"""Episode-by-episode alignment: the first episode, cross-evaluated.

The two loops are identical in every component -- the same GP, the same RBF
controller, the same ``minimize`` with the same number of line searches, the same
initial dataset (same seed, same random rollout) and the same initial policy --
so episode 1 is the one place where a difference cannot be blamed on divergent
data.

This script optimises twice from that shared starting point, once against the
analytic value and once against the Monte Carlo value, and then evaluates **both**
resulting policies with **both** values plus the true plant cost.  That separates
two very different explanations:

* the MC objective leads the optimiser to a different (worse) policy -- then the
  cross-evaluation rows differ, or
* the MC objective is simply harder to optimise -- then the MC-optimised policy is
  worse under *every* metric, including the analytic one.

Usage::

    python scripts/align_episode1.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PILCO = os.path.join(os.path.dirname(_ROOT), "pilco_pytorch")
for p in (_ROOT, _PILCO):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_default_dtype(torch.float64)

from pilco.gaussian import Gaussian                              # noqa: E402
from pilco.models.gp import (build_dynamics_op,                   # noqa: E402
                             fit_hyperparameters_pilco)
from pilco.optim.minimize import (ParamVector, autograd_objective,  # noqa: E402
                                  minimize)
from pilco.scenarios.cartpole import make_cartpole as pilco_cartpole  # noqa: E402
from pilco.trajectory import PilcoModel, plan                     # noqa: E402

from pilco_mc import make_cartpole                                # noqa: E402

N_PARTICLES = 64
HORIZON = 40


def mc_value(model: PilcoModel, initial: Gaussian, particles: int = N_PARTICLES,
             seed: int = 0) -> torch.Tensor:
    """Sample-mean cost of ``particles`` rollouts through the same GP."""
    gen = torch.Generator().manual_seed(seed)
    mean, cov = initial.mean.detach(), initial.cov.detach()
    L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(len(mean)))
    x = mean + torch.randn(particles, len(mean), generator=gen) @ L.T
    total = torch.zeros(())
    beta = model.policy.beta()
    centres = model.policy.centers
    ell = torch.exp(model.policy.log_lengthscales)
    for _ in range(HORIZON):
        th = x[:, 3:4]
        x5 = torch.cat([x[:, :3], torch.sin(th), torch.cos(th)], 1)
        k = torch.exp(-0.5 * (((x5[:, None, :] - centres[None]) / ell[None, None, :])
                              ** 2).sum(-1))
        raw = k @ beta
        u = 10.0 * (9.0 * torch.sin(raw) + torch.sin(3.0 * raw)) / 8.0
        z = torch.cat([x5, u], 1)
        delta = torch.stack([model.dynamics.models[e].eval()(z).mean
                             for e in range(4)], 1)
        var = torch.stack([model.dynamics.models[e].eval()(z).variance
                           for e in range(4)], 1)
        x = x + delta + torch.sqrt(var.clamp(min=0.0) + 1e-12) * torch.randn(
            particles, 4, generator=gen)
        d2 = (x[:, 0] + 0.5 * torch.sin(th)) ** 2 + 0.25 * (torch.cos(th) + 1.0) ** 2
        total = total + (1.0 - torch.exp(-0.5 * d2 / 0.25 ** 2)).mean()
    return total


def true_cost(model: PilcoModel, scenario, n: int = 8, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.zeros(4)
        total = 0.0
        for _ in range(HORIZON):
            total += float(scenario.cost(torch.as_tensor(x)))
            x5 = torch.cat([torch.as_tensor(x)[:3],
                            torch.tensor([np.sin(x[3]), np.cos(x[3])])])
            u = float(model.policy.moments(x5, torch.zeros(5, 5)).mean[0])
            x = scenario.step_true(x, u)
            x = x + rng.standard_normal(4) * scenario.process_noise
        out.append(total)
    return float(np.mean(out))


def main() -> int:
    scenario = make_cartpole()
    pilco_scenario = pilco_cartpole()
    rng = np.random.default_rng(1000)

    states, controls = [], []
    x = np.zeros(4)
    for _ in range(HORIZON):                     # exactly PILCO's first rollout
        a = float(rng.uniform(-scenario.u_max, scenario.u_max))
        controls.append(a)
        states.append(x.copy())
        x = scenario.step_true(x, a)
        x = x + rng.standard_normal(4) * scenario.process_noise
    states.append(x.copy())
    states = np.asarray(states)
    idx, tgt = pilco_scenario.data_from_rollout(states, np.asarray(controls))
    op = build_dynamics_op(torch.as_tensor(idx), torch.as_tensor(tgt))
    fit_hyperparameters_pilco(op, iterations=150, MFEPLS=30)

    centres0 = (0.1 * torch.randn(5, 12, generator=torch.Generator().manual_seed(1))).T
    targets0 = 0.1 * torch.randn(12, 1, generator=torch.Generator().manual_seed(1))

    def fresh_policy():
        from pilco.ops.policy import RbfPolicyOp
        return RbfPolicyOp(centres0.clone(), targets0.clone(),
                           torch.ones(5), noise=0.01, signal=1.0)

    x0 = Gaussian(torch.zeros(4), torch.eye(4) * 0.01)
    results = {}
    for kind in ("analytic", "mc"):
        policy = fresh_policy()
        model = PilcoModel(pilco_scenario.spec, op, policy, pilco_scenario.cost,
                           max_u=[10.0])
        vec = ParamVector([policy.centers, policy.targets,
                           policy.log_lengthscales])
        objective = (lambda: plan(model, x0, HORIZON,
                                  with_jacobians=False).value()) if kind == "analytic" \
            else (lambda: mc_value(model, x0))
        evaluate = autograd_objective(objective, vec)
        res = minimize(vec.flat(), evaluate, length=20, MFEPLS=30)
        vec.set(res.x)
        with torch.no_grad():
            results[kind] = {
                "optimised": res.final,
                "analytic": float(plan(model, x0, HORIZON,
                                       with_jacobians=False).value()),
                "mc": float(mc_value(model, x0, seed=7)),
                "true": true_cost(model, scenario),
            }

    print("episode 1, shared data and starting policy, 20 line searches\n")
    print(f"{'policy from':<14} {'optimised':>10} {'analytic':>10} "
          f"{'mc':>9} {'true':>7}")
    for kind in ("analytic", "mc"):
        r = results[kind]
        print(f"{kind:<14} {r['optimised']:10.2f} {r['analytic']:10.2f} "
              f"{r['mc']:9.2f} {r['true']:7.2f}")
    print("\n'optimised' is the objective the optimiser saw; the other three are")
    print("post-hoc evaluations of the resulting policy on the same model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
