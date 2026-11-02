"""Frozen-data alignment: same dataset, same start, only the objective differs.

After episode 1 the two loops diverge because they deploy different policies, so
later episodes can no longer be compared.  This script removes that confound: one
shared dataset, one shared GP (fitted once per episode from PILCO's own code), one
shared starting policy, and two optimisations that differ *only* in the objective
-- analytic moment matching versus a Monte Carlo sample mean.

The analytic-optimised policy is the one deployed, so the dataset grows the way
PILCO's would and the MC objective is being tested on PILCO's own data.  That is
the question: can the MC objective exploit the same data that the analytic one
exploits?

Usage::

    python scripts/align_frozen.py --episodes 3
"""
from __future__ import annotations

import argparse
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
from pilco.pilco import initial_policy                            # noqa: E402
from pilco.scenarios.cartpole import make_cartpole as pilco_cartpole  # noqa: E402
from pilco_mc import make_cartpole                                # noqa: E402
from pilco.trajectory import PilcoModel, plan                     # noqa: E402

HORIZON = 40
SEARCHES = 20
PARTICLES = 32


def mc_value(model, initial, particles: int = PARTICLES, seed: int = 0):
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


def control(model, x: np.ndarray) -> float:
    x5 = torch.cat([torch.as_tensor(x)[:3],
                    torch.tensor([np.sin(x[3]), np.cos(x[3])])])
    return float(model.policy.moments(x5, torch.zeros(5, 5)).mean[0])


def true_cost(model, scenario, n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.zeros(4)
        total = 0.0
        for _ in range(HORIZON):
            total += float(scenario.cost(torch.as_tensor(x)))
            x = scenario.step_true(x, control(model, x))
            x = x + rng.standard_normal(4) * scenario.process_noise
        out.append(total)
    return float(np.mean(out)), float(np.std(out))


def deploy(model, scenario, x0: np.ndarray, rng):
    states, controls = [x0.copy()], []
    x = x0.copy()
    for _ in range(HORIZON):
        u = control(model, x)
        controls.append(u)
        x = scenario.step_true(x, u)
        x = x + rng.standard_normal(4) * scenario.process_noise
        states.append(x.copy())
    return np.asarray(states), np.asarray(controls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    scenario = make_cartpole()
    pilco_scen = pilco_cartpole()
    rng = np.random.default_rng(1000)
    spec = pilco_scen.spec
    x0 = Gaussian(torch.zeros(4), torch.eye(4) * 0.01)

    inputs, targets = [], []
    # PILCO's first random rollout
    states, controls = [np.zeros(4)], []
    x = np.zeros(4)
    for _ in range(HORIZON):
        a = float(rng.uniform(-scenario.u_max, scenario.u_max))
        controls.append(a)
        x = scenario.step_true(x, a) + rng.standard_normal(4) * scenario.process_noise
        states.append(x.copy())
    i, t = pilco_scen.data_from_rollout(np.asarray(states), np.asarray(controls))
    inputs.append(torch.as_tensor(i))
    targets.append(torch.as_tensor(t))

    print(f"{'ep':>2} {'pts':>4} | {'objective':<9} {'analytic':>9} {'mc':>8} "
          f"{'true':>7} {'sd':>5}")
    for episode in range(args.episodes):
        op = build_dynamics_op(torch.cat(inputs), torch.cat(targets))
        fit_hyperparameters_pilco(op, iterations=150, MFEPLS=30)

        best_model = None
        for kind in ("analytic", "mc"):
            policy = initial_policy(pilco_scen, x0, 12,
                                    torch.Generator().manual_seed(1), legacy=False)
            model = PilcoModel(spec, op, policy, pilco_scen.cost, max_u=[10.0])
            vec = ParamVector([policy.centers, policy.targets,
                               policy.log_lengthscales])
            objective = (lambda: plan(model, x0, HORIZON,
                                      with_jacobians=False).value()) \
                if kind == "analytic" else (lambda: mc_value(model, x0))
            evaluate = autograd_objective(objective, vec)
            res = minimize(vec.flat(), evaluate, length=SEARCHES, MFEPLS=30)
            vec.set(res.x)
            with torch.no_grad():
                an = float(plan(model, x0, HORIZON, with_jacobians=False).value())
                mc = float(mc_value(model, x0, seed=7))
            tr, sd = true_cost(model, scenario)
            print(f"{episode + 1:2d} {torch.cat(inputs).shape[0]:4d} | "
                  f"{kind:<9} {an:9.2f} {mc:8.2f} {tr:7.2f} {sd:5.2f}")
            if kind == "analytic":
                best_model = model

        # deploy the analytic-optimised policy, as PILCO's loop would
        xs, us = deploy(best_model, scenario, x0.mean.numpy(), rng)
        i, t = pilco_scen.data_from_rollout(xs, us)
        inputs.append(torch.as_tensor(i))
        targets.append(torch.as_tensor(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
