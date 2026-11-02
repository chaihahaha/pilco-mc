"""Is the line search the last structural difference?  PILCO's `minimize` vs LBFGS.

Everything else has been aligned: the same 40-point dataset, the same GP, the same
RBF policy and the same initialisation.  What still differs from PILCO is that the
objective here is a sampled one and the optimiser is ``torch.optim.LBFGS`` with
strong-Wolfe, whereas PILCO uses BFGS with **cubic extrapolation** -- which can
take a long step when the gradient is small, and the gradient at this starting
point *is* small (measured 0.04, with the imagined cost flat at 39.996).

To isolate the line search from the sampling noise, the objective here is the
**mean** rollout through the model (one particle, no noise).  That is the
"smooth" end of the Monte Carlo family and it is a deterministic function of the
policy, so both optimisers are comparing numbers of the same quality; the only
difference left is how they search.

Usage::

    python scripts/linesearch_race.py
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

from pilco.optim.minimize import (ParamVector, autograd_objective,  # noqa: E402
                                  minimize)

from pilco_mc import MCConfig, RbfPolicy, fit_model, make_cartpole   # noqa: E402
from pilco_mc.nn import initialise_output_scale                       # noqa: E402

HORIZON = 40
DT = 0.1


def mean_rollout_cost(policy, model, scenario, x0, horizon=HORIZON):
    """Deterministic imagined cost: the model's mean trajectory from ``x0``."""
    x = x0.clone()
    total = torch.zeros((), dtype=x.dtype)
    for _ in range(horizon):
        total = total + scenario.cost(x).sum()
        u = policy(x)
        mean, _ = model.predict(x, u)
        x = x + mean
    return total


def true_cost(scenario, policy, n=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.zeros(4)
        total = 0.0
        for _ in range(HORIZON):
            total += float(scenario.cost(torch.as_tensor(x)))
            with torch.no_grad():
                u = float(policy(torch.as_tensor(x)[None])[0, 0])
            x = scenario.step_true(x, u)
            x = x + rng.standard_normal(4) * scenario.process_noise
        out.append(total)
    return float(np.mean(out)), float(np.std(out))


def main() -> int:
    scenario = make_cartpole()
    rng = np.random.default_rng(1000)

    # one 40-step random rollout -- PILCO's starting dataset
    x, s, u = np.zeros(4), [], []
    for _ in range(HORIZON):
        a = float(rng.uniform(-scenario.u_max, scenario.u_max))
        u.append(a)
        s.append(x.copy())
        x = scenario.step_true(x, a) + rng.standard_normal(4) * scenario.process_noise
    s.append(x.copy())
    states, controls, deltas = scenario.transitions(np.asarray(s), np.asarray(u))

    model = fit_model(states, controls, deltas,
                      MCConfig(verbose=False, model_kind="gp", gp_iterations=150),
                      seed=0)
    gen = torch.Generator().manual_seed(0)
    base = RbfPolicy(4, 1, [scenario.u_max], 12, gen, angle_indices=(3,),
                     squash="pilco")
    initialise_output_scale(base, torch.zeros(1, 4), target=0.5)
    x0 = torch.zeros(1, 4)

    print(f"40 points, GP fitted, RBF policy initialised to u(x0) = "
          f"{float(base(x0)):.2f} N")
    print(f"imagined cost before optimisation: "
          f"{float(mean_rollout_cost(base, model, scenario, x0)):.3f}")
    print(f"true cost before optimisation:     {true_cost(scenario, base)[0]:.2f}\n")

    def run(label, optimise):
        policy = RbfPolicy(4, 1, [scenario.u_max], 12,
                           torch.Generator().manual_seed(0), angle_indices=(3,),
                           squash="pilco")
        initialise_output_scale(policy, x0, target=0.5)
        n_evals = optimise(policy)
        imagined = float(mean_rollout_cost(policy, model, scenario, x0))
        real, spread = true_cost(scenario, policy)
        print(f"  {label:<26} imagined {imagined:7.2f}   "
              f"true {real:6.2f} +- {spread:.2f}   ({n_evals} evals)")

    def lbfgs(policy):
        opt = torch.optim.LBFGS(list(policy.parameters()), lr=1.0, max_iter=200,
                                history_size=30, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-14)
        def closure():
            opt.zero_grad()
            loss = mean_rollout_cost(policy, model, scenario, x0)
            loss.backward()
            return loss
        for _ in range(5):
            opt.step(closure)
        return "5x200"

    def pilco_minimise(policy):
        vec = ParamVector(list(policy.parameters()))
        evaluate = autograd_objective(
            lambda: mean_rollout_cost(policy, model, scenario, x0), vec)
        res = minimize(vec.flat(), evaluate, length=40, MFEPLS=30)
        vec.set(res.x)
        return f"{len(res.fX)} searches"

    run("torch LBFGS (strong-Wolfe)", lbfgs)
    run("PILCO minimize (cubic)", pilco_minimise)
    print("\n  a true cost near 40 is 'the pole has fallen'; PILCO reaches ~14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
