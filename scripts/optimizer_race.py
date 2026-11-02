"""Is the optimiser the problem?  PILCO's `minimize` vs torch LBFGS.

The history of `pilco_pytorch` says yes: policy search with `torch.optim.LBFGS`
learned cart-pole to 11.7 and then collapsed, while the port of PILCO's MATLAB
`minimize` was stable.  The reason is in the line search -- PILCO's `minimize`
uses BFGS with **cubic extrapolation**, so it can take a long step across a flat
region, which `torch.optim.LBFGS` with the strong-Wolfe conditions will not.

That matters here because the policy landscape has exactly such a flat region:
from the hanging state every gentle policy costs 1.0 per step, so the gradient is
tiny and a conservative line search stops immediately.

This script races the two optimisers on the *same* objective, objective value and
policy initialisation, in three settings of increasing realism:

1. deterministic single trajectory (true dynamics, one particle, no noise);
2. the fixed particle batch with the true dynamics (the "oracle" model);
3. the fixed particle batch with the learned model.

Usage::

    python scripts/optimizer_race.py
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

from pilco_mc import MCConfig, fit_model, make_cartpole              # noqa: E402
from pilco_mc.nn import RbfPolicy                                    # noqa: E402
from pilco_mc.objective import cost_statistics, expected_cost        # noqa: E402
from pilco_mc.oracle import OracleDynamics, rk4_step                 # noqa: E402
from pilco_mc.particles import draw_batch, rollout_batch             # noqa: E402


def make_policy(seed=0, aggressive=True):
    gen = torch.Generator().manual_seed(seed)
    policy = RbfPolicy(4, 1, [10.0], 12, gen, angle_indices=(3,))
    if aggressive:                       # leave the "do nothing" plateau
        with torch.no_grad():
            policy.weights.mul_(4.5)
    return policy


def true_cost(policy, scenario, x0, horizon=40, substeps=8):
    x = x0.clone()
    total = torch.zeros((), dtype=x.dtype)
    for _ in range(horizon):
        total = total + scenario.cost(x).sum()
        x = rk4_step(x, policy(x), scenario.dt, substeps)
    return total


def race(objective_fn, policy, label, truth_fn):
    """Run both optimisers from identical initial parameters."""
    init = {k: v.clone() for k, v in policy.state_dict().items()}

    # --- torch LBFGS, several restarts of the inner loop -------------------
    policy.load_state_dict(init)
    optimiser = torch.optim.LBFGS(list(policy.parameters()), lr=1.0, max_iter=200,
                                  history_size=30, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-14)
    def closure():
        optimiser.zero_grad()
        loss = objective_fn()
        loss.backward()
        return loss
    for _ in range(5):
        optimiser.step(closure)
    lbfgs_value = float(truth_fn(policy))

    # --- PILCO's minimise: BFGS + cubic extrapolation ----------------------
    policy.load_state_dict(init)
    vec = ParamVector(list(policy.parameters()))
    evaluate = autograd_objective(objective_fn, vec)
    result = minimize(vec.flat(), evaluate, length=150, MFEPLS=30)
    vec.set(result.x)
    minimize_value = float(truth_fn(policy))

    print(f"  {label:<34} LBFGS {lbfgs_value:7.2f}   "
          f"minimize {minimize_value:7.2f}   "
          f"({len(result.fX)} line searches)")
    return lbfgs_value, minimize_value


def main() -> int:
    scenario = make_cartpole()
    config = MCConfig(horizon=40, verbose=False)
    mean0, cov0 = scenario.initial_belief()
    x0 = torch.zeros(1, 4)
    gen = torch.Generator().manual_seed(0)

    print("policy search from an identical initialisation, cost of the result\n")
    print(f"  {'setting':<34} {'LBFGS':>12} {'minimize':>15}")

    # 1) deterministic, true dynamics --------------------------------------
    def det_obj():
        return true_cost(policy_det, scenario, x0)

    policy_det = make_policy()
    race(det_obj, policy_det, "deterministic, true dynamics",
         lambda p: true_cost(p, scenario, x0))

    # 2) particle batch, true dynamics (oracle) -----------------------------
    batch = draw_batch(mean0, cov0, 64, 40, 1, gen)
    oracle = OracleDynamics(dt=scenario.dt)

    def oracle_obj():
        return expected_cost(rollout_batch(policy_orc, oracle, batch),
                             scenario.cost)

    policy_orc = make_policy()
    race(oracle_obj, policy_orc, "particles + true dynamics",
         lambda p: cost_statistics(rollout_batch(p, oracle, batch),
                                   scenario.cost)[0])

    # 3) particle batch, learned model --------------------------------------
    states, controls, deltas = [], [], []
    rng = np.random.default_rng(1000)
    from pilco_mc.trainer import collect_seed_data
    collect_seed_data(scenario, config, mean0, cov0, rng,
                      states, controls, deltas)
    learned = fit_model(torch.cat(states), torch.cat(controls),
                        torch.cat(deltas), config, seed=0)

    def learned_obj():
        return expected_cost(rollout_batch(policy_lrn, learned, batch),
                             scenario.cost)

    policy_lrn = make_policy()
    race(learned_obj, policy_lrn, "particles + learned model",
         lambda p: cost_statistics(rollout_batch(p, learned, batch),
                                   scenario.cost)[0])

    print("\n  reference: PILCO's own controller scores 12.8 on the true system;")
    print("  a cost of ~40 means the policy learned to do nothing.")
    print("\n  (the third row is measured on the *model*, so it says what the")
    print("   optimiser achieves, not what transfers to the plant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
