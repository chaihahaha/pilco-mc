"""Receding-horizon control with the learned model (MPC), and why it is the point.

The measurement in ``docs/NOTES.md`` is that the learned model's imagined
trajectory matches reality for about ten steps and then drifts onto "the pole has
fallen".  A 40-step open-loop plan is therefore optimised against a fiction, no
matter how good the policy class or the optimiser is.

MPC makes the planning horizon short enough that the model is right, and pays for
it by re-planning every step.  Only the first action is executed, then the state
is observed again and the problem is re-solved, so the error never compounds:
each plan is 8-10 steps long, the model is accurate over that span, and the
feedback closes the loop that the open-loop plan could not.

This is also the natural way to *use* a 90%-accurate model, and it is what the
"branching tree" idea reduces to once you stop insisting on a single 40-step
commitment.

Usage::

    python scripts/mpc_cartpole.py [--episodes 5]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

torch.set_default_dtype(torch.float64)

from pilco_mc import MCConfig, fit_model, make_cartpole          # noqa: E402
from pilco_mc.trainer import collect_seed_data                    # noqa: E402


def terminal_cost(state: torch.Tensor, weight: float) -> torch.Tensor:
    """Cost-to-go beyond the planning horizon.

    A short plan is myopic: from the hanging state no 8-step action sequence can
    get the pole up, so the greedy objective is flat and MPC does nothing
    (measured: 39.99, the same as no control).  The terminal term pays for
    *progress towards upright* at the end of the plan, which is what lets a short
    horizon pump energy over many replans.
    """
    x, theta = state[..., 0], state[..., 3]
    return weight * ((1.0 - torch.cos(theta)) + 0.5 * x ** 2)


def plan_and_act(model, cost_fn, state: torch.Tensor, plan_horizon: int,
                 u_max: float, dt: float, previous: torch.Tensor,
                 iters: int = 60, terminal_weight: float = 4.0
                 ) -> tuple[float, torch.Tensor]:
    """Solve the short-horizon problem and return the first control.

    ``previous`` is the previous solution, shifted by one step and re-optimised;
    warm starting this way is what makes a 60-iteration solve enough at every
    control step.
    """
    warm = torch.cat([previous[1:], torch.zeros(1)])
    raw = torch.atanh((warm / u_max).clamp(-0.99, 0.99)).requires_grad_(True)
    optimiser = torch.optim.LBFGS([raw], lr=1.0, max_iter=iters, history_size=20,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimiser.zero_grad()
        x = state.clone()
        total = torch.zeros((), dtype=state.dtype)
        for t in range(plan_horizon):
            total = total + cost_fn(x).sum()
            u = (u_max * torch.tanh(raw[t])).reshape(1, 1)
            x = x + model.predict(x, u)[0]
        total = total + terminal_cost(x, terminal_weight)
        total.backward()
        return total

    optimiser.step(closure)
    with torch.no_grad():
        u = u_max * torch.tanh(raw.detach())
    return float(u[0]), u


@torch.no_grad()
def run_mpc(model, scenario, x0: np.ndarray, steps: int, plan_horizon: int,
            rng: np.random.Generator, with_noise: bool = True,
            iters: int = 60, terminal_weight: float = 4.0):
    x = np.asarray(x0, dtype=np.float64).copy()
    if with_noise:
        x = x + rng.standard_normal(4) * scenario.process_noise
    previous = torch.zeros(plan_horizon)
    states, costs, applied = [x.copy()], [], []
    for _ in range(steps):
        costs.append(float(scenario.cost(torch.as_tensor(x))))
        u, previous = plan_and_act(model, scenario.cost,
                                   torch.as_tensor(x)[None], plan_horizon,
                                   scenario.u_max, scenario.dt, previous, iters,
                                   terminal_weight)
        applied.append(u)
        x = scenario.step_true(x, u)
        if with_noise:
            x = x + rng.standard_normal(4) * scenario.process_noise
        states.append(x.copy())
    return np.asarray(states), costs, np.asarray(applied)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--plan-horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--terminal-weight", type=float, default=4.0)
    args = ap.parse_args()

    scenario = make_cartpole(horizon=args.horizon)
    config = MCConfig(horizon=args.horizon, episodes=args.episodes,
                      seed=args.seed, verbose=True)
    mean0, cov0 = scenario.initial_belief()
    rng = np.random.default_rng(args.seed + 1000)

    states, controls, deltas = [], [], []
    collect_seed_data(scenario, config, mean0, cov0, rng, states, controls, deltas)
    model = fit_model(torch.cat(states), torch.cat(controls), torch.cat(deltas),
                      config, seed=args.seed)

    print(f"\nreceding horizon: plan {args.plan_horizon} steps, execute 1, "
          f"repeat {args.horizon} times")
    print("  episode   realised cost   (open-loop policy search was ~39.9)")
    for episode in range(args.episodes):
        x0 = mean0.numpy() + np.linalg.cholesky(cov0.numpy()) @ rng.standard_normal(4)
        states_mpc, costs, applied = run_mpc(model, scenario, x0, args.horizon,
                                             args.plan_horizon, rng,
                                             iters=args.iters,
                                             terminal_weight=args.terminal_weight)
        print(f"  {episode + 1:7d}   {np.sum(costs):13.2f}")
        # add the closed-loop trajectory to the data and re-fit: the model then
        # covers the states MPC actually visits, which is the region that matters
        i, j, k = scenario.transitions(states_mpc, applied)
        states.append(i); controls.append(j); deltas.append(k)
        model = fit_model(torch.cat(states), torch.cat(controls),
                          torch.cat(deltas), config, seed=args.seed + episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
