"""The one-variable substitution: PILCO's own loop with a Monte Carlo value.

Everything is PILCO's -- the GP, the RBF controller, the curb, ``minimize`` with
cubic extrapolation, the data collection, the number of line searches.  The only
difference is how a candidate policy is scored:

* ``analytic`` -- ``E[C]`` under the moment-matched belief (``plan``), and
* ``mc``       -- the sample mean of the cost of ``P`` particle rollouts.

If the MC value is the root cause of the difference, this script will show the
loop degrading from PILCO's 37.9 / 28.0 / 14.5 to something near 40.

Usage::

    python scripts/analytic_vs_mc_value.py --mode both --episodes 3
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

import pilco.objectives as objectives                      # noqa: E402
from pilco.objectives import expected_cumulative_cost      # noqa: E402
from pilco.pilco import PilcoConfig, learn                 # noqa: E402
from pilco.scenarios.cartpole import make_cartpole         # noqa: E402

N_PARTICLES = 16


def _x5(states: torch.Tensor) -> torch.Tensor:
    """``[x, dx, dtheta, sin, cos]`` -- the controller's input layout."""
    th = states[:, 3:4]
    return torch.cat([states[:, :3], torch.sin(th), torch.cos(th)], 1)


def mc_value(model, initial, particles: int = N_PARTICLES, seed: int = 0):
    """Sample mean of the cumulative cost of ``particles`` rollouts."""
    gen = torch.Generator().manual_seed(seed)
    mean = initial.mean.detach()
    cov = initial.cov.detach()
    L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(len(mean)))
    x = mean + torch.randn(particles, len(mean), generator=gen) @ L.T
    total = torch.zeros(())
    policy = model.policy
    beta = policy.beta()
    for _ in range(40):
        u_raw = torch.exp(
            -0.5 * (((_x5(x)[:, None, :] - policy.centers[None]) /
                     torch.exp(policy.log_lengthscales)[None, None, :]) ** 2).sum(-1)
        ) @ beta
        u = (9.0 * torch.sin(u_raw) + torch.sin(3.0 * u_raw)) / 8.0
        u = u[:, 0] * 10.0
        z = torch.cat([_x5(x), u[:, None]], 1)
        # NOT under no_grad: the policy gradient has to flow through the
        # dynamics, exactly as it does through plan() in the analytic version.
        delta = torch.stack([model.dynamics.models[e].eval()(z).mean
                             for e in range(4)], 1)
        var = torch.stack([model.dynamics.models[e].eval()(z).variance
                           for e in range(4)], 1)
        x = x + delta + torch.sqrt(var.clamp(min=0.0) + 1e-12) * torch.randn(
            particles, 4, generator=gen)
        # cart-pole cost, same formula as the scenario
        th = x[:, 3]
        d2 = (x[:, 0] + 0.5 * torch.sin(th)) ** 2 + 0.25 * (torch.cos(th) + 1.0) ** 2
        total = total + (1.0 - torch.exp(-0.5 * d2 / 0.25 ** 2)).mean()
    return total


def main() -> int:
    global N_PARTICLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["analytic", "mc", "both"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--searches", type=int, default=20)
    ap.add_argument("--particles", type=int, default=16)
    args = ap.parse_args()
    N_PARTICLES = args.particles

    original = objectives.ExplorationObjective.__call__

    def run(mode):
        if mode == "mc":
            def call(self, traj, model):
                return mc_value(model, traj.states[0])
            objectives.ExplorationObjective.__call__ = call
        else:
            objectives.ExplorationObjective.__call__ = original
        cfg = PilcoConfig()
        cfg.episodes = args.episodes
        cfg.policy_iterations = args.searches
        cfg.policy_optimizer = "pilco"
        cfg.verbose = False
        _, logs = learn(make_cartpole(), cfg, log=None)
        costs = [round(float(np.sum(l.realised_costs)), 2) for l in logs]
        print(f"  {mode:<9} {costs}")
        return costs

    print(f"identical loop, only the value differs "
          f"({args.searches} line searches, P={N_PARTICLES})")
    if args.mode in ("analytic", "both"):
        run("analytic")
    if args.mode in ("mc", "both"):
        run("mc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
