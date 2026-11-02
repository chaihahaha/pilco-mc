"""Per-episode alignment taken from inside ``learn()`` itself.

Earlier attempts to study PILCO by writing a simplified copy of its loop produced
comparisons that did not even reproduce PILCO's baseline.  This script does the
opposite: it leaves ``learn()`` untouched and wraps ``optimise_policy``, so every
row below is measured on the model and policy that the real loop just produced.

For each episode it records

* the analytic value of the optimised policy (``plan``),
* the Monte Carlo value of the same policy (particle rollouts through the same GP),
* the true plant cost, averaged over several rollouts, with its spread.

Two runs: the unmodified analytic objective, and the same loop with
``ExplorationObjective.__call__`` replaced by the MC value.

Usage::

    python scripts/align_instrumented.py --episodes 4 --searches 20
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
import pilco.pilco as pilco_module                          # noqa: E402
from pilco.gaussian import Gaussian                         # noqa: E402
from pilco.pilco import PilcoConfig, learn                  # noqa: E402
from pilco.scenarios.cartpole import make_cartpole as pilco_cartpole  # noqa: E402
from pilco.trajectory import plan                           # noqa: E402

from pilco_mc import make_cartpole                          # noqa: E402

HORIZON = 40
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
    """The control the loop deploys: the RBF output **after** the saturation.

    ``policy.moments`` returns ``u_raw``; PILCO then applies
    ``gSat = u_max (9 sin u + sin 3u) / 8``.  Forgetting that step made every
    policy look like it never moved the pole (true cost exactly 40.00 with zero
    spread), which invalidated the first version of this table.
    """
    x5 = torch.cat([torch.as_tensor(x)[:3],
                    torch.tensor([np.sin(x[3]), np.cos(x[3])])])
    raw = float(model.policy.moments(x5, torch.zeros(5, 5)).mean[0])
    return float(10.0 * (9.0 * np.sin(raw) + np.sin(3.0 * raw)) / 8.0)


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
    return float(np.mean(out)), float(np.std(out) / np.sqrt(n))


def run(mode: str, episodes: int, searches: int):
    scenario = make_cartpole()
    records = []
    original = pilco_module.optimise_policy

    def wrapped(model, initial_state, config, objective, verbose=False):
        out = original(model, initial_state, config, objective, verbose)
        with torch.no_grad():
            an = float(plan(model, initial_state, config.horizon,
                            with_jacobians=False).value())
            mc = float(mc_value(model, initial_state, seed=7))
        tr, se = true_cost(model, scenario)
        records.append((an, mc, tr, se))
        return out

    pilco_module.optimise_policy = wrapped
    original_call = objectives.ExplorationObjective.__call__
    if mode == "mc":
        objectives.ExplorationObjective.__call__ = \
            lambda self, traj, model: mc_value(model, traj.states[0], seed=11)
    try:
        cfg = PilcoConfig()
        cfg.episodes = episodes
        cfg.policy_iterations = searches
        cfg.policy_optimizer = "pilco"
        cfg.verbose = False
        learn(pilco_cartpole(), cfg, log=None)
    finally:
        pilco_module.optimise_policy = original
        objectives.ExplorationObjective.__call__ = original_call
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--searches", type=int, default=20)
    args = ap.parse_args()

    print(f"measured inside learn(), {args.searches} line searches per episode")
    print(f"{'objective':<9} {'ep':>2} {'analytic':>9} {'mc':>8} {'true':>7} {'se':>6}")
    for mode in ("analytic", "mc"):
        for ep, (an, mc, tr, se) in enumerate(run(mode, args.episodes,
                                                  args.searches), 1):
            print(f"{mode:<9} {ep:2d} {an:9.2f} {mc:8.2f} {tr:7.2f} {se:6.3f}")
    print("\nall rows come from the model and policy the real loop produced;")
    print("'true' is the mean over 8 plant rollouts, 'se' its standard error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
