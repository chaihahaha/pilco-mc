"""What does the neural model buy?  Scaling, since it cannot buy accuracy here.

With 40-320 points a GP is excellent and the network is not: that is why every
"match PILCO" experiment in this repo uses the GP.  The reason to replace the GP
is that the GP's cost is cubic in the number of points, and PILCO's planner calls
it once per particle per step.  The network is trained once and is then O(1) per
query.

So the honest claim to test is not "the network is more accurate" but "the
network is the only one of the two that survives more data".  This script fits
both on the same data at increasing sizes and reports

* fit time,
* one-step accuracy on held-out data (the thing the planner needs),
* the cost of one planning step (128 particles through 40 steps),
* and, for the GP, the size at which it stops being feasible.

Usage::

    python scripts/scaling.py [--max-gp 3000]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

from pilco_mc import MCConfig, fit_model, make_cartpole          # noqa: E402
from pilco_mc.particles import draw_batch, rollout_batch         # noqa: E402


def generate(scenario, n, seed=0):
    """States/controls/deltas from random-control rollouts of the true plant."""
    rng = np.random.default_rng(seed)
    S, U, D = [], [], []
    collected = 0
    while collected < n:
        x, s, u = np.zeros(4), [], []
        for _ in range(200):
            a = float(rng.uniform(-scenario.u_max, scenario.u_max))
            u.append(a)
            s.append(x.copy())
            x = scenario.step_true(x, a) + rng.standard_normal(4) * scenario.process_noise
        s.append(x.copy())
        i, j, k = scenario.transitions(np.asarray(s), np.asarray(u))
        S.append(i)
        U.append(j)
        D.append(k)
        collected += i.shape[0]
    return (torch.cat(S)[:n], torch.cat(U)[:n], torch.cat(D)[:n])


def accuracy(model, states, controls, deltas):
    with torch.no_grad():
        pred, _ = model.predict(states, controls)
        ss_res = ((deltas - pred) ** 2).sum()
        ss_tot = ((deltas - deltas.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def planning_step_cost(model, n_particles=128, horizon=40):
    gen = torch.Generator().manual_seed(0)
    from pilco_mc.nn import RbfPolicy
    policy = RbfPolicy(4, 1, [10.0], 12, gen, angle_indices=(3,))
    batch = draw_batch(torch.zeros(4), torch.eye(4) * 0.01, n_particles,
                       horizon, 1, gen)
    t0 = time.perf_counter()
    with torch.no_grad():
        rollout_batch(policy, model, batch)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gp", type=int, default=3000)
    ap.add_argument("--gp-iterations", type=int, default=150)
    args = ap.parse_args()

    scenario = make_cartpole()
    data = generate(scenario, max(30000, args.max_gp * 2))
    all_s, all_u, all_d = data
    print("fitting the GP and the network on identical data\n")
    print(f"{'n':>6} | {'GP fit':>9} {'GP R2':>7} {'GP step':>9} | "
          f"{'NN fit':>9} {'NN R2':>7} {'NN step':>9}")
    print("-" * 78)

    sizes = [100, 300, 1000, 3000, 10000, 30000]
    perm = torch.randperm(all_s.shape[0], generator=torch.Generator().manual_seed(0))
    all_s, all_u, all_d = all_s[perm], all_u[perm], all_d[perm]
    for n in sizes:
        n_val = max(20, n // 5)
        sel = torch.arange(n)
        states, controls, deltas = (t[sel[:-n_val]] for t in (all_s, all_u, all_d))
        vs, vu, vd = (t[sel[-n_val:]] for t in (all_s, all_u, all_d))

        gp_fit = gp_r2 = gp_step = float("nan")
        if n <= args.max_gp:
            try:
                t0 = time.perf_counter()
                gp = fit_model(states, controls, deltas,
                               MCConfig(verbose=False, model_kind="gp",
                                        gp_iterations=args.gp_iterations),
                               seed=0)
                gp_fit = time.perf_counter() - t0
                gp_r2 = accuracy(gp, vs, vu, vd)
                gp_step = planning_step_cost(gp)
                del gp
                gc.collect()
            except Exception as exc:                      # memory, mostly
                gp_fit = float("nan")
                print(f"  (GP failed at n={n}: {type(exc).__name__})")

        t0 = time.perf_counter()
        nn = fit_model(states, controls, deltas,
                       MCConfig(verbose=False, model_kind="nn",
                                model_epochs=1500), seed=0)
        nn_fit = time.perf_counter() - t0
        nn_r2 = accuracy(nn, vs, vu, vd)
        nn_step = planning_step_cost(nn)
        del nn
        gc.collect()

        def fmt(x, scale=1.0):
            return "       -" if not np.isfinite(x) else f"{x * scale:8.2f}"

        print(f"{n:6d} | {fmt(gp_fit):>8}s {fmt(gp_r2):>7} {fmt(gp_step, 1e3):>8}ms | "
              f"{fmt(nn_fit):>8}s {fmt(nn_r2):>7} {fmt(nn_step, 1e3):>8}ms")
    print("\n  GP: one Cholesky per output per fit, O(n^3); the planner queries it")
    print("      O(particles x horizon) times per policy evaluation.")
    print("  NN: one training pass; a query is a 3-layer MLP, independent of n.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
