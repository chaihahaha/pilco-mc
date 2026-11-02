"""Does the Monte Carlo value surface grow structure with data, and does variance
reduction make it usable sooner?

The previous experiment showed why the Monte Carlo objective cannot drive the
policy search at a small data budget: along a line through the current policy the
surface spans 0.015 while its seed-to-seed spread is also about 0.015, i.e. it is
flat below its own sampling resolution.  The analytic value manufactures a 0.54
deep basin instead.

This script asks two follow-up questions, with one measurement:

1. **More data.**  Fit the neural model on n = 40, 160, 640, 2560 points and
   measure the surface's structure (the drop from a random policy to a verified
   good one) against its noise (spread across seeds).
2. **Variance reduction.**  Repeat with antithetic sampling -- each noise draw is
   paired with its negation -- which cancels the first-order effect of the noise
   for a deterministic integrand.

The policies are a random residual MLP and the controller distilled from PILCO
(true cost 12.84); interpolating between them is a line in policy-parameter space
that genuinely goes from "the pole falls" to "the pole swings up", so the value
function should show a clear drop if it can see anything at all.

Usage::

    python scripts/mc_surface_vs_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_default_dtype(torch.float64)

from pilco_mc import (BoundedPolicy, MCConfig, fit_model,  # noqa: E402
                      make_cartpole)

HORIZON = 40


def generate(scenario, n, seed=0):
    rng = np.random.default_rng(seed)
    S, U, D = [], [], []
    total = 0
    while total < n:
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
        total += i.shape[0]
    return torch.cat(S)[:n], torch.cat(U)[:n], torch.cat(D)[:n]


def mc_value(model, policy, mean0, cov0, particles, noise, antithetic):
    """Sample-mean cost of ``particles`` rollouts, optionally antithetic."""
    def rollout(sign):
        x = mean0 + sign * noise["init"]
        total = torch.zeros(())
        for t in range(HORIZON):
            u = policy(x)
            delta, var = model.predict(x, u)
            x = x + delta + sign * torch.sqrt(var.clamp(min=0.0) + 1e-12) * noise["step"][t]
            total = total + model_cost(model, x)
        return total

    if antithetic:
        return 0.5 * (rollout(1.0) + rollout(-1.0))
    return rollout(1.0)


def model_cost(model, x):
    th = x[:, 3]
    d2 = (x[:, 0] + 0.5 * torch.sin(th)) ** 2 + 0.25 * (torch.cos(th) + 1.0) ** 2
    return (1.0 - torch.exp(-0.5 * d2 / 0.25 ** 2)).mean()


def make_noise(particles, seed):
    gen = torch.Generator().manual_seed(seed)
    return {"init": 0.1 * torch.randn(particles, 4, generator=gen),
            "step": [torch.randn(particles, 4, generator=gen)
                     for _ in range(HORIZON)]}


def main() -> int:
    scenario = make_cartpole()
    good = BoundedPolicy(4, 1, [scenario.u_max], width=64, blocks=2,
                         angle_indices=(3,))
    good.load_state_dict(torch.load(os.path.join(_ROOT, "results",
                                                 "distilled_mlp.pt"),
                                    weights_only=True))
    torch.manual_seed(0)
    rand = BoundedPolicy(4, 1, [scenario.u_max], width=64, blocks=2,
                         angle_indices=(3,))
    theta_good = {k: v.clone() for k, v in good.state_dict().items()}
    theta_rand = {k: v.clone() for k, v in rand.state_dict().items()}

    def interpolate(alpha):
        return {k: (1 - alpha) * theta_rand[k] + alpha * theta_good[k]
                for k in theta_good}

    mean0 = torch.zeros(4)
    cov0 = torch.eye(4) * 0.01
    print("surface structure (random -> good) vs sampling noise")
    print(f"{'n':>5} {'mode':<11} {'value@0':>9} {'value@1':>9} "
          f"{'structure':>10} {'noise':>8} {'S/N':>6}")
    for n in (40, 160, 640, 2560):
        states, controls, deltas = generate(scenario, n)
        model = fit_model(states, controls, deltas,
                          MCConfig(verbose=False, model_kind="nn",
                                   model_epochs=400, angle_indices=(3,)),
                          seed=0)
        # calibrate the variance, as the loop does
        model.calibrate(states, controls, deltas)
        for antithetic in (False, True):
            vals = {}
            for alpha in (0.0, 1.0):
                policy = BoundedPolicy(4, 1, [scenario.u_max], width=64,
                                       blocks=2, angle_indices=(3,))
                policy.load_state_dict(interpolate(alpha))
                samples = []
                for seed in range(4):
                    noise = make_noise(64, 100 * n + seed)
                    samples.append(float(mc_value(model, policy, mean0, cov0,
                                                  64, noise, antithetic)))
                vals[alpha] = (float(np.mean(samples)), float(np.std(samples)))
            structure = vals[0.0][0] - vals[1.0][0]
            noise = max(vals[0.0][1], vals[1.0][1])
            print(f"{n:5d} {'antithetic' if antithetic else 'plain':<11} "
                  f"{vals[0.0][0]:9.3f} {vals[1.0][0]:9.3f} {structure:10.4f} "
                  f"{noise:8.4f} {structure / noise if noise else float('inf'):6.2f}")
    print("\nstructure = V(random policy) - V(good policy); noise = spread over 4 seeds")
    print("the objective is usable only when structure/noise is well above 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
