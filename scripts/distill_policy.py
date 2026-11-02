"""Can a neural policy represent the swing-up at all?  Distil it in and check.

The MLP policy never leaves the "do nothing" plateau under policy search (see
``docs/NOTES.md``), so the question "is the neural policy good enough" cannot be
answered by search.  It can be answered by supervised learning: roll the known
good controller out, regress the MLP onto ``(state, control)`` pairs, and measure
what the MLP actually does on the plant.

If the distilled MLP swings up, then the function class is fine and the problem
is purely the *search*; if it does not, then the MLP needs more than a better
starting point.

Usage::

    python scripts/distill_policy.py
"""
from __future__ import annotations

import importlib.util
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

from pilco_mc import BoundedPolicy, MCConfig, fit_model, make_cartpole   # noqa: E402
from pilco_mc.trainer import collect_seed_data                           # noqa: E402


def load_controller():
    path = os.path.join(_PILCO, "results", "cartpole_swingup_policy.npz")
    spec = importlib.util.spec_from_file_location(
        "anim", os.path.join(_PILCO, "scripts", "make_animation.py"))
    anim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anim)
    return anim.Controller(path)


def collect_teacher_data(scenario, control, n_rollouts=30, steps=40, seed=0):
    """(state, control) pairs along the teacher's trajectories, with perturbations."""
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for _ in range(n_rollouts):
        # perturbed start so the student also learns to recover
        x = rng.standard_normal(4) * np.array([0.15, 0.2, 0.2, 0.35])
        for _ in range(steps):
            u = float(np.clip(control(x), -scenario.u_max, scenario.u_max))
            states.append(x.copy())
            actions.append(u)
            x = scenario.step_true(x, u)
            x = x + rng.standard_normal(4) * scenario.process_noise
            if abs(x[0]) > 2.5:                       # keep the cart on the rail
                break
    return (torch.as_tensor(np.asarray(states)),
            torch.as_tensor(np.asarray(actions)).reshape(-1, 1))


def fit_student(student, states, actions, epochs=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = ((student(states) - actions) ** 2).mean()
        loss.backward()
        optimiser.step()
    return float(loss)


def true_cost(scenario, policy, n=8, seed=0):
    rng = np.random.default_rng(seed)
    costs = []
    for _ in range(n):
        x = np.zeros(4)
        total = 0.0
        for _ in range(scenario.horizon):
            total += float(scenario.cost(torch.as_tensor(x)))
            with torch.no_grad():
                u = float(policy(torch.as_tensor(x)[None])[0, 0])
            x = scenario.step_true(x, u)
            x = x + rng.standard_normal(4) * scenario.process_noise
        costs.append(total)
    return float(np.mean(costs)), float(np.std(costs))


def main() -> int:
    scenario = make_cartpole()
    control = load_controller()
    teacher_cost, _ = true_cost(scenario, lambda s: torch.as_tensor(
        [[np.clip(control(s[0].numpy()), -10, 10)]]))
    print(f"teacher (PILCO's controller) on the plant: {teacher_cost:.2f}\n")

    states, actions = collect_teacher_data(scenario, control)
    print(f"teacher data: {states.shape[0]} (state, control) pairs")

    for seed in (0, 1):
        gen = torch.Generator().manual_seed(seed)
        student = BoundedPolicy(4, 1, [scenario.u_max], angle_indices=(3,))
        with torch.no_grad():                 # start from a sensible output scale
            student.net.out.bias.zero_()
        loss = fit_student(student, states, actions, seed=seed)
        cost, spread = true_cost(scenario, student, seed=seed)
        print(f"  student seed {seed}: regression MSE {loss:8.4f}   "
              f"cost on the plant {cost:6.2f} +- {spread:.2f}")

    print("\n  a cost of ~13 is a swing-up, ~40 is 'the pole has fallen'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
