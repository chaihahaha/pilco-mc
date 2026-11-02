"""Is multi-step prediction the problem, and does PILCO do better at it?

The diagnosis in ``docs/NOTES.md`` says the learned model's 40-step imagination
drifts.  That only explains the failure if PILCO's model does *not* drift on the
same data -- PILCO also fits a one-step model, so if its imagination drifts just
as much, "multi-step accuracy" is not the differentiator and the diagnosis is
wrong.

This script settles it by fitting **both models on exactly the same data** and
comparing, for the same known-good controller:

* the 40-step **mean trajectory** against ground truth (multi-step prediction
  error);
* the **imagined cost** against the true cost;
* the **belief standard deviation** PILCO carries along (the thing the analytic
  planner can use and the sampler cannot).

Usage::

    python scripts/pilco_vs_nn_multistep.py
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

from pilco.gaussian import Gaussian                                  # noqa: E402
from pilco.models.gp import build_dynamics_op, fit_hyperparameters_pilco  # noqa: E402
from pilco.ops.policy import RbfPolicyOp                              # noqa: E402
from pilco.scenarios.cartpole import make_cartpole as pilco_scenario  # noqa: E402
from pilco.trajectory import PilcoModel, plan                         # noqa: E402

from pilco_mc import MCConfig, fit_model, make_cartpole               # noqa: E402

HORIZON = 40


def load_controller():
    path = os.path.join(_PILCO, "results", "cartpole_swingup_policy.npz")
    spec = importlib.util.spec_from_file_location(
        "anim", os.path.join(_PILCO, "scripts", "make_animation.py"))
    anim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anim)
    return anim.Controller(path)


def seed_data(scenario, rollouts=5, steps=100, seed=1000):
    """Random-control rollouts -- the data both models are fitted on."""
    rng = np.random.default_rng(seed)
    states, controls, deltas = [], [], []
    for _ in range(rollouts):
        x, s, u = np.zeros(4), [], []
        for _ in range(steps):
            a = float(rng.uniform(-scenario.u_max, scenario.u_max))
            u.append(a)
            s.append(x.copy())
            x = scenario.step_true(x, a) + rng.standard_normal(4) * scenario.process_noise
        s.append(x.copy())
        i, j, k = scenario.transitions(np.asarray(s), np.asarray(u))
        states.append(i)
        controls.append(j)
        deltas.append(k)
    return torch.cat(states), torch.cat(controls), torch.cat(deltas)


def true_trajectory(scenario, control, steps=HORIZON):
    x = np.zeros(4)
    states, costs = [x.copy()], []
    for _ in range(steps):
        costs.append(float(scenario.cost(torch.as_tensor(x))))
        x = scenario.step_true(x, float(np.clip(control(x), -scenario.u_max,
                                                scenario.u_max)))
        states.append(x.copy())
    return np.asarray(states), float(np.sum(costs))


def main() -> int:
    scenario = make_cartpole()
    control = load_controller()
    states, controls, deltas = seed_data(scenario)
    true_states, true_cost = true_trajectory(scenario, control)
    print(f"data: {states.shape[0]} transitions from random controls")
    print(f"the good controller's TRUE cost over {HORIZON} steps: {true_cost:.2f}\n")
    x0_np = torch.zeros(4)
    eye = torch.eye(4) * 0.01

    # ---------------- PILCO: GP + analytic moment propagation -------------
    # PILCO's dynamics GP reads [x, dx, dtheta, sin(theta), cos(theta), u], i.e.
    # the trig-augmented state (plant.dyni = [0 1 2 4 5]); the neural model reads
    # [x, u] and learns the trig itself.  Same data, different feature maps.
    theta = states[:, 3:4]
    pilco_inputs = torch.cat([states[:, :3], torch.sin(theta), torch.cos(theta),
                              controls], dim=1)
    op = build_dynamics_op(pilco_inputs, deltas)
    fit_hyperparameters_pilco(op, iterations=150, MFEPLS=30)
    npz = np.load(os.path.join(_PILCO, "results", "cartpole_swingup_policy.npz"))
    centers = torch.as_tensor(npz["centers"])
    ell = torch.as_tensor(npz["lengthscales"])
    gram = torch.exp(-0.5 * (((centers[:, None, :] - centers[None]) / ell) ** 2).sum(-1))
    A = gram + 0.01 ** 2 * torch.eye(len(centers))
    targets = A @ torch.as_tensor(npz["targets"])       # so that beta matches
    policy_op = RbfPolicyOp(centers, targets, ell, noise=0.01, signal=1.0)
    pmodel = PilcoModel(pilco_scenario().spec, op, policy_op,
                        pilco_scenario().cost, max_u=[10.0])
    with torch.no_grad():
        ptraj = plan(pmodel, Gaussian(x0_np, eye), HORIZON, with_jacobians=False)
    pilco_cost = float(ptraj.value())
    pilco_mean = np.stack([s.mean.numpy() for s in ptraj.states])
    pilco_sd = np.asarray([float(np.sqrt(np.clip(np.diagonal(s.cov.numpy()), 0, None).sum()))
                           for s in ptraj.states])

    # ---------------- the neural model: same data --------------------------
    config = MCConfig(horizon=HORIZON, members=4, verbose=False)
    nn = fit_model(states, controls, deltas, config, seed=0)
    with torch.no_grad():
        x = x0_np.clone()[None]
        nn_mean, nn_cost = [x[0].numpy()], 0.0
        for t in range(HORIZON):
            nn_cost += float(scenario.cost(x).sum())
            u = torch.as_tensor([[np.clip(control(x[0].numpy()), -10, 10)]])
            x = x + nn.predict(x, u)[0]
            nn_mean.append(x[0].numpy())
    nn_mean = np.stack(nn_mean)

    # ---------------- comparison -------------------------------------------
    print(f"{'t':>3} | {'PILCO mean err':>14} {'belief sd':>10} | "
          f"{'NN mean err':>12} | {'|true|':>8}")
    print("-" * 62)
    for t in (0, 5, 10, 15, 20, 30, 40):
        pe = float(np.linalg.norm(pilco_mean[t] - true_states[t]))
        ne = float(np.linalg.norm(nn_mean[t] - true_states[t]))
        print(f"{t:3d} | {pe:14.3f} {pilco_sd[t]:10.3f} | {ne:12.3f} | "
              f"{np.linalg.norm(true_states[t]):8.3f}")
    print()
    print(f"imagined cost of the good controller:")
    print(f"  PILCO (analytic moments): {pilco_cost:7.2f}")
    print(f"  neural network (mean)   : {nn_cost:7.2f}")
    print(f"  truth                   : {true_cost:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
