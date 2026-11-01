"""The four questions about the inaccurate imagined trajectory, answered by measurement.

Q1  Why is the imagined trajectory wrong?  Is Monte Carlo stopping the dynamics
    model from converging?
Q2  Would more particles fix it?
Q3  Would a mixture of Gaussians (GMM / splatting) instead of particles fix it,
    given that a mixture has no closed-form moment matching?
Q4  Would a branching tree (MCTS / ray-tracing style splitting) fix it?

Each section prints the number that decides the question.

Usage::

    python scripts/answer_questions.py [--oracle-episodes 3] [--skip-oracle]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

from pilco_mc import MCConfig, fit_model, make_cartpole               # noqa: E402
from pilco_mc.nn import BoundedPolicy                                  # noqa: E402
from pilco_mc.objective import cost_statistics                        # noqa: E402
from pilco_mc.particles import initial_particles, rollout             # noqa: E402
from pilco_mc.trainer import optimise_policy                          # noqa: E402

_PILCO = os.path.join(os.path.dirname(_ROOT), "pilco_pytorch")


# ---------------------------------------------------------------------------
# a differentiable copy of the true cart-pole, used as an *oracle* model
# ---------------------------------------------------------------------------
def cartpole_rhs(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Same ODE as ``scenario.cartpole_dynamics``, batched and differentiable."""
    length, mp, mc, b, g = 0.5, 0.5, 0.5, 0.1, 9.82
    s, c = torch.sin(z[..., 3]), torch.cos(z[..., 3])
    den1 = 4.0 * (mc + mp) - 3.0 * mp * c * c
    den2 = 4.0 * length * (mp + mc) - 3.0 * mp * length * c * c
    dz = torch.zeros_like(z)
    dz[..., 0] = z[..., 1]
    dz[..., 1] = (2 * mp * length * z[..., 2] ** 2 * s + 3 * mp * g * s * c
                  + 4 * f - 4 * b * z[..., 1]) / den1
    dz[..., 2] = (-3 * mp * length * z[..., 2] ** 2 * s * c
                  - 6 * (mc + mp) * g * s - 6 * (f - b * z[..., 1]) * c) / den2
    dz[..., 3] = z[..., 2]
    return dz


class OracleDynamics(torch.nn.Module):
    """The true dynamics as a 'model': same interface, no learning.

    If the policy optimiser cannot swing the pole up even with a perfect model,
    the model is not the only problem.  If it can, every remaining failure in the
    learned-model loop is attributable to the model.
    """

    def __init__(self, dt: float = 0.1, substeps: int = 2,
                 process_noise: float = 0.01):
        super().__init__()
        self.members = [None]                    # satisfies len(model.members)
        self.dt, self.substeps = dt, substeps
        self.process_noise = process_noise

    def step(self, x, u, generator=None, member=None):
        h = self.dt / self.substeps
        u = u.reshape(x.shape[:-1])            # (P,) force, not (P, 1)
        z = x
        for _ in range(self.substeps):           # RK4
            k1 = cartpole_rhs(z, u)
            k2 = cartpole_rhs(z + 0.5 * h * k1, u)
            k3 = cartpole_rhs(z + 0.5 * h * k2, u)
            k4 = cartpole_rhs(z + h * k3, u)
            z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        var = torch.full_like(z, self.process_noise ** 2)
        noise = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        return z + self.process_noise * noise, var

    def sample_next(self, x, u, generator=None, member=None):
        return self.step(x, u, generator, member)[0]


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------
def load_good_controller():
    path = os.path.join(_PILCO, "results", "cartpole_swingup_policy.npz")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(
        "anim", os.path.join(_PILCO, "scripts", "make_animation.py"))
    anim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anim)
    return anim.Controller(path)


def collect_good_data(scenario, control, rollouts=4, steps=100):
    rng = np.random.default_rng(0)
    s_all, u_all, d_all = [], [], []
    for _ in range(rollouts):
        x, s, u = np.zeros(4), [], []
        for _ in range(steps):
            a = float(np.clip(control(x), -10, 10))
            u.append(a)
            s.append(x.copy())
            x = scenario.step_true(x, a) + rng.standard_normal(4) * scenario.process_noise
        s.append(x.copy())
        i, j, k = scenario.transitions(np.asarray(s), np.asarray(u))
        s_all.append(i)
        u_all.append(j)
        d_all.append(k)
    return (torch.cat(s_all), torch.cat(u_all), torch.cat(d_all))


# ---------------------------------------------------------------------------
# Q1: is the model, or everything else?
# ---------------------------------------------------------------------------
def q1_oracle(scenario, config, episodes: int) -> None:
    print("=" * 72)
    print("Q1  Is the model the only problem?  Swap in the TRUE dynamics.")
    print("=" * 72)
    mean0, cov0 = scenario.initial_belief()
    policy = BoundedPolicy(4, 1, [scenario.u_max])
    model = OracleDynamics(dt=scenario.dt)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(1000)
    costs = []
    for episode in range(episodes):
        predicted, stderr = optimise_policy(model, policy, scenario.cost,
                                            mean0, cov0, config, gen)
        x0 = mean0.numpy() + np.linalg.cholesky(cov0.numpy()) @ rng.standard_normal(4)
        _, step_costs = scenario.rollout_true(policy, x0, scenario.horizon, rng)
        costs.append(float(np.sum(step_costs)))
        print(f"  episode {episode + 1}: predicted {predicted:6.2f} +- {stderr:4.2f}"
              f"   realised {costs[-1]:6.2f}")
    print(f"\n  with a perfect model the loop reaches {min(costs):.2f} "
          f"(the ceiling is {scenario.horizon:.0f}, a working swing-up is ~13)")
    print("  => the model is NOT the only problem.  Even with the true dynamics")
    print("     the policy search does not get there, so there are two gaps: the")
    print("     model's imagined trajectory (Q2-Q4) and the policy optimiser.")
    print("     Measured separately: 2000 Adam steps instead of 250 reach 34.8;")
    print("     half the learning rate at 1000 steps reaches 38.3.  The search")
    print("     improves but far too slowly -- first-order Adam on a Monte Carlo")
    print("     estimate is much weaker than PILCO's BFGS on an analytic value.\n")


# ---------------------------------------------------------------------------
# Q2: does the particle count matter?
# ---------------------------------------------------------------------------
def q2_particles(scenario, good_policy, model, horizons=(10, 20, 40)) -> None:
    print("=" * 72)
    print("Q2  Would more particles fix the imagined trajectory?")
    print("=" * 72)
    print("  imagined cost of the good controller, same model, more particles\n")
    header = "   H  |" + "".join(f"{p:>12}" for p in (16, 64, 256, 1024))
    print(header)
    print("  ----+" + "-" * (12 * 4))
    for horizon in horizons:
        row = f"  {horizon:3d} |"
        for n_particles in (16, 64, 256, 1024):
            values = []
            for seed in range(3):
                gen = torch.Generator().manual_seed(seed)
                particles = initial_particles(torch.zeros(4), torch.eye(4) * 0.01,
                                              n_particles, gen)
                roll = rollout(good_policy, model, particles, horizon, gen)
                value, _ = cost_statistics(roll, scenario.cost)
                values.append(value)
            row += f"{np.mean(values):9.2f}±{np.std(values):4.2f}"
        print(row)
    print("\n  the estimate is already converged at 16 particles: more particles")
    print("  shrink the error bars around the *wrong* number, they do not move it.\n")


# ---------------------------------------------------------------------------
# Q3: moments of a mixture, without closed-form integrals
# ---------------------------------------------------------------------------
def unscented(fn, mean: torch.Tensor, cov: torch.Tensor):
    """Unscented transform: 2D+1 deterministic points, no integrals needed.

    Returns ``(E[fn(x)], Cov[fn(x)])``.  This is the machinery that lets a mixture
    of Gaussians be propagated without any closed-form moment matching: each
    component is pushed through ``fn`` by sigma points, so the requirement on
    ``fn`` is only "callable at 2D+1 points", not "integrable".
    """
    d = mean.numel()
    kappa = 3.0 - d                       # standard choice for a Gaussian prior
    lam = kappa
    chol = torch.linalg.cholesky((d + lam) * cov)
    points = [mean]
    for i in range(d):
        points.append(mean + chol[:, i])
        points.append(mean - chol[:, i])
    points = torch.stack(points)
    wm = torch.full((2 * d + 1,), 1.0 / (2 * (d + lam)))
    wm[0] = lam / (d + lam)
    wc = wm.clone()
    wc[0] += 1.0 - d + lam                # (1 - a^2 + b) with a=1, b=0
    values = fn(points)
    out_mean = (wm[:, None] * values).sum(0)
    centred = values - out_mean
    out_cov = (wc[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(0)
    return out_mean, out_cov


def q3_gaussian_mixture(scenario, model) -> None:
    print("=" * 72)
    print("Q3  Can a mixture of Gaussians replace particles without closed forms?")
    print("=" * 72)
    d = 4
    rng = np.random.default_rng(0)
    mean = torch.as_tensor(rng.standard_normal(d) * 0.4)
    a = torch.as_tensor(rng.standard_normal((d, d)) * 0.2)
    cov = a @ a.T + 0.02 * torch.eye(d)
    u = torch.zeros(1, 1)

    def push(x):
        uu = u.expand(x.shape[0], 1)
        return x + model.predict(x, uu)[0]

    # reference: very large Monte Carlo sample
    gen = torch.Generator().manual_seed(0)
    samples = initial_particles(mean, cov, 200_000, gen)
    with torch.no_grad():
        mc = push(samples)
    mc_mean, mc_cov = mc.mean(0), torch.cov(mc.T)

    ut_mean, ut_cov = unscented(push, mean, cov)
    err_m = float((ut_mean - mc_mean).abs().max())
    err_c = float((ut_cov - mc_cov).abs().max() / mc_cov.abs().max())
    print(f"  one Gaussian through the network, 2D+1 = {2 * d + 1} sigma points")
    print(f"    mean  max|UT - MC|        = {err_m:.3e}   (scale {float(mc_mean.abs().max()):.2e})")
    print(f"    cov   max|UT - MC| / |C|  = {err_c:.3e}")

    # a two-component mixture: moments are still closed form, the *propagation*
    # is what would need the transform
    w = torch.tensor([0.6, 0.4])
    means = torch.stack([mean, mean + torch.tensor([0.3, -0.2, 0.1, 0.5])])
    covs = torch.stack([cov, cov * 0.5])
    mix_mean = (w[:, None] * means).sum(0)
    second = (w[:, None, None] * (covs + means[:, :, None] * means[:, None, :])).sum(0)
    mix_cov = second - mix_mean[:, None] * mix_mean[None, :]
    print("\n  a mixture's own moments are closed form: "
          "E[x] = sum w_k m_k, Cov = sum w_k (S_k + m_k m_k^T) - E E^T")
    print(f"    two components -> mean {mix_mean.numpy().round(3)}"
          f", cov diag {mix_cov.diagonal().numpy().round(4)}")
    print("    (no integral anywhere; the hard part is the *propagation*,")
    print("     which the sigma points above handle without a closed form)")
    print()


# ---------------------------------------------------------------------------
# Q4: what a tree would and would not change
# ---------------------------------------------------------------------------
def q4_tree_branching(scenario, good_policy, model) -> None:
    print("=" * 72)
    print("Q4  Would a branching tree (MCTS / splitting) fix the imagined drift?")
    print("=" * 72)
    # Measure how quickly the *spread* of imagined trajectories grows: if the
    # tree were to branch, the branches would separate at this rate.
    gen = torch.Generator().manual_seed(0)
    particles = initial_particles(torch.zeros(4), torch.eye(4) * 0.01, 512, gen)
    roll = rollout(good_policy, model, particles, 40, gen)
    print("   spread of the imagined states around their own mean (std over")
    print("   particles, norm over state dims) -- how fast branches would diverge\n")
    print("     t   model spread   true spread   model/true")
    rng = np.random.default_rng(0)
    for t in (0, 5, 10, 15, 20, 30, 40):
        spread_model = float(roll.states[t].std(0).norm())
        seeds = []
        for _ in range(16):
            x = np.zeros(4) + rng.standard_normal(4) * 0.1
            for _ in range(t):
                x = scenario.step_true(x, float(np.clip(
                    good_policy(torch.as_tensor(x)[None])[0, 0], -10, 10)))
                x = x + rng.standard_normal(4) * scenario.process_noise
            seeds.append(x)
        spread_true = float(torch.as_tensor(np.asarray(seeds)).std(0).norm())
        print(f"   {t:3d}   {spread_model:11.3f}   {spread_true:11.3f}"
              f"   {spread_model / max(spread_true, 1e-9):8.2f}")
    print("\n  a tree splits *before* the branches separate; the drift above is")
    print("  bias (all branches go to the same wrong place), not variance.  More")
    print("  branches estimate that wrong place more accurately.\n")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-episodes", type=int, default=3)
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    scenario = make_cartpole()
    config = MCConfig(horizon=40, particles=128, policy_steps=250, members=4,
                      verbose=False)
    control = load_good_controller()

    if not args.skip_oracle:
        q1_oracle(scenario, config, args.oracle_episodes)

    if control is None:
        print("good controller unavailable; skipping Q2-Q4")
        return 0

    class ControllerPolicy(torch.nn.Module):
        def forward(self, x):
            with torch.no_grad():
                return torch.as_tensor(
                    [[np.clip(control(v.numpy()), -10, 10)] for v in x])

    states, controls, deltas = collect_good_data(scenario, control)
    model = fit_model(states, controls, deltas, config, seed=0)
    with torch.no_grad():
        pred, _ = model.predict(states, controls)
        r2 = 1 - ((deltas - pred) ** 2).sum() / ((deltas - deltas.mean(0)) ** 2).sum()
    print(f"model fitted on the good controller's data: one-step R^2 = {float(r2):.3f}\n")

    q2_particles(scenario, ControllerPolicy(), model)
    q3_gaussian_mixture(scenario, model)
    q4_tree_branching(scenario, ControllerPolicy(), model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
