"""Does Monte Carlo reproduce the analytic moments it replaces?

This is the test the whole package rests on.  For every operator whose moments
``pilco_pytorch`` computes in closed form, sample the *same* distribution, push
the samples through the *same* nonlinearity, and check that the empirical mean
and covariance converge to the analytic ones within Monte Carlo error.

If these pass, replacing ``plan()`` with ``rollout()`` is a change of estimator,
not a change of model.

The reference package is optional: the tests that need it are skipped when
``../pilco_pytorch`` is not importable.

Usage::

    python tests/test_particles.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REF = os.path.join(os.path.dirname(_ROOT), "pilco_pytorch")
for p in (_ROOT, _REF):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_default_dtype(torch.float64)

from pilco_mc import make_cartpole                                # noqa: E402
from pilco_mc.particles import initial_particles, rollout          # noqa: E402

TOLERANCE_SE = 4.0          # max |empirical - analytic| in standard errors


def _gaussian(d, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(d)
    a = rng.standard_normal((d, d))
    s = a @ a.T + 0.35 * np.eye(d)
    return torch.as_tensor(m), torch.as_tensor(s)


def _check(name, analytic, samples):
    """Compare ``samples`` (P,) with an analytic mean / variance."""
    mean_a, var_a = analytic
    n = samples.numel()
    mean_e = samples.mean()
    var_e = samples.var(unbiased=True)
    se_mean = samples.std(unbiased=True) / n ** 0.5
    # standard error of the variance estimator for a roughly Gaussian sample
    se_var = var_e * (2.0 / max(n - 1, 1)) ** 0.5
    z_mean = abs(float(mean_e - mean_a)) / max(float(se_mean), 1e-12)
    z_var = abs(float(var_e - var_a)) / max(float(se_var), 1e-12)
    ok = z_mean < TOLERANCE_SE and z_var < TOLERANCE_SE
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} "
          f"mean z={z_mean:5.2f}  var z={z_var:5.2f}")
    return ok


# ---------------------------------------------------------------------------
def test_initial_particles(n=200_000):
    """Sampling a Gaussian must reproduce its mean and covariance."""
    m, s = _gaussian(4, seed=0)
    x = initial_particles(m, s, n, torch.Generator().manual_seed(0))
    err_m = float((x.mean(0) - m).abs().max())
    xc = x - x.mean(0)                       # the sample mean is not exactly m
    err_s = float((xc.T @ xc / n - s).abs().max())
    # Monte Carlo tolerance: each covariance entry has standard error
    # sqrt(2/n) * (its own scale), so compare against 6 of those
    scale = float(s.abs().max())
    tol_m = 6.0 * scale ** 0.5 / n ** 0.5
    tol_s = 6.0 * (2.0 / n) ** 0.5 * scale
    ok = err_m < tol_m and err_s < tol_s
    print(f"[{'PASS' if ok else 'FAIL'}] initial_particles            "
          f"mean err={err_m:.2e}/{tol_m:.2e}  cov err={err_s:.2e}/{tol_s:.2e}")
    return ok


def test_trig_moments(n=400_000):
    """``E[sin]``/``Var[sin]`` from samples vs the closed form in fourier.py."""
    try:
        from pilco.ops.fourier import TrigAugmentOp
    except ImportError:
        print("[SKIP] trig moments (pilco_pytorch not importable)")
        return True
    m, s = _gaussian(4, seed=1)
    op = TrigAugmentOp([1, 3], n_in=4)
    with torch.no_grad():
        analytic = op.moments(m, s)
        x = initial_particles(m, s, n, torch.Generator().manual_seed(1))
        g = torch.cat([torch.sin(x[:, [1]]), torch.cos(x[:, [1]]),
                       torch.sin(x[:, [3]]), torch.cos(x[:, [3]])], dim=1)
    ok = _check("E[sin(x1)]", (analytic.mean[0], analytic.cov[0, 0]), g[:, 0])
    ok &= _check("E[cos(x1)]", (analytic.mean[1], analytic.cov[1, 1]), g[:, 1])
    ok &= _check("E[sin(x3)]", (analytic.mean[2], analytic.cov[2, 2]), g[:, 2])
    # the cross-covariance is what makes the augmentation non-trivial
    cross = (g[:, 0] * g[:, 1]).mean() - g[:, 0].mean() * g[:, 1].mean()
    se = float(g[:, 0].std() * g[:, 1].std() / g.shape[0] ** 0.5)
    z = abs(float(cross) - float(analytic.cov[0, 1])) / max(se, 1e-12)
    ok &= z < TOLERANCE_SE
    print(f"[{'PASS' if z < TOLERANCE_SE else 'FAIL'}] Cov[sin,cos](x1)              "
          f"z={z:5.2f}")
    return ok


def test_cost_moments(n=1_200_000):
    """``E[c]`` and ``Var[c]`` from samples vs ``SaturatingCostOp.moments``."""
    try:
        from pilco.ops.cost import SaturatingCostOp
    except ImportError:
        print("[SKIP] cost moments (pilco_pytorch not importable)")
        return True
    m, s = _gaussian(3, seed=2)
    target = torch.tensor([0.3, -0.2, 0.5])
    weight = torch.diag(torch.tensor([2.0, 1.0, 0.5]))
    op = SaturatingCostOp(target, weight, [0.4])
    with torch.no_grad():
        value, variance = op.moments(m, s)
        x = initial_particles(m, s, n, torch.Generator().manual_seed(2))
        # the operator uses W = weight / width^2
        d2 = ((x - target) @ (weight / 0.4 ** 2) * (x - target)).sum(-1)
        c = 1.0 - torch.exp(-0.5 * d2)
    return _check("E[c]", (value, variance), c)


def test_gp_moments(n=20_000, n_train=25):
    """Sampling through the GP posterior vs ``GpDynamicsOp.moments``.

    This is the operator that replaces ``gp0d``'s closed-form integral: draw
    ``x_i ~ N(m, S)``, draw ``f_i | x_i`` from the GP posterior, and compare the
    empirical mean/covariance of ``{f_i}`` with the analytic ones.
    """
    try:
        from pilco.ops.gp import GpDynamicsOp
    except ImportError:
        print("[SKIP] GP moments (gpytorch / pilco_pytorch not importable)")
        return True

    rng = np.random.default_rng(3)
    d, E = 3, 2
    X = torch.as_tensor(rng.uniform(-1, 1, (n_train, d)))
    Y = torch.as_tensor(np.sin(2 * X.numpy()[:, :E]) + 0.05 * rng.standard_normal((n_train, E)))
    hyp = np.zeros((d + 2, E))
    hyp[:d] = np.log(rng.uniform(0.5, 1.5, (d, E)))
    hyp[d], hyp[d + 1] = 0.0, np.log(0.1)
    op = GpDynamicsOp.from_log_hyp(X, Y, torch.as_tensor(hyp))

    m, s = _gaussian(d, seed=3)
    with torch.no_grad():
        analytic = op.moments(m, s)
        x = initial_particles(m, s, n, torch.Generator().manual_seed(3))
        draws = []
        for e in range(E):
            model = op.models[e]
            model.eval()
            # Marginal sampling: f_i ~ N(m(x_i), s(x_i)).  Drawing the *joint*
            # posterior would need an n x n Cholesky for n = 20000; sampling each
            # point marginally is the same mixture and is what the particle
            # propagator does anyway (particles are independent).
            dist = model(x)
            mean = dist.mean
            sd = dist.variance.clamp(min=1e-12).sqrt()
            eps = torch.randn(mean.shape, generator=torch.Generator().manual_seed(e))
            draws.append(mean + sd * eps)
    ok = True
    for e in range(E):
        ok &= _check(f"E[f{e}]", (analytic.mean[e], analytic.cov[e, e]), draws[e])
    cross = ((draws[0] - draws[0].mean()) * (draws[1] - draws[1].mean())).mean()
    se = float(draws[0].std() * draws[1].std() / n ** 0.5)
    z = abs(float(cross) - float(analytic.cov[0, 1])) / max(se, 1e-12)
    ok &= z < TOLERANCE_SE
    print(f"[{'PASS' if z < TOLERANCE_SE else 'FAIL'}] Cov[f0,f1]                    "
          f"z={z:5.2f}")
    return ok


def test_gradient_flows():
    """The MC objective must be differentiable w.r.t. the policy parameters."""
    from pilco_mc import BoundedPolicy, DynamicsEnsemble
    from pilco_mc.objective import expected_cost
    from pilco_mc.particles import initial_particles, rollout

    torch.manual_seed(0)
    model = DynamicsEnsemble(4, 1, members=2, hidden=16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    policy = BoundedPolicy(4, 1, [10.0], width=16)
    m0 = torch.zeros(4)
    s0 = torch.eye(4) * 0.01
    gen = torch.Generator().manual_seed(0)
    particles = initial_particles(m0, s0, 32, gen)
    roll = rollout(policy, model, particles, 10, gen)
    loss = expected_cost(roll, make_cartpole().cost)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    ok = bool(grads) and all(torch.isfinite(g).all() for g in grads) \
        and any(float(g.abs().sum()) > 0 for g in grads)
    print(f"[{'PASS' if ok else 'FAIL'}] policy gradient flows       "
          f"({len(grads)} tensors, |g|={sum(float(g.abs().sum()) for g in grads):.3e})")
    return ok


def test_rollout_shape():
    from pilco_mc import BoundedPolicy, DynamicsEnsemble

    model = DynamicsEnsemble(4, 1, members=2, hidden=16)
    policy = BoundedPolicy(4, 1, [10.0], width=16)
    gen = torch.Generator().manual_seed(0)
    particles = initial_particles(torch.zeros(4), torch.eye(4), 16, gen)
    roll = rollout(policy, model, particles, 7, gen)
    ok = (tuple(roll.states.shape) == (8, 16, 4)
          and tuple(roll.controls.shape) == (7, 16, 1)
          and torch.isfinite(roll.states).all())
    print(f"[{'PASS' if ok else 'FAIL'}] rollout shapes              "
          f"{tuple(roll.states.shape)} {tuple(roll.controls.shape)}")
    return ok


def main() -> int:
    print("Monte Carlo vs analytic moments\n")
    results = [
        test_initial_particles(),
        test_trig_moments(),
        test_cost_moments(),
        test_gp_moments(),
        test_rollout_shape(),
        test_gradient_flows(),
    ]
    print("\nALL OK" if all(results) else "\nFAILURES PRESENT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
