"""The full moment-matching propagation law, for *any* model, via sigma points.

``splice`` in PILCO needs exactly three quantities for each nonlinear operator
``g`` under a Gaussian input ``N(m, S)``:

    E[g(x)],   Cov[g(x)],   C = S^{-1} Cov[x, g]

PILCO obtains them in closed form, which is what forced it to use a GP: the
nonlinearities have to be integrable.  They do not have to be.  The **unscented
transform** gives all three from ``2n + 1`` deterministic evaluations of ``g``,

    X_0 = m,  X_i = m +- sqrt((n + lambda) S)_i,
    E[g]   = sum_i w_i^m g(X_i),
    Cov[g] = sum_i w_i^c (g(X_i) - E[g])(g(X_i) - E[g])^T + Sigma,
    Cov[x,g] = sum_i w_i^c (X_i - m)(g(X_i) - E[g])^T,

which is exact for the mean to third order and for the covariance to second.  It
is deterministic (no Monte Carlo noise), costs a fixed 2n+1 forward passes, and
makes no assumption about ``g`` beyond being callable -- a network, a simulator,
anything.

Applied through the same augmented-state bookkeeping PILCO uses

    [ state | trig | u_raw | u | dyn ]

this reproduces the *whole* propagation law -- including the marginal-versus-
conditional choice that turned out to matter -- with a neural network in place of
the GP.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

DTYPE = torch.float64

__all__ = ["unscented_moments", "splice_ut", "UnscentedRollout",
           "unscented_value"]


def unscented_moments(fn: Callable[[torch.Tensor], torch.Tensor],
                      mean: torch.Tensor, cov: torch.Tensor,
                      noise: torch.Tensor | None = None,
                      alpha: float = 1.0, beta: float = 0.0,
                      kappa: float = 0.0):
    """``(E[g], Cov[g], S^{-1}Cov[x,g])`` for a callable ``fn``.

    The third return value uses PILCO's convention (already multiplied by
    ``S^{-1}``), so it can be passed straight to a ``splice``-style assembly.
    """
    # alpha=1, beta=0, kappa=0 gives lambda=0: the central weight vanishes and
    # every remaining weight is 1/(2n) > 0.  The textbook alpha=0.3 is not usable
    # here -- for n=1 it gives a central weight of -10.1, and one curved operator
    # (the saturation) then inflates the covariance 58x (0.072 -> 4.18, measured).
    n = mean.numel()
    lam = alpha ** 2 * (n + kappa) - n
    scale = n + lam
    # A symmetric square root.  The splice assembly and a network's
    # heteroscedastic head can both produce a matrix that is only PSD up to
    # round-off, so escalate the ridge until the factorisation succeeds.
    eye = torch.eye(n, dtype=cov.dtype)
    S, L = None, None
    for jitter in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        S = cov + jitter * eye
        S = (S + S.T) / 2.0
        try:
            L = torch.linalg.cholesky(scale * S)
            break
        except Exception:
            continue
    if L is None:                      # give up on the spread, keep the mean
        S = torch.eye(n, dtype=cov.dtype)
        L = torch.linalg.cholesky(scale * S)

    points = [mean]
    for i in range(n):
        points.append(mean + L[:, i])
        points.append(mean - L[:, i])
    stack = torch.stack(points)                                   # (2n+1, n)

    wm = torch.full((2 * n + 1,), 1.0 / (2.0 * scale), dtype=mean.dtype)
    wc = wm.clone()
    wm[0] = lam / scale
    wc[0] = lam / scale + (1.0 - alpha ** 2 + beta)

    ys = torch.stack([fn(p).reshape(-1) for p in stack])          # (2n+1, k)
    m_out = (wm[:, None] * ys).sum(0)

    dx = stack - mean
    dy = ys - m_out
    cov_out = (wc[:, None, None] * dy[:, :, None] * dy[:, None, :]).sum(0)
    if noise is not None:
        cov_out = cov_out + noise
    cross_xy = (wc[:, None, None] * dx[:, :, None] * dy[:, None, :]).sum(0)
    cross = torch.linalg.solve(S, cross_xy)
    return m_out, cov_out, cross


def splice_ut(mean: torch.Tensor, cov: torch.Tensor,
              fn: Callable[[torch.Tensor], torch.Tensor],
              in_idx: Sequence[int], out_idx: Sequence[int],
              noise: torch.Tensor | None = None):
    """Insert ``fn``'s output into an augmented Gaussian -- PILCO's ``splice``,
    with the moments supplied by :func:`unscented_moments`."""
    in_idx = torch.as_tensor(list(in_idx))
    out_idx = torch.as_tensor(list(out_idx))
    m_out, cov_out, cross = unscented_moments(
        fn, mean[in_idx], cov[in_idx][:, in_idx], noise)

    proj = cov[:, in_idx] @ cross
    new_mean = mean.index_copy(0, out_idx, m_out)
    new_cov = cov.clone()
    new_cov = new_cov.index_copy(1, out_idx, proj)
    new_cov = new_cov.index_copy(0, out_idx, proj.transpose(0, 1))
    k = out_idx.numel()
    rows = out_idx.repeat_interleave(k)
    cols = out_idx.repeat(k)
    new_cov = new_cov.index_put((rows, cols), cov_out.reshape(-1))
    return new_mean, (new_cov + new_cov.T) / 2.0


class UnscentedRollout:
    """Closed-loop belief propagation with UT moments, any differentiable model.

    The layout mirrors PILCO's: ``[state (d) | trig (2a) | u_raw (du) | u (du) |
    dyn (d)]``, filled in the same four splices plus the final projection.
    """

    def __init__(self, state_dim: int, angle_indices: Sequence[int],
                 policy: Callable, model, u_max: float):
        self.d = state_dim
        self.angles = list(angle_indices)
        self.policy = policy
        self.model = model
        self.u_max = u_max
        self.a = len(self.angles)
        self.i_state = list(range(state_dim))
        self.i_trig = list(range(state_dim, state_dim + 2 * self.a))
        self.i_u_raw = [state_dim + 2 * self.a]
        self.i_u = [state_dim + 2 * self.a + 1]
        self.i_dyn = list(range(state_dim + 2 * self.a + 2,
                                state_dim + 2 * self.a + 2 + state_dim))
        self.n_aug = state_dim + 2 * self.a + 2 + state_dim
        # the controller reads [state | trig] with the angle replaced by its
        # sin/cos (PILCO's plant.poli), i.e. the angle column is dropped
        self.pi_idx = ([i for i in self.i_state if i not in
                        {a for a in self.angles}] + self.i_trig)
        # the dynamics reads [dyn inputs | u]; PILCO drops theta in favour of
        # sin/cos, matching the columns the model was fitted on
        self.gp_idx = ([i for i in self.i_state if i not in
                        {a for a in self.angles}] + self.i_trig + self.i_u)
        self.proj = torch.zeros(state_dim, self.n_aug, dtype=DTYPE)
        for j, i in enumerate(self.i_state):
            self.proj[j, i] = 1.0
        for j, i in enumerate(self.i_dyn):
            self.proj[j, i] = 1.0

    def step(self, mean: torch.Tensor, cov: torch.Tensor):
        aug_m = torch.cat([mean, torch.zeros(self.n_aug - self.d, dtype=DTYPE)])
        aug_S = torch.zeros(self.n_aug, self.n_aug, dtype=DTYPE)
        aug_S[:self.d, :self.d] = cov

        def trig(x):
            return torch.stack([f(x[a]) for a in self.angles
                                for f in (torch.sin, torch.cos)])

        aug_m, aug_S = splice_ut(aug_m, aug_S, trig, self.i_state, self.i_trig)
        aug_m, aug_S = splice_ut(aug_m, aug_S, self.policy, self.pi_idx,
                                 self.i_u_raw)

        def sat(x):
            return self.u_max * (9.0 * torch.sin(x) + torch.sin(3.0 * x)) / 8.0

        aug_m, aug_S = splice_ut(aug_m, aug_S, sat, self.i_u_raw, self.i_u)

        # The model works on the raw state [x, dx, dtheta]; the augmented slots
        # carry sin/cos, so recover theta.  atan2 is exact for |theta| < pi, which
        # is the whole range this task visits.
        def to_state(z):
            parts = z[:-1]
            sin_t, cos_t = parts[-2], parts[-1]
            return torch.cat([parts[:-2], torch.atan2(sin_t, cos_t)[None]])

        def dynamics(z):
            x4, u = to_state(z), z[-1:]
            delta, _ = self.model.predict(x4[None], u.reshape(1, -1))
            return delta[0]

        _, var_now = self.model.predict(
            to_state(aug_m[self.gp_idx])[None],
            aug_m[self.i_u].reshape(1, -1))
        aug_m, aug_S = splice_ut(aug_m, aug_S, dynamics, self.gp_idx,
                                 self.i_dyn,
                                 noise=torch.diag(var_now[0].clamp(min=0.0)))

        next_m = self.proj @ aug_m
        next_S = self.proj @ aug_S @ self.proj.T
        return next_m, (next_S + next_S.T) / 2.0

    def rollout(self, mean: torch.Tensor, cov: torch.Tensor, horizon: int):
        states = [(mean.clone(), cov.clone())]
        for _ in range(horizon):
            mean, cov = self.step(mean, cov)
            states.append((mean.clone(), cov.clone()))
        return states


def unscented_value(model, policy, mean0: torch.Tensor, cov0: torch.Tensor,
                    horizon: int, cost_fn, u_max: float | None = None,
                    angle_indices: Sequence[int] = ()):
    """PILCO's objective -- ``sum_t E[c(x_t)]`` -- computed for **any** model.

    The belief is propagated with sigma points (see :class:`UnscentedRollout`), so
    the value is deterministic and keeps the policy gradient, and the model can be
    a network: no closed-form nonlinearity moments are needed anywhere.

    This is the quantity whose absence made the Monte Carlo value useless at a
    small data budget -- measured at 39.99 / 39.71 / 39.99 / 39.96 for every
    episode where the analytic value gives 37.44 / 36.88 / 31.54 / 34.53.
    """
    if u_max is None:
        u_max = float(policy.u_max[0])
    angles = tuple(angle_indices) or tuple(getattr(policy, "angle_indices", ()))

    def controller(x5: torch.Tensor) -> torch.Tensor:
        return policy.on_augmented(x5[None])[0]

    rollout = UnscentedRollout(mean0.numel(), angles, controller, model, u_max)
    mean, cov = mean0.clone(), cov0.clone()
    total = torch.zeros((), dtype=mean0.dtype)
    for _ in range(horizon):
        mean, cov = rollout.step(mean, cov)
        step_cost, _, _ = unscented_moments(
            lambda x: cost_fn(x[None]).reshape(-1), mean, cov)
        total = total + step_cost.sum()
    return total
