"""Linearised (EKF-style) belief propagation for any differentiable model.

PILCO evaluates the expected cost under a belief propagated by *moment matching*:
for every nonlinearity it inserts the **marginal** covariance of the output,
``Cov[g(x)]``, plus the cross-covariance ``Cov[x, g]``.  The practical consequence
is that the belief stays informative along the horizon, so the expected cost stays
sensitive to the policy.

Sampling particles instead is *honest* and, at this data budget, useless: the cloud
wanders off the data, the model's prior variance inflates it, and the expected
cost saturates at 1.0 per step so every policy scores the same.  Measured in
PILCO's own loop, changing nothing but the value:

    analytic          37.44  36.88  31.54  34.53
    Monte Carlo P=16  39.99  39.71  39.99  39.96
    Monte Carlo P=256 39.99  39.71            (identical: not estimator noise)

The point of this module is that linearisation needs only a **Jacobian**, not a
closed-form integral.  Any differentiable model -- including a neural network --
has one, so PILCO's optimism can be reproduced without the analytic nonlinearity
moments that forced it to use a GP.

For a nonlinear map ``g`` the moments used here are

    E[g(x)]   ~ g(m)
    Cov[g(x)] ~ J S J^T + Sigma
    Cov[x, g] ~ S J^T

with ``J = dg/dm`` from autograd and ``Sigma`` the model's predictive covariance
where it has one (the GP does; a network exposes its heteroscedastic head).
"""
from __future__ import annotations

from typing import Callable

import torch

DTYPE = torch.float64

__all__ = ["jacobian", "linearised_rollout"]


def jacobian(fn: Callable[[torch.Tensor], torch.Tensor],
             x: torch.Tensor) -> torch.Tensor:
    """``dg/dx`` at ``x`` for a vector-valued ``fn``; shape ``(out, in)``."""
    x = x.detach().clone().requires_grad_(True)
    y = fn(x).reshape(-1)
    rows = [torch.autograd.grad(y[i], x, retain_graph=True,
                                create_graph=False)[0].detach()
            for i in range(y.numel())]
    return torch.stack(rows)


def linearised_rollout(model, policy, mean0: torch.Tensor,
                       cov0: torch.Tensor, horizon: int,
                       augment: Callable[[torch.Tensor], torch.Tensor]):
    """Propagate ``(mean, cov)`` through the closed loop for ``horizon`` steps.

    ``model.predict(x, u)`` must return the mean and variance of the *state
    increment*; ``augment`` maps a state ``(d,)`` to the controller input
    ``(1, k)`` (the trig augmentation).
    """
    mean, cov = mean0.clone().detach(), cov0.clone().detach()
    states = [(mean.clone(), cov.clone())]
    for _ in range(horizon):
        def step_map(x: torch.Tensor) -> torch.Tensor:
            u = policy(augment(x))[0]
            delta, _ = model.predict(x[None], u.reshape(1, -1))
            return x + delta[0]

        A = jacobian(step_map, mean)
        u_now = policy(augment(mean))
        _, variance = model.predict(mean[None], u_now)
        noise = torch.diag(variance[0].clamp(min=0.0))
        mean = step_map(mean).detach()
        cov = (A @ cov @ A.T + noise).detach()
        states.append((mean.clone(), cov.clone()))
    return states
