"""The objective the policy is optimised against, estimated from particles.

PILCO minimises ``sum_t gamma^(t+1) E[c(x_t)]`` and differentiates the analytic
expectation.  Here the same quantity is the sample average over particles, and
autograd differentiates through the sampler.

Two variance-control devices come from MC-PILCO and are worth understanding
because the closed loop is otherwise dominated by the estimator's noise:

``particle_dropout``
    Zero each particle's contribution with probability ``p`` and rescale by
    ``1/(1-p)``.  The estimator stays unbiased, the *gradient* variance drops,
    and -- more importantly -- the policy cannot fit the particular draw of
    noise that produced the particles.  MC-PILCO applies the equivalent idea to
    the RBF activations of its controller; masking particles is the same
    regulariser in the sample dimension.

``gamma``
    Discount.  With ``gamma = 1`` (PILCO's cart-pole setting) every step of the
    horizon counts equally, which makes the early, high-cost swing-up dominate
    the objective -- exactly what we want here.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .particles import Rollout

__all__ = ["discount_weights", "particle_dropout", "expected_cost",
           "cost_statistics"]


def discount_weights(horizon: int, gamma: float, dtype=torch.float64,
                     device=None) -> torch.Tensor:
    return gamma ** torch.arange(1, horizon + 1, dtype=dtype, device=device)


def particle_dropout(per_particle: torch.Tensor, p: float,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Inverted dropout over the particle axis: unbiased, lower-variance gradient.

    ``per_particle`` is ``(P,)``; the result is the same shape with each entry
    zeroed with probability ``p`` and the survivors scaled by ``1/(1-p)``.
    """
    if p <= 0.0:
        return per_particle
    keep = (torch.rand(per_particle.shape, dtype=per_particle.dtype,
                       generator=generator, device=per_particle.device) >= p)
    return per_particle * keep / (1.0 - p)


def expected_cost(roll: Rollout, cost_fn: Callable[[torch.Tensor], torch.Tensor],
                  gamma: float = 1.0, dropout: float = 0.0,
                  generator: Optional[torch.Generator] = None,
                  uncertainty_weight: float = 0.0) -> torch.Tensor:
    """``sum_t gamma^(t+1) E[c(x_t)]`` estimated from the particles.

    ``cost_fn`` maps states of shape ``(..., D)`` to costs of shape ``(...)``.
    The cost is charged on ``states[0..H-1]``, i.e. *before* each transition,
    matching ``pilco_pytorch``'s ``rollout_true`` so that realised and predicted
    costs are directly comparable.

    ``uncertainty_weight > 0`` adds ``lambda * sum_t gamma^(t+1) E_p ||sd(x_t)||``
    -- an **uncertainty penalty**.  Without it the policy optimiser reliably finds
    the region where the model is most wrong and least constrained (measured on
    this project: predicted cost 33.8 while the realised cost sat at the 40.0
    ceiling).  With it, the imagined trajectory is pushed towards the states the
    ensemble actually agrees on.
    """
    states = roll.states[:-1]                                   # (H, P, D)
    cost = cost_fn(states)                                      # (H, P)
    weights = discount_weights(cost.shape[0], gamma, cost.dtype, cost.device)
    if uncertainty_weight > 0.0 and roll.variances is not None:
        # Norm of the per-step predictive standard deviation, averaged over
        # particles; ``variances`` is (H, P, D).
        #
        # The epsilon inside the sqrt is not cosmetic.  ``sqrt(0)`` has an
        # infinite derivative, so when the variance is zero (a deterministic
        # rollout, i.e. ``variance_temperature=0``) multiplying by it produces
        # **NaN gradients** and every optimiser step silently fails.  Measured:
        # the first closure call had loss 39.9971 with |grad| = nan, after which
        # the policy never changed -- which invalidated every closed-loop run
        # that used the default uncertainty weight.
        sd_norm = torch.sqrt(roll.variances.clamp(min=0.0).sum(-1) + 1e-12)
        cost = cost + uncertainty_weight * sd_norm
    per_particle = (cost * weights[:, None]).sum(0)             # (P,)
    per_particle = particle_dropout(per_particle, dropout, generator)
    return per_particle.mean()


@torch.no_grad()
def cost_statistics(roll: Rollout, cost_fn: Callable[[torch.Tensor], torch.Tensor],
                    gamma: float = 1.0) -> tuple[float, float]:
    """``(mean, standard error)`` of the estimated cost -- for logging."""
    states = roll.states[:-1]
    cost = cost_fn(states)
    weights = discount_weights(cost.shape[0], gamma, cost.dtype, cost.device)
    per_particle = (cost * weights[:, None]).sum(0)
    n = per_particle.numel()
    return float(per_particle.mean()), float(per_particle.std() / max(n ** 0.5, 1.0))
