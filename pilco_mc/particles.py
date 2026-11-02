"""Monte Carlo moment propagation: the replacement for PILCO's analytic planner.

PILCO's ``plan()`` carries a Gaussian ``(m_t, S_t)`` through the horizon and
computes, at every step, the *closed-form* moments of the next state.  That is
only possible because every stage is integrable, which in turn forces the model
class (ARD GP) and the controller class (RBF network).

Here the same object is estimated from samples.  A **particle** is one imagined
state; ``P`` particles are drawn from the initial belief and pushed through the
controller and the dynamics, and the mean, covariance and expected cost of the
trajectory are empirical averages over the particles:

    m_t   ~= (1/P) sum_p x_t^(p)
    S_t   ~= (1/P) sum_p (x_t^(p) - m_t)(x_t^(p) - m_t)^T
    E[c]  ~= (1/P) sum_p c(x_t^(p))

Two consequences matter:

1. **Nothing has to be integrable.**  The only requirement on the model is that
   it can be *sampled*, so it can be any differentiable network
   (:mod:`pilco_mc.nn`).
2. **The estimator is differentiable.**  Sampling is reparameterised
   (``x' = mu + sigma * eps``, ``eps ~ N(0, I)``), so the gradient of the
   estimated cost flows back through the horizon into the policy parameters.
   That is the whole trick behind MC-PILCO.

What replaces what
------------------
=========================  =====================================
``pilco.trajectory.plan``  :func:`rollout`
``pilco.ops.base.splice``  nothing -- particles carry the joint
                           distribution implicitly
``pilco.trajectory.mean_jacobian``  nothing -- autograd handles it
``pilco.objectives``       :mod:`pilco_mc.objective`
=========================  =====================================

Price of the exchange
---------------------
The analytic moments are exact to ~1e-12 and cost one pass; the Monte Carlo
estimate has standard error ``O(1/sqrt(P))`` and its *variance* is what limits
how aggressively the policy may be optimised.  ``objective.particle_dropout``
and the ensemble in :mod:`pilco_mc.nn` exist to control it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .nn import BoundedPolicy, DynamicsEnsemble

__all__ = ["Rollout", "ParticleBatch", "initial_particles", "draw_batch",
           "rollout", "rollout_batch", "moment_match"]


@dataclass
class Rollout:
    """Particle trajectories.

    Attributes
    ----------
    states:
        ``(H + 1, P, D)`` -- ``states[0]`` is the sampled initial belief.
    controls:
        ``(H, P, Du)`` -- the controls the policy applied.
    variances:
        ``(H, P, D)`` -- the predictive variance the model used at every step;
        an empty tensor when the caller did not ask for it.
    """

    states: torch.Tensor
    controls: torch.Tensor
    variances: Optional[torch.Tensor] = None

    @property
    def horizon(self) -> int:
        return self.controls.shape[0]

    @property
    def n_particles(self) -> int:
        return self.states.shape[1]

    def mean(self) -> torch.Tensor:
        return self.states.mean(1)

    def cov(self) -> torch.Tensor:
        """``(H + 1, D, D)`` empirical covariance at every step."""
        centred = self.states - self.states.mean(1, keepdim=True)
        return centred.transpose(1, 2) @ centred / (self.n_particles - 1)


def initial_particles(mean: torch.Tensor, cov: torch.Tensor, n: int,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Draw ``n`` samples of ``N(mean, cov)`` by reparameterisation.

    ``L @ z`` with ``z ~ N(0, I)`` keeps the samples differentiable with respect
    to ``mean``/``cov``, which is what lets a caller optimise an initial state
    distribution (PILCO never needed that, but a learned initial belief is free
    here).
    """
    d = mean.numel()
    chol = torch.linalg.cholesky(cov + 1e-12 * torch.eye(d, dtype=cov.dtype))
    z = torch.randn(n, d, dtype=mean.dtype, generator=generator)
    return mean + z @ chol.transpose(0, 1)


def rollout(policy: BoundedPolicy, model: DynamicsEnsemble,
            particles: torch.Tensor, horizon: int,
            generator: Optional[torch.Generator] = None,
            members: Optional[torch.Tensor] = None) -> Rollout:
    """Push ``particles`` through ``policy`` and ``model`` for ``horizon`` steps.

    Gradients flow through the policy; the model is expected to be frozen (its
    parameters are the *model* of the environment, not something to optimise
    here).  Sampling is reparameterised, so the result is a differentiable
    function of the policy parameters.

    Parameters
    ----------
    particles:
        ``(P, D)`` initial states.
    members:
        Optional ``(P,)`` ensemble indices.  Pinning one member per particle for
        the whole trajectory is the PETS / MC-PILCO convention: a particle is one
        imagined rollout under one model, so the ensemble spread becomes
        *epistemic* uncertainty along the trajectory instead of noise that
        averages out after a few steps.
    """
    n = particles.shape[0]
    if members is None:
        members = torch.randint(len(model.members), (n,), generator=generator)
    states = [particles]
    controls, variances = [], []
    x = particles
    for _ in range(horizon):
        u = policy(x)
        x, var = model.step(x, u, generator=generator, member=members)
        states.append(x)
        controls.append(u)
        variances.append(var)
    return Rollout(torch.stack(states), torch.stack(controls),
                   torch.stack(variances))


@torch.no_grad()
def moment_match(roll: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical ``(m_t, S_t)`` of a particle rollout -- PILCO's ``plan`` output.

    Provided for diagnostics and for comparing against the analytic propagator;
    the learning loop never needs it, because the objective is computed directly
    from the particles.
    """
    return roll.mean(), roll.cov()


@dataclass
class ParticleBatch:
    """A **fixed** set of particles, ensemble members and noise draws.

    The Monte Carlo objective is only a function of the policy parameters if the
    randomness is held fixed.  Resampling the particles at every optimisation
    step turns the objective into a noisy sequence, and a line search cannot
    compare two points of a noisy function -- which is why the first version of
    this project could only use Adam, and why it could not optimise hard enough.

    Holding the batch fixed is the "common random numbers" variance-reduction
    device: the objective becomes deterministic, its gradient is exact for that
    batch, and L-BFGS with a strong-Wolfe line search works.  The batch is
    re-drawn between optimisation rounds so the policy does not overfit one draw.

    Attributes
    ----------
    particles:
        ``(P, D)`` initial states.
    members:
        ``(P,)`` ensemble member per particle.
    noise:
        ``(H, P, D)`` standard-normal draws, one per step.
    """

    particles: torch.Tensor
    members: torch.Tensor
    noise: torch.Tensor


def draw_batch(mean: torch.Tensor, cov: torch.Tensor, n_particles: int,
               horizon: int, n_members: int,
               generator: Optional[torch.Generator] = None,
               antithetic: bool = False) -> ParticleBatch:
    """Draw one fixed batch (see :class:`ParticleBatch`).

    With ``antithetic`` the second half of the particles is the mirror of the
    first: identical initial state, negated noise.  The pair ``(x(eps),
    x(-eps))`` is then an antithetic pair, which cancels the first-order effect of
    the sampling noise.  Measured on this task: the noise of the objective falls
    3-11x and its signal-to-noise ratio rises 2-11x (at 40 points, 0.024 -> 0.002).
    """
    if antithetic:
        half = max(1, n_particles // 2)
        particles = initial_particles(mean, cov, half, generator)
        members = torch.randint(n_members, (half,), generator=generator)
        noise = torch.randn(horizon, half, mean.numel(), generator=generator)
        particles = torch.cat([particles, particles])
        members = torch.cat([members, members])
        noise = torch.cat([noise, -noise], dim=1)
        return ParticleBatch(particles, members, noise)
    particles = initial_particles(mean, cov, n_particles, generator)
    members = torch.randint(n_members, (n_particles,), generator=generator)
    noise = torch.randn(horizon, n_particles, mean.numel(), generator=generator)
    return ParticleBatch(particles, members, noise)


def rollout_batch(policy: BoundedPolicy, model: DynamicsEnsemble,
                  batch: ParticleBatch) -> Rollout:
    """Deterministic given ``batch``: same interface as :func:`rollout`."""
    states = [batch.particles]
    controls, variances = [], []
    x = batch.particles
    for t in range(batch.noise.shape[0]):
        u = policy(x)
        x, var = model.step(x, u, member=batch.members, noise=batch.noise[t])
        states.append(x)
        controls.append(u)
        variances.append(var)
    return Rollout(torch.stack(states), torch.stack(controls),
                   torch.stack(variances))
