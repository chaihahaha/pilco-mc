"""A differentiable copy of the true cart-pole, used for shooting and diagnostics.

Two uses:

* **shooting** -- optimise the 40 controls directly against the true dynamics.
  This is the classic way to solve a swing-up and it has none of the problems of
  policy search: 40 parameters, no plateau, and an exact gradient.  Its solution
  is then distilled into a state-feedback policy (see
  :func:`pilco_mc.trainer.warm_start_from_shooting`);
* **oracle** -- swap the learned model for the true one, to separate "the model
  is wrong" from "the search is weak".

The integrator is RK4 with ``substeps`` per control interval.  It is *not* the
same code path as :meth:`pilco_mc.scenario.CartPole.step_true` (scipy's adaptive
RK45); with ``substeps=4`` the two agree to ~1e-6 over a 40-step horizon, which
is far below the model error we care about.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["cartpole_rhs", "rk4_step", "ShootingResult", "shoot",
           "OracleDynamics"]

DTYPE = torch.float64


def cartpole_rhs(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable cart-pole ODE (same constants as the scenario)."""
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


def rk4_step(z: torch.Tensor, u: torch.Tensor, dt: float,
             substeps: int = 20) -> torch.Tensor:
    """One zero-order-hold step of the true dynamics, differentiable in ``u``/``z``."""
    u = u.reshape(z.shape[:-1])
    h = dt / substeps
    for _ in range(substeps):
        k1 = cartpole_rhs(z, u)
        k2 = cartpole_rhs(z + 0.5 * h * k1, u)
        k3 = cartpole_rhs(z + 0.5 * h * k2, u)
        k4 = cartpole_rhs(z + h * k3, u)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


class ShootingResult:
    """A solved action sequence and the trajectory it produces."""

    def __init__(self, controls: torch.Tensor, states: torch.Tensor,
                 cost: float):
        self.controls = controls
        self.states = states
        self.cost = cost


def shoot(cost_fn, x0: torch.Tensor, horizon: int, u_max: float,
          dt: float = 0.1, iters: int = 400, restarts: int = 3,
          generator: Optional[torch.Generator] = None,
          substeps: int = 20) -> ShootingResult:
    """Optimise ``u_0..u_{H-1}`` directly against the true dynamics.

    This is the escape hatch from the policy-search plateau documented in
    ``docs/NOTES.md``: from the hanging state every gentle policy costs 1.0 per
    step, so the gradient with respect to *policy* parameters vanishes.  In
    control space the same problem is 40-dimensional, has an informative
    gradient (push the cart, the pole swings), and L-BFGS solves it.

    Several random restarts are used because a single start can stall on the
    upside-down symmetry.
    """
    best: Optional[ShootingResult] = None
    # a pendulum of length 0.5 has omega = sqrt(g / l) ~ 4.4 rad/s; pumping
    # energy means pushing the cart at that frequency, so a sinusoid is a far
    # better starting guess than white noise
    omega = float(torch.sqrt(torch.tensor(9.82 / 0.5)))
    tau = torch.arange(horizon, dtype=DTYPE) * dt
    for restart in range(restarts):
        if restart == 0:
            raw = torch.atanh((torch.sin(omega * tau)).clamp(-0.99, 0.99))
        elif restart == 1:
            raw = torch.atanh((-torch.sin(omega * tau)).clamp(-0.99, 0.99))
        else:
            raw = 0.5 * torch.randn(horizon, dtype=DTYPE, generator=generator)
        # ``u = u_max * tanh(raw)``: the bound is smooth, so the gradient does
        # not vanish when the actuator saturates.  Clamping did, and that is
        # exactly where a swing-up spends most of its time (measured: 25.4 with
        # clamp against ~13 for a real swing-up).
        raw = raw.requires_grad_(True)
        optimiser = torch.optim.LBFGS([raw], lr=1.0, max_iter=iters,
                                      history_size=40,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-10,
                                      tolerance_change=1e-14)

        def closure():
            optimiser.zero_grad()
            x = x0.clone()
            total = torch.zeros((), dtype=x0.dtype)
            for t in range(horizon):
                total = total + cost_fn(x).sum()
                x = rk4_step(x, u_max * torch.tanh(raw[t]), dt, substeps)
            total.backward()
            return total

        optimiser.step(closure)
        with torch.no_grad():
            uc = u_max * torch.tanh(raw.detach())
            x = x0.clone()
            states = [x.clone()]
            total = torch.zeros((), dtype=x0.dtype)
            for t in range(horizon):
                total = total + cost_fn(x).sum()
                x = rk4_step(x, uc[t], dt, substeps)
                states.append(x.clone())
            result = ShootingResult(uc, torch.stack(states), float(total))
        if best is None or result.cost < best.cost:
            best = result
    return best


class OracleDynamics(torch.nn.Module):
    """The true dynamics behind the ``DynamicsEnsemble`` interface.

    Used to answer "would this work with a perfect model?".  ``members`` is a
    one-element list so that ``len(model.members)`` works in the sampler.
    """

    def __init__(self, dt: float = 0.1, substeps: int = 20,
                 process_noise: float = 0.01):
        super().__init__()
        self.members = [None]
        self.dt, self.substeps = dt, substeps
        self.process_noise = process_noise

    def step(self, x: torch.Tensor, u: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             member: Optional[torch.Tensor] = None,
             noise: Optional[torch.Tensor] = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        z = rk4_step(x, u, self.dt, self.substeps)
        var = torch.full_like(z, self.process_noise ** 2)
        if noise is None:
            noise = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        return z + self.process_noise * noise, var

    def predict(self, x: torch.Tensor, u: torch.Tensor):
        """Deterministic successor mean/variance (the ``DynamicsEnsemble`` API)."""
        z = rk4_step(x, u, self.dt, self.substeps)
        return z - x, torch.full_like(x, self.process_noise ** 2)
