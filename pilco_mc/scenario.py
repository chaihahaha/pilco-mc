"""Cart-pole: the true dynamics, the task cost and the data plumbing.

Deliberately identical to ``pilco_pytorch/pilco/scenarios/cartpole.py`` -- same
ODE constants, same cost, same initial belief -- so the two packages can be
compared on the same task.  The one difference is that no ``sin``/``cos``
augmentation is needed anywhere: with particles there is no Gaussian integral to
keep closed-form, so the networks see the raw state ``[x, dx, dtheta, theta]``.

The state is ``[x, dx, dtheta, theta]`` and the control is the cart force in
``[-u_max, u_max]``.  ``theta = 0`` is hanging down, ``theta = pi`` is upright.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_ivp

__all__ = ["CartPole", "make_cartpole"]

DTYPE = torch.float64


def cartpole_dynamics(t: float, z: np.ndarray, f: float) -> np.ndarray:
    """Continuous-time cart-pole dynamics (Deisenroth thesis, Appendix C)."""
    length, mp, mc, b, g = 0.5, 0.5, 0.5, 0.1, 9.82
    z = np.asarray(z, dtype=np.float64).ravel()
    s, c = np.sin(z[3]), np.cos(z[3])
    den1 = 4.0 * (mc + mp) - 3.0 * mp * c * c
    den2 = 4.0 * length * (mp + mc) - 3.0 * mp * length * c * c
    dz = np.zeros(4)
    dz[0] = z[1]
    dz[1] = (2 * mp * length * z[2] ** 2 * s + 3 * mp * g * s * c
             + 4 * f - 4 * b * z[1]) / den1
    dz[2] = (-3 * mp * length * z[2] ** 2 * s * c - 6 * (mc + mp) * g * s
             - 6 * (f - b * z[1]) * c) / den2
    dz[3] = z[2]
    return dz


@dataclass
class CartPole:
    horizon: int = 40
    dt: float = 0.1
    length: float = 0.5
    width: float = 0.25
    u_max: float = 10.0
    process_noise: float = 0.01
    init_variance: float = 0.01

    state_dim: int = 4
    control_dim: int = 1

    # ------------------------------------------------------------------
    #: ``"swingup"`` starts hanging (``theta = 0``), ``"balance"`` starts near
    #: upright (``theta = pi``).  The distinction matters: the balance task only
    #: needs the model to be accurate for a few steps, the swing-up needs ~15.
    task: str = "swingup"

    def initial_belief(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The belief the policy is planned from and evaluated at."""
        mean = torch.zeros(self.state_dim, dtype=DTYPE)
        if self.task == "balance":
            mean[3] = np.pi
        return (mean,
                torch.eye(self.state_dim, dtype=DTYPE) * self.init_variance)

    # ------------------------------------------------------------------
    def cost(self, state: torch.Tensor) -> torch.Tensor:
        """``1 - exp(-d^2 / 2 w^2)`` on states of shape ``(..., 4)``.

        ``d`` is the distance from the pole tip to the upright target at
        ``(0, length)``; the cost is 0 at the goal and ~1 when hanging.
        """
        x, theta = state[..., 0], state[..., 3]
        d2 = ((x + self.length * torch.sin(theta)) ** 2
              + self.length ** 2 * (torch.cos(theta) + 1.0) ** 2)
        return 1.0 - torch.exp(-0.5 * d2 / self.width ** 2)

    # ------------------------------------------------------------------
    def step_true(self, z: np.ndarray, u: float) -> np.ndarray:
        """One ``dt`` step of the true system with a zero-order-hold control."""
        sol = solve_ivp(lambda t, y: cartpole_dynamics(t, y, u), (0.0, self.dt),
                        np.asarray(z, dtype=np.float64).ravel(), method="RK45",
                        t_eval=[self.dt], rtol=1e-12, atol=1e-12)
        return sol.y[:, -1]

    def rollout_true(self, policy, x0: np.ndarray, steps: int,
                     rng: np.random.Generator, with_noise: bool = True):
        """Roll the policy out on the **true** system; returns states and costs.

        Process noise is added to every component of the successor state, and the
        noisy states are what get logged -- the same convention as PILCO, so the
        data the model sees has the same statistics.
        """
        x = np.asarray(x0, dtype=np.float64).ravel()
        if with_noise:
            x = x + rng.standard_normal(4) * self.process_noise
        states, costs = [x.copy()], []
        for _ in range(steps):
            with torch.no_grad():
                u = float(policy(torch.as_tensor(x, dtype=DTYPE)[None])[0, 0])
            costs.append(float(self.cost(torch.as_tensor(x, dtype=DTYPE))))
            x = self.step_true(x, u)
            if with_noise:
                x = x + rng.standard_normal(4) * self.process_noise
            states.append(x.copy())
        return np.asarray(states), np.asarray(costs)

    # ------------------------------------------------------------------
    def transitions(self, states: np.ndarray, controls: np.ndarray):
        """``(x_t, u_t, x_{t+1} - x_t)`` triples for supervised learning."""
        s = np.asarray(states[:-1])
        u = np.asarray(controls, dtype=np.float64).reshape(s.shape[0], -1)
        return (torch.as_tensor(s, dtype=DTYPE),
                torch.as_tensor(u, dtype=DTYPE),
                torch.as_tensor(states[1:] - states[:-1], dtype=DTYPE))


def make_cartpole(**kwargs) -> CartPole:
    return CartPole(**kwargs)
