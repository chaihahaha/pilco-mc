"""A GP dynamics model, as the alternative to the neural ensemble.

The measured reason for adding this: the heteroscedastic network's predictive
variance is fitted to the *average* squared error over the whole dataset, so at
any particular state it is 20x too wide (sd 0.245 against a true one-step error
of 0.0066) and does not grow the way a GP's does when the query leaves the data.
Wide sampling diffuses the particles until the Monte Carlo cost saturates at "the
pole has fallen"; narrowing it makes the model overconfident and the optimiser
exploits it.  A GP's posterior variance is narrow exactly where the data is and
wide where it is not, which is the shape the planner needs.

Implementation: one exact GP per state dimension, ARD squared-exponential kernel,
hyper-parameters fitted by maximising the marginal likelihood with L-BFGS --
i.e. PILCO's model, in GPyTorch, behind the same interface as
:class:`~pilco_mc.nn.DynamicsEnsemble` so the particle machinery does not change.
"""
from __future__ import annotations

from typing import Optional, Sequence

import gpytorch
import torch
import torch.nn as nn

__all__ = ["GpDynamics"]

DTYPE = torch.float64


class _SingleGP(gpytorch.models.ExactGP):
    """Zero mean, ARD squared-exponential kernel, Gaussian noise."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        super().__init__(inputs, targets,
                         gpytorch.likelihoods.GaussianLikelihood(
                             # PILCO's curb in spirit: without a floor the
                             # marginal likelihood on 40 points drives the noise
                             # to ~0 and the Cholesky returns NaN (measured).
                             noise_constraint=gpytorch.constraints.GreaterThan(1e-4)))
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=inputs.shape[1],
                lengthscale_constraint=gpytorch.constraints.Interval(1e-2, 1e2)))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class GpDynamics(nn.Module):
    """``E`` independent exact GPs, one per state dimension.

    Stands in for :class:`~pilco_mc.nn.DynamicsEnsemble`; ``members`` is a
    one-element list so that callers which draw an ensemble index keep working.
    """

    def __init__(self, state_dim: int, control_dim: int, **_: object):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.members = [None]
        self.models: list[_SingleGP] = []
        # normalisation, same convention as the network
        self.register_buffer("x_mu", torch.zeros(state_dim, dtype=DTYPE))
        self.register_buffer("x_sd", torch.ones(state_dim, dtype=DTYPE))
        self.register_buffer("u_mu", torch.zeros(control_dim, dtype=DTYPE))
        self.register_buffer("u_sd", torch.ones(control_dim, dtype=DTYPE))
        self.register_buffer("y_mu", torch.zeros(state_dim, dtype=DTYPE))
        self.register_buffer("y_sd", torch.ones(state_dim, dtype=DTYPE))
        self.epistemic_scale = 0.0
        self.var_temperature = 1.0

    # -- interface parity with DynamicsEnsemble ----------------------------
    def set_normalisation(self, states, controls, deltas) -> None:
        with torch.no_grad():
            self.x_mu, self.x_sd = states.mean(0), states.std(0).clamp(min=1e-6)
            self.u_mu, self.u_sd = controls.mean(0), controls.std(0).clamp(min=1e-6)
            self.y_mu, self.y_sd = deltas.mean(0), deltas.std(0).clamp(min=1e-6)

    def _encode(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([(x - self.x_mu) / self.x_sd,
                          (u - self.u_mu) / self.u_sd], dim=-1)

    def set_epistemic(self, *args, **kwargs) -> None:
        """No-op: the GP posterior already carries the right epistemic shape."""

    def fit(self, states: torch.Tensor, controls: torch.Tensor,
            deltas: torch.Tensor, iterations: int = 80,
            init_from: Optional["GpDynamics"] = None) -> "GpDynamics":
        """Maximise the marginal likelihood of each output (L-BFGS)."""
        self.set_normalisation(states, controls, deltas)
        z = self._encode(states, controls).detach()
        y = ((deltas - self.y_mu) / self.y_sd).detach()
        self.models = []
        for e in range(self.state_dim):
            model = _SingleGP(z, y[:, e])
            if init_from is not None and e < len(init_from.models):
                # warm start: last episode's hyper-parameters, then one short fit
                # on the new data.  From scratch the marginal-likelihood optimum
                # can land in a different basin every episode, which is what made
                # the closed loop oscillate between 13 and 25.
                with torch.no_grad():
                    model.load_state_dict(init_from.models[e].state_dict())
            model.train()
            model.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            optimiser = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                          max_iter=iterations,
                                          line_search_fn="strong_wolfe")
            def closure(model=model, mll=mll, e=e):
                optimiser.zero_grad()
                loss = -mll(model(z), y[:, e]) * len(z)
                loss.backward()
                return loss
            optimiser.step(closure)
            model.eval()
            model.likelihood.eval()
            self.models.append(model)
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, x: torch.Tensor, u: torch.Tensor,
                member: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._encode(x, u)
        means, vars_ = [], []
        for e, model in enumerate(self.models):
            dist = model(z)
            # Differentiable in ``z``: the policy gradient has to flow through
            # the dynamics, exactly as it does through the network.
            means.append(self.y_mu[e] + self.y_sd[e] * dist.mean)
            vars_.append((self.y_sd[e] ** 2) * dist.variance)
        return (torch.stack(means, dim=-1),
                self.var_temperature * torch.stack(vars_, dim=-1))

    def step(self, x: torch.Tensor, u: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             member: Optional[torch.Tensor] = None,
             noise: Optional[torch.Tensor] = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.predict(x, u)
        if noise is None:
            noise = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        return x + mean + var.clamp(min=1e-12).sqrt() * noise, var

    def sample_next(self, x, u, generator=None, member=None):
        return self.step(x, u, generator, member)[0]

    def nll(self, states, controls, deltas) -> torch.Tensor:
        """Marginal likelihood objective (only used for reporting)."""
        z = self._encode(states, controls)
        y = (deltas - self.y_mu) / self.y_sd
        total = 0.0
        for e, model in enumerate(self.models):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            total = total - len(z) * mll(model(z), y[:, e])
        return total
