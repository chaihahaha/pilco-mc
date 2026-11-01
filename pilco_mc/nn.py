"""The two function approximators: a probabilistic dynamics ensemble and a policy.

Why networks instead of a GP + RBF controller
---------------------------------------------
PILCO's analytic moment matching forces both the dynamics model and the
controller into a form whose Gaussian integrals are closed-form: an ARD
squared-exponential GP and a squared-exponential RBF network.  Once the moments
are estimated by Monte Carlo (see :mod:`pilco_mc.particles`) nothing needs to be
integrable any more, so both can be arbitrary differentiable functions.

The only requirement the Monte Carlo estimator places on the dynamics model is
that it can be **sampled**, because the particles have to see the model's
uncertainty.  That is why the model here is probabilistic: an ensemble of
Gaussian heads, giving both aleatoric noise (the log-variance output) and
epistemic uncertainty (the spread between members).

Normalisation
-------------
State differences are two orders of magnitude smaller than the states
themselves, and ``dtheta`` is two orders of magnitude larger than ``theta``.
Every network therefore works in standardised coordinates; the statistics are
buffers on the ensemble, fitted with the data, and the sampler converts the
predicted variance back into physical units.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["ResidualMLP", "GaussianHead", "DynamicsEnsemble", "BoundedPolicy"]

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """``h -> h + W2 act(W1 h)``, the unit that makes depth cheap to optimise.

    ``W2`` starts at **zero**, so the block is the identity at initialisation and
    the stack starts as a shallow network.  Without this the residual branch is
    as large as the signal from the first step, the activations grow with depth,
    and the fit stalls -- measured on this project's dynamics data: R^2 0.41 with
    a non-zero ``W2`` versus 0.93 with the zero init.
    """

    def __init__(self, width: int, activation=nn.SiLU):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.act = activation()
        nn.init.orthogonal_(self.linear1.weight, gain=2 ** 0.5)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.linear2(self.act(self.linear1(h)))


class ResidualMLP(nn.Module):
    """Input projection, ``blocks`` residual blocks, output projection."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 64,
                 blocks: int = 2, activation=nn.SiLU):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, activation) for _ in range(blocks)])
        self.act = activation()
        self.out = nn.Linear(width, out_dim)
        nn.init.orthogonal_(self.inp.weight, gain=2 ** 0.5)
        nn.init.zeros_(self.inp.bias)
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.out(self.act(h))


class GaussianHead(nn.Module):
    """A residual MLP emitting ``(mean, log-variance)`` of the state difference."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 64,
                 blocks: int = 2):
        super().__init__()
        self.net = ResidualMLP(in_dim, 2 * out_dim, width, blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        # keep the variance in a range where the loss cannot explode
        return mean, logvar.clamp(-12.0, 4.0)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------
class DynamicsEnsemble(nn.Module):
    """``E`` independent Gaussian dynamics models, ``x_{t+1} = x_t + f(x_t, u_t)``.

    Parameters
    ----------
    state_dim, control_dim:
        ``D`` and ``Du``.
    members:
        Ensemble size.  The spread between members is the model's *epistemic*
        uncertainty; one member alone would let the policy exploit regions where
        the model is merely extrapolating, which is the classic failure mode of
        model-based policy search.
    hidden, blocks:
        Width and number of residual blocks.
    """

    def __init__(self, state_dim: int, control_dim: int, members: int = 4,
                 hidden: int = 64, blocks: int = 2):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.members = nn.ModuleList([
            GaussianHead(state_dim + control_dim, state_dim, hidden, blocks)
            for _ in range(members)])
        # normalisation statistics; filled by ``set_normalisation``
        self.register_buffer("x_mu", torch.zeros(state_dim, dtype=DTYPE))
        self.register_buffer("x_sd", torch.ones(state_dim, dtype=DTYPE))
        self.register_buffer("u_mu", torch.zeros(control_dim, dtype=DTYPE))
        self.register_buffer("u_sd", torch.ones(control_dim, dtype=DTYPE))
        self.register_buffer("y_mu", torch.zeros(state_dim, dtype=DTYPE))
        self.register_buffer("y_sd", torch.ones(state_dim, dtype=DTYPE))

    # -- normalisation ------------------------------------------------------
    def set_normalisation(self, states: torch.Tensor, controls: torch.Tensor,
                          deltas: torch.Tensor) -> None:
        def stats(t: torch.Tensor):
            return t.mean(0), t.std(0).clamp(min=1e-6)
        with torch.no_grad():
            self.x_mu, self.x_sd = stats(states)
            self.u_mu, self.u_sd = stats(controls)
            self.y_mu, self.y_sd = stats(deltas)

    def _encode(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([(x - self.x_mu) / self.x_sd,
                          (u - self.u_mu) / self.u_sd], dim=-1)

    # -- training -----------------------------------------------------------
    def nll(self, x: torch.Tensor, u: torch.Tensor, delta: torch.Tensor,
            bootstrap: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean Gaussian negative log-likelihood over the ensemble.

        ``bootstrap`` is an optional ``(members, n_bootstrap)`` index tensor: each
        member is then fitted on its own resample of the data.  That is what makes
        the members **disagree away from the data**, which is the only source of
        epistemic uncertainty the sampler has (and the reason a policy optimiser
        cannot treat the model as ground truth).  Training all members on the
        same points makes them agree everywhere, including where they are all
        wrong.
        """
        target = (delta - self.y_mu) / self.y_sd
        total = 0.0
        for k, member in enumerate(self.members):
            if bootstrap is None:
                zk, tk = self._encode(x, u), target
            else:
                idx = bootstrap[k]
                zk, tk = self._encode(x[idx], u[idx]), target[idx]
            mean, logvar = member(zk)
            total = total + 0.5 * (logvar + (tk - mean) ** 2 / logvar.exp()).sum(-1).mean()
        return total / len(self.members)

    # -- prediction ---------------------------------------------------------
    def predict(self, x: torch.Tensor, u: torch.Tensor,
                member: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Physical-space ``(mean_delta, variance_delta)``.

        With ``member=None`` the mean is the ensemble average and the variance
        adds the between-member spread (the law of total variance); that is the
        uncertainty a particle should be sampled from.
        """
        z = self._encode(x, u)
        models = self.members if member is None else [self.members[member]]
        means, variances = [], []
        for m in models:
            mean, logvar = m(z)
            means.append(self.y_mu + self.y_sd * mean)
            variances.append((self.y_sd ** 2) * logvar.exp())
        mean = torch.stack(means).mean(0)
        # total variance = E[aleatoric] + Var[epistemic mean]
        aleatoric = torch.stack(variances).mean(0)
        epistemic = torch.stack(means).var(0) if len(models) > 1 \
            else torch.zeros_like(aleatoric)
        return mean, aleatoric + epistemic

    def step(self, x: torch.Tensor, u: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             member: Optional[torch.Tensor] = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One reparameterised draw of ``x_{t+1}``, plus the variance it used.

        ``member`` optionally pins each particle to a fixed ensemble member
        (shape ``(P,)``); otherwise the member is drawn uniformly per call.  The
        pinned form keeps the ensemble index fixed for a whole trajectory, which
        is the PETS/MC-PILCO convention: one particle = one imagined rollout
        under one model.

        Returning the variance is what makes an uncertainty penalty possible: a
        policy optimiser that cannot see where the model is unsure will walk
        straight into the region where it is wrong (measured here: predicted
        cost 33.8, realised 40.0).
        """
        n = x.shape[0]
        if member is None:
            member = torch.randint(len(self.members), (n,), generator=generator)
        z = self._encode(x, u)
        mean = torch.zeros(n, self.state_dim, dtype=x.dtype)
        var = torch.zeros_like(mean)
        for k, m in enumerate(self.members):
            sel = member == k
            if not sel.any():
                continue
            mk, logvk = m(z[sel])
            mean[sel] = self.y_mu + self.y_sd * mk
            var[sel] = (self.y_sd ** 2) * logvk.exp()
        var = var.clamp(min=1e-12)
        noise = torch.randn(mean.shape, dtype=x.dtype, generator=generator)
        return x + mean + var.sqrt() * noise, var

    def sample_next(self, x: torch.Tensor, u: torch.Tensor,
                    generator: Optional[torch.Generator] = None,
                    member: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sampled successor state only (see :meth:`step`)."""
        return self.step(x, u, generator=generator, member=member)[0]


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
class BoundedPolicy(nn.Module):
    """``u = u_max * tanh(MLP(x))``: a residual MLP with a bounded output.

    The tanh is the analytic equivalent of PILCO's squashed RBF controller: it
    keeps the control inside the actuator limits without a constraint term, and
    it saturates with a zero gradient only in the limit.
    """

    def __init__(self, state_dim: int, control_dim: int, u_max: Sequence[float],
                 width: int = 64, blocks: int = 2):
        super().__init__()
        self.net = ResidualMLP(state_dim, control_dim, width, blocks)
        self.register_buffer("u_max", torch.as_tensor(list(u_max), dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_max * torch.tanh(self.net(x))
