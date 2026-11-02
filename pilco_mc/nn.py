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

__all__ = ["ResidualMLP", "GaussianHead", "DynamicsEnsemble",
           "BoundedPolicy", "RbfPolicy", "initialise_output_scale"]

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
        #: standardised training inputs, kept so the model can tell how far a
        #: query is from anything it has seen
        self.register_buffer("train_z", torch.zeros(0, state_dim + control_dim,
                                                    dtype=DTYPE))
        #: saturating distance-based epistemic variance (see ``epistemic``)
        self.register_buffer("epi_prior", torch.zeros(state_dim, dtype=DTYPE))
        self.register_buffer("epi_length", torch.ones(1, dtype=DTYPE))
        self.epistemic_scale = 0.0
        #: temperature on the predictive variance.  The heteroscedastic head fits
        #: the *average* squared error over the dataset, which is dominated by
        #: the model's own misfit, so at any particular state it is far too wide
        #: (measured: sd 0.305 where the GP says 0.0145 and the true error is
        #: 0.0066).  Over-wide sampling diffuses the particle cloud until the
        #: Monte Carlo cost saturates at "the pole has fallen" for every policy.
        self.var_temperature = 1.0
        #: per-state variance recalibration ``var <- a * var^b``, fitted on held-out
        #: data by :meth:`calibrate`.  A single global temperature cannot fix a
        #: variance whose *shape* is wrong: too wide on the data (so every policy
        #: looks equally bad and the optimiser stops) and not wide enough off it.
        self.cal_a = 1.0
        self.cal_b = 1.0

    # -- normalisation ------------------------------------------------------
    def set_normalisation(self, states: torch.Tensor, controls: torch.Tensor,
                          deltas: torch.Tensor) -> None:
        def stats(t: torch.Tensor):
            return t.mean(0), t.std(0).clamp(min=1e-6)
        with torch.no_grad():
            self.x_mu, self.x_sd = stats(states)
            self.u_mu, self.u_sd = stats(controls)
            self.y_mu, self.y_sd = stats(deltas)
            self.train_z = self._encode(states, controls).clone()

    def _encode(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([(x - self.x_mu) / self.x_sd,
                          (u - self.u_mu) / self.u_sd], dim=-1)

    def set_epistemic(self, alpha: float = 1.0, quantile: float = 0.5) -> None:
        """Configure the distance-based epistemic variance.

        ``epi(z) = epi_prior * (1 - exp(-d^2 / (2 ell^2)))`` where ``d`` is the
        distance (in standardised inputs) to the nearest training point.  It is 0
        on the data and saturates at ``epi_prior`` far away -- the shape a GP's
        predictive variance has, and the shape the ensemble's lack of agreement
        does *not* have.

        ``epi_prior`` is ``alpha^2 * y_sd^2`` (the GP's ``sf^2`` prior in PILCO's
        parameterisation) and ``ell`` is the ``quantile`` of the distance to the
        nearest neighbour, so "far" is measured in units of the data spacing.
        """
        if self.train_z.numel() == 0:
            return
        with torch.no_grad():
            d2 = ((self.train_z[:, None, :] - self.train_z[None]) ** 2).sum(-1)
            d2.diagonal().fill_(float("inf"))
            nn2 = d2.min(dim=1).values
            self.epi_length = torch.sqrt(torch.quantile(
                nn2, quantile).clamp(min=1e-8)).reshape(1)
            self.epi_prior = (alpha ** 2) * (self.y_sd ** 2)
        self.epistemic_scale = 1.0

    def calibrate(self, states: torch.Tensor, controls: torch.Tensor,
                  deltas: torch.Tensor) -> None:
        """Fit ``var <- a * var^b`` by maximising the held-out Gaussian likelihood.

        The heteroscedastic head is trained to fit the *average* squared error, so
        its variance is the wrong size at any particular state.  A power law is
        the smallest correction that can also change the *shape* of the variance
        across states, which is what matters: the planner needs "narrow where the
        model is good, wide where it is not".
        """
        with torch.no_grad():
            mean, var = self.predict(states, controls)
            err2 = ((deltas - mean) ** 2).mean(-1)          # (n,) over dims
            v = var.mean(-1).clamp(min=1e-12)
            best = (float("inf"), 1.0, 1.0)
            for b in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                vb = v ** b
                a = float((err2 / vb).mean())
                nll = float((torch.log(a * vb) + err2 / (a * vb)).mean())
                if nll < best[0]:
                    best = (nll, a, b)
        self.cal_a, self.cal_b = best[1], best[2]

    def epistemic(self, z: torch.Tensor) -> torch.Tensor:
        """Saturating distance-based epistemic variance (see :meth:`set_epistemic`).

        This is the one thing a GP does that an ensemble of identically-trained
        networks does not: its predictive variance **grows away from the data**
        (measured on the same states: GP 0.0145 -> 0.0885 going from a training
        state to upright).  The ensemble's does not (3.18 -> 2.60, it shrinks),
        because every member saw the same points and they all extrapolate the
        same way.  Without a signal like this the policy optimiser can walk into
        a region the model has never seen and be told it is an improvement.
        """
        if self.epistemic_scale <= 0.0 or self.train_z.numel() == 0:
            return torch.zeros(z.shape[:-1] + (self.state_dim,),
                               dtype=z.dtype, device=z.device)
        d2 = ((z[:, None, :] - self.train_z[None]) ** 2).sum(-1).min(dim=1).values
        gate = 1.0 - torch.exp(-0.5 * d2 / (self.epi_length ** 2))
        return gate[:, None] * self.epi_prior[None, :]

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
        # total variance = E[aleatoric] + Var(epistemic mean) + distance term
        aleatoric = torch.stack(variances).mean(0)
        epistemic = torch.stack(means).var(0) if len(models) > 1 \
            else torch.zeros_like(aleatoric)
        var = self.var_temperature * (aleatoric + epistemic + self.epistemic(z))
        if self.cal_b != 1.0 or self.cal_a != 1.0:
            var = self.cal_a * var.clamp(min=1e-12) ** self.cal_b
        return mean, var

    def step(self, x: torch.Tensor, u: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             member: Optional[torch.Tensor] = None,
             noise: Optional[torch.Tensor] = None
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
            var[sel] = self.var_temperature * (
                (self.y_sd ** 2) * logvk.exp() + self.epistemic(z[sel]))
            if self.cal_b != 1.0 or self.cal_a != 1.0:
                var[sel] = self.cal_a * var[sel].clamp(min=1e-12) ** self.cal_b
        var = var.clamp(min=1e-12)
        if noise is None:
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
def _augment(x: torch.Tensor, angle_indices) -> torch.Tensor:
    """Append ``sin``/``cos`` of the angle coordinates.

    PILCO's controller reads ``[x, dx, dtheta, sin(theta), cos(theta)]``, and the
    trig features matter: without them the controller has to represent a periodic
    function of ``theta`` with a kernel on the real line.  The dynamics network
    does not need them (it can learn them), but the policy does.
    """
    if not angle_indices:
        return x
    # PILCO replaces the angle by its sin/cos (`policy_indices` in the MATLAB
    # code drops `theta`); a kernel on the real line is a poor way to represent a
    # periodic function, and keeping both would just add a redundant feature.
    keep = [i for i in range(x.shape[1]) if i not in set(angle_indices)]
    extras = [f(x[:, a]) for a in angle_indices for f in (torch.sin, torch.cos)]
    return torch.cat([x[:, keep]] + [e[:, None] for e in extras], dim=-1)


class RbfPolicy(nn.Module):
    """PILCO's controller: ``u = u_max * tanh(sum_i w_i k(x, c_i))``.

    A *linear* model in squared-exponential features.  PILCO used exactly this,
    and the reason is worth keeping in mind: the policy search optimises over
    ``n_basis * (n_in + 1) + n_in`` parameters instead of thousands, and the
    cost is far better conditioned in that space.  The MLP is available too
    (:class:`BoundedPolicy`), but this is the one that optimises reliably.
    """

    def __init__(self, state_dim: int, control_dim: int,
                 u_max: Sequence[float], n_basis: int = 12,
                 generator: Optional[torch.Generator] = None,
                 angle_indices: Sequence[int] = (),
                 squash: str = "pilco",
                 centre_scale: float = 0.1):
        super().__init__()
        self.angle_indices = tuple(angle_indices)
        self.squash = squash
        n_in = state_dim + len(self.angle_indices)     # angle -> sin, cos
        # Centres drawn from the *initial belief* (std ~0.1), as PILCO does.
        # With spread-out centres every kernel is ~0.6 for every state and the
        # controller is a constant: the pole settles at a tilted equilibrium, the
        # cost saturates at 0.99966 per step and the gradient is exactly zero.
        # With centres at the origin k(x, c) ~ exp(-|x|^2 / 2) is a genuine
        # function of the state, which is what lets the first trajectory move.
        self.centers = nn.Parameter(centre_scale * torch.randn(
            n_basis, n_in, dtype=DTYPE, generator=generator))
        self.log_lengthscales = nn.Parameter(torch.zeros(n_in, dtype=DTYPE))
        #: The scale of the initial weights decides whether the policy search
        #: starts on the "do nothing" plateau: with u(x0) ~ 0.3 N the cost is
        #: flat at 1.0 per step and the gradient vanishes (measured: LBFGS does
        #: not move at all).  ~5 N, the scale PILCO's random controller happens
        #: to produce, is enough to see the swing-up.
        self.weights = nn.Parameter(0.45 * torch.randn(n_basis, control_dim,
                                                       dtype=DTYPE, generator=generator))
        self.register_buffer("u_max", torch.as_tensor(list(u_max), dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _augment(x, self.angle_indices)
        inv_ell2 = torch.exp(-2.0 * self.log_lengthscales)      # 1 / ell^2
        d2 = ((x[:, None, :] - self.centers[None]) ** 2 * inv_ell2).sum(-1)
        raw = torch.exp(-0.5 * d2) @ self.weights
        if self.squash == "pilco":
            # PILCO's saturating output: (9 sin u + sin 3u) / 8, which reaches
            # exactly 1 at u = pi/2.  With this the policy can represent PILCO's
            # controller *exactly*, which is what makes the warm start below an
            # exact transfer rather than a least-squares fit.
            return self.u_max * (9.0 * torch.sin(raw) + torch.sin(3.0 * raw)) / 8.0
        return self.u_max * torch.tanh(raw)


class BoundedPolicy(nn.Module):
    """``u = u_max * tanh(MLP(x))``: a residual MLP with a bounded output.

    The tanh is the analytic equivalent of PILCO's squashed RBF controller: it
    keeps the control inside the actuator limits without a constraint term, and
    it saturates with a zero gradient only in the limit.
    """

    def __init__(self, state_dim: int, control_dim: int, u_max: Sequence[float],
                 width: int = 64, blocks: int = 2,
                 angle_indices: Sequence[int] = ()):
        super().__init__()
        self.angle_indices = tuple(angle_indices)
        self.net = ResidualMLP(state_dim + len(self.angle_indices),
                               control_dim, width, blocks)
        self.register_buffer("u_max", torch.as_tensor(list(u_max), dtype=DTYPE))
        # Start with a non-trivial **constant** control.  From the hanging state
        # every gentle policy costs 1.0 per step, so at u(x0) ~ 0.3 N the cost is
        # flat and the gradient is exactly zero: the optimiser does not move at
        # all (measured: 39.99 +- 0.00 for four episodes, with either dynamics
        # model).  A bias giving ~0.3 u_max breaks the symmetry; the state
        # dependence is learned from there.
        with torch.no_grad():
            self.net.out.bias.fill_(float(torch.atanh(torch.tensor(0.3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_max * torch.tanh(self.net(_augment(x, self.angle_indices)))


@torch.no_grad()
def initialise_output_scale(policy: nn.Module, x0: torch.Tensor,
                            target: float = 0.5) -> float:
    """Scale ``policy`` so that ``|u(x0)| = target * u_max``.

    The single most important hyper-parameter in this project.  From the hanging
    state the cost is a stationary point of the state, so a policy whose control
    is small keeps the pole hanging and every particle pays exactly 0.99966 at
    every step -- the sampled cost is constant, the gradient is *exactly* zero,
    and no optimiser can move (measured: four episodes of ``39.99 +- 0.00``).

    PILCO escapes this by accident: its random RBF controller happens to output
    about 5 N.  Setting the scale explicitly removes the luck.

    Returns the scale that was applied (1.0 if the policy already had the right
    magnitude, and the gain is capped to keep the controller sane).
    """
    u0 = float(policy(x0)[0, 0])
    u_max = float(policy.u_max[0])
    if abs(u0) < 1e-9:
        # no output at all: use the bias instead of a scale
        if hasattr(policy, "net"):
            policy.net.out.bias.fill_(float(torch.atanh(torch.tensor(target))))
        else:
            policy.weights.mul_(1.0)
        return 1.0
    gain = (target * u_max) / abs(u0)
    gain = min(max(gain, 0.1), 20.0)
    if hasattr(policy, "net"):
        policy.net.out.weight.mul_(gain)
        policy.net.out.bias.mul_(gain)
    else:
        policy.weights.mul_(gain)
    return gain
