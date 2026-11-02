"""The learning loop: fit the ensemble, optimise the policy inside it, repeat.

This is PILCO's loop with the analytic planner swapped for particle propagation
and the GP/RBF pair swapped for networks:

1. roll the current policy out on the **true** system and append the data;
2. re-fit the dynamics ensemble by maximum likelihood;
3. optimise the policy against the **model**, with the Monte Carlo estimate of
   the expected cost as the objective and the model held fixed;
4. repeat.

Step 3 is where the old PILCO needed ``minimize`` with hand-tuned line searches:
its optimiser had to be weak, because a strong optimiser exploits model errors.
Two things replace that here, both from MC-PILCO: an **ensemble** (so the policy
sees epistemic uncertainty rather than a single confident extrapolation) and
**particle dropout** (so it cannot fit the particular noise draw).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .nn import BoundedPolicy, DynamicsEnsemble, RbfPolicy
from .objective import cost_statistics, expected_cost
from .particles import draw_batch, rollout_batch

__all__ = ["MCConfig", "EpisodeLog", "fit_model", "optimise_policy",
           "learn", "collect_seed_data", "load_policy_npz"]


@dataclass
class MCConfig:
    """Everything the loop needs, with the defaults used for cart-pole."""

    horizon: int = 40
    episodes: int = 8
    #: random-control rollouts used to seed the dataset.  PILCO starts from one
    #: rollout of ``horizon`` steps because a GP prior is informative there; a
    #: network is not, so the default is larger -- see docs/NOTES.md.
    initial_rollouts: int = 5
    #: steps per seed rollout (they need not match ``horizon``)
    initial_steps: int = 100
    seed: int = 0

    # -- dynamics model ----------------------------------------------------
    #: ``"nn"`` = bootstrapped residual-MLP ensemble; ``"gp"`` = one exact GP per
    #: state dimension (PILCO's model).  The network's variance is fitted to the
    #: average squared error and comes out 20x too wide; a GP's posterior
    #: variance is narrow on the data and wide off it.
    model_kind: str = "nn"
    members: int = 4
    hidden: int = 64
    blocks: int = 2
    #: the heteroscedastic variance head converges far more slowly than the
    #: mean.  At 600 epochs the predicted sd is ~3.0 where the actual one-step
    #: error is ~0.007, so every particle samples noise 300x the process noise;
    #: at 3000 epochs it reaches ~0.32, which is the right order of magnitude.
    model_epochs: int = 1500
    model_lr: float = 3e-3
    #: weight decay is off: applied to the log-variance head it pulls the
    #: predictive variance back to 1 (in normalised units), i.e. ~300x the true
    #: process noise, which is what made every particle sample garbage.
    model_weight_decay: float = 0.0
    #: early stopping is off by default: the variance head converges long after
    #: the validation loss plateaus, and stopping there leaves the model
    #: sampling noise ~300x the process noise (see docs/NOTES.md)
    validation_fraction: float = 0.0

    # -- policy -----------------------------------------------------------
    #: ``"rbf"`` (PILCO's controller, few parameters) or ``"mlp"``
    policy_kind: str = "mlp"
    #: ``"pilco"`` (PILCO's 9 sin + sin 3 saturation) or ``"tanh"``
    squash: str = "pilco"
    n_basis: int = 12
    #: path to a ``.npz`` controller from ``../pilco_pytorch`` to warm-start from
    warm_start: Optional[str] = None
    #: optimisation rounds; each round re-draws the particle batch and runs an
    #: L-BFGS line search on the (now deterministic) objective
    policy_rounds: int = 8
    lbfgs_max_iter: int = 20
    #: ``"cem"`` = cross-entropy search then an LBFGS polish (default),
    #: ``"lbfgs"`` = LBFGS only.  Gradient search from a random initialisation
    #: reliably lands in a local minimum around 31-35 on cart-pole even with a
    #: perfect model; the population search finds the basin first.
    policy_optimizer: str = "cem"
    cem_iters: int = 30
    cem_population: int = 48
    cem_elite: int = 8
    cem_sigma: float = 0.3
    cem_sigma_floor: float = 0.02
    policy_steps: int = 250          # kept for the Adam fallback
    policy_lr: float = 3e-3
    #: particles per Monte Carlo estimate; the estimator's standard error is
    #: ``O(1/sqrt(particles))``
    particles: int = 128
    #: probability of dropping a particle from the gradient (MC-PILCO's trick)
    dropout: float = 0.1
    #: weight of the uncertainty penalty.  Without it the optimiser drives the
    #: imagined trajectory into the model's most wrong region; see
    #: :func:`pilco_mc.objective.expected_cost`.
    uncertainty_weight: float = 0.2
    gamma: float = 1.0
    #: stop the policy search early once the predicted cost is this low
    stop_cost: float = 0.0
    #: temperature on the predictive variance (see ``DynamicsEnsemble``).  The
    #: heteroscedastic head fits the average squared error, which is dominated by
    #: model misfit, so it is wide enough to diffuse the particles until the
    #: Monte Carlo cost saturates at "the pole has fallen" for every policy.
    #: PILCO's GP reports sd 0.0145 in-data; the raw network reports 0.245.
    variance_temperature: float = 0.02
    #: scale of the distance-based epistemic variance, ``alpha^2 * y_sd^2``
    #: saturating away from the data (0 disables it).  The target is a
    #: ``sd(off-data)/sd(in-data)`` ratio like a GP's (measured 6.1).
    epistemic_alpha: float = 0.0

    verbose: bool = True


@dataclass
class EpisodeLog:
    episode: int
    realised_cost: float
    predicted_cost: float
    predicted_stderr: float
    n_data: int


# ---------------------------------------------------------------------------
# model fitting
# ---------------------------------------------------------------------------
def fit_model(states: torch.Tensor, controls: torch.Tensor, deltas: torch.Tensor,
              config: MCConfig, seed: int = 0) -> DynamicsEnsemble:
    """Maximum-likelihood fit of the ensemble, with a validation-based stop.

    Early stopping matters more than it looks: the ensemble is fitted on a few
    hundred points that are concentrated wherever the current policy has been,
    so an over-trained model extrapolates confidently and the planner walks into
    the extrapolation.
    """
    d, du = states.shape[1], controls.shape[1]
    if config.model_kind == "gp":
        from .gp import GpDynamics
        model = GpDynamics(d, du)
        model.var_temperature = config.variance_temperature
        return model.fit(states, controls, deltas,
                         iterations=max(40, config.model_epochs // 20))
    model = DynamicsEnsemble(d, du, members=config.members,
                             hidden=config.hidden, blocks=config.blocks)
    model.set_normalisation(states, controls, deltas)
    if config.epistemic_alpha > 0.0:
        model.set_epistemic(config.epistemic_alpha)
    model.var_temperature = config.variance_temperature

    n = states.shape[0]
    n_val = int(config.validation_fraction * n) if n > 20 else 0
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val, train = perm[:n_val], perm[n_val:]

    # one bootstrap resample per ensemble member
    gen = torch.Generator().manual_seed(seed)
    # indices into the *training subset* passed to model.nll below
    bootstrap = torch.randint(len(train), (config.members, len(train)),
                              generator=gen)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.model_lr,
                                 weight_decay=config.model_weight_decay)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for _ in range(config.model_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model.nll(states[train], controls[train], deltas[train],
                         bootstrap=bootstrap)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
        if n_val == 0:
            continue
        with torch.no_grad():
            val_loss = float(model.nll(states[val], controls[val], deltas[val]))
        if val_loss < best - 1e-6:
            best, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale > 80:
                break
    if n_val:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# policy optimisation inside the model
# ---------------------------------------------------------------------------
def _flatten(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _assign(params: list[torch.Tensor], vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vector[offset:offset + n].reshape(p.shape))
            offset += n


def cross_entropy_search(policy, model, cost_fn, mean0, cov0,
                         config: MCConfig, generator: torch.Generator
                         ) -> float:
    """Population search over the policy parameters (CEM / CMA-lite).

    Derivative-free, so it does not care that the objective is only a Monte
    Carlo estimate, and it explores several basins before committing.  On this
    task a gradient method from a random start reliably stops at a local minimum
    worth ~31-35 while a good swing-up is worth ~13, so the basin has to be
    found before the gradient can be used.
    """
    params = list(policy.parameters())
    mu = _flatten(params)
    sigma = torch.full_like(mu, config.cem_sigma)
    batch = draw_batch(mean0, cov0, config.particles, config.horizon,
                       len(model.members), generator)
    best, best_state = float("inf"), copy.deepcopy(policy.state_dict())
    for _ in range(config.cem_iters):
        noise = torch.randn(config.cem_population, mu.numel(), generator=generator)
        candidates = mu + sigma * noise
        scores = []
        with torch.no_grad():
            for i in range(config.cem_population):
                _assign(params, candidates[i])
                scores.append(float(expected_cost(
                    rollout_batch(policy, model, batch), cost_fn,
                    config.gamma, 0.0, None, config.uncertainty_weight)))
        scores_t = torch.as_tensor(scores)
        elite = torch.argsort(scores_t)[:config.cem_elite]
        mu = candidates[elite].mean(0)
        sigma = candidates[elite].std(0).clamp(min=config.cem_sigma_floor)
        if float(scores_t[elite[0]]) < best:
            best = float(scores_t[elite[0]])
            _assign(params, candidates[elite[0]])
            best_state = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    return best


def optimise_policy(model: DynamicsEnsemble, policy,
                    cost_fn: Callable[[torch.Tensor], torch.Tensor],
                    mean0: torch.Tensor, cov0: torch.Tensor,
                    config: MCConfig, generator: torch.Generator
                    ) -> tuple[float, float]:
    """Minimise the Monte Carlo estimate of the expected cost (in place).

    Two changes make this strong enough to find a swing-up, and both come from
    the measurement that a perfect model still failed with Adam:

    * the particle batch (initial states, ensemble members, noise) is **fixed
      within a round**, so the objective is a deterministic function of the
      policy parameters and its gradient is exact for that batch;
    * the optimiser is therefore **L-BFGS with a strong-Wolfe line search**, not
      a fixed-step first-order method.

    The batch is re-drawn between rounds so the policy cannot overfit one draw,
    and each round's result is scored on a held-out batch, so the policy
    returned is the best one seen rather than the last one.
    """
    if config.policy_optimizer == "cem":
        cross_entropy_search(policy, model, cost_fn, mean0, cov0, config,
                             generator)

    params = list(policy.parameters())
    optimiser = torch.optim.LBFGS(params, lr=1.0,
                                  max_iter=config.lbfgs_max_iter,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=20,
                                  line_search_fn="strong_wolfe")
    #: fixed batch used to compare rounds -- a fresh draw every round would make
    #: the comparison noise-dominated
    val_batch = draw_batch(mean0, cov0, config.particles, config.horizon,
                           len(model.members), generator)

    best_score = float("inf")
    best_state = copy.deepcopy(policy.state_dict())

    for _ in range(config.policy_rounds):
        batch = draw_batch(mean0, cov0, config.particles, config.horizon,
                           len(model.members), generator)

        def closure():
            optimiser.zero_grad()
            roll = rollout_batch(policy, model, batch)
            # No dropout here: a fresh dropout mask per evaluation would make the
            # objective random again, which is exactly what the fixed batch is
            # there to avoid.  Re-drawing the batch between rounds already does
            # the regularising job dropout was doing.
            loss = expected_cost(roll, cost_fn, config.gamma, 0.0,
                                 None, config.uncertainty_weight)
            loss.backward()
            for p in params:
                if p.grad is not None and not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
            return loss

        optimiser.step(closure)

        with torch.no_grad():
            score = float(expected_cost(rollout_batch(policy, model, val_batch),
                                        cost_fn, config.gamma, 0.0, None,
                                        config.uncertainty_weight))
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(policy.state_dict())
        if best_score < config.stop_cost:
            break

    policy.load_state_dict(best_state)
    with torch.no_grad():
        roll = rollout_batch(policy, model, val_batch)
        value, stderr = cost_statistics(roll, cost_fn, config.gamma)
    return value, stderr


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def collect_seed_data(scenario, config: MCConfig, mean0, cov0, rng,
                      states_all, controls_all, deltas_all) -> None:
    """Uniformly random controls, so the first model sees the whole actuator range.

    The alternative -- let the untrained policy generate the first data -- biases
    the model towards whatever near-zero output a random network happens to
    produce.
    """
    for _ in range(config.initial_rollouts):
        x = mean0.numpy() + np.linalg.cholesky(cov0.numpy()) @ rng.standard_normal(4)
        states, costs = [], []
        for _ in range(config.initial_steps):
            u = float(rng.uniform(-scenario.u_max, scenario.u_max))
            costs.append(float(scenario.cost(torch.as_tensor(x, dtype=torch.float64))))
            nxt = scenario.step_true(x, u)
            nxt = nxt + rng.standard_normal(4) * scenario.process_noise
            states.append((x.copy(), u))
            x = nxt
        states.append((x.copy(), 0.0))
        s = np.asarray([v[0] for v in states])
        u = np.asarray([v[1] for v in states[:-1]])
        i, j, k = scenario.transitions(s, u)
        states_all.append(i)
        controls_all.append(j)
        deltas_all.append(k)


def load_policy_npz(policy, path: str) -> None:
    """Copy an ``RbfPolicy`` from a controller saved by ``pilco_pytorch``.

    The two parameterisations match store-for-store (``centers``,
    ``lengthscales``, ``beta``), so this is an exact transfer when
    ``squash="pilco"``.
    """
    data = np.load(path)
    centers = np.asarray(data["centers"])
    ell = np.asarray(data["lengthscales"])
    if "beta" in data:
        beta = np.asarray(data["beta"])
    else:
        # an older cache stores only the pseudo-targets; the network *weights*
        # are beta = (K + sigma_n^2 I)^{-1} T with sigma_n = 0.01 and unit signal
        diff = ((centers[:, None, :] - centers[None, :, :]) / ell[None, None, :]) ** 2
        gram = np.exp(-0.5 * diff.sum(-1))
        beta = np.linalg.solve(gram + 1e-4 * np.eye(len(centers)),
                               np.asarray(data["targets"]))
    with torch.no_grad():
        policy.centers.copy_(torch.as_tensor(centers))
        policy.log_lengthscales.copy_(torch.log(torch.as_tensor(ell)))
        policy.weights.copy_(torch.as_tensor(beta))


def make_policy(scenario, config: MCConfig, generator: torch.Generator):
    """The controller for this run (PILCO-style RBF, or a residual MLP)."""
    if config.policy_kind == "rbf":
        return RbfPolicy(scenario.state_dim, scenario.control_dim,
                         [scenario.u_max], config.n_basis, generator,
                         angle_indices=scenario.angle_indices,
                         squash=config.squash)
    return BoundedPolicy(scenario.state_dim, scenario.control_dim,
                         [scenario.u_max],
                         angle_indices=scenario.angle_indices)


def collect_policy_rollout(scenario, policy, config: MCConfig, mean0, cov0,
                           rng, states_all, controls_all, deltas_all) -> float:
    """One policy rollout on the true system, appended to the dataset."""
    x0 = mean0.numpy() + np.linalg.cholesky(cov0.numpy()) @ rng.standard_normal(4)
    traj, costs = scenario.rollout_true(policy, x0, config.horizon, rng)
    us = []
    for t in range(config.horizon):
        with torch.no_grad():
            us.append(float(policy(torch.as_tensor(traj[t:t + 1]))[0, 0]))
    i, j, k = scenario.transitions(traj, np.asarray(us))
    states_all.append(i)
    controls_all.append(j)
    deltas_all.append(k)
    return float(np.sum(costs))


def learn(scenario, config: Optional[MCConfig] = None,
          log: Optional[Callable[[str], None]] = print):
    """Run the full loop on ``scenario``; returns ``(policy, model, logs)``."""
    config = config or MCConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    rng = np.random.default_rng(config.seed + 1000)
    mean0, cov0 = scenario.initial_belief()

    states_all, controls_all, deltas_all = [], [], []
    collect_seed_data(scenario, config, mean0, cov0, rng,
                      states_all, controls_all, deltas_all)

    policy = make_policy(scenario, config, generator)
    if config.warm_start:
        load_policy_npz(policy, config.warm_start)
        if log is not None and config.verbose:
            log(f"  warm start: policy loaded from {config.warm_start}")
        # The model has only seen random-control data at this point, i.e. it has
        # never seen the pole near upright.  Without this rollout the model rates
        # a 13-cost controller at 39.9 and the first policy optimisation throws
        # it away (measured).
        collect_policy_rollout(scenario, policy, config, mean0, cov0, rng,
                               states_all, controls_all, deltas_all)

    logs: list[EpisodeLog] = []
    model = None
    for episode in range(config.episodes):
        states = torch.cat(states_all)
        controls = torch.cat(controls_all)
        deltas = torch.cat(deltas_all)
        model = fit_model(states, controls, deltas, config,
                          seed=config.seed + episode)
        predicted, stderr = optimise_policy(model, policy, scenario.cost,
                                            mean0, cov0, config, generator)
        realised = collect_policy_rollout(scenario, policy, config, mean0, cov0,
                                          rng, states_all, controls_all,
                                          deltas_all)
        logs.append(EpisodeLog(episode, realised, predicted, stderr,
                               int(states.shape[0])))
        if log is not None and config.verbose:
            log(f"  episode {episode + 1}/{config.episodes}: "
                f"data={states.shape[0]:4d}  "
                f"predicted {predicted:7.2f} +- {stderr:5.2f}  "
                f"realised {realised:7.2f}")
    return policy, model, logs
