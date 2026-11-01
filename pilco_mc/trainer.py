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

from .nn import BoundedPolicy, DynamicsEnsemble
from .objective import cost_statistics, expected_cost
from .particles import initial_particles, rollout

__all__ = ["MCConfig", "EpisodeLog", "fit_model", "optimise_policy", "learn"]


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

    # -- dynamics ensemble ------------------------------------------------
    members: int = 4
    hidden: int = 64
    blocks: int = 2
    model_epochs: int = 600
    model_lr: float = 3e-3
    model_weight_decay: float = 1e-4
    validation_fraction: float = 0.15

    # -- policy -----------------------------------------------------------
    policy_steps: int = 250
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
    model = DynamicsEnsemble(d, du, members=config.members,
                             hidden=config.hidden, blocks=config.blocks)
    model.set_normalisation(states, controls, deltas)

    n = states.shape[0]
    n_val = max(1, int(config.validation_fraction * n)) if n > 20 else 0
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
def optimise_policy(model: DynamicsEnsemble, policy: BoundedPolicy,
                    cost_fn: Callable[[torch.Tensor], torch.Tensor],
                    mean0: torch.Tensor, cov0: torch.Tensor,
                    config: MCConfig, generator: torch.Generator
                    ) -> tuple[float, float]:
    """Minimise the Monte Carlo estimate of the expected cost (in place).

    Returns ``(best_cost, standard_error)``.  The policy returned is the **best
    one seen**, not the last one: the particle estimate is noisy, so the last
    iterate is not necessarily the best, and keeping it protects against a late
    excursion into a region where the model is wrong.
    """
    params = list(policy.parameters())
    optimiser = torch.optim.Adam(params, lr=config.policy_lr)
    best = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    best_se = float("nan")
    for step in range(config.policy_steps):
        for group in optimiser.param_groups:
            group["lr"] = config.policy_lr * (1.0 - 0.9 * step / config.policy_steps)
        particles = initial_particles(mean0, cov0, config.particles, generator)
        members = torch.randint(len(model.members), (config.particles,),
                                generator=generator)
        roll = rollout(policy, model, particles, config.horizon, generator, members)
        loss = expected_cost(roll, cost_fn, config.gamma, config.dropout, generator,
                             config.uncertainty_weight)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimiser.step()
        if float(loss) < best:
            best = float(loss)
            best_state = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    with torch.no_grad():
        particles = initial_particles(mean0, cov0, config.particles, generator)
        members = torch.randint(len(model.members), (config.particles,),
                                generator=generator)
        roll = rollout(policy, model, particles, config.horizon, generator, members)
        best, best_se = cost_statistics(roll, cost_fn, config.gamma)
    return best, best_se


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def learn(scenario, config: Optional[MCConfig] = None,
          log: Optional[Callable[[str], None]] = print):
    """Run the full loop on ``scenario``; returns ``(policy, model, logs)``."""
    config = config or MCConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    rng = np.random.default_rng(config.seed + 1000)
    mean0, cov0 = scenario.initial_belief()

    # -- initial data: uniformly random controls, so the first model sees the
    # whole actuator range rather than whatever the untrained policy happens to
    # output.
    states_all, controls_all, deltas_all = [], [], []
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

    policy = BoundedPolicy(scenario.state_dim, scenario.control_dim,
                           [scenario.u_max])
    logs: list[EpisodeLog] = []
    model = None

    for episode in range(config.episodes):
        states = torch.cat(states_all)
        controls = torch.cat(controls_all)
        deltas = torch.cat(deltas_all)
        model = fit_model(states, controls, deltas, config, seed=config.seed + episode)
        predicted, stderr = optimise_policy(model, policy, scenario.cost,
                                            mean0, cov0, config, generator)

        # -- roll the policy out on the true system and append the data
        x0 = mean0.numpy() + np.linalg.cholesky(cov0.numpy()) @ rng.standard_normal(4)
        traj, costs = scenario.rollout_true(policy, x0, config.horizon, rng)
        u = []
        for t in range(config.horizon):
            with torch.no_grad():
                u.append(float(policy(torch.as_tensor(traj[t:t + 1]))[0, 0]))
        i, j, k = scenario.transitions(traj, np.asarray(u))
        states_all.append(i)
        controls_all.append(j)
        deltas_all.append(k)

        realised = float(np.sum(costs))
        logs.append(EpisodeLog(episode, realised, predicted, stderr,
                               int(states.shape[0])))
        if log is not None and config.verbose:
            log(f"  episode {episode + 1}/{config.episodes}: "
                f"data={states.shape[0]:4d}  "
                f"predicted {predicted:7.2f} +- {stderr:5.2f}  "
                f"realised {realised:7.2f}")
    return policy, model, logs
