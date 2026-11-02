"""Monte Carlo PILCO: particle moment propagation with neural networks.

Replaces PILCO's closed-form Gaussian moment matching with sampled particles, so
the dynamics model and the controller no longer have to be analytically
integrable -- they are ordinary residual MLPs (an ensemble for the model).
"""

from .nn import BoundedPolicy, DynamicsEnsemble, RbfPolicy, ResidualMLP
from .objective import cost_statistics, expected_cost
from .particles import Rollout, initial_particles, moment_match, rollout
from .scenario import CartPole, make_cartpole
from .trainer import (EpisodeLog, MCConfig, fit_model, learn,
                      load_policy_npz, optimise_policy)

__all__ = [
    "BoundedPolicy", "DynamicsEnsemble", "RbfPolicy", "ResidualMLP",
    "cost_statistics", "expected_cost",
    "Rollout", "initial_particles", "moment_match", "rollout",
    "CartPole", "make_cartpole",
    "EpisodeLog", "MCConfig", "fit_model", "learn", "optimise_policy",
    "load_policy_npz",
]
