"""Learn cart-pole with Monte Carlo PILCO and report the realised cost per episode.

Usage::

    python scripts/run_cartpole.py --episodes 8 --seed 0
    python scripts/run_cartpole.py --episodes 8 --smoke      # 1-minute check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

from pilco_mc import MCConfig, learn, make_cartpole            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--policy-steps", type=int, default=250)
    ap.add_argument("--model-epochs", type=int, default=600)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warm-start", default=None,
                    help="path to a pilco_pytorch controller .npz to start from")
    ap.add_argument("--model-kind", default="nn", choices=["gp", "nn"])
    ap.add_argument("--policy-kind", default="mlp", choices=["rbf", "mlp"])
    ap.add_argument("--initial-rollouts", type=int, default=5)
    ap.add_argument("--initial-steps", type=int, default=100)
    ap.add_argument("--gp-iterations", type=int, default=0)
    ap.add_argument("--policy-width", type=int, default=64)
    ap.add_argument("--policy-blocks", type=int, default=2)
    ap.add_argument("--lbfgs-restarts", type=int, default=5)
    ap.add_argument("--particle-dropout", type=float, default=0.0)
    ap.add_argument("--value-kind", default="particles",
                    choices=["particles", "unscented"])
    ap.add_argument("--variance-temperature", type=float, default=1.0)
    ap.add_argument("--policy-rounds", type=int, default=8)
    ap.add_argument("--lbfgs-iter", type=int, default=20)
    ap.add_argument("--policy-optimizer", default="cem",
                    choices=["cem", "lbfgs"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings that finish in about a minute")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.episodes, args.particles = 2, 32
        args.policy_steps, args.model_epochs = 40, 150

    scenario = make_cartpole(horizon=args.horizon)
    config = MCConfig(horizon=args.horizon, episodes=args.episodes,
                      seed=args.seed, particles=args.particles,
                      policy_steps=args.policy_steps,
                      model_epochs=args.model_epochs,
                      members=args.members, dropout=args.dropout,
                      warm_start=args.warm_start,
                      model_kind=args.model_kind,
                      policy_kind=args.policy_kind,
                      initial_rollouts=args.initial_rollouts,
                      initial_steps=args.initial_steps,
                      gp_iterations=args.gp_iterations,
                      policy_width=args.policy_width,
                      policy_blocks=args.policy_blocks,
                      lbfgs_restarts=args.lbfgs_restarts,
                      particle_dropout=args.particle_dropout,
                      value_kind=args.value_kind,
                      variance_temperature=args.variance_temperature,
                      policy_rounds=args.policy_rounds,
                      lbfgs_max_iter=args.lbfgs_iter,
                      policy_optimizer=args.policy_optimizer,
                      verbose=True)
    t0 = time.time()
    policy, model, logs = learn(scenario, config)
    elapsed = time.time() - t0
    costs = [log.realised_cost for log in logs]
    print(f"\nMC-PILCO seed={args.seed}: total={sum(costs):.2f} "
          f"costs={[round(c, 2) for c in costs]}  [{elapsed:.0f}s]")

    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    path = args.out or os.path.join(_ROOT, "results",
                                    f"cartpole_mc_seed{args.seed}.json")
    with open(path, "w") as fh:
        json.dump({"seed": args.seed, "horizon": args.horizon,
                   "costs": costs, "elapsed": elapsed,
                   "config": config.__dict__}, fh, indent=1)
    torch.save({"policy": policy.state_dict(), "model": model.state_dict()},
               os.path.join(_ROOT, "results", f"cartpole_mc_seed{args.seed}.pt"))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
