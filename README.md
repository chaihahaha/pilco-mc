# `pilco-mc`

PILCO with the analytic Gaussian moment matching replaced by **Monte Carlo
particle propagation**, and the GP + RBF pair replaced by **neural networks**
(a residual-MLP ensemble for the dynamics, a bounded residual MLP for the
controller).  Borrows the particle-based algorithm from
[MC-PILCO](https://github.com/merlresearch/MC-PILCO) (Amadio et al., 2023).

The point of the exchange: PILCO's closed-form moments are only closed-form
because every stage is an ARD squared-exponential kernel.  Once the moments are
estimated from samples, nothing has to be integrable, so the model and the
controller can be arbitrary differentiable functions.

**Status: the estimator is validated, the loop runs, the cart-pole swing-up is
not yet learned — and the reason is measured, not guessed.**  See
[What happens on cart-pole](#what-happens-on-cart-pole).

---

## What replaces what

| `pilco_pytorch` | here |
| --- | --- |
| `trajectory.plan()` — carries `(m_t, S_t)` analytically | `particles.rollout()` — carries `P` particles |
| `ops/base.splice()` — attaches a new block to the joint Gaussian | nothing: particles carry the joint implicitly |
| `ops/fourier.TrigAugmentOp` — `E[sin]`, `E[cos]` in closed form | nothing: the network sees the raw angle |
| `ops/cost.SaturatingCostOp` — `E[c]`, `Var[c]` in closed form | `objective.expected_cost()` — sample mean |
| `ops/gp.GpDynamicsOp` — ARD GP + uncertain-input moments | `nn.DynamicsEnsemble` — bootstrapped residual MLPs |
| `ops/policy.RbfPolicyOp` — RBF network with analytic moments | `nn.BoundedPolicy` — residual MLP + `tanh` |
| `optim/minimize.py` — PILCO's own weak line-search optimiser | Adam on the reparameterised samples |
| `trajectory.mean_jacobian()` | nothing: autograd through the sampler |

Two things do **not** disappear, and they are the interesting ones: the
objective is still "expected discounted cost under the initial belief", and the
policy is still optimised inside the model.  Only the estimator changed.

## Install and run

```bash
pip install -r requirements.txt

# validate the estimator against the analytic moments (needs ../pilco_pytorch)
python tests/test_particles.py

# learn cart-pole
python scripts/run_cartpole.py --episodes 8 --seed 0

# why the swing-up does not work yet -- the key measurement
python scripts/diagnose_imagination.py
```

## Validation: Monte Carlo reproduces the moments it replaces

`tests/test_particles.py` samples the same distribution, pushes the samples
through the same nonlinearity, and compares the empirical mean and covariance
with the closed form that `pilco_pytorch` computes.  Four operators, 4σ
tolerance:

```
[PASS] initial_particles            sampling N(m, S) reproduces (m, S)
[PASS] E[sin(x1)] / E[cos(x1)]      vs pilco.ops.fourier.TrigAugmentOp
[PASS] Cov[sin,cos]                 the cross term that makes augmentation work
[PASS] E[c]                         vs pilco.ops.cost.SaturatingCostOp
[PASS] E[f0] / E[f1] / Cov[f0,f1]   vs pilco.ops.gp.GpDynamicsOp
[PASS] rollout shapes
[PASS] policy gradient flows
```

So replacing `plan()` with `rollout()` is a change of *estimator*, not of model.

## What happens on cart-pole

The loop runs and the model is good.  It does not learn:

| task | realised cost per episode (H=40 / H=30) |
| --- | --- |
| swing-up, H=40 | 39.99, 39.99, 39.99, 39.93 |
| balance from upright, H=30 | 25.8, 26.4, 26.2, 24.2 |

Both sit at the "the pole has fallen and stays down" ceiling (1.0 per step).
`scripts/diagnose_imagination.py` explains why, and it is not an optimisation
bug:

```
one-step fit: R^2 = 0.977 on 400 transitions

  H   imagined     true   ratio
  5       4.95     4.99     1.0x
 10       9.84     9.98     1.0x
 15      14.71    12.79     1.2x
 20      19.37    12.79     1.5x
 30      28.91    12.80     2.3x
 40      38.72    12.80     3.0x
```

The model is fitted on data from a *known-good* swing-up controller (true cost
12.8) and still imagines that controller's cost as 38.7.  One-step accuracy is
excellent; the imagined trajectory drifts off the data after about ten steps,
and once it has drifted the pole is simply down, which costs 1 per step.

**The swing-up needs ~15 steps of trustworthy imagination before the first
payoff appears. A network trained on one-step transitions gives about ten.**

That is the difference from the analytic version: `pilco_pytorch` propagates the
GP's *calibrated* predictive uncertainty through every step, so an imagined
trajectory that leaves the data becomes uncertain rather than confidently wrong,
and the planner is charged for that uncertainty.  Here the sampler does have an
ensemble, but the members are all fitted to the same 400 points and agree almost
everywhere.

Observed failure modes along the way (numbers in [`docs/NOTES.md`](docs/NOTES.md)):

* **without an uncertainty penalty** the optimiser exploits the model:
  predicted cost 33.8, realised 40.0;
* **with a strong penalty** (λ = 2) it freezes at 39.9 — safe and useless;
* adding data (500 → 620 points), a larger optimisation budget (200 → 800
  steps), and ensemble bootstrapping all move the numbers by less than 1.

## What would fix it

In order of expected effect, none of them implemented yet:

1. **Multi-step training.**  Unroll the model `k` steps during the *fit* and
   penalise the `k`-step error, instead of fitting one-step transitions.  This is
   the direct fix for the table above: it makes the model pay for the drift.
2. **A calibrated model.**  GP per ensemble member, or a deep ensemble with
   temperature/last-layer variance — the current spread is not calibrated, so the
   uncertainty penalty is the wrong size everywhere.
3. **Plan with a horizon the model can support** (≈10 steps) and re-plan, i.e.
   MPC.  This trades the swing-up for a shorter promise.

## Layout

```
pilco_mc/
  nn.py          residual MLPs, Gaussian heads, bootstrapped ensemble, bounded policy
  particles.py   initial_particles, rollout  (replaces plan/splice)
  objective.py   expected cost from particles, particle dropout, uncertainty penalty
  trainer.py     the loop: fit model -> optimise policy inside it -> roll out on truth
  scenario.py    cart-pole: true dynamics, cost, data plumbing
scripts/
  run_cartpole.py          learn and log the realised cost per episode
  diagnose_imagination.py  imagined vs true cost as a function of horizon
tests/
  test_particles.py        Monte Carlo vs the analytic moments
docs/
  NOTES.md                 the engineering log and the measurements behind the claims
```

## Borrowed from MC-PILCO

* particle propagation with reparameterised sampling, so the policy gradient
  flows through the horizon;
* ensemble of dynamics models, one member pinned per particle and per trajectory;
* particle dropout as a variance-reduction / anti-overfitting device (MC-PILCO
  applies it to the controller's RBF activations; here it masks particles);
* keeping the *best* policy seen rather than the last iterate.
