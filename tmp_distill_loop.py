import sys, os, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
torch.set_default_dtype(torch.float64)
from pilco_mc import MCConfig, fit_model, make_cartpole, BoundedPolicy
from pilco_mc.trainer import collect_seed_data, optimise_policy
from pilco_mc.particles import draw_batch, rollout_batch
from pilco_mc.objective import cost_statistics
import importlib.util
spec=importlib.util.spec_from_file_location("d","scripts/distill_policy.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

scenario=make_cartpole(); control=d.load_controller()
states, actions = d.collect_teacher_data(scenario, control)
gen=torch.Generator().manual_seed(0)
student=BoundedPolicy(4,1,[scenario.u_max],angle_indices=(3,))
print("distil MSE %.4f" % d.fit_student(student, states, actions, seed=0))
print("student on plant: %.2f" % d.true_cost(scenario, student)[0])

cfg=MCConfig(horizon=40, model_kind="gp", verbose=False,
             particles=32, policy_rounds=4, lbfgs_max_iter=15,
             policy_optimizer="lbfgs")
mean0,cov0=scenario.initial_belief(); rng=np.random.default_rng(1000)
S,U,D=[],[],[]
collect_seed_data(scenario, cfg, mean0, cov0, rng, S,U,D)
# one rollout of the distilled policy so the model has seen the swing-up
x0=np.zeros(4); tr=[]; us=[]
x=x0.copy()
for t in range(40):
    with torch.no_grad(): u=float(student(torch.as_tensor(x)[None])[0,0])
    tr.append(x.copy()); us.append(u); x=scenario.step_true(x,u); x=x+rng.standard_normal(4)*scenario.process_noise
tr.append(x.copy())
i,j,k=scenario.transitions(np.asarray(tr),np.asarray(us)); S.append(i);U.append(j);D.append(k)
model=fit_model(torch.cat(S),torch.cat(U),torch.cat(D),cfg,seed=0)

b=draw_batch(mean0,cov0,64,40,1,torch.Generator().manual_seed(0))
pred=cost_statistics(rollout_batch(student,model,b),scenario.cost)[0]
real=d.true_cost(scenario,student)[0]
print("model's view: %.2f   plant: %.2f" % (pred, real))

g=torch.Generator().manual_seed(0)
p2,se=optimise_policy(model,student,scenario.cost,mean0,cov0,cfg,g)
pred2=cost_statistics(rollout_batch(student,model,b),scenario.cost)[0]
real2=d.true_cost(scenario,student)[0]
print("after policy optimisation: model %.2f -> plant %.2f" % (pred2, real2))
