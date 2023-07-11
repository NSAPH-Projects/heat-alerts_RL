import numpy as np
import matplotlib.pyplot as plt
import json
from cmdstanpy import CmdStanModel

# read sim data
with open("bayesian_model/simulated_data/sim.json", "r") as f:
    sim_data = json.load(f)

# load stan model
model = CmdStanModel(stan_file="bayesian_model/model_not_spatial.stan")

# make fitting data
keys = ["N", "S", "DX", "DW", "M", "node1", "node2",  "A", "offset", "X", "W", "sind", "mu", "Y"]
data = {key: sim_data[key] for key in keys}


# fit model, use variational inference
fit = model.variational(
    data=data,
    seed=123,
    algorithm="meanfield",
    iter=5000,
    tol_rel_obj=0.001,
    eval_elbo=50,
    require_converged=False,
    show_console=True,
    output_dir="bayesian_model/results",
    output_samples=1,
)

# fit = model.sample(
#     data=data,
#     seed=123,
#     iter_sampling=1000,
#     chains=1,
#     show_console=True,
#     output_dir="bayesian_model/results",
#     thin=10,
# )

# save fit
var = "tau"
x_ =  getattr(fit, var).reshape(-1)
x = np.array(sim_data[var]).reshape(-1)
plt.scatter(x, x_, alpha=0.02, c="blue", label="spatial")
plt.xlabel("truth")
plt.ylabel("estimated")
plt.savefig(f"bayesian_model/fit_simulation_{var}_cmdstan.png")
plt.close()
