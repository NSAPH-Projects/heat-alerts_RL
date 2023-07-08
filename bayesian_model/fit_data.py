# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from patsy import dmatrix
from cmdstanpy import CmdStanModel
from sklearn.preprocessing import StandardScaler


# %% read processed data
X = pd.read_parquet("data/processed/states.parquet")
A = pd.read_parquet("data/processed/actions.parquet")
Y = pd.read_parquet("data/processed/outcomes.parquet")
sind = pd.read_parquet("data/processed/location_indicator.parquet")
W = pd.read_parquet("data/processed/spatial_feats.parquet")
P = pd.read_parquet("data/processed/population.parquet")

with open("data/processed/scalers.json") as io:
    scalers = json.load(io)

with open("data/processed/fips2idx.json") as io:
    fips2idx = json.load(io)

# %% pick offset smartly
# %% for the offset we want mean(log(Y/offset)) ~ 0
pop = P.Population.values
loc = sind.sind.values
tmp = np.log((1 + Y.other_hosps.values) / pop[loc])
M = np.exp(np.mean(tmp))

#%% get means by location
tmp = (
    pd.concat([Y, sind], axis=1)
    .groupby("sind")
    .mean()
    .reset_index()
    .rename(columns={"other_hosps": "mean_other_hosps"})
)
locmeans = (
    pd.concat([Y, sind], axis=1)
    .merge(tmp, on="sind", how="left")
)
offset = locmeans.mean_other_hosps.values
locmeans

# %%
# offset = pop[loc] * M
# np.log((1e-3 + Y.other_hosps.values) / offset).mean()

# %%
model = CmdStanModel(stan_file="model_not_spatial.stan")

# %% fit using variational inference
N = X.shape[0]
DX = X.shape[1]
DW = W.shape[1]
S = W.shape[0]

data = {
    "N": N,
    "DX": DX,
    "DW": DW,
    "S": S,
    "A": A.values[:, 0],
    "Y": Y.values[:, 0],
    "X": X.values,
    "W": W.values,
    "offset": offset,
    "sind": sind.values[:, 0] + 1,
}

# %% fit model, use variational inference
fit = model.variational(
    data=data,
    seed=123,
    algorithm="meanfield",
    iter=5000,
    tol_rel_obj=0.001,
    require_converged=False,
    show_console=True,
    adapt_engaged=False,
    output_dir="bayesian_model/results_fit_data",
    output_samples=1,
    eta=0.03,
    inits={
        "beta_unstruct": np.zeros((S, DX)),
        "gamma_unstruct": np.zeros((S, DX)),
        "delta_beta": np.zeros((DW, DX)),
        "delta_gamma": np.zeros((DW, DX)),
        "omega_beta": np.full((DX, ), 0.05),
        "omega_gamma": np.full((DX, ), 0.05),
    }
)

tau = fit.tau

e# %%
plt.hist(tau)

# %%
