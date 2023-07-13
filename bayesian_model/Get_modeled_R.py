# Run from within bayesian_model/

import argparse
import json
import pandas as pd
import numpy as np
import torch
import itertools

def main(name):

    with open("Plots_params/fit_data_pyro_base_" + name + ".json","r") as f:
        params = json.load(f)

    with open("data/processed/fips2idx.json","r") as f:
        crosswalk = json.load(f)

    n_days = 153
    n_years = 10
    Fips = np.array(list(itertools.chain(*[itertools.repeat(i, n_days*n_years) for i in crosswalk.keys()])))

    dir = "data/processed/"
    X = pd.read_parquet(f"{dir}/states.parquet")
    A = pd.read_parquet(f"{dir}/actions.parquet")
    Y = pd.read_parquet(f"{dir}/outcomes.parquet")
    sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
    W = pd.read_parquet(f"{dir}/spatial_feats.parquet")

    A_ = torch.tensor(A.values[:, 0], dtype=torch.float32)
    Y_ = torch.tensor(Y.values[:, 0], dtype=torch.float32)
    X_ = torch.tensor(X.values, dtype=torch.float32)
    W_ = torch.tensor(W.values, dtype=torch.float32)
    sind_ = torch.tensor(sind.values[:, 0], dtype=torch.long)

    ## Get predictions:
    betas = []
    gammas = []
    for i in range(args.n_samples):
        delta_beta = torch.tensor(params["delta_beta"][i])
        delta_gamma = torch.tensor(params["delta_gamma"][i])
        omega_beta = torch.tensor(params["omega_beta"][i])
        omega_gamma = torch.tensor(params["omega_gamma"][i])
        unstruct_beta = torch.tensor(params["unstruct_beta"][i])
        unstruct_gamma = torch.tensor(params["unstruct_gamma"][i])
        beta = W_ @ delta_beta + omega_beta * unstruct_beta
        gamma = W_ @ delta_gamma + omega_gamma * unstruct_gamma
        betas.append(beta.numpy())
        gammas.append(gamma.numpy())

    betas = np.stack(betas, axis=0)
    gammas = np.stack(gammas, axis=0)

    betas_means = betas.mean((0))
    betas_std = betas.std((0))
    gammas_means = gammas.mean((0))
    gammas_std = gammas.std((0))

    lam = torch.exp((torch.tensor(betas_means)[sind_] * X_).sum(-1).clamp(-20, 10))  # baseline rate
    tau = torch.sigmoid((torch.tensor(gammas_means)[sind_] * X_).sum(-1))  # heat alert effectiveness

    # mu = lam * (1.0 - A_ * tau) 
    R0 = lam.detach().numpy()
    R1 = lam * (1.0 - torch.ones(len(tau)) * tau).detach().numpy() 

    df = pd.DataFrame(np.stack([Fips, R0, R1], axis = 1))
    df.columns = ["fips", "R0", "R1"]
    df.to_csv("../Bayesian_models/Bayesian_R_" + name + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()
    main(args.name)