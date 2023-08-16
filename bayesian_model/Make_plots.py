
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt

import pyro
import pytorch_lightning as pl
import torch

from pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
                             HeatAlertModel)
from pyro.infer import Predictive, predictive

def main(params):
    # params = {"model_name": "Full_8-14", "n_samples": 1}
    ## Read in data:
    n_days = 153
    years = set(range(2006, 2017))
    n_years = len(years)
    dm = HeatAlertDataModule(
            dir="data/processed", # dir="data/processed",
            batch_size=n_days*n_years,
            num_workers=4,
            for_gym=True
        )
    data = dm.gym_dataset
    hosps = data[0]
    loc_ind = data[1].long()
    county_summer_mean = data[2]
    alert = data[3]
    baseline_features = data[4]
    eff_features = data[5]
    index = data[6]

    ## Set up the rewards model, previously trained using pyro:
    model = HeatAlertModel(
            spatial_features=dm.spatial_features,
            data_size=dm.data_size,
            d_baseline=dm.d_baseline,
            d_effectiveness=dm.d_effectiveness,
            baseline_constraints=dm.baseline_constraints,
            baseline_feature_names=dm.baseline_feature_names,
            effectiveness_constraints=dm.effectiveness_constraints,
            effectiveness_feature_names=dm.effectiveness_feature_names,
            hidden_dim= 32, #cfg.model.hidden_dim,
            num_hidden_layers= 1, #cfg.model.num_hidden_layers,
        )
    model.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_model.pt"))
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)
    guide.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_guide.pt"))

    ## Sample coefficients from the rewards model:
    predictive_outputs = Predictive(
            model,
            guide=guide, # including the guide includes all sites by default
            num_samples= 1,#params["n_samples"],  
            return_sites=["_RETURN"],
        )
    inputs = [
        hosps, 
        loc_ind, 
        county_summer_mean, 
        alert,
        baseline_features, 
        eff_features, 
        index
    ]
    outputs = predictive_outputs(*inputs, condition=False, return_outcomes=True)["_RETURN"][0]
    eff, baseline, outcome_mean = (
                    outputs[:, 0],
                    outputs[:, 1],
                    outputs[:, 2],
                )
    sample = guide(*inputs)
    keys = [k for k in sample.keys() if not "dos_" in k]
    base_keys = [k for k in keys if not "eff" in k]
    eff_keys = [k for k in keys if not "base" in k]
    base_means = [sample[k].mean().detach().numpy() for k in base_keys]
    base_stddevs = [sample[k].std().detach().numpy() for k in base_keys]
    eff_means = [sample[k].mean().detach().numpy() for k in eff_keys]
    eff_stddevs = [sample[k].std().detach().numpy() for k in eff_keys]
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(x=base_keys, y=base_means, yerr=base_stddevs, fmt="o")
    plt.xticks(rotation=90)
    ax.set_title("Baseline Coeff Distribution")
    ax.set_ylabel("Coeff Value")
    plt.subplots_adjust(bottom=0.6)
    fig.savefig("Plots_params/Baseline_Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(x=eff_keys, y=eff_means, yerr=eff_stddevs, fmt="o")
    plt.xticks(rotation=90)
    ax.set_title("Effectiveness Coeff Distribution")
    ax.set_ylabel("Coeff Value")
    plt.subplots_adjust(bottom=0.6)
    fig.savefig("Plots_params/Effectiveness_Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")

    # now a plot of the effect of day of summer
    n_basis = dm.dos_spline_basis.shape[1]
    basis = dm.dos_spline_basis
    eff_coefs = [sample[f"effectiveness_dos_{i}"] for i in range(n_basis)]
    baseline_coefs = [sample[f"baseline_dos_{i}"] for i in range(n_basis)]
    eff_contribs = [
        basis[:, i] * eff_coefs[i][:, None] for i in range(n_basis)
    ]  # list of len(n_basis) each of size (S, T)
    baseline_contribs = [
        basis[:, i] * baseline_coefs[i][:, None] for i in range(n_basis)
    ]
    dos_beta_eff = sum(baseline_contribs).detach().numpy()
    dos_gamma_eff = sum(eff_contribs).detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(dos_beta_eff.T, color="k", alpha=0.05, lw=0.5)
    ax[0].plot(dos_beta_eff.mean(0), color="k", lw=2)
    ax[0].set_xlabel("Day of summer")
    ax[0].set_title("Baseline rate")
    ax[1].plot(dos_gamma_eff.T, color="k", alpha=0.05, lw=0.5)
    ax[1].plot(dos_gamma_eff.mean(0), color="k", lw=2)
    ax[1].set_xlabel("Day of summer")
    ax[1].set_title("Heat alert effectiveness")
    fig.savefig("Plots_params/DOS_" + params["model_name"] + ".png", bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Full_8-7", help="model name")
    parser.add_argument("--n_samples", type=int, default=1000, help="number of samples to take")
    args = parser.parse_args()
    main(args)