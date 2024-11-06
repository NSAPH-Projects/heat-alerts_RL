
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import json
import csv
import pandas as pd

import pyro
from pyro.distributions import Poisson
import pytorch_lightning as pl
import torch

# from heat_alerts.bayesian_model.pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
#                              HeatAlertModel)
from pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
                             HeatAlertModel)
from pyro.infer import Predictive, predictive

def main(params):
    # params = {"model_name": "FF_C-M_wide-EB-prior", "n_samples": 100, "SC": "F", "county": 36005, "constrain": "mixed"}
    params = vars(params)
    ## Read in data:
    n_days = 153
    years = set(range(2006, 2017))
    n_years = len(years)
    dm = HeatAlertDataModule(
            dir="data/processed", # dir="data/processed",
            batch_size=n_days*n_years,
            num_workers=4,
            for_gym=True,
            constrain=params["constrain"], 
            load_outcome=False,
        )
    data = dm.gym_dataset
    hosps = data[0]
    loc_ind = data[1].long()
    county_summer_mean = data[2]
    alert = data[3]
    baseline_features = data[4]
    eff_features = data[5]
    index = data[6]

    ### Observed data:
    inputs = [
        hosps, 
        loc_ind, 
        county_summer_mean, 
        alert,
        baseline_features, 
        eff_features, 
        index
    ]

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

    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)
    guide.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_guide.pt"))
    
    sample = guide(*inputs)
    
    keys0 = [k for k in sample.keys() if k.startswith("effectiveness_")]
    keys1 = [k for k in sample.keys() if k.startswith("baseline_")]
    keys0 = keys0[0:-4]
    keys1 = keys1[0:-4]
    # keys0.remove("effectiveness_bias", 'effectiveness_dos_0', 'effectiveness_dos_1', 'effectiveness_dos_2')
    # keys1.remove("baseline_bias", 'baseline_dos_0', 'baseline_dos_1', 'baseline_dos_2')
    medians_0 = np.array(
        [torch.quantile(sample[k], 0.5).item() for k in keys0]
    )
    medians_1 = np.array(
        [torch.quantile(sample[k], 0.5).item() for k in keys1]
    )
    q25_0 = np.array(
        [torch.quantile(sample[k], 0.25).item() for k in keys0]
    )
    q25_1 = np.array(
        [torch.quantile(sample[k], 0.25).item() for k in keys1]
    )
    q75_0 = np.array(
        [torch.quantile(sample[k], 0.75).item() for k in keys0]
    )
    q75_1 = np.array(
        [torch.quantile(sample[k], 0.75).item() for k in keys1]
    )
    l0, u0 = medians_0 - q25_0, q75_0 - medians_0
    l1, u1 = medians_1 - q25_1, q75_1 - medians_1
    base_names = [k.split("baseline_")[1] for k in keys1]
    eff_names = [k.split("effectiveness_")[1] for k in keys0]
    base_names = ["QHI", "QHI>25", "QHI>75", "Excess QHI", "(-)Alert Lag1", "(-)Alerts 2wks", "Weekend"
                  #, "DOS_0", "DOS_1", "DOS_2" #, "Bias"
                  ]
    eff_names = ["(+)QHI", "(+)Excess QHI", "Alert Lag1", "Alerts 2wks", "Weekend"
                 #, "DOS_0", "DOS_1", "DOS_2" #, "Bias"
                 ]

    Sample = [guide(*inputs) for i in range(100)]
    Medians_0 = [np.median(np.array([sample[k].detach().numpy() for sample in Sample])) for k in keys0]
    Medians_1 = [np.median(np.array([sample[k].detach().numpy() for sample in Sample])) for k in keys1]
    Q25_0 = [np.percentile(np.array([sample[k].detach().numpy() for sample in Sample]), 25) for k in keys0]
    Q25_1 = [np.percentile(np.array([sample[k].detach().numpy() for sample in Sample]), 25) for k in keys1]
    Q75_0 = [np.percentile(np.array([sample[k].detach().numpy() for sample in Sample]), 75) for k in keys0]
    Q75_1 = [np.percentile(np.array([sample[k].detach().numpy() for sample in Sample]), 75) for k in keys1]
    L0, U0 = np.subtract(Medians_0, Q25_0), np.subtract(Q75_0, Medians_0)
    L1, U1 = np.subtract(Medians_1, Q25_1), np.subtract(Q75_1, Medians_1)

    # make coefficient distribution plots for coefficients, error bars are iqr
    # width = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(6, 2))
    # ax[0].bar(np.arange(len(eff_names)) - width/2, medians_0, width, yerr=[l0, u0], label='1 sample', capsize=5)
    # ax[0].bar(np.arange(len(eff_names)) + width/2, Medians_0, width, yerr=[L0, U0], label='100 samples', capsize=5)
    ax[0].errorbar(x=eff_names, y=medians_0, yerr=[l0, u0], fmt="o")
    # ax[0].errorbar(x=eff_names, y=Medians_0, yerr=[L0, U0], fmt="s")
    # ax[0].set_xticks(np.arange(len(eff_names)))
    # ax[0].set_xticklabels(eff_names)
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    ax[0].set_title("Effectiveness Coefs Distribution")
    # ax[0].set_ylabel("Coef Value")
    # ax[1].bar(np.arange(len(base_names)) - width/2, medians_1, width, yerr=[l1, u1], label='1 sample', capsize=5)
    # ax[1].bar(np.arange(len(base_names)) + width/2, Medians_1, width, yerr=[L1, U1], label='100 samples', capsize=5)
    ax[1].errorbar(x=base_names, y=medians_1, yerr=[l1, u1], fmt="o")
    # ax[1].errorbar(x=base_names, y=Medians_1, yerr=[L1, U1], fmt="s")
    # ax[1].set_xticks(np.arange(len(base_names)))
    # ax[1].set_xticklabels(base_names)
    # ax[1].set_xlabel('Categories')
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    ax[1].set_title("Baseline Coefs Distribution")
    # ax[1].set_ylabel("Coef Value")
    # plt.subplots_adjust(hspace=1.2)
    fig.savefig("heat_alerts/bayesian_model/Plots_params/Coefficients_" + params["model_name"] + "_horizontal.png", bbox_inches="tight")

    print("Saved coef plot")

  
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
    fig.savefig("heat_alerts/bayesian_model/Plots_params/DOS_" + params["model_name"] + ".png", bbox_inches="tight")

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Full_8-7", help="model name")
    parser.add_argument("--constrain", type=str, default="all", help="model constraints?")
    parser.add_argument("--n_samples", type=int, default=100, help="number of samples to take")
    parser.add_argument("--SC", type=str, default="F", help="Make plot for single county?")
    parser.add_argument("--county", type=int, default=36005, help="county to make plots for")
    args = parser.parse_args()
    main(args)
