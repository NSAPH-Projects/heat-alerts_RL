
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
            constrain=params["constrain"]
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
    model.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_model.pt"))
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)
    guide.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_guide.pt"))
    
    sample = guide(*inputs)
    
    keys0 = [k for k in sample.keys() if k.startswith("effectiveness_")]
    keys1 = [k for k in sample.keys() if k.startswith("baseline_")]
    keys0.remove("effectiveness_bias")
    keys1.remove("baseline_bias")
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
    base_names = ["QHI", "QHI>25", "QHI>75", "Excess QHI", "Alert Lag1", "Alerts 2wks", "Weekend",
                  "DOS_0", "DOS_1", "DOS_2" #, "Bias"
                  ]
    eff_names = ["QHI", "Excess QHI", "Alert Lag1", "Alerts 2wks", "Weekend", 
                 "DOS_0", "DOS_1", "DOS_2" #, "Bias"
                 ]

    # make coefficient distribution plots for coefficients, error bars are iqr
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(x=eff_names, y=medians_0, yerr=[l0, u0], fmt="o")
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    ax[0].set_title("Effectiveness Coefs Distribution")
    # ax[0].set_ylabel("Coef Value")
    ax[1].errorbar(x=base_names, y=medians_1, yerr=[l1, u1], fmt="o")
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    ax[1].set_title("Baseline Coefs Distribution")
    # ax[1].set_ylabel("Coef Value")
    plt.subplots_adjust(bottom=0.6)
    fig.savefig("heat_alerts/bayesian_model/Plots_params/Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")

    print("Saved coef plot")
    
    ### Custom data:
    new_alert = torch.ones(alert.shape)
    new_base = baseline_features.detach().clone()
    new_base[:, dm.baseline_feature_names.index("alert_lag1")] = torch.zeros(alert.shape)
    # new_base[:, dm.baseline_feature_names.index("alert_lag1")] = torch.ones(alert.shape)
    p = torch.tensor((0 - dm.prev_alert_mean)/(2 * dm.prev_alert_std), dtype=torch.float32)
    # p = torch.tensor((1 - dm.prev_alert_mean)/(2 * dm.prev_alert_std), dtype=torch.float32)
    # p = torch.tensor((2 - dm.prev_alert_mean)/(2 * dm.prev_alert_std), dtype=torch.float32)
    new_base[:, dm.baseline_feature_names.index("previous_alerts")] = p.repeat(alert.shape)
    new_eff = eff_features.detach().clone()
    new_eff[:, dm.effectiveness_feature_names.index("alert_lag1")] = torch.zeros(alert.shape)
    # new_eff[:, dm.effectiveness_feature_names.index("alert_lag1")] = torch.ones(alert.shape)
    new_eff[:, dm.effectiveness_feature_names.index("previous_alerts")] = p.repeat(alert.shape)

    inputs = [
        hosps, 
        loc_ind, 
        county_summer_mean, 
        new_alert,
        new_base, 
        new_eff, 
        index
    ]

    ## Sample coefficients from the rewards model:
    predictive_outputs = Predictive(
            model,
            guide=guide, # including the guide includes all sites by default
            num_samples= params["n_samples"],  
            return_sites=["_RETURN"],
        )
    
    outputs = predictive_outputs(*inputs, condition=False, return_outcomes=True)["_RETURN"]
    # outputs = torch.mean(outputs, dim=0)
    r0 = outputs[:, :, 1]
    r1 = r0*(1-outputs[:, :, 0])
    N = int(r0.shape[1]/n_days)
    # dos = torch.tensor(np.repeat(np.arange(0,n_days), N))
    effect = r1-r0
    Effect = torch.mean(effect, dim=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if params["SC"] == "F":
        for s in np.unique(loc_ind):
            county_dos = torch.reshape(Effect[loc_ind == s], (n_years, n_days))
            ax.plot(county_dos.mean(0), color="k", alpha=0.1, lw=0.5)
        Effect = torch.reshape(Effect, (N, n_days))
        ax.set_ylim(-0.1,0)
        # Upper = torch.quantile(Effect, 0.975, dim = 0)
        # Lower = torch.quantile(Effect, 0.025, dim = 0)
        ax.plot(Effect.mean(0), color="b", lw=2)
        # ax.plot(Upper, color="k", alpha = 0.1, lw=2)
        # ax.plot(Lower, color="k", alpha = 0.1, lw=2)
        ax.set_xlabel("Day of Summer")
        ax.set_title("Effect of Issuing an Alert")
        fig.savefig("heat_alerts/bayesian_model/Plots_params/Overall_effect_w-CI_" + params["model_name"] + ".png", bbox_inches="tight")
        # fig.savefig("heat_alerts/bayesian_model/Plots_params/Lagged_effect_" + params["model_name"] + ".png", bbox_inches="tight")
        # fig.savefig("heat_alerts/bayesian_model/Plots_params/Prev_alerts-2_effect_" + params["model_name"] + ".png", bbox_inches="tight")
    else:
        with open("data/processed/fips2idx.json", "r") as f:
            fips2ix = json.load(f)
            fips2ix = {int(k): v for k, v in fips2ix.items()}
        s = fips2ix[params["county"]]
        county_dos = torch.reshape(Effect[loc_ind == s], (n_years, n_days))
        ax.plot(county_dos.mean(0), color="k")
        ax.set_xlabel("Day of Summer")
        ax.set_title("Effect of Issuing an Alert: County " + str(params["county"]))
        fig.savefig("heat_alerts/bayesian_model/Plots_params/Overall_effect_county-" + str(params["county"]) + "_" + params["model_name"] + ".png", bbox_inches="tight")
    

    # ############################ OLD code:
    # Outputs = torch.mean(outputs, dim=0)
    # eff, baseline, outcome_mean = (
    #                 Outputs[:, 0],
    #                 Outputs[:, 1],
    #                 Outputs[:, 2],
    #             )
    
    # # keys = [k for k in sample.keys() if not "dos_" in k]
    # # base_keys = [k for k in keys if not "eff" in k]
    # # eff_keys = [k for k in keys if not "base" in k]
    # # base_means = [sample[k].mean().detach().numpy() for k in base_keys]
    # # base_stddevs = [sample[k].std().detach().numpy() for k in base_keys]
    # # eff_means = [sample[k].mean().detach().numpy() for k in eff_keys]
    # # eff_stddevs = [sample[k].std().detach().numpy() for k in eff_keys]
    
    # # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # # ax.errorbar(x=base_keys, y=base_means, yerr=base_stddevs, fmt="o")
    # # plt.xticks(rotation=90)
    # # ax.set_title("Baseline Coeff Distribution")
    # # ax.set_ylabel("Coeff Value")
    # # plt.subplots_adjust(bottom=0.6)
    # # fig.savefig("Plots_params/Baseline_Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")

    # # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # # ax.errorbar(x=eff_keys, y=eff_means, yerr=eff_stddevs, fmt="o")
    # # plt.xticks(rotation=90)
    # # ax.set_title("Effectiveness Coeff Distribution")
    # # ax.set_ylabel("Coeff Value")
    # # plt.subplots_adjust(bottom=0.6)
    # # fig.savefig("Plots_params/Effectiveness_Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")

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
