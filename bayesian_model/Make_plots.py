
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
    # params = {"model_name": "Full_8-7", "n_samples": 1}
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
    keys = [k for k in keys if not "eff" in k]
    # means = [sample[k].mean().detach().numpy() for k in keys]
    # stddevs = [sample[k].std().detach().numpy() for k in keys]
    means = [sample[k].mean().detach().numpy() for k in keys if not "eff" in k]
    stddevs = [sample[k].std().detach().numpy() for k in keys if not "eff" in k]
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(x=keys, y=means, yerr=stddevs, fmt="o")
    plt.xticks(rotation=90)
    ax.set_title("coeff distribution of one sample across locs")
    ax.set_ylabel("coeff value")
    plt.subplots_adjust(bottom=0.6)
    fig.savefig("Plots_params/Coefficients_" + params["model_name"] + ".png", bbox_inches="tight")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Full_8-7", help="model name")
    parser.add_argument("--n_samples", type=int, default=1000, help="number of samples to take")
    args = parser.parse_args()
    main(args)