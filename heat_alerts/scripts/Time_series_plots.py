
import numpy as np
import json
import pandas as pd

from heat_alerts.bayesian_model.pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
                             HeatAlertModel)

from heat_alerts.online_rl.datautils import load_rl_states_by_county

import pyro
import torch

counties = [41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025]

Region = [
    "Marine", "Marine", "Mixed Humid", "Mixed Humid", "Hot Humid", "Hot Humid",
    "Cold", "Cold", "Cold", "Hot Dry", "Hot Dry", "Cold", "Cold", "Cold", "Cold", 
    "Mixed Humid", "Mixed Humid", "Mixed Humid", "Mixed Humid", "Mixed Humid", "Mixed Humid", 
    "Hot Humid", "Hot Humid", "Hot Humid", "Hot Humid", "Hot Humid", "Marine",
    "Hot Dry", "Hot Dry", "Hot Dry"
]

years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
n_days = 153

dm = HeatAlertDataModule(
        dir="data/processed", load_outcome=False, constrain="mixed", 
    )

model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectiveness_constraints=dm.effectiveness_constraints,
        effectiveness_feature_names=dm.effectiveness_feature_names,
        hidden_dim=32,
        num_hidden_layers=1,
    )

guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
guide(*dm.dataset.tensors)
guide.load_state_dict(torch.load("ckpts/FF_C-M_wide-EB-prior_guide.pt", map_location=torch.device("cpu")))

n_samples = 100
with torch.no_grad():
    Samples = [guide(*dm.dataset.tensors) for _ in range(n_samples)]

def _clean_key(k: str):
    "some transforms caused by code inconsistencies, should eventually remove"
    return k.replace("qi_base", "qi").replace("qi1", "qi").replace("qi2", "qi")

with open("data/processed/fips2idx.json", "r") as f:
    fips2ix = json.load(f)
    fips2ix = {int(k): v for k, v in fips2ix.items()}

for r in np.unique(Region):
    county = counties[Region.index(r)]
    ## Read in data from counties in the region:
    base_dict, effect_dict, extra_dict, other_dict = load_rl_states_by_county(
            county,
            "data/processed",
            years,
            match_similar=True,
            include_COI=True,
            as_tensors=False,
            incorp_forecasts=True,
        )
    ## Start building the output dataframe:
    QHI = base_dict['baseline_heat_qi']
    Data = pd.DataFrame(np.array(QHI))
    Data["Type"] = "Heat Index"
    Data["ID"] = QHI.index
    ## Calculate averages across the time series, for calculating the rewards:
    baseline_states = {
        k: np.mean(base_dict[k], axis=0) for k in base_dict.keys()
    }
    effectiveness_states = {
        k: np.mean(effect_dict[k], axis=0) for k in effect_dict.keys()
    }
    ## Calculate rewards:
    region_counties = [x for ind,x in enumerate(counties) if Region[ind] == r]
    for rc in region_counties:
        ix = fips2ix[rc]
        samples = {
            _clean_key(k): np.array([s[k][ix].item() for s in Samples]) for k in Samples[0].keys()
        }
        for j in range(0, n_samples):
            baseline_contribs = np.array([
                baseline_states[k] * samples[k][j]
                for k in baseline_states
            ])
            effectiveness_contribs = np.array([
                effectiveness_states[k] * samples[k][j]
                for k in effectiveness_states
            ])
            baseline = np.exp(np.sum(baseline_contribs, axis=0) +
                            0 * samples["baseline_previous_alerts"][j] +
                            ((0 - dm.prev_alert_mean)/(2 * dm.prev_alert_std)) * samples["baseline_alert_lag1"][j] +
                            samples["baseline_bias"][j])
            effectiveness = np.exp(np.sum(effectiveness_contribs, axis=0) +
                            0 * samples["effectiveness_previous_alerts"][j] +
                            ((0 - dm.prev_alert_mean)/(2 * dm.prev_alert_std)) * samples["effectiveness_alert_lag1"][j] +
                            samples["effectiveness_bias"][j])
            lam = pd.DataFrame(np.append(baseline, ["Lambda", rc]))
            Lam = lam.T
            Lam.columns = Data.columns
            Data = pd.concat([Data, Lam])
            tau = pd.DataFrame(np.append(effectiveness, ["Tau", rc]))
            Tau = tau.T
            Tau.columns = Data.columns
            Data = pd.concat([Data, Tau])
        print(rc)
    Data.to_csv("heat_alerts/time_series_data/HI-lam-tau_" + r + ".csv")

    




