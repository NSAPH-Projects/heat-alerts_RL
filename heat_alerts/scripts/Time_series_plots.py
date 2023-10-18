
import numpy as np

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

for r in np.unique(Region):
    county = counties[Region.index(r)]

    base_dict, effect_dict, extra_dict, other_dict = load_rl_states_by_county(
            county,
            "data/processed",
            years,
            match_similar=True,
            include_COI=True,
            as_tensors=False,
            incorp_forecasts=True,
        )

    




