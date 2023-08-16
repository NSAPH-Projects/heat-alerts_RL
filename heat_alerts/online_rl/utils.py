from heat_alerts.bayesian_model import HeatAlertDataModule
import numpy as np
import pandas as pd


WESTERN_STATES = [
    "AZ",
    "CA",
    "CO",
    "ID",
    "MT",
    "NM",
    "NV",
    "OR",
    "WA",
    "ND",
    "SD",
    "NE",
    "KS",  # adds 10 counties to the Cold western group
]


# def get_similar_counties(
#     loc,
#     spatial_features=global_spatial_features,
#     locations=global_locations,
#     western=global_western,
# ):
#     this_spat = spatial_features.loc[loc]
#     group = locations[
#         spatial_features[
#             ["Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold"]
#         ]
#         .eq(
#             this_spat[
#                 ["Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold"]
#             ]
#         )
#         .all(axis=1)
#     ]
#     if (
#         this_spat["Cold"] == 1.0
#     ):  # additionally separating the "Cold" region into western and eastern
#         this_w = this_spat["State"] in western
#         if this_w:
#             group = group[np.isin(spatial_features.loc[group]["State"], western)]
#         else:
#             group = group[
#                 np.isin(spatial_features.loc[group]["State"], western, invert=True)
#             ]
#     return group


def load_rl_data(path: str):
    # we will load the same data used for the bayesian model
    # to have the same features and computation
    # but perhaps there could be an even simpler solution
    dm = HeatAlertDataModule(path, load_outcome=False, for_gym=True)
    (
        _,
        loc,
        county_summer_mean,
        alert,
        baseline_features_tensor,
        effectiveness_features_tensor,
        _,
        year,
        budget,
        state,
        hi_mean,
    ) = dm.gym_dataset

    years = np.unique(year)
    num_years = len(years)

    # extract feature names
    base_feats = pd.DataFrame(
        baseline_features_tensor.numpy(), columns=dm.baseline_feature_names
    )
    dos = base_feats["dos_0"]
    n_days = len(dos.unique())
    base_feats["hi_mean"] = hi_mean.numpy()
    eff_feats = pd.DataFrame(
        effectiveness_features_tensor.numpy(), columns=dm.effectiveness_feature_names
    )
    eff_feats["hi_mean"] = hi_mean.numpy()
    spatial_feats = pd.DataFrame(
        dm.spatial_features.numpy(),
        columns=dm.spatial_features_names,
        index=dm.spatial_features_idx,
    )
    county2ix = {c: i for i, c in enumerate(dm.spatial_features_idx)}
    counties = list(county2ix.keys())

    # fixed baseline features
    base_names = [
        "heat_qi_base",
        "heat_qi1_above_25",
        "heat_qi2_above_75",
        "excess_heat",
        "hi_mean",
        "weekend",
        "dos_0",
        "dos_1",
        "dos_2",
    ]
    # base = base_feats[base_names]
    eff_names = [
        "heat_qi",
        "excess_heat",
        "weekend",
        "hi_mean",
        "dos_0",
        "dos_1",
        "dos_2",
    ]
    eff = eff_feats[eff_names]

    # baseline feats
    base = {}
    for f in base_names:
        D = base_feats[f]
        pass


#     # s

# ## Read in data:
# global_n_days = 153
# global_years = set(range(2006, 2017))
# global_n_years = len(global_years)

# global_baseline_feature_names = dm.baseline_feature_names
# global_effectiveness_feature_names = dm.effectiveness_feature_names

# global_baseline_weather_names = [
#     "heat_qi_base",
#     "heat_qi1_above_25",
#     "heat_qi2_above_75",
#     "excess_heat",
# ]
# global_effectiveness_weather_names = ["heat_qi", "excess_heat"]

# data = dm.gym_dataset
# # Call these "global_*" so we don't accidentally use them directly inside the environment...
# global_hosps = data[0]
# global_loc_ind = data[1].long()
# global_county_summer_mean = data[2]
# global_alert = data[3]
# global_baseline_features = data[4]
# global_eff_features = data[5]
# global_index = data[6]
# global_year = data[7]
# global_Budget = data[8]
# global_hi_mean = data[10]
# global_state = data[9]
# # Get unique state IDs:
# W_state = global_state.loc[
#     np.arange(0, len(global_state), global_n_days * global_n_years)
# ]
# s = np.unique(W_state)
# W_state = W_state["state"].replace(s, np.arange(0, len(s)))

# # The "Cold" zone is very large, so split on additional east-west feature:
# global_western = [
#     " AZ",
#     " CA",
#     " CO",
#     " ID",
#     " MT",
#     " NM",
#     " NV",
#     " OR",
#     " WA",
#     " ND",
#     " SD",
#     " NE",
#     " KS",  # adds 10 counties to the Cold western group
# ]
# global_western = pd.DataFrame(global_western).replace(s, np.arange(0, len(s)))
# global_western.columns = ["Num"]

# # Add state ID to spatial features data frame:
# global_spatial_features = dm.spatial_features
# global_spatial_features = torch.cat(
#     (
#         global_spatial_features,
#         torch.tensor(np.array(W_state), dtype=torch.int).reshape(-1, 1),
#     ),
#     dim=1,
# )
# global_spatial_features = pd.DataFrame(global_spatial_features.numpy())

# global_spatial_feature_names = dm.spatial_features_names
# global_spatial_feature_names = global_spatial_feature_names.union(["State"], sort=False)
# global_spatial_features.columns = global_spatial_feature_names

# global_locations = np.arange(0, global_spatial_features.shape[0])


# # Function for obtaining counties with similar weather:
# def get_similar_counties(
#     loc,
#     spatial_features=global_spatial_features,
#     locations=global_locations,
#     western=global_western,
# ):
#     this_spat = spatial_features.loc[loc]
#     group = locations[
#         spatial_features[
#             ["Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold"]
#         ]
#         .eq(
#             this_spat[
#                 ["Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold"]
#             ]
#         )
#         .all(axis=1)
#     ]
#     if (
#         this_spat["Cold"] == 1.0
#     ):  # additionally separating the "Cold" region into western and eastern
#         this_w = this_spat["State"] in western
#         if this_w:
#             group = group[np.isin(spatial_features.loc[group]["State"], western)]
#         else:
#             group = group[
#                 np.isin(spatial_features.loc[group]["State"], western, invert=True)
#             ]
#     return group

if __name__ == "__main__":
    dir = "data/processed/bayesian_model"
    county = "48453"
    load_rl_data(dir)
