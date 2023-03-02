
from d3rlpy.dataset import MDPDataset

import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing as skprep

import torch
from torch.nn import functional as F

def make_data(
    filename="data/Train_smaller-for-Python.csv", 
    outcome="deaths", 
    modeled_r = False, 
    log_r = True,
    random_effects = False
):
    ## Read in data:
    Train = pd.read_csv(filename)

    # n_counties = Train["fips"].nunique()
    # n_years = 11
    n_days = 153

    actions = Train["alert"]
    terminals = Train["dos"] == n_days

    ## Prepare rewards:
    if modeled_r == False:
        if outcome == "all_hosps":
            rewards = -1*(Train["all_hosps"]/Train["total_count"])
        elif outcome == "other_hosps":
            rewards = -1*(Train["other_hosps"]/Train["total_count"])
        else:
            rewards = -1*(Train["N"]/Train["Pop.65"])
        rewards = rewards.to_numpy()
    else:
        if outcome == "all_hosps":
            rewards = pd.read_csv("Fall_results/R_2-28_all-hosps_all.csv") # don't use this one
        elif outcome == "other_hosps":
            rewards = pd.read_csv("Fall_results/R_2-28_other-hosps_all.csv")
        else:
            rewards = pd.read_csv("Fall_results/R_1-23_deaths.csv") # would need to get deaths for d=153
        rewards = torch.gather(torch.FloatTensor(rewards.to_numpy()), 1, torch.LongTensor(actions).view(-1, 1) +1).view(-1).detach().numpy()

    if log_r == True:
        rewards = -np.log(-rewards + 0.0000000001)

    rewards = (rewards - rewards.mean()) / np.max(np.abs(rewards)) # removed scaling by 0.5
    
    ## Prepare observations (S):
    budget = Train[Train["dos"] == n_days]["alert_sum"]
    Budget = list(itertools.chain(*[itertools.repeat(b, n_days) for b in budget]))
    Train["More_alerts"] = Budget - Train["alert_sum"]

    States = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "BA_zone", "l.Pop_density", "l.Med.HH.Income",
                        "year", "dos", "holiday", "dow", 
                        "alert_lag1", "alert_lag2", "alerts_2wks", "T_since_alert",
                        "alert_sum", "More_alerts",
                        "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                        "broadband.usage", "Democrat", "Republican", "pm25"]]

    ## One-hot encode non-numeric variables
    S_enc = skprep.OneHotEncoder(drop = "first")
    S_enc.fit(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]])
    S_ohe = S_enc.transform(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]]).toarray()
    S_names = S_enc.get_feature_names_out(["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"])
    S_OHE = pd.DataFrame(S_ohe, columns=S_names)

    ## Standardize numeric variables
    num_vars = ["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "l.Pop_density", "l.Med.HH.Income", 
                        "year", "dos", "alerts_2wks", "T_since_alert",
                         "alert_sum", "More_alerts",
                         "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                         "broadband.usage", "Democrat", "Republican", "pm25"]

    s_means = States[num_vars].mean(0)
    s_stds = States[num_vars].std(0)
    S = (States[num_vars] - s_means)/s_stds

    if random_effects == True:
        if outcome == "all_hosps":
            rand_effs = pd.read_csv("Fall_results/R_2-11_all-hosps_random-effects_all.csv")
        elif outcome == "other_hosps":
            rand_effs = pd.read_csv("Fall_results/R_2-11_other-hosps_random-effects_all.csv")
        else:
            rand_effs = pd.read_csv("Fall_results/R_1-23_deaths_random-effects.csv") # would need to get deaths for d=153
        S["rand_ints"] = rand_effs["Rand_Ints"]
        S["rand_slopes"] = F.softplus(torch.FloatTensor(rand_effs["Rand_Slopes"].to_numpy()))
    
    observations = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)
    observations.drop("index", axis=1)

    ## Put everything together:
    dataset = MDPDataset(
        observations.to_numpy(), actions.to_numpy(), 
        rewards, terminals.to_numpy()
    )

    return dataset


if __name__ == "__main__":
    D = make_data()