from d3rlpy.dataset import MDPDataset

import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing as skprep
from sklearn.decomposition import PCA

import torch
from torch.nn import functional as F


def symlog(x, shift=1):
    if x >= 0:
        return np.log(x+shift)-np.log(shift)
    else:
        return -np.log(-x+shift)+np.log(shift)


def make_data(
    filename="data/Summer23_Train_smaller-for-Python.csv", 
    fips = 4013,
    modeled_r = "F", 
    eligible = "all",
    std_budget = 0
):
    ## Subset the county-level data and prepare the rewards:
    Train = pd.read_csv(filename)
    county_pos = Train.index[Train["fips"] == fips].tolist()
    elig = Train.loc[county_pos]

    n_days = 153
    if std_budget == 0:
        budget = elig[elig["dos"] == n_days]["alert_sum"]
        Budget = list(itertools.chain(*[itertools.repeat(b, n_days) for b in budget]))
    else: 
        Budget = std_budget
    elig["More_alerts"] = Budget - elig["alert_sum"]

    if modeled_r == "F":
        rewards = -1*(elig["other_hosps"]/elig["total_count"])
        rewards = (rewards - rewards.mean())/rewards.std()
        # rewards = rewards.apply(symlog,shift=1)
        rewards = rewards.to_numpy()
    else:
        # rewards = pd.read_csv("Summer_results/R_6-28_forced_small-S_all.csv")
        rewards = pd.read_csv("Summer_results/Bayesian_R_7-12.csv")
        rewards = rewards.loc[rewards["fips"] == fips]
        rewards = torch.gather(torch.FloatTensor(-rewards.to_numpy()), 1, torch.LongTensor(elig["alert"].to_numpy()).view(-1, 1) +2).view(-1)
        rewards = -torch.log(-rewards).detach().numpy()
        rewards = 0.5 * (rewards - rewards.mean()) / np.max(np.abs(rewards))

    if eligible == "all":
        terminals = Train["dos"] == 153
    elif eligible == "90pct":
        rewards = rewards[elig["quant_HI_county"] >= 0.9]
        HI_pos = elig.index[elig["quant_HI_county"] >= 0.9].tolist()
        elig = elig.loc[HI_pos]
        terminals = []
        y = 2006
        d = 1
        while d < len(elig):
            if elig["year"].iloc[d] == y:
                terminals.append(0)
            else: 
                terminals.append(1)
                y += 1
            d += 1
        terminals.append(1)

    ## Prepare observations (S)... 
    # States = elig[["quant_HI_county", "HI_mean", "year", "dos", 
    #                "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate"]]
    States = elig[[
        # "quant_HI_county", "quant_HI_3d_county", "year", "weekend", "T_since_alert",
        "quant_HI", "quant_HI_3d", "dos", "alert_sum", "More_alerts"
    ]]

    s_means = States.mean(0)
    s_stds = States.std(0)
    S = (States - s_means)/s_stds

    observations = S.reset_index().drop("index", axis=1)

    # S_enc = skprep.OneHotEncoder(drop = "first")
    # S_enc.fit(elig[["dow"]])
    # S_ohe = S_enc.transform(elig[["dow"]]).toarray()
    # S_names = S_enc.get_feature_names_out(["dow"]) 
    # S_OHE = pd.DataFrame(S_ohe, columns=S_names)

    # observations["weekend"] = S_OHE["dow_Saturday"] + S_OHE["dow_Sunday"]
    print(observations.columns)

    dataset = MDPDataset(
        observations.to_numpy(), elig["alert"].to_numpy(),
        rewards, np.array(terminals)
    )
    
    return [dataset, s_means, s_stds]