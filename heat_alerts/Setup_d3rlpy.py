
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
    filename="data/Train_smaller-for-Python.csv", 
    outcome="other_hosps", 
    modeled_r = False, 
    # log_r = True,
    random_effects = False,
    std_budget = 0,
    eligible = "all",
    pca = False, pca_var_thresh = 0.5, 
    manual_S_size = "medium"
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
    else:
        if outcome == "all_hosps":
            # rewards = pd.read_csv("Fall_results/R_2-28_all-hosps_all.csv") # don't use this one
            rewards = pd.read_csv("Fall_results/R_3-2_all-hosps_all.csv")
        elif outcome == "other_hosps":
            # rewards = pd.read_csv("Fall_results/R_2-28_other-hosps_all.csv")
            rewards = pd.read_csv("Fall_results/R_3-2_other-hosps_all.csv")
        else:
            rewards = pd.read_csv("Fall_results/R_1-23_deaths.csv") # would need to get deaths for d=153
        rewards = torch.gather(torch.FloatTensor(rewards.to_numpy()), 1, torch.LongTensor(actions).view(-1, 1) +1).view(-1).detach().numpy()

     # if log_r == True:
    #     rewards = -np.log(-rewards + 0.0000000001)

    rewards = (rewards - rewards.mean())/rewards.std()
    # rewards = (rewards - rewards.mean()) / np.max(np.abs(rewards))
    # rewards = rewards / np.max(np.abs(rewards)) # include scaling by 0.5?
    
    rewards = rewards.apply(symlog,shift=1)
    # rewards = rewards.apply(symlog, shift=0.01)
    rewards = rewards.to_numpy()

    ## Prepare observations (S):
    if std_budget == 0:
        budget = Train[Train["dos"] == n_days]["alert_sum"]
        Budget = list(itertools.chain(*[itertools.repeat(b, n_days) for b in budget]))
    else: 
        Budget = std_budget
    Train["More_alerts"] = Budget - Train["alert_sum"]

    States = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "BA_zone", "l.Pop_density", "l.Med.HH.Income",
                        "year", "dos", "holiday", "dow", 
                        "alert_lag1", "alert_lag2", "alerts_2wks", "T_since_alert",
                        "alert_sum", "More_alerts",
                        "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                        "broadband.usage", "Democrat", "Republican", "pm25"]] #, "STNAME"

    ## One-hot encode non-numeric variables
    S_enc = skprep.OneHotEncoder(drop = "first")
    S_enc.fit(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]]) #, "STNAME"
    S_ohe = S_enc.transform(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]]).toarray() #, "STNAME"
    S_names = S_enc.get_feature_names_out(["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]) #, "STNAME"
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
            rand_effs = pd.read_csv("Fall_results/R_3-2_all-hosps_random-effects_all.csv")
        elif outcome == "other_hosps":
            rand_effs = pd.read_csv("Fall_results/R_3-2_other-hosps_random-effects_all.csv")
        else:
            rand_effs = pd.read_csv("Fall_results/R_1-23_deaths_random-effects.csv") # would need to get deaths for d=153
        S["rand_ints"] = rand_effs["Rand_Ints"]
        S["rand_slopes"] = F.softplus(torch.FloatTensor(rand_effs["Rand_Slopes"].to_numpy()))
    
    observations = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)
    observations = observations.drop("index", axis=1)
    # observations = observations[["quant_HI_county","More_alerts"]]
    # print(observations.columns)

    if pca == True:
        pca_fit = PCA().fit(observations)
        n_large = np.sum(pca_fit.explained_variance_ > pca_var_thresh)
        observations = np.matmul(pca_fit.components_[0:n_large, :], observations.transpose()).transpose() # each component is a row 

    if manual_S_size == "medium":
        observations["weekend"] = observations["dow_Saturday"] + observations["dow_Sunday"]
        observations = observations[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate", 
            "Republican", "pm25", "weekend", 'BA_zone_Hot-Dry',
            'BA_zone_Hot-Humid', 'BA_zone_Marine', 'BA_zone_Mixed-Dry',
            'BA_zone_Mixed-Humid', 'BA_zone_Very Cold'
        ]
        ]
    elif manual_S_size == "small":
        observations["weekend"] = observations["dow_Saturday"] + observations["dow_Sunday"]
        observations = observations[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate", 
            "weekend"
        ]
        ]
    elif manual_S_size == "tiny":
        observations = observations[[
            "quant_HI_county", "More_alerts", "dos", "T_since_alert", "all_hosp_mean_rate"
        ]
        ]


    ## Put everything together:
    # summer = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(observations.shape[0]/(n_days-1)))]))
    fips = Train.fips.unique()
    a = []
    b = []
    c = []
    d = []

    if eligible == "all":
        dataset = MDPDataset(
            observations.to_numpy(), actions.to_numpy(), 
            rewards, terminals.to_numpy()
        )
        ## Calculate constants (+-) for alerts and budget violations:
        for f in fips:
            pos = np.where(Train["fips"] == f)
            n = len(pos[0])
            inds = np.nonzero(actions.to_numpy()[pos])
            a.extend([np.min(rewards[pos][inds])]*n)
            b.extend([np.max(np.delete(rewards[pos], inds))]*n)
            c.extend([np.max(rewards[pos][inds])]*n)
            d.extend([np.min(np.delete(rewards[pos], inds))]*n)
        # inds = np.nonzero(actions.to_numpy())
        # a = np.min(rewards[inds])
        # b = np.max(np.delete(rewards, inds))
        # c = np.max(rewards[inds])
        # d = np.min(np.delete(rewards, inds))
    else: 
        elig = pd.read_csv("data/Pct_90_eligible.csv") # could include other options too
        Elig = elig.index[elig["Pct_90_eligible"]]
        terminals = pd.read_csv("data/Pct_90_eligible_terminals.csv")
        dataset = MDPDataset(
            observations.iloc[Elig].to_numpy(), actions[Elig].to_numpy(), 
            rewards[Elig], terminals.to_numpy()
        )
        # summer = summer[Elig]
        ## Calculate constants (+-) for alerts and budget violations:
        for f in fips:
            pos = np.where(Train.iloc[Elig]["fips"] == f)
            n = len(pos[0])
            try:
                inds = np.nonzero(actions.to_numpy()[Elig][pos])
                a.extend([np.min(rewards[Elig][pos][inds])]*n)
                b.extend([np.max(np.delete(rewards[Elig][pos], inds))]*n)
                c.extend([np.max(rewards[Elig][pos][inds])]*n)
                d.extend([np.min(np.delete(rewards[Elig][pos], inds))]*n)
            except:
                a.extend([0]*n)
                b.extend([0]*n)
                c.extend([0]*n)
                d.extend([0]*n)
                # print(f)

    boost = np.array(b) - np.array(a) + 1e-10 # to be added when alert is issued (not in violation of budget)  
    penalty = np.array(c) - np.array(d) + 1e-10 # to be subtracted when budget is violated (in CPQ)
    return [dataset, boost, penalty, s_means, s_stds] # , summer


    


if __name__ == "__main__":
    D = make_data()
