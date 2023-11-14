
import numpy as np
import pandas as pd
import itertools
# import pyreadr
from sklearn import preprocessing as skprep
# import matplotlib.pyplot as plt
from scipy.special import expit, softmax

#%%

def symlog(x, shift=1):
    if x >= 0:
        return np.log(x+shift)-np.log(shift)
    else:
        return -np.log(-x+shift)+np.log(shift)

def make_data(
    filename="data/Summer23_Train_smaller-for-Python.csv", 
    # budget_file="data/Over_budget_S_t3.csv",
    # prob_constraint="Fall_results/BART_preds_near-zero_11-20.csv",
    data_only=True,
    outcome="other-hosps",
    manual_S_size = "medium",
    eligible = "all",
    all_data=False # whether it's for RL or not
):
    ## Read in data
    Train = pd.read_csv(filename)

    ## Subset out the indices for today (vs tomorrow)
    n_counties = Train["fips"].nunique()
    n_years = 11
    n_days = 153

    ## Get county-year IDs:
    # ID = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(S.shape[0]/(n_days-1)))]))
    county_ids = range(0, n_counties)
    # ids_over_years = list(itertools.chain(*itertools.repeat(county_ids, n_years)))

    if all_data == False:
        n_seq_s = range(n_days-1, Train.shape[0], n_days)
        n_seq_s1 = range(0, Train.shape[0], n_days)
        ID = list(itertools.chain(*[itertools.repeat(i, (n_days-1)*n_years) for i in county_ids]))
    else:
        n_seq_s = []
        n_seq_s1 = []
        ID = list(itertools.chain(*[itertools.repeat(i, n_days*n_years) for i in county_ids]))

    A = Train["alert"].drop(n_seq_s)
    if outcome == "all_hosps":
        R = -1*(Train["all_hosps"]/Train["total_count"]).drop(n_seq_s)
    elif outcome == "other_hosps":
        R = -1*(Train["other_hosps"]/Train["total_count"]).drop(n_seq_s)
    else:
        R = -1*(Train["N"]/Train["Pop.65"]).drop(n_seq_s)
    ep_end = Train["dos"].drop(n_seq_s) == 152

    ## Adding new column for overall budget:
    budget = Train[Train["dos"] == n_days]["alert_sum"]
    Budget = list(itertools.chain(*[itertools.repeat(b, n_days) for b in budget]))
    Train["More_alerts"] = Budget - Train["alert_sum"]

    ## Subset out state variables
    States_1 = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "BA_zone", "l.Pop_density", "l.Med.HH.Income",
                        "year", "dos", "holiday", "dow", 
                        "alert_lag1", "alert_lag2", "alerts_2wks", "T_since_alert",
                        "alert_sum", "More_alerts",
                        "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                        "all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate",     
                        "heat_hosp_3dMA_rate", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                        "broadband.usage", "Democrat", "Republican", "pm25"]]
    States = States_1.drop(n_seq_s)
    States_1 = States_1.drop(n_seq_s1)

    ## One-hot encode non-numeric variables
    S_enc = skprep.OneHotEncoder(drop = "first")
    S_enc.fit(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]])
    S_ohe = S_enc.transform(States[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]]).toarray()
    S_names = S_enc.get_feature_names_out(["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"])
    S_OHE = pd.DataFrame(S_ohe, columns=S_names)

    S1_enc = skprep.OneHotEncoder(drop = "first")
    S1_enc.fit(States_1[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]])
    S1_ohe = S1_enc.transform(States_1[["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"]]).toarray()
    S1_names = S1_enc.get_feature_names_out(["BA_zone", "holiday", "dow", "alert_lag1", "alert_lag2"])
    S1_OHE = pd.DataFrame(S1_ohe, columns=S1_names)

    ## Standardize numeric variables
    num_vars = ["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "l.Pop_density", "l.Med.HH.Income", 
                        "year", "dos", "alerts_2wks", "T_since_alert",
                         "alert_sum", "More_alerts",
                         "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                         "all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate",     
                        "heat_hosp_3dMA_rate", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                         "broadband.usage", "Democrat", "Republican", "pm25"]

    s_means = States[num_vars].mean(0)
    s_stds = States[num_vars].std(0)
    S = (States[num_vars] - s_means)/s_stds
    S = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)
    S = S.drop("index", axis=1)

    s_1_means = States_1[num_vars].mean(0)
    s_1_stds = States_1[num_vars].std(0)
    S_1 = (States_1[num_vars] - s_1_means)/s_1_stds
    S_1 = pd.concat([S_1.reset_index(), S1_OHE.reset_index()], axis = 1)
    S_1 = S_1.drop("index", axis=1)

    ## Get subsets:

    if manual_S_size == "medium":
        S["weekend"] = S["dow_Saturday"] + S["dow_Sunday"]
        S = S[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate",
             "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate", 
             "age_65_74_rate", "age_75_84_rate", "dual_rate",
            "Republican", "pm25", "weekend", 'BA_zone_Hot-Dry',
            'BA_zone_Hot-Humid', 'BA_zone_Marine', 'BA_zone_Mixed-Dry',
            'BA_zone_Mixed-Humid', 'BA_zone_Very Cold'
        ]
        ]
        S_1["weekend"] = S_1["dow_Saturday"] + S_1["dow_Sunday"]
        S_1 = S_1[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate",
             "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate", 
             "age_65_74_rate", "age_75_84_rate", "dual_rate",
            "Republican", "pm25", "weekend", 'BA_zone_Hot-Dry',
            'BA_zone_Hot-Humid', 'BA_zone_Marine', 'BA_zone_Mixed-Dry',
            'BA_zone_Mixed-Humid', 'BA_zone_Very Cold'
        ]
        ]
    elif manual_S_size == "small":
        S["weekend"] = S["dow_Saturday"] + S["dow_Sunday"]
        S = S[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate", 
            "weekend"
        ]
        ]
        S_1["weekend"] = S_1["dow_Saturday"] + S_1["dow_Sunday"]
        S_1 = S_1[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", "More_alerts", "all_hosp_mean_rate", 
            "weekend"
        ]
        ]

    ## Get budget
    # over_budget = pd.read_csv(budget_file)
    # over = over_budget["over_budget"] == 1
    over = S["More_alerts"] == 0

    ## Get behavior policy probability constraint:
    # near_zero = pd.read_csv(prob_constraint).drop(n_seq_s)
    # near_zero = pd.read_csv(prob_constraint)
    
    # behav_prob = pd.read_csv("Fall_results/Alerts_model_1-23.csv")
    # near_zero = (np.array(behav_prob)[:,1] < 0.01).astype(int)

    if eligible == "all":
        if data_only == True:
            output = dict(
                S = S, A = A, R = R, S_1 = S_1, 
                ep_end = ep_end, over = over, # near_zero = near_zero, ID = ID
                )
        else:
            ## Get summary stats of all outcomes:
            R_deaths = -1*(Train["N"]/Train["Pop.65"]).drop(n_seq_s)
            R_all_hosps = -1*(Train["all_hosps"]/Train["total_count"]).drop(n_seq_s)
            R_other_hosps = -1*(Train["other_hosps"]/Train["total_count"]).drop(n_seq_s)
            deaths_shift = R_deaths.mean()
            all_hosps_shift = R_all_hosps.mean()
            other_hosps_shift = R_other_hosps.mean()
            deaths_scale = np.max(np.abs(R_deaths))
            all_hosps_scale = np.max(np.abs(R_all_hosps))
            other_hosps_scale = np.max(np.abs(R_other_hosps))
            ## Return:
            output = dict(
                S = S, A = A, R = R, S_1 = S_1, 
                ep_end = ep_end, over = over, # near_zero = near_zero, ID = ID,
                Budget = Budget, n_seq_s = n_seq_s,
                s_means = s_means, s_stds = s_stds,
                R_deaths = R_deaths, R_all_hosps = R_all_hosps, R_other_hosps = R_other_hosps,
                deaths_shift = deaths_shift, deaths_scale = deaths_scale, 
                all_hosps_shift = all_hosps_shift, all_hosps_scale = all_hosps_scale,
                other_hosps_shift = other_hosps_shift, other_hosps_scale = other_hosps_scale
                )

    else:
        elig = pd.read_csv("data/Pct_90_eligible.csv") # could include other options too
        Elig = elig.index[elig["Pct_90_eligible"]]
        if data_only == True:
            output = dict(
                S = S.iloc[Elig], A = A[Elig], R = R[Elig], S_1 = S_1.iloc[Elig], 
                ep_end = ep_end[Elig], over = over[Elig] #, near_zero = near_zero, ID = ID
                )
        else:
            ## Get summary stats of all outcomes:
            R_deaths = -1*(Train["N"]/Train["Pop.65"]).drop(n_seq_s)
            R_all_hosps = -1*(Train["all_hosps"]/Train["total_count"]).drop(n_seq_s)
            R_other_hosps = -1*(Train["other_hosps"]/Train["total_count"]).drop(n_seq_s)
            deaths_shift = R_deaths.mean()
            all_hosps_shift = R_all_hosps.mean()
            other_hosps_shift = R_other_hosps.mean()
            deaths_scale = np.max(np.abs(R_deaths))
            all_hosps_scale = np.max(np.abs(R_all_hosps))
            other_hosps_scale = np.max(np.abs(R_other_hosps))
            ## Return:
            output = dict(
                S = S.iloc[Elig], A = A[Elig], R = R[Elig], S_1 = S_1.iloc[Elig],  
                ep_end = ep_end[Elig], over = over[Elig], # near_zero = near_zero, ID = ID,
                Budget = Budget, n_seq_s = n_seq_s[Elig],
                s_means = s_means, s_stds = s_stds,
                R_deaths = R_deaths[Elig], R_all_hosps = R_all_hosps[Elig], R_other_hosps = R_other_hosps[Elig],
                deaths_shift = deaths_shift, deaths_scale = deaths_scale, 
                all_hosps_shift = all_hosps_shift, all_hosps_scale = all_hosps_scale,
                other_hosps_shift = other_hosps_shift, other_hosps_scale = other_hosps_scale
                )
    return output

if __name__ == "__main__":
    D = make_data()