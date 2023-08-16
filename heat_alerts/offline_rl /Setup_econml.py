
import numpy as np
import pandas as pd
from sklearn import preprocessing as skprep

def prep_data(
    filename="data/Train_smaller-for-Python.csv", 
    eligible = "90pct",
    manual_S_size = "small"
):
    ## Read in data:
    Train = pd.read_csv(filename)

    actions = Train["alert"]

    rewards = Train["other_hosps"]/Train["total_count"]
    rewards = (rewards - rewards.mean())/rewards.std()
    rewards = rewards.to_numpy()

    ## Prepare observations:
    States = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "BA_zone", "l.Pop_density", "l.Med.HH.Income",
                        "year", "dos", "holiday", "dow", 
                        "alert_lag1", "alert_lag2", "alerts_2wks", "T_since_alert",
                        "alert_sum", # "More_alerts",
                        "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                        "broadband.usage", "Democrat", "Republican", "pm25"]]

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
                         "alert_sum", # "More_alerts",
                         "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                         "broadband.usage", "Democrat", "Republican", "pm25"]

    s_means = States[num_vars].mean(0)
    s_stds = States[num_vars].std(0)
    S = (States[num_vars] - s_means)/s_stds
    
    observations = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)
    observations = observations.drop("index", axis=1)

    if manual_S_size == "medium":
        observations["weekend"] = observations["dow_Saturday"] + observations["dow_Sunday"]
        observations = observations[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", # "More_alerts", 
            "all_hosp_mean_rate", "Republican", "pm25", "weekend", 'BA_zone_Hot-Dry',
            'BA_zone_Hot-Humid', 'BA_zone_Marine', 'BA_zone_Mixed-Dry',
            'BA_zone_Mixed-Humid', 'BA_zone_Very Cold'
        ]
        ]
    elif manual_S_size == "small":
        observations["weekend"] = observations["dow_Saturday"] + observations["dow_Sunday"]
        observations = observations[[
            "quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "T_since_alert", "alert_sum", # "More_alerts", 
            "all_hosp_mean_rate", "weekend"
        ]
        ]
    elif manual_S_size == "tiny":
        observations = observations[[
            "quant_HI_county", #"More_alerts", 
            "dos", "T_since_alert", "all_hosp_mean_rate"
        ]
        ]

    if eligible == "all":
        X = observations.to_numpy()
        T = actions.to_numpy()
        Y = rewards
    else:
        elig = pd.read_csv("data/Pct_90_eligible.csv") # could include other options too
        Elig = elig.index[elig["Pct_90_eligible"]]
        X = observations.iloc[Elig].to_numpy()
        T = actions[Elig].to_numpy()
        Y = rewards[Elig]
    
    return [X, T, Y]

