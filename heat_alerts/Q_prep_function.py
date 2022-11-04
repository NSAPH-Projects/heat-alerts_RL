
# import numpy as np
import pandas as pd
import itertools
# import pyreadr
from sklearn import preprocessing as skprep
# import matplotlib.pyplot as plt
from scipy.special import expit, softmax

#%%


def make_data(
    filename="data/Train_smaller-for-Python.csv", 
    # budget_file="data/Over_budget_S_t3.csv",
    prob_constraint="Fall_results/BART_preds_near-zero.csv",
    data_only=True,
    outcome="deaths"
):
    ## Read in data
    Train = pd.read_csv(filename)

    ## Subset out the indices for today (vs tomorrow)
    n_counties = Train["GEOID"].nunique()
    n_years = 11
    n_days = 153

    n_seq_s = range(n_days-1, Train.shape[0], n_days)

    A = Train["alert"].drop(n_seq_s)
    if outcome == "hosps":
        R = -1*(Train["all_hosps"]/Train["total_count"]).drop(n_seq_s)
    else:
        R = -1*(Train["N"]*10000/Train["Pop.65"]).drop(n_seq_s)
    ep_end = Train["dos"].drop(n_seq_s) == 152

    ## Adding new column for overall budget:
    budget = Train[Train["dos"] == n_days]["alert_sum"]
    Budget = list(itertools.chain(*[itertools.repeat(b, n_days) for b in budget]))
    Train["More_alerts"] = Budget - Train["alert_sum"]

    ## Subset out state variables
    States_1 = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                        "BA_zone", "Pop_density", "Med.HH.Income",
                        "year", "dos", "holiday", "dow", "alert_sum", "More_alerts"]]
    States = States_1.drop(n_seq_s)
    States_1 = States_1.drop(range(0, Train.shape[0], n_days))

    ## One-hot encode non-numeric variables
    S_enc = skprep.OneHotEncoder(drop = "first")
    S_enc.fit(States[["BA_zone", "holiday", "dow"]])
    S_ohe = S_enc.transform(States[["BA_zone", "holiday", "dow"]]).toarray()
    S_names = S_enc.get_feature_names_out(["BA_zone", "holiday", "dow"])
    S_OHE = pd.DataFrame(S_ohe, columns=S_names)

    S1_enc = skprep.OneHotEncoder(drop = "first")
    S1_enc.fit(States_1[["BA_zone", "holiday", "dow"]])
    S1_ohe = S1_enc.transform(States_1[["BA_zone", "holiday", "dow"]]).toarray()
    S1_names = S1_enc.get_feature_names_out(["BA_zone", "holiday", "dow"])
    S1_OHE = pd.DataFrame(S1_ohe, columns=S1_names)

    ## Standardize numeric variables
    num_vars = ["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                        "Pop_density", "Med.HH.Income",
                        "year", "dos", "alert_sum", "More_alerts"]

    s_means = States[num_vars].mean(0)
    s_stds = States[num_vars].std(0)
    S = (States[num_vars] - s_means)/s_stds
    S = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)

    s_1_means = States_1[num_vars].mean(0)
    s_1_stds = States_1[num_vars].std(0)
    S_1 = (States_1[num_vars] - s_1_means)/s_1_stds
    S_1 = pd.concat([S_1.reset_index(), S1_OHE.reset_index()], axis = 1)

    ## Get budget
    # over_budget = pd.read_csv(budget_file)
    # over = over_budget["over_budget"] == 1
    over = S["More_alerts"] == 0

    ## Get behavior policy probability constraint:
    near_zero = pd.read_csv(prob_constraint).drop(n_seq_s)

    if data_only == True:
        output = dict(
            S = S, A = A, R = R, S_1 = S_1, 
            ep_end = ep_end, over = over, near_zero = near_zero)
    else:
        output = dict(
            S = S, A = A, R = R, S_1 = S_1, 
            ep_end = ep_end, over = over, near_zero = near_zero,
            Budget = Budget, n_seq_s = n_seq_s,
            s_means = s_means, s_stds = s_stds)

    return output

if __name__ == "__main__":
    D = make_data()