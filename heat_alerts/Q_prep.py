## Prepare for Q-learning / DQN using pytorch

import os
import time
import numpy as np
import pandas as pd
import itertools
# import pyreadr
from sklearn import preprocessing as skprep
import matplotlib.pyplot as plt
from scipy.special import expit, softmax

import torch 
from torch import nn # creating modules
from torch.nn import functional as F # losses, etc.
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from tqdm import tqdm
from copy import deepcopy

os.chdir("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")


## Read in the data
Train = pd.read_csv("data/Train_smaller-for-Python.csv") # had to use write_csv from readr package to ensure this was all in utf-8



## Subset out the indices for today (vs tomorrow)
n_counties = Train["GEOID"].nunique()
n_years = 11
n_days = 153

n_seq_s = range(n_days-1, Train.shape[0], n_days)

A = Train["alert"].drop(n_seq_s)
R = -1*(Train["N"]*10000/Train["Pop.65"]).drop(n_seq_s)
ep_end = Train["dos"].drop(n_seq_s) == 152
gamma = 0.99

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
over_budget = pd.read_csv("data/Over_budget_S_t3.csv")
over = over_budget["over_budget"] == 1

