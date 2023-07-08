# %%
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from patsy import dmatrix
from sklearn.preprocessing import StandardScaler

# %%
data = pd.read_csv("../data/Train_smaller-for-Python.csv")
data.columns = [col.replace(".", "_") for col in data.columns]
data.keys()

# %%
all_cols = [
    "fips",
    "Date",
    "alert",
    "alert_lag1",
    "alert_lag2",
    "other_hosps",
    "HI_mean",
    "HI_lag1",
    "HI_lag2",
    "holiday",
    "dos",
    "dow",
    "Population",
    "Pop_density",
    "Med_HH_Income",
    "broadband_usage",
    "Democrat",
    "pm25"
]
# drop rows with missing values
data = data[all_cols].dropna()
data.head()

# %%
data.shape

# %% space invariate feats
space_keys = [
    "fips",
    "Pop_density",
    "Med_HH_Income",
    "broadband_usage",
    "Democrat",
    "Population",
]
W = (
    data[space_keys]
    .drop_duplicates()
    .set_index("fips")
    .groupby("fips")
    .mean()
    .assign(Lop_Pop_density=lambda x: np.log(x["Pop_density"]))
    .assign(Log_Med_HH_Income=lambda x: np.log(x["Med_HH_Income"]))
    .drop(columns=["Pop_density", "Med_HH_Income"])
)

# standardize W
wscaler = StandardScaler()
wscaler_cols = ["broadband_usage", "Democrat", "Lop_Pop_density", "Log_Med_HH_Income"]
W[wscaler_cols] = wscaler.fit_transform(W[wscaler_cols])
W["intercept"] = 1.0

P = W[["Population"]]
W = W.drop(columns=["Population"])
fips = W.index.values
fips2idx = {fips: idx for idx, fips in enumerate(data["fips"].unique())}

W.head()

# %% make splines of time of summer
# use patsy to make bspline basis for dos, degree 3, df 5
dos = data["dos"] - 1

# make day of week to be 0 on monday
dow2num = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
dow = data["dow"].map(dow2num)

Bdos = dmatrix("bs(dos, df=5, degree=3) - 1", {"dos": dos}, return_type="dataframe")
Bdos.columns = [f"dos_{i}" for i in range(Bdos.shape[1])]
Bdow = dmatrix("bs(dow, df=5, degree=3) - 1", {"dow": dow}, return_type="dataframe")
Bdow.columns = [f"dow_{i}" for i in range(Bdow.shape[1])]

# %% time varying features
time_keys = [
    "fips",
    "Date",
    "alert_lag1",
    "alert_lag2",
    "HI_mean",
    "HI_lag1",
    "HI_lag2",
    "holiday",
]
X = data[time_keys].set_index(["fips", "Date"])

# paste splines onto X
Bdos.index = X.index
Bdow.index = X.index
X = pd.concat([X, Bdos, Bdow], axis=1)

# standardize X if not binary cols or splines
xscaler = StandardScaler()
xscaler_cols = ["alert_lag1", "alert_lag2", "HI_mean", "HI_lag1", "HI_lag2"]
X[xscaler_cols] = xscaler.fit_transform(X[xscaler_cols])
X["intercept"] = 1.0
X.head()


# %% get alerts and outcome data
A = data[["fips", "Date", "alert"]].set_index(["fips", "Date"])
Y = data[["fips", "Date", "other_hosps"]].set_index(["fips", "Date"])

# plot histograms of alerts and hospos in (1, 2) pane
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(A.values, bins=20)
ax[0].set_title("Alerts")
ax[1].hist(Y.values, bins=20)
ax[1].set_title("Other Hosps")


# %% naive estimate
# without adjutment the mean on alert days is higher
Y.loc[A.values == 1].mean() - Y.loc[A.values == 0].mean()

# %% location indicator and population
sind = data.fips.map(fips2idx).values
sind = pd.DataFrame({"sind": sind}, Y.index)
P = data[["Population"]] / 1000 # better to work on thousands

# %% save time varying features, treatment, outcomes
os.makedirs("data/processed", exist_ok=True)

# %% save states (X)
X.to_parquet("data/processed/states.parquet")

# %% save outcomes (Y)
Y.to_parquet("data/processed/outcomes.parquet")

# %% save treatment/action (A)
A.to_parquet("data/processed/actions.parquet")

# %% save time-varying features (W)
# loc indicator (sind)
# Population (P)
W.to_parquet("data/processed/spatial_feats.parquet")
sind.to_parquet("data/processed/location_indicator.parquet")
P.to_parquet("data/processed/population.parquet")

# %% save scaler info as json
scaler_info = {
    "xscaler": {
        "mean": xscaler.mean_.tolist(),
        "scale": xscaler.scale_.tolist(),
        "columns": xscaler_cols,
    },
    "wscaler": {
        "mean": wscaler.mean_.tolist(),
        "scale": wscaler.scale_.tolist(),
        "columns": wscaler_cols,
    },
}
with open("data/processed/scalers.json", "w") as f:
    json.dump(scaler_info, f)

# %% save fips2idx as json, make sure keys and values are int
with open("data/processed/fips2idx.json", "w") as f:
    json.dump({int(k): int(v) for k, v in fips2idx.items()}, f)

# %%
