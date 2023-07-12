# %%
import os
import json

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import StandardScaler


# %%
data_raw = pd.read_csv("../data/Summer23_Train_smaller-for-Python.csv")
data_raw.columns = [col.replace(".", "_") for col in data_raw.columns]
data_raw.keys()

# %%
unique_fips = data_raw["fips"].unique()
fips_ix = np.random.choice(len(unique_fips), 16, replace=False)
subfips = unique_fips[fips_ix]

# %%

# make a plot panel of 16 counties with a scatter plot
# each panel create a scatter plot with a fitted loess curve 
# of hospitalizations vs. heat index (HI)



# import seaborn as sns
# minq = 0.0
# maxq = 1.0
# subdata = data_raw[data_raw["fips"].isin(subfips) & (data_raw["quant_HI"] > minq) & (data_raw["quant_HI"] < maxq)]

# hosps_var = "other_hosps"
# for hi_var in ("HImaxF_PopW", "quant_HI"):
#     sns.lmplot(
#         x=hi_var,
#         y=hosps_var,
#         col="fips",
#         col_wrap=4,
#         data=subdata,
#         order=2,
#         # lowess=True,
#         # no ci
#         # ci=None,
#         # scatter_kws={"alpha": 0.1},
#         line_kws={"color": "red"},
#         height=1.5,
#         aspect=1.5,
#         sharey=False,
#         scatter=False,
#     )
    # plt.savefig(f"preprocessing_{hi_var}_vs_{hosps_var}.png", dpi=300)
    


# %%
all_cols = [
    "fips",
    "Date",
    "year",
    "quant_HI_county",
    "quant_HI_3d_county",
    "alert",
    # "alert_lag1", # if doing online or hybrid RL
    "alerts_2wks",
    "other_hosps",
    "pm25",
    "holiday",
    "dos",
    "dow",
    "Population",
    "Pop_density",
    "Med_HH_Income",
    "broadband_usage",
    "Democrat",
    "BA_zone"
]

# drop rows with missing values
data = data_raw[all_cols].dropna()
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
    "pm25",
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

# %% also add BA_zone, make one-hot encoding (sklearn) using the Hot-Humid as control
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(data[["BA_zone"]])
BA_zone = enc.transform(data[["BA_zone"]]).toarray()
categories = enc.categories_[0]
BA_zone = pd.DataFrame(BA_zone, columns=categories)
BA_zone = BA_zone.drop(columns=["Hot-Humid"], axis=1)
W = pd.concat([W, BA_zone], axis=1)


# %% standardize W
wscaler = StandardScaler()
wscaler_cols = ["broadband_usage", "Democrat", "Lop_Pop_density", "Log_Med_HH_Income", "pm25"]
W[wscaler_cols] = wscaler.fit_transform(W[wscaler_cols])
W["intercept"] = 1.0

P = W[["Population"]]
W = W.drop(columns=["Population"])
fips = W.index.values
fips2idx = {fips: idx for idx, fips in enumerate(data["fips"].unique())}

W.head()

# %% make splines of time of summer, use patsy to make bspline basis for dos, degree 3, df 5
dos = data["dos"] - 1
M = dos.max()

Bdos = dmatrix(f"bs(dos, df=5, degree=3, lower_bound=0, upper_bound={M}) - 1", {"dos": dos}, return_type="dataframe")
Bdos.columns = [f"dos_{i}" for i in range(Bdos.shape[1])]


# %% time varying features
time_keys = [
    "fips",
    "Date",
    "year",
    "quant_HI_county",
    "quant_HI_3d_county",
    "alerts_2wks",
    "holiday",
]
X = (
    data[time_keys].set_index(["fips", "Date"])
    .assign(weekend=data.dow.isin(["Saturday", "Sunday"]).values.astype(int))
    # insert the square of quant_HI after quant_HI
    # make sure it is the third column
    .assign(quant_HI_county_pow2=lambda x: (x["quant_HI_county"]  - 0.5)** 2)
    .assign(quant_HI_3d_county_pow2=lambda x: (x["quant_HI_3d_county"]  - 0.5)** 2)
    .assign(intercept=1.0)
)
reorder = ["intercept", "quant_HI_county", "quant_HI_county_pow2", "quant_HI_3d_county", "quant_HI_3d_county_pow2",
           "year", "weekend", "alerts_2wks"]
X = X[reorder]

# paste splines onto X
Bdos.index = X.index
X = pd.concat([X, Bdos], axis=1)

# standardize X if not binary cols or splines
# xscaler = StandardScaler()
# xscaler_cols = ["HImaxF_PopW", "HI_3days", "alerts_2wks"]
# X[xscaler_cols] = xscaler.fit_transform(X[xscaler_cols])

X.head()


# %% get alerts and outcome data
A = data[["fips", "Date", "alert"]].set_index(["fips", "Date"])
Y = data[["fips", "Date", "other_hosps"]].set_index(["fips", "Date"])
year = data[["fips", "Date", "year"]].set_index(["fips", "Date"])

# plot histograms of alerts and hospos in (1, 2) pane
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].hist(A.values, bins=20)
# ax[0].set_title("Alerts")
# ax[1].hist(Y.values, bins=20)
# ax[1].set_title("Other Hosps")


# %% naive estimate
# without adjutment the mean on alert days is higher
Y.loc[A.values == 1].mean() - Y.loc[A.values == 0].mean()

# %% location indicator and population
sind = data.fips.map(fips2idx).values
sind = pd.DataFrame({"sind": sind}, Y.index)
P = data[["Population"]] / 1000 # better to work on thousands

# %% offset = location means
df = pd.DataFrame({"other_hosps": Y.values[:, 0], "sind": sind.values[:, 0], "year": year.values[:, 0]})
tmp = (
    df.groupby(["sind","year"])
    .mean()
    .reset_index()
    .rename(columns={"other_hosps": "mean_other_hosps"})
)
offset = df.merge(tmp, on=["sind","year"], how="left")[["mean_other_hosps"]]


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
offset.to_parquet("data/processed/offset.parquet")

# %% save scaler info as json
scaler_info = {
    "xscaler": {
        # "mean": xscaler.mean_.tolist(),
        # "scale": xscaler.scale_.tolist(),
        # "columns": xscaler_cols,
        "mean": [],
        "scale": [],
        "columns": [],
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


# %% save splines
t_dos = np.arange(0, M + 1)
# t_dow = np.arange(0, 7)
Btdos = dmatrix(f"bs(t_dos, df=5, degree=3, lower_bound=0, upper_bound={M}) - 1", return_type="dataframe")
# Btdow = dmatrix("bs(t_dow, df=5, degree=3, lower_bound=0, upper_bound=6) - 1", return_type="dataframe")
Btdos.columns = [f"dos_{i}" for i in range(Btdos.shape[1])]
# Btdow.columns = [f"dow_{i}" for i in range(Btdow.shape[1])]

#
Btdos.to_parquet("data/processed/Btdos.parquet")

# %%
