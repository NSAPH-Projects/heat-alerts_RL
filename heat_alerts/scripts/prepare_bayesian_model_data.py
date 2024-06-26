# %%
import json
import os

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %%
data_raw = pd.read_csv("data/Summer23_Train_smaller-for-Python.csv")
data_raw.columns = [col.replace(".", "_") for col in data_raw.columns]
data_raw.keys()

# %%
all_cols = [
    "fips",
    "Date",
    "year",
    "quant_HI_county",
    "quant_HI_3d_county",
    "HI_mean",
    'HImaxF_PopW', # to provide "forecasts" to the RL
    "alert",
    "alert_lag1",  # if doing online or hybrid RL
    "alerts_2wks",
    "alert_sum",
    "other_hosps",
    "pm25",
    "holiday",
    "dos",
    "dow",
    "Population",
    "total_count",  # Medicare enrollees
    "Pop_density",
    "Med_HH_Income",
    "broadband_usage",
    "Democrat",
    "BA_zone",
    "state",
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
    "pm25",
]

# %% also add BA_zone, make one-hot encoding (sklearn) using the Hot-Humid as control
enc = OneHotEncoder()
enc.fit(data[["BA_zone"]])
BA_zone = enc.transform(data[["BA_zone"]]).toarray()
categories = enc.categories_[0]
BA_zone = pd.DataFrame(BA_zone, columns=categories, index=data.index)
BA_zone = BA_zone.drop(columns=["Hot-Humid"], axis=1)

W = (
    data[space_keys]
    .merge(BA_zone, left_index=True, right_index=True)
    .drop_duplicates()
    .set_index("fips")
    .groupby("fips")
    .mean()
    .assign(Log_Pop_density=lambda x: np.log(x["Pop_density"]))
    .assign(Log_Med_HH_Income=lambda x: np.log(x["Med_HH_Income"]))
    .drop(columns=["Pop_density", "Med_HH_Income"])
)


# %% standardize W
wscaler = StandardScaler()
wscaler_cols = [
    "broadband_usage",
    "Democrat",
    "Log_Pop_density",
    "Log_Med_HH_Income",
    "pm25",
]
W[wscaler_cols] = wscaler.fit_transform(W[wscaler_cols])
W["intercept"] = 1.0

fips = data["fips"].unique()
W = W.loc[fips]  # needed to ensure they have the same ordering!

fips2idx = {fips: idx for idx, fips in enumerate(fips)}
fips2state = {fips: state for fips, state in zip(data["fips"], data["state"])}
# W.head()

# %% make splines of time of summer, use patsy to make bspline basis for dos, degree 3, df 5
dos = data["dos"] - 1
M = dos.max()

# note: we need M + 1 below  because how bs works
Bdos = dmatrix(
    f"bs(dos, df=3, degree=3, lower_bound=0, upper_bound={M + 1}) - 1",
    {"dos": dos},
    return_type="dataframe",
)
Bdos.columns = [f"dos_{i}" for i in range(Bdos.shape[1])]


# %% time varying features
time_keys = [
    "fips",
    "Date",
    "quant_HI_county",
    "quant_HI_3d_county",
    "HI_mean",
    "alert_lag1",
    "alerts_2wks",
    "holiday",
]

X = (
    data[time_keys]
    .set_index(["fips", "Date"])
    .assign(weekend=data.dow.isin(["Saturday", "Sunday"]).values.astype(int))
    # insert the square of quant_HI after quant_HI
    # make sure it is the third column
    .assign(quant_HI_county_pow2=lambda x: (x["quant_HI_county"] - 0.5) ** 2)
    .assign(quant_HI_3d_county_pow2=lambda x: (x["quant_HI_3d_county"] - 0.5) ** 2)
    .assign(
        heat_qi_above_25=lambda x: (x["quant_HI_county"] - 0.25)
        * (x["quant_HI_county"] > 0.25)
    )
    .assign(
        heat_qi_above_75=lambda x: (x["quant_HI_county"] - 0.75)
        * (x["quant_HI_county"] > 0.75)
    )
    .assign(excess_heat=lambda x: x["quant_HI_county"] - x["quant_HI_3d_county"])
    .assign(intercept=1.0)
)
reorder = [
    "intercept",
    "quant_HI_county",
    "quant_HI_county_pow2",
    "quant_HI_3d_county",
    "quant_HI_3d_county_pow2",
    "heat_qi_above_25",
    "heat_qi_above_75",
    "excess_heat",
    "HI_mean",
    "weekend",
    "alert_lag1",
    "alerts_2wks",
]  # "year",
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

alert_sum = data[["fips", "Date", "alert_sum"]]
n_days = 153
end_seq = range(n_days - 1, len(alert_sum), n_days)
budget = alert_sum["alert_sum"][end_seq]
Budget = pd.DataFrame(np.repeat(budget.values, n_days, axis=0))
Budget["fips"] = data["fips"]
Budget["Date"] = data["Date"]
Budget = Budget.set_index(["fips", "Date"])

abs_HI = data[["fips", "Date", 'HImaxF_PopW']].set_index(["fips", "Date"])


# %% naive estimate
# without adjutment the mean on alert days is higher
Y.loc[A.values == 1].mean() - Y.loc[A.values == 0].mean()

# %% location indicator and population
sind = data.fips.map(fips2idx).values
sind = pd.DataFrame({"sind": sind}, Y.index)
P = data[["Population"]] / 1000  # better to work on thousands
Enrolled = data[["total_count"]]
State = data[["state"]]

# %% offset = location means
df = pd.DataFrame(
    {
        "other_hosps": Y.values[:, 0],
        "sind": sind.values[:, 0],
        "year": year.values[:, 0],
    }
)
tmp = (
    df.groupby(["sind", "year"])
    .mean()
    .reset_index()
    .rename(columns={"other_hosps": "mean_other_hosps"})
)
offset = df.merge(tmp, on=["sind", "year"], how="left")[["mean_other_hosps"]]


# %% save time varying features, treatment, outcomes
os.makedirs("data/processed", exist_ok=True)

# %% save states (X)
X.to_parquet("data/processed/states.parquet")

# %% save outcomes (Y)
Y.to_parquet("data/processed/outcomes.parquet")

# %% save treatment/action (A)
A.to_parquet("data/processed/actions.parquet")

# %% save time-varying features (W)
abs_HI.to_parquet("data/processed/abs_HI.parquet")
# loc indicator (sind)
# Population (P)
W.to_parquet("data/processed/spatial_feats.parquet")
sind.to_parquet("data/processed/location_indicator.parquet")
P.to_parquet("data/processed/population.parquet")
Enrolled.to_parquet("data/processed/Medicare_denominator.parquet")
offset.to_parquet("data/processed/offset.parquet")
State.to_parquet("data/processed/state.parquet")
year.to_parquet("data/processed/year.parquet")
Budget.to_parquet("data/processed/budget.parquet")

# %% save scaler info as json
scaler_info = {
    "xscaler": {
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

# %% save fips2idx, fips2state as json, make sure keys and values are int
with open("data/processed/fips2idx.json", "w") as f:
    json.dump({int(k): int(v) for k, v in fips2idx.items()}, f)

with open("data/processed/fips2state.json", "w") as f:
    json.dump({int(k): v for k, v in fips2state.items()}, f)

# %% save splines
t_dos = np.arange(0, M)
Btdos = dmatrix(
    f"bs(t_dos, df=3, degree=3, lower_bound=0, upper_bound={M + 1}) - 1",
    return_type="dataframe",
)
Btdos.columns = [f"dos_{i}" for i in range(Btdos.shape[1])]

Btdos.to_parquet("data/processed/Btdos.parquet")

# %%

#### Calculate future quantiles (for RL):
import itertools

def QQ(x, q):
    n = len(x)
    for i in range(0,n-1):
        x[i+n] = np.quantile(x[(i+1):n], q)
    x[(n-1)+n] = x[(n-2)+n] # just repeating because we're at the end of the episode
    return(x)

with open("data/processed/fips2idx.json", "r") as io:
    fips2idx = json.load(io)

idx2fips = {v: k for k, v in fips2idx.items()}
sind=sind.sind
n_counties = len(sind.unique())
n_years = len(year.year.unique())
dos_index = list(itertools.chain(*[np.arange(0,n_days) for i in np.arange(0,n_years*n_counties)]))

qhi = data[["fips", "Date", "quant_HI_county"]].set_index(["fips", "Date"])
qhi = qhi.assign(fips=sind.map(idx2fips).astype(int)).assign(dos_index=dos_index).assign(year=year)
f = "quant_HI_county"
D = qhi[[f, "fips", "year", "dos_index"]]
future = D.pivot(index=["fips", "year"], columns="dos_index", values="quant_HI_county")

for q in [5, 6, 7, 8, 9, 10]:
    quant = future.apply(QQ, axis=1, args=(q*0.1,)).iloc[:, n_days:]
    quant.columns = future.columns
    Quant = data[["fips", "Date"]]
    Quant["q" + str(q) + "0"] = quant.to_numpy().flatten()
    Quant = Quant.set_index(["fips", "Date"])
    Quant.to_parquet("data/processed/future_q" + str(q) + "0.parquet")
    print(q)

#### Calculate averages evenly split by remaining time (for RL):
def avg_4ths(x):
    n = len(x)
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for i in range(0,n-5):
        l = n - 1 - i
        q = int(np.floor(l/4))
        j = i+1
        y1.append(np.mean(x[j:(np.min([j+q, n]))]))
        j += q
        y2.append(np.mean(x[j:np.min([j+q, n])]))
        j += q
        y3.append(np.mean(x[j:np.min([j+q, n])]))
        j += q
        y4.append(np.mean(x[j:n]))
    for k in np.arange(0,5):
        y1.append(y1[n-6]) # just repeating because we're at the end of the episode
        y2.append(y2[n-6])
        y3.append(y3[n-6])
        y4.append(y4[n-6])
    return(y1, y2, y3, y4)

T1 = []
T2 = []
T3 = []
T4 = []
for i in np.arange(future.shape[0]):
    t1, t2, t3, t4 = avg_4ths(future.iloc[i])
    T1.append(t1)
    T2.append(t2)
    T3.append(t3)
    T4.append(t4)
    print(i)

Avgs = data[["fips", "Date"]]
Avgs["T4_1"] = np.array(T1).flatten()
Avgs["T4_2"] = np.array(T2).flatten()
Avgs["T4_3"] = np.array(T3).flatten()
Avgs["T4_4"] = np.array(T4).flatten()
Avgs = Avgs.set_index(["fips", "Date"])
Avgs.to_parquet("data/processed/future_quarters.parquet")
