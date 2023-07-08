# %%
import numpy as np
import geopandas as gpd
import requests
import zipfile
import os
import json
import networkx as nx
from cmdstanpy import CmdStanModel


# %%
# download the shapefile of counties form tiger for 2010
url = "https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/tl_2010_us_county10.zip"
if not os.path.exists("bayesian_model/data/tl_2010_us_county10"):
    os.makedirs("bayesian_model/data/tl_2010_us_county10")
    # download .zip, unzip and delte
    r = requests.get(url)
    with open("tl_2010_us_county10.zip", "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile("tl_2010_us_county10.zip", "r") as zip_ref:
        zip_ref.extractall("bayesian_model/data/tl_2010_us_county10")
    os.remove("bayesian_model/data/tl_2010_us_county10.zip")

# %%
# read in the shapefile
file = "bayesian_model/data/tl_2010_us_county10/tl_2010_us_county10.shp"
counties = gpd.read_file(file)
# remove counties that are not in the mainland
counties = counties[~counties.STATEFP10.isin(["02", "15", "72", "78"])]


# %%
# obtain edge list
edge_list = set()
for i in range(len(counties)):
    mask = counties.iloc[i + 1 :].geometry.touches(counties.iloc[i].geometry)
    if mask.any():
        which = np.where(mask)[0]
        for j in which:
            src, tgt = (i, j) if i < j else (j, i)
            edge_list.add((src, tgt))

edge_list = list(edge_list)
edge_list = np.array(edge_list)
edge_list[:5]

# %% compute number of connected components
graph = nx.Graph()
graph.add_edges_from(edge_list)
nx.number_connected_components(graph)

# %% compute average degree
degrees = [graph.degree(node) for node in graph.nodes]
np.mean(degrees)

# %% save edge list
np.savetxt("bayesian_model/data/edge_list.csv", edge_list, delimiter=",", fmt="%d")

# # %%
model = CmdStanModel(stan_file="bayesian_model/model_not_spatial.stan")
print(model)

# %% simulate a panel design
S = len(counties)
T = 100
sind = np.repeat(np.arange(S), T)
N = S * T
node1 = edge_list[:, 0]
node2 = edge_list[:, 1]

# simulate the time-vary covariates as panel using an AR(1)
X = np.zeros((S, T, 3))
for i in range(S):
    X[i, 0] = 0.1 * np.random.normal(0, 1, size=(3,))
    for j in range(1, T):
        X[i, j] = 0.1 * np.random.normal(0.5 * X[i, j - 1], 1, size=(3,))
X = X.reshape(-1, 3)

# simulate the space-varying covariates
W = 0.1 * np.random.normal(0, 1, (S, 2))

# add a column of ones to X and W
X = np.hstack((np.ones((N, 1)), X))
W = np.hstack((np.ones((S, 1)), W))

# simulate population
P = np.random.poisson(10, size=(S,))

data = {
    "N": N,
    "S": S,
    "DX": 4,
    "DW": 3,
    "M": edge_list.shape[0],
    "node1": node1 + 1,
    "node2": node2 + 1,
    "A": np.random.binomial(1, 0.1, size=(N,)),
    "y": np.random.poisson(10, size=(N,)),  # not used in fixed sampling
    "P": P,
    "X": X,
    "W": W,
    "sind": sind + 1,
}

# %% simulate data
init = {
    "omega_beta": 0.1 * np.ones((4,)),
    "omega_gamma": 0.1 * np.ones((4,)),
    # "xi": 1.0
}
sample = model.sample(
    data=data,
    output_dir="bayesian_model/simulated_data",
    iter_sampling=1,
    chains=1,
    seed=1234,
    fixed_param=True,
    show_console=True,
    inits=init,
)

# %%
# print variable names in sample
vars = list(sample.metadata.stan_vars_cols.keys())

# add all vars to data
for var in vars:
    data[var] = sample.stan_variable(var)[0]

# %% simulate outcome
# y ~ neg_bin2(mu, xi)
# importantly, negbin needs to be parameterized by mean and overdispersion
mu = data["mu"]
# xi = data["xi"]
# phi = 1 / xi  # overdispersion parameter, reciprocal of xi
# n = phi
# p = n / (n + mu)
# y = np.random.negative_binomial(n, p)
data["y"] = np.random.poisson(mu)

# %%
# serialize data as json.
# make sure that ll numpy arrays are mapped as lists

# convert numpy arrays to lists
for key, value in data.items():
    if isinstance(value, np.ndarray):
        data[key] = value.tolist()

# save data
# save in pretty longer format
with open("bayesian_model/simulated_data/sim.json", "w") as f:
    json.dump(data, f)

# %%
