
import gym
from gym import spaces

import torch
from pyro.infer import Predictive, predictive

import pandas as pd
import numpy as np
import pyro
# from bayesian_model.pyro_heat_alert import HeatAlertDataModule, HeatAlertModel
from pyro_heat_alert import HeatAlertDataModule, HeatAlertModel

# Read in data:
n_days = 153
n_years = 10
dm = HeatAlertDataModule(
        dir="data/processed", # dir="bayesian_model/data/processed",
        batch_size=n_days*n_years,
        num_workers=4,
        for_gym=True
    )
data = dm.gym_dataset

# X = pd.read_parquet(f"{dir}/states.parquet").drop(columns="intercept")
# A = pd.read_parquet(f"{dir}/actions.parquet")
# Y = pd.read_parquet(f"{dir}/outcomes.parquet")
# W = pd.read_parquet(f"{dir}/spatial_feats.parquet")
# sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
# offset = pd.read_parquet(f"{dir}/offset.parquet")
# dos = pd.read_parquet(f"{dir}/Btdos.parquet")
# year = pd.read_parquet(f"{dir}/year.parquet")
# budget = pd.read_parquet(f"{dir}/budget.parquet")

hosps = data[0]
loc_ind = data[1].long()
county_summer_mean = data[2]
alert = data[3]
baseline_features = data[4]
eff_features = data[5]
index = data[6]
year = data[7]
budget = data[8]

baseline_feature_names = dm.baseline_feature_names
effectiveness_feature_names = dm.effectiveness_feature_names

# Rewards model:
# model = torch.load("bayesian_model/ckpts/test_model.pt") # may need to read in the Model class first?
# guide = torch.load("bayesian_model/ckpts/test_guide.pt")
model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectiveness_constraints=dm.effectiveness_constraints,
        effectiveness_feature_names=dm.effectiveness_feature_names,
        hidden_dim= 32, #cfg.model.hidden_dim,
        num_hidden_layers= 1, #cfg.model.num_hidden_layers,
    )
# model.load_state_dict(torch.load("bayesian_model/ckpts/test_model.pt"))
model.load_state_dict(torch.load("ckpts/test_model.pt"))

guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
guide(*dm.dataset.tensors)
# guide.load_state_dict(torch.load("bayesian_model/ckpts/test_guide.pt"))
guide.load_state_dict(torch.load("ckpts/test_guide.pt"))

predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=1, # or can do >1 and take average
        return_sites=["_RETURN"],
    )


# Define your custom environment class
class HASDM_Env(gym.Env):
    def __init__(self, loc):
        # Initialize your environment variables and parameters
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(baseline_feature_names),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.loc = torch.tensor(loc).long()
        self.county = loc_ind == self.loc
        self.y = np.random.randint(2006, 2016)
        self.year = year[self.county] == self.y
        self.budget = budget[self.county][self.year][0]
        self.day = 0
        self.alerts = []
        obs = baseline_features[self.county][self.year][self.day]
        obs[baseline_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        eff = eff_features[self.county][self.year][self.day]
        eff[effectiveness_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.effectiveness_vars = eff
    def step(self, action):
        print("Day = " + str(self.day))
        # print("Index = " + str(self.county[self.year][self.day]))
        # Take an action in the environment and return the next state, reward, done flag, and additional information
        # Update new action according to the alert budget:
        if action == 1 and self.budget > 0:
            action = 1
            self.budget -= 1
        else:
            action = 0
        self.alerts.append(action)
        # Obtain reward:
        inputs = [
            hosps[self.county][self.year][self.day].reshape(1,1), 
            self.loc.reshape(1,1), 
            county_summer_mean[self.county][self.year][self.day].reshape(1,1), 
            torch.tensor(action, dtype=torch.float32).reshape(1,1), 
            self.observation[0:(len(self.observation)-1)].reshape(1,-1), 
            self.effectiveness_vars.reshape(1,-1), 
            index[self.county][self.year][self.day].reshape(1,1)
        ]
        r = predictive_outputs(*inputs, condition=False, return_outcomes=True)["_RETURN"][0]
        reward = r[0][2][0].item()
        # format = effectiveness, baseline, outcome_mean
        # R = county_summer_mean * baseline * (1 - alert[this_loc] * effectiveness)
        # Set up next observation:
        self.day += 1
        obs = baseline_features[self.county][self.year][self.day]
        eff = eff_features[self.county][self.year][self.day]
        if self.day < 14:
            s = torch.tensor(sum(self.alerts), dtype=torch.float32)
            obs[baseline_feature_names.index("previous_alerts")] = s
            eff[effectiveness_feature_names.index("previous_alerts")] = s
        else: 
            s = torch.tensor(sum(self.alerts[(self.day - 14):self.day]), dtype=torch.float32)
            obs[baseline_feature_names.index("previous_alerts")] = s
            eff[effectiveness_feature_names.index("previous_alerts")] = s
        next_observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        self.observation = next_observation
        self.effectiveness_vars = eff
        if self.day == n_days:
            terminal = True
        else:
            terminal = False
        info = {} # could keep track of extra metrics here?
        return(next_observation, reward, terminal, info)
    def reset(self):
        # Reset the environment to its initial state
        self.y = np.random.randint(2006, 2016)
        self.year = year[self.county] == self.y
        self.budget = budget[self.county][self.year][0]
        self.day = 0
        self.alerts = []
        obs = baseline_features[self.county][self.year][self.day]
        obs[baseline_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        eff = eff_features[self.county][self.year][self.day]
        eff[effectiveness_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.effectiveness_vars = eff
        return(self.observation)


## Test the env:

env = HASDM_Env(loc=2)

d=0
while d < 10:
    next_observation, reward, terminal, info = env.step(1)
    print(reward)
    if terminal:
        env.reset()
    d+= 1

