
import gym
from gym import spaces

import torch
from pyro.infer import Predictive, predictive

import pandas as pd
import numpy as np
from itertools import groupby
import pyro

from bayesian_model.pyro_heat_alert import HeatAlertDataModule, HeatAlertModel
# from pyro_heat_alert import HeatAlertDataModule, HeatAlertModel

## Read in data:
n_days = 153
n_years = 10
dm = HeatAlertDataModule(
        dir="bayesian_model/data/processed", # dir="data/processed",
        batch_size=n_days*n_years,
        num_workers=4,
        for_gym=True
    )
data = dm.gym_dataset

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

## Rewards model:
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
model.load_state_dict(torch.load("bayesian_model/ckpts/Full_7-19_model.pt"))
# model.load_state_dict(torch.load("ckpts/Full_7-19_model.pt"))

guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
guide(*dm.dataset.tensors)
guide.load_state_dict(torch.load("bayesian_model/ckpts/Full_7-19_guide.pt"))
# guide.load_state_dict(torch.load("ckpts/Full_7-19_guide.pt"))

predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=1, # or can do >1 and take average
        return_sites=["_RETURN"],
    )

def avg_streak_length(inds): # inds = indices (days) of alerts
    n = len(inds)
    diffs = inds[1:n] - inds[0:(n-1)]
    rle = [(k, sum(1 for i in g)) for k,g in groupby(diffs)]
    D = dict()
    for tup in rle:
        if str(tup[0]) in D.keys():
            D[str(tup[0])].append(tup[1])
        else:
            D[str(tup[0])] = [tup[1]]
    if "1" in D.keys():
        return(np.mean(np.array(D["1"])+1))
    else:
        return(0)

## Define the custom environment class:

class HASDM_Env(gym.Env):
    def __init__(self, loc):
        # Initialize your environment variables and parameters
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(baseline_feature_names)+1,),
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
        # create a couple of metrics to save and return at end:
        self.episode_sum = []
        self.episode_budget = []
        self.episode_avg_dos = []
        self.episode_avg_streak_length = []
    def step(self, action): # Take an action in the environment and return the next state, reward, done flag, and additional information
        # print("Day = " + str(self.day))
        # print("Index = " + str(self.county[self.year][self.day]))
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
        # R = county_summer_mean * baseline * (1 - alert * effectiveness)
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
        if self.day == n_days-1:
            terminal = True
            k = sum(self.alerts)
            self.episode_sum.append(k)
            if k > 0:
                i = np.where(np.array(self.alerts) == 1)[0]
                self.episode_avg_dos.append(np.mean(i+1))
                if k > 1:
                    self.episode_avg_streak_length.append(avg_streak_length(i))
                else:
                    self.episode_avg_streak_length.append(0)
            else:
                self.episode_avg_dos.append(-1)
                self.episode_avg_streak_length.append(0)
        else:
            terminal = False
        info = {} 
        return(next_observation.reshape(-1,).detach().numpy(), reward, terminal, info)
    def reset(self):
        # Reset the environment to its initial state
        self.y = np.random.randint(2006, 2016)
        self.year = year[self.county] == self.y
        self.budget = budget[self.county][self.year][0]
        self.episode_budget.append(self.budget.item()) # saving for later reference
        self.day = 0
        self.alerts = []
        obs = baseline_features[self.county][self.year][self.day]
        obs[baseline_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        eff = eff_features[self.county][self.year][self.day]
        eff[effectiveness_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.effectiveness_vars = eff
        return(self.observation.reshape(-1,).detach().numpy())


# ## Test the env:

# env = HASDM_Env(loc=2)
# env.reset()
# d=0
# while d < 200:
#     next_observation, reward, terminal, info = env.step(1)
#     # print(reward)
#     print(next_observation)
#     if terminal:
#         env.reset()
#     d+= 1

