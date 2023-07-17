
import gym
from gym import spaces

import torch
from pyro.infer import Predictive

import pandas as pd
import numpy as np


# Set up data:
n_days = 153
vars = ["quant_HI_county", "quant_HI_3d_county", "dos", "alert_sum", "More_alerts", 
        "alerts_2wks", "year"]
Train = pd.read_csv("data/Summer23_Train_smaller-for-Python.csv") 
Train["More_alerts"] = 0 # filler column

# Rewards model:
model = torch.load("Bayesian_models/Pyro_model_7-13.pt") # may need to read in the Model class first?
guide = torch.load("Bayesian_models/Pyro_guide_7-13.pt")
predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=1, # or can do >1 and take average
        return_sites=["_RETURN"],
    )


# Define your custom environment class
class CustomEnv(gym.Env):
    def __init__(self, fips):
        # Initialize your environment variables and parameters
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(vars),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.county_data = Train.loc[Train["fips"] == "fips"][vars]
        self.year = np.random.randint(2006, 2016)
        self.data = self.county_data.loc[self.county_data["year"] == self.year]
        self.budget = self.data["alert_sum"].iloc[n_days-1]
        # standardize self.data?
        self.day = 0
        self.alerts = []

    def step(self, action):
        # Take an action in the environment and return the next state, reward, done flag, and additional information

        # Update new action according to the alert budget:
        if action == 1 and self.budget > 0:
            action = 1
            self.budget -= 1
        else:
            action = 0
        self.alerts.append(action)

        # Obtain reward:
        inputs = [] # need to be tensors
        reward = predictive_outputs(*inputs, return_all=True)["_RETURN"]
        
        # Set up next observation:
        obs = self.data[vars].iloc[self.day]
        obs["alert_sum"] = sum(self.alerts)
        obs["More_alerts"] = self.budget
        if self.day < 14:
            obs["alerts_2wks"] = sum(self.alerts)
        else: 
            obs["alerts_2wks"] = sum(self.alerts[(self.day - 14):self.day])
        next_observation = np.array(obs)

        self.day += 1
        if self.day == n_days:
            terminal = True
        else:
            terminal = False

        info = {}
        return(next_observation, reward, terminal, info)

def reset(self):
        # Reset the environment to its initial state
        self.year = np.random.randint(2006, 2016)
        self.data = self.county_data.loc[self.county_data["year"] == self.year]
        self.budget = self.data["alert_sum"].iloc[n_days-1]
        # standardize self.data?
        self.day = 0
        self.alerts = []
        obs = self.data[vars].iloc[self.day]
        obs["alert_sum"] = sum(self.alerts)
        obs["More_alerts"] = self.budget 
        obs["alerts_2wks"] = 0 # sum(self.alerts[(self.day - 13):self.day])
        self.observation = np.array(obs)
        return(self.observation)
