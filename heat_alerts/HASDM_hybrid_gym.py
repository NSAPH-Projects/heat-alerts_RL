
import gym
from gym import spaces

import torch
from pyro.infer import Predictive, predictive

import pandas as pd
import numpy as np
from itertools import groupby
import pyro
import json

from bayesian_model.pyro_heat_alert import HeatAlertDataModule, HeatAlertModel
# from pyro_heat_alert import HeatAlertDataModule, HeatAlertModel

## Read in data:
n_days = 153
years = set(range(2006, 2017))
n_years = len(years)
dm = HeatAlertDataModule(
        dir="bayesian_model/data/processed", # dir="data/processed",
        batch_size=n_days*n_years,
        num_workers=4,
        for_gym=True
    )

baseline_feature_names = dm.baseline_feature_names
effectiveness_feature_names = dm.effectiveness_feature_names

baseline_weather_names = ['heat_qi_base', 'heat_qi1_above_25', 'heat_qi2_above_75', 'excess_heat']
effectiveness_weather_names = ['heat_qi', 'excess_heat']

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
hi_mean = data[10]
state = data[9]
# Get unique state IDs:
W_state = state.loc[np.arange(0, len(state), n_days*n_years)]
s = np.unique(W_state)
W_state = W_state["state"].replace(s, np.arange(0,len(s)))

# The "Cold" zone is very large, so split on additional east-west feature:
western = [' AZ', ' CA', ' CO', ' ID', ' MT', ' NM', ' NV', ' OR', ' WA' 
           ,' ND', ' SD', ' NE', ' KS' # adds 10 counties to the Cold western group
           ] 
western = pd.DataFrame(western).replace(s, np.arange(0,len(s)))
western.columns = ["Num"]

# Add state ID to spatial features data frame:
spatial_features = dm.spatial_features
spatial_features = torch.cat((spatial_features,torch.tensor(np.array(W_state), dtype=torch.int).reshape(-1,1)), dim=1)
spatial_features = pd.DataFrame(spatial_features.numpy())

spatial_feature_names = dm.spatial_features_names
spatial_feature_names = spatial_feature_names.union(["State"], sort=False)
spatial_features.columns = spatial_feature_names

locations = np.arange(0, spatial_features.shape[0])

# Function for obtaining counties with similar weather:
def get_similar_counties(loc):
    this_spat = spatial_features.loc[loc]
    group = locations[spatial_features[['Cold', 'Hot-Dry', 'Marine',
       'Mixed-Dry', 'Mixed-Humid', 'Very Cold']].eq(this_spat[['Cold', 'Hot-Dry', 'Marine',
       'Mixed-Dry', 'Mixed-Humid', 'Very Cold']]).all(axis=1)]
    if this_spat["Cold"] == 1.0: # additionally separating the "Cold" region into western and eastern
        this_w = this_spat["State"] in western
        if this_w:
            group = group[np.isin(spatial_features.loc[group]["State"], western)]
        else:
            group = group[np.isin(spatial_features.loc[group]["State"], western, invert=True)]
    return(group)


## Set up the rewards model, previously trained using pyro:
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
model.load_state_dict(torch.load("bayesian_model/ckpts/Full_8-7_model.pt"))
# model.load_state_dict(torch.load("ckpts/Full_8-4_model.pt"))

predictive_outputs = Predictive(
        model,
        # guide=guide, # including the guide includes all sites by default
        num_samples=n_days, 
        return_sites=["_RETURN"],
    )

# Function for getting the average streak length of alerts:
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

## Testing:
# loc = torch.tensor(2).long()
# county = loc_ind == loc
# y = 2009
# this_y = year[county] == y

# inputs = [ hosps[county][this_y][0].reshape(1,1), loc.reshape(1,1), county_summer_mean[county][this_y][0].reshape(1,1), torch.tensor(1, dtype=torch.float32).reshape(1,1), baseline_features[county][this_y][0].reshape(1,-1), eff_features[county][this_y][0].reshape(1,-1), index[county][this_y][0].reshape(1,1) ]


## Define the custom environment class:
class HASDM_Env(gym.Env):
    def __init__( ## Initialize the environment variables and parameters...
            self, loc, y = None, # if y (a year) is passed, we are doing evaluation
            P = -10, # the penalty applied during training when the RL goes over the alert budget
            hold_out=[2015]
    ): 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(baseline_feature_names)+2,), # Don't forget to update this if you change the length of the observations!!
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        # Get the set of years allocated for training (as opposed to evaluation):
        self.year_options = np.array(list(years-set(hold_out)))
        if y is None: # training
            self.y = np.random.choice(self.year_options) # select the year
            # Obtain codes of counties in the same region:
            self.similar_counties = get_similar_counties(loc) # loc is not yet a tensor
            # Sample a county from the region:
            self.weather_loc = torch.tensor(np.random.choice(self.similar_counties)).long()
            self.weather_county = loc_ind == self.weather_loc 
        else: # evaluation
            self.y = y
        ## Get the initial environment data:
        self.loc = torch.tensor(loc).long()
        self.county = loc_ind == self.loc
        self.year = year[self.county] == self.y
        self.weather_year = year[self.weather_county] == self.y
        self.b = budget[self.county][self.year][0]
        if y is None: # training
            # Sample the budget:
            self.budget = torch.tensor(np.random.randint(0.5*self.b, 1.5*self.b+1)).long()
            self.R = torch.empty((n_days,
                                  len(effectiveness_feature_names)+len(baseline_feature_names)+2, 1), 
                                  dtype=torch.float32)
        else: # evaluation
            self.budget = self.b
        self.P = P # given penalty for the RL going over the alert budget during training
        self.day = 0
        self.alerts = []
        obs = baseline_features[self.county][self.year][self.day]
        obs[baseline_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        eff = eff_features[self.county][self.year][self.day]
        eff[effectiveness_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.effectiveness_vars = eff
        if y is None: # training
            # Use the weather features from the sampled county:
            for v in baseline_weather_names:
                pos = baseline_feature_names.index(v)
                self.observation[pos] = baseline_features[self.weather_county][self.weather_year][self.day][pos]
            for v in effectiveness_weather_names:
                pos = effectiveness_feature_names.index(v)
                self.effectiveness_vars[pos] = eff_features[self.weather_county][self.weather_year][self.day][pos]
            # Include rolling mean of heat index, which is not given to the rewards model:
            self.observation = torch.cat((self.observation,hi_mean[self.weather_county][self.weather_year][self.day].reshape(-1)))
        else:
            # Include rolling mean of heat index, which is not given to the rewards model:
            self.observation = torch.cat((self.observation,hi_mean[self.county][self.year][self.day].reshape(-1)))
        ## Create a few additional metrics to save and return at end of training:
        self.episode_sum = []
        self.episode_budget = []
        self.episode_avg_dos = []
        self.episode_avg_streak_length = []
    def step(self, action, y = None): ## Take an action in the environment and return the next state, reward, done flag, and additional information...
        ## Sample coefficients from the rewards model, at the beginning of each episode for speed:
        if self.day == 0:
            inputs = [
                hosps[self.county][self.year][self.day].reshape(1,1), 
                self.loc.reshape(1,1), 
                county_summer_mean[self.county][self.year][self.day].reshape(1,1), 
                torch.tensor(action, dtype=torch.float32).reshape(1,1), 
                self.observation[0:(len(self.observation)-2)].reshape(1,-1), 
                self.effectiveness_vars.reshape(1,-1), 
                index[self.county][self.year][self.day].reshape(1,1)
            ]
            R = predictive_outputs(*inputs, condition=False, return_outcomes=True)["_RETURN"]#[0]
            if y is not None: # evaluation
                # Calculate the average of each coefficient across the samples:
                r = torch.mean(R, dim=0)
                j = len(effectiveness_feature_names)
                self.eff_coef = r[0:j]
                k = len(baseline_feature_names)
                self.base_coef = r[j:(j+k)]
                j += k
                self.eff_bias = r[j:(j+1)]
                self.base_bias = r[(j+1):(j+2)]
            else: # training
                self.R = R
        if y is None: # training
            r = self.R[self.day] # using a different sample each time
            j = len(effectiveness_feature_names)
            self.eff_coef = r[0:j]
            k = len(baseline_feature_names)
            self.base_coef = r[j:(j+k)]
            j += k
            self.eff_bias = r[j:(j+1)]
            self.base_bias = r[(j+1):(j+2)]
        ## Update new action according to the alert budget, and penalize if the RL exceeded it:
        penalty = 0
        if action == 1 and self.budget > 0:
            action = 1
            self.budget -= 1
        elif action == 1 and self.budget == 0:
            action = 0
            penalty = self.P 
        self.alerts.append(action)
        ## Calculate reward:
        baseline_contribs = torch.matmul(self.base_coef.reshape(-1), self.observation[0:(len(self.observation)-2)])
        baseline = torch.exp(baseline_contribs + self.base_bias)
        baseline = baseline.clamp(max=1e6)
        effectiveness_contribs = torch.matmul(self.eff_coef.reshape(-1), self.effectiveness_vars)
        effectiveness = torch.exp(effectiveness_contribs + self.eff_bias)
        effectiveness = effectiveness.clamp(1e-6, 1 - 1e-6)
        reward = baseline * (1 - torch.tensor(action, dtype=torch.float32) * effectiveness) # relative scale
        # reward = county_summer_mean[self.county][self.year][self.day] * baseline * (1 - torch.tensor(action, dtype=torch.float32) * effectiveness) # absolute scale
        if y is not None: # evaluation
            Reward = -reward # Note that reward is negative so higher is better
        else: # training
            Reward = -reward + torch.tensor(penalty, dtype=torch.float32)
        ## Set up next observation:
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
        self.effectiveness_vars = eff
        if y is None: # training
            # Use the weather features from the sampled county:
            for v in baseline_weather_names:
                pos = baseline_feature_names.index(v)
                self.observation[pos] = baseline_features[self.weather_county][self.weather_year][self.day][pos]
            for v in effectiveness_weather_names:
                pos = effectiveness_feature_names.index(v)
                self.effectiveness_vars[pos] = eff_features[self.weather_county][self.weather_year][self.day][pos]
            # Include rolling mean of heat index, which is not given to the rewards model:
            next_observation = torch.cat((next_observation,hi_mean[self.weather_county][self.weather_year][self.day].reshape(-1)))
        else:
            # Include rolling mean of heat index, which is not given to the rewards model:
            next_observation = torch.cat((next_observation,hi_mean[self.county][self.year][self.day].reshape(-1)))
        self.observation = next_observation
        if self.day == n_days-1:
            terminal = True
            ## Calculate additional metrics:
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
        return(next_observation.reshape(-1,).detach().numpy(), Reward, terminal, info) 
    def reset(self, y = None): ## Reset the environment to its initial state...
        if y is None: # training
            self.y = np.random.choice(self.year_options) # select the year
            # Sample a county from the region:
            self.weather_loc = torch.tensor(np.random.choice(self.similar_counties)).long()
            self.weather_county = loc_ind == self.weather_loc
        else: # evaluation
            self.y = y
        self.year = year[self.county] == self.y
        self.weather_year = year[self.weather_county] == self.y
        self.b = budget[self.county][self.year][0]
        if y is None: # training
            # Sample the budget:
            self.budget = torch.tensor(np.random.randint(0.5*self.b, 1.5*self.b+1)).long()
        else: # evaluation
            self.budget = self.b
        self.episode_budget.append(self.budget.item()) # saving for later reference
        self.day = 0
        self.alerts = []
        ## Get the initial environment data:
        obs = baseline_features[self.county][self.year][self.day]
        obs[baseline_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.observation = torch.cat((obs,self.budget.reshape(-1))) # so the RL knows the budget
        eff = eff_features[self.county][self.year][self.day]
        eff[effectiveness_feature_names.index("previous_alerts")] = torch.tensor(0.0, dtype=torch.float32)
        self.effectiveness_vars = eff
        if y is None: # training
            # Use the weather features from the sampled county:
            for v in baseline_weather_names:
                pos = baseline_feature_names.index(v)
                self.observation[pos] = baseline_features[self.weather_county][self.weather_year][self.day][pos]
            for v in effectiveness_weather_names:
                pos = effectiveness_feature_names.index(v)
                self.effectiveness_vars[pos] = eff_features[self.weather_county][self.weather_year][self.day][pos]
            # Include rolling mean of heat index, which is not given to the rewards model:
            self.observation = torch.cat((self.observation,hi_mean[self.weather_county][self.weather_year][self.day].reshape(-1)))
        else:
            # Include rolling mean of heat index, which is not given to the rewards model:
            self.observation = torch.cat((self.observation,hi_mean[self.county][self.year][self.day].reshape(-1)))
        return(self.observation.reshape(-1,).detach().numpy())



# ## Test the env:
# env = HASDM_Env(loc=400)
# env.reset() # y = 2009
# d = 0
# y = 2016
# while d < 20:
#     next_observation, reward, terminal, info = env.step(1) #,y
#     print(reward)
#     # print(next_observation)
#     if terminal:
#         env.reset(y)
#     d+= 1


# #### Evaluate observed actions:

# with open("bayesian_model/data/processed/fips2idx.json","r") as f:
#         crosswalk = json.load(f)

# counties = np.fromiter(crosswalk.keys(), dtype=int)
# locations = np.fromiter(crosswalk.values(), dtype=int)

# Results = pd.DataFrame(columns=["Actions", "Rewards", "Year", "County"])

# for i in range(0, len(locations)):  
#     env = HASDM_Env(loc=locations[i])
#     Rewards = []
#     Actions = []
#     Year = []
#     for y in years:
#         obs = env.reset(y)
#         obs = torch.tensor(obs,dtype=torch.float32).reshape(1,-1)
#         # action = alert[env.county][env.year][env.day].item()
#         action = 0
#         terminal = False
#         while terminal == False:
#             if action == 1 and env.budget == 0:
#                 action = 0
#             Actions.append(action)
#             Year.append(y)
#             obs, reward, terminal, info = env.step(action, y)
#             Rewards.append(reward.item())
#             obs = torch.tensor(obs,dtype=torch.float32).reshape(1,-1)
#             # action = alert[env.county][env.year][env.day].item()
#             action = 0
#         print(y)
#     results = pd.DataFrame(np.array([Actions, Rewards]).T)
#     results.columns = ["Actions", "Rewards"]
#     results["Year"] = Year
#     results["County"] = counties[i]
#     Results = pd.concat([Results, results], ignore_index=True)
#     print(locations[i])

# # Results.to_csv("Summer_results/ORL_eval_NWS.csv")
# Results.to_csv("Summer_results/ORL_eval_zero.csv")
