import json
import logging
from functools import partial

import hydra
import numpy as np
import pandas as pd
import pyro
import torch
from omegaconf import DictConfig

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces

from heat_alerts.bayesian_model import HeatAlertDataModule, HeatAlertModel
from heat_alerts.online_rl.datautils import load_rl_states_by_county
from heat_alerts.online_rl.env import HeatAlertEnv
from heat_alerts.online_rl.callbacks import FinalEvalCallback


class No_alert_policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super(No_alert_policy, self).__init__(observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space
    def _predict(self, obs, deterministic = True):
        action = 0
        return action, None
    def forward(self, obs):
        action = 0
        return action, None
    
class NWS_policy(BasePolicy):
    def __init__(self, env, observation_space, action_space):
        super(NWS_policy, self).__init__(env, observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
    def _predict(self, obs, deterministic = True):
        action = self.env.other_data["nws_alert"][self.env.feature_ep_index, self.env.t]
        return action
    def forward(self, obs):
        action = self.env.other_data["nws_alert"][self.env.feature_ep_index, self.env.t]
        return action, None
    
# hydra.initialize(config_path="conf/online_rl/sb3", version_base=None)
# cfg = hydra.compose(config_name="config")

@hydra.main(config_path="conf/online_rl/sb3", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set seed
    set_random_seed(cfg.seed)

    # instantiate guide
    # TODO: I wish the guide could be loaded more elegantly!
    logging.info("Instantiating guide")
    dm = HeatAlertDataModule(dir=cfg.datadir, load_outcome=False)

    # Load model
    logging.info("Creating model")
    model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectiveness_constraints=dm.effectiveness_constraints,
        effectiveness_feature_names=dm.effectiveness_feature_names,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
    )

    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)  # initializes the guide

    # Load checkpoint
    logging.info("Loading checkpoint")
    guide.load_state_dict(torch.load(cfg.guide_ckpt, map_location=torch.device("cpu")))

    # Load states data
    logging.info("Loading RL states data")
    base_dict_val, effect_dict_val, extra_dict_val, other_dict_val = load_rl_states_by_county(
        cfg.county,
        cfg.datadir,
        years=cfg.val_years if cfg.eval.val_years else cfg.train_years,
        match_similar=cfg.eval.match_similar,
        as_tensors=True,
    )

    logging.info("Loading supporting county data (index mapping)")
    with open(f"{cfg.datadir}/fips2idx.json", "r") as f:
        fips2ix = json.load(f)
        fips2ix = {int(k): v for k, v in fips2ix.items()}
    ix = fips2ix[cfg.county]
    
    # take cfg.num_posterior_samples from the guide and make numpy arrays
    logging.info(f"Sampling posterior from guide {cfg.num_posterior_samples} times")
    with torch.no_grad():
        samples = [guide(*dm.dataset.tensors) for _ in range(cfg.num_posterior_samples)]

    def _clean_key(k: str):
        "some transforms caused by code inconsistencies, should eventually remove"
        return k.replace("qi_base", "qi").replace("qi1", "qi").replace("qi2", "qi")

    samples = {
        _clean_key(k): np.array([s[k][ix].item() for s in samples]) for k in samples[0].keys()
    }

    # make RL env
    logging.info("Making RL environment")
    val_kwargs = dict(
        posterior_coefficient_samples=samples,
        baseline_states=base_dict_val,
        effectiveness_states=effect_dict_val,
        extra_states=extra_dict_val,
        other_data = other_dict_val,
        penalty=cfg.eval.penalty,
        prev_alert_mean = dm.prev_alert_mean,
        prev_alert_std = dm.prev_alert_std,
        eval_mode = cfg.final_eval_EM,
        years = cfg.val_years,
    )

    eval_env = HeatAlertEnv(**val_kwargs)

    if cfg.policy_type == "RL":
        model = hydra.utils.instantiate(
            cfg.algo, policy="MlpPolicy", env = eval_env, verbose=0 #, tensorboard_log="./logs/rl_tensorboard/"
        )
        model.load(f"./logs/SB/{cfg.model_name}/best_model/best_model")
    elif cfg.policy_type == "no alerts":
        model = No_alert_policy(observation_space = eval_env.observation_space, action_space = spaces.Discrete(1))
    elif cfg.policy_type == "NWS":
        model = NWS_policy(env = eval_env, observation_space = eval_env.observation_space, action_space = eval_env.action_space)

    my_callback = FinalEvalCallback(filename = "./logs/SB/{cfg.model_name}/evaluation.csv")
    results = evaluate_policy(model, eval_env, n_eval_episodes=3, callback = my_callback)



    
if __name__ == "__main__":
    main()
