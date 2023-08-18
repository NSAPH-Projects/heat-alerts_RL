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
from stable_baselines3.common.logger import configure

from heat_alerts.bayesian_model import HeatAlertDataModule, HeatAlertModel
from heat_alerts.online_rl.datautils import load_rl_states_by_county
from heat_alerts.online_rl.env import HeatAlertEnv
from heat_alerts.online_rl.callbacks import AlertLoggingCallback


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
        years=cfg.val_years,
        match_similar=False,
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
        rl_model = hydra.utils.instantiate(
            cfg.algo, policy="MlpPolicy", env = eval_env, verbose=0 #, tensorboard_log="./logs/rl_tensorboard/"
        )
        rl_model.load(f"./logs/SB/{cfg.model_name}/best_model/best_model")
    else:
        rl_model = None


    def get_action(policy_type, obs, env, rl_model=None):
        if policy_type == "RL":
            return(rl_model.predict(obs)[0].item())
        elif policy_type == "no alerts":
            return(0)
        elif policy_type == "NWS":
            a = env.other_data["nws_alert"][env.feature_ep_index, env.t]
            # print(a)
            return(a)

    rewards = []
    actions = []
    year = []
    i = 0
    a = 0
    n_reps = 1 if cfg.final_eval_EM else cfg.num_posterior_samples
    val_years = [x for x in cfg.val_years]

    for y in val_years*n_reps:
        obs = eval_env.reset(year=y)[0]
        action = get_action(cfg.policy_type, obs, eval_env, rl_model)
        terminal = False
        while terminal == False:
            if action == 1 and eval_env.over_budget() == False:
                a = i
            elif action == 1 and eval_env.over_budget():
                action = 0
                actions[a] = 0 
            actions.append(action)
            year.append(y)
            obs, reward, terminal, trunc, info = eval_env.step(action)
            rewards.append(reward)
            action = get_action(cfg.policy_type, obs, eval_env, rl_model)
            i += 1
        print(y)

    results = pd.DataFrame(
        {'Actions': actions, 'Rewards': rewards, 'Year': year}
    )
    if cfg.policy_type == "RL":
        results.to_csv(f"Summer_results/ORL_train_{cfg.model_name}_fips_{cfg.county}.csv")
    elif cfg.policy_type == "no alerts":
        results.to_csv(f"Summer_results/ORL_train_No_alerts_fips_{cfg.county}.csv")
    elif cfg.policy_type == "NWS":
        results.to_csv(f"Summer_results/ORL_train_NWS_fips_{cfg.county}.csv")

if __name__ == "__main__":
    main()
