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

from heat_alerts.bayesian_model import HeatAlertDataModule, HeatAlertModel
from heat_alerts.online_rl.datautils import load_rl_states_by_county
from heat_alerts.online_rl.env import HeatAlertEnv
from heat_alerts.online_rl.callbacks import AlertLoggingCallback


# hydra.initialize(config_path="conf/online_rl/sb3", version_base=None)
# cfg = hydra.compose(config_name="config")
# cfg.policy_type="NWS"
# cfg.policy_type="NA"

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
        eval_mode = cfg.eval.eval_mode,
        sample_budget = False,
        years = cfg.val_years,
    )

    eval_env = HeatAlertEnv(**val_kwargs)

    if cfg.policy_type == "RL":
        if cfg.algo['_target_'] == 'sb3_contrib.RecurrentPPO':
            rl_model = hydra.utils.instantiate(
                cfg.algo, policy="MlpLstmPolicy", env = eval_env, verbose=0 #, tensorboard_log="./logs/rl_tensorboard/"
            )
        else:
            rl_model = hydra.utils.instantiate(
                cfg.algo, policy="MlpPolicy", env = eval_env, verbose=0 #, tensorboard_log="./logs/rl_tensorboard/"
            )
        rl_model.load(f"./logs/SB/{cfg.model_name}/best_model/best_model")
    else:
        rl_model = None


    def get_action(policy_type, obs, env, rl_model=None):
        if policy_type == "RL":
            return(rl_model.predict(obs)[0].item())
        elif policy_type == "NA":
            return(0)
        elif policy_type == "NWS":
            return(env.other_data["nws_alert"][env.feature_ep_index, env.t])

    rewards = []
    actions = []
    year = []
    budget = []

    for i in range(0, cfg.final_eval_episodes):
        obs, info = eval_env.reset()
        action = get_action(cfg.policy_type, obs, eval_env, rl_model)
        terminal = False
        while terminal == False:
            if action == 1 and info["over_budget"]:
                action = 0
            obs, reward, terminal, trunc, info = eval_env.step(action)
            rewards.append(reward)
            year.append(eval_env.other_data["y"][eval_env.feature_ep_index, eval_env.t].item())
            budget.append(eval_env.other_data["budget"][eval_env.feature_ep_index, eval_env.t].item())
            action = get_action(cfg.policy_type, obs, eval_env, rl_model)
        actions.extend([x.item() if torch.is_tensor(x) else x for x in eval_env.allowed_alert_buffer])
        print(i)

    results = pd.DataFrame(
        {'Year': year, 'Budget': budget, 'Actions': actions, 'Rewards': rewards}
    )

    year_set = "eval" if cfg.eval.val_years else "train"
    posterior = "avg-R" if cfg.eval.eval_mode else "samp-R"
    weather = "samp-W" if cfg.eval.match_similar else "obs-W"
    results.to_csv(f"Summer_results/ORL_{cfg.policy_type}_{year_set}_{posterior}_{weather}_{cfg.model_name}_fips_{cfg.county}.csv")

if __name__ == "__main__":
    main()
