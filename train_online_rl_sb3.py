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

from old_evaluation_SB3 import custom_eval


def make_env(rank: int, seed: int, **kwargs) -> HeatAlertEnv:
    """Auxiliary function to make parallel vectorized envs"""
    set_random_seed(seed)

    def _init() -> HeatAlertEnv:
        return HeatAlertEnv(global_seed = rank, **kwargs)

    return _init

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
    base_dict, effect_dict, extra_dict, other_dict = load_rl_states_by_county(
        cfg.county,
        cfg.datadir,
        years=cfg.train_years,
        match_similar=cfg.match_similar,
        as_tensors=True,
    )
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
    kwargs = dict(
        posterior_coefficient_samples=samples,
        baseline_states=base_dict,
        effectiveness_states=effect_dict,
        extra_states=extra_dict,
        other_data = other_dict,
        eval_mode=cfg.eval_mode,
        penalty=cfg.penalty,
        prev_alert_mean = dm.prev_alert_mean,
        prev_alert_std = dm.prev_alert_std,
        sample_budget = cfg.sample_budget,
        explore_budget = cfg.explore_budget,
        penalty_decay = cfg.penalty_decay,
        restrict_alerts = cfg.restrict_alerts,
        HI_restriction = cfg.HI_restriction,
    )
    val_kwargs = dict(
        posterior_coefficient_samples=samples,
        baseline_states=base_dict_val,
        effectiveness_states=effect_dict_val,
        extra_states=extra_dict_val,
        other_data = other_dict_val,
        eval_mode=cfg.eval.eval_mode,
        penalty=cfg.eval.penalty,
        prev_alert_mean = dm.prev_alert_mean,
        prev_alert_std = dm.prev_alert_std,
        sample_budget = cfg.sample_budget,
        explore_budget = False,
        restrict_alerts = cfg.restrict_alerts,
        HI_restriction = cfg.HI_restriction,
    )
    env_promise = [make_env(i, cfg.seed, **kwargs) for i in range(cfg.num_envs)]
    env_promise_val = [make_env(i, cfg.seed, **val_kwargs) for i in range(cfg.num_envs)]

    # vectorize environments
    if cfg.parallel:
        fun = partial(SubprocVecEnv) # , start_method="spawn"
    else:
        fun = DummyVecEnv

    env = fun(env_promise)
    env_val = fun(env_promise_val)

    # RL training code here
    logging.info("Creating RL model")
    logger = configure(f"./logs/SB/{cfg.model_name}/training_metrics", ["csv", "tensorboard"])
    # print(cfg.algo)
    rl_model = hydra.utils.instantiate(
            cfg.algo, env = env, verbose=0 #, tensorboard_log="./logs/rl_tensorboard/"
        )
    rl_model.set_logger(logger)

    # Create a callback to evaluate the agent
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=f"./logs/SB/{cfg.model_name}/best_model",
        log_path=f"./logs/SB/{cfg.model_name}/results",
        eval_freq=cfg.eval.freq,  # Evaluation frequency
        n_eval_episodes=cfg.eval.episodes,
    )
    alert_logging_callback = AlertLoggingCallback()

    # Training the agent
    logging.info("Training RL model")
    rl_model.learn(
        total_timesteps=cfg.training_timesteps,
        callback=[eval_callback, alert_logging_callback],
        progress_bar=True,
    )

    logging.info("Performing evaluations")
    new_cfg = cfg
    for v in [True, False]:
        for m in [True, False]:
            new_cfg.eval.val_years = v
            new_cfg.eval.match_similar = m
            custom_eval(new_cfg, dm=dm, samples=samples)
            logging.info("Completed eval with eval.val_years="+str(v)+" and eval.match_similar="+str(m))


if __name__ == "__main__":
    main()
