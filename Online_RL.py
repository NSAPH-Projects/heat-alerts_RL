

import datetime
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import random

import gym

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN, SAC
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.metrics.scorer import td_error_scorer, average_value_estimation_scorer

# from HASDM_hybrid_gym import HASDM_Env
from heat_alerts.HASDM_hybrid_gym import HASDM_Env

def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(params):
    params = vars(params)
    seed = params["seed"]
    # seed = 321
    set_seed(seed)
    d3rlpy.seed(seed)

    # params=dict(
    #     fips = 4013, model_name = "test_4013", algo="DQN",
    #     n_hidden = 32, n_layers = 2,
    #     n_epochs = 50, sa = 1, sync_rate = 3,
    #     b_size = 153, lr = 0.1, gamma = 0.999,
    #     n_gpus = 0
    # )

    n_days=153
    H=n_days-1
    n_years=10
    # n_steps_per_epoch=h*n_years
    name = params["model_name"]

    n_hidden = params["n_hidden"]
    encoder_factory = VectorEncoderFactory(hidden_units=[n_hidden]*params["n_layers"], activation='relu') # doesn't allow for 'elu'

    gpu = False
    device = "cpu"
    if params["n_gpus"] > 0: 
        gpu = True
        device = "cuda"

    functions = [DQN, DoubleDQN, SAC]
    func_names = ["DQN", "DoubleDQN", "SAC"]
    algos = dict(zip(func_names, functions))
    algo = algos[params["algo"]]

    dqn = algo(
        encoder_factory=encoder_factory,
        use_gpu=gpu, 
        batch_size=params["b_size"],
        learning_rate=params["lr"],
        gamma=params["gamma"],
        target_update_interval=H*params["sync_rate"],
        scaler = None,
        reward_scaler = None
        )
    
    with open("bayesian_model/data/processed/fips2idx.json","r") as f:
        crosswalk = json.load(f)
    
    env = HASDM_Env(crosswalk[str(params["fips"])])
    # eval_env = HASDM_Env(params["loc"])

    buffer = ReplayBuffer(maxlen=H*params["n_epochs"], env=env)

    if params["xpl"] == "T":
        explorer = LinearDecayEpsilonGreedy(start_epsilon=params["eps_0"],
                                    end_epsilon=params["eps_t"],
                                    duration=H*params["n_epochs"]*params["eps_dur"])
    else: 
        explorer = None

    dqn.fit_online(env,
               buffer,
               explorer,
               experiment_name=name,
               with_timestamp=False,
               n_steps=H*params["n_epochs"],
               # eval_env=eval_env,
               n_steps_per_epoch=n_days, 
               update_interval=10,
               # update_start_step=1000,
               save_interval = params["sa"])
    
    B = env.episode_budget 
    del B[len(B)-1] # env.reset() gets called one extra time at the end
    DF = pd.DataFrame(np.array([env.episode_sum, B, env.episode_avg_dos, env.episode_avg_streak_length]).T)
    DF.columns = ["Alert_sum", "Budget", "Avg_DOS", "Avg_StrkLn"]
    DF.to_csv("d3rlpy_logs/" + name + "/custom_metrics.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fips", type=int, default=4013, help="fips code")
    parser.add_argument("--algo", type=str, default="DQN", help="RL algorithm")
    parser.add_argument("--xpl", type=str, default="F", help="Use explorer?")
    parser.add_argument("--eps_0", type=float, default=1.0, help="epsilon start")
    parser.add_argument("--eps_t", type=float, default=0.00000001, help="epsilon end")
    parser.add_argument("--eps_dur", type=float, default=1.0, help="epsilon duration (fraction)")
    parser.add_argument("--seed", type=int, default=321, help="set seed")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--b_size", type=int, default=500, help="size of the batches")
    parser.add_argument("--n_layers", type=int, default=2, help="how many hidden layers in DQN")
    parser.add_argument("--n_hidden", type=int, default=32, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to run")
    parser.add_argument("--n_gpus", type=int, default=0, help="number of gpus")
    parser.add_argument("--sa", type=int, default=10, help="save model params every X episodes")

    args = parser.parse_args()
    main(args)
