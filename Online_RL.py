
import glob
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import random

import torch
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN, DiscreteSAC
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.metrics.scorer import td_error_scorer, average_value_estimation_scorer

from heat_alerts.HASDM_hybrid_gym import HASDM_Env

def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    d3rlpy.seed(seed)

def main(params):
    params = vars(params)
    seed = params["seed"]
    # seed = 321
    set_seed(seed)

    print("Holding out:" + str(params["hold_out"]))

    # params=dict(
    #     fips = 4013, model_name = "test_sac_4013", algo="SAC",
    #     n_hidden = 32, n_layers = 2,
    #     n_epochs = 5, sa = 1, sync_rate = 3,
    #     b_size = 153, lr = 0.1, gamma = 0.999,
    #     n_gpus = 0, update_rate = 5,
    #     eps_0 = 1.0, eps_t = 0.00000001, eps_dur = 1.0,
    #     hold_out = [2015], penalty = -5
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

    functions = [DQN, DoubleDQN, DiscreteSAC]
    func_names = ["DQN", "DoubleDQN", "SAC"]
    algos = dict(zip(func_names, functions))
    algo = algos[params["algo"]]

    if algo == DQN or algo == DoubleDQN:
        RL = algo(
            encoder_factory=encoder_factory,
            use_gpu=gpu, 
            batch_size=params["b_size"],
            learning_rate=params["lr"],
            gamma=params["gamma"],
            target_update_interval=H*params["sync_rate"],
            scaler = None,
            reward_scaler = None
            )
    elif algo == DiscreteSAC:
        RL = algo(
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            use_gpu=gpu, 
            batch_size=params["b_size"],
            actor_learning_rate=params["lr"],
            critic_learning_rate=params["lr"],
            temp_learning_rate=params["lr"],
            gamma=params["gamma"],
            target_update_interval=H*params["sync_rate"],
            scaler = None,
            reward_scaler = None
            )
    
    with open("bayesian_model/data/processed/fips2idx.json","r") as f:
        crosswalk = json.load(f)
    
    hold_out = params["hold_out"]
    if type(hold_out) == list:
        env = HASDM_Env(
            loc=crosswalk[str(params["fips"])],
            P=params["penalty"],
            hold_out=hold_out
        )
    else: 
        hold_out = [hold_out]
        env = HASDM_Env(
            loc=crosswalk[str(params["fips"])],
            P=params["penalty"],
            hold_out=hold_out
        )
    # eval_env = HASDM_Env(params["loc"])

    buffer = ReplayBuffer(maxlen=H*params["n_epochs"], env=env)

    if params["xpl"] == "T":
        explorer = LinearDecayEpsilonGreedy(start_epsilon=params["eps_0"],
                                    end_epsilon=params["eps_t"],
                                    duration=H*params["n_epochs"]*params["eps_dur"])
    else: 
        explorer = None

    
    RL.fit_online(env,
               buffer,
               explorer,
               experiment_name=name, 
               with_timestamp=False,
               n_steps=H*params["n_epochs"], 
               # eval_env=eval_env,
               n_steps_per_epoch=n_days, 
               update_interval=params["update_rate"],
               # update_start_step=1000,
               save_interval = params["sa"])
    
    B = env.episode_budget 
    del B[len(B)-1] # env.reset() gets called one extra time at the end
    DF = pd.DataFrame(np.array([env.episode_sum, B, env.episode_avg_dos, env.episode_avg_streak_length]).T)
    DF.columns = ["Alert_sum", "Budget", "Avg_DOS", "Avg_StrkLn"]
    DF.to_csv("d3rlpy_logs/" + name + "/custom_metrics.csv")

    ## Evaluation:
    models = glob.glob("d3rlpy_logs/" + name + "/model_*")

    Training_Results = pd.DataFrame(columns=["Actions", "Rewards", "Year", "Model"])
    Training_Penalty_Results = pd.DataFrame(columns=["Actions", "Rewards", "Year", "Model"])
    Evaluation_Results = pd.DataFrame(columns=["Actions", "Rewards", "Year", "Model"])
    i = 0
    for m in models:
        RL.load_model(m)
        T_Rewards = []
        TP_Rewards = []
        T_Actions = []
        T_Year = []
        E_Rewards = []
        E_Actions = []
        E_Year = []
        eval_env = HASDM_Env(loc=crosswalk[str(params["fips"])])
        for y in range(2006, 2017):
            obs = eval_env.reset(y)
            obs = torch.tensor(obs,dtype=torch.float32).reshape(1,-1)
            action = RL.predict(obs).item()
            terminal = False
            if y in hold_out:
                while terminal == False:
                    if action == 1 and eval_env.budget == 0:
                        action = 0
                    E_Actions.append(action)
                    E_Year.append(y)
                    obs, reward, terminal, info = eval_env.step(action, y, absolute=True)
                    E_Rewards.append(reward.item())
                    obs = torch.tensor(obs,dtype=torch.float32).reshape(1,-1)
                    action = RL.predict(obs).item()
            else:
                while terminal == False:
                    penalty = 0
                    if action == 1 and eval_env.budget == 0:
                        action = 0
                        penalty = params["penalty"]
                    T_Actions.append(action)
                    T_Year.append(y)
                    obs, reward, terminal, info = eval_env.step(action, y)
                    T_Rewards.append(reward.item())
                    TP_Rewards.append(reward.item() + penalty)
                    obs = torch.tensor(obs,dtype=torch.float32).reshape(1,-1)
                    action = RL.predict(obs).item()
            print(y)
        T_results = pd.DataFrame(np.array([T_Actions, T_Rewards]).T)
        T_results.columns = ["Actions", "Rewards"]
        T_results["Year"] = T_Year
        T_results["Model"] = i
        Training_Results = pd.concat([Training_Results, T_results], ignore_index=True)
        TP_results = pd.DataFrame(np.array([T_Actions, TP_Rewards]).T)
        TP_results.columns = ["Actions", "Rewards"]
        TP_results["Year"] = T_Year
        TP_results["Model"] = i
        Training_Penalty_Results = pd.concat([Training_Results, T_results], ignore_index=True)
        E_results = pd.DataFrame(np.array([E_Actions, E_Rewards]).T)
        E_results.columns = ["Actions", "Rewards"]
        E_results["Year"] = E_Year
        E_results["Model"] = i
        Evaluation_Results = pd.concat([Evaluation_Results, E_results], ignore_index=True)
        i += 1

    Training_Results.to_csv("Summer_results/ORL_training_" + name + ".csv")
    Training_Penalty_Results.to_csv("Summer_results/ORL_training_penalty_" + name + ".csv")
    Evaluation_Results.to_csv("Summer_results/ORL_eval_" + name + ".csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fips", type=int, default=4013, help="fips code")
    parser.add_argument("--hold_out", nargs='+', type=int, default=2015, help="evaluation years")
    parser.add_argument("--algo", type=str, default="DQN", help="RL algorithm")
    parser.add_argument("--xpl", type=str, default="F", help="Use explorer?")
    parser.add_argument("--eps_0", type=float, default=1.0, help="epsilon start")
    parser.add_argument("--eps_t", type=float, default=0.00000001, help="epsilon end")
    parser.add_argument("--eps_dur", type=float, default=1.0, help="epsilon duration (fraction)")
    parser.add_argument("--penalty", type=float, default=-10.0, help="penalty for going over the alert budget (during training)")
    parser.add_argument("--seed", type=int, default=321, help="set seed")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--b_size", type=int, default=500, help="size of the batches")
    parser.add_argument("--n_layers", type=int, default=2, help="how many hidden layers in DQN")
    parser.add_argument("--n_hidden", type=int, default=32, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--update_rate", type=int, default=5, help="how often (in epochs) to update the online RL")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to run")
    parser.add_argument("--n_gpus", type=int, default=0, help="number of gpus")
    parser.add_argument("--sa", type=int, default=10, help="save model params every X episodes")

    args = parser.parse_args()
    main(args)
