
from argparse import ArgumentParser
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN

# from heat_alerts.Setup_d3rlpy import make_data
from Setup_d3rlpy import make_data
# from heat_alerts.cpq import CPQ
from cpq import CPQ

def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(params):

    d3rlpy.seed(321)
    params = vars(params)

    data = make_data(
        eligible = "90pct", manual_S_size = "small"
    )

    # data = make_data(
    #     outcome = params["outcome"], modeled_r = params["modeled_r"], # log_r = True,
    #     random_effects = params["random_effects"], eligible = params["eligible"],
    #     pca = params["pca"], pca_var_thresh = params["pca_var_thresh"], manual_S_size = params["S_size"]
    # )
    dataset = data[0]
    s_means = data[3]
    s_stds = data[4]
    # summer = data[3] # index of episode (county-summer)

    if params["policy_type"] == "DQN":
        name = params["model_name"]

        ## Set up NN:
        n_hidden = params["n_hidden"]
        encoder_factory = VectorEncoderFactory(hidden_units=[n_hidden]*3, activation='relu') # doesn't allow for 'elu'

        gpu = False
        if params["n_gpus"] > 0: gpu = True

        functions = [DQN, DoubleDQN, CPQ]
        func_names = ["DQN", "DoubleDQN", "CPQ"]
        algos = dict(zip(func_names, functions))
        algo = algos[params["algo"]] # DQN, DoubleDQN, CPQ

        dqn = algo(
            encoder_factory=encoder_factory,
            use_gpu=gpu, 
            batch_size=params["b_size"],
            learning_rate=params["lr"],
            gamma=params["gamma"],
            target_update_interval=params["b_size"]*params["sync_rate"]
            ) 
        
        dqn.build_with_dataset(dataset) 
        dqn.load_model("d3rlpy_logs/" + params["final_model"])
    # elif params["policy_type"] == "random":
    #     name = params["model_name"] 
    elif params["policy_type"] == "NWS":
        name = "NWS"
        policy = dataset.actions

    ## Iterate through each episode (county-summer):
    Policy = np.zeros(len(dataset.observations))
    p = 0
    for i in range(0, len(dataset)): # test with i=1 for nonzero constraint
        alert_sum = 0
        t_since_alert = dataset.episodes[i].observations[0][6]*s_stds["T_since_alert"] + s_means["T_since_alert"] - (dataset.episodes[i].observations[0][5]*s_stds["dos"] + s_means["dos"])
        budget = dataset.episodes[i].observations[0][8]*s_stds["More_alerts"] + s_means["More_alerts"]
        d = 0
        dos_alert = -t_since_alert
        while d < len(dataset.episodes[i].observations):
            new_s = dataset.observations[p]
            new_s[7] = (alert_sum - s_means["alert_sum"])/s_stds["alert_sum"]
            new_s[8] = (budget - alert_sum - s_means["More_alerts"])/s_stds["More_alerts"]
            dos = new_s[5]*s_stds["dos"] + s_means["dos"]
            new_s[6] = (dos - dos_alert - s_means["T_since_alert"])/s_stds["T_since_alert"]
            dataset.observations[p] = new_s
            ## Get new policy:
            if params["policy_type"] == "DQN":
                output = dqn.predict(dataset.observations[p:(p+1)])
                if output == 1:
                    if params["algo"] == "CPQ" and alert_sum >= budget:
                        action = 0
                    else:
                        Policy[p] = 1
                        action = 1
                else: 
                    action = 0
            else: 
                action = policy.iloc[p]
            if action == 1:
                alert_sum += 1
                dos_alert = dos
            d+=1
            p+=1
        print(i)

    if params["policy_type"] != "NWS":
        pd.DataFrame(Policy).to_csv("Policies/Policy_" + name + ".csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="other_hosps", help = "deaths or hosps")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--eligible", type=str, default="all", help="days to include in RL")
    parser.add_argument("--S_size", type=str, default="medium", help="Manual size of state matrix")
    parser.add_argument("--algo", type=str, default="DQN", help="RL algorithm")
    parser.add_argument("--policy_type", type=str, default="DQN", help="DQN, NWS, or random")
    parser.add_argument("--final_model", type=str, default="CPQ_observed-alerts_small-S_lr5e-3_20230426131737/model_390000.pt", help="file path of model to get policy from")

    args = parser.parse_args()
    main(args)
