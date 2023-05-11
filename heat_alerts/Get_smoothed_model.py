
import glob
from argparse import ArgumentParser
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN
from d3rlpy.dataset import TransitionMiniBatch

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

def get_steps_per_epoch(folder):
    models = sorted(glob.glob(folder + "/model_*"))
    m1 = int(models[0].split("model_")[1].split(".pt")[0])
    m2 = int(models[1].split("model_")[1].split(".pt")[0])
    return(m2 - m1)

def main(params):

    d3rlpy.seed(321)
    params = vars(params)

    data = make_data(
        eligible = "90pct", manual_S_size = "small"
        )
    dataset = data[0]
    # dataset.next_observations = np.expand_dims(dataset.episodes[0][0].next_observation, axis=0)
    # for e in dataset.episodes[1:]:
    #     for d in range(0, len(e)):
    #         dataset.next_observations = np.append(dataset.next_observations, np.expand_dims(e[d].next_observation, axis=0), axis=0)
    # Dataset = [torch.FloatTensor(d) for d in Dataset] # doesn't work

    ## Prepare the NN architecture:
    n_hidden = params["n_hidden"]
    encoder_factory = VectorEncoderFactory(hidden_units=[n_hidden]*3, activation='relu') # doesn't allow for 'elu'

    gpu = False
    device = "cpu"
    if params["n_gpus"] > 0: 
        gpu = True
        device = "cuda"

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

    folder = glob.glob("d3rlpy_logs/" + params["model_name"] + "_2*")[0]
    steps_per_epoch = get_steps_per_epoch(folder)
    Total_Alerts = []
    NN_sum = torch.load(folder + "/model_" + str(steps_per_epoch) + ".pt", map_location=torch.device(device))
    NN_list = []
    NN_list.append(NN_sum)
    n_models = 1
    for i in range(1, params["n_epochs"]): # params["n_epochs"]
        new = torch.load(folder + "/model_" + str(steps_per_epoch*(i+1)) + ".pt", map_location=torch.device(device))
        NN_list.append(new)
        n_models += 1
        for key in NN_sum["_q_func"]:
            if n_models <= params["ma"]:
                NN_sum["_q_func"][key] = NN_sum["_q_func"][key] + new["_q_func"][key]
                new["_q_func"][key] = NN_sum["_q_func"][key] / n_models
            else:
                NN_sum["_q_func"][key] = NN_sum["_q_func"][key] + new["_q_func"][key] - NN_list[i-params["ma"]]["_q_func"][key]
                new["_q_func"][key] = NN_sum["_q_func"][key] / params["ma"]
        for key in NN_sum["_targ_q_func"]:
            if n_models <= params["ma"]:
                NN_sum["_targ_q_func"][key] = NN_sum["_targ_q_func"][key] + new["_targ_q_func"][key]
                new["_targ_q_func"][key] = NN_sum["_targ_q_func"][key] / n_models
            else:
                NN_sum["_targ_q_func"][key] = NN_sum["_targ_q_func"][key] + new["_targ_q_func"][key] - NN_list[i-params["ma"]]["_targ_q_func"][key]
                new["_targ_q_func"][key] = NN_sum["_targ_q_func"][key] / params["ma"]
        torch.save(new, folder + "/MA_" + str(i) + ".pt")
        dqn.load_model(folder + "/MA_" + str(i) + ".pt")
        a = sum(dqn.predict(dataset.observations))
        Total_Alerts.append(a)
        print(i)

    np.savetxt("Fall_results/MA_" + str(params["ma"]) + "_total_alerts_" + params['model_name'] + ".csv", Total_Alerts)

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
    parser.add_argument("--ma", type=int, default=200, help="number of epochs in moving average")

    args = parser.parse_args()
    main(args)
