import glob
from argparse import ArgumentParser
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import math

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

# from heat_alerts.Single_county_setup import make_data
from Single_county_setup import make_data
# from heat_alerts.cpq import CPQ
# from cpq import CPQ

def get_steps_per_epoch(folder):
    models = sorted(glob.glob(folder + "/model_*"))
    m1 = int(models[0].split("model_")[1].split(".pt")[0])
    m2 = int(models[1].split("model_")[1].split(".pt")[0])
    return(m2 - m1)

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
    
    ## For now:
    # params = dict(
    #     fips = 4013, n_hidden = 256, n_layers = 3,
    #     n_gpus=0, b_size=32, n_epochs=3000,
    #     lr=0.001, gamma=0.999, sync_rate = 3,
    #     modeled_r = "F", random_effects = False,
    #     model_name = "test_1-fips",
    #     eligible = "90pct", S_size = "small",
    #     algo = "CPQ", std_budget = 0,
    #     )

    ## Prepare data:
    data = make_data(
        fips = params["fips"], modeled_r = params["modeled_r"], 
        eligible = params["eligible"], std_budget = params["std_budget"]
    )
    dataset = data[0]
    s_means = data[1]
    s_stds = data[2]

    ####### Q-LEARNING:

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2) # uses np.random.seed
    # iters_per_epoch = round(len(dataset.observations)*0.8/params["b_size"])
    obs = 0
    for t in train_episodes:
        obs += len(t.observations)
    iters_per_epoch = math.floor(obs/params["b_size"])

    ## Set up algorithm and NN:
    n_hidden = params["n_hidden"]
    encoder_factory = VectorEncoderFactory(hidden_units=[n_hidden]*params["n_layers"], activation='relu') # doesn't allow for 'elu'

    gpu = False
    device = "cpu"
    if params["n_gpus"] > 0: 
        gpu = True
        device = "cuda"

    if params["algo"] == "CPQ":
        with open('/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL/heat_alerts/cpq_global.py', 'w') as f:
            if params["HER"]:
                f.write('her = True \n')
            else: 
                f.write('her = False \n')
            MA_mean = s_means["More_alerts"]
            f.write('MA_mean = ' + str(MA_mean) + ' \n')
            MA_sd = s_stds["More_alerts"]
            f.write('MA_sd = ' + str(MA_sd) + ' \n')
            SA_mean = s_means["alert_sum"]
            f.write('SA_mean = ' + str(SA_mean) + ' \n')
            SA_sd = s_stds["alert_sum"]
            f.write('SA_sd = ' + str(SA_sd) + ' \n')
            if gpu:
                f.write('device = ' + "'cuda'" + ' \n')
            else: 
                f.write('device = ' + "'cpu'" + ' \n')

    from cpq import CPQ # putting this here so it includes the correct global variables

    functions = [DQN, DoubleDQN, CPQ]
    func_names = ["DQN", "DoubleDQN", "CPQ"]
    algos = dict(zip(func_names, functions))
    algo = algos[params["algo"]]

    dqn = algo(
        encoder_factory=encoder_factory,
        use_gpu=gpu, 
        batch_size=params["b_size"],
        learning_rate=params["lr"],
        gamma=params["gamma"],
        target_update_interval=iters_per_epoch*params["sync_rate"]
        ) 
    
    dqn.build_with_dataset(dataset) # initialize neural networks

    if params["continue"] != "false":
        dqn.load_model("d3rlpy_logs/" + params["continue"])

    ## Train:

    name = params["model_name"]

    dqn.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=params["n_epochs"],
        experiment_name=name,
        scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    ####### MODEL AVERAGING:

    folder = glob.glob("d3rlpy_logs/" + name + "_2*")[0]
    Total_Alerts = []
    NN_sum = torch.load(folder + "/model_" + str(iters_per_epoch) + ".pt", map_location=torch.device(device))
    NN_list = []
    NN_list.append(NN_sum)
    n_models = 1
    for i in range(1, params["n_epochs"]): 
        new = torch.load(folder + "/model_" + str(iters_per_epoch*(i+1)) + ".pt", map_location=torch.device(device))
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

    np.savetxt("Summer_results/" + name + "_MA_" + str(params["ma"]) + "_total_alerts_" + ".csv", Total_Alerts)

    ####### SEQUENTIAL POLICY EXTRACTION:

    final_model = glob.glob("d3rlpy_logs/" + name + "_2*")[0] + "/MA_" + str(params["n_epochs"]-1) + ".pt"
    dqn.load_model(final_model)

    ## Iterate through each episode (summer) for the county:
    Policy = np.zeros(len(dataset.observations))
    p = 0
    for i in range(0, len(dataset)):
        alert_sum = 0
        t_since_alert = dataset.episodes[i].observations[0][4]*s_stds["T_since_alert"] + s_means["T_since_alert"] - (dataset.episodes[i].observations[0][5]*s_stds["dos"] + s_means["dos"])
        budget = dataset.episodes[i].observations[0][6]*s_stds["More_alerts"] + s_means["More_alerts"]
        d = 0
        dos_alert = -t_since_alert
        while d < len(dataset.episodes[i].observations):
            new_s = dataset.observations[p]
            new_s[5] = (alert_sum - s_means["alert_sum"])/s_stds["alert_sum"]
            new_s[6] = (budget - alert_sum - s_means["More_alerts"])/s_stds["More_alerts"]
            dos = new_s[3]*s_stds["dos"] + s_means["dos"]
            new_s[4] = (dos - dos_alert - s_means["T_since_alert"])/s_stds["T_since_alert"]
            dataset.observations[p] = new_s
            ## Get new policy:
            output = dqn.predict(dataset.observations[p:(p+1)])
            if output == 1:
                if params["algo"] == "CPQ" and alert_sum >= budget:
                    action = 0
                else:
                    Policy[p] = 1
                    action = 1
            else: 
                action = 0
            
            if action == 1:
                alert_sum += 1
                dos_alert = dos
            d+=1
            p+=1
        print(i)

    pd.DataFrame(Policy).to_csv("Policies/Policy_" + name + ".csv")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fips", type=int, default=4013, help="fips code of the county")
    parser.add_argument("--eligible", type=str, default="all", help="days to include in RL")
    # parser.add_argument("--S_size", type=str, default="medium", help="Manual size of state matrix")
    parser.add_argument("--modeled_r", type=str, default="F", help="use modeled rewards? T or F")
    parser.add_argument("--algo", type=str, default="DQN", help="RL algorithm")
    parser.add_argument("--seed", type=int, default=321, help="set seed")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--std_budget", type=int, default=0, help="same budget for all episodes?")
    parser.add_argument("--HER", type=bool, default=False, help="Use hindsight experience replay for CPQ?")
    parser.add_argument("--b_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_layers", type=int, default=2, help="how many hidden layers in DQN")
    parser.add_argument("--n_hidden", type=int, default=32, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs to run")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--continue", type=str, default="false", help="continue fitting an existing model")
    parser.add_argument("--ma", type=int, default=50, help="number of epochs in moving average for smoothed model")

    args = parser.parse_args()
    main(args)
