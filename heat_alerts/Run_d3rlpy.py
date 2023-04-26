
import glob
from argparse import ArgumentParser
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DQN, DoubleDQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

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
    set_seed(321)
    d3rlpy.seed(321)
    params = vars(params)

    ## For now:
    # params = dict(
    #     outcome = "other_hosps", n_hidden = 256,
    #     n_gpus=1, b_size=2048, n_epochs=2,
    #     lr=0.0001, gamma=1.0, sync_rate = 3,
    #     modeled_r = False, random_effects = False,
    #     model_name = "test_cpq",
    #     eligible = "90pct",
    #     algo = "CPQ"
    #     )

    ## Prepare data:
    print(params["modeled_r"])
    print(params["random_effects"])

    dataset = make_data(
        outcome = params["outcome"], modeled_r = params["modeled_r"], 
        log_r = True, random_effects = params["random_effects"], eligible = params["eligible"],
        pca = params["pca"], pca_var_thresh = params["pca_var_thresh"], manual_S_size = params["S_size"]
    )

    # dataset.episodes[0][0].observation
    # dataset.episodes[0][0].next_observation

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2) # uses np.random.seed

    ## Set up algorithm:

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
    
    dqn.build_with_dataset(dataset) # initialize neural networks

    ## Train:
    
    if params["continue"] != "false":
        dqn.load_model("d3rlpy_logs/" + params["continue"])
        # dqn.load_model("d3rlpy_logs/vanilla_DQN_lr1e-3sr10_modeled-R_20230408124037" + "/model_" + str(39*20000) + ".pt")

    dqn.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=params["n_epochs"],
        experiment_name=params["model_name"],
        scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    ## Check how many alerts are being sent:
    folder = glob.glob("d3rlpy_logs/" + params["model_name"] + "_2*")[0]
    dqn2 = algo(
        encoder_factory=encoder_factory,
        use_gpu=gpu, 
        batch_size=params["b_size"],
        learning_rate=params["lr"],
        gamma=params["gamma"],
        target_update_interval=params["b_size"]*params["sync_rate"]
        )  
    dqn2.build_with_dataset(dataset)
    steps_per_epoch = get_steps_per_epoch(folder)
    # steps_per_epoch=int(np.trunc(len(train_episodes)*len(dataset.episodes[0])/params["b_size"]))
    Total_Alerts = []
    for i in range(0, params["n_epochs"]): # params["n_epochs"]
        dqn2.load_model(folder + "/model_" + str(steps_per_epoch*(i+1)) + ".pt")
        a = sum(dqn2.predict(dataset.observations))
        Total_Alerts.append(a)
        print(i)
    
    # pd.DataFrame(Total_Alerts).to_csv("Fall_results/Total_alerts_" + params['model_name'] + ".csv")
    np.savetxt("Fall_results/Total_alerts_" + params['model_name'] + ".csv", Total_Alerts)
    
    # ## Save final results for easy access:
    # Action = dqn.predict(dataset.observations) # return actions based on the greedy-policy
    # Value_0 = dqn.predict_value(dataset.observations, np.repeat(0,len(Action))) # estimate action-values
    # Value_1 = dqn.predict_value(dataset.observations, np.repeat(1,len(Action)))
    # Value = np.vstack((Value_0,Value_1)).transpose()
    # pd.DataFrame(Value).to_csv("Fall_results/" + params['model_name'] + ".csv")

    ## Note: logger saves model parameters automatically
    # dqn.save_model("Fall_results/" + params['model_name'] + ".pt") # save full parameters

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="other_hosps", help = "deaths or hosps")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--modeled_r", type=bool, default=False, help="use modeled rewards?")
    parser.add_argument("--random_effects", type=bool, default=False, help="use random effects from modeled rewards?")
    parser.add_argument("--eligible", type=str, default="all", help="days to include in RL")
    parser.add_argument("--pca", type=bool, default=False, help="perform PCA?")
    parser.add_argument("--pca_var_thresh", type=float, default=0.5, help="PCA variance threshold")
    parser.add_argument("--S_size", type=str, default="medium", help="Manual size of state matrix")
    parser.add_argument("--algo", type=str, default="DQN", help="RL algorithm")
    parser.add_argument("--continue", type=str, default="false", help="continue fitting an existing model")

    args = parser.parse_args()
    main(args)
