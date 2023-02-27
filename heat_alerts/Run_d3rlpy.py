
from argparse import ArgumentParser
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import d3rlpy
from d3rlpy.algos import DQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

# from heat_alerts.Setup_d3rlpy import make_data
from Setup_d3rlpy import make_data

def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(params):
    params = vars(params)

    ## Prepare data:

    dataset = make_data(outcome = "other_hosps", modeled_r = False, log_r = True, random_effects = False)
    # dataset.episodes[0][0].observation
    # dataset.episodes[0][0].next_observation

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2) # uses np.random.seed

    ## Set up algorithm:

    dqn = DQN( # DoubleDQN
        use_gpu=True, 
        batch_size=2048,
        learning_rate=0.003,
        gamma=1.0,
        target_update_interval=2048*3) 
    
    dqn.build_with_dataset(dataset) # initialize neural networks

    td_error = td_error_scorer(dqn, test_episodes) # calculate metrics

    dqn.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=1,
        scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    # # return actions based on the greedy-policy
    # Action = dqn.predict([observation])[0]
    # # estimate action-values
    # Value = dqn.predict_value([observation], [action])[0]

    # # save full parameters
    # dqn.save_model('dqn.pt')
    # # load full parameters
    # dqn2 = DQN()
    # dqn2.build_with_dataset(dataset)
    # dqn2.load_model('dqn.pt')
    # # save the greedy-policy as TorchScript
    # dqn.save_policy('policy.pt')

if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="deaths", help = "deaths or hosps")
    parser.add_argument("--prob_constraint", type=bool, default=True, help="constrained by behavior model probablities?")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--mtm", type=float, default=0.0, help="momentum")
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--xpt_name", type=str, default="test", help="name for the experiment log")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--loss", type=str, default="mse", choices=("huber", "mse"))
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("sgd", "adam"))
    parser.add_argument("--n_workers", type=int, default=0, help="number of workers in the data loader")

    args = parser.parse_args()
    main(args)