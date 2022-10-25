
# import os
import argparse
from typing import Tuple
from collections import OrderedDict
import time
import numpy as np
import random
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
from scipy.special import expit, softmax

#%%

import torch 
from torch import nn # creating modules
from torch.nn import functional as F # losses, etc.
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp
from heat_alerts.Q_prep_function import make_data
from torch.utils.data import Dataset


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Set up model:
class FFN(nn.Module):
    def __init__(self, n_col: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(n_col),
            nn.Linear(n_col, n_hidden),
            nn.ELU(),
            # nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            # nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            # nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_out)
        )
    def forward(self, x):
        return self.net(x)


class DQN_Lightning(pl.LightningModule):
    def __init__(self, n_col, n_hidden, b_size, lr, gamma, sync_rate, loss="huber",  optimizer="adam", **kwargs) -> None: # add more params here from argparse
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
        self.net = FFN(n_col, n_hidden, 2)
        self.target_net = FFN(n_col, n_hidden, 2)
        self.reward_net = FFN(n_col, n_hidden, 2)
        self.target_net.eval()  # in case using layer normalization
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.b_size = b_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.training_epochs = 0

    def make_pred_and_targets(self, s, a, r, s1, ee, o):
        preds = self.net(s).gather(1, a.view(-1, 1)).view(-1)
        rhat = self.reward_net(s).gather(1, a.view(-1, 1)).view(-1)
        with torch.no_grad():
            target = rhat + self.gamma * (1-ee) * self.eval_Q_double(s1, o)
        return preds, target, rhat

    def eval_Q_double(self, S1, over = None): 
        Q = self.net(S1)
        Qtgt = self.target_net(S1)
        best_action = Q.argmax(axis=1)
        if over is not None:
            best_action = best_action * (1 - over)
        best_Q = torch.gather(Qtgt, 1, best_action.view(-1, 1)).view(-1) 
        return best_Q

    def configure_optimizers(self):
        if self.optimizer_fn == "adam":
            optimizer0 = optim.Adam(self.net.parameters(), lr = self.lr, betas=(0.9, 0.99), weight_decay=1e-6)
            optimizer1 = optim.Adam(self.reward_net.parameters(), lr = self.lr, betas=(0.9, 0.99), weight_decay=1e-6)
        elif self.optimizer_fn == "sgd":
            optimizer0 = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.5, weight_decay=1e-6)
            optimizer1 = optim.SGD(self.reward_net.parameters(), lr = self.lr, momentum=0.5, weight_decay=1e-6)
        return [optimizer0, optimizer1]

    def training_step(self, batch, b_idx, optimizer_idx): # latter is batch index
        s, a, r, s1, ee, o  = batch
        preds, targets, rhat = self.make_pred_and_targets(s, a, r, s1, ee, o)
        if optimizer_idx == 0:
            lossq = self.loss_fn(preds, targets)
            self.log("ep_loss_q", lossq, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            bias = (preds - targets).mean()
            self.log("ep_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return lossq
        elif optimizer_idx == 1:
            lossr = self.loss_fn(rhat, r)
            self.log("ep_loss_r", lossr, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            r2 = (r - rhat).pow(2).sum() / (r - r.mean()).pow(2).sum()
            self.log("r2", 1 - r2, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return lossr

    def on_train_epoch_start(self) -> None:
        if self.training_epochs % self.sync_rate == 0:  # copy params over target network
            self.target_net.load_state_dict(self.net.state_dict())
        self.training_epochs += 1

def main(params):
    # params = [256, 0.01, 0.99, 50, 100, 10, 0, 20] # remove when calling script from 
    # params = dict(...)  # better to keep names
    # command line
    
    if isinstance(params, argparse.Namespace):
        params = vars(params)  # convert args to dictionary 

    ## Set up data:
    D = make_data()    
    S,A,R,S_1,ep_end,over = [D[k] for k in ("S","A","R","S_1","ep_end","over")]
    R = 0.5 * (R - R.mean()) / np.max(np.abs(R)) # tranformation greatly helps with convergence

    state_dim = S.drop("index", axis = 1).shape[1]

    # Make data loader
    N = len(D['R'])
    perm = np.random.permutation(N)  # for preshuffling
    data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, over]
    tensors = [v.to_numpy()[perm] for v in data]

    # get the right data types
    for j in [0, 2, 3, 4]: tensors[j] = torch.FloatTensor(tensors[j])
    for j in [1, 5]: tensors[j] = torch.LongTensor(tensors[j])

    DS = TensorDataset(*tensors)
    DL = DataLoader(
        DS,
        batch_size = params['b_size'],
        num_workers=params['n_workers'],
        persistent_workers=(params['n_workers'] > 0)
    )

    model = DQN_Lightning(state_dim, **params)
    logger_name = params["experiment_name"]
    logger = CSVLogger("lightning_logs", name=logger_name)

    trainer = pl.Trainer(
        # gpus=params[6], # n_gpu
        strategy="dp",
        # limit_train_batches=params["k_size"], # k_size
        max_epochs = params["n_epochs"], # n_epochs
        logger = logger,
        accelerator="auto",
        devices=params['n_gpus'],
        # gradient_clip_val=100.0,
        # gradient_clip_algorithm="norm",
        enable_progress_bar=(not params['silent'])
        # precision=16, amp_backend="native"
    )
    trainer.fit(model, train_dataloaders=DL)
    # Eventually, save the model and the losses for later


if __name__ == "__main__":
    set_seed(321)
    parser = argparse.ArgumentParser()
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--experiment_name", type=str, default="test", help="name for the experiment logs")
    parser.add_argument("--loss", type=str, default="mse", choices=("huber", "mse"))
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("sgd", "adam"))
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (epochs) to sync the target model")
    parser.add_argument("--print", type=int, default=10, help="progress updates on epochs")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_workers", type=int, default=0, help="number of workers in the data loader")
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs to run")

    args = parser.parse_args()
    print(args)
    main(args)

# %%
