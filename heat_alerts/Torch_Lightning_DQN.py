
# import os
from argparse import ArgumentParser
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
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import optim
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp

from Q_prep_function import make_data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Set up model:
class DQN(nn.Module):
    def __init__(self, n_col: int, n_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_col, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

def eval_Q_double(Q_model, Qtgt_model, S, over = None): 
    Q = Q_model(S) # n x 2
    Qtgt = Qtgt_model(S) # n x 2
    best_action = Q.argmax(axis=1).view(-1, 1)
    best_Q = torch.gather(Qtgt, 1, best_action).view(-1) # n 
    if over is not None:
        final_Q = torch.where(over, Qtgt[:,0], best_Q) # n
        return final_Q
    return best_Q


class DQN_Lightning(pl.LightningModule):
    def __init__(self, n_col, n_hidden, b_size, lr, gamma, sync_rate, loss="huber",  optimizer="adam", momentum=0.0, **kwargs) -> None:
        super().__init__()
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
        self.momentum = momentum
        self.net = DQN(n_col, n_hidden)
        self.target_net = DQN(n_col, n_hidden)
        # self.target_net.eval()  # in case using layer normalization
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.b_size = b_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.training_epochs = 0
    def make_pred_and_targets(self, batch):
        s, a, r, s1, ee, o = batch
        preds = self.net(s).gather(1, a.view(-1, 1)).view(-1)
        with torch.no_grad():
            target = r + self.gamma * (1-ee) * self.eval_Q_double(s1, o)
        return preds, target
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
            optimizer = optim.Adam(self.net.parameters(), lr = self.lr, betas=(self.momentum, 0.9), eps=1e-4, weight_decay=1e-4)
        elif self.optimizer_fn == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum=self.momentum, weight_decay=1e-4)
        return optimizer
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets)
        self.log("epoch_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        bias = (preds - targets).mean()
        self.log("epoch_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def on_train_epoch_start(self) -> None:
        if self.training_epochs % self.sync_rate == 0:  # copy params over target network
            self.target_net.load_state_dict(self.net.state_dict())
        self.training_epochs += 1

def main(params):
    params = vars(params)
    # if isinstance(params, ArgumentParser):
    #     params = vars(params)  # convert args to dictionary 

    ## Set up data:
    if params["outcome"] == "hosps":
        D = make_data(outcome="hosps")
        S,A,R,S_1,ep_end,over,near_zero = [D[k] for k in ("S","A","R","S_1","ep_end","over","near_zero")]
        over = over or 
    else:
        D = make_data()
        S,A,R,S_1,ep_end,over,near_zero = [D[k] for k in ("S","A","R","S_1","ep_end","over","near_zero")]
    
    R = 0.5 * (R - R.mean()) / np.max(np.abs(R))  # centered rewards in (-0.5, 0.5) stabilizes the Q function
    
    if params["prob_constraint"] == True:
        over = over | [n for n in near_zero["x"]]
    
    state_dim = S.drop("index", axis = 1).shape[1]

    N = len(D['R'])
    perm = np.random.permutation(N)  # for preshuffling
    data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, over]

    # Make data loader
    tensors = [v.to_numpy()[perm] for v in data]
    for j in [0, 2, 3]: tensors[j] = torch.FloatTensor(tensors[j])
    for j in [1, 4, 5]: tensors[j] = torch.LongTensor(tensors[j])
    DS = TensorDataset(*tensors)
    DL = DataLoader(
        DS,
        batch_size = params['b_size'],
        num_workers=params['n_workers'],
        persistent_workers=(params['n_workers'] > 0)
    )

    model = DQN_Lightning(state_dim, **params)
    logger_name = params["xpt_name"]
    logger = CSVLogger("lightning_logs", name=logger_name)
    
    trainer = pl.Trainer(
        # distributed_backend="dp",
        # limit_train_batches=params["k_size"], # k_size
        max_epochs = params["n_epochs"], # n_epochs
        logger = logger,
        accelerator="auto",
        devices=params["n_gpus"],
        enable_progress_bar=(not params['silent'])
        # precision=16, amp_backend="native"
    )
    trainer.fit(model, train_dataloaders=DL)
    
    torch.save(model, "Fall_results/" + params['model_name'] + ".pt")

if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="deaths", help = "deaths or hosps")
    parser.add_argument("--prob_constraint", default=True, action="constrained by behavior model probablities?")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--mtm", type=float, default=0.0, help="momentum")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=3, help="how often (in epochs) to sync the target model")
    # parser.add_argument("--k_size", type=int, default=100, help="how many batches per epoch")
    # parser.add_argument("--print", type=int, default=10, help="progress updates on epochs")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--xpt_name", type=str, default="test", help="name for the experiment log")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--loss", type=str, default="mse", choices=("huber", "mse"))
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("sgd", "adam"))
    parser.add_argument("--n_workers", type=int, default=0, help="number of workers in the data loader")

    args = parser.parse_args()
    # print(args)
    main(args)

# %%
