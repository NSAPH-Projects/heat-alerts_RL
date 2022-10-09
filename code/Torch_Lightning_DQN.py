
# import os
from argparse import ArgumentParser
from typing import Tuple
from collections import OrderedDict
import time
import numpy as np
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
from tqdm import tqdm
from copy import deepcopy

from code.Q_prep_function import make_data


## Set up data:
D = make_data()
S,A,R,S_1,ep_end,over = [D[k] for k in ("S","A","R","S_1","ep_end","over")]
data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, over]
state_dim = S.drop("index", axis = 1).shape[1]

## Set up model:
class DQN(nn.Module):
    def __init__(self, n_col) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_col, n_col^2),
            nn.SiLU(),
            # nn.Linear(n_col**2, n_col**2),
            # nn.SiLU(),
            nn.Linear(n_col^2, 2)
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
    def __init__(self, n_col, params) -> None: # add more params here from argparse
        super().__init__()
        self.net = DQN(n_col)
        self.target_net = DQN(n_col)
        self.iter = 0
        self.b_size = params[0]
        self.lr = params[1]
        self.gamma = params[2]
        self.sync_rate = params[3]
        self.k_size = params[4]
        self.n_epochs = params[7] # params[6] is n_gpu
    def forward(self, x):
        return self.net(x)
    def dqn_huber_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        s, a, r, s1, ee, o = batch
        Q = self.net(s.float()).gather(1, a.view(-1, 1)).view(-1)
        with torch.no_grad():
            target = r + self.gamma*(1-ee.float())*eval_Q_double(self.net, self.target_net, s1.float(), o)
        return F.smooth_l1_loss(Q, target)
    def configure_optimizers(self):
        optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.25)
        return [optimizer]
    def train_dataloader(self):
        tensors = [torch.from_numpy(v.to_numpy()) for v in data]
        DS = TensorDataset(*tensors)
        DL = DataLoader(DS, batch_size = self.b_size, shuffle = True)
        return DL
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        loss = self.dqn_huber_loss(batch)
        self.iter += 1
        if self.iter % self.sync_rate*self.k_size == 0:
            self.target_net = deepcopy(self.net)
        return OrderedDict({'loss': loss})

def main(params):
    params = [256, 0.01, 0.99, 50, 100, 10, 0, 20] # remove when calling script from command line
    model = DQN_Lightning(state_dim, params)
    trainer = pl.Trainer(
        gpus=params[6], # n_gpu
        distributed_backend="dp", # not sure what to do with this?
        limit_train_batches=params[4], # k_size
        max_epochs = params[7] # n_epochs
    )
    trainer.fit(model)
    # Eventually, save the model and the losses for later

if __name__ == "__main__":
    torch.manual_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--b_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=50, help="how often to sync the target model")
    parser.add_argument("--k_size", type=int, default=100, help="how many batches per epoch")
    parser.add_argument("--print", type=int, default=10, help="progress updates on epochs")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to run")

    args = parser.parse_args()
    main(args)

# %%
