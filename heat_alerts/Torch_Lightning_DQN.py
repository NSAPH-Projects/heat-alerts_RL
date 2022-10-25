
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
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp
from heat_alerts.Q_prep_function import make_data


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
            nn.SiLU(),
            # nn.Linear(n_col**2, n_col**2),
            # nn.SiLU(),
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
    def __init__(self, n_col, n_hidden, b_size, lr, gamma, sync_rate, **kwargs) -> None: # add more params here from argparse
        super().__init__()
        self.net = DQN(n_col, n_hidden)
        self.target_net = DQN(n_col, n_hidden)
        self.b_size = b_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        loss = self.dqn_huber_loss(batch)
        self.log(
            "loss", loss, sync_dist = False,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if b_idx % self.sync_rate == 0:  # copy params over target network
            self.target_net.load_state_dict(self.net.state_dict())
        return OrderedDict({'loss': loss})


def main(params):
    # params = [256, 0.01, 0.99, 50, 100, 10, 0, 20] # remove when calling script from 
    # params = dict(...)  # better to keep names
    # command line
    
    if isinstance(params, ArgumentParser):
        params = vars(params)  # convert args to dictionary 

    ## Set up data:
    D = make_data()
    S,A,R,S_1,ep_end,over = [D[k] for k in ("S","A","R","S_1","ep_end","over")]
    data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, over]
    state_dim = S.drop("index", axis = 1).shape[1]

    # Make data loader
    tensors = [torch.from_numpy(v.to_numpy()) for v in data]
    DS = TensorDataset(*tensors)
    DL = DataLoader(
        DS, batch_size = params['b_size'], shuffle = True, 
        num_workers=(mp.cpu_count()//2), persistent_workers=True
    )

    model = DQN_Lightning(state_dim, **params)
    logger = CSVLogger("lightning_logs", name="test_logs")

    trainer = pl.Trainer(
        # gpus=params[6], # n_gpu
        distributed_backend="dp",
        limit_train_batches=params["k_size"], # k_size
        max_epochs = params["n_epochs"], # n_epochs
        logger = logger,
        accelerator="auto",
        devices="auto",
        # precision=16, amp_backend="native"
    )
    trainer.fit(model, train_dataloaders=DL)
    # Eventually, save the model and the losses for later


if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--b_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=4, help="how often to sync the target model")
    parser.add_argument("--k_size", type=int, default=100, help="how many batches per epoch")
    parser.add_argument("--print", type=int, default=10, help="progress updates on epochs")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to run")

    args = parser.parse_args()
    main(args)

# %%
