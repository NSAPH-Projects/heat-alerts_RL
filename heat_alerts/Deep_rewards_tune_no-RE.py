
#### Based tuning off of https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646

import os
from typing import Tuple
from collections import OrderedDict
import numpy as np
import random
import pandas as pd
from scipy.special import expit, softmax
from random import sample

import torch 
from torch import nn # creating modules
from torch.nn import functional as F # losses, etc.
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import optim
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
import multiprocessing as mp

os.chdir("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")

# from heat_alerts.Q_prep_function import make_data
from Q_prep_function import make_data

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Set up model:
class my_NN(nn.Module):
    def __init__(self, n_col: int, n_hidden: int, dropout_prob: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_col, n_hidden),
            nn.ELU(),
            nn.Dropout(dropout_prob), # remove using prob = 0.0
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )
    def forward(self, x):
        step1 = self.net(x)
        return step1

class DQN_Lightning(pl.LightningModule):
    def __init__(self, n_col, config, N, b_size, lr, loss="huber",  optimizer="adam", **kwargs) -> None:
        super().__init__()
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
        self.net = my_NN(n_col, config["n_hidden"], config["dropout_prob"])
        # self.target_net.eval()  # in case using layer normalization
        self.N = N
        self.b_size = b_size
        self.lr = lr
        self.training_epochs = 0
        self.w_decay = config["w_decay"]
    def make_pred_and_targets(self, batch):
        s, a, r = batch
        preds = self.net(s)
        Preds = torch.where(a == 0, preds[:,0], preds[:,0] + preds[:,1])
        Preds = -torch.exp(Preds)
        return Preds, r
    def configure_optimizers(self):
        if self.optimizer_fn == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr = self.lr, eps=1e-4, weight_decay=self.w_decay)
        elif self.optimizer_fn == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr = self.lr, weight_decay=self.w_decay)
        return optimizer
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets)
        self.log("epoch_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        bias = (preds - targets).mean()
        self.log("epoch_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    def on_train_epoch_start(self) -> None:
        self.training_epochs += 1
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx) -> None: # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        bias = (preds - targets).mean()
        self.log("val_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)


def train_model(config, params, state_dim, N, train_DL, val_DL):
    model = DQN_Lightning(state_dim, config, N = N, **params)
    metrics = {"loss": "val_loss", "bias": "val_bias"}
    
    trainer = pl.Trainer(
        max_epochs = params["n_epochs"],
        accelerator="auto",
        devices=params["n_gpus"],
        enable_progress_bar=(not params['silent']),
        auto_lr_find=True,
        callbacks=[TuneReportCallback(metrics, on="validation_end")]
    )
    trainer.tune(model, train_DL, val_DL)
    trainer.fit(model, train_DL, val_DL)
    
    # torch.save(model, "Summer_results/" + params['model_name'] + ".pt")

### Run everything:

## Set up data:
set_seed(321)
params = {
    "outcome": "other_hosps",
    "S_size": "medium",
    "b_size": 2048,
    "n_hidden": 256,
    "lr": 0.003,
    "n_gpus": 0, # 1
    "n_epochs": 2, # 200,
    "xpt_name": "tuning-hypers_other-hosps",
    "model_name": "Summer_R_hosps_003_medium",
    "loss": "mse", # "huber",
    "silent": False,
    "optimizer": "adam",
    "n_workers": 8 # 4
}

if params["outcome"] == "all_hosps":
    D = make_data(outcome="all_hosps", manual_S_size = params["S_size"])
elif params["outcome"] == "other_hosps":
    D = make_data(outcome="other_hosps", manual_S_size = params["S_size"])
else:
    D = make_data(manual_S_size = params["S_size"])

S,A,R = [D[k] for k in ("S","A","R")]

# R = 0.5 * (R - R.mean()) / np.max(np.abs(R))  # centered rewards in (-0.5, 0.5) stabilizes the Q function
# R = 0.5 * R / np.max(np.abs(R))
R = R*1000

state_dim = S.shape[1]

N = len(D['R'])
perm = np.random.permutation(N)  # for preshuffling
data = [S, A, R]

# Make data loader
tensors = [v.to_numpy()[perm] for v in data]
for j in [0, 2]: tensors[j] = torch.FloatTensor(tensors[j])

tensors[1] = torch.LongTensor(tensors[1])

random.seed(321)
train = sample(list(range(0,N)), round(0.8*N))
val = list(set(list(range(0,N))) - set(train))

train_tensors = [t[train] for t in tensors]
train_DS = TensorDataset(*train_tensors)
val_tensors = [t[val] for t in tensors]
val_DS = TensorDataset(*val_tensors)

train_DL = DataLoader(
    train_DS,
    batch_size = params['b_size'],
    num_workers=params['n_workers'],
    persistent_workers=(params['n_workers'] > 0)
)

val_DL = DataLoader(
    val_DS,
    batch_size = params['b_size'],
    num_workers=params['n_workers'],
    persistent_workers=(params['n_workers'] > 0)
)

## Set up hyperparameter tuning:

config = {
    "dropout_prob": tune.grid_search([0.0, 0.1, 0.25]), # 0.5, 0.75
    "n_hidden": tune.grid_search([32, 64, 128, 256]),
    # "n_hidden": 256,
    "w_decay": tune.grid_search([0.0, 1e-4])
    # "w_decay": 1e-4
}

trainable = tune.with_parameters(
    train_model, 
    params=params,
    state_dim=state_dim, N=N,
    train_DL=train_DL, val_DL=val_DL
)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 4,
        "gpu": params["n_gpus"]
    },
    metric = "loss",
    mode = "min",
    config = config,
    name = params["xpt_name"],
    num_samples = 1
)

print(analysis.best_config)

torch.save(analysis, "Summer_results/R_model_tuning_medium-S.pt")

# Analysis = torch.load("Fall_results/R_model_tuning_medium-S.pt")
# # dir(Analysis)
# Analysis.best_config
# Analysis.results_df