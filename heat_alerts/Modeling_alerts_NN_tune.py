#### Based tuning off of https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646

import os
from argparse import ArgumentParser
from typing import Tuple
from collections import OrderedDict
import time
import numpy as np
import random
import pandas as pd
from scipy.special import expit, softmax
from random import sample
from imblearn.under_sampling import NearMiss

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
class NN_logit(nn.Module):
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
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x):
        return self.net(x) 

class Logit_Lightning(pl.LightningModule):
    def __init__(self, n_col, config, b_size, lr,  optimizer="adam", **kwargs) -> None:
        super().__init__()
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.binary_cross_entropy_with_logits
        self.optimizer_fn = optimizer
        self.net = NN_logit(n_col, config["n_hidden"], config["dropout_prob"])
        # self.target_net.eval()  # in case using layer normalization
        self.b_size = b_size
        self.lr = lr
        self.training_epochs = 0
        self.w_decay = config["w_decay"]
    def make_pred_and_targets(self, batch):
        s, a = batch
        preds = self.net(s)[:,0]
        return preds, a.float()
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
        # bias = (preds - targets).mean()
        # self.log("epoch_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    def on_train_epoch_start(self) -> None:
        self.training_epochs += 1
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx) -> None: # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # bias = (preds - targets).mean()
        # self.log("val_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)

def train_model(config, params, state_dim, train_DL, val_DL):
    model = Logit_Lightning(state_dim, config, **params)
    metrics = {"loss": "val_loss"}
    
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
    
    # torch.save(model, "Fall_results/" + params['model_name'] + ".pt")


### Run everything:

## Set up data:
set_seed(321)
params = {
    "b_size": 2048,
    "n_hidden": 256,
    "lr": 0.003,
    "n_gpus": 1,
    "n_epochs": 2, # 200,
    "xpt_name": "tuning-hypers_alerts_model",
    "model_name": "alerts_sgd_003",
    "silent": False,
    "optimizer": "adam",
    "n_workers": 3 # 4
}

D = make_data()

S,A = [D[k] for k in ("S","A")]

state_dim = S.drop("index", axis = 1).shape[1]

N = len(D['A'])
perm = np.random.permutation(N)  # for preshuffling
data = [S.drop("index", axis = 1), A]

# Make data loader, re-balanced
shuffled = [v.to_numpy()[perm] for v in data]
train = sample(list(range(0,N)), round(0.8*N))
val = list(set(list(range(0,N))) - set(train))
train_data = [d[train] for d in shuffled]
val_data = [d[val] for d in shuffled]

nm = NearMiss()

S_train_miss, A_train_miss = nm.fit_resample(train_data[0], train_data[1].ravel())

train_tensors = [torch.FloatTensor(S_train_miss), torch.LongTensor(A_train_miss)]
val_tensors = [torch.FloatTensor(val_data[0]), torch.LongTensor(val_data[1])]

train_DS = TensorDataset(*train_tensors)
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
    # "dropout_prob": tune.grid_search([0.0, 0.1, 0.25, 0.5, 0.75]),
    "dropout_prob": tune.grid_search([0.0, 0.1, 0.25]),
    "n_hidden": tune.grid_search([32, 64, 128, 256]),
    # "n_hidden": 256,
    # "w_decay": tune.grid_search([1e-3, 1e-4, 1e-5])
    "w_decay": 1e-4
}

trainable = tune.with_parameters(
    train_model, 
    params=params,
    state_dim=state_dim,
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

torch.save(analysis, "Fall_results/Alerts_model_tuning_DP-NH.pt")

# Analysis = torch.load("Fall_results/R_model_tuning_DP-NH-WD.pt")
# # dir(Analysis)
# Analysis.best_config
# Analysis.results_df