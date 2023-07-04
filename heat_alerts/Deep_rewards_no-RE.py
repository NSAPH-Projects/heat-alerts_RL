
import os
from argparse import ArgumentParser
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
from pytorch_lightning.callbacks import LearningRateFinder
from tqdm import tqdm
import multiprocessing as mp

os.chdir("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")

# from heat_alerts.Q_prep_function import make_data
from Q_prep_function import make_data

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

class Rewards_Lightning(pl.LightningModule):
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
        # Preds = torch.where(a == 0, preds[:,0], preds[:,0] - F.softplus(preds[:,1]))
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

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones
    def on_fit_start(self, *args, **kwargs):
        return
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def main(params):
    params = vars(params)

    # params = {
    #     "outcome": "other_hosps",
    #     "S_size": "medium",
    #     "eligible": "90pct",
    #     "b_size": 2048,
    #     "n_hidden": 256,
    #     "lr": 0.003,
    #     "n_gpus": 0, # 1
    #     "n_epochs": 2, # 200,
    #     "xpt_name": "testing_other-hosps",
    #     "model_name": "Summer_R_hosps_003_medium",
    #     "loss": "mse", # "huber",
    #     "silent": False,
    #     "optimizer": "adam",
    #     "n_workers": 4
    # }

    if params["outcome"] == "all_hosps":
        D = make_data(outcome="all_hosps", manual_S_size = params["S_size"], 
                    eligible = params["eligible"], all_data = True)
    elif params["outcome"] == "other_hosps":
        D = make_data(outcome="other_hosps", manual_S_size = params["S_size"], 
                    eligible = params["eligible"], all_data = True)
    else:
        D = make_data(manual_S_size = params["S_size"], 
                    eligible = params["eligible"], all_data = True)
        
    S,A,R = [D[k] for k in ("S","A","R")]
    state_dim = S.shape[1]
    R = R*1000
    N = len(D['R'])
    perm = np.random.permutation(N)  # for preshuffling
    data = [S, A, R]

    # Select validation set by episode, ensure equal representation by all counties
    n_years = 11
    n_days = 152
    all_days = n_years*n_days
    n_counties = int(N/all_days)
    years = S["year"].unique()

    val = []
    for i in range(0,n_counties):
        y_val = sample(list(years), 2)
        start = i*all_days
        end = (i+1)*all_days
        val.extend([start + x for x in np.where(S["year"][start:end] == y_val[0])][0])
        val.extend([start + x for x in np.where(S["year"][start:end] == y_val[1])][0])
        # print(i)

    # pd.DataFrame(val).to_csv("data/Python_val_set_by-county.csv")
    train = list(set(list(range(0,N))) - set(val))

    # Make data loader, shuffled:
    train_perm = np.random.permutation(len(train))
    val_perm = np.random.permutation(len(val))
    train_data = [v.to_numpy()[train][train_perm] for v in data]
    val_data = [v.to_numpy()[val][val_perm] for v in data]

    train_tensors = train_data
    val_tensors = val_data
    for j in [0, 2]: train_tensors[j] = torch.FloatTensor(train_tensors[j])

    train_tensors[1] = torch.LongTensor(train_tensors[1])
    for j in [0, 2]: val_tensors[j] = torch.FloatTensor(val_tensors[j])

    val_tensors[1] = torch.LongTensor(val_tensors[1])

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

    config = { # results from tuning
        "n_hidden": 256,
        "dropout_prob": 0.0,
        "w_decay": 1e-4
    }

    model = Rewards_Lightning(state_dim, config, N = N, **params)
    logger_name = params["xpt_name"]
    logger = CSVLogger("lightning_logs", name=logger_name)
    
    trainer = pl.Trainer(
        # distributed_backend="dp",
        # limit_train_batches=params["k_size"], # k_size
        max_epochs = params["n_epochs"], # n_epochs
        logger = logger,
        accelerator="auto",
        devices=params["n_gpus"],
        enable_progress_bar=(not params['silent']),
        # auto_lr_find = True
        # precision=16, amp_backend="native"
        callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))]
    )
    # trainer.tune(model, train_DL, val_DL)
    trainer.fit(model, train_DL, val_DL)
    
    torch.save(model, "Summer_results/" + params['model_name'] + ".pt")

    ## Save modeled rewards:
    d = make_data(filename="data/Summer23_Train_smaller-for-Python.csv", 
                  manual_S_size = params["S_size"], all_data = True)
    s = torch.FloatTensor(d["S"].to_numpy())
    model.eval() # turns off dropout for the predictions
    r_hat = model.net(s)
    R_hat = r_hat
    # R_hat[:,1] = R_hat[:,0] - F.softplus(R_hat[:,1])
    R_hat[:,1] = R_hat[:,0] + R_hat[:,1]
    R_hat = -torch.exp(R_hat)
    n = R_hat.detach().numpy()
    df = pd.DataFrame(n)
    df.to_csv("Summer_results/" + params['model_name'] + "_" + params["eligible"] + ".csv")


if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="deaths", help = "deaths or hosps")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--eligible", type=str, default="all", help="days to include in model")
    parser.add_argument("--S_size", type=str, default="medium", help="Manual size of state matrix")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
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
