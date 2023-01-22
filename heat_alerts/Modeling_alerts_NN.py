
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

# from heat_alerts.Q_prep_function import make_data
from Q_prep_function import make_data

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


def main(params):
    params = vars(params)
    # if isinstance(params, ArgumentParser):
    #     params = vars(params)  # convert args to dictionary 

    ## Set up data:
    
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

    # nm = NearMiss(sampling_strategy=0.5)

    # S_train_miss, A_train_miss = nm.fit_resample(train_data[0], train_data[1].ravel())

    # train_tensors = [torch.FloatTensor(S_train_miss), torch.LongTensor(A_train_miss)]
    train_tensors = [torch.FloatTensor(train_data[0]), torch.LongTensor(train_data[1])]
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

    config = { # results from tuning
        "dropout_prob": 0.5,
        "n_hidden": 256,
        "w_decay": 0.0
    }

    model = Logit_Lightning(state_dim, config, **params)
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
        auto_lr_find=True
        # precision=16, amp_backend="native"
    )
    trainer.tune(model, train_DL, val_DL)
    trainer.fit(model, train_DL, val_DL)
    
    torch.save(model, "Fall_results/" + params['model_name'] + ".pt")

    # model = torch.load("Fall_results/Alerts_model_1-20.pt", map_location=torch.device('cpu'))
    
    s = torch.FloatTensor(S.drop("index", axis = 1).to_numpy())
    model.eval() # turns off dropout for the predictions
    a_hat = model.net(s)
    A_hat = torch.exp(a_hat) # gives odds
    # A_hat = F.softplus(a_hat) # gives truncated odds
    A_hat_prob = A_hat / (1 + A_hat) # gives probabilities
    a = A_hat_prob.detach().numpy()
    df = pd.DataFrame(a)
    df.to_csv("Fall_results/" + params['model_name'] + ".csv")

if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--xpt_name", type=str, default="test", help="name for the experiment log")
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--silent", default=False, action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("sgd", "adam"))
    parser.add_argument("--n_workers", type=int, default=0, help="number of workers in the data loader")

    args = parser.parse_args()
    # print(args)
    main(args)
