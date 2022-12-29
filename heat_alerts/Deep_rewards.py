
from argparse import ArgumentParser
from typing import Tuple
from collections import OrderedDict
import time
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

from Q_prep_function import make_data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Set up model:
class my_NN(nn.Module):
    def __init__(self, n_col: int, n_hidden: int, dropout_prob: float, n_randeff: int) -> None:
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
        self.randeff = nn.Parameter(torch.zeros(n_randeff).to("cuda"))
        self.lsigma = nn.Parameter(torch.tensor(0.0).to("cuda"))
        # self.rePrior = torch.distributions.Normal(0, F.softplus(self.lsigma))
        # self.sigmaPrior = torch.distributions.HalfCauchy(1.0)
    def forward(self, x, id):
        step1 = self.net(x)
        # print(step1)
        # print(self.randeff[id])
        # print(step1 + self.randeff[id])
        return step1 + F.softplus(self.lsigma)*self.randeff[id]#.unsqueeze(1) 

class DQN_Lightning(pl.LightningModule):
    def __init__(self, n_col, config, n_randeff, N, b_size, lr, loss="huber",  optimizer="adam", momentum=0.0, **kwargs) -> None:
        super().__init__()
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
        self.momentum = momentum
        self.net = my_NN(n_col, config["n_hidden"], config["dropout_prob"], n_randeff)
        # self.target_net.eval()  # in case using layer normalization
        self.N = N
        self.b_size = b_size
        self.lr = lr
        self.training_epochs = 0
        self.w_decay = config["w_decay"]
    def make_pred_and_targets(self, batch):
        s, a, r, s1, ee, o, id = batch
        # preds = self.net(s).gather(1, a.view(-1, 1)).view(-1)
        preds = self.net(s, id)
        Preds = torch.where(a == 0, preds[:,1] - F.softplus(preds[:,0]), preds[:,1])
        return Preds, r
    def configure_optimizers(self):
        if self.optimizer_fn == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr = self.lr, betas=(self.momentum, 0.9), eps=1e-4, weight_decay=self.w_decay)
        elif self.optimizer_fn == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum=self.momentum, weight_decay=self.w_decay)
        return optimizer
    def prior(self):
        re = self.net.randeff
        lsig = self.net.lsigma
        loss1 = -torch.distributions.Normal(0, 1).log_prob(re)
        loss2 = -torch.distributions.HalfCauchy(1.0).log_prob(F.softplus(lsig))
        # print(F.softplus(lsig))
        #print(loss1.sum() + loss2)
        return loss1.sum() + loss2
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets) + (1/self.N)*self.prior()
        self.log("epoch_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        bias = (preds - targets).mean()
        self.log("epoch_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    def on_train_epoch_start(self) -> None:
        self.training_epochs += 1
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx) -> None: # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets) + (1/self.N)*self.prior()
        self.log("val_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        bias = (preds - targets).mean()
        self.log("val_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=False, logger=True)


def main(params):
    params = vars(params)
    # if isinstance(params, ArgumentParser):
    #     params = vars(params)  # convert args to dictionary 

    ## Set up data:
    if params["outcome"] == "all_hosps":
        D = make_data(outcome="all_hosps")
    elif params["outcome"] == "other_hosps":
        D = make_data(outcome="other_hosps")
    else:
        D = make_data()
    
    S,A,R,S_1,ep_end,over,near_zero,ID = [D[k] for k in ("S","A","R","S_1","ep_end","over","near_zero","ID")]
    
    R = 0.5 * (R - R.mean()) / np.max(np.abs(R))  # centered rewards in (-0.5, 0.5) stabilizes the Q function
    
    state_dim = S.drop("index", axis = 1).shape[1]

    N = len(D['R'])
    perm = np.random.permutation(N)  # for preshuffling
    data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, over, pd.DataFrame(ID)]

    # Make data loader
    tensors = [v.to_numpy()[perm] for v in data]
    for j in [0, 2, 3]: tensors[j] = torch.FloatTensor(tensors[j])
    for j in [1, 4, 5, 6]: tensors[j] = torch.LongTensor(tensors[j])
    
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

    config = { # results from tuning
        "dropout_prob": 0.1,
        "n_hidden": 256,
        "w_decay": 1e-4
    }

    model = DQN_Lightning(state_dim, config, n_randeff = len(np.unique(ID)), N = N, **params)
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
    trainer.fit(model, train_DL, val_DL)
    
    torch.save(model, "Fall_results/" + params['model_name'] + ".pt")

    # model = torch.load("Fall_results/R_11-27_deaths.pt", map_location=torch.device('cpu'))
    
    s = torch.FloatTensor(S.drop("index", axis = 1).to_numpy())
    id = torch.LongTensor(pd.DataFrame(ID).to_numpy())
    model.eval() # turns off dropout for the predictions
    r_hat = model.net(s,id)
    R_hat = r_hat
    R_hat[:,0] = r_hat[:,1] - torch.exp(r_hat[:,0])
    R = D["R"]
    final = R_hat*np.max(np.abs(R))/0.5 + R.mean()
    n = final.detach().numpy()
    df = pd.DataFrame(n)
    df.to_csv("Fall_results/" + params['model_name'] + ".csv")

if __name__ == "__main__":
    set_seed(321)
    parser = ArgumentParser()
    parser.add_argument("--outcome", type=str, default="deaths", help = "deaths or hosps")
    parser.add_argument("--b_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of params in DQN hidden layers")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--mtm", type=float, default=0.0, help="momentum")
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

