import numpy as np
import pandas as pd
from scipy import stats
import math
from typing import Tuple
import itertools
import torch 
from torch import optim
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl 

# from heat_alerts.Q_prep_function import make_data
from Q_prep_function import make_data


class my_NN(nn.Module): # change name!
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
        self.randeff_slopes = nn.Parameter(torch.zeros(n_randeff).to("cuda"))
        self.lsigma_slopes = nn.Parameter(torch.tensor(0.0).to("cuda"))
        # self.rePrior = torch.distributions.Normal(0, F.softplus(self.lsigma))
        # self.sigmaPrior = torch.distributions.HalfCauchy(1.0)
    def forward(self, x, id):
        step1 = self.net(x)
        # print(step1)
        # print(self.randeff[id])
        # print(step1 + self.randeff[id])
        return step1

class DQN_Lightning(pl.LightningModule): # change name!
    def __init__(self, n_col, config, n_randeff, N, b_size, lr, loss="huber",  optimizer="adam", **kwargs) -> None:
        super().__init__()
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
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
        random_slopes = F.softplus(self.net.lsigma_slopes)*self.net.randeff_slopes[id]
        Preds = torch.where(a == 0, preds[:,0], preds[:,0] + preds[:,1] + random_slopes)
        Preds = Preds + F.softplus(self.net.lsigma)*self.net.randeff[id]
        Preds = -torch.exp(Preds)
        # preds = -F.softplus(self.net(s, id))
        # Preds = torch.where(a == 0, preds[:,1] + preds[:,0], preds[:,1])
        # Preds = Preds - F.softplus(self.lsigma)*self.randeff[id]#.unsqueeze(1) 
        return Preds, r
    def configure_optimizers(self):
        if self.optimizer_fn == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr = self.lr, eps=1e-4, weight_decay=self.w_decay)
        elif self.optimizer_fn == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr = self.lr, weight_decay=self.w_decay)
        return optimizer
    def prior(self):
        re = self.net.randeff
        lsig = self.net.lsigma
        loss1 = -torch.distributions.Normal(0, 1).log_prob(re)
        loss2 = -torch.distributions.HalfCauchy(1.0).log_prob(F.softplus(lsig))
        re_s = self.net.randeff_slopes
        lsig_s = self.net.lsigma_slopes
        loss3 = -torch.distributions.Normal(0, 1).log_prob(re_s)
        loss4 = -torch.distributions.HalfCauchy(1.0).log_prob(F.softplus(lsig_s))
        # print(F.softplus(lsig))
        #print(loss1.sum() + loss2)
        return loss1.sum() + loss2 + loss3.sum() + loss4
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


## Main code:

deaths_model = torch.load("Fall_results/R_1-23_deaths.pt", map_location=torch.device('cpu'))
other_hosps_model = torch.load("Fall_results/R_1-23_other-hosps.pt", map_location=torch.device('cpu'))
all_hosps_model = torch.load("Fall_results/R_1-23_all-hosps.pt", map_location=torch.device('cpu'))

## Remove dropout from evaluation of them all:
deaths_model.eval()
other_hosps_model.eval()
all_hosps_model.eval()

model = other_hosps_model

D = make_data(data_only=False)
S, A, ID = [D[k] for k in ("S", "A", "ID")]
S = S.drop("index", axis = 1)

id = torch.LongTensor(pd.DataFrame(ID).to_numpy())

history_values = np.arange(-5, 25, 1)
Diffs = []
A1_Diffs = []

for h in history_values:
    S["all_hosp_mean_rate"] = h
    s = torch.FloatTensor(S.to_numpy())
    r_hat = model.net(s,id)
    random_slopes = F.softplus(model.net.lsigma_slopes)*model.net.randeff_slopes[id]
    R_hat = r_hat
    R_hat[:,1] = R_hat[:,0] + R_hat[:,1] + random_slopes[:,0]
    random_intercepts = F.softplus(model.net.lsigma)*model.net.randeff[id]
    R_hat = R_hat + random_intercepts
    R_hat = -torch.exp(R_hat)
    n = R_hat.detach().numpy()
    df = pd.DataFrame(n)
    df.columns=["R0", "R1"]
    ## Now calculate differences, both overall and on days that A=1:
    diffs = df["R1"] - df["R0"]
    a1_diffs = diffs.where(A == 1, float('nan'))
    Diffs.append(diffs.mean(axis=0))
    A1_Diffs.append(a1_diffs.mean(axis=0))
    print(h)


## Save results:
df = pd.DataFrame([history_values, Diffs, A1_Diffs])
df.to_csv("Fall_results/Hosp_preds_vs_History.csv")