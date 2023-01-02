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

from heat_alerts.Q_prep_function import make_data

## NN / Lightning setup from Deep_rewards.py:

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

def get_rewards(model, s, id, shift=0, scale=1):
    r_hat = model.net(s,id)
    R_hat = - F.softplus(r_hat)
    R_hat[0] = R_hat[1] + R_hat[0]
    final = R_hat*scale/0.5 # + shift
    n = final.detach().numpy()
    # n = R_hat.detach().numpy()
    df = pd.DataFrame(n)
    return(df)

## Main code:

deaths_model = torch.load("Fall_results/R_1-2_deaths.pt", map_location=torch.device('cpu'))
other_hosps_model = torch.load("Fall_results/R_1-2_other-hosps.pt", map_location=torch.device('cpu'))
all_hosps_model = torch.load("Fall_results/R_1-2_all-hosps.pt", map_location=torch.device('cpu'))

## Remove dropout from evaluation of them all:
deaths_model.eval()
other_hosps_model.eval()
all_hosps_model.eval()

name = "12-29_deaths" # eventually, switch to argparse?
policy = pd.read_csv("Fall_results/DQN_" + name + "_constrained_policy.csv")["policy"]

D = make_data(data_only=False)
S = D["S"]
S = S.drop("index", axis = 1)
Budget,n_seq_s,s_means,s_stds = [D[k] for k in ("Budget","n_seq_s","s_means","s_stds")]
Constraint = pd.DataFrame(Budget).drop(n_seq_s)
deaths_shift,deaths_scale,all_hosps_shift,all_hosps_scale,other_hosps_shift,other_hosps_scale = [D[k] for k in ("deaths_shift","deaths_scale","all_hosps_shift","all_hosps_scale","other_hosps_shift","other_hosps_scale")]
 
n_years = 11
n_days = 153
ID = D["ID"]
id = torch.LongTensor(pd.DataFrame(ID).to_numpy())
summer = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(S.shape[0]/(n_days-1)))]))

Deaths = np.zeros(len(ID))
All_hosps = np.zeros(len(ID))
Other_hosps = np.zeros(len(ID))
for i in range(0, max(summer)): # test with i=6 for nonzero constraint
    pos = np.where(np.array(summer) == i)
    this_id = id[pos[0][0]]
    d=0
    alerts = 0
    death_rate_sum = S.iloc[pos[0][0]]["death_mean_rate"]*s_stds["death_mean_rate"] + s_means["death_mean_rate"]
    hosp_rate_sum = S.iloc[pos[0][0]]["all_hosp_mean_rate"]*s_stds["all_hosp_mean_rate"] + s_means["all_hosp_mean_rate"]
    other_hosp_rate_sum = hosp_rate_sum - (S.iloc[pos[0][0]]["heat_hosp_mean_rate"]*s_stds["heat_hosp_mean_rate"] + s_means["heat_hosp_mean_rate"])
    while d < n_days - 2:
        new_s = S.iloc[pos[0][d]]
        ## Update alerts based on new policy:
        action = policy.iloc[pos[0][d]]
        alerts += action
        new_s["alert_sum"] = (alerts - s_means["alert_sum"])/s_stds["alert_sum"]
        new_s["More_alerts"] = (Constraint.iloc[pos[0][d]] - alerts - s_means["More_alerts"])/s_stds["More_alerts"]
        ## Update past health outcomes based on new data:
        new_s["death_mean_rate"] = (death_rate_sum/(d+1) - s_means["death_mean_rate"])/s_stds["death_mean_rate"]
        new_s["all_hosp_mean_rate"] = (hosp_rate_sum/(d+1) - s_means["all_hosp_mean_rate"])/s_stds["all_hosp_mean_rate"]
        new_s["heat_hosp_mean_rate"] = ((hosp_rate_sum - other_hosp_rate_sum)/(d+1) - s_means["heat_hosp_mean_rate"])/s_stds["heat_hosp_mean_rate"]
        v = torch.FloatTensor(new_s)
        deaths = get_rewards(deaths_model, v, this_id, deaths_shift, deaths_scale)
        all_hosps = get_rewards(all_hosps_model, v, this_id, all_hosps_shift, all_hosps_scale)
        other_hosps = get_rewards(other_hosps_model, v, this_id, other_hosps_shift, other_hosps_scale)
        death_rate_sum += deaths[0][action]
        hosp_rate_sum += all_hosps[0][action]
        other_hosp_rate_sum += other_hosps[0][action]
        ## Record estimated outcomes for OPE:
        Deaths[pos[0][d]] = deaths[0][action]
        All_hosps[pos[0][d]] = all_hosps[0][action]
        Other_hosps[pos[0][d]] = other_hosps[0][action]
        d+=1
    print(i)

DF = np.column_stack((Deaths, All_hosps, Other_hosps))
pd.DataFrame(DF).to_csv("Fall_results/Estimated_rewards_" + name + "_policy.csv")
