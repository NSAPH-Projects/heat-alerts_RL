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


## NN / Lightning setup from Deep_rewards.py:

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


def get_rewards(model, s, id, shift=0, scale=1):
    r_hat = model.net(s,id)
    random_slopes = F.softplus(model.net.lsigma_slopes)*model.net.randeff_slopes[id]
    R_hat = r_hat
    R_hat[1] = R_hat[0] + R_hat[1] + random_slopes
    R_hat = R_hat + F.softplus(model.net.lsigma)*model.net.randeff[id]
    R_hat = -torch.exp(R_hat)
    n = R_hat.detach().numpy()
    df = pd.DataFrame(n)
    return(df)

def get_alert_prob(model, s):
    a_hat = model.net(s)
    A_hat = torch.exp(a_hat) # gives odds
    A_hat_prob = A_hat / (1 + A_hat) # gives probabilities
    a = A_hat_prob.detach().numpy()
    df = pd.DataFrame(a)
    return(df)

## Main code:

deaths_model = torch.load("Fall_results/R_1-23_deaths.pt", map_location=torch.device('cpu'))
other_hosps_model = torch.load("Fall_results/R_1-23_other-hosps.pt", map_location=torch.device('cpu'))
all_hosps_model = torch.load("Fall_results/R_1-23_all-hosps.pt", map_location=torch.device('cpu'))

alerts_model = torch.load("Fall_results/Alerts_model_1-23.pt", map_location=torch.device('cpu'))

## Remove dropout from evaluation of them all:
deaths_model.eval()
other_hosps_model.eval()
all_hosps_model.eval()
alerts_model.eval()

D = make_data(data_only=False)
S = D["S"]
S = S.drop("index", axis = 1)
A, R_deaths, R_all_hosps, R_other_hosps = [D[k] for k in ("A","R_deaths","R_all_hosps","R_other_hosps")]
Budget,n_seq_s,s_means,s_stds = [D[k] for k in ("Budget","n_seq_s","s_means","s_stds")]
Constraint = pd.DataFrame(Budget).drop(n_seq_s)
deaths_shift,deaths_scale,all_hosps_shift,all_hosps_scale,other_hosps_shift,other_hosps_scale = [D[k] for k in ("deaths_shift","deaths_scale","all_hosps_shift","all_hosps_scale","other_hosps_shift","other_hosps_scale")]

n_years = 11
n_days = 153
ID = D["ID"]
id = torch.LongTensor(pd.DataFrame(ID).to_numpy())
summer = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(S.shape[0]/(n_days-1)))]))

# name = "12-29_deaths" # eventually, switch to argparse?
# policy = pd.read_csv("Fall_results/DQN_" + name + "_constrained_policy.csv")["policy"]
# policy = A
# name = "NWS_hosps-only"
policy = pd.DataFrame(np.zeros(len(ID)))
name = "No_alerts_hosps-only-b"
dqn = "get_new_post_run"
DQN = False

Policy = np.zeros(len(ID))
Deaths = np.zeros(len(ID))
All_hosps = np.zeros(len(ID))
Other_hosps = np.zeros(len(ID))
for i in range(0, max(summer)): # test with i=6 for nonzero constraint
    pos = np.where(np.array(summer) == i)
    this_id = id[pos[0][0]]
    d = 0
    alerts = 0
    ## Get past health outcomes at start
    death_rate_sum = S.iloc[pos[0][0]]["death_mean_rate"]*s_stds["death_mean_rate"] + s_means["death_mean_rate"]
    hosp_rate_sum = S.iloc[pos[0][0]]["all_hosp_mean_rate"]*s_stds["all_hosp_mean_rate"] + s_means["all_hosp_mean_rate"]
    other_hosp_rate_sum = hosp_rate_sum - (S.iloc[pos[0][0]]["heat_hosp_mean_rate"]*s_stds["heat_hosp_mean_rate"] + s_means["heat_hosp_mean_rate"])
    while d < n_days - 2:
        p = pos[0][d]
        new_s = S.iloc[p]
        ## Estimate health outcomes using models and observed history:
        v0 = torch.FloatTensor(new_s)
        deaths_0 = get_rewards(deaths_model, v0, this_id, deaths_shift, deaths_scale)
        all_hosps_0 = get_rewards(all_hosps_model, v0, this_id, all_hosps_shift, all_hosps_scale)
        other_hosps_0 = get_rewards(other_hosps_model, v0, this_id, other_hosps_shift, other_hosps_scale)
        ## Update alert-related features based on past actions:
        new_s["alert_sum"] = (alerts - s_means["alert_sum"])/s_stds["alert_sum"]
        new_s["More_alerts"] = (Constraint.iloc[p] - alerts - s_means["More_alerts"])/s_stds["More_alerts"]
        ## Update past health outcomes based on new data:
        # new_s["death_mean_rate"] = (death_rate_sum/(d+1) - s_means["death_mean_rate"])/s_stds["death_mean_rate"]
        new_s["all_hosp_mean_rate"] = (hosp_rate_sum/(d+1) - s_means["all_hosp_mean_rate"])/s_stds["all_hosp_mean_rate"]
        new_s["heat_hosp_mean_rate"] = ((hosp_rate_sum - other_hosp_rate_sum)/(d+1) - s_means["heat_hosp_mean_rate"])/s_stds["heat_hosp_mean_rate"]
        v1 = torch.FloatTensor(new_s)
        ## Get new policy:
        alert_prob = get_alert_prob(alerts_model, v1.float())
        if (alert_prob >= 0.01).bool() and (alerts < Constraint.iloc[p]).bool():
            if DQN == True:
                output = dqn.net(v1.float()).detach().numpy()
                if output[1] > output[0]:
                    Policy[p] = 1
                    action = 1
            else:
                action = policy.iloc[p]
        else:
            action = 0
        ## Update alerts based on new policy:
        alerts += action
        ## Estimate health outcomes using models and new history:
        deaths_1 = get_rewards(deaths_model, v1, this_id, deaths_shift, deaths_scale)
        all_hosps_1 = get_rewards(all_hosps_model, v1, this_id, all_hosps_shift, all_hosps_scale)
        other_hosps_1 = get_rewards(other_hosps_model, v1, this_id, other_hosps_shift, other_hosps_scale)
        ## Adjust observed health outcomes:
        deaths = R_deaths.iloc[p]*1000 # - deaths_0[0][A.iloc[p]] + deaths_1[0][action]
        all_hosps = R_all_hosps.iloc[p]*1000 - all_hosps_0[0][A.iloc[p]] + all_hosps_1[0][action]
        other_hosps = R_other_hosps.iloc[p]*1000 - other_hosps_0[0][A.iloc[p]] + other_hosps_1[0][action]
        ## Record estimated outcomes for OPE:
        Deaths[p] = deaths
        All_hosps[p] = all_hosps
        Other_hosps[p] = other_hosps
        ## Update rolling means of health outcomes:
        death_rate_sum += deaths_1[0][action]/1000
        hosp_rate_sum += all_hosps_1[0][action]/1000
        other_hosp_rate_sum += other_hosps_1[0][action]/1000
        d+=1
    print(i)

DF = np.column_stack((Deaths, All_hosps, Other_hosps))
pd.DataFrame(DF).to_csv("Fall_results/Estimated_rewards_" + name + "_policy.csv")
if DQN == True:
    pd.DataFrame(Policy).to_csv("Fall_results/New_policy_" + name + "_policy.csv")

