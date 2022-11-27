import numpy as np
import pandas as pd
import math
from typing import Tuple
import itertools
import torch 
from torch import optim
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl 

from heat_alerts.Q_prep_function import make_data

## Set up model:
class DQN(nn.Module):
    def __init__(self, n_col: int, n_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_col, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

class DQN_Lightning(pl.LightningModule):
    def __init__(self, n_col, n_hidden, b_size, lr, gamma, sync_rate, loss="huber",  optimizer="adam", momentum=0.0, **kwargs) -> None:
        super().__init__()
        assert loss in ("huber", "mse")
        assert optimizer in ("adam", "sgd")
        self.save_hyperparameters()
        self.loss_fn = F.smooth_l1_loss if loss=="huber" else F.mse_loss
        self.optimizer_fn = optimizer
        self.momentum = momentum
        self.net = DQN(n_col, n_hidden)
        self.target_net = DQN(n_col, n_hidden)
        # self.target_net.eval()  # in case using layer normalization
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.b_size = b_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.training_epochs = 0
    def make_pred_and_targets(self, batch):
        s, a, r, s1, ee, o = batch
        preds = self.net(s).gather(1, a.view(-1, 1)).view(-1)
        with torch.no_grad():
            target = r + self.gamma * (1-ee) * self.eval_Q_double(s1, o)
        return preds, target
    def eval_Q_double(self, S1, over = None): 
        Q = self.net(S1)
        Qtgt = self.target_net(S1)
        # best_action = Q.argmax(axis=1)
        best_action = torch.gt(torch.exp(Q[:,1]), 0)
        if over is not None:
            best_action = torch.tensor(best_action * (1 - over))
        # best_Q = torch.gather(Qtgt, 1, best_action.view(-1, 1)).view(-1) 
        best_Q = torch.where(best_action == 1, Qtgt[:,0] + torch.exp(Qtgt[:,1]), Qtgt[:,0])
        return best_Q
    def configure_optimizers(self):
        if self.optimizer_fn == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr = self.lr, betas=(self.momentum, 0.9), eps=1e-4, weight_decay=1e-4)
        elif self.optimizer_fn == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum=self.momentum, weight_decay=1e-4)
        return optimizer
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], b_idx): # latter is batch index
        preds, targets = self.make_pred_and_targets(batch)
        loss = self.loss_fn(preds, targets)
        self.log("epoch_loss", loss, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        bias = (preds - targets).mean()
        self.log("epoch_bias", bias, sync_dist = False, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def on_train_epoch_start(self) -> None:
        if self.training_epochs % self.sync_rate == 0:  # copy params over target network
            self.target_net.load_state_dict(self.net.state_dict())
        self.training_epochs += 1


### Prep data:
D = make_data(data_only=False)
S = D["S"]
S = S.drop("index", axis = 1)

prob_constraint = True # change this as needed

if prob_constraint:
    near_zero = D["near_zero"]["x"]


n_years = 11
n_days = 153
ID = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(S.shape[0]/(n_days-1)))]))

Budget,n_seq_s,s_means,s_stds = [D[k] for k in ("Budget","n_seq_s","s_means","s_stds")]
Constraint = pd.DataFrame(Budget).drop(n_seq_s)

### Look at results from model:
# pred_model = torch.load("Fall_results/DQN_11-5_deaths_constrained.pt", map_location=torch.device('cpu'))
pred_model = torch.load("Fall_results/DQN_11-5_hosps_constrained.pt", map_location=torch.device('cpu'))

new_alerts = np.zeros(len(ID))
policy = np.zeros(len(ID))
for i in range(0, max(ID)):
    pos = np.where(np.array(ID) == i)
    d = 0
    while (d < n_days - 2) & (new_alerts[pos[0][d]] < Constraint.iloc[pos[0][d]]).all():
        if prob_constraint == False:
            alerts_scaled = (new_alerts[pos[0][d]] - s_means["alert_sum"])/s_stds["alert_sum"]
            more_scaled = (Constraint.iloc[pos[0][d]] - new_alerts[pos[0][d]] - s_means["More_alerts"])/s_stds["More_alerts"]
            new_s = S.iloc[pos[0][d]]
            new_s["alert_sum"] = alerts_scaled
            new_s["More_alerts"] = more_scaled
            v = torch.tensor(new_s)
            output = pred_model.net(v.float()).detach().numpy()
            # if output[1] > output[0]:
            if math.exp(output[1]) > 0:
                policy[pos[0][d]] = 1
                new_alerts[pos[0][d:(n_days-1)]] += 1
        elif (prob_constraint == True) & (near_zero.iloc[pos[0][d]] == False):
            alerts_scaled = (new_alerts[pos[0][d]] - s_means["alert_sum"])/s_stds["alert_sum"]
            more_scaled = (Constraint.iloc[pos[0][d]] - new_alerts[pos[0][d]] - s_means["More_alerts"])/s_stds["More_alerts"]
            new_s = S.iloc[pos[0][d]]
            new_s["alert_sum"] = alerts_scaled
            new_s["More_alerts"] = more_scaled
            v = torch.tensor(new_s)
            output = pred_model.net(v.float()).detach().numpy()
            # if output[1] > output[0]:
            if math.exp(output[1]) > 0:
                policy[pos[0][d]] = 1
                new_alerts[pos[0][d:(n_days-1)]] += 1
        d+=1
    print(i)

pol = pd.DataFrame(policy, columns = ["policy"])
pol.to_csv("Fall_results/DQN_11-5_hosps_constrained_policy.csv")
# pol.to_csv("Fall_results/DQN_11-5_deaths_constrained_policy.csv")

############

# Q = model(tensors[0].float()) # n x 2
# best_action = Q.argmax(axis=1).view(-1, 1)
# final_action = best_action.numpy()
# DF = pd.concat([pd.DataFrame(ID,columns=["ID"]), pd.DataFrame(final_action,columns=["A"])], axis=1)
# sum_alerts = DF.groupby("ID")["A"].agg("cumsum")
# here_over = sum_alerts > S["alert_sum"] + S["More_alerts"]
# final_action[here_over] = 0