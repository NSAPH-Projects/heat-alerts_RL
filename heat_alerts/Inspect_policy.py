import numpy as np
import pandas as pd
import itertools
import torch 
from torch import nn

from heat_alerts.Q_prep_function import make_data

class DQN(nn.Module):
    def __init__(self, n_col) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_col, n_col^2),
            nn.SiLU(),
            # nn.Linear(n_col**2, n_col**2),
            # nn.SiLU(),
            nn.Linear(n_col^2, 2)
        )
    def forward(self, x):
        return self.net(x)

### Prep data:
D = make_data(data_only=False)
S,A,R,S_1,ep_end,over = [D[k] for k in ("S","A","R","S_1","ep_end","over")]
S = S.drop("index", axis = 1)
S_1 = S_1.drop("index", axis = 1)
data = [S, A, R, S_1, ep_end, over]
tensors = [torch.from_numpy(v.to_numpy()).to("cpu") for v in data]

n_years = 11
n_days = 153
ID = list(itertools.chain(*[itertools.repeat(i, n_days-1) for i in range(0,int(S.shape[0]/(n_days-1)))]))

Budget,n_seq_s,s_means,s_stds = [D[k] for k in ("Budget","n_seq_s","s_means","s_stds")]
Constraint = pd.DataFrame(Budget).drop(n_seq_s)

### Look at results from model:
model = torch.load("Fall_results/DQN_10-15d.pt", map_location=torch.device('cpu'))

new_alerts = np.zeros(len(ID))
policy = np.zeros(len(ID))
for i in range(0, max(ID)):
    pos = np.where(np.array(ID) == i)
    d = 0
    while (d < n_days - 2) & (new_alerts[pos[0][d]] < Constraint.iloc[pos[0][d]]).all():
        alerts_scaled = (new_alerts[pos[0][d]] - s_means["alert_sum"])/s_stds["alert_sum"]
        more_scaled = (Constraint.iloc[pos[0][d]] - new_alerts[pos[0][d]] - s_means["More_alerts"])/s_stds["More_alerts"]
        new_s = S.iloc[pos[0][d]]
        new_s["alert_sum"] = alerts_scaled
        new_s["More_alerts"] = more_scaled
        v = torch.tensor(new_s)
        output = model(v.float()).detach().numpy()
        if output[1] > output[0]:
            policy[pos[0][d]] = 1
            new_alerts[pos[0][d:(n_days-1)]] += 1
        d+=1

pol = pd.DataFrame(policy, columns = ["policy"])
pol.to_csv("Fall_results/DQN_10-15d_policy.csv")


############

# Q = model(tensors[0].float()) # n x 2
# best_action = Q.argmax(axis=1).view(-1, 1)
# final_action = best_action.numpy()
# DF = pd.concat([pd.DataFrame(ID,columns=["ID"]), pd.DataFrame(final_action,columns=["A"])], axis=1)
# sum_alerts = DF.groupby("ID")["A"].agg("cumsum")
# here_over = sum_alerts > S["alert_sum"] + S["More_alerts"]
# final_action[here_over] = 0