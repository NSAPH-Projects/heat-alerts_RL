
# import os
import time
import numpy as np
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
from scipy.special import expit, softmax

#%%

import torch 
from torch import nn # creating modules
from torch.nn import functional as F # losses, etc.
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from tqdm import tqdm
from copy import deepcopy

from Q_prep_function import make_data

## Set up the model

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

## Methods to evaluate the model

def eval_Q_double(Q_model, Qtgt_model, S, not_allowed = None): 
    Q = Q_model(S) # n x 2
    Qtgt = Qtgt_model(S) # n x 2
    best_action = Q.argmax(axis=1).view(-1, 1)
    best_Q = torch.gather(Qtgt, 1, best_action).view(-1) # n 
    if not_allowed is not None:
        final_Q = torch.where(not_allowed, Qtgt[:,0], best_Q) # n
        return final_Q
    return best_Q


## Set up dataloader
D = make_data()
S,A,R,S_1,ep_end,over,near_zero = [D[k] for k in ("S","A","R","S_1","ep_end","over","near_zero")]
not_allowed = over | [n for n in near_zero["x"]]
data = [S.drop("index", axis = 1), A, R, S_1.drop("index", axis = 1), ep_end, not_allowed]

gamma = 0.99

n_years = 11
n_days = 153
n_counties = S.shape[0]/(n_years*(n_days-1))

dev = "cuda"
# dev = "cpu"
tensors = [torch.from_numpy(v.to_numpy()).to(dev) for v in data]

DS = TensorDataset(*tensors)

torch.manual_seed(321)

DL = DataLoader(
    DS, batch_size = 256, shuffle = True
    # , pin_memory=True
    # , num_workers=32
    # , persistent_workers=True
)
# DL = DataLoader(DS, batch_size = 1024, shuffle = True, pin_memory = True) # if using cpu; actually doesn't matter

## Initialize the model

state_dim = S.drop("index", axis = 1).shape[1]
num_episodes = n_counties*n_years
# print(num_episodes)
n = num_episodes*(n_days-1)

model = DQN(state_dim)
num_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
model = model.to(dev)

# optimizer = optim.Adam(model.parameters(), lr = 0.0001, betas=(0.25, 0.99))
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.25)
optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum=0.25)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.25)
# optimizer = optim.SGD(model.parameters(), lr = 0.0005, momentum=0.25)

update_tgt_every = 100
print_every = 10
tgt_model = deepcopy(model)

## Train

epochs = 60000

epoch_loss_means = []
epoch_loss_full = []
start = time.time()
for k in range(len(epoch_loss_means), epochs):
    # if k == 100:
    #     break
    l = []
    iter = 1
    for s, a, r, s1, e, nope in tqdm(DL, disable=True):
        # break
        with torch.no_grad():
            target = r + gamma*(1-e.float())*eval_Q_double(model, tgt_model, s1.float(), nope)
        optimizer.zero_grad()
        output = model(s.float()) # n x 2
        Q = torch.gather(output, 1, a.view(-1, 1)).view(-1)
        # loss = F.mse_loss(Q, target.float())
        loss = F.smooth_l1_loss(Q, target)  # huber loss as in DQN paper
        loss.backward()
        optimizer.step()
        l.append(loss.item())
        iter+=1
        # print(iter)
        if iter == 100:
            break
    # break
    epoch_loss = np.mean(l)
    epoch_loss_means.append(epoch_loss)
    with torch.no_grad(): # Note: tensors order is S, A, R, S1, EE, O
        Target = tensors[2] + gamma*(1-tensors[4].float())*eval_Q_double(model, tgt_model, 
            tensors[3].float(), tensors[5])
        Output = model(tensors[0].float()) # n x 2
        Q = torch.gather(Output, 1, tensors[1].view(-1, 1)).view(-1)
        full_loss = F.smooth_l1_loss(Q, Target).item()
        epoch_loss_full.append(full_loss)
    if k % print_every == 0:
        # full_loss = 0
        print(f"Epoch: {k}, average loss: {epoch_loss:.4f}, full loss: {full_loss:.4f}")
    if k % update_tgt_every == 0:
        tgt_model = deepcopy(model)
    #     break
    # if k == 20:
    #     break

print("--- %s seconds ---" % (time.time() - start))


torch.save(model, "Fall_results/DQN_10-18b.pt")
## Convert these to pd dataframes and then .to_csv
EL = pd.DataFrame(epoch_loss_means, columns = ["Means"])
EL["Full"] = epoch_loss_full
EL.to_csv("Fall_results/DQN_10-18b_epoch-losses.csv")


