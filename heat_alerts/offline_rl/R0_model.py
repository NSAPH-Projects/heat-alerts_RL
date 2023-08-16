import os
from argparse import ArgumentParser
from typing import Tuple
from collections import OrderedDict
import numpy as np
import random
import pandas as pd
from random import sample
from sklearn import preprocessing as skprep

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

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def symlog(x, shift=1):
    if x >= 0:
        return np.log(x+shift)-np.log(shift)
    else:
        return -np.log(-x+shift)+np.log(shift)

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
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x):
        step1 = self.net(x)
        return step1

class Rewards_Lightning(pl.LightningModule):
    def __init__(self, n_col, config, N, b_size, lr, loss="huber",  optimizer="adam") -> None:
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
        s, r = batch
        preds = self.net(s)
        Preds = -torch.exp(preds)
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

    set_seed(321)

    ## Get dataset with only the days when A=0:
    filename="data/Summer23_Train_smaller-for-Python.csv"
    Train = pd.read_csv(filename)
    
    R = -1*(Train["other_hosps"]/Train["total_count"])
    R = R.apply(symlog,shift=1)

    States = Train[["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "BA_zone", "l.Pop_density", "l.Med.HH.Income",
                        "year", "dos", "holiday", "dow", 
                        "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                        "all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate",     
                        "heat_hosp_3dMA_rate", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                        "broadband.usage", "Democrat", "Republican", "pm25"]]
    
    ## One-hot encode non-numeric variables
    S_enc = skprep.OneHotEncoder(drop = "first")
    S_enc.fit(States[["BA_zone", "holiday", "dow"]])
    S_ohe = S_enc.transform(States[["BA_zone", "holiday", "dow"]]).toarray()
    S_names = S_enc.get_feature_names_out(["BA_zone", "holiday", "dow"])
    S_OHE = pd.DataFrame(S_ohe, columns=S_names)

    num_vars = ["HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                        "quant_HI_3d_county", "quant_HI_fwd_avg_county", "HI_mean",
                        "l.Pop_density", "l.Med.HH.Income", "year", "dos",
                         "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                         "all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate",     
                        "heat_hosp_3dMA_rate", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                         "broadband.usage", "Democrat", "Republican", "pm25"]
    
    s_means = States[num_vars].mean(0)
    s_stds = States[num_vars].std(0)
    S = (States[num_vars] - s_means)/s_stds
    S = pd.concat([S.reset_index(), S_OHE.reset_index()], axis = 1)
    S = S.drop("index", axis=1)

    S["weekend"] = S["dow_Saturday"] + S["dow_Sunday"]
    S = S[[
            "quant_HI_county", "HI_mean", "quant_HI_3d_county",
            "l.Pop_density", "l.Med.HH.Income",
            "year", "dos", "all_hosp_mean_rate",
             "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate", 
             "age_65_74_rate", "age_75_84_rate", "dual_rate",
            "Republican", "pm25", "weekend", 'BA_zone_Hot-Dry',
            'BA_zone_Hot-Humid', 'BA_zone_Marine', 'BA_zone_Mixed-Dry',
            'BA_zone_Mixed-Humid', 'BA_zone_Very Cold'
        ]]

    state_dim = S.shape[1]
    N = len(R)
    data = [S,R]

    # Select validation set by episode, ensure equal representation by all counties
    n_years = 10
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

    train = list(set(list(range(0,N))) - set(val))

    # Make data loader, shuffled:
    train_perm = np.random.permutation(len(train))
    val_perm = np.random.permutation(len(val))
    train_data = [v.to_numpy()[train][train_perm] for v in data]
    val_data = [v.to_numpy()[val][val_perm] for v in data]

    A = (Train["alert"] == 1).to_numpy()

    train_tensors = [np.delete(v, A[train][train_perm], axis=0) for v in train_data]
    val_tensors = [np.delete(v, A[val][val_perm], axis=0) for v in val_data]
    for j in [0, 1]: train_tensors[j] = torch.FloatTensor(train_tensors[j])

    for j in [0, 1]: val_tensors[j] = torch.FloatTensor(val_tensors[j])

    train_DS = TensorDataset(*train_tensors)
    val_DS = TensorDataset(*val_tensors)

    train_DL = DataLoader(
        train_DS,
        batch_size= 2048,
        num_workers= 4,
        persistent_workers= True
    )

    val_DL = DataLoader(
        val_DS,
        batch_size= 2048,
        num_workers= 4,
        persistent_workers= True
    )

    config = { # results from tuning
        "n_hidden": 256,
        "dropout_prob": 0.0,
        "w_decay": 1e-4
    }

    model = Rewards_Lightning(state_dim, config, N = N, b_size = 2048, lr = 0.03)
    logger_name = "R0_model_no_alerts"
    logger = CSVLogger("lightning_logs", name=logger_name)
    
    trainer = pl.Trainer(
        max_epochs= 300,
        logger= logger,
        accelerator= "auto",
        devices= 1,
        enable_progress_bar= True,
        callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))]
    )
    # trainer.tune(model, train_DL, val_DL)
    trainer.fit(model, train_DL, val_DL)
    
    torch.save(model, "Summer_results/R0_model_no_alerts.pt")

    ## Save modeled rewards:
    model.eval() # turns off dropout for the predictions
    r_hat = model.net(torch.FloatTensor(S.to_numpy()))
    R_hat = -torch.exp(r_hat)
    n = R_hat.detach().numpy()
    df = pd.DataFrame(n)
    df.to_csv("Summer_results/R0_model_no_alerts_7-5.csv")



if __name__ == "__main__":
    
    main()
