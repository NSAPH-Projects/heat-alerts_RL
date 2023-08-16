import logging

import hydra
import numpy as np
import pyro
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from heat_alerts.bayesian_model import (
    HeatAlertDataModule,
    HeatAlertLightning,
    HeatAlertModel,
)


@hydra.main(config_path="conf/online_rl", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    logging.info("Loading data")
    dm = HeatAlertDataModule(dir=cfg.datadir,  load_outcome=False, for_gym=True)
    (
        _,
        loc,
        county_summer_mean,
        alert,
        baseline_features_tensor,
        effectiveness_features_tensor,
        _,
        year,
        budget,
        state,
        hi_mean
    ) = dm.gym_dataset

   


if __name__ == "__main__":
    main()
