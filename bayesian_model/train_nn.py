import logging

import hydra
import pyro
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
                             HeatAlertModel)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    logging.info("Loading data")
    dm = HeatAlertDataModule(
        dir=cfg.datadir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # Load model
    logging.info("Creating model")
    model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectivess_constraints=dm.effectiveness_constraints,
        effectivess_feature_names=dm.effectiveness_feature_names,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
    )

    # use low-rank normal guide and initialize by calling it once
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)  # always needed to initialize guide

    # create lightning module for training
    module = HeatAlertLightning(
        model=model,
        guide=guide,
        num_particles=cfg.training.num_particles,
        lr=cfg.training.lr,
        jit=cfg.training.jit,
        dos_spline_basis=dm.dos_spline_basis,
    )

    # Train model
    logger = pl.loggers.TensorBoardLogger("logs/", name=cfg.model.name)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        enable_checkpointing=False,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        max_steps=cfg.training.max_steps,
    )
    logging.info("Training model")
    trainer.fit(module, dm)

    # test saving the model using pytorch lightning
    logging.info("Saving ckpts")
    ckpt_lightning = f"ckpts/{cfg.model.name}_lightning.ckpt"
    ckpt_guide = f"ckpts/{cfg.model.name}_guide.pt"
    ckpt_model = f"ckpts/{cfg.model.name}_model.pt"

    trainer.save_checkpoint(ckpt_lightning)
    torch.save(model.state_dict(), ckpt_model)
    torch.save(guide.state_dict(), ckpt_guide)

    # save config for easy reproducibility
    with open(f"ckpts/{cfg.model.name}_cfg.yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main()
