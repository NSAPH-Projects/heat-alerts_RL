import argparse
import logging

import pyro
import torch
from omegaconf import OmegaConf

from pyro_heat_alert import HeatAlertDataModule, HeatAlertModel


def main(args: argparse.Namespace):
    # Load data
    logging.info("Loading config")
    cfg = OmegaConf.load(args.config)

    logging.info("Loading data module")
    dm = HeatAlertDataModule(dir=cfg.datadir)

    # Load model
    logging.info("Creating model")
    model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectivess_constraints=dm.effectivess_constraints,
        effectivess_feature_names=dm.effectiveness_feature_names,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
    )

    logging.info("Loading model weights")
    model.load_state_dict(torch.load(f"ckpts/{cfg.model.name}_model.pt"))

    # Guide, initialize and load state dict
    logging.info("Creating guide")
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)

    logging.info("Loading guide weights")
    guide.load_state_dict(torch.load(f"ckpts/{cfg.model.name}_guide.pt"))

    # run some tests
    logging.info("Running tests")

    # test sampling from the posterior using guide
    logging.info("Sampling from posterior")
    sample = guide(*dm.dataset.tensors)

    shapes = {k: v.shape for k, v in sample.items()}
    logging.info(f"Obtained a sampel from guide:\n{shapes}")

    # test using posterior predictive, 10 samples
    logging.info("Sampling from posterior predictive (10 samples)")
    sites = list(sample.keys()) + ["_RETURN"]
    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=10, return_sites=sites
    )
    pp_sample = predictive(*dm.dataset.tensors, return_outcomes=True)
    shapes = {k: v.shape for k, v in pp_sample.items()}
    logging.info(f"Obtain samples using Pyro's posterior predictive: \n{shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # configure logging to print hour and message
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    main(args)
