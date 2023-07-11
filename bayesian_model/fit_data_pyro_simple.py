
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

# import torch.nn as nn
from pyro.infer import Predictive
from pyro.infer.autoguide import init_to_median

# from pyro.nn import PyroParam
from tqdm import tqdm


def train(model, guide, data, lr, n_epochs, batch_size, num_particles=1):
    # initialize
    pyro.clear_param_store()
    loss_fn = pyro.infer.Trace_ELBO(num_particles=num_particles)(model, guide)
    loss_fn(*data)  # initialize parameters
    num_pars = [p.numel() for p in loss_fn.parameters() if p.requires_grad]
    print("Number of parameters: ", sum(num_pars))
    opt = torch.optim.Adam(loss_fn.parameters(), lr=lr)

    # # Create a dataloader.
    dataset = torch.utils.data.TensorDataset(*data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True,
    )

    epoch_loss = np.nan
    for epoch in range(n_epochs):
        epoch_losses = []
        # add epoch indicator in tqdm
        pbar = tqdm(dataloader, leave=False)
        for batch in pbar:
            opt.zero_grad()
            reg = 1e-4 * sum([x.pow(2).sum() for x in loss_fn.guide.parameters()])
            elbo_loss = loss_fn(*batch)
            loss = elbo_loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_value_(loss_fn.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())
            pbar_desc = f"[epoch {epoch + 1}/{n_epochs}, {loss:.4f}]"
            pbar.set_description(pbar_desc, refresh=False)
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch == 0 or ((epoch + 1) % (n_epochs // 10)) == 0:
            print(
                "[epoch {}/{}]  av. loss: {:.4f}".format(
                    epoch + 1, n_epochs, epoch_loss
                )
            )


class Model(pyro.nn.PyroModule):
    def __init__(self, W, X):
        super().__init__()
        N, DX = X.shape
        S, DW = W.shape
        self.S = S
        self.N = N
        self.DX = DX
        self.DW = DW
        self.W = W
        # in this version, the weights of lin_beta are not sampled/constrained
        self.lin_beta = pyro.nn.PyroModule[torch.nn.Linear](DW, DX)
        self.lin_gamma = pyro.nn.PyroModule[torch.nn.Linear](DW, DX)


    def forward(self, A, X, offset, sind, Y=None, return_all=False):
        # omega_beta = pyro.sample(
        #     "omega_beta", dist.InverseGamma(1.1, 1.1).expand([self.DX]).to_event(1)
        # )
        # omega_gamma = pyro.sample(
        #     "omega_gamma", dist.InverseGamma(1.1, 1.1).expand([self.DX]).to_event(1)
        # )
        omega_beta = pyro.sample(
            "omega_beta", dist.HalfCauchy(1.0).expand([self.DX]).to_event(1)
        )
        omega_gamma = pyro.sample(
            "omega_gamma", dist.HalfCauchy(1.0).expand([self.DX]).to_event(1)
        )
        beta = pyro.sample(
            "beta",
            dist.Normal(self.lin_beta(self.W), omega_beta.sqrt()).to_event(2),
        ) # beta[s]~ N(beta_mean[s], omega_beta),  where beta_mean[s] = W[s]'delta + intercept
        gamma = pyro.sample(
            "gamma",
            dist.Normal(self.lin_gamma(self.W), omega_gamma.sqrt()).to_event(2),
        )
        beta = beta[sind, :]
        gamma = gamma[sind, :]
        tau = torch.sigmoid((beta * X).sum(-1))
        lam = torch.exp((gamma * X).sum(-1).clamp(-20, 10))
        mu = offset * lam * (1.0 - A * tau)
        with pyro.plate("obs_plate", self.N, X.shape[0]):
            obs = pyro.sample("obs", dist.Poisson(mu + 1e-6), obs=Y)

        if not return_all:
            return obs
        else:
            return torch.stack([tau, lam, mu, obs], axis=1)


def main(args):
    # %% read processed data
    dir = "bayesian_model/data/processed/"
    X = pd.read_parquet(f"{dir}/states.parquet")
    A = pd.read_parquet(f"{dir}/actions.parquet")
    Y = pd.read_parquet(f"{dir}/outcomes.parquet")
    sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
    W = pd.read_parquet(f"{dir}/spatial_feats.parquet")
    offset = pd.read_parquet(f"{dir}/offset.parquet")

    # make fitting data
    A_ = torch.tensor(A.values[:, 0], dtype=torch.float32)
    Y_ = torch.tensor(Y.values[:, 0], dtype=torch.float32)
    X_ = torch.tensor(X.values, dtype=torch.float32)
    W_ = torch.tensor(W.values, dtype=torch.float32)
    sind_ = torch.tensor(sind.values[:, 0], dtype=torch.long)
    offset_ = torch.tensor(offset.values[:, 0], dtype=torch.float32)

    # render pyro model
    S, DW = W_.shape
    N, DX = X_.shape
    model = Model(W_, X_)
    inputs = [A_, X_, offset_, sind_, Y_]
    # pyro.render_model(
    #     model,
    #     model_args=inputs,
    #     render_distributions=True,
    #     filename="bayesian_model/fit_data_pyro_diag.png",
    # )

    # inference with autoguide
    # guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    init_fn = init_to_median

    # guide = pyro.infer.autoguide.AutoDelta(model, init_loc_fn=init_fn)
    # guide = pyro.infer.autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fn)
    # guide = LowRankGuideNudge(model, init_loc_fn=init_fn)
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(
        model, init_loc_fn=init_fn
    )

    train(
        model,
        guide,
        inputs,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=X.shape[0] // S,
        num_particles=args.num_particles,
    )

    # extract tau
    sites = [
        "omega_beta",
        "omega_gamma",
        "beta",
        "gamma",
    ]
    params = Predictive(model, guide=guide, num_samples=50, return_sites=sites)(*inputs)
    params = {k: v[:, 0] for k, v in params.items()}

    # save a json of all samples on sites, make sure to save as float
    results = {}
    for s in sites:
        results[s] = params[s].numpy().astype(float).tolist()
    with open("bayesian_model/fit_data_pyro_simple.json", "w") as io:
        json.dump(results, io)

    predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=50,
        return_sites=["_RETURN"],
    )
    outputs = predictive_outputs(*inputs, return_all=True)["_RETURN"]

    # plot histograms
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    for i, var in enumerate(("tau", "lam", "mu")):
        v = outputs[..., i].mean(0)
        ax[i].hist(v, bins=50)
        ax[i].set_title(var)

    # add mu vs real obs
    ax[3].scatter(outputs[..., 2].mean(0), outputs[..., 3].mean(0), alpha=0.03)
    ax[3].set_title("mu vs real obs")
    ax[3].set_xlabel("mu")
    ax[3].set_ylabel("real obs")

    fig.savefig("bayesian_model/fit_data_pyro_simple.png", bbox_inches="tight")

    # make errorplot of gamma and beta coefficients
    betas = params["beta"]
    gammas = params["gamma"]

    betas_means = betas.mean((0, 1))
    betas_std = betas.std((0, 1))
    gammas_means = gammas.mean((0, 1))
    gammas_std = gammas.std((0, 1))
    colnames = [c for c in X.columns if not c.startswith("dos")]
    colidx = np.array([i for i, c in enumerate(X.columns) if not c.startswith("dos")])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].errorbar(
        np.arange(len(colnames)),
        betas_means[colidx],
        yerr=betas_std[colidx],
        fmt="o",
    )
    ax[0].set_xticks(np.arange(len(colnames)))
    ax[0].set_xticklabels(colnames, rotation=45)
    ax[0].set_title("heat alert effectiveness")
    ax[1].errorbar(
        np.arange(len(colnames)),
        gammas_means[colidx],
        yerr=gammas_std[colidx],
        fmt="o",
    )
    ax[1].set_xticks(np.arange(len(colnames)))
    ax[1].set_title("expected baseline hospitalizations")
    fig.savefig("bayesian_model/fit_data_pyro_coefficients_simple.png", bbox_inches="tight")

    # load spline design matrix
    dos = pd.read_parquet("bayesian_model/data/processed/Btdos.parquet")  # T x num feats
    dos = torch.tensor(dos.values, dtype=torch.float32)
    dos_cols = np.array([i for i, c in enumerate(X.columns) if c.startswith("dos")], dtype=int)
    dos_beta = betas[..., dos_cols].reshape(-1, 1, len(dos_cols))  #  (num samples*locs) x n1 x um feats
    dos_eff = (dos_beta * dos).sum(-1)  # num samples x T

    # plot as spaghetti, gray, alpha
    ix = np.random.choice(dos_eff.shape[0], 200, replace=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dos_eff[ix].T, color="k", alpha=0.1, lw=0.5)
    ax.plot(dos_eff.mean(0), color="k", lw=2)
    ax.set_xlabel("Day of summer")
    ax.set_title("Day of summer effect")
    fig.savefig("bayesian_model/fit_data_pyro_splines_simple.png", bbox_inches="tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_particles", type=int, default=10)
    args = parser.parse_args()
    main(args)
