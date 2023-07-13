import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch


from pyro.infer import Predictive
from pyro.infer.autoguide import init_to_median
from tqdm import tqdm


def train(model, guide, data, lr, n_epochs, batch_size, num_particles=1):
    # initialize
    pyro.clear_param_store()
    loss_fn = pyro.infer.Trace_ELBO(num_particles=num_particles)(model, guide)
    loss_fn(*data)  # initialize parameters
    num_pars = [p.numel() for p in loss_fn.parameters() if p.requires_grad]
    print("Number of parameters: ", sum(num_pars))
    dataset = torch.utils.data.TensorDataset(*data)
    # # Create a dataloader.
    if batch_size is not None:
        opt = torch.optim.Adam(loss_fn.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        Epoch_losses = []
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
            Epoch_losses.append(epoch_loss)
            if epoch == 0 or ((epoch + 1) % (n_epochs // 10)) == 0:
                print(
                    "[epoch {}/{}]  av. loss: {:.4f}".format(
                        epoch + 1, n_epochs, epoch_loss
                    )
                )
        return Epoch_losses
    else:
        epoch_loss = np.nan
        pbar = tqdm(range(n_epochs), leave=False)
        opt = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(model, guide, opt, loss_fn)
        epoch_losses = []
        for epoch in pbar:
            epoch_loss = svi.step(data)  # for some reason currently complaining
            pbar_desc = f"[epoch {epoch + 1}/{n_epochs}, {epoch_loss:.4f}]"
            pbar.set_description(pbar_desc, refresh=False)
            if epoch == 0 or ((epoch + 1) % (n_epochs // 10)) == 0:
                print(
                    "[epoch {}/{}]  av. loss: {:.4f}".format(
                        epoch + 1, n_epochs, epoch_loss
                    )
                )
            epoch_losses.append(epoch_loss.item())
        return epoch_losses


class Model(pyro.nn.PyroModule):
    def __init__(self, N, S):
        super().__init__()
        self.S = S
        self.N = N
    def forward(self, A, X, W, offset, sind, Y=None, return_all=False):
        batch_size, DX = X.shape
        _, DW = W.shape
        S = self.S
        # spatial random effects and common variance
        unstruct_beta = pyro.sample(
            "unstruct_beta", dist.Normal(0, 1).expand([S, DX]).to_event(2)
        )
        unstruct_gamma = pyro.sample(
            "unstruct_gamma", dist.Normal(0, 1).expand([S, DX]).to_event(2)
        )
        omega_beta = pyro.sample(
            "omega_beta", dist.HalfCauchy(1.0).expand([DX]).to_event(1)
        )
        omega_gamma = pyro.sample(
            "omega_gamma", dist.HalfCauchy(1.0).expand([DX]).to_event(1)
        )
        # coefficients for the mean of the random effects as
        # explained by the spatial features
        delta_beta = pyro.sample(
            "delta_beta", dist.Normal(0, 1).expand([DW, DX]).to_event(2)
        )
        delta_gamma = pyro.sample(
            "delta_gamma", dist.Normal(0, 1).expand([DW, DX]).to_event(2)
        )
        # linear coefficients of heat alert effectivness
        beta_prior_mean = W @ delta_beta  # matrix multiplication
        gamma_prior_mean = W @ delta_gamma
        beta = beta_prior_mean + omega_beta * unstruct_beta[sind]
        gamma = gamma_prior_mean + omega_gamma * unstruct_gamma[sind]
        lam = torch.exp((beta * X).sum(-1).clamp(-20, 10))  # baseline rate
        tau = torch.sigmoid((gamma * X).sum(-1))  # heat alert effectiveness
        # expected number of cases
        mu = offset * lam * (1.0 - A * tau)
        with pyro.plate("obs_plate", self.N, batch_size):
            obs = pyro.sample("obs", dist.Poisson(mu + 1e-6), obs=Y)
        if not return_all:
            return obs
        else:
            return torch.stack([tau, lam, mu, obs], axis=1)


def main(args):
    # %% load processed data
    dir = "data/processed/"
    X = pd.read_parquet(f"{dir}/states.parquet")
    A = pd.read_parquet(f"{dir}/actions.parquet")
    Y = pd.read_parquet(f"{dir}/outcomes.parquet")
    sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
    W = pd.read_parquet(f"{dir}/spatial_feats.parquet").iloc[sind.values[:, 0]]
    offset = pd.read_parquet(f"{dir}/offset.parquet")

    # make tensors
    A_ = torch.tensor(A.values[:, 0], dtype=torch.float32)
    Y_ = torch.tensor(Y.values[:, 0], dtype=torch.float32)
    X_ = torch.tensor(X.values, dtype=torch.float32)
    W_ = torch.tensor(W.values, dtype=torch.float32)
    sind_ = torch.tensor(sind.values[:, 0], dtype=torch.long)
    offset_ = torch.tensor(offset.values[:, 0], dtype=torch.float32)

    # render pyro model
    N, DX = X_.shape
    _, DW = W_.shape
    S = sind_.max().item() + 1
    model = Model(N, S)
    inputs = [A_, X_, W_, offset_, sind_, Y_]

    init_fn = init_to_median

    # guide = pyro.infer.autoguide.AutoDelta(model, init_loc_fn=init_fn)
    # guide = pyro.infer.autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fn)
    # guide = LowRankGuideNudge(model, init_loc_fn=init_fn)
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(
        model, init_loc_fn=init_fn
    )

    Epoch_losses = train(
        model,
        guide,
        inputs,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=X.shape[0] // S if not args.full_batch else None,
        num_particles=args.num_particles,
    )

    plt.plot(np.log(np.array(Epoch_losses)))
    plt.savefig("Plots_params/fit_data_pyro_base_" + args.name + "_log-Loss.png")
    plt.clf()

    # extract tau
    sites = [
        "omega_beta",
        "omega_gamma",
        "delta_beta",
        "delta_gamma",
        "unstruct_beta",
        "unstruct_gamma"
    ]
    params = Predictive(model, guide=guide, num_samples=args.n_samples, return_sites=sites)(*inputs)
    params = {k: v[:, 0] for k, v in params.items()}

    # save a json of all samples on sites, make sure to save as float
    results = {}
    for s in sites:
        results[s] = params[s].numpy().astype(float).tolist()
    with open("Plots_params/fit_data_pyro_base_" + args.name + ".json", "w") as io:
        json.dump(results, io)

    predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=args.n_samples,
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

    fig.savefig("Plots_params/fit_data_pyro_base_" + args.name + ".png", bbox_inches="tight")

    # make errorplot of gamma and beta coefficients
    betas = []
    gammas = []
    W__ = torch.tensor(
        pd.read_parquet(f"{dir}/spatial_feats.parquet").values, dtype=torch.float32
    )
    for i in range(args.n_samples):
        delta_beta = params["delta_beta"][i]
        delta_gamma = params["delta_gamma"][i]
        omega_beta = params["omega_beta"][i]
        omega_gamma = params["omega_gamma"][i]
        unstruct_beta = params["unstruct_beta"][i]
        unstruct_gamma = params["unstruct_gamma"][i]
        beta = W__ @ delta_beta + omega_beta * unstruct_beta
        gamma = W__ @ delta_gamma + omega_gamma * unstruct_gamma
        betas.append(beta.numpy())
        gammas.append(gamma.numpy())

    betas = np.stack(betas, axis=0)
    gammas = np.stack(gammas, axis=0)

    betas_means = np.nanmean(betas,(0, 1))
    betas_std = np.nanstd(betas,(0, 1))
    gammas_means = np.nanmean(gammas,(0, 1))
    gammas_std = np.nanstd(gammas,(0, 1))
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
    ax[0].set_title("baseline rate")
    ax[1].errorbar(
        np.arange(len(colnames)),
        gammas_means[colidx],
        yerr=gammas_std[colidx],
        fmt="o",
    )
    ax[1].set_xticks(np.arange(len(colnames)))
    ax[1].set_xticklabels(colnames, rotation=45)
    ax[1].set_title("heat alert effectiveness")
    fig.savefig("Plots_params/fit_data_pyro_coefficients_base_" + args.name + ".png", bbox_inches="tight")

    # load spline design matrix
    dos = pd.read_parquet("data/processed/Btdos.parquet").values  # T x num feats

    dos_cols = np.array(
        [i for i, c in enumerate(X.columns) if c.startswith("dos")], dtype=int
    )
    dos_beta = betas[..., dos_cols].reshape(
        -1, 1, len(dos_cols)
    )  #  (num samples*locs) x n1 x um feats
    dos_gamma = gammas[..., dos_cols].reshape(
        -1, 1, len(dos_cols)
    )  #  (num samples*locs) x n1 x um feats
    dos_beta_eff = (dos_beta * dos).sum(-1)  # num samples x T
    dos_gamma_eff = (dos_gamma * dos).sum(-1)  # num samples x T

    # plot as spaghetti, gray, alpha
    ix = np.random.choice(dos_beta_eff.shape[0], 200, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(dos_beta_eff[ix].T, color="k", alpha=0.1, lw=0.5)
    ax[0].plot(dos_beta_eff.mean(0), color="k", lw=2)
    ax[0].set_xlabel("Day of summer")
    ax[0].set_title("Baseline rate")
    ax[1].plot(dos_gamma_eff[ix].T, color="k", alpha=0.1, lw=0.5)
    ax[1].plot(dos_gamma_eff.mean(0), color="k", lw=2)
    ax[1].set_xlabel("Day of summer")
    ax[1].set_title("Heat alert effectiveness")
    fig.savefig("Plots_params/fit_data_pyro_splines_base_" + args.name + ".png", bbox_inches="tight")

    torch.save(model, "../Bayesian_models/Pyro_model_" + args.name + ".pt")
    torch.save(guide, "../Bayesian_models/Pyro_guide_" + args.name + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--full_batch", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
