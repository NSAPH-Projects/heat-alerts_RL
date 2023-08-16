import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn as nn
import torch.distributions as D
from torch.distributions.transforms import ExpTransform
from torch.nn.functional import softplus
from pyro.infer import Predictive
from pyro.infer.autoguide import init_to_median
from torch.distributions import constraints
from tqdm import tqdm


class CoefDist(D.Distribution):
    def __init__(self, scale, positive_contraints=[]):
        super().__init__()
        self.distributions = []
        for i in range(len(scale)):
            if i not in positive_contraints:
                self.distributions.append(D.Normal(0, scale[i]))
            else:
                self.distributions.append(D.HalfNormal(scale[i]))
        batch_shape = torch.Size([d.batch_shape for d in self.distributions])
        event_shape = torch.Size([d.event_shape for d in self.distributions])
        super().__init__(batch_shape, event_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape + self.event_shape
        samples = torch.stack(
            [dist.sample(sample_shape) for dist in self.distributions], dim=-1
        )
        return samples

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape + self.event_shape
        samples = torch.stack(
            [dist.rsample(sample_shape) for dist in self.distributions], dim=-1
        )
        return samples

    def log_prob(self, value):
        assert value.shape[-1] == len(
            self.distributions
        ), "Input 'value' must have the same number of dimensions as 'distributions'"
        log_probs = torch.stack(
            [
                dist.log_prob(v)
                for dist, v in zip(self.distributions, value.unbind(dim=-1))
            ],
            dim=-1,
        )
        return torch.sum(log_probs, dim=-1)


def train(
    model,
    guide,
    data,
    lr,
    n_epochs,
    batch_size,
    num_particles=1,
    jit=True,
    beta_positive_constraints=[],
    gamma_positive_constraints=[],
):
    # initialize
    pyro.clear_param_store()
    elbo_fun = pyro.infer.Trace_ELBO if not jit else pyro.infer.JitTrace_ELBO
    model(*data)
    guide(*data)
    loss_fn = elbo_fun(num_particles=num_particles)(model, guide)
    loss_fn(*data)  # initialize parameters
    num_pars = [p.numel() for p in loss_fn.parameters() if p.requires_grad]
    logging.info(f"Number of parameters: {sum(num_pars)}")
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
        epoch_losses = []
        epoch_loss = np.nan
        for epoch in range(n_epochs):
            losses_buffer = []
            # add epoch indicator in tqdm
            pbar = tqdm(dataloader, leave=False)
            for batch in pbar:
                opt.zero_grad()
                elbo_loss = loss_fn(*batch)
                loss = elbo_loss
                loss.backward()
                torch.nn.utils.clip_grad_value_(loss_fn.parameters(), 1.0)
                opt.step()
                losses_buffer.append(loss.item())
                pbar_desc = f"[epoch {epoch + 1}/{n_epochs}, {loss:.4f}]"
                pbar.set_description(pbar_desc, refresh=False)
            epoch_loss = sum(losses_buffer) / len(losses_buffer)
            epoch_losses.append(epoch_loss)
            if epoch == 0 or ((epoch + 1) % max(n_epochs // 10, 1) == 0):
                logging.info(
                    "[epoch {}/{}]  av. loss: {:.4f}".format(
                        epoch + 1, n_epochs, epoch_loss
                    )
                )
        return epoch_losses
    else:
        epoch_loss = np.nan
        pbar = tqdm(range(n_epochs), leave=False)
        opt = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(
            model, guide, opt, loss=elbo_fun(num_particles=num_particles)
        )
        losses_buffer = []
        for epoch in pbar:
            epoch_loss = svi.step(*data)  # for some reason currently complaining
            pbar_desc = f"[epoch {epoch + 1}/{n_epochs}, {epoch_loss:.4f}]"
            pbar.set_description(pbar_desc, refresh=False)
            if epoch == 0 or ((epoch + 1) % max(n_epochs // 10, 1)) == 0:
                logging.info(
                    "[epoch {}/{}]  av. loss: {:.4f}".format(
                        epoch + 1, n_epochs, epoch_loss
                    )
                )
            losses_buffer.append(epoch_loss)
        return losses_buffer


class Model(pyro.nn.PyroModule):
    def __init__(
        self,
        W,
        X,
        num_hidden_layers,
        hdim,
        beta_positive_constraints=[],
        gamma_positive_constraints=[],
    ):
        super().__init__()
        N, DX = X.shape
        S, DW = W.shape
        self.S = S
        self.N = N
        self.DX = DX
        self.DW = DW
        self.W = W
        self.beta_pc = beta_positive_constraints
        self.gamma_pc = gamma_positive_constraints

        def make_nn(indim, outdim, hdim, num_hidden, sample_weights=False, act=nn.ReLU):
            module = []
            for i in range(num_hidden):
                d = indim if i == 0 else hdim
                module.append(nn.Linear(d, hdim))
                module.append(act())
            d = indim if num_hidden == 0 else hdim
            module.append(nn.Linear(d, outdim))
            module = nn.Sequential(*module)

        self.nn_beta = make_nn(DW, DX, hdim, num_hidden_layers)
        self.nn_gamma = make_nn(DW, DX, hdim, num_hidden_layers)

    def model(self, A, X, offset, sind, subsample, Y=None):
        # scales of each parameter
        omega_beta = pyro.sample(
            "omega_beta",
            D.HalfNormal(1.0).expand([self.DX]).to_event(1),
        )
        omega_gamma = pyro.sample(
            "omega_gamma",
            D.HalfNormal(1.0).expand([self.DX]).to_event(1),
        )
        beta = pyro.sample(
            "beta",
            CoefDist(omega_beta, self.beta_pc).expand([self.S]).to_event(2),
        )
        gamma = pyro.sample(
            "gamma",
            CoefDist(omega_gamma, self.gamma_pc).expand([self.S]).to_event(2),
        )
        lam_bias = pyro.sample(
            "lam_bias",
            D.Uniform(-0.5, 0.5).expand([self.S]).to_event(1),
        )
        tau_bias = pyro.sample(
            "tau_bias",
            D.Uniform(-0.5, 0.5).expand([self.S]).to_event(1),
        )

        lam = torch.exp((beta[sind] * X).sum(-1).clamp(max=3) + lam_bias[sind])
        tau = torch.sigmoid((gamma[sind] * X).sum(-1) + tau_bias)

        mu = offset * lam * (1.0 - A * tau)
        with pyro.plate("obs_plate", self.N, subsample=subsample):
            obs = pyro.sample("obs", D.Poisson(mu + 1e-6), obs=Y)

        return torch.stack([tau, lam, mu, obs], axis=-1)

    def guide(self, A, X, offset, sind, idx, Y=None):
        # before we were using autoguide, now we define our own guide
        # it is obviously more manual work. We can impose some 
        # reasonable assumptions on the conditionals of the paramters.
        # the random effects will be independent in each location and
        # independent of the scale.

        # omega beta and gamma as a lognormal
        omloc_b = pyro.param("omloc_b", torch.zeros((self.DX,)))
        omloc_g = pyro.param("omloc_g", torch.zeros((self.DX,)))
        omdiag_b = pyro.param("omscale_b", torch.ones((self.DX,)))
        omdiag_g = pyro.param("omscale_g", torch.ones((self.DX,)))
        omcovf_b = pyro.param("omcovf_b", 0.01 * torch.randn((self.DX, self.DX)))
        omcovf_g = pyro.param("omcovf_g", 0.01 * torch.randn((self.DX, self.DX)))
        mvn_b = D.LowRankMultivariateNormal(omloc_b, omcovf_b, omdiag_b)
        mvn_g = D.LowRankMultivariateNormal(omloc_g, omcovf_g, omdiag_g)
        trans_b = D.TransformedDistribution(mvn_b, ExpTransform())
        trans_g = D.TransformedDistribution(mvn_g, ExpTransform())
        omega_beta = pyro.sample("omega_b", trans_b)
        omega_gamma = pyro.sample("omega_g", trans_g)
     
        # beta and gamma
        pyro.module("nn_beta", self.nn_beta)
        pyro.module("nn_gamma", self.nn_gamma)
        bm = self.nn_beta(self.W)
        gm = self.nn_gamma(self.W)
        beta_mean = torch.zeros([self.S, self.DX])
        gamma_mean = torch.zeros([self.S, self.DX])
        for i in range(self.DX):
            if i in self.beta_pc:
                beta_mean[..., i] = softplus(bm[..., i])
            else:
                beta_mean[..., i] = bm[..., i]
            if i in self.gamma_pc:
                gamma_mean[..., i] = softplus(gm[..., i])
            else:
                gamma_mean[..., i] = gm[..., i]

        # sample from our defined distribution for the loss
        with pyro.plate("S", self.S):
            pyro.sample("beta", CoefDist(omega_beta, self.beta_pc))
            pyro.sample("gamma", CoefDist(omega_gamma, self.gamma_pc))


def main(args):
    # %% read processed data
    dir = "data/processed/"
    X = pd.read_parquet(f"{dir}/states.parquet").drop(columns="intercept")
    A = pd.read_parquet(f"{dir}/actions.parquet")
    Y = pd.read_parquet(f"{dir}/outcomes.parquet")
    sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
    W = pd.read_parquet(f"{dir}/spatial_feats.parquet").drop(columns="intercept")

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
    model = Model(W_, X_, args.num_hidden_layers, args.hidden_dim, args.sample_weights)
    ix = torch.arange(X_.shape[0])
    inputs = [A_, X_, offset_, sind_, Y_, ix]
    # pyro.render_model(
    #     model,
    #     model_args=inputs,
    #     render_distributions=True,
    #     filename=f"{args.name}_diag.png",
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
        batch_size=X.shape[0] // S if not args.full_batch else None,
        num_particles=args.num_particles,
        jit=args.jit,
        gamma_positive_constraints=[0] if args.poscoef else [],
    )

    # extract tau
    params = Predictive(model, guide=guide, num_samples=50, return_sites=["_RETURN"])(
        *inputs, return_beta=True
    )
    betas, gammas = params["_RETURN"][:, ..., 0], params["_RETURN"][:, ..., 1]

    betas_means = betas.mean((0, 1))
    betas_std = betas.std((0, 1))
    gammas_means = gammas.mean((0, 1))
    gammas_std = gammas.std((0, 1))

    # save a json of all samples on sites, make sure to save as float
    params = Predictive(model, guide=guide, num_samples=50)(*inputs)
    sites = list(params.keys())
    results = {}
    for s in sites:
        results[s] = params[s].numpy().astype(float).tolist()
    with open(f"Plots_params/samples_{args.name}.json", "w") as io:
        json.dump(results, io)

    predictive_outputs = Predictive(
        model,
        guide=guide,
        num_samples=args.n_samples,
        return_sites=["_RETURN"],
    )
    outputs = predictive_outputs(*inputs, return_all_outcomes=True)["_RETURN"]

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

    fig.savefig(f"Plots_params/outcomes_{args.name}.png", bbox_inches="tight")

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
    fig.savefig(f"Plots_params/coefficients_{args.name}.png", bbox_inches="tight")

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
    fig.savefig(f"Plots_params/dos_{args.name}.png", bbox_inches="tight")

    os.makedirs("../Bayesian_models", exist_ok=True)
    torch.save(model.state_dict(), f"../Bayesian_models/Pyro_model_{args.name}.pt")
    torch.save(guide.state_dict(), f"../Bayesian_models/Pyro_guide_{args.name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--name", type=str, default="nn")
    parser.add_argument("--full_batch", default=False, action="store_true")
    parser.add_argument("--sample_weights", default=False, action="store_true")
    parser.add_argument("--nojit", default=True, action="store_false", dest="jit")
    parser.add_argument("--poscoef", default=False, action="store_true")
    args = parser.parse_args()

    # %% configure logger using args.name as filename, print time, also print stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"Plots_params/log_{args.name}.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"==== Starting Experiment {args.name} ====")
    main(args)
