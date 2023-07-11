import json

import matplotlib.pyplot as plt
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from tqdm import tqdm


def train(model, guide, data, lr, n_steps):
    pyro.clear_param_store()
    adam_params = {"lr": lr}
    adam = pyro.optim.Adam(adam_params)
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=10, vectorize_particles=True))

    for step in range(n_steps):
        loss = svi.step(*data)
        if step == 0 or (step % (n_steps // 10)) == 0:
            print("[iter {}]  loss: {:.8f}".format(step, loss))


def train2(model, guide, data, lr, n_epochs, batch_size):
    # initialize
    pyro.clear_param_store()
    loss_fn = pyro.infer.Trace_ELBO()(model, guide)
    loss_fn(*data)  # initialize parameters
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

    for epoch in range(n_epochs):
        epoch_losses = []
        for batch in tqdm(dataloader):
            opt.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        loss = sum(epoch_losses) / len(epoch_losses)
        if epoch == 0 or ((epoch + 1) % (n_epochs // 10)) == 0:
            print("[epoch {}/{}]  loss: {:.8f}".format(epoch + 1, n_epochs, loss))


class Model(pyro.nn.PyroModule):
    def __init__(self, W, N, S, DX, DW):
        super().__init__()
        self.linWbeta = PyroModule[torch.nn.Linear](DW, DX)
        self.linWbeta.weight = PyroSample(
            dist.Normal(0, 1).expand([DX, DW]).to_event(2)
        )
        self.linWbeta.bias = PyroSample(dist.Normal(0, 1).expand([DX]).to_event(1))

        self.linWgamma = pyro.nn.PyroModule[torch.nn.Linear](DW, DX)
        self.linWgamma.weight = PyroSample(
            dist.Normal(0, 1).expand([DX, DW]).to_event(2)
        )
        self.linWgamma.bias = PyroSample(dist.Normal(0, 1).expand([DX]).to_event(1))

        unstruct = PyroModule()
        unstruct.beta = PyroSample(dist.Normal(0, 1).expand([S, DX]).to_event(2))
        unstruct.gamma = PyroSample(dist.Normal(0, 1).expand([S, DX]).to_event(2))
        unstruct.omega_beta = PyroSample(dist.Gamma(1.1, 1.1).expand([DX]).to_event(1))
        unstruct.omega_gamma = PyroSample(dist.Gamma(1.1, 1.1).expand([DX]).to_event(1))
        self.unstruct = unstruct
        self.register_buffer("W", W)
        self.N = N

    def forward(self, A, X, offset, sind, Y=None):
        B, DX = X.shape
        W = self.W
        _, DW = W.shape

        spatial_beta = self.linWbeta(W)
        spatial_gamma = self.linWgamma(W)
        beta = spatial_beta + self.unstruct.omega_beta * self.unstruct.beta
        gamma = spatial_gamma + self.unstruct.omega_gamma * self.unstruct.gamma

        # predictors
        tau = torch.sigmoid((gamma[sind, :] * X).sum(-1))
        lam = torch.exp((beta[sind, :] * X).sum(-1))
        mu = offset * lam * (1.0 - A * tau)

        # likelihood, Y ~ poisson(mu)
        with pyro.plate("data", self.N, B):
            obs = pyro.sample("obs", dist.Poisson(mu), obs=Y)

        return torch.stack([tau, lam, mu], axis=1)


def main():
    # read sim data
    with open("bayesian_model/simulated_data/sim.json", "r") as f:
        sim_data = json.load(f)

    # make fitting data
    A = torch.tensor(sim_data["A"], dtype=torch.float32)
    Y = torch.tensor(sim_data["Y"], dtype=torch.float32)
    X = torch.tensor(sim_data["X"], dtype=torch.float32)
    W = torch.tensor(sim_data["W"], dtype=torch.float32)
    sind = torch.tensor(sim_data["sind"], dtype=torch.long) - 1

    # make offset
    # compute mean by each element of sind then left join
    df = pd.DataFrame({"other_hosps": Y, "sind": sind})
    tmp = (
        df.groupby("sind")
        .mean()
        .reset_index()
        .rename(columns={"other_hosps": "mean_other_hosps"})
    )
    locmeans = df.merge(tmp, on="sind", how="left")
    offset = torch.tensor(locmeans.mean_other_hosps.values, dtype=torch.float32)

    # render pyro model
    N, DX = X.shape
    S, DW = W.shape
    model = Model(W, N, S, DX, DW)
    pyro.render_model(
        model,
        model_args=(A, X, offset, sind, Y),
        render_distributions=True,
        filename="bayesian_model/fit_simulation_pyro_diag.png",
    )

    # inference with autoguide
    # guide = pyro.infer.autoguide.AutoDelta(model)
    # guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    # guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model, rank=500)
    inputs = [A, X, offset, sind, Y]

    # train(model, guide, inputs, lr=0.003, n_steps=5000)
    train2(model, guide, inputs, lr=0.003, n_epochs=10, batch_size=X.shape[0] // W.shape[0])

    # extract tau
    num_samples = 1
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=("_RETURN",),
    )
    svi_samples = predictive(A, X, offset, sind)

    # plot real vs estimated tau
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    for i, var in enumerate(("tau", "lam", "mu")):
        v = svi_samples["_RETURN"][0, :, i]
        ax[i].scatter(sim_data[var], v, alpha=0.1)
        ax[i].set_xlabel(f"real {var}")
        ax[i].set_ylabel(f"estimated {var}")
        ax[i].set_title(f"real vs estimated {var}")

    fig.savefig("bayesian_model/fit_simulation_pyro.png")


if __name__ == "__main__":
    main()
