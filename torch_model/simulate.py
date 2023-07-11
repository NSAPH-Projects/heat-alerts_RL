# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader



class Model(pl.LightningModule):
    def __init__(self, n: int, s: int, dx: int, dw: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.n = n
        self.s = s
        self.dx = dx
        self.dw = dw
        self.lr = lr

        self.delta_beta = torch.nn.Parameter(torch.randn(dw, dx))
        self.delta_gamma = torch.nn.Parameter(torch.randn(dw, dx))
        self.unstruct_beta = torch.nn.Parameter(torch.randn(s, dx))
        self.unstruct_gamma = torch.nn.Parameter(torch.randn(s, dx))
        self.omega_beta = torch.nn.Parameter(torch.randn(dx))
        self.omega_gamma = torch.nn.Parameter(torch.randn(dx))

    def beta(self, W: torch.Tensor, sind: torch.Tensor) -> torch.Tensor:
        return W @ self.delta_beta + self.omega_beta[None] * self.unstruct_beta[sind]

    def gamma(self, W: torch.Tensor, sind: torch.Tensor) -> torch.Tensor:
        return W @ self.delta_gamma + self.omega_gamma[None] * self.unstruct_gamma[sind]

    def logits_tau(
        self, W: torch.Tensor, X: torch.Tensor, sind: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.gamma(W, sind)
        return (X * gamma).sum(-1)

    def forward(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        sind: torch.Tensor,
        loff: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        beta = self.beta(W, sind)
        logits_tau = self.logits_tau(W, X, sind)
        log_lam = (X * beta).sum(-1)
        log_lam = log_lam + A * torch.log(1 - logits_tau.sigmoid()) + loff
        return log_lam, logits_tau

    def training_step(self, batch, _):
        W, X, sind, loff, A, Y, _ = batch
        log_lam, _ = self.forward(W, X, sind, loff, A)
        loss = torch.nn.functional.poisson_nll_loss(log_lam, Y, log_input=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        W, X, sind, loff, A, Y, true_tau = batch
        log_lam, logits_tau = self.forward(W, X, sind, loff, A)
        loss = torch.nn.functional.poisson_nll_loss(log_lam, Y, log_input=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        tau = logits_tau.sigmoid()
        self.tau_loss = torch.nn.functional.mse_loss(tau, true_tau)
        self.log("tau_loss", self.tau_loss, prog_bar=True, on_epoch=True, on_step=False)

        # make a plot of true vs predicted tau
        if batch_idx == 0:
            fig, ax = plt.subplots()
            ax.scatter(true_tau, tau)
            ax.set_xlabel("true tau")
            ax.set_ylabel("predicted tau")
            self.logger.experiment.add_figure("tau", fig, self.current_epoch)
            plt.close(fig)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


# %% make tensor datasets
class Data(pl.LightningDataModule):
    def __init__(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        sind: torch.Tensor,
        P: torch.Tensor,
        A: torch.Tensor,
        Y: torch.Tensor,
        true_tau: torch.Tensor,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        n = X.shape[0]
        dx = X.shape[1]
        dw = W.shape[1]

        # train ixs
        test_frac = 0.2
        n_train = int(n * (1 - test_frac))
        ix = np.arange(n)
        np.random.shuffle(ix)
        self.ix_train = ix[:n_train]
        self.ix_test = ix[n_train:]

        self.W = W[sind]
        self.X = X
        self.sind = sind
        self.loff = torch.log(P[sind])
        self.A = A
        self.Y = Y
        self.true_tau = true_tau
        self.batch_size = batch_size
        self.num_workers = num_workers

        # train datset
        self.train_dataset = TensorDataset(
            self.W[self.ix_train],
            self.X[self.ix_train],
            self.sind[self.ix_train],
            self.loff[self.ix_train],
            self.A[self.ix_train],
            self.Y[self.ix_train],
            self.true_tau[self.ix_train],
        )

        # val dataset
        self.val_dataset = TensorDataset(
            self.W[self.ix_test],
            self.X[self.ix_test],
            self.sind[self.ix_test],
            self.loff[self.ix_test],
            self.A[self.ix_test],
            self.Y[self.ix_test],
            self.true_tau[self.ix_test],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

def main():
    # %% read edge
    edge_list = np.loadtxt("torch_model/edge_list.csv", delimiter=",", dtype=int)
    with open("bayesian_model/simulated_data/sim.json", "r") as f:
        sim_data = json.load(f)

    # %%
    sim_data.keys()
    # %%
    delta_beta = torch.FloatTensor(sim_data["delta_beta"])
    delta_gamma = torch.FloatTensor(sim_data["delta_gamma"])
    W = torch.FloatTensor(sim_data["W"])
    X = torch.FloatTensor(sim_data["X"])
    omega_beta = torch.FloatTensor(sim_data["omega_beta"])
    omega_gamma = torch.FloatTensor(sim_data["omega_gamma"])
    unstruct_beta = torch.FloatTensor(sim_data["beta_unstruct"])
    unstruct_gamma = torch.FloatTensor(sim_data["gamma_unstruct"])
    sind = torch.LongTensor(sim_data["sind"]) - 1
    P = torch.FloatTensor(sim_data["P"])
    A = torch.FloatTensor(sim_data["A"])


    # %% compute time varying coefficients
    beta = W @ delta_beta + omega_beta[None] * unstruct_beta
    gamma = W @ delta_gamma + omega_gamma[None] * unstruct_gamma


    # %%
    tau = torch.sigmoid((X * gamma[sind]).sum(-1))
    # tau = 0.1 * torch.ones_like(tau) # !!
    lam = torch.exp((X * beta[sind]).sum(-1))
    # lam = torch.ones_like(lam) # !!
    mu = P[sind] * lam * torch.where(A > 0, 1 - tau, 1.0)
    Y = torch.poisson(mu)

    # plt.subplot(2, 2, 1)
    # plt.hist(tau)
    # plt.title("tau")
    # plt.subplot(2, 2, 2)
    # plt.hist(lam)
    # plt.title("lam")
    # plt.subplot(2, 2, 3)
    # plt.hist(mu)
    # plt.title("mu")
    # plt.subplot(2, 2, 4)
    # plt.hist(Y)
    # plt.title("Y")


    # %%
    n = sim_data["N"]
    s = sim_data["S"]
    dx = sim_data["DX"]
    dw = sim_data["DW"]


    # %%
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=100,
    )
    model = Model(n, s, dx, dw)
    data = Data(W, X, sind, P, A, Y, true_tau=tau)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()