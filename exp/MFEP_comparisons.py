#!/usr/bin/env python
"""
Batch-train DPA / AE / VAE many times and quantify how well each parametrises
the Müller–Brown MFEP.
"""

# ────────────────────────────────────────────────────────────────────────────
#  Imports
# ────────────────────────────────────────────────────────────────────────────
import json
import pathlib
import random
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path += ["../", "../src/DistributionalPrincipalAutoencoder", "../src/engression", "../src/mlcolvar",
    "../src/PyTorch-VAE", ]
from engression.data import loader
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.data import DictModule
from mlcolvar.cvs import VariationalAutoEncoderCV
from utils import mfep_utils
from models.beta_vae import BetaVAE
from models.betatc_vae import BetaTCVAE

# ────────────────────────────────────────────────────────────────────────────
#  Matplotlib defaults
# ────────────────────────────────────────────────────────────────────────────
plt.style.use('default')

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['contour.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['axes.linewidth'] = 2

mpl.rcParams['axes.grid'] = False
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linewidth'] = 1

mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 1.0
mpl.rcParams['legend.edgecolor'] = 'black'

mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

class BetaVAE_FC(BetaVAE):
    """
    Thin wrapper around AntixK/PyTorch‑VAE's `BetaVAE` that swaps the
    convolutional blocks for the same 100‑100 fully‑connected topology
    used by AE/VAE in this script.
    """

    def __init__(self, in_dim: int, latent_dim: int = 2, hidden_dims: list[int] = (100, 100), beta: int = 4):
        super().__init__(in_channels=1, latent_dim=latent_dim, hidden_dims=[1], beta=beta, loss_type='H')

        # ── replace Conv encoder/decoder with MLPs ──────────────────
        enc = []
        last = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(last, h), nn.ELU()]
            last = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_var = nn.Linear(last, latent_dim)

        dec = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(last, h), nn.ELU()]
            last = h
        dec.append(nn.Linear(last, in_dim))
        self.decoder_input = nn.Identity()
        self.decoder = nn.Sequential(*dec)
        self.final_layer = nn.Identity()

    def encode(self, x):
        h = self.encoder(x)
        return [self.fc_mu(h), self.fc_var(h)]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]


class BetaTCVAE_FC(BetaTCVAE):
    """
    Re‑uses PyTorch‑VAE's `BetaTCVAE` loss and bookkeeping but swaps the
    original Conv‑Net encoder/decoder for the same 100‑100 MLP used by the
    other (vector) benchmarks.
    """

    def __init__(self, in_dim: int, latent_dim: int = 2, hidden_dims: list[int] = (100, 100), anneal_steps: int = 200,
                 alpha: float = 1., beta: float = 6., gamma: float = 1.):
        super().__init__(in_channels=1, latent_dim=latent_dim, hidden_dims=[1], anneal_steps=anneal_steps, alpha=alpha,
                         beta=beta, gamma=gamma)

        # ── fully‑connected encoder ──────────────────────────────────
        enc = []
        last = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(last, h), nn.ELU()]
            last = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_var = nn.Linear(last, latent_dim)

        # ── fully‑connected decoder ──────────────────────────────────
        dec = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(last, h), nn.ELU()]
            last = h
        dec.append(nn.Linear(last, in_dim))
        self.decoder_input = nn.Identity()
        self.decoder = nn.Sequential(*dec)
        self.final_layer = nn.Identity()

    def encode(self, x):
        h = self.encoder(x)
        return [self.fc_mu(h), self.fc_var(h)]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var, z]


def _as_tensor(batch):
    """
    Robustly convert whatever the DataLoader returns into a 2‑D tensor
    [batch_size × n_features] without adding spurious dimensions.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            return batch[0]
        return torch.stack(list(batch))
    raise TypeError(f"Unexpected batch type: {type(batch)}")


class EncoderModule(torch.nn.Module):
    def __init__(self, trained_model, standardize=False, x_mean=None, x_std=None):
        super(EncoderModule, self).__init__()
        self.trained_model = trained_model
        self.trained_model.raw = trained_model.model
        self.encoder = self.trained_model.model.encoder
        self.standardize = self.trained_model.standardize
        self.x_mean = self.trained_model.x_mean
        self.x_std = self.trained_model.x_std
        self.device = next(self.encoder.parameters()).device

    def forward(self, x, k=None, mean=True, gen_sample_size=100):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        x = x.to(self.device)

        if not x.requires_grad:
            x.requires_grad_(True)

        if self.standardize:
            x = (x - self.x_mean.to(self.device)) / self.x_std.to(self.device)

        if k is None:
            k = self.trained_model.latent_dim
        if self.trained_model.encoder_k:
            x = self.trained_model.get_k_embedding(k, x)

        z = self.encoder(x)
        return z[:, :k]


class ForwardCVModule(torch.nn.Module):
    def __init__(self, vae_cv: VariationalAutoEncoderCV):
        super(ForwardCVModule, self).__init__()
        self.norm_in = vae_cv.norm_in
        self.encoder = vae_cv.encoder
        self.mean_nn = vae_cv.mean_nn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        return self.mean_nn(x)


# ────────────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────────────
N_SEEDS = 25
SEED_LIST = list(range(42, 42 + N_SEEDS))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS_DPA = 1200
N_EPOCHS_AE = 1200
N_EPOCHS_VAE = 1200
PARAM_TEST_KW = dict(n_steps=250, iters=200, tol=1e-7, batch=True)

SAVE_DIR = pathlib.Path("../res/MFEP_tests-betatc")
FIG_DIR = SAVE_DIR / "figs"
for d in (SAVE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUNS_CSV = SAVE_DIR / "all_runs.csv"
SUMMARY_CSV = SAVE_DIR / "summary_mean_std.csv"
RAW_JSON = SAVE_DIR / "all_runs.json"

# ────────────────────────────────────────────────────────────────────────────
#  Data loading
# ────────────────────────────────────────────────────────────────────────────
mfep_xy = np.load("../res/aux/MB/mfep_xy.npy")
start_xy = mfep_xy[0]

filenames = ["../src/mlcolvar/docs/notebooks/tutorials/data/muller-brown/unbiased/high-temp/COLVAR"]
dataset, df = create_dataset_from_files(filenames, filter_args={"regex": "p.x|p.y"}, return_dataframe=True)
n_input = dataset[:]["data"].shape[1]

datamodule = DictModule(dataset, lengths=[0.8, 0.2], random_split=True)
X_loader = loader.make_dataloader(dataset[:]["data"].to(DEVICE), batch_size=5000, shuffle=True, device="cpu")


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_savefig(fig, path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"[WARN] Could not save figure {path}: {e}", flush=True)
    finally:
        plt.close(fig)


def add_failure(all_runs, model, seed, exc):
    all_runs.append({"model": model, "seed": seed, "status": "failed", "error": str(exc).replace("\n", " "), })


# ────────────────────────────────────────────────────────────────────────────
#  Main loop
# ────────────────────────────────────────────────────────────────────────────
all_runs = []

for seed in SEED_LIST:
    print(f"\n=== Seed {seed} ===", flush=True)
    try:
        set_all_seeds(seed)
        dpa = DPA(beta=2, dist_enc="deterministic", dist_dec="stochastic", data_dim=n_input, latent_dims=[2, 1, 0],
            num_layer=4, hidden_dim=100, noise_dim=100, resblock=True, standardize=False, device=DEVICE, seed=seed, )
        dpa.train(X_loader, batch_size=len(dataset) // 2, num_epochs=N_EPOCHS_DPA, lr=5e-4, save_model_every=999999,
            save_dir=SAVE_DIR / "tmp", save_loss=False, )
        enc_dpa = EncoderModule(dpa).to(DEVICE)

        scores = mfep_utils.parametrisation_test(encoder=enc_dpa.to(DEVICE), mfep_xy=mfep_xy, start_minimum_xy=start_xy,
            device=DEVICE, **PARAM_TEST_KW, grad=True, )

        all_runs.append({"model": "DPA", "seed": seed, "status": "ok", **scores})

        fig, ax = plt.subplots()
        ax.scatter(*mfep_xy.T, s=1, c="purple", label="MFEP")
        ax.scatter(*scores["reconstructed_path"].T, s=10, c="red", label="DPA")
        ax.scatter(*scores["reconstructed_path_grad"].T, s=10, c="orange", alpha=0.75, label="DPA-grad")
        ax.legend()
        safe_savefig(fig, FIG_DIR / f"dpa_seed_{seed}.png")

    except Exception as exc:
        traceback.print_exc()
        add_failure(all_runs, "DPA", seed, exc)

    try:
        set_all_seeds(seed)
        modelBETA = BetaVAE_FC(in_dim=n_input, latent_dim=2, beta=4).to(DEVICE)
        opt = torch.optim.Adam(modelBETA.parameters(), lr=1e-3)

        for epoch in range(N_EPOCHS_VAE):
            for batch in X_loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(DEVICE)

                opt.zero_grad()
                recons, x_in, mu, log_var = modelBETA(xb)
                loss_dict = modelBETA.loss_function(recons, x_in, mu, log_var, M_N=1.0)
                loss_dict["loss"].backward()
                opt.step()

        enc_beta = lambda x: modelBETA.encode(x)[0]
        scores = mfep_utils.parametrisation_test(encoder=enc_beta, mfep_xy=mfep_xy, start_minimum_xy=start_xy,
            device=DEVICE, **PARAM_TEST_KW, grad=False)

        all_runs.append({"model": "β‑VAE", "seed": seed, "status": "ok", **scores})

        fig, ax = plt.subplots()
        ax.scatter(*mfep_xy.T, s=1, c="purple", label="MFEP")
        ax.scatter(*scores["reconstructed_path"].T, s=10, c="red", label="β‑VAE")
        ax.legend()
        safe_savefig(fig, FIG_DIR / f"beta_vae_seed_{seed}.png")

    except Exception as exc:
        traceback.print_exc()
        add_failure(all_runs, "β‑VAE", seed, exc)

    try:
        set_all_seeds(seed)
        modelTC = BetaTCVAE_FC(in_dim=n_input, latent_dim=2).to(DEVICE)

        X_loader_tc = torch.utils.data.DataLoader(X_loader.dataset, batch_size=256, shuffle=True, drop_last=True,
            num_workers=0)
        dataset_size = len(X_loader_tc.dataset)
        opt = torch.optim.Adam(modelTC.parameters(), lr=1e-3)

        for epoch in range(N_EPOCHS_VAE):
            for batch in X_loader_tc:
                xb = _as_tensor(batch).to(DEVICE)
                opt.zero_grad()

                recons, x_in, mu, log_var, z = modelTC(xb)
                M_N = xb.size(0) / dataset_size
                loss_dict = modelTC.loss_function(recons, x_in, mu, log_var, z, M_N=M_N)
                loss_dict["loss"].backward()
                opt.step()

        enc_tc = lambda x: modelTC.encode(x)[0]
        scores = mfep_utils.parametrisation_test(encoder=enc_tc, mfep_xy=mfep_xy, start_minimum_xy=start_xy,
            device=DEVICE, **PARAM_TEST_KW, grad=False)

        all_runs.append({"model": "β‑TC‑VAE", "seed": seed, "status": "ok", **scores})

        fig, ax = plt.subplots()
        ax.scatter(*mfep_xy.T, s=1, c="purple", label="MFEP")
        ax.scatter(*scores["reconstructed_path"].T, s=10, c="orange", label="β‑TC‑VAE")
        ax.legend()
        safe_savefig(fig, FIG_DIR / f"betatc_vae_seed_{seed}.png")

    except Exception as exc:
        traceback.print_exc()
        add_failure(all_runs, "β‑TC‑VAE", seed, exc)

    try:
        set_all_seeds(seed)
        modelAE = AutoEncoderCV([n_input, 100, 100, 2], options={"encoder": {"activation": "shifted_softplus"},
                                                                 "decoder": {"activation": "shifted_softplus"}}, ).to(
            DEVICE)
        trainerAE = lightning.Trainer(max_epochs=N_EPOCHS_AE, logger=None, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, )
        trainerAE.fit(modelAE, datamodule)

        scores = mfep_utils.parametrisation_test(encoder=modelAE.to(DEVICE), mfep_xy=mfep_xy, start_minimum_xy=start_xy,
            device=DEVICE, **PARAM_TEST_KW, grad=False, )
        all_runs.append({"model": "AE", "seed": seed, "status": "ok", **scores})

        fig, ax = plt.subplots()
        ax.scatter(*mfep_xy.T, s=1, c="purple", label="MFEP")
        ax.scatter(*scores["reconstructed_path"].T, s=10, c="red", label="AE")
        ax.legend()
        safe_savefig(fig, FIG_DIR / f"ae_seed_{seed}.png")

    except Exception as exc:
        traceback.print_exc()
        add_failure(all_runs, "AE", seed, exc)

    try:
        set_all_seeds(seed)
        modelVAE = VariationalAutoEncoderCV(n_cvs=2, encoder_layers=[n_input, 100, 100],
            options={"encoder": {"activation": "elu"}, "decoder": {"activation": "elu"}}, ).to(DEVICE)
        trainerVAE = lightning.Trainer(max_epochs=N_EPOCHS_VAE, logger=None, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, )
        trainerVAE.fit(modelVAE, datamodule)

        enc_vae = ForwardCVModule(modelVAE).to(DEVICE)
        scores = mfep_utils.parametrisation_test(encoder=enc_vae.to(DEVICE), mfep_xy=mfep_xy, start_minimum_xy=start_xy,
            device=DEVICE, **PARAM_TEST_KW, grad=False, )
        all_runs.append({"model": "VAE", "seed": seed, "status": "ok", **scores})

        fig, ax = plt.subplots()
        ax.scatter(*mfep_xy.T, s=1, c="purple", label="MFEP")
        ax.scatter(*scores["reconstructed_path"].T, s=10, c="red", label="VAE")
        ax.legend()
        safe_savefig(fig, FIG_DIR / f"vae_seed_{seed}.png")

    except Exception as exc:
        traceback.print_exc()
        add_failure(all_runs, "VAE", seed, exc)

# ────────────────────────────────────────────────────────────────────────────
#  Save all-runs table
# ────────────────────────────────────────────────────────────────────────────
df_runs = pd.DataFrame(all_runs)
df_runs.to_csv(RUNS_CSV, index=False)

try:
    with open(RAW_JSON, "w") as fp:
        json.dump(all_runs, fp, indent=2)
except Exception as e:
    pass

# ────────────────────────────────────────────────────────────────────────────
#  Summary: drop worst Chamfer, report mean & std
# ────────────────────────────────────────────────────────────────────────────
summary_rows = []
for model, grp in df_runs[df_runs.status == "ok"].groupby("model"):

    if len(grp) < 3:
        continue

    keep = grp.nsmallest(len(grp) - 1, "chamfer")
    means = keep.mean(numeric_only=True).add_suffix("_mean")
    stds = keep.std(numeric_only=True, ddof=1).add_suffix("_std")

    row = pd.concat([pd.Series({"model": model}, dtype="object"), means, stds])

    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows).sort_values("model")
df_summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nPer-run results  → {RUNS_CSV}")
print(f"Mean / std table → {SUMMARY_CSV}")
