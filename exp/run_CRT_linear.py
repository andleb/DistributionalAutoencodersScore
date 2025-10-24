# Stage-1 testing
################################################################################
SEED = 69
N_REPS = 100
B_BOOTSTRAP = 400

################################################################################

import copy
import os
import re

import gc

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys


def slurm_cpus():
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if var in os.environ:
            return int(os.environ[var])
    try:
        import psutil
        return len(psutil.Process().cpu_affinity())
    except Exception:
        return None


n_threads = slurm_cpus() or 1
print("Use", n_threads, "threads", flush=True)

os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"

import torch
torch.set_num_threads(n_threads)

import numpy as np
import scipy
from scipy import stats

import GPUtil

if torch.cuda.is_available():
    available_gpus = GPUtil.getAvailable(order='memory', limit=1)

    if available_gpus:
        selected_gpu = available_gpus[0]

        device = torch.device(f"cuda:{selected_gpu}")

        print(f"Using GPU: {selected_gpu} with the lowest memory usage.", flush=True)
    else:
        print("No GPUs available with low memory usage.", flush=True)
else:
    device = torch.device("cpu")
    print("Using CPU.", flush=True)

torch.manual_seed(SEED)

import joblib
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal
import math
import time

from causallearn.utils.cit import CIT
from scipy.stats import kstest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from causallearn.utils.KCI.KCI import KCI_UInd

sys.path += [
    "../src/DistributionalPrincipalAutoencoder",
    "../src/engression",
    "../src/PyTorch-VAE",
]

from dpa.dpa_fit import DPA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# ────────────────────────────────────────────────────────────────────────────
#  Matplotlib defaults  (unchanged, but condensed)
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


def extract_first_numbers_after_epoch(filename):
    with open(filename, 'r') as file:
        text = file.read()

    pattern = r'\[Epoch (\d+)\]\s+([-*\d.]+)'
    matches = re.findall(pattern, text)
    numbers = np.array([(float(m1), float(m2)) for m1, m2 in matches])
    return numbers


class EncoderModule(torch.nn.Module):
    def __init__(self, trained_model, standardize=False, x_mean=None, x_std=None):
        super(EncoderModule, self).__init__()
        self.trained_model = trained_model
        self.trained_model.raw = trained_model.model
        self.encoder = self.trained_model.model.encoder
        if standardize is None:
            self.standardize = self.trained_model.standardize
        else:
            self.standardize = standardize
        if x_mean is None:
            self.x_mean = self.trained_model.x_mean
        else:
            self.x_mean = x_mean
        if x_std is None:
            self.x_std = self.trained_model.x_std
        else:
            self.x_std = x_std

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


def _fit_gaussian_gp(u: np.ndarray, z: np.ndarray, jitter: float = 1e-3):
    """
    Fit N(u | μ(z), σ²(z)) via Gaussian‑process regression.
    """
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(z.shape[1])) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
    gpr.fit(z, u.ravel())
    mu, sigma = gpr.predict(z, return_std=True)
    sigma = np.maximum(sigma, jitter)
    return mu, sigma, gpr


def _hsic_stat(u: np.ndarray, x: np.ndarray) -> float:
    """
    Unconditional HSIC‑V statistic (biased) with analytic bandwidths.
    """
    kci = KCI_UInd(approx=True)
    _, stat = kci.compute_pvalue(u, x)
    return float(stat)


def double_crt(
        u: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        model,
        latent_dim: int,
        B: int = 500,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
        verbose: bool = False):
    """
    Return (p_value, observed_statistic, bootstrap_stats[B]).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, K = z.shape

    mu, sigma, _ = _fit_gaussian_gp(u, z)

    t_obs = _hsic_stat(u, x)

    z_full = np.zeros((n, latent_dim), dtype=np.float32)
    z_full[:, :K] = z
    z_full = torch.from_numpy(z_full).to(device)

    t_null = np.empty(B)
    model.model.eval()

    for b in range(B):
        u_b = mu + sigma * rng.standard_normal(n)
        u_b = u_b.reshape(-1, 1).astype(np.float32)

        with torch.no_grad():
            x_b = model.decode(
                z_full,
                mean=False,
                gen_sample_size=1
            ).cpu().numpy()

        if len(x_b.shape) > 2:
            x_b = x_b.squeeze()

        t_null[b] = _hsic_stat(u_b, x_b)

    p_val = (1 + np.sum(t_null >= t_obs)) / (B + 1)

    if verbose:
        print(f"Double‑CRT p = {p_val:.4g}   (B = {B})", flush=True)

    return p_val, t_obs, t_null


class DummyDecoder(torch.nn.Module):
    """Identity decoder for testing purposes."""

    def decode(self, z, *args, **kwargs):
        return z

    @property
    def model(self):
        return self

    def eval(self):
        return None


class DummyORD:
    """Identity decoder with Gaussian noise:  X = [Z, ε]."""

    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.model = self

    def eval(self):
        pass

    def decode(self, z_full, mean=False, gen_sample_size=1):
        z = z_full[:, :1]
        eps = self.sigma * torch.randn_like(z_full[:, :1])
        return torch.cat([z, eps], dim=1)


class DummyORDLeak(DummyORD):
    def __init__(self, beta=0.3, sigma=0.1):
        super().__init__(sigma)
        self.beta = beta

    def decode(self, z_full, mean=False, gen_sample_size=1):
        z = z_full[:, :1]
        u = z_full[:, 1:2]
        eps = self.sigma * torch.randn_like(z)
        x0 = z + self.beta * u + eps

        return torch.cat([x0, eps], dim=1)


X_line = torch.load("../data/indep/gauss_line/X_line.pt")

n_feats = 2
k = 2
num_layer = 4
hidden_dim = 100
latent_dims = list(range(k + 1))[::-1]

prefix = f"../res/indep/gauss_line/{k}k{num_layer}l{hidden_dim}h/"

print(f"Model prefix: {prefix}", flush=True)

dpaModel_line = DPA(beta=1.,
                    dist_enc="deterministic",
                    dist_dec="stochastic",
                    data_dim=n_feats,
                    latent_dims=latent_dims,
                    num_layer=num_layer,
                    hidden_dim=hidden_dim,
                    noise_dim=100,
                    resblock=True,
                    standardize=False,
                    device=device,
                    seed=SEED
                    )

dpaModel_line.model.load_state_dict(torch.load(f"{prefix}/model_{4000}.pt",
                                               ),
                                    )

dpaModel_line.model = dpaModel_line.model.to(device)
dpaModel_line.model.eval();
dpaeFunc_line = EncoderModule(dpaModel_line)

model = dpaModel_line
data = X_line
print("Running double CRT with the small model ...", flush=True)

with torch.no_grad():
    z_lat = model.encode(data.to(device))

Z1 = z_lat[:, :1].cpu().numpy()
U2 = z_lat[:, 1:2].cpu().numpy()
latent_dim = z_lat.shape[1]

pvals = []
t_obss = []
t_nulls = []

rng = np.random.default_rng(SEED)

for iter in range(N_REPS):

    print(f"Iteration {iter + 1}/{N_REPS} ...", flush=True)

    with torch.no_grad():
        X_obs = model.decode(
            z_lat.to(device),
            mean=False,
            gen_sample_size=1
        ).squeeze().cpu().numpy()

    p_val, t_obs, t_null = double_crt(
        u=U2,
        x=X_obs,
        z=Z1,
        model=model,
        latent_dim=latent_dim,
        B=B_BOOTSTRAP,
        device=device,
        rng=rng,
        verbose=True
    )

    pvals.append(p_val)
    t_obss.append(t_obs)
    t_nulls.append(t_null)

    print(f"p_value = {p_val}", flush=True)

RES_PREFIX = f"../res/indep/gauss_line/crt_{SEED}_l{num_layer}h{hidden_dim}/"

if not os.path.exists(RES_PREFIX):
    os.makedirs(RES_PREFIX)

results = {
    "pvals": np.array(pvals),
    "t_obss": np.array(t_obss),
    "t_nulls": np.array(t_nulls)
}

joblib.dump(results, f"{RES_PREFIX}/results_{N_REPS}reps_{B_BOOTSTRAP}boot.pkl")

plt.scatter(np.linspace(0, 1, len(pvals), endpoint=False), np.sort(pvals), s=12);
plt.plot([0, 1], [0, 1], lw=1);
plt.xlabel("Uniform(0,1) quantile");
plt.ylabel("Observed p‑value");
plt.savefig(f"{RES_PREFIX}/pvals_{N_REPS}reps_{B_BOOTSTRAP}boot.png");
plt.show();
plt.close();

pvals = np.array(pvals)
D_KS, p_KS = kstest(pvals, "uniform")

with open(f"{RES_PREFIX}/ks_stats.txt", "w") as f:
    f.write(f"KS stat = {D_KS :.3f},  p = {p_KS :.3e}")
    f.write(f"% p < 0.05 = {(pvals < 0.05).mean()*100:.1f}")
