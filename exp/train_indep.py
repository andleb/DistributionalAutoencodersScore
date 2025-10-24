# Simple datasets training
################################################################################
SEED = 42
STANDARDIZE = False
################################################################################

import argparse

import os
import re

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


n_threads = min(slurm_cpus(), 4)
print("Use", n_threads, "threads", flush=True)

os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"

import torch

torch.set_num_threads(n_threads)

import numpy as np

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

sys.path += ["../src/DistributionalPrincipalAutoencoder", "../src/engression", "../src/mlcolvar",
             "../src/PyTorch-VAE", ]

from dpa.dpa_fit import DPA
from engression.data import loader
from mlcolvar.utils.plot import plot_isolines_2D

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

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


parser = argparse.ArgumentParser(description="Train on specific dataset.")
parser.add_argument('--n_samples', type=int, default=10000, help='Number of data points to generate.')
parser.add_argument('--dataset', type=str, default='line',
                    choices=['parabola', 'exponential', 'helix_slice', 'grid_sum', 'gauss_line'],
                    help='Which dataset to use.')
args = parser.parse_args()


def make_linear_line(n=3_000, noise_std=0.1):
    z = np.random.randn(n, 1)
    eps = noise_std * np.random.randn(n, 1)
    X = np.concatenate([z, eps], axis=1)

    return X.astype(np.float32)


def parabola(n=2000):
    t = np.random.uniform(-1, 1, n)[:, None]
    return np.hstack([t, t ** 2])


def exponential(n=2000):
    t = np.random.uniform(-3, 3, n)[:, None]
    return np.hstack((t, np.exp(t)))


def helix_slice(n=2000):
    t = np.random.uniform(0, 4 * np.pi, n)[:, None]
    return np.hstack((t, np.cos(t)))


def grid_sum(n_side=60, *args, **kwargs):
    s = np.linspace(-1, 1, n_side)
    xx, yy = np.meshgrid(s, s)
    Z = np.column_stack([xx.ravel(), yy.ravel()])
    U = Z.sum(axis=1, keepdims=True)
    return np.hstack((Z, U))


GENS = dict(parabola=parabola, exponential=exponential, helix_slice=helix_slice, grid_sum=grid_sum,
            gauss_line=make_linear_line)

X_data = torch.tensor(GENS[args.dataset](n=args.n_samples), dtype=torch.float32)

DATA_PRREFIX = f"../data/indep/{args.dataset}/"
os.makedirs(DATA_PRREFIX, exist_ok=True)

plt.scatter(X_data[:, 0], X_data[:, 1])
plt.savefig(f"{DATA_PRREFIX}/data_{args.n_samples}_{SEED}.png");
plt.show();
plt.close()

torch.save(X_data, f"{DATA_PRREFIX}/data_{args.n_samples}_{SEED}.pt")

X_data_loader = loader.make_dataloader(X_data, device="cpu")

n_feats = X_data.shape[1]
k = n_feats
num_layer = 4
hidden_dim = 100
latent_dims = list(range(k + 1))[::-1]

prefix = f"../res/indep/{args.dataset}/{k}k{num_layer}l{hidden_dim}h/"

print(f"Model prefix: {prefix}", flush=True)

dpaModel = DPA(beta=1., dist_enc="deterministic", dist_dec="stochastic", data_dim=n_feats, latent_dims=latent_dims,
               num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=hidden_dim, resblock=True, standardize=STANDARDIZE,
               device=device, seed=SEED)

print("Training", flush=True)
if STANDARDIZE:
    dpaModel._standardize_data_and_record_stats(X_data.to(device));

dpaModel.train(x=X_data_loader, num_epochs=2000,

               batch_size=len(X_data),

               lr=5e-4, save_model_every=100, print_every_nepoch=100, save_dir=prefix, save_loss=True, )

print("Plotting the results", flush=True)
dpaModel.model = dpaModel.model.to(device)
dpaModel.model.eval();
dpaeFunc = EncoderModule(dpaModel)

a = extract_first_numbers_after_epoch(f"{prefix}/log.txt")
plt.plot(a[1:, 0], a[1:, 1])
argmin = np.argmin(a[1:, 1])
plt.axvline(a[argmin + 1, 0], c="r")
plt.axhline(0, c="k", ls="--");
plt.savefig(f"{prefix}/loss.png");
plt.show();
plt.close()

dpaModel.plot_energy_loss(X_data.to(device), xscale='linear', save_dir=f"{prefix}/energy_loss.png")

dpaModel.plot_mse(X_data.to(device), xscale='linear', save_dir=f"{prefix}/mse.png")

limits = ((X_data[:, 0].min() * 1.5, X_data[:, 0].max() * 1.5), (X_data[:, 1].min() * 1.5, X_data[:, 1].max() * 1.5),)
n_components = X_data.shape[1]
fig, axs = plt.subplots(1, n_components, figsize=(10 * n_components, 8))

if n_components == 1:
    axs = [axs]
for i in range(n_components):
    ax = axs[i]
    plot_isolines_2D(dpaeFunc, component=i, levels=25, ax=ax, limits=limits)
    plot_isolines_2D(dpaeFunc, component=i, mode='contour', levels=25, ax=ax, limits=limits)

    ax.scatter(X_data[:, 0], X_data[:, 1], s=1, c="k", alpha=0.5)

plt.savefig(f"{prefix}/level-sets.png");
plt.show();
plt.close()
