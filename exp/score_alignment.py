import glob
import math
import os
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=mpl._api.deprecation.MatplotlibDeprecationWarning)

use_threads = 32


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
print(f"Nthreads {n_threads}")
n_threads = min(n_threads, use_threads)
print("Use", n_threads, "threads")

import torch

os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
torch.set_num_threads(n_threads)

import numpy as np

import GPUtil

if torch.cuda.is_available():

    available_gpus = GPUtil.getAvailable(order='memory', limit=1)

    if available_gpus:
        selected_gpu = available_gpus[0]

        device = torch.device(f"cuda:{selected_gpu}")

        print(f"Using GPU: {selected_gpu} with the lowest memory usage.")
    else:
        print("No GPUs available with low memory usage.")
else:
    device = torch.device("cpu")

torch.manual_seed(42);

sys.path.append("src/DistributionalPrincipalAutoencoder")
sys.path.append("src/engression")
sys.path.append("src/mlcolvar")
sys.path.append("src/PyTorch-VAE")

from dpa.dpa_fit import DPA

_TINY = 1e-12


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


def _gradient_norm(encoder: torch.nn.Module, xy: np.ndarray, component: int | None, device="cpu") -> np.ndarray:
    pts = torch.tensor(xy, dtype=torch.float32, device=device, requires_grad=True)
    out = encoder(pts)
    if component is not None:
        out = out[:, component]
    grad, = torch.autograd.grad(out.sum(), pts)
    return grad.norm(dim=1).detach().cpu().numpy()


def levelset_stats(encoder, density_fn, contour_pts, component=None, device="cpu"):
    """Return {'mass':Z, 'center':c, 'variance':V} for one discretized isoline."""

    ds = np.linalg.norm(np.roll(contour_pts, -1, axis=0) - contour_pts, axis=1)
    pdf = density_fn(contour_pts)
    gnorm = _gradient_norm(encoder, contour_pts, component, device)
    w = pdf * ds / np.maximum(gnorm, 1e-12)

    Z = w.sum()
    c = (w[:, None] * contour_pts).sum(axis=0) / Z
    V = (w * np.square(np.linalg.norm(contour_pts - c, axis=1))).sum()

    return dict(mass=float(Z), center=c, variance=float(V))


def _collect_levelset_stats(encoder, density_fn, limits, grid_res, component, device, atol=1e-5):
    nx, ny = grid_res
    xs = np.linspace(limits[0][0], limits[0][1], nx)
    ys = np.linspace(limits[1][0], limits[1][1], ny)
    xv, yv = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([xv.ravel(), yv.ravel()], 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        z_val = encoder(pts)
        if component is not None:
            z_val = z_val[:, component]
    z_val = z_val.cpu().numpy().reshape(xv.shape)

    levels = np.unique(z_val)
    levels = np.asarray([levels[0], *[l for l in levels[1:] if abs(l - levels[0]) > atol]])
    stats = {}
    for L in levels:
        cs = plt.contour(xv, yv, z_val, levels=[L])
        for col in cs.collections:
            for path in col.get_paths():
                verts = path.vertices
                st = levelset_stats(encoder, density_fn, verts, component=component, device=device)
                if st['mass'] < _TINY:
                    continue
                if L in stats:
                    old = stats[L]
                    Z = old['mass'] + st['mass']
                    c = (old['center'] * old['mass'] + st['center'] * st['mass']) / Z
                    V = old['variance'] + st['variance']
                    stats[L] = dict(mass=Z, center=c, variance=V)
                else:
                    stats[L] = st
        plt.close(cs.figure)

    return stats


def cosine_alignment_theorem2p6(encoder, score_fn, density_fn, limits, grid_res, component, device="cpu",
                                density_cutoff=0.01, ignore_sign=False):
    if isinstance(grid_res, int):
        grid_res = (grid_res, grid_res)

    stats = _collect_levelset_stats(encoder, density_fn, limits, grid_res, component, device=device)
    nx, ny = grid_res
    xs = np.linspace(limits[0][0], limits[0][1], nx)
    ys = np.linspace(limits[1][0], limits[1][1], ny)
    xv, yv = np.meshgrid(xs, ys)
    grid_np = np.stack([xv.ravel(), yv.ravel()], 1)
    pdf = density_fn(grid_np)
    keep = pdf >= density_cutoff * pdf.max()

    grid = torch.tensor(grid_np, dtype=torch.float32, device=device, requires_grad=True)
    enc_out = encoder(grid)
    if component is not None:
        enc_out = enc_out[:, component]
    enc_out = enc_out.squeeze(-1)

    grad, = torch.autograd.grad(enc_out.sum(), grid)
    grad_np = grad.detach().cpu().numpy()
    score_np = score_fn(grid_np)

    cosines = []
    for i, (y, g, s) in enumerate(zip(grid_np, grad_np, score_np)):
        if not keep[i]:
            continue
        key = min(stats, key=lambda k: abs(k - enc_out[i].item()))
        st = stats[key]
        if st['mass'] < _TINY:
            continue
        c = st['center']
        VZ = st['variance'] / st['mass']
        r2 = np.dot(y - c, y - c)
        denom = VZ - r2
        if abs(denom) < _TINY:
            continue

        radial = 2.0 * (y - c) / denom
        lhs = np.dot(radial, g) * g
        rhs = np.dot(s, g) * g

        ln, rn = np.linalg.norm(lhs), np.linalg.norm(rhs)

        if ln < _TINY or rn < _TINY:
            continue

        cos = np.dot(lhs, rhs) / (ln * rn)

        if ignore_sign:
            cos = abs(cos)
        cosines.append(cos)

    cosines = np.array(cosines, dtype=np.float32)
    mean_cos = float(cosines.mean()) if cosines.size else float("nan")
    std_cos = float(cosines.std()) if cosines.size else float("nan")
    perc95 = float(np.percentile(cosines, 95.0)) if cosines.size else float("nan")

    return mean_cos, std_cos, perc95, cosines


def gaussian_pdf(xy):
    return (1 / (2 * math.pi)) * np.exp(-0.5 * np.sum(xy ** 2, 1))


def gaussian_score(xy):
    return -xy


MEANS_GMM = np.array([[-1.1, -1.1], [1.1, -0.9], [-0.33, 1.0]], dtype=np.float32)
WEIGHTS_GMM = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
SIGMA_GMM = 0.66


def _gmm_base(xy):
    diff = xy[:, None, :] - MEANS_GMM[None]
    g = np.exp(-0.5 * np.sum(diff ** 2, 2) / (SIGMA_GMM ** 2))
    p = g @ WEIGHTS_GMM
    return g, p


def gmm_pdf(xy):
    _, p = _gmm_base(xy);
    return p


def gmm_score(xy):
    g, p = _gmm_base(xy)
    gamma = (g * WEIGHTS_GMM) / p[:, None]
    diff = xy[:, None, :] - MEANS_GMM[None]
    cs = -diff / (SIGMA_GMM ** 2)
    return np.sum(gamma[..., None] * cs, 1).astype(np.float32)


def available_ckpts(prefix):
    return sorted(
        int(m.group(1)) for m in (re.search(r"model_(\d+)\.pt", p) for p in glob.glob(f"{prefix}/model_*.pt")) if
        m and int(m.group(1)) >= 300)


def load_encoder(prefix: str, epoch: int, n_feats=2, k=3, num_layer=4, hidden_dim=100, standardize=False,
                 device: str = "cpu") -> torch.nn.Module:
    latent_dims = list(range(k + 1))[::-1]

    model = DPA(beta=2, dist_enc="deterministic", dist_dec="stochastic", data_dim=n_feats, latent_dims=latent_dims,
                num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=hidden_dim, resblock=True,
                standardize=standardize, device=device, seed=42)

    ckpt = torch.load(f"{prefix}/model_{epoch}.pt", map_location="cpu")
    model.model.load_state_dict(ckpt)
    model.model = model.model.to(device)
    model.model.eval();

    return EncoderModule(model)


def eval_family(label: str, prefix: str, pdf_fn: Callable, score_fn: Callable, limits: tuple, grid_res: int,
                device: str, out_dir: str, density_cutoff: float = 0.01):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in available_ckpts(prefix):
        enc = load_encoder(prefix, ep).to(device)
        for comp in (0, 1):
            for ign in (False, True):
                mean, std, p95, *_ = cosine_alignment_theorem2p6(enc, score_fn, pdf_fn, limits, grid_res, comp,
                                                                 device=device, density_cutoff=density_cutoff,
                                                                 ignore_sign=ign)

                rows.append(dict(epoch=ep, component=comp, ignore_sign=ign, mean=mean, std=std, p95=p95))
        print(f"[{label}] epoch {ep} done")

    df = pd.DataFrame(rows)
    csv_file = out_path / f"{label.lower()}_alignment_all.csv"
    df.to_csv(csv_file, index=False)

    best_lines = []
    for comp in (0, 1):
        sub = df[(df.component == comp) & (df.ignore_sign)]
        best = sub.loc[(sub.mean - 1.0).abs().idxmin()]
        best_lines.append(f"**{label}, component {comp}:** "
                          f"epoch {best.epoch}, mean |cos| = {best.mean:.4f}, "
                          f"std ≈ {best.std:.4f}, 95-ile = {best.p95:.4f}")

    md_file = out_path / f"{label.lower()}_alignment_best.md"
    md_file.write_text("\n\n".join(best_lines))

    return best_lines


if __name__ == "__main__":
    LIMITS = ((-4, 4), (-4, 4))
    GRID = 100

    md_lines = []

    best_md_gauss = eval_family(label="Gaussian", prefix="res/models/gaussian/1k3l256h_old", pdf_fn=gaussian_pdf,
                                score_fn=gaussian_score, limits=((-4, 4), (-4, 4)), grid_res=100, device=device,
                                out_dir="res/gauss_results")

    best_md_trim = eval_family(label="Trimodal", prefix="res/models/trimodal/1k3l256h", pdf_fn=gmm_pdf,
                               score_fn=gmm_score, limits=((-4, 4), (-4, 4)), grid_res=100, device=device,
                               out_dir="res/trimodal_results")

    Path("res").mkdir(exist_ok=True)
    with open("res/alignment_best.md", "w") as fh:
        fh.write("# Alignment‑metric summary\n\n")
        fh.write("\n\n".join(md_lines))
    print("Markdown summary → res/alignment_best.md")
