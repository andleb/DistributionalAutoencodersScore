# Utility functions for the MFEP analysis

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

minima = [(-0.558, 1.442), (-0.050, 0.467), (0.623, 0.028)]


def gradMBPos(x, y):
    comp1 = (-25.5 * (11 * (y - 1.5) - 13 * (x + 0.5)) * np.exp(
        11 * (x + 0.5) * (y - 1.5) - 6.5 * (x + 0.5) ** 2 - 6.5 * (y - 1.5) ** 2) + 2.25 * (
                         1.4 * (x + 1) + 0.6 * (y - 1)) * np.exp(
        0.6 * (x + 1) * (y - 1) + 0.7 * (x + 1) ** 2 + 0.7 * (y - 1) ** 2) + 30 * x * np.exp(
        -x ** 2 - 10 * (y - 0.5) ** 2) + 60 * (x - 1) * np.exp(-(x - 1) ** 2 - 10 * y ** 2))

    comp2 = (-25.5 * (11 * (x + 0.5) - 13 * (y - 1.5)) * np.exp(
        11 * (x + 0.5) * (y - 1.5) - 6.5 * (x + 0.5) ** 2 - 6.5 * (y - 1.5) ** 2) + 2.25 * (
                         0.6 * (x + 1) + 1.4 * (y - 1)) * np.exp(
        0.6 * (x + 1) * (y - 1) + 0.7 * (x + 1) ** 2 + 0.7 * (y - 1) ** 2) + 300 * (y - 0.5) * np.exp(
        -x ** 2 - 10 * (y - 0.5) ** 2) + 600 * y * np.exp(-(x - 1) ** 2 - 10 * y ** 2))

    return comp1, comp2


def string_method(start, end, n_points=30, n_iterations=1000, step_size=0.001):
    """Implement the string method to find the MFEP"""

    string = np.array([np.linspace(s, e, n_points) for s, e in zip(start, end)]).T

    for _ in range(n_iterations):
        gradients = np.array([np.clip(gradMBPos(x, y), -100, 100) for x, y in string])

        string = string - step_size * gradients

        distances = np.cumsum(np.sqrt(np.sum(np.diff(string, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        distances /= distances[-1]

        new_string = np.array([interp1d(distances, string[:, i])(np.linspace(0, 1, n_points)) for i in range(2)]).T

        if np.allclose(string, new_string, atol=1e-6):
            break

        string = new_string

    return string


def chamfer_and_p95(reco_xy: np.ndarray, reference_xy: np.ndarray):
    """Return Chamfer distance and 95-percentile of bidirectional errors."""

    tree_r = cKDTree(reference_xy)
    tree_p = cKDTree(reco_xy)
    d_ref_to_p = tree_p.query(reference_xy, k=1)[0]
    d_p_to_ref = tree_r.query(reco_xy, k=1)[0]
    chamfer = 0.5 * (d_ref_to_p.mean() + d_p_to_ref.mean())
    p95 = np.percentile(np.hstack([d_ref_to_p, d_p_to_ref]), 95)

    return chamfer, p95


def _z_and_grad(x, encoder, comp_idx):
    """
    Utility that computes the latent coordinate z_i and its gradient
    """

    x = x.detach().clone().requires_grad_(True)

    z = encoder(x.unsqueeze(0))[0, comp_idx]
    z.backward()
    g = x.grad

    return z, g


def newton_single(latent_target, x_init, encoder, comp_idx, tol=1e-6, max_iter=40, damp=0.5, device="cpu", **kwargs):
    """
    Newton's method with damping
    """

    x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=False)

    for _ in range(max_iter):
        z, g = _z_and_grad(x, encoder, comp_idx)
        diff = z - latent_target
        if diff.abs() < tol:
            break
        step = damp * diff * g / (g @ g + 1e-12)
        x = (x - step).detach()

    return x.cpu().numpy()


def invert_single_adam(latent_target, x_init, encoder, comp_idx, lr=3e-3, max_iter=80, tol=1e-6, device="cpu",
                       **kwargs):
    """
       Root-finding inversion using the Adam optimizer.
    """
    x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        diff = encoder(x.unsqueeze(0))[0, comp_idx] - latent_target
        if diff.abs() < tol:
            break
        diff.backward()
        opt.step()
    return x.detach().cpu().numpy()


def invert_march(latent_vals, start_xy, encoder, comp_idx, solver="newton", device="cpu", **solver_kw):
    """
    March along the latent values using the specified solver.
    """

    xs = np.empty((len(latent_vals), 2), dtype=np.float32)
    xs[0] = start_xy
    inv = newton_single if solver == "newton" else invert_single_adam

    for k in range(1, len(latent_vals)):
        xs[k] = inv(latent_vals[k], xs[k - 1], encoder, comp_idx=comp_idx, device=device, **solver_kw)
    return xs


def invert_batch_progressive(latent_vals, start_xy, encoder, comp_idx, tol=1e-6, max_iter=40, damp=0.5, device="cuda"):
    """
    Batched progressive inversion of the latent values using Newton's method.

    """
    xs = [start_xy]
    for t in latent_vals[1:]:
        x_next = newton_single(t, xs[-1], encoder, comp_idx, tol=tol, max_iter=max_iter, damp=damp, device=device)
        xs.append(x_next)
    return np.stack(xs)


def march_along_gradient(encoder, start_xy: np.ndarray, comp_idx: int, z_max: float, dz: float = 0.01,
                         max_steps: int = 20_000, device: str = "cpu"):
    """
    March along the latents using the gradient of the encoder.
    """
    x = torch.as_tensor(start_xy, dtype=torch.float32, device=device)
    path = [x.cpu().numpy()]

    for _ in range(max_steps):
        z, g = _z_and_grad(x, encoder, comp_idx)
        if z >= z_max:
            break

        step = dz * g / (g @ g + 1e-12)

        x = (x + step).detach()

        path.append(x.cpu().numpy())

    return np.stack(path)


def parametrisation_test(encoder, mfep_xy: np.ndarray, start_minimum_xy: np.ndarray, n_steps: int = 200,
        z_range: tuple[float, float] | None = None, z_vals: np.ndarray | None = None, z_parallel_idx=None,
        device: str = "cpu", lr=1e-3, tol=1e-6, iters=500, solver="newton", batch=True, grad=False,

):
    """
        The main parameterization finding and evaluation routine
    """

    with torch.no_grad():
        z_mfep = encoder(torch.as_tensor(mfep_xy, dtype=torch.float32, device=device)).detach().cpu().numpy()

    arc = np.insert(np.cumsum(np.linalg.norm(np.diff(mfep_xy, axis=0), axis=1)), 0, 0.0).reshape(-1, 1)

    best_i, best_r2 = None, -np.inf
    for i in range(z_mfep.shape[1]):
        mdl = LinearRegression().fit(arc, z_mfep[:, i])
        r2 = mdl.score(arc, z_mfep[:, i])
        if r2 > best_r2:
            best_r2, best_i, best_mdl = r2, i, mdl

    if z_parallel_idx is None:
        z_parallel_idx = best_i

    if z_vals is None:
        if z_range is None:
            z_start = best_mdl.predict([[arc[0, 0]]])[0]
            z_end = best_mdl.predict([[arc[-1, 0]]])[0]
        else:
            z_start, z_end = z_range

        z_vals = np.linspace(z_start, z_end, n_steps)
    else:
        z_vals = np.asarray(z_vals).ravel()
        n_steps = len(z_vals)

    recon_xy = np.empty((n_steps, 2))
    recon_xy[0] = start_minimum_xy

    if grad:
        recon_xy_grad = march_along_gradient(encoder=encoder, start_xy=start_minimum_xy, comp_idx=z_parallel_idx,
            z_max=z_vals[-1], dz=(z_vals[-1] - z_vals[0]) / n_steps, max_steps=n_steps, device=device)

    if batch:
        recon_xy = invert_batch_progressive(z_vals, start_minimum_xy, encoder, comp_idx=z_parallel_idx, device=device,
                                            tol=tol, max_iter=iters, )



    else:
        recon_xy = invert_march(latent_vals=z_vals, start_xy=start_minimum_xy, encoder=encoder, comp_idx=z_parallel_idx,
            solver=solver, device=device, tol=tol, max_iter=iters, lr=lr

        )

    tree_mfep = cKDTree(mfep_xy)
    tree_reco = cKDTree(recon_xy)
    d_m2r, _ = tree_reco.query(mfep_xy, k=1)
    d_r2m, _ = tree_mfep.query(recon_xy, k=1)

    chamfer, p95 = chamfer_and_p95(recon_xy, mfep_xy)

    ret = dict(parallel_component=int(z_parallel_idx), R2=float(best_r2), mean_MFEP_to_path=float(d_m2r.mean()),
        mean_path_to_MFEP=float(d_r2m.mean()), Hausdorff=float(max(d_m2r.max(), d_r2m.max())), chamfer=float(chamfer),
        p95=float(p95), z_values=z_vals, z_mfep=z_mfep, reconstructed_path=recon_xy, )

    if grad:
        ret["reconstructed_path_grad"] = recon_xy_grad

        tree_reco_grad = cKDTree(recon_xy_grad)
        d_m2r_grad, _ = tree_reco_grad.query(mfep_xy, k=1)
        d_r2m_grad, _ = tree_mfep.query(recon_xy_grad, k=1)

        chamfer_grad, p95_grad = chamfer_and_p95(recon_xy_grad, mfep_xy)

        ret["mean_MFEP_to_path_grad"] = float(d_m2r_grad.mean())
        ret["mean_path_to_MFEP_grad"] = float(d_r2m_grad.mean())
        ret["Hausdorff_grad"] = float(max(d_m2r_grad.max(), d_r2m_grad.max()))
        ret["chamfer_grad"] = float(chamfer_grad)
        ret["p95_grad"] = float(p95_grad)

    return ret
