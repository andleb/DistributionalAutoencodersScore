# NOTE: Modified and extended version of
# https://github.com/luigibonati/mlcolvar/blob/main/mlcolvar/utils/plot.py
# Licensed under the MIT License, original license reproduced in third_party_licenses/mlcolvar/LICENSE

import matplotlib as mpl
import matplotlib.pyplot as plt
import mlcolvar
import numpy as np
import torch
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ColorConverter

__all__ = ["paletteFessa", "paletteCortina", "plot_metrics", "plot_features_distribution"]

paletteFessa = ["#1F3B73", "#2F9294", "#50B28D", "#A7D655", "#FFE03E", "#FFA955", "#D6573B", ]

cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa)
mpl.colormaps.register(cmap=cm_fessa)
mpl.colormaps.register(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f"fessa{i}"] = paletteFessa[i]

paletteCortina = [[0.0, 0.0, 0.803921568627451, 1], [0.4823529411764706, 0.40784313725490196, 0.9333333333333333, 1],
                  [0.0, 0.9803921568627451, 0.6039215686274509, 1],
                  [0.23529411764705882, 0.7019607843137254, 0.44313725490196076, 1],
                  [0.8588235294117647, 0.4392156862745098, 0.5764705882352941, 1],
                  [0.7803921568627451, 0.08235294117647059, 0.5215686274509804, 1], ]

cm_cortina = LinearSegmentedColormap.from_list("cortina80", paletteCortina)
mpl.colormaps.register(cmap=cm_cortina)
mpl.colormaps.register(cmap=cm_cortina.reversed())

import matplotlib.pyplot as plt
import numpy as np
import torch


def muller_brown_potential(x, y):
    """Muller-Brown analytical potential"""
    prefactor = 0.15
    A = (-200, -100, -170, 15)
    a = (-1, -1, -6.5, 0.7)
    b = (0, 0, 11, 0.6)
    c = (-10, -10, -6.5, 0.7)
    x0 = (1, 0, -0.5, -1)
    y0 = (0, 0.5, 1.5, 1)
    offset = -146.7

    v = -prefactor * offset
    for i in range(4):
        v += (prefactor * A[i] * np.exp(
            a[i] * (x - x0[i]) ** 2 + b[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i]) ** 2))
    return v


def muller_brown_mfep():
    mfep = np.loadtxt(mlcolvar.__path__[0] + "/../docs/notebooks/tutorials/data/muller-brown/mfep.txt",
                      usecols=(0, 1), )
    return mfep


def muller_brown_potential_three_states(x, y):
    """Muller-Brown analytical potential"""
    prefactor = 0.15
    A = (-280, -170, -170, 15)
    a = (-15, -1, -6.5, 0.7)
    b = (0, 0, 11, 0.6)
    c = (-10, -10, -6.5, 0.7)
    x0 = (1, 0.2, -0.5, -1)
    y0 = (0, 0.5, 1.5, 1)
    offset = -146.7

    v = -prefactor * offset
    for i in range(4):
        v += (prefactor * A[i] * np.exp(
            a[i] * (x - x0[i]) ** 2 + b[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i]) ** 2))
    return v


def muller_brown_three_states_mfep():
    mfep = np.loadtxt(mlcolvar.__path__[0] + "/../docs/notebooks/tutorials/data/muller-brown-3states/mfep.txt",
                      usecols=(0, 1), )
    return mfep


def plot_metrics(metrics, keys=["train_loss_epoch", "valid_loss"], x=None, labels=None, linestyles=None, colors=None,
                 xlabel="Epoch", ylabel="Loss", title="Learning curves", yscale=None, ax=None, ):
    """Plot logged metrics."""

    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(5, 4), dpi=100)

    auto_x = True if x is None else False
    for i, key in enumerate(keys):
        y = metrics[key]
        lstyle = linestyles[i] if linestyles is not None else None
        label = labels[i] if labels is not None else key
        color = colors[i] if colors is not None else None
        if auto_x:
            x = np.arange(len(y))
        ax.plot(x, y, linestyle=lstyle, label=label, color=color)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)

    ax.legend(ncol=1, frameon=False)

    if return_axs:
        return ax
    else:
        return None


def plot_features_distribution(dataset, features, titles=None, axs=None):
    """Plot distribution of the given features.

    Parameters
    ----------
    dataset : DictDataset
        dataset
    features : list
        list of features names
    titles : list,optional
        list titles to be displayed, by default None
    axs : optional
        matplotlib axs, by default None

    """

    if isinstance(features, dict):
        raise TypeError('features should be a list of feature names, not a dictionary')

    n_feat = len(features)

    if axs is None:
        if n_feat <= 5:
            fig, axs = plt.subplots(1, n_feat, figsize=(3 * n_feat + 1, 3))
        else:
            fig, axs = plt.subplots(n_feat, 1, figsize=(3, 3 * n_feat))

        plt.suptitle('Features distribution')
        init_ax = True
    else:
        if n_feat != len(axs):
            raise ValueError(f'Number of features ({len(features)}) != number of axis ({len(axs)})')
        init_ax = False

    axs[0].set_ylabel('Distribution')

    if "labels" in dataset.keys:
        labels = sorted(dataset['labels'].unique().numpy())
        for l in labels:
            id_l = np.argwhere(dataset['labels'] == l)[0, :]
            data_label = dataset['data'][id_l, :]

            for i, feat in enumerate(features):
                ax = axs[i]
                id = np.argwhere(dataset.feature_names == feat)[0]
                x = data_label[:, id].numpy()
                ax.hist(x, bins=50, label=f"State {int(l)}", histtype='step')
                ax.set_yticks([])
                ax.set_xlabel(feat)
                if i == 0:
                    if titles is not None:
                        ax.legend(title=titles[i], loc='upper center', framealpha=0.8, edgecolor='white')
                    else:
                        ax.legend(loc='upper center', framealpha=0.8, edgecolor='white')
                else:
                    if titles is not None:
                        ax.legend([], [], title=titles[i], loc='upper center', framealpha=0.8, edgecolor='white')
    else:
        for i, feat in enumerate(features):
            ax = axs[i]
            id = np.argwhere(dataset.feature_names == feat)[0]
            data = dataset['data']
            x = data[:, id].numpy()
            ax.hist(x, bins=100, )
            ax.set_yticks([])
            ax.legend([], [], title=feat, loc='upper center', frameon=False)


def test_utils_plot():
    import matplotlib

    x = np.linspace(-1.5, 1.5)
    y = np.linspace(-0.5, 2.5)

    mp = muller_brown_potential(x, y)
    mp = muller_brown_potential_three_states(x, y)

    pal = paletteFessa
    pal = paletteCortina

    cmap = matplotlib.colors.Colormap("fessa", 2)
    cmap = matplotlib.colors.Colormap("fessa_r", 2)
    cmap = matplotlib.colors.Colormap("cortina80", 2)
    cmap = matplotlib.colors.Colormap("cortina80_r", 2)


def plot_isolines_2D(function, component=None, limits=((-1.8, 1.2), (-0.4, 2.1)), num_points=(100, 100),
                     mode="contourf", levels=12, cmap=None, colorbar=None, max_value=None, ax=None, allow_grad=False,
                     xx=None, yy=None, **kwargs, ):
    """Plot isolines of a function/model in a 2D space."""

    if type(num_points) == int:
        num_points = (num_points, num_points)
    if xx is None or yy is None:
        xx = np.linspace(limits[0][0], limits[0][1], num_points[0])
        yy = np.linspace(limits[1][0], limits[1][1], num_points[1])

    xv, yv = np.meshgrid(xx, yy)

    if isinstance(function, torch.nn.Module):
        z = np.zeros_like(xv)
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                xy = torch.Tensor([xv[i, j], yv[i, j]])
                if allow_grad:
                    s = function(xy.unsqueeze(0)).squeeze(0).detach().numpy()
                else:
                    with torch.no_grad():
                        train_mode = function.training
                        function.eval()
                        s = function(xy.unsqueeze(0)).squeeze(0).cpu().numpy()
                        function.training = train_mode
                if component is not None:
                    s = s[component]
                z[i, j] = s
    else:
        z = function(xv, yv)

    if max_value is not None:
        z[z > max_value] = max_value

    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6, 4.0), dpi=100)

    if cmap is None:
        if mode == "contourf":
            cmap = "fessa"
        elif mode == "contour":
            if 'colors' not in kwargs:
                cmap = "Greys_r"

    if colorbar is None:
        if mode == "contourf":
            colorbar = True
        elif mode == "contour":
            colorbar = False

    if mode == "contourf":
        pp = ax.contourf(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            plt.colorbar(pp, ax=ax)
    else:
        pp = ax.contour(xv, yv, z, levels=levels, cmap=cmap, **kwargs)

    if return_axs:
        return ax
    else:
        return xv, yv, z


def plot_gradient_2D(function, component=None, fin_diff=True, limits=((-1.8, 1.2), (-0.4, 2.1)), num_points=(20, 20),
                     clip_grad=None, cmap=None, ax=None, allow_grad=False, xx=None, yy=None, **kwargs, ):
    """Plot gradient of a function/model in a 2D space."""

    if isinstance(num_points, int):
        num_points = (num_points, num_points)
    if xx is None or yy is None:
        xx = np.linspace(limits[0][0], limits[0][1], num_points[0])
        yy = np.linspace(limits[1][0], limits[1][1], num_points[1])

    xv, yv = np.meshgrid(xx, yy)

    if isinstance(function, torch.nn.Module) and not fin_diff:
        grid_points = torch.tensor(np.stack([xv.flatten(), yv.flatten()], axis=1), dtype=torch.float32,
                                   requires_grad=True).to(next(function.parameters()).device)

        if not allow_grad:
            train_mode = function.training
            function.eval()

        with torch.enable_grad():
            outputs = function(grid_points)
            if component is not None:
                outputs = outputs[:, component]

            if not outputs.requires_grad:
                outputs.requires_grad_(True)

            loss = outputs.sum()

            if not any(p.requires_grad for p in [grid_points, loss]):
                raise RuntimeError("No tensor in the computation graph requires gradients")

            gradients = \
                torch.autograd.grad(loss, grid_points, create_graph=False, retain_graph=False, allow_unused=True)[0]

            if gradients is None:
                raise RuntimeError("Gradient computation returned None")

        dz_dx = gradients[:, 0].reshape(xv.shape).detach().cpu().numpy()
        dz_dy = gradients[:, 1].reshape(xv.shape).detach().cpu().numpy()
        z = outputs.reshape(xv.shape).detach().cpu().numpy()

    if fin_diff or not isinstance(function, torch.nn.Module):

        if not isinstance(function, torch.nn.Module):
            z = function(xv, yv)
        else:
            grid_points = torch.tensor(np.stack([xv.flatten(), yv.flatten()], axis=1), dtype=torch.float32).to(
                next(function.parameters()).device)

            outputs = function(grid_points)

            expected_shape = xv.shape
            z = outputs.detach().cpu().numpy()

            if component is not None:
                z = z[:, component]

            z = z.reshape(expected_shape)

        if component is not None and not isinstance(function, torch.nn.Module):
            z = z[component]

        dx = (limits[0][1] - limits[0][0]) / (num_points[0] - 1)
        dy = (limits[1][1] - limits[1][0]) / (num_points[1] - 1)
        dz_dy, dz_dx = np.gradient(z, dx, dy)

    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6, 4.0), dpi=100)

    if cmap is None:
        cmap = "viridis"

    magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    if clip_grad is not None:
        dz_dx = np.clip(dz_dx, -clip_grad, clip_grad)
        dz_dy = np.clip(dz_dy, -clip_grad, clip_grad)

    ax.quiver(xv, yv, dz_dx, dz_dy, cmap=cmap, **kwargs)

    if return_axs:
        return ax
    else:
        return dz_dx, dz_dy
