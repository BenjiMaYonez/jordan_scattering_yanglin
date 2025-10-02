#!/usr/bin/env python3
"""Visualize Fourier-domain wavelet filters and their energy coverage."""
import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

_CACHE_ROOT = Path(os.environ.get("JORDAN_MPL_CACHE_DIR", ".mpl-cache")).resolve()
(_CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str((_CACHE_ROOT / "xdg-cache")))

import matplotlib
import numpy as np
import torch

from configs import save_config
from jordan_scatter.wavelets import filter_bank as build_filter_bank
from jordan_scatter.helpers import LoggerManager


DEFAULT_CONFIG = {
    "wavelet": "meyer",
    "max_scale": 3,
    "nb_orients": 8,
    "image_size": 256,
    "max_per_figure": 10,
    "columns": 4,
    "cmap": "viridis",
    "show": False,
    "output_dir": "experiments",
    "run_prefix": "filters",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot wavelet filters in the frequency domain")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/plot_filters.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_settings(path: str) -> SimpleNamespace:
    cfg = DEFAULT_CONFIG.copy()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parsing error in {path}: {exc}") from exc

    if loaded is None:
        loaded = {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping of settings.")

    cfg.update(loaded)
    return SimpleNamespace(**cfg)


def to_numpy(array_like: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def prepare_image(data: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(data)


def plot_batches(
    settings: SimpleNamespace,
    batches: list[list[tuple[str, np.ndarray]]],
    output_dir: Path,
    plt,
    logger,
) -> None:
    for idx, batch in enumerate(batches, start=1):
        ncols = max(1, settings.columns)
        nrows = math.ceil(len(batch) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax_idx, (title, image) in enumerate(batch):
            ax = axes[ax_idx]
            prepared = prepare_image(image)
            if prepared.size == 0:
                vmax = 1.0
                vmin = 0.0
            elif np.all(prepared >= 0):
                vmax = prepared.max() if prepared.max() > 0 else 1.0
                vmin = 0.0
            else:
                max_abs = np.max(np.abs(prepared))
                vmax = max_abs if max_abs > 0 else 1.0
                vmin = -vmax
            im = ax.imshow(prepared, cmap=settings.cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes[len(batch):]:
            ax.axis("off")
        fig.suptitle(
            f"Filters: {settings.wavelet}, J={settings.max_scale}, L={settings.nb_orients}, N={settings.image_size}",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = output_dir / (
            f"filters_grid_{settings.wavelet}_J{settings.max_scale}_L{settings.nb_orients}_N{settings.image_size}_batch{idx:02d}.png"
        )
        fig.savefig(output_path, dpi=150)
        logger.info("Saved filter grid to %s", output_path)
        if settings.show:
            plt.show()
        else:
            plt.close(fig)


def plot_sum_of_squares(settings: SimpleNamespace, energy: np.ndarray, output_dir: Path, plt, logger) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    prepared = prepare_image(energy)
    vmax = prepared.max() if prepared.size > 0 else 1.0
    im = ax.imshow(prepared, cmap=settings.cmap, vmin=0.0, vmax=vmax)
    ax.set_title(
        f"Sum of Squares\n{settings.wavelet}, J={settings.max_scale}, L={settings.nb_orients}, N={settings.image_size}",
        fontsize=14,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = output_dir / (
        f"sum_of_squares_{settings.wavelet}_J{settings.max_scale}_L{settings.nb_orients}_N{settings.image_size}.png"
    )
    fig.savefig(output_path, dpi=150)
    logger.info("Saved energy map to %s", output_path)
    if settings.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    cli_args = parse_args()
    settings = load_settings(cli_args.config)

    if settings.max_per_figure <= 0:
        raise ValueError("max_per_figure must be positive")

    if not settings.show:
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    base_dir = Path(settings.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{settings.run_prefix}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = LoggerManager.get_logger(log_dir=str(run_dir), name="plot_filters")
    logger.info("Loaded settings from %s", cli_args.config)
    logger.info("Saving plots to %s", run_dir)

    config_snapshot = vars(settings).copy()
    config_snapshot["source_config"] = os.path.abspath(cli_args.config)
    save_config(str(run_dir), config_snapshot)

    filters = build_filter_bank(settings.wavelet, settings.max_scale, settings.nb_orients, settings.image_size)
    lp = to_numpy(filters["lp"])
    hp = to_numpy(filters["hp"])

    lp_energy = np.abs(lp) ** 2
    hp_energy = np.abs(hp) ** 2
    frame_energy = lp_energy + hp_energy.sum(axis=(0, 1))

    logger.info(
        "LP energy stats (first level): min=%.6f, max=%.6f",
        lp_energy.min(),
        lp_energy.max(),
    )
    logger.info(
        "Frame bounds: min=%.6f, max=%.6f, mean|E-1|=%.6f",
        frame_energy.min(),
        frame_energy.max(),
        np.mean(np.abs(frame_energy - 1.0)),
    )

    filters_to_plot: list[tuple[str, np.ndarray]] = [("low-pass", np.real(lp))]
    for j in range(hp.shape[0]):
        for theta in range(hp.shape[1]):
            title = f"j={j}, θ={theta}"
            filters_to_plot.append((title, np.real(hp[j, theta])))

    batches: list[list[tuple[str, np.ndarray]]] = []
    for start in range(0, len(filters_to_plot), settings.max_per_figure):
        end = start + settings.max_per_figure
        batches.append(filters_to_plot[start:end])

    plot_batches(settings, batches, run_dir, plt, logger)

    # Append sum of squares figure separately for full-energy view
    plot_sum_of_squares(settings, frame_energy, run_dir, plt, logger)

    logger.info("Finished plotting filters")


if __name__ == "__main__":
    main()
