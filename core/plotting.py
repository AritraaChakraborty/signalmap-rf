"""CO6: Well-labelled Matplotlib visualizations for wireless signal coverage."""

from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.models import DEFAULT_AP_ORIGIN, DEFAULT_STRONG_THRESHOLD, DEFAULT_WEAK_THRESHOLD

PALETTE = {
    "Strong": "#157A6E",
    "Fair": "#C27803",
    "Weak": "#C83E2B",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFC8BC")
    ax.spines["bottom"].set_color("#CFC8BC")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, color="#B8B0A4")


def plot_scatter_map(
    df: pd.DataFrame,
    output_path: str,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=160)
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FFFFFF")

    for zone in ("Strong", "Fair", "Weak"):
        sub = df[df["zone"] == zone]
        ax.scatter(
            sub["x"],
            sub["y"],
            c=PALETTE[zone],
            label=f"{zone} ({len(sub)})",
            s=52,
            edgecolors="#FFFFFF",
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
        )

    ax.scatter(
        [DEFAULT_AP_ORIGIN[0]],
        [DEFAULT_AP_ORIGIN[1]],
        marker="^",
        s=150,
        c="#161614",
        label="Access Point (AP-01)",
        zorder=5,
    )
    ax.set_title(
        f"Synthetic Floor Wireless Coverage Map (Weak < {weak_threshold:.0f} dBm, Strong >= {strong_threshold:.0f} dBm)",
        fontsize=11,
        fontweight="bold",
        color="#161614",
        pad=12,
    )
    ax.set_xlabel("Floor X Coordinate (meters)", color="#161614")
    ax.set_ylabel("Floor Y Coordinate (meters)", color="#161614")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 30)
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="#FAF7F2", edgecolor="#E6E0D6", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coverage_heatmap(
    XX: np.ndarray,
    YY: np.ndarray,
    blended: np.ndarray,
    nearest_dist: np.ndarray,
    df: pd.DataFrame,
    output_path: str,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    gap_radius_m: float = 6.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FFFFFF")

    mesh = ax.pcolormesh(
        XX, YY, blended, shading="auto", cmap="RdYlGn", vmin=-95, vmax=-40
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Estimated RSSI (dBm)", color="#161614")

    # Contour line at weak threshold
    ax.contour(
        XX,
        YY,
        blended,
        levels=[weak_threshold],
        colors=["#161614"],
        linewidths=1.4,
        linestyles="--",
    )

    # Hatch unsurveyed gap region (nearest sample > gap_radius_m)
    gap_mask = (nearest_dist > gap_radius_m).astype(float)
    ax.contourf(
        XX,
        YY,
        gap_mask,
        levels=[0.5, 1.5],
        colors=["none"],
        hatches=["///"],
    )
    ax.contour(
        XX,
        YY,
        gap_mask,
        levels=[0.5],
        colors=["#161614"],
        linewidths=1.0,
    )

    ax.scatter(
        df["x"],
        df["y"],
        c="#161614",
        s=14,
        alpha=0.55,
        label="Measured samples",
    )
    ax.scatter(
        [DEFAULT_AP_ORIGIN[0]],
        [DEFAULT_AP_ORIGIN[1]],
        marker="^",
        s=140,
        c="#161614",
        edgecolors="#FFFFFF",
        linewidth=1.0,
        label="AP-01 Origin",
    )
    ax.set_title(
        f"Interpolated Coverage Heatmap (Hatched = Unsurveyed Gap > {gap_radius_m:.0f} m)",
        fontsize=11,
        fontweight="bold",
        color="#161614",
        pad=12,
    )
    ax.set_xlabel("Floor X Coordinate (meters)")
    ax.set_ylabel("Floor Y Coordinate (meters)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 30)
    ax.legend(loc="lower right", frameon=True, facecolor="#FAF7F2", edgecolor="#E6E0D6", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rssi_histogram(
    df: pd.DataFrame,
    output_path: str,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FFFFFF")

    vals = df["signal_dbm"].to_numpy()
    mean_v = float(np.mean(vals))
    med_v = float(np.median(vals))

    ax.hist(vals, bins=12, color="#157A6E", alpha=0.82, edgecolor="#FFFFFF", linewidth=1.0)
    ax.axvline(mean_v, color="#161614", linestyle="--", linewidth=1.5, label=f"Mean ({mean_v:.1f} dBm)")
    ax.axvline(med_v, color="#C27803", linestyle=":", linewidth=1.8, label=f"Median ({med_v:.1f} dBm)")
    ax.axvline(weak_threshold, color="#C83E2B", linestyle="-", linewidth=1.6, label=f"Weak Threshold ({weak_threshold:.0f} dBm)")

    ax.set_title("Distribution of Measured Signal Strength (Synthetic Dataset)", fontsize=11, fontweight="bold", color="#161614", pad=12)
    ax.set_xlabel("Signal Strength RSSI (dBm)")
    ax.set_ylabel("Number of Survey Points")
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="#FAF7F2", edgecolor="#E6E0D6", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_path_loss_fit(
    df: pd.DataFrame,
    model: Dict[str, Any],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=160)
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FFFFFF")

    dists = model["distances"]
    rssi = df["signal_dbm"].to_numpy()
    d_grid = np.linspace(max(0.5, float(np.min(dists))), float(np.max(dists)), 100)
    y_fit = model["A_dbm"] - 10.0 * model["path_loss_exponent_n"] * np.log10(d_grid)

    ax.scatter(dists, rssi, c="#157A6E", alpha=0.72, s=36, edgecolors="#FFFFFF", linewidth=0.5, label="Measured samples")
    ax.plot(
        d_grid,
        y_fit,
        color="#C83E2B",
        linewidth=2.2,
        label=f"Fit: A={model['A_dbm']:.2f} dBm, n={model['path_loss_exponent_n']:.2f} (RMSE={model['rmse_db']:.2f} dB)",
    )

    ax.set_title("Log-Distance Path-Loss Model Fit: RSSI(d) = A - 10*n*log10(d)", fontsize=11, fontweight="bold", color="#161614", pad=12)
    ax.set_xlabel("Distance from Access Point (meters)")
    ax.set_ylabel("Signal Strength RSSI (dBm)")
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="#FAF7F2", edgecolor="#E6E0D6", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(
    df: pd.DataFrame,
    model: Dict[str, Any],
    XX: np.ndarray,
    YY: np.ndarray,
    blended: np.ndarray,
    nearest_dist: np.ndarray,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plot_scatter_map(df, os.path.join(output_dir, "scatter_map.png"))
    plot_coverage_heatmap(XX, YY, blended, nearest_dist, df, os.path.join(output_dir, "coverage_heatmap.png"))
    plot_rssi_histogram(df, os.path.join(output_dir, "rssi_histogram.png"))
    plot_path_loss_fit(df, model, os.path.join(output_dir, "path_loss_fit.png"))
