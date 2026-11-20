"""CO4 (NumPy arrays/operations) and CO5 (Pandas Series/DataFrames & CSV/JSON).

Implements vectorized NumPy distance computation, 2D spatial grid summaries,
log-distance path-loss regression via NumPy polyfit, and Pandas DataFrame
location aggregations with CSV and JSON export.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from core.models import (
    DEFAULT_AP_ORIGIN,
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    SurveyDataset,
)


def dataset_to_dataframe(
    dataset: SurveyDataset,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    origin: Tuple[float, float] = DEFAULT_AP_ORIGIN,
) -> pd.DataFrame:
    """CO5: Convert a SurveyDataset into a structured Pandas DataFrame."""
    records = [p.to_dict(weak_threshold, strong_threshold, origin) for p in dataset]
    return pd.DataFrame(records)


def numpy_path_loss_fit(
    df: pd.DataFrame,
    origin: Tuple[float, float] = DEFAULT_AP_ORIGIN,
) -> Dict[str, Any]:
    """CO4: Vectorized distance calculation and log-distance path-loss fit using NumPy."""
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    rssi = df["signal_dbm"].to_numpy(dtype=float)

    distances = np.maximum(np.hypot(x - origin[0], y - origin[1]), 0.5)
    u = 10.0 * np.log10(distances)

    # Fit RSSI = A + slope * (10 * log10(d)), where n = -slope
    slope, intercept_a = np.polyfit(u, rssi, deg=1)
    path_loss_n = float(-slope)
    intercept_a = float(intercept_a)

    predicted = intercept_a - 10.0 * path_loss_n * np.log10(distances)
    residuals = rssi - predicted

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((rssi - np.mean(rssi)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "ap_origin": {"x": origin[0], "y": origin[1]},
        "A_dbm": round(intercept_a, 3),
        "path_loss_exponent_n": round(path_loss_n, 3),
        "mae_db": round(mae, 3),
        "rmse_db": round(rmse, 3),
        "r_squared": round(r2, 3),
        "distances": distances,
        "predicted": predicted,
        "residuals": residuals,
    }


def numpy_grid_summary(
    df: pd.DataFrame,
    cols: int = 4,
    rows: int = 3,
    x_max: float = 40.0,
    y_max: float = 30.0,
) -> np.ndarray:
    """CO4: Compute a 2D NumPy array of mean RSSI per spatial grid cell (NaN if unsurveyed)."""
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    rssi = df["signal_dbm"].to_numpy(dtype=float)

    grid = np.full((rows, cols), np.nan, dtype=float)
    dx = x_max / cols
    dy = y_max / rows

    for r in range(rows):
        for c in range(cols):
            mask = (
                (x >= c * dx)
                & (x < (c + 1) * dx if c < cols - 1 else x <= x_max)
                & (y >= r * dy)
                & (y < (r + 1) * dy if r < rows - 1 else y <= y_max)
            )
            if np.any(mask):
                grid[r, c] = np.mean(rssi[mask])
    return grid


def numpy_interpolated_surface(
    df: pd.DataFrame,
    model: Dict[str, Any],
    nx: int = 40,
    ny: int = 30,
    x_max: float = 40.0,
    y_max: float = 30.0,
    gap_radius_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CO4: Interpolate a 2D floor RSSI grid and nearest-sample distance map."""
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    rssi = df["signal_dbm"].to_numpy(dtype=float)

    gx = np.linspace(0.5, x_max - 0.5, nx)
    gy = np.linspace(0.5, y_max - 0.5, ny)
    XX, YY = np.meshgrid(gx, gy)

    # Pairwise distances from each grid cell to each survey point
    dx = XX[:, :, None] - xs[None, None, :]
    dy = YY[:, :, None] - ys[None, None, :]
    dists = np.hypot(dx, dy)

    nearest_dist = np.min(dists, axis=2)
    weights = 1.0 / ((dists + 0.75) ** 2)
    idw_surface = np.sum(weights * rssi[None, None, :], axis=2) / np.sum(weights, axis=2)

    origin = model["ap_origin"]
    d_ap = np.maximum(np.hypot(XX - origin["x"], YY - origin["y"]), 0.5)
    model_surface = model["A_dbm"] - 10.0 * model["path_loss_exponent_n"] * np.log10(d_ap)

    alpha = np.clip(nearest_dist / (gap_radius_m * 1.5), 0.0, 1.0)
    blended = (1.0 - alpha) * idw_surface + alpha * model_surface
    return XX, YY, blended, nearest_dist


def pandas_location_comparison(
    df: pd.DataFrame,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> pd.DataFrame:
    """CO5: Aggregate RSSI metrics and weak-point percentages by location."""
    grouped = (
        df.groupby("location", as_index=False)
        .agg(
            sample_count=("signal_dbm", "count"),
            mean_dbm=("signal_dbm", "mean"),
            median_dbm=("signal_dbm", "median"),
            std_dbm=("signal_dbm", "std"),
            min_dbm=("signal_dbm", "min"),
            max_dbm=("signal_dbm", "max"),
            mean_distance_m=("distance_m", "mean"),
            weak_count=("signal_dbm", lambda s: int((s < weak_threshold).sum())),
        )
        .sort_values("mean_dbm", ascending=True)
    )
    grouped["weak_pct"] = (100.0 * grouped["weak_count"] / grouped["sample_count"]).round(1)
    for col in ("mean_dbm", "median_dbm", "std_dbm", "min_dbm", "max_dbm", "mean_distance_m"):
        grouped[col] = grouped[col].round(2)
    return grouped


def export_outputs(
    df: pd.DataFrame,
    loc_df: pd.DataFrame,
    model: Dict[str, Any],
    meta: Dict[str, Any],
    output_dir: str = "outputs",
) -> None:
    """CO5: Export location summaries to CSV and full metrics to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    loc_df.to_csv(os.path.join(output_dir, "location_summary.csv"), index=False)

    payload = {
        "meta": meta,
        "overall_stats": {
            "mean_dbm": round(float(df["signal_dbm"].mean()), 2),
            "median_dbm": round(float(df["signal_dbm"].median()), 2),
            "std_dbm": round(float(df["signal_dbm"].std(ddof=1)), 2),
            "min_dbm": round(float(df["signal_dbm"].min()), 2),
            "max_dbm": round(float(df["signal_dbm"].max()), 2),
            "weak_count_at_neg75": int((df["signal_dbm"] < -75.0).sum()),
        },
        "path_loss_model": {
            "ap_origin": model["ap_origin"],
            "A_dbm": model["A_dbm"],
            "path_loss_exponent_n": model["path_loss_exponent_n"],
            "mae_db": model["mae_db"],
            "rmse_db": model["rmse_db"],
            "r_squared": model["r_squared"],
        },
        "location_summary": loc_df.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
