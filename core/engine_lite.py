"""Lightweight pure-Python analysis engine for Vercel serverless deployment.

Implements the exact same mathematical formulas as core/analysis_np_pd.py
(log-distance path-loss OLS fit, 2D floor grid interpolation, unsurveyed gap
detection, location aggregation, histogram binning, and threshold sensitivity)
without requiring NumPy/Pandas inside the 250 MB Vercel Lambda bundle.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple

from core.models import (
    DEFAULT_AP_ORIGIN,
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    SurveyDataset,
    classify_signal,
    euclidean_distance,
)


def fit_log_distance_model(
    dataset: SurveyDataset,
    origin: Tuple[float, float] = DEFAULT_AP_ORIGIN,
) -> Dict[str, Any]:
    """Fit RSSI(d) = A - 10 * n * log10(d) via ordinary least squares."""
    if len(dataset) < 2:
        raise ValueError("At least 2 valid survey points are required to fit the model.")

    us: List[float] = []
    ys: List[float] = []
    dists: List[float] = []

    for p in dataset:
        d = p.distance_to(origin)
        dists.append(d)
        us.append(10.0 * math.log10(d))
        ys.append(p.signal_dbm)

    mean_u = sum(us) / len(us)
    mean_y = sum(ys) / len(ys)

    ss_uu = sum((u - mean_u) ** 2 for u in us)
    ss_uy = sum((u - mean_u) * (y - mean_y) for u, y in zip(us, ys))

    slope = ss_uy / ss_uu if ss_uu > 1e-12 else -2.0
    intercept_a = mean_y - slope * mean_u
    path_loss_n = -slope

    preds = [intercept_a - 10.0 * path_loss_n * math.log10(d) for d in dists]
    residuals = [y - yhat for y, yhat in zip(ys, preds)]

    mae = sum(abs(r) for r in residuals) / len(residuals)
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Smooth reference curve for plotting
    d_min = max(0.5, min(dists))
    d_max = max(dists)
    curve = []
    steps = 40
    for i in range(steps + 1):
        d = d_min + (d_max - d_min) * (i / steps)
        curve.append(
            {
                "distance_m": round(d, 2),
                "predicted_dbm": round(intercept_a - 10.0 * path_loss_n * math.log10(d), 2),
            }
        )

    pairs = []
    for p, d, yhat, res in zip(dataset, dists, preds, residuals):
        pairs.append(
            {
                "point_id": p.point_id,
                "location": p.location,
                "distance_m": round(d, 2),
                "measured_dbm": round(p.signal_dbm, 2),
                "predicted_dbm": round(yhat, 2),
                "residual_db": round(res, 2),
            }
        )

    return {
        "ap_origin": {"x": origin[0], "y": origin[1]},
        "A_dbm": round(intercept_a, 3),
        "path_loss_exponent_n": round(path_loss_n, 3),
        "mae_db": round(mae, 3),
        "rmse_db": round(rmse, 3),
        "r_squared": round(r2, 3),
        "equation": f"RSSI(d) = {intercept_a:.2f} - 10 * {path_loss_n:.2f} * log10(d)",
        "curve": curve,
        "points": pairs,
    }


def build_interpolated_grid(
    dataset: SurveyDataset,
    model: Dict[str, Any],
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    nx: int = 20,
    ny: int = 15,
    gap_radius_m: float = 6.0,
) -> Dict[str, Any]:
    """Interpolate RSSI over the floor plane using hybrid IDW + log-distance fallback.

    Cells whose nearest measured survey point is farther than gap_radius_m are
    flagged as unsurveyed gaps (low confidence).
    """
    xs = [p.x for p in dataset]
    ys = [p.y for p in dataset]
    x_min, x_max = 0.0, max(40.0, math.ceil(max(xs)))
    y_min, y_max = 0.0, max(30.0, math.ceil(max(ys)))

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    a_val = model["A_dbm"]
    n_val = model["path_loss_exponent_n"]
    origin = (model["ap_origin"]["x"], model["ap_origin"]["y"])

    cells = []
    weak_cells = 0
    gap_cells = 0

    for j in range(ny):
        cy = y_min + (j + 0.5) * dy
        for i in range(nx):
            cx = x_min + (i + 0.5) * dx

            # Find distances to all measured points
            nearest_d = float("inf")
            weighted_sum = 0.0
            weight_total = 0.0

            for p in dataset:
                d = math.hypot(cx - p.x, cy - p.y)
                if d < nearest_d:
                    nearest_d = d
                w = 1.0 / ((d + 0.75) ** 2)
                weighted_sum += w * p.signal_dbm
                weight_total += w

            idw_est = weighted_sum / weight_total if weight_total > 0 else -80.0
            d_ap = euclidean_distance(cx, cy, origin=origin)
            model_est = a_val - 10.0 * n_val * math.log10(d_ap)

            # Blend: near measured points trust IDW; in unsurveyed gaps smoothly
            # shift weight toward the physical log-distance model.
            alpha = min(1.0, nearest_d / (gap_radius_m * 1.5))
            blended = (1.0 - alpha) * idw_est + alpha * model_est

            is_gap = nearest_d > gap_radius_m
            zone = classify_signal(blended, weak_threshold, strong_threshold)
            if zone == "Weak":
                weak_cells += 1
            if is_gap:
                gap_cells += 1

            cells.append(
                {
                    "col": i,
                    "row": j,
                    "cx": round(cx, 2),
                    "cy": round(cy, 2),
                    "rssi_dbm": round(blended, 2),
                    "model_dbm": round(model_est, 2),
                    "idw_dbm": round(idw_est, 2),
                    "nearest_sample_m": round(nearest_d, 2),
                    "is_gap": is_gap,
                    "zone": zone,
                }
            )

    total_cells = nx * ny
    return {
        "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "nx": nx,
        "ny": ny,
        "gap_radius_m": gap_radius_m,
        "total_cells": total_cells,
        "weak_cell_count": weak_cells,
        "weak_area_pct": round(100.0 * weak_cells / total_cells, 1),
        "gap_cell_count": gap_cells,
        "gap_area_pct": round(100.0 * gap_cells / total_cells, 1),
        "cells": cells,
    }


def compute_location_summary(
    dataset: SurveyDataset,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Group survey points by location and compute descriptive & weak-zone stats."""
    groups: Dict[str, List[float]] = {}
    dists: Dict[str, List[float]] = {}
    for p in dataset:
        groups.setdefault(p.location, []).append(p.signal_dbm)
        dists.setdefault(p.location, []).append(p.distance_to())

    summary = []
    for loc in sorted(groups.keys()):
        vals = groups[loc]
        d_vals = dists[loc]
        weak_cnt = sum(1 for v in vals if v < weak_threshold)
        strong_cnt = sum(1 for v in vals if v >= strong_threshold)
        mean_v = statistics.mean(vals)
        summary.append(
            {
                "location": loc,
                "count": len(vals),
                "mean_dbm": round(mean_v, 2),
                "median_dbm": round(statistics.median(vals), 2),
                "std_dbm": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                "min_dbm": round(min(vals), 2),
                "max_dbm": round(max(vals), 2),
                "mean_distance_m": round(statistics.mean(d_vals), 2),
                "weak_count": weak_cnt,
                "weak_pct": round(100.0 * weak_cnt / len(vals), 1),
                "strong_count": strong_cnt,
                "status": classify_signal(mean_v, weak_threshold, strong_threshold),
            }
        )
    summary.sort(key=lambda r: r["mean_dbm"])
    return summary


def compute_grid_summary(
    dataset: SurveyDataset,
    cols: int = 4,
    rows: int = 3,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Aggregate measured points into a coarse 4x3 spatial grid."""
    xs = [p.x for p in dataset]
    ys = [p.y for p in dataset]
    x_max = max(40.0, math.ceil(max(xs)))
    y_max = max(30.0, math.ceil(max(ys)))
    dx = x_max / cols
    dy = y_max / rows

    bins: Dict[Tuple[int, int], List[float]] = {}
    for p in dataset:
        c = min(cols - 1, max(0, int(p.x // dx)))
        r = min(rows - 1, max(0, int(p.y // dy)))
        bins.setdefault((r, c), []).append(p.signal_dbm)

    grid_rows = []
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            vals = bins.get((r, c), [])
            grid_rows.append(
                {
                    "cell": f"R{r + 1}C{c + 1}",
                    "x_range": f"{c * dx:.0f}-{(c + 1) * dx:.0f} m",
                    "y_range": f"{r * dy:.0f}-{(r + 1) * dy:.0f} m",
                    "count": len(vals),
                    "mean_dbm": round(statistics.mean(vals), 2) if vals else None,
                    "weak_count": sum(1 for v in vals if v < weak_threshold),
                    "unsurveyed": len(vals) == 0,
                }
            )
    return grid_rows


def compute_histogram(dataset: SurveyDataset, bin_width: int = 5) -> List[Dict[str, Any]]:
    """Compute RSSI frequency histogram in 5 dB bins."""
    vals = [p.signal_dbm for p in dataset]
    lo = int(math.floor(min(vals) / bin_width) * bin_width)
    hi = int(math.ceil(max(vals) / bin_width) * bin_width)
    if lo == hi:
        hi = lo + bin_width

    bins = []
    edge = lo
    while edge < hi:
        next_edge = edge + bin_width
        cnt = sum(
            1
            for v in vals
            if (edge <= v < next_edge) or (next_edge == hi and v == hi)
        )
        bins.append({"label": f"{edge} to {next_edge}", "min_dbm": edge, "max_dbm": next_edge, "count": cnt})
        edge = next_edge
    return bins


def compute_threshold_sensitivity(dataset: SurveyDataset) -> List[Dict[str, Any]]:
    """BL5: Evaluate how changing the weak-signal threshold alters conclusions."""
    thresholds = [
        (-70.0, "Strict (-70 dBm)", "VoIP, video calls, high-throughput streaming"),
        (-75.0, "Default (-75 dBm)", "Standard web browsing & classroom workloads"),
        (-80.0, "Lenient (-80 dBm)", "Basic messaging & low-rate IoT telemetry"),
    ]
    results = []
    total = len(dataset)
    for thr, label, use_case in thresholds:
        weak_pts = [p for p in dataset if p.signal_dbm < thr]
        loc_stats = compute_location_summary(dataset, weak_threshold=thr)
        weak_locs = [r["location"] for r in loc_stats if r["mean_dbm"] < thr]
        results.append(
            {
                "threshold_dbm": thr,
                "label": label,
                "use_case": use_case,
                "weak_points": len(weak_pts),
                "weak_pct": round(100.0 * len(weak_pts) / total, 1) if total else 0.0,
                "weak_mean_locations": weak_locs,
            }
        )
    return results


def run_full_analysis_lite(
    dataset: SurveyDataset,
    rejected: List[Dict[str, Any]],
    meta: Dict[str, Any],
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> Dict[str, Any]:
    """Produce complete JSON payload consumed by all frontend pages."""
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 valid survey points.")

    vals = [p.signal_dbm for p in dataset]
    sorted_vals = sorted(vals)
    q1 = sorted_vals[len(sorted_vals) // 4]
    q3 = sorted_vals[(3 * len(sorted_vals)) // 4]

    weak_count = sum(1 for v in vals if v < weak_threshold)
    strong_count = sum(1 for v in vals if v >= strong_threshold)
    fair_count = len(vals) - weak_count - strong_count

    model = fit_log_distance_model(dataset)
    heatmap = build_interpolated_grid(
        dataset,
        model,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
    )
    locations = compute_location_summary(dataset, weak_threshold, strong_threshold)
    grid_summary = compute_grid_summary(dataset, weak_threshold=weak_threshold)
    histogram = compute_histogram(dataset)
    sensitivity = compute_threshold_sensitivity(dataset)

    return {
        "meta": meta,
        "thresholds": {
            "weak_dbm": weak_threshold,
            "strong_dbm": strong_threshold,
        },
        "kpis": {
            "valid_points": len(dataset),
            "rejected_points": len(rejected),
            "mean_dbm": round(statistics.mean(vals), 2),
            "median_dbm": round(statistics.median(vals), 2),
            "std_dbm": round(statistics.stdev(vals), 2),
            "min_dbm": round(min(vals), 2),
            "max_dbm": round(max(vals), 2),
            "q1_dbm": round(q1, 2),
            "q3_dbm": round(q3, 2),
            "strong_count": strong_count,
            "fair_count": fair_count,
            "weak_count": weak_count,
            "weak_pct": round(100.0 * weak_count / len(vals), 1),
        },
        "points": [p.to_dict(weak_threshold, strong_threshold) for p in dataset],
        "rejected": rejected,
        "model": model,
        "heatmap": heatmap,
        "locations": locations,
        "grid_summary": grid_summary,
        "histogram": histogram,
        "sensitivity": sensitivity,
    }
