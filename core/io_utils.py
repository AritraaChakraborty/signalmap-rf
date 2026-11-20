"""CO2 (data structures and file I/O).

Handles CSV/text loading, deterministic synthetic dataset generation (explicitly
labelled as simulated), row-by-row validation with rejection reasons, and
cleaned CSV serialization.
"""

from __future__ import annotations

import csv
import io
import math
import random
from typing import Any, Dict, List, Tuple

from core.models import (
    DEFAULT_AP_ORIGIN,
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    SurveyDataset,
    SurveyPoint,
)

REQUIRED_COLUMNS = ("point_id", "x", "y", "signal_dbm", "location")


def generate_synthetic_rows(seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a reproducible synthetic indoor Wi-Fi survey dataset.

    Floor size: 40 m x 30 m. AP located at (4.0, 15.0).
    True underlying log-distance model:
        RSSI(d) = -34.0 - 10 * 2.65 * log10(d) - wall_loss + noise(0, 2.6)
    Deliberately leaves an unsurveyed gap in x in [28, 39], y in [18, 29]
    to test missing-location interpolation and confidence flagging, and appends
    6 invalid rows to verify data validation.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    zones = [
        # (name, x_range, y_range, count, extra_wall_loss_db, floor, ap_id)
        ("North Wing", (1.5, 18.0), (16.5, 28.5), 32, 1.5, "Floor 1", "AP-01"),
        ("Central Hall", (2.0, 20.0), (4.0, 16.0), 34, 0.0, "Floor 1", "AP-01"),
        ("Lab Block", (20.5, 38.5), (1.5, 15.5), 34, 5.5, "Floor 1", "AP-01"),
        # Deliberately stop x at 27.5 in East Corridor so [28..39, 18..29] is unsurveyed
        ("East Corridor", (18.5, 27.5), (16.5, 28.5), 24, 4.0, "Floor 1", "AP-01"),
    ]

    idx = 1
    for name, (x0, x1), (y0, y1), count, wall_loss, floor, ap_id in zones:
        for _ in range(count):
            x = round(rng.uniform(x0, x1), 2)
            y = round(rng.uniform(y0, y1), 2)
            dist = max(math.hypot(x - DEFAULT_AP_ORIGIN[0], y - DEFAULT_AP_ORIGIN[1]), 0.5)
            rssi = -34.0 - 10.0 * 2.65 * math.log10(dist) - wall_loss + rng.gauss(0.0, 2.6)
            rssi = round(max(-102.0, min(-22.0, rssi)), 1)
            rows.append(
                {
                    "point_id": f"SP{idx:03d}",
                    "x": x,
                    "y": y,
                    "signal_dbm": rssi,
                    "location": name,
                    "floor": floor,
                    "ap_id": ap_id,
                }
            )
            idx += 1

    # Deliberately malformed rows to test CO1/CO2/CO3 validation
    invalid_rows = [
        {"point_id": "SP901", "x": -5.0, "y": 12.0, "signal_dbm": -62.0, "location": "Central Hall", "floor": "Floor 1", "ap_id": "AP-01"},
        {"point_id": "SP902", "x": 14.0, "y": 140.0, "signal_dbm": -68.0, "location": "North Wing", "floor": "Floor 1", "ap_id": "AP-01"},
        {"point_id": "SP903", "x": 18.0, "y": 10.0, "signal_dbm": 15.0, "location": "Lab Block", "floor": "Floor 1", "ap_id": "AP-01"},
        {"point_id": "SP904", "x": 22.0, "y": 8.0, "signal_dbm": -145.0, "location": "Lab Block", "floor": "Floor 1", "ap_id": "AP-01"},
        {"point_id": "SP905", "x": "abc", "y": 11.0, "signal_dbm": -71.0, "location": "East Corridor", "floor": "Floor 1", "ap_id": "AP-01"},
        {"point_id": "SP906", "x": 10.0, "y": 9.0, "signal_dbm": -64.0, "location": "", "floor": "Floor 1", "ap_id": "AP-01"},
    ]
    rows.extend(invalid_rows)
    return rows


def write_rows_to_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    """CO2: Write list of row dicts to a CSV file."""
    fieldnames = ["point_id", "x", "y", "signal_dbm", "location", "floor", "ap_id"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def validate_csv_text(
    csv_text: str,
    source_label: str = "Uploaded CSV",
    is_synthetic: bool = False,
) -> Tuple[SurveyDataset, List[Dict[str, Any]], Dict[str, Any]]:
    """Parse CSV text, validate every row, and return (dataset, rejected_rows, meta)."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    headers = [h.strip() for h in (reader.fieldnames or [])]
    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

    dataset = SurveyDataset(source_label=source_label, is_synthetic=is_synthetic)
    rejected: List[Dict[str, Any]] = []
    total_rows = 0

    for line_num, raw_row in enumerate(reader, start=2):
        total_rows += 1
        cleaned_row = {
            (k.strip() if k else ""): (v.strip() if isinstance(v, str) else v)
            for k, v in raw_row.items()
        }
        try:
            point = SurveyPoint.from_dict(cleaned_row)
            dataset.add(point)
        except ValueError as exc:
            rejected.append(
                {
                    "row_number": line_num,
                    "point_id": cleaned_row.get("point_id", ""),
                    "x": cleaned_row.get("x", ""),
                    "y": cleaned_row.get("y", ""),
                    "signal_dbm": cleaned_row.get("signal_dbm", ""),
                    "location": cleaned_row.get("location", ""),
                    "reason": str(exc),
                }
            )

    meta = {
        "source_label": source_label,
        "is_synthetic": is_synthetic,
        "total_rows": total_rows,
        "valid_rows": len(dataset),
        "rejected_rows": len(rejected),
    }
    return dataset, rejected, meta


def load_csv_file(
    filepath: str,
    source_label: str = "Synthetic Floor Survey (sample_survey.csv)",
    is_synthetic: bool = True,
) -> Tuple[SurveyDataset, List[Dict[str, Any]], Dict[str, Any]]:
    """CO2: Read a CSV file from disk and validate its survey points."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return validate_csv_text(text, source_label=source_label, is_synthetic=is_synthetic)


def dataset_to_csv_string(
    dataset: SurveyDataset,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> str:
    """Serialize a validated SurveyDataset to CSV text with computed zone & distance."""
    output = io.StringIO()
    fieldnames = [
        "point_id",
        "x",
        "y",
        "signal_dbm",
        "location",
        "floor",
        "ap_id",
        "distance_m",
        "zone",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for p in dataset:
        writer.writerow(p.to_dict(weak_threshold, strong_threshold))
    return output.getvalue()
