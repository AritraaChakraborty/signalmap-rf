"""CO1 (control flow & functions) and CO3 (classes & special methods).

Defines coordinate/RSSI validation functions, zone classification, Euclidean
distance computation, the SurveyPoint entity class, and the SurveyDataset
collection class implementing __init__, __len__, __getitem__, and __iter__.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Physical floor bounds (meters) and plausible 2.4/5 GHz Wi-Fi RSSI bounds (dBm)
MIN_X, MAX_X = 0.0, 100.0
MIN_Y, MAX_Y = 0.0, 100.0
MIN_RSSI, MAX_RSSI = -110.0, -10.0

DEFAULT_AP_ORIGIN: Tuple[float, float] = (4.0, 15.0)
DEFAULT_STRONG_THRESHOLD: float = -60.0
DEFAULT_WEAK_THRESHOLD: float = -75.0


def is_valid_coordinate(x: float, y: float) -> bool:
    """CO1: Return True if (x, y) are finite numbers within floor bounds."""
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return False
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False
    return (MIN_X <= x <= MAX_X) and (MIN_Y <= y <= MAX_Y)


def is_valid_rssi(signal_dbm: float) -> bool:
    """CO1: Return True if RSSI is finite and within [-110, -10] dBm."""
    if not isinstance(signal_dbm, (int, float)):
        return False
    if math.isnan(signal_dbm) or math.isinf(signal_dbm):
        return False
    return MIN_RSSI <= signal_dbm <= MAX_RSSI


def classify_signal(
    signal_dbm: float,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> str:
    """CO1: Classify RSSI into 'Strong', 'Fair', or 'Weak'."""
    if signal_dbm >= strong_threshold:
        return "Strong"
    if signal_dbm >= weak_threshold:
        return "Fair"
    return "Weak"


def euclidean_distance(
    x: float,
    y: float,
    origin: Tuple[float, float] = DEFAULT_AP_ORIGIN,
    min_distance: float = 0.5,
) -> float:
    """CO1: Compute distance from AP origin, floored at min_distance to avoid log10(0)."""
    dist = math.hypot(x - origin[0], y - origin[1])
    return max(dist, min_distance)


class SurveyPoint:
    """CO3: Represents one validated wireless signal survey observation.

    Special methods implemented:
    - __init__: validates coordinates, RSSI, and location label on construction.
    - __repr__: provides a concise diagnostic representation for logs/debugging.
    - __eq__ / __lt__: enables sorting and comparing points by signal strength.
    """

    def __init__(
        self,
        point_id: str,
        x: float,
        y: float,
        signal_dbm: float,
        location: str,
        floor: str = "Floor 1",
        ap_id: str = "AP-01",
    ) -> None:
        if not str(point_id).strip():
            raise ValueError("point_id cannot be empty")
        if not str(location).strip():
            raise ValueError("location cannot be empty")
        if not is_valid_coordinate(x, y):
            raise ValueError(
                f"Coordinates ({x}, {y}) out of valid range [{MIN_X}, {MAX_X}] x [{MIN_Y}, {MAX_Y}]"
            )
        if not is_valid_rssi(signal_dbm):
            raise ValueError(
                f"signal_dbm ({signal_dbm}) out of plausible range [{MIN_RSSI}, {MAX_RSSI}] dBm"
            )

        self.point_id: str = str(point_id).strip()
        self.x: float = float(x)
        self.y: float = float(y)
        self.signal_dbm: float = float(signal_dbm)
        self.location: str = str(location).strip()
        self.floor: str = str(floor).strip() or "Floor 1"
        self.ap_id: str = str(ap_id).strip() or "AP-01"

    def distance_to(self, origin: Tuple[float, float] = DEFAULT_AP_ORIGIN) -> float:
        return euclidean_distance(self.x, self.y, origin=origin)

    def zone(
        self,
        weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    ) -> str:
        return classify_signal(self.signal_dbm, weak_threshold, strong_threshold)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "SurveyPoint":
        """Parse and validate a raw CSV dictionary row into a SurveyPoint."""
        try:
            x_val = float(row["x"])
            y_val = float(row["y"])
            rssi_val = float(row["signal_dbm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric or missing x/y/signal_dbm: {exc}") from exc

        return cls(
            point_id=str(row.get("point_id", "")).strip(),
            x=x_val,
            y=y_val,
            signal_dbm=rssi_val,
            location=str(row.get("location", "")).strip(),
            floor=str(row.get("floor", "Floor 1")).strip() or "Floor 1",
            ap_id=str(row.get("ap_id", "AP-01")).strip() or "AP-01",
        )

    def to_dict(
        self,
        weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        origin: Tuple[float, float] = DEFAULT_AP_ORIGIN,
    ) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "signal_dbm": round(self.signal_dbm, 2),
            "location": self.location,
            "floor": self.floor,
            "ap_id": self.ap_id,
            "distance_m": round(self.distance_to(origin), 2),
            "zone": self.zone(weak_threshold, strong_threshold),
        }

    def __repr__(self) -> str:
        return (
            f"SurveyPoint(id={self.point_id!r}, x={self.x:.1f}, y={self.y:.1f}, "
            f"rssi={self.signal_dbm:.1f} dBm, loc={self.location!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SurveyPoint):
            return NotImplemented
        return (
            self.point_id == other.point_id
            and math.isclose(self.x, other.x)
            and math.isclose(self.y, other.y)
            and math.isclose(self.signal_dbm, other.signal_dbm)
            and self.location == other.location
        )

    def __lt__(self, other: "SurveyPoint") -> bool:
        if not isinstance(other, SurveyPoint):
            return NotImplemented
        return self.signal_dbm < other.signal_dbm


class SurveyDataset:
    """CO3: Collection container for validated SurveyPoint instances.

    Special methods implemented:
    - __init__: initializes internal list of SurveyPoint objects and metadata.
    - __len__: allows len(dataset) to return the number of valid survey points.
    - __getitem__: supports indexing (dataset[0]) and slicing (dataset[:5]).
    - __iter__: makes the dataset directly iterable in loops and comprehensions.
    """

    def __init__(
        self,
        points: Optional[List[SurveyPoint]] = None,
        source_label: str = "Synthetic Floor Survey",
        is_synthetic: bool = True,
    ) -> None:
        self._points: List[SurveyPoint] = list(points) if points else []
        self.source_label: str = source_label
        self.is_synthetic: bool = is_synthetic

    def add(self, point: SurveyPoint) -> None:
        if not isinstance(point, SurveyPoint):
            raise TypeError("Expected a SurveyPoint instance")
        self._points.append(point)

    def locations(self) -> List[str]:
        return sorted({p.location for p in self._points})

    def filter_by_location(self, location: str) -> "SurveyDataset":
        if not location or location == "All":
            return self
        return SurveyDataset(
            [p for p in self._points if p.location == location],
            source_label=f"{self.source_label} ({location})",
            is_synthetic=self.is_synthetic,
        )

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[SurveyPoint, List[SurveyPoint]]:
        return self._points[index]

    def __iter__(self) -> Iterator[SurveyPoint]:
        return iter(self._points)

    def __repr__(self) -> str:
        return f"SurveyDataset(count={len(self)}, source={self.source_label!r})"
