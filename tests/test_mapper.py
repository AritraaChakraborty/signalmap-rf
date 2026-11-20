"""Automated test suite covering CO1-CO5 logic and Flask web routes.

Run with either:
    python -m unittest discover -s tests -v
    pytest -v
"""

from __future__ import annotations

import csv
import io
import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from api.index import app
    HAS_FLASK = True
except ImportError:
    app = None
    HAS_FLASK = False

from core.analysis_np_pd import (
    dataset_to_dataframe,
    numpy_grid_summary,
    numpy_path_loss_fit,
    pandas_location_comparison,
)
from core.engine_lite import (
    compute_threshold_sensitivity,
    fit_log_distance_model,
    run_full_analysis_lite,
)
from core.io_utils import generate_synthetic_rows, validate_csv_text
from core.models import (
    SurveyDataset,
    SurveyPoint,
    classify_signal,
    is_valid_coordinate,
    is_valid_rssi,
)


class TestWirelessCoverageMapper(unittest.TestCase):
    def test_co1_validation_and_classification(self):
        self.assertTrue(is_valid_coordinate(12.5, 24.0))
        self.assertFalse(is_valid_coordinate(-1.0, 10.0))
        self.assertFalse(is_valid_coordinate(10.0, 150.0))
        self.assertTrue(is_valid_rssi(-65.0))
        self.assertFalse(is_valid_rssi(-130.0))
        self.assertFalse(is_valid_rssi(5.0))

        self.assertEqual(classify_signal(-55.0), "Strong")
        self.assertEqual(classify_signal(-68.0), "Fair")
        self.assertEqual(classify_signal(-82.0), "Weak")

    def test_co3_special_methods(self):
        p1 = SurveyPoint("SP1", 5.0, 10.0, -58.0, "Central Hall")
        p2 = SurveyPoint("SP2", 25.0, 12.0, -81.0, "Lab Block")
        self.assertLess(p2, p1)
        self.assertIn("SP1", repr(p1))

        ds = SurveyDataset([p1, p2], source_label="Test", is_synthetic=True)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], p1)
        self.assertEqual(ds[1], p2)
        self.assertEqual([p.point_id for p in ds], ["SP1", "SP2"])

        with self.assertRaises(ValueError):
            SurveyPoint("BAD", -3.0, 5.0, -60.0, "Central Hall")

    def test_co2_csv_validation_rejects_bad_rows(self):
        csv_text = (
            "point_id,x,y,signal_dbm,location\n"
            "P1,5,15,-48,Central Hall\n"
            "P2,25,10,-78,Lab Block\n"
            "P3,-4,10,-60,Central Hall\n"
            "P4,10,10,12,Lab Block\n"
            "P5,abc,10,-70,North Wing\n"
            "P6,10,10,-70,\n"
        )
        ds, rejected, meta = validate_csv_text(csv_text)
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(rejected), 4)
        self.assertEqual(meta["total_rows"], 6)

    def test_co4_co5_numpy_pandas_parity_with_lite_engine(self):
        rows = generate_synthetic_rows(seed=42)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["point_id", "x", "y", "signal_dbm", "location", "floor", "ap_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

        ds, rejected, meta = validate_csv_text(buf.getvalue(), is_synthetic=True)
        self.assertEqual(len(ds), 124)
        self.assertEqual(len(rejected), 6)

        df = dataset_to_dataframe(ds)
        np_model = numpy_path_loss_fit(df)
        lite_model = fit_log_distance_model(ds)

        self.assertAlmostEqual(np_model["A_dbm"], lite_model["A_dbm"], places=2)
        self.assertAlmostEqual(
            np_model["path_loss_exponent_n"], lite_model["path_loss_exponent_n"], places=2
        )
        self.assertAlmostEqual(np_model["rmse_db"], lite_model["rmse_db"], places=2)

        grid = numpy_grid_summary(df)
        self.assertEqual(grid.shape, (3, 4))
        self.assertTrue(any(str(v) == "nan" for v in grid.flatten()))

        loc_df = pandas_location_comparison(df)
        self.assertEqual(
            set(loc_df["location"]),
            {"North Wing", "Central Hall", "Lab Block", "East Corridor"},
        )

        payload = run_full_analysis_lite(ds, rejected, meta)
        self.assertEqual(payload["kpis"]["valid_points"], 124)
        self.assertEqual(payload["kpis"]["rejected_points"], 6)

        sens = compute_threshold_sensitivity(ds)
        self.assertGreaterEqual(sens[0]["weak_points"], sens[1]["weak_points"])
        self.assertGreaterEqual(sens[1]["weak_points"], sens[2]["weak_points"])

    @unittest.skipUnless(HAS_FLASK, "Flask not installed in this environment")
    def test_flask_routes_and_custom_404(self):
        client = app.test_client()
        for route in ["/", "/data", "/map", "/heatmap", "/model", "/stats", "/analysis", "/about"]:
            resp = client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed")

        resp_404 = client.get("/does-not-exist-route")
        self.assertEqual(resp_404.status_code, 404)
        self.assertIn(b"Signal lost at this address", resp_404.data)

        api_resp = client.get("/api/sample?weak_threshold=-75")
        self.assertEqual(api_resp.status_code, 200)
        payload = api_resp.get_json()
        self.assertEqual(payload["kpis"]["valid_points"], 124)
        self.assertEqual(payload["kpis"]["rejected_points"], 6)


if __name__ == "__main__":
    unittest.main()
