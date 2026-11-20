# Wireless-Signal Coverage Mapper

An individual Python and web-based RF survey analysis tool for studying indoor wireless signal strength (RSSI), fitting a log-distance path-loss model, comparing zones, and flagging unsurveyed coverage gaps.

> **Dataset Disclosure:** The built-in dataset (`data/sample_survey.csv`, 130 rows: 124 valid and 6 deliberately malformed) is **synthetically generated** using a fixed random seed (`seed=42`) over a 40 m × 30 m floor plan. It is provided for reproducible testing and is never presented as real field measurements.

## Features
- **Validation & Cleaning (`/data`)**: Rejects out-of-range coordinates (`0–100 m`), implausible RSSI values (`-110 to -10 dBm`), missing labels, and non-numeric inputs with row-by-row reasons, plus cleaned CSV export.
- **Interactive Coverage Map (`/map`)**: Live threshold sliders (`Strong` / `Fair` / `Weak`) and location filters.
- **Interpolated Heatmap (`/heatmap`)**: Blends inverse-distance weighting (IDW) near measured points with the fitted log-distance path-loss model inside unsurveyed gaps, hatching low-confidence cells (> 6 m from the nearest sample).
- **Log-Distance Path-Loss Model (`/model`)**: Fits `RSSI(d) = A - 10 * n * log10(d)` via ordinary least squares and reports `A`, path-loss exponent `n`, MAE, RMSE, and `R²`.
- **Statistical Breakdown (`/stats`)**: RSSI histogram, room/location comparisons, and 4 × 3 spatial grid summary.
- **Analytical Findings (`/analysis`)**: Threshold sensitivity evaluation (`-70` vs `-75` vs `-80 dBm`) and missing-location reliability discussion.

## Repository Structure
```text
wireless-coverage-mapper/
├── vercel.json               # Routes all paths to api/index.py (prevents 404 on deploy)
├── requirements.txt          # Minimal Vercel serverless dependency (Flask)
├── requirements-local.txt    # Full scientific stack (Flask, NumPy, Pandas, Matplotlib, pytest)
├── api/index.py              # Flask WSGI app (9 routes + JSON/CSV endpoints)
├── core/
│   ├── models.py             # CO1 & CO3: validation functions, SurveyPoint & SurveyDataset
│   ├── io_utils.py           # CO2: CSV I/O, synthetic generator, validation log
│   ├── engine_lite.py        # Pure-Python math mirror for lightweight Vercel serverless execution
│   ├── analysis_np_pd.py     # CO4 & CO5: NumPy arrays, path-loss fit, Pandas summaries & export
│   └── plotting.py           # CO6: Matplotlib scatter map, heatmap, histogram & model plots
├── run_pipeline.py           # Local CLI runner generating outputs/*.png, CSV & JSON
├── data/sample_survey.csv    # Reproducible synthetic dataset
├── outputs/                  # Generated Matplotlib plots and CSV/JSON summaries
├── templates/ + static/      # Light-themed editorial frontend
└── tests/test_mapper.py      # Automated pytest verification suite
```

## Run Locally
```bash
pip install -r requirements-local.txt

# 1. Run the full CO1-CO6 NumPy/Pandas/Matplotlib pipeline (writes to outputs/)
python run_pipeline.py

# 2. Run automated tests
pytest -v

# 3. Start the web dashboard locally at http://127.0.0.1:5000
python api/index.py
```

## Deploy to Vercel (Zero 404 & Under Bundle Limits)
1. Push this repository to GitHub.
2. Import the repository in Vercel (Framework Preset: **Other**; Root Directory: `./`).
3. Click **Deploy**.

`vercel.json` rewrites all routes (`/(.*)`) to `api/index.py`, and `requirements.txt` installs only `Flask`, keeping the serverless bundle well under Vercel's 250 MB limit while serving every page and custom 404 cleanly.
