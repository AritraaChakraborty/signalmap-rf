"""End-to-end local Python pipeline demonstrating CO1 through CO6.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import os

from core.analysis_np_pd import (
    dataset_to_dataframe,
    export_outputs,
    numpy_grid_summary,
    numpy_interpolated_surface,
    numpy_path_loss_fit,
    pandas_location_comparison,
)
from core.io_utils import generate_synthetic_rows, load_csv_file, write_rows_to_csv
from core.plotting import generate_all_plots


def main() -> None:
    data_path = os.path.join("data", "sample_survey.csv")
    os.makedirs("data", exist_ok=True)

    # Ensure deterministic synthetic dataset exists on disk
    if not os.path.exists(data_path):
        rows = generate_synthetic_rows(seed=42)
        write_rows_to_csv(rows, data_path)

    # CO1, CO2, CO3: Load CSV & validate into SurveyDataset of SurveyPoint objects
    dataset, rejected, meta = load_csv_file(
        data_path,
        source_label="Synthetic Floor Survey (sample_survey.csv)",
        is_synthetic=True,
    )
    print(f"Loaded {len(dataset)} valid points ({len(rejected)} rejected) from {data_path}")
    print(f"First point via __getitem__: {dataset[0]}")

    # CO4 & CO5: NumPy + Pandas analysis
    df = dataset_to_dataframe(dataset)
    model = numpy_path_loss_fit(df)
    grid = numpy_grid_summary(df)
    XX, YY, blended, nearest_dist = numpy_interpolated_surface(df, model)
    loc_df = pandas_location_comparison(df)

    print(
        f"Log-Distance Model: RSSI(d) = {model['A_dbm']:.2f} - 10 * "
        f"{model['path_loss_exponent_n']:.2f} * log10(d) | "
        f"MAE={model['mae_db']:.2f} dB, RMSE={model['rmse_db']:.2f} dB, R2={model['r_squared']:.3f}"
    )
    print("\nLocation Comparison (Pandas):")
    print(loc_df.to_string(index=False))
    print("\nCoarse 3x4 Mean RSSI Grid (NumPy, NaN = unsurveyed):")
    print(grid)

    # CO5 & CO6: Export CSV/JSON and Matplotlib figures
    export_outputs(df, loc_df, model, meta, output_dir="outputs")
    generate_all_plots(df, model, XX, YY, blended, nearest_dist, output_dir="outputs")
    print("\nSaved CSV/JSON summaries and Matplotlib plots to outputs/.")


if __name__ == "__main__":
    main()
