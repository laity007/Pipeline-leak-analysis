"""Run MMSE (Kalman smoother) vs MAP (sparse+smooth) analysis for data2/(1).csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.mmse_map import run_full_analysis


def main() -> None:
    csv_path = Path("data2/(1).csv")
    results = run_full_analysis(csv_path)

    out_dir = Path("plot/(1)")
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel_idx, result in enumerate(results, start=1):
        metrics = {
            "channel": channel_idx,
            "dt_estimate": result.diagnostics.get("dt_estimate"),
            "duplicate_timestamp_ratio": result.diagnostics.get("duplicate_timestamp_ratio"),
            "snr_mmse_db": result.diagnostics.get("snr_mmse_db"),
            "snr_map_db": result.diagnostics.get("snr_map_db"),
            "resid_var_mmse": result.diagnostics.get("resid_var_mmse"),
            "resid_var_map": result.diagnostics.get("resid_var_map"),
            "resid_lag1_mmse": result.diagnostics.get("resid_lag1_mmse"),
            "resid_lag1_map": result.diagnostics.get("resid_lag1_map"),
            "mmse_process_var": result.mmse_process_var,
            "mmse_measurement_var": result.mmse_measurement_var,
            "map_lambda_smooth": result.map_lambda_smooth,
            "map_lambda_sparse": result.map_lambda_sparse,
            "mmse_em_iterations": result.iterations["mmse_em"],
            "map_iterations": result.iterations["map"],
            "timestamp_warning": bool(result.diagnostics.get("timestamp_warning", 0.0)),
        }
        metrics_path = out_dir / f"metrics_channel_{channel_idx}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Saved metrics for channel {channel_idx} to {metrics_path}")


if __name__ == "__main__":
    main()
