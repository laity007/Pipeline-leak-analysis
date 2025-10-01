"""Plotting utilities for MMSE vs MAP comparison scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from .mmse_map import run_channel_analysis

PlotKind = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


def plot_channel(csv_path: Path, channel_idx: int, kind: PlotKind) -> None:
    """Create the requested plot and display it interactively."""

    result = run_channel_analysis(Path(csv_path), channel_idx)
    t = result.time
    y = result.observed
    mmse = result.mmse_state
    mmse_resid = result.mmse_residual
    map_smooth = result.map_smooth
    map_sparse = result.map_sparse
    map_resid = result.map_residual
    map_signal = map_smooth + map_sparse
    metrics = result.diagnostics

    if kind == "A1":
        plt.figure(figsize=(10, 4))
        plt.plot(t, y, label="Observed")
        plt.plot(t, mmse, label="Kalman MMSE", linewidth=1.2)
        plt.title(f"Channel {channel_idx}: Observed vs Kalman MMSE")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    elif kind == "A2":
        plt.figure(figsize=(10, 3))
        plt.plot(t, mmse_resid, label="Residual")
        plt.title(f"Channel {channel_idx}: Kalman Residual")
        plt.xlabel("Time (s)")
        plt.ylabel("Residual")
        plt.grid(True)
    elif kind == "B1":
        plt.figure(figsize=(10, 4))
        plt.plot(t, y, label="Observed")
        plt.plot(t, map_smooth, label="MAP smooth x", linewidth=1.2)
        plt.plot(t, map_sparse, label="MAP sparse r", linewidth=1.2)
        plt.title(f"Channel {channel_idx}: MAP decomposition")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    elif kind == "B2":
        plt.figure(figsize=(10, 3))
        plt.plot(t, map_resid, label="Residual")
        plt.title(f"Channel {channel_idx}: MAP Residual")
        plt.xlabel("Time (s)")
        plt.ylabel("Residual")
        plt.grid(True)
    elif kind == "C1":
        plt.figure(figsize=(10, 4))
        plt.plot(t, mmse, label="Kalman MMSE", linewidth=1.2)
        plt.plot(t, map_signal, label="MAP (x + r)", linewidth=1.2)
        plt.title(f"Channel {channel_idx}: MMSE vs MAP")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    elif kind == "C2":
        plt.figure(figsize=(8, 6))
        snr_vals = [metrics.get("snr_mmse_db", 0.0), metrics.get("snr_map_db", 0.0)]
        resid_vars = [metrics.get("resid_var_mmse", 0.0), metrics.get("resid_var_map", 0.0)]
        plt.subplot(2, 1, 1)
        plt.bar(["Kalman", "MAP"], snr_vals, color=["#1f77b4", "#ff7f0e"])
        plt.ylabel("SNR (dB)")
        plt.title(f"Channel {channel_idx}: SNR comparison")
        plt.grid(axis="y")
        plt.subplot(2, 1, 2)
        plt.bar(["Kalman", "MAP"], resid_vars, color=["#1f77b4", "#ff7f0e"])
        plt.ylabel("Residual variance")
        plt.title(f"Channel {channel_idx}: Residual variance")
        plt.grid(axis="y")
        plt.tight_layout()
    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    plt.show()
