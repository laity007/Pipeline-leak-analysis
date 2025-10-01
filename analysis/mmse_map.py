"""Core MMSE (Kalman) and MAP estimators for pipeline leak analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import math
import csv

import numpy as np


@dataclass(frozen=True)
class ChannelResult:
    """Container for the results of a single channel analysis."""

    time: np.ndarray
    observed: np.ndarray
    mmse_state: np.ndarray
    mmse_residual: np.ndarray
    mmse_process_var: float
    mmse_measurement_var: float
    map_smooth: np.ndarray
    map_sparse: np.ndarray
    map_residual: np.ndarray
    map_lambda_smooth: float
    map_lambda_sparse: float
    iterations: Dict[str, int]
    diagnostics: Dict[str, float]


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the standardized CSV data file.

    The repository's CSV files follow the header
    ``time_sec,channel_1,channel_2,...``. Only text processing from the
    standard library is used so that the script works on a bare Python
    installation.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "time_sec":
            raise ValueError("CSV header must start with 'time_sec'")
        rows: List[List[float]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                rows.append([float(item) for item in row])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value on line {line_no} of {csv_path}"
                ) from exc
        if not rows:
            raise ValueError(f"No data rows found in {csv_path}")
    data_array = np.asarray(rows, dtype=float)
    time = data_array[:, 0]
    observations = data_array[:, 1:]
    return time, observations


def _estimate_dt(time: np.ndarray) -> Tuple[float, bool]:
    """Estimate the sampling period and flag degenerate timestamps."""

    if time.size < 2:
        return 1.0, True
    diffs = np.diff(time)
    positive = diffs[diffs > 1e-12]
    if positive.size == 0:
        return 1.0, True
    return float(np.median(positive)), False


def _kalman_em_smoother(
    y: np.ndarray,
    dt: float,
    max_iters: int = 15,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """Local-level Kalman smoother with EM estimation of process/measurement noise.

    Returns the smoothed state, residual, learnt process variance Q, measurement
    variance R, and the number of EM iterations executed.
    """

    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("Empty observation sequence")

    var_y = float(np.var(y)) if np.any(np.isfinite(y)) else 1.0
    q_floor = max(1e-3 * var_y, 1e-9)
    r_floor = max(1e-6 * var_y, 1e-12)
    x0 = float(y[0])
    P0 = var_y if var_y > 0 else 1.0
    R_hat = max(0.1 * var_y, r_floor) if var_y > 0 else 1.0
    Q_hat = max(0.05 * var_y, q_floor) if var_y > 0 else 1.0

    for iteration in range(1, max_iters + 1):
        x_pred = np.zeros(n)
        P_pred = np.zeros(n)
        x_filt = np.zeros(n)
        P_filt = np.zeros(n)

        x_prev = x0
        P_prev = P0
        for t in range(n):
            x_pred[t] = x_prev
            P_pred[t] = P_prev + Q_hat * dt

            S = P_pred[t] + R_hat
            K = P_pred[t] / S
            innovation = y[t] - x_pred[t]
            x_filt[t] = x_pred[t] + K * innovation
            P_filt[t] = (1.0 - K) * P_pred[t]

            x_prev = x_filt[t]
            P_prev = P_filt[t]

        x_smooth = np.zeros(n)
        P_smooth = np.zeros(n)
        C = np.zeros(max(n - 1, 0))
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]
        for t in range(n - 2, -1, -1):
            J = P_filt[t] / P_pred[t + 1]
            x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_filt[t] + J * (P_smooth[t + 1] - P_pred[t + 1]) * J
            C[t] = J * P_smooth[t + 1]

        Exx = P_smooth + x_smooth ** 2
        sum_Exx_lag = np.sum(P_smooth[:-1] + x_smooth[:-1] ** 2) if n > 1 else 0.0
        sum_cross = np.sum(C + x_smooth[:-1] * x_smooth[1:]) if n > 1 else 0.0
        numerator_Q = np.sum(Exx[1:]) - 2.0 * sum_cross + sum_Exx_lag
        Q_new = max(numerator_Q / max(n - 1, 1), q_floor)
        R_new = max(np.sum((y - x_smooth) ** 2 + P_smooth) / n, r_floor)

        change = max(abs(Q_new - Q_hat) / max(Q_hat, 1e-12), abs(R_new - R_hat) / max(R_hat, 1e-12))
        Q_hat = Q_new
        R_hat = R_new
        x0 = x_smooth[0]
        P0 = P_smooth[0]
        if change < tol:
            break

    # Final RTS pass with learnt parameters
    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    x_prev = x0
    P_prev = P0
    for t in range(n):
        x_pred[t] = x_prev
        P_pred[t] = P_prev + Q_hat * dt
        S = P_pred[t] + R_hat
        K = P_pred[t] / S
        innovation = y[t] - x_pred[t]
        x_filt[t] = x_pred[t] + K * innovation
        P_filt[t] = (1.0 - K) * P_pred[t]
        x_prev = x_filt[t]
        P_prev = P_filt[t]

    x_smooth = np.zeros(n)
    P_smooth = np.zeros(n)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(n - 2, -1, -1):
        J = P_filt[t] / P_pred[t + 1]
        x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J * (P_smooth[t + 1] - P_pred[t + 1]) * J

    residual = y - x_smooth
    return x_smooth, residual, Q_hat, R_hat, iteration


def _solve_tridiagonal(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""

    n = diag.size
    sub = sub.copy()
    diag = diag.copy()
    sup = sup.copy()
    rhs = rhs.copy()

    for i in range(1, n):
        w = sub[i - 1] / diag[i - 1]
        diag[i] -= w * sup[i - 1]
        rhs[i] -= w * rhs[i - 1]

    x = np.zeros_like(rhs)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i]
    return x


def _map_sparse_smooth(
    y: np.ndarray,
    dt: float,
    c_sparse: float = 2.5,
    c_smooth: float = 10.0,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Solve the MAP problem with sparse and smooth priors via alternating minimisation."""

    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("Empty observation sequence")

    mad = np.median(np.abs(y - np.median(y))) / 0.6745
    sigma = mad if mad > 0 else np.sqrt(np.mean((y - np.mean(y)) ** 2))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    lambda_sparse = c_sparse * sigma
    lambda_smooth = c_smooth / max(dt, 1e-12)

    diag = np.ones(n)
    if n > 1:
        diag[0] += lambda_smooth
        diag[-1] += lambda_smooth
        if n > 2:
            diag[1:-1] += 2.0 * lambda_smooth
    sub = -lambda_smooth * np.ones(max(n - 1, 0))
    sup = sub.copy()

    x = y.copy()
    r = np.zeros_like(y)
    for iteration in range(1, max_iters + 1):
        rhs = y - r
        x_new = _solve_tridiagonal(sub, diag, sup, rhs)
        r_new = np.sign(y - x_new) * np.maximum(np.abs(y - x_new) - lambda_sparse, 0.0)
        change = max(np.max(np.abs(x_new - x)), np.max(np.abs(r_new - r)))
        x = x_new
        r = r_new
        if change < tol * max(1.0, np.max(np.abs(y))):
            break

    residual = y - x - r
    return x, r, residual, lambda_smooth, lambda_sparse, iteration


def _compute_metrics(observed: np.ndarray, mmse: np.ndarray, mmse_resid: np.ndarray, map_signal: np.ndarray, map_resid: np.ndarray) -> Dict[str, float]:
    """Compute diagnostic metrics."""

    def _safe_var(arr: np.ndarray) -> float:
        value = float(np.var(arr))
        return value if value > 1e-12 else 1e-12

    snr_mmse = 10.0 * math.log10(_safe_var(mmse) / _safe_var(mmse_resid))
    snr_map = 10.0 * math.log10(_safe_var(map_signal) / _safe_var(map_resid))

    resid_autocorr_mmse = _autocorr_lag1(mmse_resid)
    resid_autocorr_map = _autocorr_lag1(map_resid)

    return {
        "snr_mmse_db": snr_mmse,
        "snr_map_db": snr_map,
        "resid_var_mmse": _safe_var(mmse_resid),
        "resid_var_map": _safe_var(map_resid),
        "resid_lag1_mmse": resid_autocorr_mmse,
        "resid_lag1_map": resid_autocorr_map,
    }


def _autocorr_lag1(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


@lru_cache(maxsize=8)
def run_channel_analysis(csv_path: Path, channel_idx: int) -> ChannelResult:
    """Run MMSE and MAP estimators for a specific channel."""

    time, data = load_dataset(csv_path)
    if channel_idx < 1 or channel_idx > data.shape[1]:
        raise ValueError(f"channel_idx must be between 1 and {data.shape[1]}")

    dt, degenerate = _estimate_dt(time)
    unique_count = len(np.unique(time))
    duplicate_ratio = 1.0 - unique_count / max(len(time), 1)
    y = data[:, channel_idx - 1]

    mmse_state, mmse_residual, q_hat, r_hat, mmse_iters = _kalman_em_smoother(y, dt)
    map_smooth, map_sparse, map_resid, lambda_smooth, lambda_sparse, map_iters = _map_sparse_smooth(y, dt)

    map_signal = map_smooth + map_sparse
    diagnostics = _compute_metrics(y, mmse_state, mmse_residual, map_signal, map_resid)
    if degenerate:
        diagnostics["timestamp_warning"] = 1.0
    diagnostics["duplicate_timestamp_ratio"] = duplicate_ratio
    diagnostics["dt_estimate"] = dt

    return ChannelResult(
        time=time,
        observed=y,
        mmse_state=mmse_state,
        mmse_residual=mmse_residual,
        mmse_process_var=q_hat,
        mmse_measurement_var=r_hat,
        map_smooth=map_smooth,
        map_sparse=map_sparse,
        map_residual=map_resid,
        map_lambda_smooth=lambda_smooth,
        map_lambda_sparse=lambda_sparse,
        iterations={"mmse_em": mmse_iters, "map": map_iters},
        diagnostics=diagnostics,
    )


def run_full_analysis(csv_path: Path) -> List[ChannelResult]:
    """Run the comparison for every channel in the dataset."""

    time, data = load_dataset(csv_path)
    results = []
    for idx in range(1, data.shape[1] + 1):
        results.append(run_channel_analysis(csv_path, idx))
    return results
