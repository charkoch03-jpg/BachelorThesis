"""
Pupil-size preprocessing pipeline for MATLAB .mat inputs.

Steps: 
(1) mask invalid samples (0/32700/NaN ± window), 
(2) remove extreme fluctuations using first-derivative outliers (z-threshold), 
(3) interpolate NaNs,
(4) median filter. 

Returns lists of 1D NumPy arrays.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    bad_values: Tuple[int, ...] = (0, 32700)   # values to mask as invalid
    dilate_window: int = 5                     # ± samples around invalids
    z_thresh: float = 2.0                      # threshold on first derivative
    spike_pad: int = 5                         # ± samples to NaN around spikes
    global_diff_thresholds: bool = True        # True = thresholds over all traces
    interpolate_method: str = "linear"         # pandas interpolate method
    median_kernel: int = 11                    # median filter kernel (forced odd)
    verbose: bool = True                       # print small QC summaries

# -------------------------------------------------------------------
# MATLAB helpers
# -------------------------------------------------------------------
def _matlab_cell_to_list(mat_obj: np.ndarray) -> List[np.ndarray]:
    """
    Convert MATLAB cell-like arrays (dtype=object) or numeric matrices to
    a Python list of 1D float arrays (one per trace).
    """
    if mat_obj.dtype != object:
        # Numeric matrix: treat each row as a trace
        return [np.asarray(row, dtype=float).ravel() for row in mat_obj]
    out: List[np.ndarray] = []
    for cell in mat_obj.ravel():
        arr = np.asarray(cell, dtype=float).ravel()
        out.append(arr)
    return out

def load_mat_list(path: str | Path, varname: str) -> List[np.ndarray]:
    """Load a MATLAB variable into a list of 1D float arrays."""
    m = sio.loadmat(str(path))
    if varname not in m:
        raise KeyError(f"Variable '{varname}' not found in {path}")
    return _matlab_cell_to_list(m[varname])

def save_mat_list(path: str | Path, varname: str, traces: Sequence[np.ndarray]) -> None:
    """
    Save a list of 1D arrays as a MATLAB-style cell array shaped (N,1),
    each entry a column vector (len x 1).
    """
    cell = np.array([t.reshape(-1, 1) for t in traces], dtype=object).reshape(-1, 1)
    sio.savemat(str(path), {varname: cell})

# -------------------------------------------------------------------
# Quality check helper
# -------------------------------------------------------------------
def _qc_counts(label: str, traces: Sequence[np.ndarray]) -> None:
    if not traces:
        print(f"[QC] {label}: (no traces)")
        return
    arr = np.concatenate([np.asarray(t).ravel() for t in traces])
    zeros = int(np.sum(arr == 0))
    v32700 = int(np.sum(arr == 32700))
    nans = int(np.sum(np.isnan(arr)))
    total = arr.size
    print(f"[QC] {label}: total={total:,} zeros={zeros:,} 32700={v32700:,} NaNs={nans:,}")

# -------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------
def mask_invalid_values(
    traces: Sequence[np.ndarray],
    bad_values: Tuple[int, ...],
    dilate_window: int
) -> List[np.ndarray]:
    """
    Mask invalid samples (0/32700/NaN) and a ±window around them as NaN.
    """
    struct = np.ones(2 * dilate_window + 1, dtype=int)
    out: List[np.ndarray] = []
    for sig in traces:
        s = np.asarray(sig, dtype=float).copy()
        bad = np.isnan(s)
        for bv in bad_values:
            bad |= (s == bv)
        if bad.any():
            bad = binary_dilation(bad, structure=struct)
            s[bad] = np.nan
        out.append(s)
    return out

def _compute_diff_thresholds(traces: Sequence[np.ndarray], z: float) -> Tuple[float, float]:
    """Compute mean ± z*SD over first differences across traces (ignoring NaNs)."""
    diffs = []
    for t in traces:
        t = np.asarray(t, dtype=float)
        if t.size > 1:
            diffs.append(np.diff(t))
    if not diffs:
        return (-np.inf, np.inf)
    diffs_all = np.concatenate(diffs)
    mean = np.nanmean(diffs_all)
    sd = np.nanstd(diffs_all)
    return (mean - z * sd, mean + z * sd)

def remove_extreme_fluctuations(
    traces: Sequence[np.ndarray],
    z_thresh: float,
    spike_pad: int,
    global_thresholds: bool
) -> List[np.ndarray]:
    """
    Replace windows around derivative outliers with NaN.
    If global_thresholds=True, thresholds are computed across all traces;
    otherwise thresholds are computed per trace.
    """
    if global_thresholds:
        lower, upper = _compute_diff_thresholds(traces, z_thresh)

    out: List[np.ndarray] = []
    for sig in traces:
        s = np.asarray(sig, dtype=float).copy()
        if s.size < 2:
            out.append(s)
            continue

        if not global_thresholds:
            lower, upper = _compute_diff_thresholds([s], z_thresh)

        d = np.diff(s)
        if not np.all(np.isfinite([lower, upper])):
            out.append(s)
            continue

        bad_idx = np.where((d < lower) | (d > upper))[0]  # transition index
        for idx in bad_idx:
            start = max(0, idx - spike_pad)
            end = min(s.size, idx + 1 + spike_pad)
            s[start:end] = np.nan
        out.append(s)
    return out

def interpolate_nans(traces: Sequence[np.ndarray], method: str = "linear") -> List[np.ndarray]:
    """Interpolate NaNs with pandas (both directions)."""
    out: List[np.ndarray] = []
    for sig in traces:
        ser = pd.Series(sig, dtype="float64")
        filled = ser.interpolate(method=method, limit_direction="both").to_numpy()
        out.append(filled)
    return out

def median_filter(traces: Sequence[np.ndarray], kernel: int) -> List[np.ndarray]:
    """Apply a median filter with an odd kernel (auto-adjusted if needed)."""
    out: List[np.ndarray] = []
    for sig in traces:
        k = max(1, kernel)
        if k % 2 == 0:
            k += 1
        if len(sig) > 0:
            k = min(k, (len(sig) // 2) * 2 + 1)  # largest odd <= len(sig)
        out.append(medfilt(sig, kernel_size=k))
    return out

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def preprocess_traces(
    traces: Sequence[np.ndarray],
    cfg: PreprocessConfig = PreprocessConfig()
) -> List[np.ndarray]:
    """
    Run the full preprocessing pipeline on a list of 1D arrays.
    Returns: list of 1D NumPy arrays (same lengths as inputs).
    """
    if cfg.verbose:
        _qc_counts("raw", traces)

    step1 = mask_invalid_values(traces, cfg.bad_values, cfg.dilate_window)
    if cfg.verbose:
        print(" → after mask_invalid_values")
        _qc_counts("masked", step1)

    step2 = remove_extreme_fluctuations(step1, cfg.z_thresh, cfg.spike_pad, cfg.global_diff_thresholds)
    if cfg.verbose:
        print(" → after remove_extreme_fluctuations")
        _qc_counts("spike-removed", step2)

    step3 = interpolate_nans(step2, cfg.interpolate_method)
    if cfg.verbose:
        print(" → after interpolate_nans")
        _qc_counts("interpolated", step3)

    step4 = median_filter(step3, cfg.median_kernel)
    if cfg.verbose:
        print(" → after median_filter")
        _qc_counts("filtered", step4)

    return step4

def preprocess_mat_files(
    left_mat_path: str | Path,
    right_mat_path: str | Path,
    left_var: str = "pupilSizeLeft",
    right_var: str = "pupilSizeRight",
    cfg: PreprocessConfig = PreprocessConfig()
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load left/right MATLAB files, preprocess with the same pipeline, and return
    (left_processed, right_processed) as lists of 1D arrays.
    """
    left_raw = load_mat_list(left_mat_path, left_var)
    right_raw = load_mat_list(right_mat_path, right_var)
    left_proc = preprocess_traces(left_raw, cfg)
    right_proc = preprocess_traces(right_raw, cfg)
    return left_proc, right_proc
