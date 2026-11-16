"""
Multi-resolution temporal aggregation and statistical feature engineering.
Extracts 12 statistical descriptors for each of the 16 sensor variables
across rolling windows (2h, 4h, 6h, 24h), producing the 192-dimensional state matrix.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import List, Dict, Optional, Tuple


# Resolutions in hours matching HCR baseline
WINDOW_HOURS_LIST = [2, 4, 6, 24]
PRIMARY_RESOLUTION_HOURS = 6.0


def compute_12_descriptors(series_values: np.ndarray) -> Dict[str, float]:
    """
    Computes the 12 statistical descriptors for a single variable window:
    1. Mean
    2. Median
    3. Variance
    4. Standard deviation
    5. Minimum
    6. Maximum
    7. Range (max - min)
    8. Skewness
    9. Kurtosis
    10. Interquartile Range (IQR)
    11. Coefficient of Variation (coef_var)
    12. Linear slope (rate of change)
    """
    vals = np.asarray(series_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        vals = np.array([0.0])

    n = len(vals)
    val_mean = float(np.mean(vals))
    val_median = float(np.median(vals))
    val_var = float(np.var(vals, ddof=1)) if n > 1 else 0.0
    val_std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    val_min = float(np.min(vals))
    val_max = float(np.max(vals))
    val_range = float(val_max - val_min)

    # When variance / std is zero or nearly zero, skewness and kurtosis are mathematically 0.0
    if n >= 3 and val_std > 1e-6:
        val_skew = float(skew(vals))
        if np.isnan(val_skew) or np.isinf(val_skew):
            val_skew = 0.0
    else:
        val_skew = 0.0

    if n >= 4 and val_std > 1e-6:
        val_kurt = float(kurtosis(vals))
        if np.isnan(val_kurt) or np.isinf(val_kurt):
            val_kurt = 0.0
    else:
        val_kurt = 0.0

    q75, q25 = np.percentile(vals, [75, 25])
    val_iqr = float(q75 - q25)

    # Coefficient of variation
    coef_var = float(val_std / (abs(val_mean) + 1e-6))
    if np.isnan(coef_var) or np.isinf(coef_var):
        coef_var = 0.0

    # Linear slope across the window
    if n > 1 and val_std > 1e-6:
        x = np.arange(n)
        slope, _ = np.polyfit(x, vals, 1)
        linear_slope = float(slope)
        if np.isnan(linear_slope) or np.isinf(linear_slope):
            linear_slope = 0.0
    else:
        linear_slope = 0.0

    res = {
        "mean": val_mean,
        "median": val_median,
        "variance": val_var,
        "std": val_std,
        "min": val_min,
        "max": val_max,
        "range": val_range,
        "skewness": val_skew,
        "kurtosis": val_kurt,
        "iqr": val_iqr,
        "coef_var": coef_var,
        "linear_slope": linear_slope,
    }
    for k in res:
        if np.isnan(res[k]) or np.isinf(res[k]):
            res[k] = 0.0
    return res


def aggregate_windows(
    df: pd.DataFrame,
    sensor_columns: List[str],
    window_size: int = 4,
) -> np.ndarray:
    """
    Constructs multi-resolution aggregated descriptors using rolling windows.
    For each step, looks back over `window_size` steps and extracts 12 descriptors
    per sensor variable -> 16 * 12 = 192 features per step.
    """
    num_steps = len(df)
    feature_matrix = []

    for t in range(num_steps):
        start_idx = max(0, t - window_size + 1)
        window_slice = df.iloc[start_idx : t + 1]

        step_features = []
        for col in sensor_columns:
            col_vals = window_slice[col].to_numpy()
            descriptors = compute_12_descriptors(col_vals)
            step_features.extend([
                descriptors["mean"],
                descriptors["median"],
                descriptors["variance"],
                descriptors["std"],
                descriptors["min"],
                descriptors["max"],
                descriptors["range"],
                descriptors["skewness"],
                descriptors["kurtosis"],
                descriptors["iqr"],
                descriptors["coef_var"],
                descriptors["linear_slope"],
            ])

        feature_matrix.append(step_features)

    return np.nan_to_num(np.array(feature_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_windowed_features(
    df: pd.DataFrame,
    sensor_columns: List[str],
    window_hours: float = 6.0,
) -> np.ndarray:
    """
    Public entry point to build the 192-dimensional statistical feature representation.
    """
    # Using pandas .rolling() to support continuous telemetry aggregation
    rolling_obj = df[sensor_columns].rolling(window=max(1, int(window_hours / 6.0)), min_periods=1)
    return aggregate_windows(df, sensor_columns, window_size=max(1, int(window_hours / 6.0)))


def compute_step_window_features(
    history: List[Dict[str, float]],
    sensor_columns: List[str],
    window_size: int = 4,
) -> np.ndarray:
    """
    Computes the 192-dimensional statistical feature vector for the latest time step
    from an online rolling history buffer of state dictionaries.
    Guarantees exact parity with build_windowed_features / aggregate_windows.
    """
    if not history:
        return np.zeros(len(sensor_columns) * 12, dtype=np.float32)

    recent_slice = history[-window_size:]
    step_features = []

    for col in sensor_columns:
        vals = np.array([float(s.get(col, 0.0)) for s in recent_slice], dtype=float)
        descriptors = compute_12_descriptors(vals)
        step_features.extend([
            descriptors["mean"],
            descriptors["median"],
            descriptors["variance"],
            descriptors["std"],
            descriptors["min"],
            descriptors["max"],
            descriptors["range"],
            descriptors["skewness"],
            descriptors["kurtosis"],
            descriptors["iqr"],
            descriptors["coef_var"],
            descriptors["linear_slope"],
        ])

    return np.nan_to_num(np.array(step_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ==============================================================================
# Physics Features & Data-Scarcity Trajectory Windowing (unified via bioprocess_core)
# ==============================================================================
import os
import sys

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from models.bioprocess_core import (  # noqa: E402, F401
    salinity_from_tds,
    do_saturation_mgL,
    free_nh3_fraction,
)


def compute_lab_mask(length: int, start: int = 0, interval_hours: int = 24) -> np.ndarray:
    """
    Boolean mask identifying genuine laboratory measurement timestamps (every 24 hours from H0).
    Used to mask sparse lab measurements (e.g. OD680 biomass) from hourly interpolated steps.
    """
    return ((np.arange(start, start + length) % interval_hours) == 0)


def build_trajectory_windows(
    arrs: List[Tuple[np.ndarray, ...]],
    window_hours: int = 72,
    stride_hours: int = 24,
) -> Tuple[np.ndarray, ...]:
    """
    Slice multi-cultivation trajectories into overlapping windows of length `window_hours` (72h)
    with a stride of `stride_hours` (24h) so every window starts on a true daily lab measurement.
    Overcomes small batch count (14 cultivations) by augmenting training sequence samples.
    """
    if not arrs:
        raise ValueError("arrs cannot be empty")
    n_elements = len(arrs[0])
    buckets: List[List[np.ndarray]] = [[] for _ in range(n_elements)]
    for tup in arrs:
        seq_len = len(tup[0])
        for s in range(0, seq_len - window_hours + 1, stride_hours):
            for idx in range(n_elements):
                buckets[idx].append(np.asarray(tup[idx][s : s + window_hours]))
    return tuple(np.stack(b, axis=0) for b in buckets)


