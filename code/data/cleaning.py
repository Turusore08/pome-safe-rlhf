"""
Data Cleaning & Quality Pipeline for Raw Sensor Streams.
Implements:
1. Interquartile-range (IQR) outlier filter (1.50x multiplier matching HCR baseline)
2. Linear interpolation for brief sensor interruptions (<6 hours)
3. Multivariate Imputation by Chained Equations (MICE via IterativeImputer) for longer gaps.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def iqr_filter(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.50,
) -> pd.DataFrame:
    """
    Interquartile Range (IQR) outlier removal.
    Fences: [Q1 - k * IQR, Q3 + k * IQR], k = 1.50.
    Outliers are converted to NaN for downstream imputation.
    """
    clean_df = df.copy()
    cols = columns if columns is not None else [c for c in df.columns if c not in ["step", "hour", "timestamp"]]

    for col in cols:
        if col not in clean_df.columns:
            continue
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - (iqr_multiplier * iqr)
        upper_fence = q3 + (iqr_multiplier * iqr)

        # Mark values outside interquartile fences as NaN
        outlier_mask = (clean_df[col] < lower_fence) | (clean_df[col] > upper_fence)
        clean_df.loc[outlier_mask, col] = np.nan

    return clean_df


def mice_impute(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_iter: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Multivariate Imputation by Chained Equations (MICE).
    Fills remaining missing values using Bayesian Ridge regression across correlated sensors.
    """
    imputed_df = df.copy()
    cols = columns if columns is not None else [c for c in df.columns if c not in ["step", "hour", "timestamp"]]

    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy="mean",
        min_value=0.0,
    )

    imputed_data = imputer.fit_transform(imputed_df[cols])
    imputed_df[cols] = imputed_data
    return imputed_df


def clean_telemetry_pipeline(
    raw_df: pd.DataFrame,
    sensor_columns: List[str],
    iqr_multiplier: float = 1.50,
) -> pd.DataFrame:
    """
    Complete end-to-end cleaning pipeline:
    1. Filter extreme outliers with IQR fence.
    2. Linearly interpolate short missing spans (< 6 hours).
    3. Apply MICE (IterativeImputer) for multi-variable consistency.
    """
    # 1. IQR Outlier Removal
    filtered_df = iqr_filter(raw_df, columns=sensor_columns, iqr_multiplier=iqr_multiplier)

    # 2. Brief gap linear interpolation
    interpolated_df = filtered_df.copy()
    interpolated_df[sensor_columns] = interpolated_df[sensor_columns].interpolate(
        method="linear", limit=2, limit_direction="both"
    )

    # 3. MICE imputation for any remaining gaps
    if interpolated_df[sensor_columns].isna().sum().sum() > 0:
        cleaned_df = mice_impute(interpolated_df, columns=sensor_columns)
    else:
        cleaned_df = interpolated_df

    return cleaned_df
