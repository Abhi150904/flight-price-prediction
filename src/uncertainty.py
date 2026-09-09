from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["route", "first_cabin_code", "isBasicEconomy"]


def fit_residual_intervals(
    calibration: pd.DataFrame,
    group_cols: list[str] | None = None,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    min_group_size: int = 200,
) -> tuple[pd.DataFrame, dict[str, float]]:
    group_cols = group_cols or GROUP_COLUMNS
    calibration = calibration.copy()
    calibration["residual"] = calibration["actual"] - calibration["prediction"]

    global_interval = {
        "lower_residual": float(calibration["residual"].quantile(lower_quantile)),
        "upper_residual": float(calibration["residual"].quantile(upper_quantile)),
    }

    grouped = (
        calibration.groupby(group_cols, dropna=False)["residual"]
        .agg(
            calibration_rows="count",
            lower_residual=lambda values: values.quantile(lower_quantile),
            upper_residual=lambda values: values.quantile(upper_quantile),
        )
        .reset_index()
    )
    grouped = grouped[grouped["calibration_rows"] >= min_group_size].copy()
    return grouped, global_interval


def apply_residual_intervals(
    scored: pd.DataFrame,
    intervals: pd.DataFrame,
    global_interval: dict[str, float],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or GROUP_COLUMNS
    out = scored.merge(intervals, on=group_cols, how="left")
    out["used_global_interval"] = out["lower_residual"].isna()
    out["lower_residual"] = out["lower_residual"].fillna(global_interval["lower_residual"])
    out["upper_residual"] = out["upper_residual"].fillna(global_interval["upper_residual"])
    out["calibration_rows"] = out["calibration_rows"].fillna(0).astype(int)
    out["predicted_lower"] = np.maximum(0, out["prediction"] + out["lower_residual"])
    out["predicted_upper"] = np.maximum(0, out["prediction"] + out["upper_residual"])
    out["interval_width"] = out["predicted_upper"] - out["predicted_lower"]
    out["covered"] = (out["actual"] >= out["predicted_lower"]) & (
        out["actual"] <= out["predicted_upper"]
    )
    return out
