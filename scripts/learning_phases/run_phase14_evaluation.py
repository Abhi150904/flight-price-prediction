from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, TARGET
from src.metrics import regression_metrics


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_PATH = ROOT / "models/phase13_best_model.joblib"
TABLE_DIR = ROOT / "reports/tables"


SEGMENT_COLUMNS = [
    "route",
    "first_airline_code",
    "first_cabin_code",
    "isBasicEconomy",
    "isNonStop",
    "departure_period",
]


def load_validation_data(limit: int | None = None) -> pd.DataFrame:
    columns = FEATURES + [TARGET, "legId", "split"]
    select_columns = ", ".join(columns)
    limit_clause = "" if limit is None else "limit ?"
    params: list[object] = ["validation"]
    if limit is not None:
        params.append(limit)

    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {select_columns}
        from airfare_model_splits
        where split = ?
        order by hash(legId)
        {limit_clause}
        """,
        params,
    ).fetchdf()


def summarize_group(scored: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in scored.groupby(group_col, dropna=False):
        if len(group) < 100:
            continue
        metrics = regression_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                "segment": group_col,
                "value": value,
                "row_count": len(group),
                "actual_median": group["actual"].median(),
                "prediction_median": group["prediction"].median(),
                "bias_mean_error": group["error"].mean(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    validation = load_validation_data()

    predictions = model.predict(validation[FEATURES])
    scored = validation.copy()
    scored["actual"] = scored[TARGET]
    scored["prediction"] = predictions
    scored["error"] = scored["prediction"] - scored["actual"]
    scored["absolute_error"] = scored["error"].abs()
    scored["fare_range"] = pd.cut(
        scored["actual"],
        bins=[0, 200, 400, 700, 1200, np.inf],
        labels=["$0-200", "$200-400", "$400-700", "$700-1200", "$1200+"],
        right=False,
    )

    overall = regression_metrics(scored["actual"], scored["prediction"])
    overall_df = pd.DataFrame(
        [
            {
                "rows": len(scored),
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "r2": overall["r2"],
                "mean_error_bias": scored["error"].mean(),
                "median_absolute_error": scored["absolute_error"].median(),
                "p90_absolute_error": scored["absolute_error"].quantile(0.90),
                "p95_absolute_error": scored["absolute_error"].quantile(0.95),
            }
        ]
    )
    overall_df.to_csv(TABLE_DIR / "phase14_overall_validation_metrics.csv", index=False)

    segment_summaries = [summarize_group(scored, col) for col in SEGMENT_COLUMNS]
    segment_summaries.append(summarize_group(scored, "fare_range"))
    by_segment = pd.concat(segment_summaries, ignore_index=True)
    by_segment = by_segment.sort_values(["segment", "mae"], ascending=[True, False])
    by_segment.to_csv(TABLE_DIR / "phase14_segment_error_summary.csv", index=False)

    large_errors = scored.sort_values("absolute_error", ascending=False).head(50)
    large_errors[
        [
            "legId",
            "route",
            "actual",
            "prediction",
            "error",
            "absolute_error",
            "lead_time_days",
            "first_airline_code",
            "first_cabin_code",
            "isBasicEconomy",
            "isNonStop",
            "travel_duration_minutes",
            "seatsRemaining",
            "departure_period",
        ]
    ].to_csv(TABLE_DIR / "phase14_largest_validation_errors.csv", index=False)

    print("\nOverall Validation Metrics")
    print(overall_df.to_string(index=False))

    print("\nError By Fare Range")
    fare_range = by_segment[by_segment["segment"] == "fare_range"].sort_values("value")
    print(fare_range.to_string(index=False))

    print("\nWorst Segment Errors By MAE")
    print(by_segment.sort_values("mae", ascending=False).head(15).to_string(index=False))

    print("\nLargest Individual Errors")
    print(
        large_errors[
            [
                "route",
                "actual",
                "prediction",
                "absolute_error",
                "lead_time_days",
                "first_airline_code",
                "first_cabin_code",
                "isBasicEconomy",
                "isNonStop",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
