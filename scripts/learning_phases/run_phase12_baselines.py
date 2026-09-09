from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import regression_metrics


DB_PATH = ROOT / "data/processed/airfare.duckdb"
TABLE_DIR = ROOT / "reports/tables"


def load_model_data() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        """
        select
            split,
            totalFare,
            route,
            isBasicEconomy,
            first_cabin_code,
            first_airline_code,
            isNonStop,
            departure_period,
            lead_time_days
        from airfare_model_splits
        """
    ).fetchdf()


def apply_group_median_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group_cols: list[str],
    global_median: float,
) -> pd.Series:
    medians = (
        train.groupby(group_cols, dropna=False)["totalFare"]
        .median()
        .rename("prediction")
        .reset_index()
    )
    scored = validation.merge(medians, on=group_cols, how="left")
    return scored["prediction"].fillna(global_median)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_model_data()
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()

    y_validation = validation["totalFare"]
    global_median = float(train["totalFare"].median())

    baselines: list[dict[str, object]] = []

    baseline_specs = [
        ("global_median", []),
        ("route_median", ["route"]),
        ("route_basic_economy_median", ["route", "isBasicEconomy"]),
        ("route_cabin_median", ["route", "first_cabin_code"]),
        ("route_airline_cabin_stop_median", ["route", "first_airline_code", "first_cabin_code", "isNonStop"]),
        (
            "route_airline_cabin_stop_departure_median",
            ["route", "first_airline_code", "first_cabin_code", "isNonStop", "departure_period"],
        ),
    ]

    for name, group_cols in baseline_specs:
        if group_cols:
            prediction = apply_group_median_baseline(
                train, validation, group_cols, global_median
            )
            fallback_rate = float(prediction.eq(global_median).mean())
        else:
            prediction = pd.Series(global_median, index=validation.index)
            fallback_rate = 0.0

        metrics = regression_metrics(y_validation, prediction)
        baselines.append(
            {
                "baseline": name,
                "grouping": ", ".join(group_cols) if group_cols else "none",
                "validation_rows": len(validation),
                "fallback_rate": fallback_rate,
                **metrics,
            }
        )

    result = pd.DataFrame(baselines).sort_values("mae")
    result.to_csv(TABLE_DIR / "phase12_baseline_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
