from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.price_context import CONTEXT_GROUPS, apply_price_context, fit_price_context


DB_PATH = ROOT / "data/processed/airfare.duckdb"
TABLE_DIR = ROOT / "reports/tables"


def load_context_data() -> pd.DataFrame:
    columns = CONTEXT_GROUPS + ["split", "totalFare", "legId", "lead_time_days"]
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {", ".join(columns)}
        from airfare_model_splits
        where split in ('train', 'validation')
        """
    ).fetchdf()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_context_data()
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()

    context, global_quantiles = fit_price_context(train, min_group_size=100)
    scored = apply_price_context(validation, context, global_quantiles)

    context.to_csv(TABLE_DIR / "phase18_price_context_thresholds.csv", index=False)

    category_summary = (
        scored.groupby("price_category", as_index=False)
        .agg(
            row_count=("totalFare", "size"),
            median_actual_fare=("totalFare", "median"),
            avg_actual_fare=("totalFare", "mean"),
            median_typical_fare=("typical_fare", "median"),
            fallback_rate=("used_global_context", "mean"),
        )
        .sort_values("median_actual_fare")
    )
    category_summary.to_csv(TABLE_DIR / "phase18_price_category_summary.csv", index=False)

    route_category_summary = (
        scored.groupby(["route", "price_category"], as_index=False)
        .agg(
            row_count=("totalFare", "size"),
            median_actual_fare=("totalFare", "median"),
            median_low_threshold=("low_threshold", "median"),
            median_high_threshold=("high_threshold", "median"),
            fallback_rate=("used_global_context", "mean"),
        )
        .sort_values(["route", "median_actual_fare"])
    )
    route_category_summary.to_csv(
        TABLE_DIR / "phase18_route_price_category_summary.csv", index=False
    )

    examples = scored[
        CONTEXT_GROUPS
        + [
            "totalFare",
            "low_threshold",
            "typical_fare",
            "high_threshold",
            "price_category",
            "comparable_count",
            "used_global_context",
        ]
    ].sort_values(["route", "price_category", "totalFare"])
    examples.head(100).to_csv(TABLE_DIR / "phase18_price_context_examples.csv", index=False)

    coverage = pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "context_groups_kept": len(context),
                "validation_global_fallback_rate": scored["used_global_context"].mean(),
                "global_p20": global_quantiles.loc[0.2],
                "global_median": global_quantiles.loc[0.5],
                "global_p80": global_quantiles.loc[0.8],
            }
        ]
    )
    coverage.to_csv(TABLE_DIR / "phase18_context_coverage.csv", index=False)

    print("\nContext Coverage")
    print(coverage.to_string(index=False))
    print("\nPrice Category Summary")
    print(category_summary.to_string(index=False))
    print("\nRoute Price Category Summary")
    print(route_category_summary.to_string(index=False))


if __name__ == "__main__":
    main()
