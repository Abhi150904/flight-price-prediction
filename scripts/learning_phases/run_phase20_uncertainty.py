from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, TARGET
from src.uncertainty import GROUP_COLUMNS, apply_residual_intervals, fit_residual_intervals


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_PATH = ROOT / "models/phase13_best_model.joblib"
TABLE_DIR = ROOT / "reports/tables"


def load_rows(split: str, limit: int, offset: int = 0) -> pd.DataFrame:
    columns = FEATURES + [TARGET, "legId"]
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {", ".join(columns)}
        from airfare_model_splits
        where split = ?
        order by hash(legId)
        limit ? offset ?
        """,
        [split, limit, offset],
    ).fetchdf()


def score_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["actual"] = scored[TARGET]
    scored["prediction"] = model.predict(frame[FEATURES])
    return scored


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)

    # The Phase 13 model was trained on the first 250k hash-ordered train rows.
    # Use later train rows for calibration so interval fitting is out-of-sample.
    calibration = score_frame(model, load_rows("train", limit=100_000, offset=250_000))
    validation = score_frame(model, load_rows("validation", limit=169_083))

    intervals, global_interval = fit_residual_intervals(calibration)
    interval_scored = apply_residual_intervals(validation, intervals, global_interval)

    intervals.to_csv(TABLE_DIR / "phase20_residual_intervals_by_group.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "calibration_rows": len(calibration),
                "validation_rows": len(validation),
                "target_interval": "80%",
                "empirical_coverage": interval_scored["covered"].mean(),
                "median_interval_width": interval_scored["interval_width"].median(),
                "p90_interval_width": interval_scored["interval_width"].quantile(0.90),
                "global_fallback_rate": interval_scored["used_global_interval"].mean(),
                "global_lower_residual": global_interval["lower_residual"],
                "global_upper_residual": global_interval["upper_residual"],
            }
        ]
    )
    overall.to_csv(TABLE_DIR / "phase20_uncertainty_overall.csv", index=False)

    by_group = (
        interval_scored.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            row_count=("actual", "size"),
            coverage=("covered", "mean"),
            median_actual=("actual", "median"),
            median_prediction=("prediction", "median"),
            median_interval_width=("interval_width", "median"),
            fallback_rate=("used_global_interval", "mean"),
        )
        .sort_values("row_count", ascending=False)
    )
    by_group.to_csv(TABLE_DIR / "phase20_uncertainty_by_group.csv", index=False)

    examples = interval_scored[
        GROUP_COLUMNS
        + [
            "actual",
            "prediction",
            "predicted_lower",
            "predicted_upper",
            "interval_width",
            "covered",
            "used_global_interval",
        ]
    ].head(100)
    examples.to_csv(TABLE_DIR / "phase20_prediction_interval_examples.csv", index=False)

    print("\nOverall Interval Performance")
    print(overall.to_string(index=False))
    print("\nLargest Groups")
    print(by_group.head(12).to_string(index=False))
    print("\nLowest-Coverage Groups With At Least 1,000 Rows")
    print(by_group[by_group["row_count"] >= 1000].sort_values("coverage").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
