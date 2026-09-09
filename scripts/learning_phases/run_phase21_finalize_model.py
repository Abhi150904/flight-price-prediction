from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, TARGET, build_preprocessor
from src.metrics import regression_metrics
from src.price_context import apply_price_context, fit_price_context
from src.uncertainty import apply_residual_intervals, fit_residual_intervals


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports/tables"


def load_split(split: str, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    columns = FEATURES + [TARGET, "legId", "split"]
    limit_clause = "" if limit is None else "limit ? offset ?"
    params: list[object] = [split]
    if limit is not None:
        params.extend([limit, offset])

    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {", ".join(columns)}
        from airfare_model_splits
        where split = ?
        order by hash(legId)
        {limit_clause}
        """,
        params,
    ).fetchdf()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    X_modeling = train[FEATURES]
    y_modeling = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=18,
                    min_samples_leaf=30,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_modeling, y_modeling)

    test_predictions = final_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)
    test_scored = test.copy()
    test_scored["actual"] = y_test
    test_scored["prediction"] = test_predictions
    test_scored["absolute_error"] = (test_scored["prediction"] - test_scored["actual"]).abs()

    test_summary = pd.DataFrame(
        [
            {
                "model": "random_forest_final_train_only",
                "train_rows": len(train),
                "validation_rows_reserved_for_calibration": len(validation),
                "test_rows": len(test),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_median_absolute_error": test_scored["absolute_error"].median(),
                "test_p90_absolute_error": test_scored["absolute_error"].quantile(0.90),
                "test_p95_absolute_error": test_scored["absolute_error"].quantile(0.95),
            }
        ]
    )
    test_summary.to_csv(TABLE_DIR / "phase21_final_test_metrics.csv", index=False)

    test_by_route = (
        test_scored.groupby("route", as_index=False)
        .agg(
            row_count=("actual", "size"),
            median_actual=("actual", "median"),
            median_prediction=("prediction", "median"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            p90_absolute_error=("absolute_error", lambda values: values.quantile(0.90)),
        )
        .sort_values("route")
    )
    test_by_route.to_csv(TABLE_DIR / "phase21_final_test_metrics_by_route.csv", index=False)

    context, global_quantiles = fit_price_context(train, min_group_size=100)

    validation_scored = validation.copy()
    validation_scored["actual"] = validation[TARGET]
    validation_scored["prediction"] = final_pipeline.predict(validation[FEATURES])
    calibration = validation_scored
    intervals, global_interval = fit_residual_intervals(calibration)

    model_path = MODEL_DIR / "final_price_model.joblib"
    artifact_path = MODEL_DIR / "final_price_artifacts.joblib"
    joblib.dump(final_pipeline, model_path)
    joblib.dump(
        {
            "model": final_pipeline,
            "features": FEATURES,
            "target": TARGET,
            "price_context": context,
            "price_context_global_quantiles": global_quantiles,
            "residual_intervals": intervals,
            "residual_global_interval": global_interval,
            "model_notes": {
                "selected_model": "RandomForestRegressor",
                "selection_basis": "Best first-pass validation MAE and strong grouped-CV performance.",
                "excluded_features": ["legId", "baseFare", "fareBasisCode", "raw searchDate", "raw flightDate"],
            },
        },
        artifact_path,
    )

    reloaded = joblib.load(model_path)
    sample = X_test.head(5)
    original_prediction = final_pipeline.predict(sample)
    reloaded_prediction = reloaded.predict(sample)
    same_predictions = bool(np.allclose(original_prediction, reloaded_prediction))

    context_examples = apply_price_context(test.head(100), context, global_quantiles)
    interval_examples = test.head(100).copy()
    interval_examples["actual"] = y_test.head(100)
    interval_examples["prediction"] = test_predictions[:100]
    interval_examples = apply_residual_intervals(interval_examples, intervals, global_interval)
    context_examples.to_csv(TABLE_DIR / "phase21_price_context_examples.csv", index=False)
    interval_examples.to_csv(TABLE_DIR / "phase21_prediction_interval_examples.csv", index=False)

    print("\nFinal Test Metrics")
    print(test_summary.to_string(index=False))
    print("\nFinal Test Metrics By Route")
    print(test_by_route.to_string(index=False))
    print("\nPersistence Check")
    print(f"Saved model: {model_path}")
    print(f"Saved artifact bundle: {artifact_path}")
    print(f"Reloaded predictions match original: {same_predictions}")
    print("\nSample Predictions")
    for actual, predicted in zip(y_test.head(5), original_prediction):
        print(f"actual={actual:.2f}, predicted={predicted:.2f}")


if __name__ == "__main__":
    main()
