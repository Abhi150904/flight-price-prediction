from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, TARGET, build_preprocessor
from src.metrics import regression_metrics


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports/tables"


def load_split(split: str, limit: int) -> pd.DataFrame:
    columns = ", ".join(FEATURES + [TARGET])
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {columns}
        from airfare_model_splits
        where split = ?
        order by hash(legId)
        limit ?
        """,
        [split, limit],
    ).fetchdf()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train", 250_000)
    validation_df = load_split("validation", 75_000)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    models = [
        (
            "linear_regression",
            LinearRegression(),
            "Linear model: simple, interpretable, but assumes mostly additive relationships.",
        ),
        (
            "ridge_regression",
            Ridge(alpha=1.0),
            "Regularized linear model: reduces unstable coefficients while staying simple.",
        ),
        (
            "decision_tree_depth_12",
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=42),
            "Single tree: captures nonlinear rules, but can overfit if too deep.",
        ),
        (
            "random_forest_small",
            RandomForestRegressor(
                n_estimators=60,
                max_depth=18,
                min_samples_leaf=30,
                n_jobs=-1,
                random_state=42,
            ),
            "Forest: averages many trees to reduce overfitting, but costs more compute.",
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=42,
            ),
            "Boosting: builds trees sequentially to correct prior errors; often strong on tabular data.",
        ),
    ]

    rows = []
    best_name = None
    best_mae = float("inf")
    best_pipeline = None

    for name, estimator, description in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        train_pred = pipeline.predict(X_train)
        validation_pred = pipeline.predict(X_validation)
        train_metrics = regression_metrics(y_train, train_pred)
        validation_metrics = regression_metrics(y_validation, validation_pred)

        row = {
            "model": name,
            "description": description,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "fit_seconds": round(fit_seconds, 2),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_r2": validation_metrics["r2"],
        }
        rows.append(row)

        print(f"\n{name}")
        print("-" * len(name))
        print(description)
        print(f"Fit seconds: {fit_seconds:.2f}")
        print(f"Train MAE: {train_metrics['mae']:.2f}")
        print(f"Validation MAE: {validation_metrics['mae']:.2f}")
        print(f"Validation RMSE: {validation_metrics['rmse']:.2f}")
        print(f"Validation R2: {validation_metrics['r2']:.3f}")

        if validation_metrics["mae"] < best_mae:
            best_mae = validation_metrics["mae"]
            best_name = name
            best_pipeline = pipeline

    results = pd.DataFrame(rows).sort_values("validation_mae")
    results.to_csv(TABLE_DIR / "phase13_model_results.csv", index=False)

    if best_pipeline is not None:
        model_path = MODEL_DIR / "phase13_best_model.joblib"
        joblib.dump(best_pipeline, model_path)
        print(f"\nBest validation model: {best_name}")
        print(f"Saved model: {model_path}")

    print("\nModel comparison:")
    print(
        results[
            [
                "model",
                "fit_seconds",
                "train_mae",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
