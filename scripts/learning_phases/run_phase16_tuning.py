from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, GROUP_ID, TARGET, build_preprocessor
from src.metrics import regression_metrics


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports/tables"


def load_split(split: str, limit: int) -> pd.DataFrame:
    columns = ", ".join(FEATURES + [TARGET, GROUP_ID])
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


def tune_model(name: str, estimator, param_distributions: dict, X, y, groups):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=4,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=2),
        random_state=42,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X, y, groups=groups)
    results = pd.DataFrame(search.cv_results_)
    results.insert(0, "model_family", name)
    return search, results


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train", 90_000)
    validation = load_split("validation", 50_000)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    groups = train[GROUP_ID]
    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    searches = []

    rf_search, rf_results = tune_model(
        "random_forest",
        RandomForestRegressor(n_jobs=-1, random_state=42),
        {
            "model__n_estimators": [30, 50],
            "model__max_depth": [12, 16, 20],
            "model__min_samples_leaf": [20, 50, 100],
            "model__max_features": [0.7, 1.0],
        },
        X_train,
        y_train,
        groups,
    )
    searches.append(("random_forest", rf_search, rf_results))

    hgb_search, hgb_results = tune_model(
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(random_state=42),
        {
            "model__max_iter": [80, 120, 160],
            "model__learning_rate": [0.05, 0.08, 0.12],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.05, 0.2],
            "model__min_samples_leaf": [30, 100],
        },
        X_train,
        y_train,
        groups,
    )
    searches.append(("hist_gradient_boosting", hgb_search, hgb_results))

    all_cv_results = pd.concat([item[2] for item in searches], ignore_index=True)
    all_cv_results.to_csv(TABLE_DIR / "phase16_randomized_search_cv_results.csv", index=False)

    comparison_rows = []
    best_model_name = None
    best_model = None
    best_validation_mae = float("inf")

    for name, search, _ in searches:
        pred = search.best_estimator_.predict(X_validation)
        metrics = regression_metrics(y_validation, pred)
        comparison_rows.append(
            {
                "model_family": name,
                "best_cv_mae": -search.best_score_,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
                "best_params": search.best_params_,
            }
        )
        if metrics["mae"] < best_validation_mae:
            best_validation_mae = metrics["mae"]
            best_model_name = name
            best_model = search.best_estimator_

    comparison = pd.DataFrame(comparison_rows).sort_values("validation_mae")
    comparison.to_csv(TABLE_DIR / "phase16_tuned_model_comparison.csv", index=False)

    if best_model is not None:
        model_path = MODEL_DIR / "phase16_best_tuned_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"\nBest tuned model: {best_model_name}")
        print(f"Saved model: {model_path}")

    print("\nTuned Model Comparison")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
