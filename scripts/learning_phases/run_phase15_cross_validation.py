from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, GROUP_ID, TARGET, build_preprocessor


DB_PATH = ROOT / "data/processed/airfare.duckdb"
TABLE_DIR = ROOT / "reports/tables"


def load_training_sample(limit: int = 180_000) -> pd.DataFrame:
    columns = ", ".join(FEATURES + [TARGET, GROUP_ID])
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {columns}
        from airfare_model_splits
        where split = 'train'
        order by hash(legId)
        limit ?
        """,
        [limit],
    ).fetchdf()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_training_sample()
    X = data[FEATURES]
    y = data[TARGET]
    groups = data[GROUP_ID]

    models = [
        (
            "random_forest_small",
            RandomForestRegressor(
                n_estimators=40,
                max_depth=18,
                min_samples_leaf=30,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=42,
            ),
        ),
    ]

    cv = GroupKFold(n_splits=3)
    rows = []

    for name, estimator in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=True,
        )

        fold_count = len(scores["test_mae"])
        for fold_idx in range(fold_count):
            rows.append(
                {
                    "model": name,
                    "fold": fold_idx + 1,
                    "train_mae": -scores["train_mae"][fold_idx],
                    "validation_mae": -scores["test_mae"][fold_idx],
                    "validation_rmse": -scores["test_rmse"][fold_idx],
                    "validation_r2": scores["test_r2"][fold_idx],
                    "fit_seconds": scores["fit_time"][fold_idx],
                }
            )

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(TABLE_DIR / "phase15_group_cv_fold_results.csv", index=False)

    summary = (
        fold_results.groupby("model")
        .agg(
            mean_validation_mae=("validation_mae", "mean"),
            std_validation_mae=("validation_mae", "std"),
            mean_validation_rmse=("validation_rmse", "mean"),
            mean_validation_r2=("validation_r2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
        .sort_values("mean_validation_mae")
    )
    summary.to_csv(TABLE_DIR / "phase15_group_cv_summary.csv", index=False)

    print("\nGrouped Cross-Validation Fold Results")
    print(fold_results.to_string(index=False))
    print("\nGrouped Cross-Validation Summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
