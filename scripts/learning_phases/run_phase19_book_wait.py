from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, GROUP_ID, build_preprocessor
from src.price_context import apply_price_context, fit_price_context


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports/tables"

TARGET = "wait_saved_10usd_7d"


def load_labeled_split(split: str, limit: int | None = None) -> pd.DataFrame:
    columns = FEATURES + [
        TARGET,
        GROUP_ID,
        "totalFare",
        "current_fare",
        "min_future_fare_7d",
        "max_savings_if_wait_7d",
        "future_observation_count",
    ]
    limit_clause = "" if limit is None else "limit ?"
    params: list[object] = [split]
    if limit is not None:
        params.append(limit)

    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {", ".join(columns)}
        from airfare_book_wait_labels
        where split = ?
        order by hash(legId)
        {limit_clause}
        """,
        params,
    ).fetchdf()


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_wait": precision_score(y_true, y_pred, zero_division=0),
        "recall_wait": recall_score(y_true, y_pred, zero_division=0),
        "f1_wait": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {key: float(value) for key, value in metrics.items()}


def evaluate_rule(name: str, y_true, y_pred) -> dict[str, object]:
    return {"model": name, **classification_metrics(y_true, y_pred)}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_labeled_split("train", 250_000)
    validation = load_labeled_split("validation", 75_000)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    context, global_quantiles = fit_price_context(train, min_group_size=100)
    validation_context = apply_price_context(validation, context, global_quantiles)

    rows = []
    rows.append(evaluate_rule("always_book", y_validation, [0] * len(y_validation)))
    rows.append(evaluate_rule("always_wait", y_validation, [1] * len(y_validation)))
    rows.append(
        evaluate_rule(
            "wait_when_price_context_high",
            y_validation,
            (validation_context["price_category"] == "HIGH").astype(int),
        )
    )

    models = [
        (
            "book_wait_random_forest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=16,
                min_samples_leaf=40,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "book_wait_hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=42,
            ),
        ),
    ]

    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, estimator in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_validation)
        if hasattr(pipeline, "predict_proba"):
            probability = pipeline.predict_proba(X_validation)[:, 1]
        else:
            probability = None

        metrics = classification_metrics(y_validation, prediction, probability)
        rows.append({"model": name, **metrics})

        if metrics["f1_wait"] > best_f1:
            best_f1 = metrics["f1_wait"]
            best_name = name
            best_model = pipeline

    results = pd.DataFrame(rows).sort_values("f1_wait", ascending=False)
    results.to_csv(TABLE_DIR / "phase19_book_wait_validation_results.csv", index=False)

    label_summary = pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_wait_rate": y_train.mean(),
                "validation_wait_rate": y_validation.mean(),
                "wait_definition": "WAIT if the same legId has a fare at least $10 cheaper within the next 7 search days.",
            }
        ]
    )
    label_summary.to_csv(TABLE_DIR / "phase19_book_wait_label_summary.csv", index=False)

    if best_model is not None:
        model_path = MODEL_DIR / "phase19_book_wait_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"\nBest BOOK/WAIT model: {best_name}")
        print(f"Saved model: {model_path}")

    print("\nLabel Summary")
    print(label_summary.to_string(index=False))
    print("\nValidation Results")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
