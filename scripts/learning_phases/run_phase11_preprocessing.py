from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_DIR = ROOT / "models"
PREPROCESSOR_PATH = MODEL_DIR / "phase11_preprocessor.joblib"


def load_split_sample(split: str, limit: int):
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

    train_df = load_split_sample("train", 100_000)
    validation_df = load_split_sample("validation", 25_000)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_validation = validation_df[FEATURES]

    preprocessor = build_preprocessor()
    X_train_prepared = preprocessor.fit_transform(X_train, y_train)
    X_validation_prepared = preprocessor.transform(X_validation)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    feature_names = preprocessor.get_feature_names_out()

    print("Numeric features:")
    for feature in NUMERIC_FEATURES:
        print(f"- {feature}")

    print("\nCategorical features:")
    for feature in CATEGORICAL_FEATURES:
        print(f"- {feature}")

    print("\nPreprocessing results:")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Validation rows: {X_validation.shape[0]}")
    print(f"Raw feature columns: {X_train.shape[1]}")
    print(f"Prepared training shape: {X_train_prepared.shape}")
    print(f"Prepared validation shape: {X_validation_prepared.shape}")
    print(f"Expanded feature count: {len(feature_names)}")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")

    print("\nFirst 25 expanded feature names:")
    for name in feature_names[:25]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
