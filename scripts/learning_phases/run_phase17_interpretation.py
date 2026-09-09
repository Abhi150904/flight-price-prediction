from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES, TARGET


DB_PATH = ROOT / "data/processed/airfare.duckdb"
MODEL_PATH = ROOT / "models/phase13_best_model.joblib"
TABLE_DIR = ROOT / "reports/tables"


def load_validation_sample(limit: int = 30_000) -> pd.DataFrame:
    columns = ", ".join(FEATURES + [TARGET, "legId"])
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    return con.execute(
        f"""
        select {columns}
        from airfare_model_splits
        where split = 'validation'
        order by hash(legId)
        limit ?
        """,
        [limit],
    ).fetchdf()


def aggregate_encoded_importance(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    encoded_names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()

    rows = []
    for encoded_name, importance in zip(encoded_names, importances):
        if encoded_name.startswith("numeric__"):
            original_feature = encoded_name.replace("numeric__", "")
        elif encoded_name.startswith("categorical__"):
            encoded_without_prefix = encoded_name.replace("categorical__", "")
            original_feature = next(
                feature
                for feature in FEATURES
                if encoded_without_prefix == feature
                or encoded_without_prefix.startswith(f"{feature}_")
            )
        else:
            original_feature = encoded_name

        rows.append(
            {
                "encoded_feature": encoded_name,
                "original_feature": original_feature,
                "importance": importance,
            }
        )

    encoded = pd.DataFrame(rows).sort_values("importance", ascending=False)
    grouped = (
        encoded.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    return encoded, grouped


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    validation = load_validation_sample()
    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    encoded_importance, grouped_importance = aggregate_encoded_importance(model)
    encoded_importance.to_csv(TABLE_DIR / "phase17_encoded_feature_importance.csv", index=False)
    grouped_importance.to_csv(TABLE_DIR / "phase17_grouped_feature_importance.csv", index=False)

    permutation = permutation_importance(
        model,
        X_validation,
        y_validation,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "mae_increase_mean": permutation.importances_mean,
            "mae_increase_std": permutation.importances_std,
        }
    ).sort_values("mae_increase_mean", ascending=False)
    permutation_df.to_csv(TABLE_DIR / "phase17_permutation_importance.csv", index=False)

    print("\nGrouped Built-In Feature Importance")
    print(grouped_importance.head(15).to_string(index=False))

    print("\nPermutation Importance")
    print(permutation_df.to_string(index=False))

    print("\nTop Encoded Feature Importances")
    print(encoded_importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
