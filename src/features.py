from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "totalFare"
GROUP_ID = "legId"
SPLIT_COLUMN = "split"

NUMERIC_FEATURES = [
    "lead_time_days",
    "seatsRemaining",
    "segment_count",
    "travel_duration_minutes",
    "elapsedDays",
    "distance_imputed_by_route_stop",
]

CATEGORICAL_FEATURES = [
    "route",
    "isBasicEconomy",
    "isNonStop",
    "missing_total_travel_distance",
    "first_airline_code",
    "first_cabin_code",
    "departure_period",
    "arrival_period",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
