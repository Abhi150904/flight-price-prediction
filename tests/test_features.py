from __future__ import annotations

from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def test_feature_lists_are_unique_and_combined() -> None:
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(FEATURES) == len(set(FEATURES))
    assert "legId" not in FEATURES
    assert "baseFare" not in FEATURES
