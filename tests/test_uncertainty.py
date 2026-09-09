from __future__ import annotations

import pandas as pd

from src.uncertainty import apply_residual_intervals, fit_residual_intervals


def test_residual_intervals_cover_expected_range() -> None:
    calibration = pd.DataFrame(
        {
            "route": ["ATL-BOS"] * 5,
            "first_cabin_code": ["coach"] * 5,
            "isBasicEconomy": [False] * 5,
            "actual": [90, 100, 110, 120, 130],
            "prediction": [100, 100, 100, 100, 100],
        }
    )
    intervals, global_interval = fit_residual_intervals(calibration, min_group_size=1)

    scored = apply_residual_intervals(
        pd.DataFrame(
            {
                "route": ["ATL-BOS"],
                "first_cabin_code": ["coach"],
                "isBasicEconomy": [False],
                "actual": [110],
                "prediction": [100],
            }
        ),
        intervals,
        global_interval,
    )

    assert scored.loc[0, "predicted_lower"] < 100
    assert scored.loc[0, "predicted_upper"] > 100
    assert bool(scored.loc[0, "covered"])
