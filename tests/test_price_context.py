from __future__ import annotations

import pandas as pd

from src.price_context import apply_price_context, fit_price_context


def test_price_context_assigns_low_typical_high() -> None:
    train = pd.DataFrame(
        {
            "route": ["ATL-BOS"] * 5,
            "first_airline_code": ["DL"] * 5,
            "first_cabin_code": ["coach"] * 5,
            "isNonStop": [True] * 5,
            "isBasicEconomy": [False] * 5,
            "totalFare": [100, 200, 300, 400, 500],
        }
    )
    context, global_quantiles = fit_price_context(train, min_group_size=1)

    scored = apply_price_context(
        pd.DataFrame(
            {
                "route": ["ATL-BOS", "ATL-BOS", "ATL-BOS"],
                "first_airline_code": ["DL", "DL", "DL"],
                "first_cabin_code": ["coach", "coach", "coach"],
                "isNonStop": [True, True, True],
                "isBasicEconomy": [False, False, False],
                "totalFare": [150, 300, 450],
            }
        ),
        context,
        global_quantiles,
    )

    assert scored["price_category"].tolist() == ["LOW", "TYPICAL", "HIGH"]
