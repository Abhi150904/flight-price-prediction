from __future__ import annotations

import pandas as pd


CONTEXT_GROUPS = ["route", "first_airline_code", "first_cabin_code", "isNonStop", "isBasicEconomy"]


def fit_price_context(
    train: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_group_size: int = 100,
) -> tuple[pd.DataFrame, pd.Series]:
    group_cols = group_cols or CONTEXT_GROUPS
    global_quantiles = train["totalFare"].quantile([0.2, 0.5, 0.8])

    context = (
        train.groupby(group_cols, dropna=False)["totalFare"]
        .agg(
            comparable_count="count",
            low_threshold=lambda values: values.quantile(0.2),
            typical_fare="median",
            high_threshold=lambda values: values.quantile(0.8),
        )
        .reset_index()
    )
    context = context[context["comparable_count"] >= min_group_size].copy()
    return context, global_quantiles


def apply_price_context(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    global_quantiles: pd.Series,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or CONTEXT_GROUPS
    scored = frame.merge(context, on=group_cols, how="left")

    scored["used_global_context"] = scored["typical_fare"].isna()
    scored["low_threshold"] = scored["low_threshold"].fillna(global_quantiles.loc[0.2])
    scored["typical_fare"] = scored["typical_fare"].fillna(global_quantiles.loc[0.5])
    scored["high_threshold"] = scored["high_threshold"].fillna(global_quantiles.loc[0.8])
    scored["comparable_count"] = scored["comparable_count"].fillna(0).astype(int)

    scored["price_category"] = "TYPICAL"
    scored.loc[scored["totalFare"] <= scored["low_threshold"], "price_category"] = "LOW"
    scored.loc[scored["totalFare"] >= scored["high_threshold"], "price_category"] = "HIGH"
    return scored
