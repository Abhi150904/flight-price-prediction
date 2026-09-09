from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


DB_PATH = Path("data/processed/airfare.duckdb")
TABLE_DIR = Path("reports/tables")


def bootstrap_median_difference(
    nonstop: np.ndarray,
    connecting: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 500,
    max_group_size: int = 50_000,
) -> tuple[float, float, float]:
    if len(nonstop) > max_group_size:
        nonstop = rng.choice(nonstop, size=max_group_size, replace=False)
    if len(connecting) > max_group_size:
        connecting = rng.choice(connecting, size=max_group_size, replace=False)

    observed = float(np.median(nonstop) - np.median(connecting))
    diffs = np.empty(iterations)

    for i in range(iterations):
        nonstop_sample = rng.choice(nonstop, size=len(nonstop), replace=True)
        connecting_sample = rng.choice(connecting, size=len(connecting), replace=True)
        diffs[i] = np.median(nonstop_sample) - np.median(connecting_sample)

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return observed, float(lower), float(upper)


def cohens_d(nonstop: np.ndarray, connecting: np.ndarray) -> float:
    n1 = len(nonstop)
    n2 = len(connecting)
    pooled_sd = np.sqrt(
        ((n1 - 1) * np.var(nonstop, ddof=1) + (n2 - 1) * np.var(connecting, ddof=1))
        / (n1 + n2 - 2)
    )
    return float((np.mean(nonstop) - np.mean(connecting)) / pooled_sd)


def mann_whitney_pvalue(
    nonstop: np.ndarray,
    connecting: np.ndarray,
    rng: np.random.Generator,
    max_group_size: int = 20_000,
) -> float:
    if len(nonstop) > max_group_size:
        nonstop = rng.choice(nonstop, size=max_group_size, replace=False)
    if len(connecting) > max_group_size:
        connecting = rng.choice(connecting, size=max_group_size, replace=False)
    result = stats.mannwhitneyu(nonstop, connecting, alternative="two-sided")
    return float(result.pvalue)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260905)
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)

    summary = con.execute(
        """
        select
            route,
            isNonStop,
            count(*) as row_count,
            round(avg(totalFare), 2) as mean_fare,
            round(median(totalFare), 2) as median_fare,
            round(stddev_samp(totalFare), 2) as sd_fare
        from airfare_clean
        group by route, isNonStop
        order by route, isNonStop
        """
    ).fetchdf()
    summary.to_csv(TABLE_DIR / "phase08_stop_type_summary.csv", index=False)

    coach_summary = con.execute(
        """
        select
            route,
            isNonStop,
            count(*) as row_count,
            round(avg(totalFare), 2) as mean_fare,
            round(median(totalFare), 2) as median_fare,
            round(stddev_samp(totalFare), 2) as sd_fare
        from airfare_clean
        where segmentsCabinCode not like '%first%'
          and segmentsCabinCode not like '%business%'
          and segmentsCabinCode not like '%premium%'
        group by route, isNonStop
        order by route, isNonStop
        """
    ).fetchdf()
    coach_summary.to_csv(TABLE_DIR / "phase08_coach_stop_type_summary.csv", index=False)

    fares = con.execute(
        """
        select
            route,
            isNonStop,
            totalFare,
            case
                when segmentsCabinCode not like '%first%'
                 and segmentsCabinCode not like '%business%'
                 and segmentsCabinCode not like '%premium%'
                then true
                else false
            end as coach_only
        from airfare_clean
        """
    ).fetchdf()

    rows = []
    for label, frame in [("all_fares", fares), ("coach_only", fares[fares["coach_only"]])]:
        for route, route_df in frame.groupby("route"):
            nonstop = route_df.loc[route_df["isNonStop"], "totalFare"].to_numpy()
            connecting = route_df.loc[~route_df["isNonStop"], "totalFare"].to_numpy()

            median_diff, ci_low, ci_high = bootstrap_median_difference(
                nonstop, connecting, rng
            )
            rows.append(
                {
                    "analysis_group": label,
                    "route": route,
                    "nonstop_rows": len(nonstop),
                    "connecting_rows": len(connecting),
                    "median_difference_nonstop_minus_connecting": round(median_diff, 2),
                    "bootstrap_ci_low": round(ci_low, 2),
                    "bootstrap_ci_high": round(ci_high, 2),
                    "cohens_d_mean_difference": round(cohens_d(nonstop, connecting), 4),
                    "mann_whitney_sample_pvalue": mann_whitney_pvalue(
                        nonstop, connecting, rng
                    ),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "phase08_stop_type_stat_tests.csv", index=False)

    print("\nStop-Type Summary")
    print(summary.to_string(index=False))
    print("\nCoach-Only Stop-Type Summary")
    print(coach_summary.to_string(index=False))
    print("\nStatistical Test Results")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
