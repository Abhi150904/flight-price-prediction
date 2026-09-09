from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/processed/airfare.duckdb")


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix())

    con.execute(
        """
        create or replace table airfare_model_splits as
        with leg_assignment as (
            select
                legId,
                hash(legId) % 100 as split_bucket
            from airfare_features
            group by legId
        )
        select
            f.*,
            case
                when a.split_bucket < 70 then 'train'
                when a.split_bucket < 85 then 'validation'
                else 'test'
            end as split
        from airfare_features as f
        inner join leg_assignment as a
            on f.legId = a.legId
        """
    )

    split_summary = con.execute(
        """
        select
            split,
            count(*) as row_count,
            count(distinct legId) as leg_ids,
            round(avg(totalFare), 2) as avg_total_fare,
            round(median(totalFare), 2) as median_total_fare
        from airfare_model_splits
        group by split
        order by
            case split
                when 'train' then 1
                when 'validation' then 2
                else 3
            end
        """
    ).fetchdf()

    route_summary = con.execute(
        """
        select
            route,
            split,
            count(*) as row_count,
            count(distinct legId) as leg_ids,
            round(median(totalFare), 2) as median_total_fare
        from airfare_model_splits
        group by route, split
        order by route,
            case split
                when 'train' then 1
                when 'validation' then 2
                else 3
            end
        """
    ).fetchdf()

    leakage_check = con.execute(
        """
        with split_counts as (
            select
                legId,
                count(distinct split) as split_count
            from airfare_model_splits
            group by legId
        )
        select
            count(*) as leg_ids,
            sum(case when split_count > 1 then 1 else 0 end) as leg_ids_spanning_multiple_splits
        from split_counts
        """
    ).fetchdf()

    print("\nSplit Summary")
    print(split_summary.to_string(index=False))
    print("\nRoute Split Summary")
    print(route_summary.to_string(index=False))
    print("\nLeakage Check")
    print(leakage_check.to_string(index=False))


if __name__ == "__main__":
    main()
