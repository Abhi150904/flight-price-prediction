from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/processed/airfare.duckdb")


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix())

    con.execute(
        """
        create or replace table airfare_book_wait_labels as
        with daily_prices as (
            select
                split,
                legId,
                searchDate,
                flightDate,
                min(totalFare) as current_fare
            from airfare_model_splits
            group by split, legId, searchDate, flightDate
        ),
        future_prices as (
            select
                current.split,
                current.legId,
                current.searchDate,
                current.flightDate,
                current.current_fare,
                min(future.current_fare) as min_future_fare_7d,
                count(*) as future_observation_count
            from daily_prices as current
            inner join daily_prices as future
                on current.legId = future.legId
               and future.searchDate > current.searchDate
               and future.searchDate <= current.searchDate + interval 7 day
            group by
                current.split,
                current.legId,
                current.searchDate,
                current.flightDate,
                current.current_fare
        )
        select
            splits.split,
            f.*,
            labels.current_fare,
            labels.min_future_fare_7d,
            labels.current_fare - labels.min_future_fare_7d as max_savings_if_wait_7d,
            labels.future_observation_count,
            case
                when labels.current_fare - labels.min_future_fare_7d >= 10 then 1
                else 0
            end as wait_saved_10usd_7d,
            case
                when labels.current_fare - labels.min_future_fare_7d >= 10 then 'WAIT'
                else 'BOOK'
            end as historical_best_action_7d
        from airfare_features as f
        inner join airfare_model_splits as splits
            on f.legId = splits.legId
           and f.searchDate = splits.searchDate
           and f.flightDate = splits.flightDate
           and f.totalFare = splits.totalFare
        inner join future_prices as labels
            on f.legId = labels.legId
           and f.searchDate = labels.searchDate
           and f.flightDate = labels.flightDate
           and f.totalFare = labels.current_fare
        """
    )

    summary = con.execute(
        """
        select
            split,
            count(*) as labeled_rows,
            count(distinct legId) as labeled_leg_ids,
            round(avg(wait_saved_10usd_7d), 4) as wait_rate,
            round(avg(max_savings_if_wait_7d), 2) as avg_max_savings_if_wait_7d,
            round(median(max_savings_if_wait_7d), 2) as median_max_savings_if_wait_7d
        from airfare_book_wait_labels
        group by split
        order by
            case split
                when 'train' then 1
                when 'validation' then 2
                else 3
            end
        """
    ).fetchdf()

    print("Created table: airfare_book_wait_labels")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
