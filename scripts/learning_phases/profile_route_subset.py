from __future__ import annotations

from pathlib import Path

import duckdb


SUBSET = Path("data/interim/expedia_route_subset.parquet")


def print_df(title: str, df) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(df.to_string(index=False))


def main() -> None:
    path = SUBSET.resolve().as_posix()
    con = duckdb.connect()

    overview = con.execute(
        """
        select
            count(*) as row_count,
            count(distinct route) as routes,
            count(distinct legId) as leg_ids,
            min(searchDate) as min_search_date,
            max(searchDate) as max_search_date,
            min(flightDate) as min_flight_date,
            max(flightDate) as max_flight_date,
            min(lead_time_days) as min_lead_time_days,
            max(lead_time_days) as max_lead_time_days
        from read_parquet(?)
        """,
        [path],
    ).fetchdf()
    print_df("Dataset Overview", overview)

    route_summary = con.execute(
        """
        select
            route,
            count(*) as row_count,
            count(distinct legId) as leg_ids,
            count(distinct searchDate) as search_dates,
            count(distinct flightDate) as flight_dates,
            count(distinct segmentsAirlineCode) as airline_patterns,
            count(distinct segmentsCabinCode) as cabin_patterns,
            round(avg(totalFare), 2) as avg_total_fare,
            round(median(totalFare), 2) as median_total_fare
        from read_parquet(?)
        group by route
        order by route
        """,
        [path],
    ).fetchdf()
    print_df("Route Summary", route_summary)

    null_summary = con.execute(
        """
        select *
        from (
            unpivot (
                select
                    count(*) as row_count,
                    sum(case when legId is null then 1 else 0 end) as legId,
                    sum(case when searchDate is null then 1 else 0 end) as searchDate,
                    sum(case when flightDate is null then 1 else 0 end) as flightDate,
                    sum(case when lead_time_days is null then 1 else 0 end) as lead_time_days,
                    sum(case when route is null then 1 else 0 end) as route,
                    sum(case when fareBasisCode is null then 1 else 0 end) as fareBasisCode,
                    sum(case when travelDuration is null then 1 else 0 end) as travelDuration,
                    sum(case when elapsedDays is null then 1 else 0 end) as elapsedDays,
                    sum(case when isBasicEconomy is null then 1 else 0 end) as isBasicEconomy,
                    sum(case when isRefundable is null then 1 else 0 end) as isRefundable,
                    sum(case when isNonStop is null then 1 else 0 end) as isNonStop,
                    sum(case when baseFare is null then 1 else 0 end) as baseFare,
                    sum(case when totalFare is null then 1 else 0 end) as totalFare,
                    sum(case when seatsRemaining is null then 1 else 0 end) as seatsRemaining,
                    sum(case when totalTravelDistance is null then 1 else 0 end) as totalTravelDistance,
                    sum(case when segmentsAirlineCode is null then 1 else 0 end) as segmentsAirlineCode,
                    sum(case when segmentsCabinCode is null then 1 else 0 end) as segmentsCabinCode
                from read_parquet(?)
            )
            on columns(* exclude (row_count))
            into name column_name value null_count
        )
        order by null_count desc, column_name
        """,
        [path],
    ).fetchdf()
    null_summary["null_rate"] = null_summary["null_count"] / null_summary["row_count"]
    print_df("Null Summary", null_summary)

    fare_distribution = con.execute(
        """
        select
            route,
            round(min(totalFare), 2) as min_fare,
            round(quantile_cont(totalFare, 0.25), 2) as p25,
            round(median(totalFare), 2) as median,
            round(avg(totalFare), 2) as mean,
            round(quantile_cont(totalFare, 0.75), 2) as p75,
            round(quantile_cont(totalFare, 0.95), 2) as p95,
            round(quantile_cont(totalFare, 0.99), 2) as p99,
            round(max(totalFare), 2) as max_fare
        from read_parquet(?)
        group by route
        order by route
        """,
        [path],
    ).fetchdf()
    print_df("Total Fare Distribution", fare_distribution)

    boolean_summary = con.execute(
        """
        select
            route,
            round(avg(case when isBasicEconomy then 1 else 0 end), 4) as basic_economy_rate,
            round(avg(case when isRefundable then 1 else 0 end), 4) as refundable_rate,
            round(avg(case when isNonStop then 1 else 0 end), 4) as nonstop_rate
        from read_parquet(?)
        group by route
        order by route
        """,
        [path],
    ).fetchdf()
    print_df("Boolean Feature Rates", boolean_summary)

    categorical_examples = con.execute(
        """
        select 'segmentsCabinCode' as column_name, segmentsCabinCode as value, count(*) as row_count
        from read_parquet(?)
        group by 1, 2
        union all
        select 'segmentsAirlineCode' as column_name, segmentsAirlineCode as value, count(*) as row_count
        from read_parquet(?)
        group by 1, 2
        order by column_name, row_count desc
        limit 25
        """,
        [path, path],
    ).fetchdf()
    print_df("Top Categorical Values", categorical_examples)

    repeated_leg_profile = con.execute(
        """
        with leg_profile as (
            select
                route,
                legId,
                count(*) as observations,
                count(distinct searchDate) as search_dates,
                count(distinct totalFare) as fare_values
            from read_parquet(?)
            group by route, legId
        )
        select
            route,
            count(*) as leg_ids,
            sum(case when search_dates > 1 then 1 else 0 end) as repeated_across_search_dates,
            sum(case when fare_values > 1 then 1 else 0 end) as multiple_fare_values,
            max(search_dates) as max_search_dates_per_leg,
            round(avg(search_dates), 2) as avg_search_dates_per_leg
        from leg_profile
        group by route
        order by route
        """,
        [path],
    ).fetchdf()
    print_df("Repeated LegId Profile", repeated_leg_profile)


if __name__ == "__main__":
    main()
