from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_CSV = Path("data/raw/itineraries.csv")


def print_df(title: str, df) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight Expedia airfare dataset validation queries."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--route",
        action="append",
        nargs=2,
        metavar=("ORIGIN", "DESTINATION"),
        help="Route to inspect, for example: --route ATL BOS. Can be repeated.",
    )
    parser.add_argument("--sample-route-limit", type=int, default=8)
    args = parser.parse_args()

    csv_path = args.csv.resolve().as_posix()
    con = duckdb.connect()

    scan = "read_csv_auto(?, sample_size=100000, header=true)"

    if args.route:
        candidate_routes = [(origin.upper(), destination.upper()) for origin, destination in args.route]
    else:
        top_routes = con.execute(
            f"""
            select
                startingAirport,
                destinationAirport,
                count(*) as rows,
                count(distinct flightDate) as flight_dates,
                count(distinct searchDate) as search_dates,
                round(avg(totalFare), 2) as avg_total_fare,
                round(min(totalFare), 2) as min_total_fare,
                round(max(totalFare), 2) as max_total_fare
            from {scan}
            group by 1, 2
            order by rows desc
            limit ?
            """,
            [csv_path, args.sample_route_limit],
        ).fetchdf()
        print_df("Top Routes By Row Count", top_routes)
        candidate_routes = list(
            top_routes[["startingAirport", "destinationAirport"]].head(3).itertuples(index=False, name=None)
        )

    for origin, destination in candidate_routes:

        profile = con.execute(
            f"""
            with route_data as (
                select
                    legId,
                    searchDate,
                    flightDate,
                    totalFare,
                    baseFare,
                    isNonStop,
                    seatsRemaining,
                    travelDuration,
                    segmentsAirlineCode,
                    segmentsCabinCode
                from {scan}
                where startingAirport = ?
                  and destinationAirport = ?
            ),
            leg_profile as (
                select
                    legId,
                    count(*) as observations,
                    count(distinct searchDate) as search_dates,
                    count(distinct flightDate) as flight_dates,
                    count(distinct totalFare) as fare_values,
                    min(searchDate) as first_seen,
                    max(searchDate) as last_seen
                from route_data
                group by legId
            )
            select
                ? as startingAirport,
                ? as destinationAirport,
                (select count(*) from route_data) as route_rows,
                count(*) as distinct_leg_ids,
                sum(case when observations > 1 then 1 else 0 end) as repeated_leg_ids,
                sum(case when search_dates > 1 then 1 else 0 end) as leg_ids_seen_on_multiple_search_dates,
                sum(case when fare_values > 1 then 1 else 0 end) as leg_ids_with_multiple_fares,
                max(observations) as max_observations_per_leg,
                max(search_dates) as max_search_dates_per_leg,
                round(avg(observations), 2) as avg_observations_per_leg
            from leg_profile
            """,
            [csv_path, origin, destination, origin, destination],
        ).fetchdf()
        print_df(f"Repeated LegId Profile: {origin} -> {destination}", profile)

        examples = con.execute(
            f"""
            with route_data as (
                select
                    legId,
                    searchDate,
                    flightDate,
                    totalFare,
                    isNonStop,
                    seatsRemaining,
                    travelDuration,
                    segmentsAirlineCode,
                    segmentsCabinCode
                from {scan}
                where startingAirport = ?
                  and destinationAirport = ?
            ),
            repeated_legs as (
                select legId
                from route_data
                group by legId
                having count(distinct searchDate) > 1
                   and count(distinct totalFare) > 1
                order by count(distinct searchDate) desc, count(*) desc
                limit 3
            )
            select
                legId,
                searchDate,
                flightDate,
                totalFare,
                isNonStop,
                seatsRemaining,
                travelDuration,
                segmentsAirlineCode,
                segmentsCabinCode
            from route_data
            where legId in (select legId from repeated_legs)
            order by legId, searchDate, totalFare
            limit 30
            """,
            [csv_path, origin, destination],
        ).fetchdf()
        print_df(f"Example Repeated Fare Observations: {origin} -> {destination}", examples)


if __name__ == "__main__":
    main()
