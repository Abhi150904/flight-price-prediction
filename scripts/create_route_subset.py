from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_CSV = Path("data/raw/itineraries.csv")
DEFAULT_OUTPUT = Path("data/interim/expedia_route_subset.parquet")
DEFAULT_ROUTES = [("ATL", "BOS"), ("LAX", "JFK")]


def route_values(routes: list[tuple[str, str]]) -> str:
    return ", ".join(f"('{origin}', '{destination}')" for origin, destination in routes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manageable Parquet subset from the Expedia airfare CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--route",
        action="append",
        nargs=2,
        metavar=("ORIGIN", "DESTINATION"),
        help="Route to include, for example: --route ATL BOS. Can be repeated.",
    )
    args = parser.parse_args()

    routes = (
        [(origin.upper(), destination.upper()) for origin, destination in args.route]
        if args.route
        else DEFAULT_ROUTES
    )

    csv_path = args.csv.resolve().as_posix()
    output_path = args.output.resolve()
    output_sql = output_path.as_posix().replace("'", "''")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    scan = "read_csv_auto(?, sample_size=100000, header=true)"

    con.execute(
        f"""
        copy (
            select
                legId,
                searchDate,
                flightDate,
                date_diff('day', searchDate, flightDate) as lead_time_days,
                startingAirport,
                destinationAirport,
                startingAirport || '-' || destinationAirport as route,
                fareBasisCode,
                travelDuration,
                elapsedDays,
                isBasicEconomy,
                isRefundable,
                isNonStop,
                baseFare,
                totalFare,
                seatsRemaining,
                totalTravelDistance,
                segmentsDepartureTimeEpochSeconds,
                segmentsDepartureTimeRaw,
                segmentsArrivalTimeEpochSeconds,
                segmentsArrivalTimeRaw,
                segmentsArrivalAirportCode,
                segmentsDepartureAirportCode,
                segmentsAirlineName,
                segmentsAirlineCode,
                segmentsEquipmentDescription,
                segmentsDurationInSeconds,
                segmentsDistance,
                segmentsCabinCode
            from {scan}
            where (startingAirport, destinationAirport) in ({route_values(routes)})
        )
        to '{output_sql}'
        (format parquet, compression zstd)
        """,
        [csv_path],
    )

    summary = con.execute(
        """
        select
            count(*) as rows,
            count(distinct route) as routes,
            count(distinct legId) as leg_ids,
            min(searchDate) as min_search_date,
            max(searchDate) as max_search_date,
            min(flightDate) as min_flight_date,
            max(flightDate) as max_flight_date,
            round(min(totalFare), 2) as min_total_fare,
            round(avg(totalFare), 2) as avg_total_fare,
            round(max(totalFare), 2) as max_total_fare
        from read_parquet(?)
        """,
        [output_path.as_posix()],
    ).fetchdf()

    print(f"Wrote {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
