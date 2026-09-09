from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/processed/airfare.duckdb")


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix())

    con.execute(
        """
        create or replace table airfare_clean as
        select
            legId,
            searchDate,
            flightDate,
            lead_time_days,
            route,
            startingAirport,
            destinationAirport,
            fareBasisCode,
            travelDuration,
            elapsedDays,
            isBasicEconomy,
            isRefundable,
            isNonStop,
            totalFare,
            seatsRemaining,
            totalTravelDistance,
            segmentsAirlineCode,
            segmentsCabinCode,
            segmentsDepartureTimeRaw,
            segmentsArrivalTimeRaw,
            segmentsDepartureAirportCode,
            segmentsArrivalAirportCode,
            segmentsDurationInSeconds,
            segmentsDistance,
            1 + length(segmentsAirlineCode) - length(replace(segmentsAirlineCode, '||', '|')) as segment_count,
            case
                when totalTravelDistance is null then true
                else false
            end as missing_total_travel_distance
        from airfare_subset
        where totalFare > 0
          and lead_time_days between 1 and 60
          and seatsRemaining >= 0
        """
    )

    summary = con.execute(
        """
        select
            count(*) as rows,
            count(distinct legId) as leg_ids,
            count(distinct route) as routes,
            sum(case when missing_total_travel_distance then 1 else 0 end) as missing_distance_rows,
            round(avg(totalFare), 2) as avg_total_fare,
            round(median(totalFare), 2) as median_total_fare
        from airfare_clean
        """
    ).fetchdf()

    print("Created table: airfare_clean")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
