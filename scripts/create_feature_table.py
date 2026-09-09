from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/processed/airfare.duckdb")


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix())

    con.execute(
        """
        create or replace table airfare_features as
        with parsed as (
            select
                legId,
                searchDate,
                flightDate,
                totalFare,

                -- Core itinerary features available before purchase.
                route,
                startingAirport,
                destinationAirport,
                lead_time_days,
                isBasicEconomy,
                isNonStop,
                seatsRemaining,
                elapsedDays,
                segment_count,
                totalTravelDistance,
                missing_total_travel_distance,

                -- Pattern features. These describe the offered itinerary but can be high-cardinality.
                segmentsAirlineCode as airline_pattern,
                split_part(segmentsAirlineCode, '||', 1) as first_airline_code,
                segmentsCabinCode as cabin_pattern,
                split_part(segmentsCabinCode, '||', 1) as first_cabin_code,

                -- Time parsing from the first flight segment.
                try_cast(regexp_extract(split_part(segmentsDepartureTimeRaw, '||', 1), 'T([0-9]{2}):', 1) as integer)
                    as departure_hour,
                try_cast(regexp_extract(split_part(segmentsArrivalTimeRaw, '||', 1), 'T([0-9]{2}):', 1) as integer)
                    as arrival_hour,

                -- Duration parsing from strings such as PT5H21M and P1DT4H46M.
                coalesce(try_cast(nullif(regexp_extract(travelDuration, 'P(?:(\\d+)D)?T(?:(\\d+)H)?(?:(\\d+)M)?', 1), '') as integer), 0) * 1440
                    + coalesce(try_cast(nullif(regexp_extract(travelDuration, 'P(?:(\\d+)D)?T(?:(\\d+)H)?(?:(\\d+)M)?', 2), '') as integer), 0) * 60
                    + coalesce(try_cast(nullif(regexp_extract(travelDuration, 'P(?:(\\d+)D)?T(?:(\\d+)H)?(?:(\\d+)M)?', 3), '') as integer), 0)
                    as travel_duration_minutes
            from airfare_clean
        )
        select
            *,
            case
                when departure_hour between 5 and 11 then 'morning'
                when departure_hour between 12 and 16 then 'afternoon'
                when departure_hour between 17 and 21 then 'evening'
                when departure_hour is null then 'unknown'
                else 'night'
            end as departure_period,
            case
                when arrival_hour between 5 and 11 then 'morning'
                when arrival_hour between 12 and 16 then 'afternoon'
                when arrival_hour between 17 and 21 then 'evening'
                when arrival_hour is null then 'unknown'
                else 'night'
            end as arrival_period,
            case
                when totalTravelDistance is not null then totalTravelDistance
                else median(totalTravelDistance) over (partition by route, isNonStop)
            end as distance_imputed_by_route_stop
        from parsed
        """
    )

    summary = con.execute(
        """
        select
            count(*) as rows,
            count(distinct route) as routes,
            count(distinct first_airline_code) as first_airlines,
            count(distinct airline_pattern) as airline_patterns,
            count(distinct first_cabin_code) as first_cabins,
            count(distinct cabin_pattern) as cabin_patterns,
            min(travel_duration_minutes) as min_duration_minutes,
            max(travel_duration_minutes) as max_duration_minutes,
            sum(case when departure_hour is null then 1 else 0 end) as null_departure_hour,
            sum(case when travel_duration_minutes <= 0 then 1 else 0 end) as nonpositive_duration_rows
        from airfare_features
        """
    ).fetchdf()

    print("Created table: airfare_features")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
