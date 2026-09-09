-- Phase 5: SQL / Data Layer
-- Database: data/processed/airfare.duckdb
-- Table: airfare_subset
--
-- Suggested runner:
-- python -c "import duckdb; con=duckdb.connect('data/processed/airfare.duckdb'); print(con.execute(open('sql/phase05_airfare_analysis.sql').read()).fetchdf())"
--
-- In practice, run one query at a time while learning.

-- 1. Basic table check
select
    count(*) as row_count,
    count(distinct route) as route_count,
    count(distinct legId) as leg_id_count,
    min(searchDate) as first_search_date,
    max(searchDate) as last_search_date,
    min(flightDate) as first_flight_date,
    max(flightDate) as last_flight_date
from airfare_subset;

-- 2. Average and median fare by route
select
    route,
    count(*) as row_count,
    round(avg(totalFare), 2) as avg_total_fare,
    round(median(totalFare), 2) as median_total_fare,
    round(quantile_cont(totalFare, 0.25), 2) as p25_total_fare,
    round(quantile_cont(totalFare, 0.75), 2) as p75_total_fare
from airfare_subset
group by route
order by median_total_fare desc;

-- 3. Fare by nonstop vs connecting flights
select
    route,
    case
        when isNonStop then 'nonstop'
        else 'connecting'
    end as stop_type,
    count(*) as row_count,
    round(avg(totalFare), 2) as avg_total_fare,
    round(median(totalFare), 2) as median_total_fare
from airfare_subset
group by route, stop_type
order by route, stop_type;

-- 4. Lead-time buckets
select
    route,
    case
        when lead_time_days between 1 and 7 then '01-07 days'
        when lead_time_days between 8 and 14 then '08-14 days'
        when lead_time_days between 15 and 30 then '15-30 days'
        when lead_time_days between 31 and 45 then '31-45 days'
        else '46-60 days'
    end as lead_time_bucket,
    count(*) as row_count,
    round(avg(totalFare), 2) as avg_total_fare,
    round(median(totalFare), 2) as median_total_fare
from airfare_subset
group by route, lead_time_bucket
order by route, lead_time_bucket;

-- 5. Airline patterns ranked within each route
with airline_summary as (
    select
        route,
        segmentsAirlineCode as airline_pattern,
        count(*) as row_count,
        round(median(totalFare), 2) as median_total_fare
    from airfare_subset
    group by route, airline_pattern
),
ranked_airlines as (
    select
        *,
        row_number() over (
            partition by route
            order by median_total_fare desc
        ) as expensive_rank
    from airline_summary
    where row_count >= 1000
)
select
    route,
    airline_pattern,
    row_count,
    median_total_fare,
    expensive_rank
from ranked_airlines
where expensive_rank <= 10
order by route, expensive_rank;

-- 6. Same-itinerary fare movement examples
with daily_leg_prices as (
    select
        route,
        legId,
        searchDate,
        flightDate,
        lead_time_days,
        min(totalFare) as observed_fare
    from airfare_subset
    group by route, legId, searchDate, flightDate, lead_time_days
),
with_previous_price as (
    select
        *,
        lag(observed_fare) over (
            partition by legId
            order by searchDate
        ) as previous_observed_fare
    from daily_leg_prices
)
select
    route,
    legId,
    searchDate,
    flightDate,
    lead_time_days,
    observed_fare,
    previous_observed_fare,
    round(observed_fare - previous_observed_fare, 2) as fare_change_from_previous_search
from with_previous_price
where previous_observed_fare is not null
order by abs(observed_fare - previous_observed_fare) desc
limit 20;
