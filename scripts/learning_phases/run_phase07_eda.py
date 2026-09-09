from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


DB_PATH = Path("data/processed/airfare.duckdb")
FIGURE_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")


def save_table(df, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def style_axes(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            va="bottom",
        )
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(colors="#333333", labelsize=9)


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fare_distribution(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute(
        """
        with limits as (
            select quantile_cont(totalFare, 0.99) as p99_fare
            from airfare_clean
        )
        select
            route,
            floor(totalFare / 50) * 50 as fare_bin,
            count(*) as row_count
        from airfare_clean, limits
        where totalFare <= p99_fare
        group by route, fare_bin
        order by route, fare_bin
        """
    ).fetchdf()
    save_table(df, "fare_distribution_bins")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"ATL-BOS": "#2f6f9f", "LAX-JFK": "#c9822b"}
    for route, group in df.groupby("route"):
        ax.plot(
            group["fare_bin"],
            group["row_count"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=route,
            color=colors.get(route, "#555555"),
        )
    style_axes(
        ax,
        "Total Fare Distribution By Route",
        "Rows are binned in $50 increments and capped at the 99th percentile.",
    )
    ax.set_xlabel("Total fare bin, USD")
    ax.set_ylabel("Observed fare rows")
    ax.legend(frameon=False)
    save_figure(fig, "phase07_total_fare_distribution")


def plot_lead_time(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute(
        """
        select
            route,
            lead_time_days,
            count(*) as row_count,
            round(median(totalFare), 2) as median_total_fare
        from airfare_clean
        group by route, lead_time_days
        order by route, lead_time_days
        """
    ).fetchdf()
    save_table(df, "lead_time_daily_median_fares")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"ATL-BOS": "#2f6f9f", "LAX-JFK": "#c9822b"}
    for route, group in df.groupby("route"):
        ax.plot(
            group["lead_time_days"],
            group["median_total_fare"],
            linewidth=2,
            label=route,
            color=colors.get(route, "#555555"),
        )
    style_axes(
        ax,
        "Median Fare By Lead Time",
        "Lead time is flightDate minus searchDate; values run from 1 to 60 days.",
    )
    ax.set_xlabel("Days before departure")
    ax.set_ylabel("Median total fare, USD")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    save_figure(fig, "phase07_median_fare_by_lead_time")


def plot_cabin_price(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute(
        """
        select
            route,
            segmentsCabinCode,
            count(*) as row_count,
            round(median(totalFare), 2) as median_total_fare
        from airfare_clean
        group by route, segmentsCabinCode
        having count(*) >= 100
        order by route, median_total_fare desc
        """
    ).fetchdf()
    save_table(df, "cabin_price_summary")

    plot_df = df.copy()
    plot_df["label"] = plot_df["route"] + " | " + plot_df["segmentsCabinCode"]
    plot_df = plot_df.sort_values("median_total_fare").tail(12)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["label"], plot_df["median_total_fare"], color="#6b8e23")
    style_axes(
        ax,
        "Median Fare By Cabin Pattern",
        "Includes cabin patterns with at least 100 observed rows.",
    )
    ax.set_xlabel("Median total fare, USD")
    ax.set_ylabel("")
    save_figure(fig, "phase07_cabin_pattern_median_fare")


def plot_repeated_leg_example(con: duckdb.DuckDBPyConnection) -> None:
    leg = con.execute(
        """
        with leg_profile as (
            select
                legId,
                route,
                count(distinct searchDate) as search_dates,
                count(distinct totalFare) as fare_values,
                max(totalFare) - min(totalFare) as fare_range
            from airfare_clean
            where isNonStop
              and segmentsCabinCode = 'coach'
            group by legId, route
        )
        select legId
        from leg_profile
        where search_dates >= 30
          and fare_values >= 5
        order by fare_range desc
        limit 1
        """
    ).fetchone()[0]

    df = con.execute(
        """
        select
            route,
            legId,
            searchDate,
            flightDate,
            lead_time_days,
            min(totalFare) as observed_fare
        from airfare_clean
        where legId = ?
        group by route, legId, searchDate, flightDate, lead_time_days
        order by searchDate
        """,
        [leg],
    ).fetchdf()
    save_table(df, "repeated_leg_example")

    route = df["route"].iloc[0]
    flight_date = df["flightDate"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        df["lead_time_days"],
        df["observed_fare"],
        marker="o",
        markersize=4,
        linewidth=2,
        color="#2f6f9f",
    )
    style_axes(
        ax,
        "Fare Path For One Repeated Itinerary",
        f"Route {route}; flight date {flight_date}; one nonstop coach legId.",
    )
    ax.set_xlabel("Days before departure")
    ax.set_ylabel("Observed total fare, USD")
    ax.invert_xaxis()
    save_figure(fig, "phase07_repeated_leg_fare_path")


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    plot_fare_distribution(con)
    plot_lead_time(con)
    plot_cabin_price(con)
    plot_repeated_leg_example(con)
    print(f"Wrote figures to {FIGURE_DIR.resolve()}")
    print(f"Wrote chart-ready tables to {TABLE_DIR.resolve()}")


if __name__ == "__main__":
    main()
