# Data Directory

This project uses the Expedia airfare dataset.

- `raw/`: original downloaded CSV files. These are large and excluded from Git.
- `interim/`: smaller working files, such as selected-route Parquet subsets.
- `processed/`: DuckDB databases and derived analytical tables.

The raw CSV is intentionally not loaded directly into Pandas. Use DuckDB scripts in `scripts/` to create smaller, reproducible working datasets.
