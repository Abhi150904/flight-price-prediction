from __future__ import annotations

from run_step import run_script


def main() -> None:
    run_script("scripts/create_clean_subset.py")
    run_script("scripts/create_feature_table.py")
    run_script("scripts/create_model_splits.py")


if __name__ == "__main__":
    main()
