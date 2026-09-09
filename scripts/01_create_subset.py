from __future__ import annotations

from run_step import run_script


def main() -> None:
    run_script("scripts/create_route_subset.py")
    run_script("scripts/learning_phases/profile_route_subset.py")


if __name__ == "__main__":
    main()
