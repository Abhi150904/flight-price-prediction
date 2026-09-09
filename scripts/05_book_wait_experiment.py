from __future__ import annotations

from run_step import run_script


def main() -> None:
    run_script("scripts/create_book_wait_labels.py")
    run_script("scripts/learning_phases/run_phase19_book_wait.py")


if __name__ == "__main__":
    main()
