from __future__ import annotations

from run_step import run_script


def main() -> None:
    run_script("scripts/learning_phases/run_phase14_evaluation.py")
    run_script("scripts/learning_phases/run_phase17_interpretation.py")
    run_script("scripts/learning_phases/run_phase18_price_context.py")
    run_script("scripts/learning_phases/run_phase20_uncertainty.py")


if __name__ == "__main__":
    main()
