from __future__ import annotations

from run_step import run_script


def main() -> None:
    run_script("scripts/learning_phases/run_phase12_baselines.py")
    run_script("scripts/learning_phases/run_phase13_models.py")
    run_script("scripts/learning_phases/run_phase15_cross_validation.py")
    run_script("scripts/learning_phases/run_phase21_finalize_model.py")


if __name__ == "__main__":
    main()
