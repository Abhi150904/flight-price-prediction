from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path: str) -> None:
    script_path = ROOT / relative_path
    print(f"\n>>> {relative_path}")
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True)
