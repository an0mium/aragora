from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tune_execution_gate.py"


def test_tune_execution_gate_help_entrypoint() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Tune execution safety gate thresholds" in result.stdout
    assert "--output" in result.stdout
