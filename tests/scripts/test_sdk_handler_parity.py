"""Tests for scripts/sdk_handler_parity.py CLI importability."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sdk_handler_parity_imports_from_scripts_cwd_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                "runpy.run_path('sdk_handler_parity.py', "
                "run_name='sdk_handler_parity_import_test')"
            ),
        ],
        cwd=ROOT / "scripts",
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
