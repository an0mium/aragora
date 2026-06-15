"""Tests for scripts/generate_receipt_schema.py."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def test_direct_script_execution_imports_aragora_without_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_receipt_schema.py"),
            "--stdout",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--- decision_receipt.v1.json ---" in proc.stdout
    assert "--- gauntlet_receipt.v1.json ---" in proc.stdout


def test_stdout_mode_suppresses_closed_consumer_broken_pipe() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = repo_root / "scripts" / "generate_receipt_schema.py"
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} --stdout | true"

    proc = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "BrokenPipeError" not in proc.stderr
