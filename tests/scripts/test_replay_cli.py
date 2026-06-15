"""Tests for direct invocation of scripts/replay_cli.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_cli.py"


def _clean_import_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_replay_cli_help_bootstraps_repo_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        cwd=REPO_ROOT,
        env=_clean_import_env(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "replay" in result.stdout
    assert "list" in result.stdout


def test_replay_cli_requires_subcommand() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        cwd=REPO_ROOT,
        env=_clean_import_env(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "the following arguments are required: command" in result.stderr
