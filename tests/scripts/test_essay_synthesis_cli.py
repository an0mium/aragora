"""Tests for scripts/essay_synthesis_cli.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_direct_help_works_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/essay_synthesis_cli.py", "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Synthesize essays from ChatGPT/Claude exports." in result.stdout
    assert "--exports" in result.stdout
