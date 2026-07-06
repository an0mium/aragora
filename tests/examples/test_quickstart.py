"""Regression tests for the root quickstart example."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_quickstart_bootstraps_bundled_debate_wedge_without_site_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }

    result = subprocess.run(
        [sys.executable, "-S", "examples/quickstart.py"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "# Decision Receipt" in result.stdout
