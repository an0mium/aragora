"""Tests for scripts/check_agent_registry_drift.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_agent_registry_drift.py"


def test_runtime_fallback_passes_without_site_packages():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT.parents[1]),
        env=env,
        timeout=20,
    )

    assert result.returncode == 0
    assert "falling back to source parsing" in result.stderr.lower()
    assert "runtime agent registry:" in result.stdout.lower()
