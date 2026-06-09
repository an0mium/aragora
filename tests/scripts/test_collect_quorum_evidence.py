"""Focused tests for ``scripts/collect_quorum_evidence.py``."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_collect_quorum_evidence_help_suppresses_broken_pipe() -> None:
    command = (
        f"{shlex.quote(sys.executable)} scripts/collect_quorum_evidence.py --help "
        "| head -5 >/dev/null"
    )
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "BrokenPipeError" not in result.stderr
