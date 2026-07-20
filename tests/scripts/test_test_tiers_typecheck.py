"""Regression tests for the typecheck tier shell helper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TIERS = REPO_ROOT / "scripts" / "test_tiers.sh"


def _run_typecheck_with_fake_python(
    tmp_path: Path, status: int
) -> subprocess.CompletedProcess[str]:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        '[[ "$*" == *scripts/ci/mypy_with_baseline.py* ]] || exit 99\n'
        f"exit {status}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["TYPECHECK_PYTHON"] = str(fake_python)
    return subprocess.run(
        ["bash", str(TEST_TIERS), "typecheck"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_typecheck_tier_delegates_to_baseline_helper(tmp_path: Path) -> None:
    result = _run_typecheck_with_fake_python(tmp_path, 0)

    assert result.returncode == 0
    assert "Type check passed (no new errors)" in result.stdout


@pytest.mark.parametrize("status", [1, 2])
def test_typecheck_tier_propagates_failure_status(tmp_path: Path, status: int) -> None:
    result = _run_typecheck_with_fake_python(tmp_path, status)

    assert result.returncode == status
    assert "Type check FAILED" in result.stdout
