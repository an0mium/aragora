"""Regression tests for the typecheck tier shell helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TIERS = REPO_ROOT / "scripts" / "test_tiers.sh"


def _run_typecheck_with_fake_mypy(
    tmp_path: Path, fake_mypy_body: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mypy = fake_bin / "mypy"
    fake_mypy.write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\n{fake_mypy_body}\n",
        encoding="utf-8",
    )
    fake_mypy.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(TEST_TIERS), "typecheck"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_typecheck_tier_fails_on_mypy_error_without_column_number(tmp_path: Path) -> None:
    proc = _run_typecheck_with_fake_mypy(
        tmp_path,
        "printf '%s\\n' 'aragora/example.py:12: error: incompatible type [assignment]'\nexit 1",
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "Found 1 mypy error(s)!" in output
    assert "aragora/example.py:12: error: incompatible type [assignment]" in output
    assert "=== Type check FAILED ===" in output


def test_typecheck_tier_passes_when_mypy_exits_cleanly(tmp_path: Path) -> None:
    proc = _run_typecheck_with_fake_mypy(tmp_path, "exit 0")

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "=== Type check passed (0 errors) ===" in output
