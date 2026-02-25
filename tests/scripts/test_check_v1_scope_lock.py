"""Tests for scripts/check_v1_scope_lock.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/check_v1_scope_lock.py")


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env=merged_env,
        check=False,
    )


def test_scope_lock_passes_for_in_scope_changes() -> None:
    result = _run(
        [
            "--files",
            "aragora/server/debate_controller.py",
            "tests/schedulers/test_settlement_review.py",
        ]
    )
    assert result.returncode == 0
    assert "passed" in result.stdout.lower()


def test_scope_lock_fails_for_blocked_prefix() -> None:
    result = _run(["--files", "aragora/server/handlers/social/slack.py"])
    assert result.returncode == 1
    assert "violation" in result.stderr.lower()
    assert "social/slack.py" in result.stderr


def test_scope_lock_can_be_overridden_by_env() -> None:
    result = _run(
        ["--files", "aragora/server/handlers/social/slack.py"],
        env={"ARAGORA_ALLOW_SCOPE_EXPANSION": "1"},
    )
    assert result.returncode == 0
    assert "bypassed" in result.stdout.lower()


def test_scope_lock_fails_when_lock_file_missing(tmp_path: Path) -> None:
    missing_lock = tmp_path / "missing.md"
    result = _run(["--files", "aragora/server/debate_controller.py", "--lock-file", str(missing_lock)])
    assert result.returncode == 1
    assert "missing" in result.stderr.lower()
