"""Tests for scripts/check_connector_exception_handling.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_connector_exception_handling.py"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ),
    )


def test_passes_when_no_silent_handlers(tmp_path: Path):
    source = tmp_path / "good.py"
    source.write_text(
        """
def f():
    try:
        x = 1 / 1
        return x
    except Exception as e:
        print(e)
        raise
""".strip()
    )

    result = _run("--path", str(tmp_path))
    assert result.returncode == 0
    assert "check passed" in result.stdout.lower()


def test_fails_on_silent_exception_handler(tmp_path: Path):
    source = tmp_path / "bad.py"
    source.write_text(
        """
def f():
    try:
        return 1
    except Exception:
        pass
""".strip()
    )

    result = _run("--path", str(tmp_path))
    assert result.returncode == 1
    assert "silent broad exception handler" in result.stdout


def test_fails_on_silent_exception_return_value(tmp_path: Path):
    source = tmp_path / "bad_return.py"
    source.write_text(
        """
def f():
    try:
        return 1
    except Exception:
        return {}
""".strip()
    )

    result = _run("--path", str(tmp_path))
    assert result.returncode == 1
    assert "silent broad exception handler (return value)" in result.stdout


def test_scans_multiple_paths(tmp_path: Path):
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()

    (good / "ok.py").write_text(
        """
def f():
    try:
        return 1
    except Exception as e:
        raise RuntimeError("boom") from e
""".strip()
    )
    (bad / "oops.py").write_text(
        """
def f():
    try:
        return 1
    except Exception:
        return None
""".strip()
    )

    result = _run("--path", str(good), str(bad))
    assert result.returncode == 1
    assert "oops.py" in result.stdout
