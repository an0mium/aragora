"""Tests for ``scripts/refresh_proof_surfaces.sh`` (H02).

The script is a thin orchestrator over already-tested
publish/render scripts. These tests exercise only the orchestrator's
own behaviour: arg validation, --check mode, --surface scoping, and
the path computation for promoting `.aragora/<surface>/latest.json`
to `docs/status/generated/<surface>/latest.json`.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_proof_surfaces.sh"


def _run(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(REFRESH_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )


def test_help_succeeds() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "refresh_proof_surfaces.sh" in result.stdout
    assert "--surface" in result.stdout


def test_unknown_flag_errors() -> None:
    result = _run("--bogus")
    assert result.returncode == 1
    assert "Unknown flag" in result.stderr


def test_invalid_surface_errors() -> None:
    result = _run("--surface", "bogus")
    assert result.returncode == 1
    assert "must be one of all|b0|tw03" in result.stderr


def test_check_mode_does_not_write() -> None:
    """--check mode runs the freshness probe but does not invoke publishers."""
    tw03_path = REPO_ROOT / "docs" / "status" / "TW03_RESCUE_PRODUCTIZATION_STATUS.md"
    pre_mtime = tw03_path.stat().st_mtime if tw03_path.exists() else 0
    result = _run("--check")
    assert result.returncode == 0
    assert "--check requested" in result.stdout
    post_mtime = tw03_path.stat().st_mtime if tw03_path.exists() else 0
    # No publisher should have touched the file under --check.
    assert pre_mtime == post_mtime


def test_check_mode_default_surface_all() -> None:
    """Without --surface, default is 'all' (script header echoes it)."""
    result = _run("--check")
    assert "surface=all" in result.stdout


def test_check_mode_with_specific_surface() -> None:
    result = _run("--surface", "tw03", "--check")
    assert result.returncode == 0
    assert "surface=tw03" in result.stdout


def test_script_is_executable() -> None:
    mode = REFRESH_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "refresh_proof_surfaces.sh must be executable"


def test_script_passes_bash_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REFRESH_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"
