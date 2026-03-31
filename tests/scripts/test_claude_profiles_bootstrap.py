"""Tests for scripts/claude_profiles_bootstrap.sh."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "claude_profiles_bootstrap.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _install_command_wrapper(bin_dir: Path, name: str) -> None:
    target = shutil.which(name)
    if target is None:
        raise AssertionError(f"required command missing from test host: {name}")
    _write_executable(
        bin_dir / name,
        f"""#!/bin/sh
exec "{target}" "$@"
""",
    )


def _prepare_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_copy = scripts_dir / "claude_profiles_bootstrap.sh"
    script_copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script_copy.chmod(script_copy.stat().st_mode | stat.S_IXUSR)

    _write_executable(
        scripts_dir / "claude_profile.sh",
        f"""#!/bin/bash
set -euo pipefail

cmd="${{1:-}}"
shift || true

case "$cmd" in
  status)
    printf '%s\\n' '{{"loggedIn": true, "email": "test@example.com"}}'
    ;;
  exec)
    shift
    if [[ "${{1:-}}" == "--" ]]; then
      shift
    fi
    exec "$@"
    ;;
  home)
    printf '%s\\n' "{tmp_path / "profile-home"}"
    ;;
  logout)
    exit 0
    ;;
  *)
    echo "unsupported: $cmd" >&2
    exit 1
    ;;
esac
""",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "claude",
        """#!/bin/sh
printf 'ok\\n'
""",
    )
    for command in ("dirname", "grep", "head", "perl", "sed"):
        _install_command_wrapper(bin_dir, command)

    env = {"PATH": str(bin_dir)}
    return script_copy, env


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script, env = _prepare_fixture(tmp_path)
    return subprocess.run(
        ["/bin/bash", str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_verify_works_without_timeout_binary(tmp_path: Path) -> None:
    result = _run(tmp_path, "verify", "test-profile")

    assert result.returncode == 0
    assert "OK  (test@example.com)" in result.stdout
    assert "1 passed, 0 failed" in result.stdout
    assert "EXPIRED" not in result.stdout


def test_login_skips_verified_profile_without_timeout_binary(tmp_path: Path) -> None:
    result = _run(tmp_path, "login", "test-profile")

    assert result.returncode == 0
    assert "Already logged in and verified; skipping. (test@example.com)" in result.stdout
