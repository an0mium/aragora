"""Tests for scripts/codex_profile.sh."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "codex_profile.sh"


def _write_codex_stub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "codex"
    stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

print(json.dumps({
    "argv": sys.argv[1:],
    "CODEX_HOME": os.environ.get("CODEX_HOME"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
}))
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _write_codex_stub(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path / "home")
    env["OPENAI_API_KEY"] = "parent-openai"
    env["ANTHROPIC_API_KEY"] = "parent-anthropic"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_home_returns_default_profile_dir(tmp_path: Path):
    result = _run(tmp_path, "home", "pro-01")
    assert result.returncode == 0
    assert result.stdout.strip() == str((tmp_path / "home" / ".aragora-codex" / "pro-01").resolve())


def test_status_uses_isolated_codex_home_and_strips_shell_keys(tmp_path: Path):
    result = _run(tmp_path, "status", "pro-02")
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["argv"] == ["login", "status"]
    assert payload["CODEX_HOME"] == str((tmp_path / "home" / ".aragora-codex" / "pro-02").resolve())
    assert payload["OPENAI_API_KEY"] is None
    assert payload["ANTHROPIC_API_KEY"] is None


def test_exec_sanitizes_keys_for_arbitrary_child_command(tmp_path: Path):
    result = _run(tmp_path, "exec", "pro-03", "--", "codex", "exec", "hello")
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["argv"] == ["exec", "hello"]
    assert payload["CODEX_HOME"] == str((tmp_path / "home" / ".aragora-codex" / "pro-03").resolve())
    assert payload["OPENAI_API_KEY"] is None
    assert payload["ANTHROPIC_API_KEY"] is None
