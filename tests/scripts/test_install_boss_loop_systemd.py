"""Render tests for ``scripts/install_boss_loop_systemd.sh``.

The installer mirrors the launchd installer's env contract but targets
systemd user units so the boss loop can run on Linux runners (de-risking the
macOS launchd monoculture). ``--dry-run`` (the default) must print the
wrapper script plus the ``.service`` and ``.timer`` unit text to stdout
without touching the filesystem; ``--install`` must refuse to run on darwin.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install_boss_loop_systemd.sh"

requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to run the installer"
)


def _run(
    *args: str,
    env_overrides: dict[str, str] | None = None,
    drop: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in drop:
        env.pop(key, None)
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_bash
def test_dry_run_is_default_and_prints_service_and_timer() -> None:
    result = _run(drop=("ARAGORA_TIER4_TRUSTED_OPERATORS",))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "aragora-boss-loop.service" in out
    assert "aragora-boss-loop.timer" in out
    assert "[Service]" in out
    assert "[Timer]" in out
    assert "ExecStart=" in out


@requires_bash
def test_dry_run_renders_env_contract_values() -> None:
    result = _run(
        env_overrides={
            "BOSS_INTERVAL_SECONDS": "77",
            "BOSS_MAX_HOURS": "5",
            "BOSS_MAX_CONSECUTIVE_FAILURES": "9",
            "BOSS_THROTTLE_SECONDS": "123",
        },
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Loop flags flow into the rendered wrapper.
    assert '--interval "77"' in out
    assert '--max-hours "5"' in out
    assert '--max-consecutive-failures "9"' in out
    # Throttle flows into the restart backoff.
    assert "RestartSec=123" in out
    # Start-limit window mirrors MAX_CONSECUTIVE_FAILURES.
    assert "StartLimitBurst=9" in out
    assert f"StartLimitIntervalSec={9 * 123}" in out


@requires_bash
def test_dry_run_renders_restart_directives() -> None:
    result = _run()
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Restart=on-failure" in out
    assert "RestartSec=" in out
    assert "StartLimitIntervalSec=" in out
    assert "StartLimitBurst=" in out
    # Exponential backoff directives are present (active or guard-commented
    # depending on the host's systemd version).
    assert "RestartMaxDelaySec=3600" in out
    assert "RestartSteps=6" in out


@requires_bash
def test_wrapper_resolves_python_at_launch_time_never_baked() -> None:
    result = _run()
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "aragora_runtime.sh" in out
    assert "resolve_aragora_python" in out
    # No interpreter path may be baked into ExecStart or the wrapper.
    for line in out.splitlines():
        if line.startswith("ExecStart="):
            assert "python" not in line, f"baked interpreter in: {line}"
    baked_venv_interpreter = "/".join((".venv", "bin", "python"))
    assert baked_venv_interpreter not in out


@requires_bash
def test_label_and_env_passthrough() -> None:
    result = _run(
        "--label",
        "custom-lane",
        "--user-id",
        "test-user",
        "--workspace-id",
        "test-ws",
        env_overrides={"ARAGORA_TIER4_TRUSTED_OPERATORS": "alice,bob"},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert '--label "custom-lane"' in out
    assert 'Environment="ARAGORA_USER_ID=test-user"' in out
    assert 'Environment="ARAGORA_WORKSPACE_ID=test-ws"' in out
    assert 'Environment="ARAGORA_TIER4_TRUSTED_OPERATORS=alice,bob"' in out


@requires_bash
def test_tier4_operators_omitted_when_unset() -> None:
    result = _run(drop=("ARAGORA_TIER4_TRUSTED_OPERATORS",))
    assert result.returncode == 0, result.stderr
    assert "ARAGORA_TIER4_TRUSTED_OPERATORS" not in result.stdout


@requires_bash
def test_install_refused_on_darwin(tmp_path: Path) -> None:
    """--install must hard-refuse when uname reports Darwin (use launchd)."""
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    fake_uname = shim_dir / "uname"
    fake_uname.write_text('#!/bin/bash\necho "Darwin"\n', encoding="utf-8")
    fake_uname.chmod(0o755)
    result = _run(
        "--install",
        env_overrides={"PATH": f"{shim_dir}:{os.environ.get('PATH', '')}"},
    )
    assert result.returncode == 1
    combined = (result.stdout + result.stderr).lower()
    assert "darwin" in combined or "macos" in combined
    assert "launchd" in combined


@requires_bash
def test_dry_run_does_not_write_unit_files(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    result = _run(env_overrides={"HOME": str(fake_home)})
    assert result.returncode == 0, result.stderr
    assert not (fake_home / ".config" / "systemd").exists()


@requires_bash
def test_installer_passes_bash_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
