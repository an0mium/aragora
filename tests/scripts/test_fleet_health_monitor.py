"""Hermetic tests for scripts/fleet_health_monitor.sh.

The monitor watches the credential slices that produce `blocked_auth_failure`
(runner / github_api), so its alert path is load-bearing for catching silent
benchmark regressions. These tests stub `gh` on PATH and point ARAGORA_REPO at a
probe-less temp dir (so the proof check is skipped) to isolate the runner slice.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR = REPO_ROOT / "scripts" / "fleet_health_monitor.sh"


def _fake_gh(bin_dir: Path, runners: list[dict], *, auth_ok: bool = True) -> None:
    """Write a fake `gh` that answers `api .../runners` and `auth status`."""
    payload = json.dumps({"runners": runners})
    script = f"""#!/usr/bin/env bash
if [ "$1" = "auth" ]; then exit {0 if auth_ok else 1}; fi
if [ "$1" = "api" ]; then
  # honor the --jq the monitor passes by emitting the already-filtered shape
  cat <<'JSON'
{json.dumps([{"name": r["name"], "status": r["status"]} for r in runners])}
JSON
  exit 0
fi
exit 0
"""
    gh = bin_dir / "gh"
    gh.write_text(script)
    gh.chmod(0o755)
    _ = payload  # the monitor uses --jq; fake emits filtered list directly


def _run(tmp_path: Path, runners: list[dict], watch: str, *, auth_ok: bool = True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_gh(bin_dir, runners, auth_ok=auth_ok)
    status_file = tmp_path / "status.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ARAGORA_REPO": str(tmp_path),  # no scripts/probe... here -> proof check skipped
        "FLEET_WATCH_RUNNERS": watch,
        "FLEET_HEALTH_STATUS": str(status_file),
    }
    proc = subprocess.run(
        ["bash", str(MONITOR)], env=env, capture_output=True, text=True, timeout=60
    )
    status = json.loads(status_file.read_text())
    return proc, status


def test_healthy_when_watched_runner_online(tmp_path):
    proc, status = _run(
        tmp_path,
        [{"name": "mac-studio-m3ultra", "status": "online"}],
        "mac-studio-m3ultra",
    )
    assert proc.returncode == 0
    assert status["degraded"] is False
    assert status["offline_runners"] == ""


def test_degraded_and_alerts_when_runner_offline(tmp_path):
    proc, status = _run(
        tmp_path,
        [{"name": "aragora-hetzner-cpu1", "status": "offline"}],
        "aragora-hetzner-cpu1",
    )
    assert proc.returncode == 1
    assert status["degraded"] is True
    assert "aragora-hetzner-cpu1" in status["offline_runners"]
    assert any("blocked_auth_failure" in a for a in status["alerts"])


def test_degraded_when_runner_absent_from_inventory(tmp_path):
    proc, status = _run(tmp_path, [], "mac-studio-m3ultra")
    assert proc.returncode == 1
    assert status["degraded"] is True
    assert "mac-studio-m3ultra(unknown)" in status["offline_runners"]


def test_github_api_slice_flagged_when_unauthenticated(tmp_path):
    proc, status = _run(
        tmp_path,
        [{"name": "mac-studio-m3ultra", "status": "online"}],
        "mac-studio-m3ultra",
        auth_ok=False,
    )
    assert proc.returncode == 1
    assert status["github_api"] == "UNAUTHENTICATED"
    assert any("GITHUB_API" in a for a in status["alerts"])
