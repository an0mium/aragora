from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_live_smoke_dry_run_json_is_side_effect_free(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "agent_bridge_live_smoke.py"
    artifact_dir = tmp_path / "agent-bridge-live-smoke"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
            "--codex-model",
            "codex-test",
            "--droid-auto",
            "high",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["repo_root"] == str(tmp_path.resolve())
    assert payload["artifact_dir"] == str(artifact_dir.resolve())
    assert payload["roles"] == ["claude", "codex", "droid"]
    assert payload["models"]["codex"] == "codex-test"
    assert payload["harness_options"]["droid"] == {"auto": "high"}
    assert payload["turn_count"] == 9
    assert payload["side_effects"] == {
        "artifact_dir_created": False,
        "artifact_written": False,
        "binaries_required": False,
        "agents_launched": False,
    }
    assert not artifact_dir.exists()


@pytest.mark.skipif(
    os.environ.get("ARAGORA_LIVE_AGENT_BRIDGE") != "1",
    reason="Set ARAGORA_LIVE_AGENT_BRIDGE=1 to run live agent bridge smoke tests",
)
def test_live_smoke_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "agent_bridge_live_smoke.py"
    artifact_dir = tmp_path / "agent-bridge-live-smoke"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo",
            str(repo_root),
            "--artifact-dir",
            str(artifact_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    artifacts = sorted(artifact_dir.glob("live-smoke-*.json"))
    assert artifacts, "expected a live smoke artifact"
