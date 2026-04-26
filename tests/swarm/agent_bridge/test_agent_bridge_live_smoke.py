from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


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
