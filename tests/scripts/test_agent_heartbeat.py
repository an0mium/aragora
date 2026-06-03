"""Tests for ``scripts/agent_heartbeat.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


heartbeat = _load_module("agent_heartbeat.py")
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "agent_heartbeat.py"


def test_heartbeat_upserts_owner_identity(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="P106-merge-gate-settlement",
        owner_session="droid-P106-merge-gate-settlement-20260521T2118Z",
        thread_id="thread-123",
        pid=12345,
        cwd="/tmp/aragora",
        worktree="/tmp/aragora",
        branch="claude/recover-merge-gate-reconciliation",
        pr_number=7423,
        last_seen_at="2026-05-21T23:00:00Z",
    )

    assert row["lane_id"] == "P106-merge-gate-settlement"
    assert row["owner_session"] == "droid-P106-merge-gate-settlement-20260521T2118Z"
    assert row["last_seen_at"] == "2026-05-21T23:00:00Z"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]


def test_heartbeat_rejects_path_traversal_owner(tmp_path: Path) -> None:
    for owner_session in ("../escape", ".", "..", ".hidden", "owner:session"):
        with pytest.raises(ValueError, match="unsafe owner_session"):
            heartbeat.record_heartbeat(
                heartbeat_path=tmp_path / "heartbeats.json",
                lane_id="P106",
                owner_session=owner_session,
            )


def test_concurrent_heartbeat_writes_preserve_all_rows(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--heartbeat-path",
                str(heartbeat_path),
                "--lane-id",
                f"lane-{idx:02d}",
                "--owner-session",
                f"owner-{idx:02d}",
                "--last-seen-at",
                "2026-05-22T00:00:00Z",
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for idx in range(12)
    ]
    results = [proc.communicate(timeout=30) + (proc.returncode,) for proc in procs]

    assert all(returncode == 0 for _stdout, _stderr, returncode in results), results
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert sorted(row["lane_id"] for row in payload) == [f"lane-{idx:02d}" for idx in range(12)]


def test_resolve_heartbeat_path_prefers_repo_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_bridge = repo_root / ".aragora" / "agent-bridge"
    shared_root = tmp_path / "shared"
    repo_bridge.mkdir(parents=True)
    (shared_root / ".aragora" / "agent-bridge").mkdir(parents=True)
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared_root))

    resolved = heartbeat.resolve_heartbeat_path(repo_root=repo_root)

    assert resolved == repo_bridge / "heartbeats.json"


def test_resolve_heartbeat_path_uses_env_checkout_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    shared_root = tmp_path / "shared"
    (shared_root / ".aragora" / "agent-bridge").mkdir(parents=True)
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared_root))

    resolved = heartbeat.resolve_heartbeat_path(repo_root=repo_root)

    assert resolved == shared_root / ".aragora" / "agent-bridge" / "heartbeats.json"


def test_resolve_heartbeat_path_accepts_direct_env_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    shared_state = tmp_path / "shared" / ".aragora"
    (shared_state / "agent-bridge").mkdir(parents=True)
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared_state))

    resolved = heartbeat.resolve_heartbeat_path(repo_root=repo_root)

    assert resolved == shared_state / "agent-bridge" / "heartbeats.json"


def test_cli_writes_to_shared_state_root_from_stateless_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    shared_root = tmp_path / "shared"
    (shared_root / ".aragora" / "agent-bridge").mkdir(parents=True)
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared_root))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo_root),
            "--lane-id",
            "Q245-primary-agent-heartbeat-shared-state",
            "--owner-session",
            "engineering-autopilot-Q245",
            "--last-seen-at",
            "2026-06-01T23:00:00Z",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    heartbeat_path = shared_root / ".aragora" / "agent-bridge" / "heartbeats.json"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert json.loads(result.stdout)["lane_id"] == "Q245-primary-agent-heartbeat-shared-state"
    assert payload[0]["owner_session"] == "engineering-autopilot-Q245"
    assert not (repo_root / ".aragora").exists()
