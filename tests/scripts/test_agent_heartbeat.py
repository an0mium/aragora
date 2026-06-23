"""Tests for ``scripts/agent_heartbeat.py``."""

from __future__ import annotations

import importlib.util
import json
import shlex
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


def test_heartbeat_renewal_updates_existing_owner_row(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"

    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-renewal",
        owner_session="codex-Q612",
        pid=111,
        branch="codex/old",
        last_seen_at="2026-06-23T10:00:00Z",
    )
    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-renewal",
        owner_session="codex-Q612",
        pid=222,
        branch="codex/new",
        last_seen_at="2026-06-23T10:05:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert payload[0]["pid"] == 222
    assert payload[0]["branch"] == "codex/new"
    assert payload[0]["last_seen_at"] == "2026-06-23T10:05:00Z"


def test_heartbeat_rejects_path_traversal_owner(tmp_path: Path) -> None:
    for owner_session in ("../escape", ".", "..", ".hidden", "owner:session"):
        with pytest.raises(ValueError, match="unsafe owner_session"):
            heartbeat.record_heartbeat(
                heartbeat_path=tmp_path / "heartbeats.json",
                lane_id="P106",
                owner_session=owner_session,
            )


def test_finalizer_receipt_appends_terminal_owner_lifecycle(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"

    receipt = heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="thread-456",
        pid=34567,
        cwd="/tmp/aragora",
        worktree="/tmp/aragora/.worktrees/codex-q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        pr_number=8560,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    assert receipt == {
        "schema_version": "aragora-agent-finalizer-receipt/1.0",
        "lane_id": "Q612-heartbeat-finalizer",
        "owner_session": "codex-Q612",
        "thread_id": "thread-456",
        "pid": 34567,
        "cwd": "/tmp/aragora",
        "worktree": "/tmp/aragora/.worktrees/codex-q612",
        "branch": "codex/lane-heartbeat-finalizer-20260623",
        "pr_number": 8560,
        "outcome": "completed",
        "reason": "published draft PR",
        "finalized_at": "2026-06-23T10:10:00Z",
    }
    payload = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert payload == [receipt]


def test_finalizer_receipt_marks_matching_heartbeat_terminal(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["last_seen_at"] == "2026-06-23T10:09:00Z"
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["terminal_reason"] == "published draft PR"
    assert payload[0]["terminal_finalized_at"] == "2026-06-23T10:10:00Z"


def test_finalizer_receipt_rejects_unknown_outcome(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outcome must be one of"):
        heartbeat.record_finalizer_receipt(
            heartbeat_path=tmp_path / "heartbeats.json",
            receipt_path=tmp_path / "finalizer-receipts.jsonl",
            lane_id="Q612",
            owner_session="codex-Q612",
            outcome="maybe",
            reason="not terminal",
        )


def test_cli_finalize_writes_to_shared_state_root_from_stateless_repo(
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
            "--finalize",
            "--repo-root",
            str(repo_root),
            "--lane-id",
            "Q612-heartbeat-finalizer",
            "--owner-session",
            "codex-Q612",
            "--outcome",
            "completed",
            "--reason",
            "published draft PR",
            "--finalized-at",
            "2026-06-23T10:10:00Z",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt_path = shared_root / ".aragora" / "agent-bridge" / "heartbeat-finalizer-receipts.jsonl"
    payload = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert json.loads(result.stdout)["outcome"] == "completed"
    assert payload[0]["lane_id"] == "Q612-heartbeat-finalizer"
    assert payload[0]["reason"] == "published draft PR"
    assert not (repo_root / ".aragora").exists()


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


def test_cli_help_is_pipe_safe_when_downstream_closes() -> None:
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT_PATH))} --help | head -5"

    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert "usage: agent_heartbeat.py" in result.stdout
    assert "BrokenPipeError" not in result.stderr
