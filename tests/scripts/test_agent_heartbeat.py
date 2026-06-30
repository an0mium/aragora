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


def test_heartbeat_accepts_dotted_owner_session_slug(tmp_path: Path) -> None:
    row = heartbeat.record_heartbeat(
        heartbeat_path=tmp_path / "heartbeats.json",
        lane_id="Q612",
        owner_session="task.v2",
    )

    assert row["owner_session"] == "task.v2"


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
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
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
    assert payload[0]["terminal_receipt_recorded"] is True


def test_finalizer_receipt_matches_stable_identity_despite_different_cwd(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=34567,
        cwd="/tmp/launcher-cwd",
        worktree="/tmp/aragora/.worktrees/q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        cwd="/tmp/manual-finalizer-cwd",
        worktree="/tmp/aragora/.worktrees/q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["terminal_receipt_recorded"] is True


def test_pidless_finalizer_with_matching_thread_marks_pid_bearing_heartbeat_terminal(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        outcome="completed",
        reason="wrapper-run-1 completed",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["terminal_receipt_recorded"] is True


def test_pidless_finalizer_with_branch_only_keeps_pid_bearing_heartbeat_live(
    tmp_path: Path,
) -> None:
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
        branch="codex/lane-heartbeat-finalizer-20260623",
        outcome="completed",
        reason="branch-only finalizer is not run identity",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["pid"] == 34567
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]


def test_threadless_pidless_finalizer_with_branch_only_keeps_threaded_pid_heartbeat_live(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        outcome="completed",
        reason="branch-only finalizer is not run identity",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["thread_id"] == "wrapper-run-1"
    assert payload[0]["pid"] == 34567
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]


def test_threadless_pidless_finalizer_with_branch_only_marks_pidless_heartbeat_terminal(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        branch="codex/lane-heartbeat-finalizer-20260623",
        outcome="completed",
        reason="pidless heartbeat can accept pidless branch finalizer",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert "pid" not in payload[0]
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"


def test_finalizer_receipt_rejects_conflicting_stable_identity(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=34567,
        cwd="/tmp/shared-cwd",
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        cwd="/tmp/shared-cwd",
        branch="codex/different-lane",
        outcome="completed",
        reason="different branch finished",
        finalized_at="2026-06-23T10:10:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["pid"] == 34567
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]


def test_heartbeat_late_renewal_preserves_terminal_row_for_same_wrapper_pid(
    tmp_path: Path,
) -> None:
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
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=34567,
        branch="codex/resurrected",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["pid"] == 34567
    assert payload[0]["branch"] == "codex/lane-heartbeat-finalizer-20260623"
    assert payload[0]["last_seen_at"] == "2026-06-23T10:09:00Z"


def test_heartbeat_relaunch_replaces_terminal_row_for_new_wrapper_pid(tmp_path: Path) -> None:
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
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=99999,
        branch="codex/relaunched",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]
    assert payload[0]["pid"] == 99999
    assert payload[0]["branch"] == "codex/relaunched"
    assert payload[0]["last_seen_at"] == "2026-06-23T10:11:00Z"


def test_heartbeat_relaunch_replaces_terminal_row_for_new_wrapper_id_even_same_pid(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )
    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=34567,
        branch="codex/relaunched",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]
    assert payload[0]["thread_id"] == "wrapper-run-2"
    assert payload[0]["pid"] == 34567
    assert payload[0]["branch"] == "codex/relaunched"


def test_threadless_relaunch_replaces_threaded_terminal_row_when_identity_differs(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        cwd="/tmp/old-worktree",
        branch="codex/old",
        last_seen_at="2026-06-23T10:09:00Z",
    )
    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/relaunched",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]
    assert "thread_id" not in payload[0]
    assert payload[0]["pid"] == 45678
    assert payload[0]["cwd"] == "/tmp/new-worktree"
    assert payload[0]["branch"] == "codex/relaunched"
    assert payload[0]["superseded_thread_ids"] == ["wrapper-run-1"]


def test_threadless_late_heartbeat_preserves_threaded_terminal_row_when_identity_matches(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        cwd="/tmp/worktree",
        branch="codex/old",
        last_seen_at="2026-06-23T10:09:00Z",
    )
    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=34567,
        cwd="/tmp/worktree",
        branch="codex/old",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["thread_id"] == "wrapper-run-1"
    assert payload[0]["last_seen_at"] == "2026-06-23T10:09:00Z"


def test_stale_wrapper_heartbeat_does_not_overwrite_successor(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/old",
        last_seen_at="2026-06-23T10:09:00Z",
    )
    successor = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=45678,
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    stale = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/old-stale",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [successor]
    assert stale == successor
    assert payload[0]["thread_id"] == "wrapper-run-2"
    assert payload[0]["branch"] == "codex/new"
    assert payload[0]["superseded_thread_ids"] == ["wrapper-run-1"]


def test_pidless_newer_heartbeat_replaces_threaded_wrapper_when_identity_differs(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        cwd="/tmp/old-worktree",
        branch="codex/old",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert "thread_id" not in payload[0]
    assert payload[0]["pid"] == 45678
    assert payload[0]["cwd"] == "/tmp/new-worktree"
    assert payload[0]["branch"] == "codex/new"
    assert payload[0]["superseded_thread_ids"] == ["wrapper-run-1"]


def test_pidless_late_heartbeat_preserves_threaded_successor_when_identity_matches(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    successor = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    stale = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [successor]
    assert stale == successor
    assert payload[0]["thread_id"] == "wrapper-run-2"


def test_finalizer_ignores_newer_relaunch_identity(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/old",
        last_seen_at="2026-06-23T10:09:00Z",
    )
    successor = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=45678,
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        outcome="cancelled",
        reason="old wrapper was killed",
        finalized_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [successor]
    assert "terminal" not in payload[0]
    assert json.loads(receipt_path.read_text(encoding="utf-8").splitlines()[0])["thread_id"] == (
        "wrapper-run-1"
    )


def test_threadless_finalizer_does_not_terminalize_threaded_successor(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    successor = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    receipt = heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        outcome="cancelled",
        reason="legacy finalizer arrived late",
        finalized_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [successor]
    assert "terminal" not in payload[0]
    assert receipt["lane_id"] == "Q612-heartbeat-finalizer"
    assert json.loads(receipt_path.read_text(encoding="utf-8").splitlines()[0])["reason"] == (
        "legacy finalizer arrived late"
    )


def test_threadless_finalizer_marks_matching_threadless_heartbeat_terminal(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    heartbeat.record_finalizer_receipt(
        heartbeat_path=heartbeat_path,
        receipt_path=receipt_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        cwd="/tmp/new-worktree",
        branch="codex/new",
        outcome="completed",
        reason="threadless worker finished",
        finalized_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["terminal_receipt_recorded"] is True


def test_cli_finalize_without_pid_keeps_pid_bearing_threadless_heartbeat_live(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--finalize",
            "--heartbeat-path",
            str(heartbeat_path),
            "--finalizer-receipt-path",
            str(receipt_path),
            "--lane-id",
            "Q612-heartbeat-finalizer",
            "--owner-session",
            "codex-Q612",
            "--outcome",
            "completed",
            "--reason",
            "threadless worker finished",
            "--finalized-at",
            "2026-06-23T10:11:00Z",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(result.stdout)
    assert "pid" not in receipt
    assert "thread_id" not in receipt
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["pid"] == 45678
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]


def test_cli_finalize_without_pid_marks_pidless_threadless_heartbeat_terminal(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=None,
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--finalize",
            "--heartbeat-path",
            str(heartbeat_path),
            "--finalizer-receipt-path",
            str(receipt_path),
            "--lane-id",
            "Q612-heartbeat-finalizer",
            "--owner-session",
            "codex-Q612",
            "--outcome",
            "completed",
            "--reason",
            "pidless worker finished",
            "--finalized-at",
            "2026-06-23T10:11:00Z",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(result.stdout)
    assert "pid" not in receipt
    assert "thread_id" not in receipt
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert "pid" not in payload[0]
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["terminal_receipt_recorded"] is True


def test_cli_finalize_with_explicit_mismatched_pid_keeps_threadless_heartbeat_live(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=45678,
        branch="codex/new",
        last_seen_at="2026-06-23T10:10:00Z",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--finalize",
            "--heartbeat-path",
            str(heartbeat_path),
            "--finalizer-receipt-path",
            str(receipt_path),
            "--lane-id",
            "Q612-heartbeat-finalizer",
            "--owner-session",
            "codex-Q612",
            "--pid",
            "99999",
            "--outcome",
            "completed",
            "--reason",
            "different worker finished",
            "--finalized-at",
            "2026-06-23T10:11:00Z",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["pid"] == 45678
    assert "terminal" not in payload[0]
    assert "terminal_outcome" not in payload[0]


def test_pidless_heartbeat_preserves_terminal_row(tmp_path: Path) -> None:
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
        pid=34567,
        outcome="completed",
        reason="published draft PR",
        finalized_at="2026-06-23T10:10:00Z",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        pid=None,
        branch="codex/pidless-resurrection",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "completed"
    assert payload[0]["pid"] == 34567
    assert payload[0]["branch"] == "codex/lane-heartbeat-finalizer-20260623"


def test_terminal_row_without_identity_allows_new_thread_relaunch(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    heartbeat_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": "aragora-agent-heartbeat/1.0",
                    "lane_id": "Q612-heartbeat-finalizer",
                    "owner_session": "codex-Q612",
                    "terminal_finalized_at": "2026-06-23T10:10:00Z",
                    "last_seen_at": "2026-06-23T10:09:00Z",
                    "branch": "codex/lane-heartbeat-finalizer-20260623",
                }
            ]
        ),
        encoding="utf-8",
    )

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-2",
        pid=45678,
        branch="codex/relaunched",
        last_seen_at="2026-06-23T10:11:00Z",
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]
    assert payload[0]["thread_id"] == "wrapper-run-2"
    assert payload[0]["branch"] == "codex/relaunched"
    assert "terminal_finalized_at" not in payload[0]


def test_finalizer_marks_heartbeat_terminal_before_receipt_append_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"
    receipt_path = tmp_path / "finalizer-receipts.jsonl"
    heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="Q612-heartbeat-finalizer",
        owner_session="codex-Q612",
        thread_id="wrapper-run-1",
        pid=34567,
        branch="codex/lane-heartbeat-finalizer-20260623",
        last_seen_at="2026-06-23T10:09:00Z",
    )

    def fail_append(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("receipt path denied")

    monkeypatch.setattr(heartbeat, "_append_jsonl", fail_append)

    with pytest.raises(PermissionError, match="receipt path denied"):
        heartbeat.record_finalizer_receipt(
            heartbeat_path=heartbeat_path,
            receipt_path=receipt_path,
            lane_id="Q612-heartbeat-finalizer",
            owner_session="codex-Q612",
            thread_id="wrapper-run-1",
            pid=34567,
            outcome="failed",
            reason="tmux launcher command exited with status 7",
            finalized_at="2026-06-23T10:10:00Z",
        )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload[0]["terminal"] is True
    assert payload[0]["terminal_outcome"] == "failed"
    assert payload[0]["terminal_receipt_recorded"] is False
    assert "PermissionError: receipt path denied" in payload[0]["terminal_receipt_error"]
    assert payload[0]["terminal_finalized_at"] == "2026-06-23T10:10:00Z"
    assert payload[0]["last_seen_at"] == "2026-06-23T10:09:00Z"


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
