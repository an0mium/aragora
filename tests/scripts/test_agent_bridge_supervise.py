"""Tests for scripts/agent_bridge_supervise.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_decide_lane_requests_approval_prompt() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="active",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
    )
    session = mod.agent_bridge.Session(
        name="codex-bridge",
        agent="codex",
        status="alive",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
        summary="Waiting for permission approval before running the next command",
    )

    decision = mod._decide_lane(
        record,
        session,
        session.summary,
        mod.WorktreeStatus(state="clean"),
        None,
    )

    assert decision.next_action == "approve_prompt"
    assert "permission prompt" in decision.reason


def test_decide_lane_waits_for_ci_when_checks_pending() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="active",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
    )
    session = mod.agent_bridge.Session(
        name="codex-bridge",
        agent="codex",
        status="alive",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
        summary="Opened the PR and waiting for checks",
    )
    pr_truth = mod.PRTruth(
        branch="codex/issue-5322",
        number=5402,
        url="https://github.com/synaptent/aragora/pull/5402",
        checks_bucket="pending",
    )

    decision = mod._decide_lane(
        record,
        session,
        session.summary,
        mod.WorktreeStatus(state="clean"),
        pr_truth,
    )

    assert decision.next_action == "wait_for_ci"
    assert decision.pr_number == 5402


def test_decide_lane_marks_conflict_blocked() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="conflict",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
        conflict_reason="conflicting active owner claim from claude-bridge",
    )
    session = mod.agent_bridge.Session(
        name="codex-bridge",
        agent="codex",
        status="alive",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
    )

    decision = mod._decide_lane(
        record,
        session,
        "",
        mod.WorktreeStatus(state="clean"),
        None,
    )

    assert decision.next_action == "blocked"
    assert "ambiguous" in decision.reason


def test_decide_lane_restarts_when_owner_missing() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="active",
        branch="codex/issue-5322",
        worktree="/tmp/missing-bridge",
    )

    decision = mod._decide_lane(
        record,
        None,
        "",
        mod.WorktreeStatus(state="missing", evidence=["worktree missing"]),
        None,
    )

    assert decision.next_action == "restart_from_main"
    assert "missing from the live session registry" in decision.reason


def test_decide_lane_ready_for_review_when_checks_pass() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="active",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
    )
    session = mod.agent_bridge.Session(
        name="codex-bridge",
        agent="codex",
        status="alive",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
        summary="Still waiting for CI even though the PR is already open",
    )
    pr_truth = mod.PRTruth(
        branch="codex/issue-5322",
        number=5402,
        url="https://github.com/synaptent/aragora/pull/5402",
        checks_bucket="pass",
    )

    decision = mod._decide_lane(
        record,
        session,
        session.summary,
        mod.WorktreeStatus(state="clean"),
        pr_truth,
    )

    assert decision.next_action == "ready_for_review"
    assert "checks passed" in decision.reason


def test_decide_lane_followup_when_worktree_dirty_even_if_checks_pass() -> None:
    import agent_bridge_supervise as mod

    record = mod.agent_bridge.LaneRecord(
        lane_id="bridge-hardening",
        owner_session="codex-bridge",
        status="active",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
    )
    session = mod.agent_bridge.Session(
        name="codex-bridge",
        agent="codex",
        status="alive",
        branch="codex/issue-5322",
        worktree="/tmp/bridge",
        summary="PR is green and ready for review",
    )
    pr_truth = mod.PRTruth(
        branch="codex/issue-5322",
        number=5402,
        url="https://github.com/synaptent/aragora/pull/5402",
        checks_bucket="pass",
    )

    decision = mod._decide_lane(
        record,
        session,
        session.summary,
        mod.WorktreeStatus(
            state="dirty",
            dirty=True,
            evidence=["git status unavailable (command timed out)"],
        ),
        pr_truth,
    )

    assert decision.next_action == "send_followup"
    assert "worktree has local changes" in decision.reason
    assert decision.pr_number == 5402


def test_inspect_worktree_treats_git_status_failure_as_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge_supervise as mod

    def fake_run_git(_worktree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args == ("status", "--short"):
            return subprocess.CompletedProcess(["git", "status"], 128, "", "fatal: bad index")
        if args == ("rev-list", "--left-right", "--count", "origin/main...HEAD"):
            return subprocess.CompletedProcess(["git", "rev-list"], 0, "0 0\n", "")
        raise AssertionError(args)

    monkeypatch.setattr(mod, "_run_git", fake_run_git)

    status = mod._inspect_worktree(str(tmp_path))

    assert status.dirty is True
    assert status.state == "dirty"
    assert status.evidence == ["git status unavailable (fatal: bad index)"]


def test_run_git_returns_degraded_process_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge_supervise as mod

    def raise_timeout(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["git", "status"], timeout=kwargs["timeout"])

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)

    proc = mod._run_git(tmp_path, "status")

    assert proc.returncode == 124
    assert "command timed out after 10s" in proc.stderr


def test_main_once_json_renders_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge_supervise as mod

    snapshot = mod.SupervisorSnapshot(
        generated_at="2026-04-13T22:30:00+00:00",
        decisions=[
            mod.LaneDecision(
                lane_id="bridge-hardening",
                owner_session="codex-bridge",
                status="active",
                next_action="send_followup",
                reason="Need one bounded follow-up prompt",
            )
        ],
        warnings=["gh unavailable"],
    )
    monkeypatch.setattr(mod, "collect_supervisor_snapshot", lambda: snapshot)

    rc = mod.main(["--once", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lanes"][0]["next_action"] == "send_followup"
    assert payload["warnings"] == ["gh unavailable"]


def test_collect_supervisor_snapshot_caps_records_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge_supervise as mod

    sessions = [
        mod.agent_bridge.Session(
            name="codex-one",
            agent="codex",
            status="alive",
            lifecycle="live",
            worktree="/tmp/one",
        ),
        mod.agent_bridge.Session(
            name="codex-two",
            agent="codex",
            status="alive",
            lifecycle="live",
            worktree="/tmp/two",
        ),
    ]
    records = [
        mod.agent_bridge.LaneRecord(lane_id="one", owner_session="codex-one", status="active"),
        mod.agent_bridge.LaneRecord(lane_id="two", owner_session="codex-two", status="active"),
    ]
    monkeypatch.setattr(mod, "_discover_current_sessions", lambda: sessions)
    monkeypatch.setattr(mod.agent_bridge, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod.agent_bridge, "_load_lane_registry", lambda: records)
    monkeypatch.setattr(mod.agent_bridge, "_sync_lane_records", lambda loaded, _sessions: loaded)
    monkeypatch.setattr(mod, "_load_pr_truth", lambda _records: ({}, []))
    monkeypatch.setattr(
        mod,
        "_inspect_worktree",
        lambda _path: mod.WorktreeStatus(state="clean"),
    )

    snapshot = mod.collect_supervisor_snapshot(max_records=1)

    assert len(snapshot.decisions) == 1
    assert snapshot.warnings == ["supervisor record cap applied: 1/2"]


def test_collect_supervisor_snapshot_treats_active_broker_session_as_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge_supervise as mod

    session = mod.agent_bridge.Session(
        name="droid-broker",
        agent="droid",
        status="active_broker",
        lifecycle="active_broker",
        branch="codex/bridge",
        worktree=str(tmp_path),
        session_id="broker-session",
    )
    records = [
        mod.agent_bridge.LaneRecord(
            lane_id="bridge-current",
            owner_session="droid-broker",
            status="active",
            branch="codex/bridge",
            worktree=str(tmp_path),
        )
    ]
    pr_truth = mod.PRTruth(
        branch="codex/bridge",
        number=7190,
        url="https://github.com/synaptent/aragora/pull/7190",
        is_draft=False,
        checks_bucket="pass",
    )
    monkeypatch.setattr(
        mod.agent_bridge,
        "_discover_with_broker_state",
        lambda **_kwargs: ([session], [{"run_id": "broker-run"}], {"broker-session"}),
    )
    monkeypatch.setattr(mod.agent_bridge, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod.agent_bridge, "_load_lane_registry", lambda: records)
    monkeypatch.setattr(mod.agent_bridge, "_sync_lane_records", lambda loaded, _sessions: loaded)
    monkeypatch.setattr(mod, "_load_pr_truth", lambda _records: ({"codex/bridge": pr_truth}, []))
    monkeypatch.setattr(
        mod,
        "_inspect_worktree",
        lambda _path: mod.WorktreeStatus(state="clean"),
    )

    snapshot = mod.collect_supervisor_snapshot()

    assert len(snapshot.decisions) == 1
    assert snapshot.decisions[0].next_action == "ready_for_review"
    assert "owner session is active_broker" not in snapshot.decisions[0].reason


def test_collect_supervisor_snapshot_fails_open_on_discovery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge_supervise as mod

    def explode() -> list[mod.agent_bridge.Session]:
        raise RuntimeError("tmux inventory unavailable")

    monkeypatch.setattr(mod, "_discover_current_sessions", explode)

    snapshot = mod.collect_supervisor_snapshot()

    assert snapshot.decisions == []
    assert snapshot.warnings == ["session discovery degraded: tmux inventory unavailable"]
