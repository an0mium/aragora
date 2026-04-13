"""Tests for scripts/agent_bridge_supervise.py."""

from __future__ import annotations

import json
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
