"""Tests for ``aragora/swarm/loop_control.py`` (pure classifier + halt audit)."""

from __future__ import annotations

import json

import pytest

from aragora.swarm.loop_control import (
    LOOP_SPECS,
    SCHEMA_VERSION,
    HaltGuards,
    HaltVerdict,
    LoopKind,
    LoopState,
    NextAction,
    _is_fault_stop,
    audit_halt_readiness,
    classify_loop,
    summarize,
)

ARBITER = LOOP_SPECS[LoopKind.MERGE_ARBITER]
PROOF = LOOP_SPECS[LoopKind.PROOF_FIRST_SHIFT]
BOSS = LOOP_SPECS[LoopKind.BOSS_LOOP]


def test_waiting_is_not_blocked() -> None:
    rec = classify_loop(ARBITER, {"source_status": "ok", "alive": True, "waiting_only": True})
    assert rec.state == LoopState.WAITING.value
    assert rec.next_action == NextAction.WAIT.value
    assert rec.blocker is None


def test_operational_fault_halts() -> None:
    rec = classify_loop(
        ARBITER,
        {"source_status": "ok", "alive": True, "stop_reason": "GitHub API 500 during evaluate"},
    )
    assert rec.state == LoopState.BLOCKED.value
    assert rec.next_action == NextAction.HALT.value
    assert "GitHub" in (rec.blocker or "")


def test_operational_fault_flag_halts_without_reason() -> None:
    rec = classify_loop(ARBITER, {"source_status": "ok", "alive": True, "operational_fault": True})
    assert rec.state == LoopState.BLOCKED.value
    assert rec.next_action == NextAction.HALT.value


def test_normal_stop_is_not_a_fault() -> None:
    rec = classify_loop(
        ARBITER,
        {"source_status": "ok", "alive": False, "stop_reason": "TimeLimit: max runtime reached"},
    )
    assert rec.state == LoopState.HALTED.value
    assert rec.next_action == NextAction.REPORT_ONLY.value


def test_unknown_is_fail_closed() -> None:
    rec = classify_loop(ARBITER, {"source_status": "unavailable"})
    assert rec.state == LoopState.UNKNOWN.value
    assert rec.next_action == NextAction.REPORT_ONLY.value


def test_budget_exhausted_halts() -> None:
    rec = classify_loop(
        BOSS,
        {
            "source_status": "ok",
            "alive": True,
            "budget": {"remaining_usd": 0.0, "source": "test", "source_status": "ok"},
        },
    )
    assert rec.state == LoopState.BUDGET_EXHAUSTED.value
    assert rec.next_action == NextAction.HALT.value


def test_negative_budget_halts() -> None:
    rec = classify_loop(
        BOSS,
        {
            "source_status": "ok",
            "alive": True,
            "budget": {"remaining_usd": -1.5, "source": "test", "source_status": "ok"},
        },
    )
    assert rec.next_action == NextAction.HALT.value


def test_human_gated_escalates() -> None:
    rec = classify_loop(
        ARBITER,
        {
            "source_status": "ok",
            "alive": True,
            "awaiting_human": True,
            "human_settlement_present": False,
        },
    )
    assert rec.state == LoopState.HUMAN_GATED.value
    assert rec.next_action == NextAction.ESCALATE_HUMAN.value


def test_human_gate_satisfied_does_not_escalate() -> None:
    rec = classify_loop(
        ARBITER,
        {
            "source_status": "ok",
            "alive": True,
            "awaiting_human": True,
            "human_settlement_present": True,
        },
    )
    assert rec.state != LoopState.HUMAN_GATED.value


def test_owner_stale_reports_only() -> None:
    rec = classify_loop(BOSS, {"source_status": "ok", "alive": True, "owner_stale": True})
    assert rec.state == LoopState.STALE_OWNER.value
    assert rec.next_action == NextAction.REPORT_ONLY.value


def test_running_continues() -> None:
    rec = classify_loop(PROOF, {"source_status": "ok", "alive": True})
    assert rec.state == LoopState.RUNNING.value
    assert rec.next_action == NextAction.CONTINUE.value


@pytest.mark.parametrize(
    "reason,is_fault",
    [
        ("max runtime reached", False),
        ("TimeLimit: 12.01h >= max 12.00h", False),
        ("no candidates", False),
        ("shift complete", False),
        ("", False),
        ("circuit breaker tripped", True),
        ("RepeatedAuthFailure", True),
        ("GitHub outage detected", True),
        ("something unrecognized", True),
    ],
)
def test_is_fault_stop(reason: str, is_fault: bool) -> None:
    assert _is_fault_stop(reason) is is_fault


def test_halt_audit_ok() -> None:
    result = audit_halt_readiness(HaltGuards(True, True, True, True))
    assert result.verdict == HaltVerdict.OK.value
    assert result.gaps == []


def test_halt_audit_incomplete_no_budget() -> None:
    result = audit_halt_readiness(HaltGuards(True, True, True, False))
    assert result.verdict == HaltVerdict.INCOMPLETE.value
    assert any("budget" in gap for gap in result.gaps)


def test_halt_audit_incomplete_fault_not_distinguished() -> None:
    result = audit_halt_readiness(HaltGuards(True, True, False, True))
    assert result.verdict == HaltVerdict.INCOMPLETE.value
    assert any("distinguish" in gap for gap in result.gaps)


def test_halt_audit_missing() -> None:
    result = audit_halt_readiness(HaltGuards(False, False, False, False))
    assert result.verdict == HaltVerdict.MISSING.value


def test_summarize_unsafe_when_blocked() -> None:
    blocked = classify_loop(
        ARBITER, {"source_status": "ok", "alive": True, "operational_fault": True}
    )
    waiting = classify_loop(ARBITER, {"source_status": "ok", "alive": True, "waiting_only": True})
    summary = summarize([blocked, waiting])
    assert summary["any_blocked"] is True
    assert summary["fleet_safe_to_continue"] is False


def test_summarize_safe_when_running_or_waiting() -> None:
    running = classify_loop(PROOF, {"source_status": "ok", "alive": True})
    waiting = classify_loop(ARBITER, {"source_status": "ok", "alive": True, "waiting_only": True})
    summary = summarize([running, waiting])
    assert summary["fleet_safe_to_continue"] is True
    assert summary["schema_version"] == SCHEMA_VERSION


def test_schema_version_on_record() -> None:
    rec = classify_loop(PROOF, {"source_status": "ok", "alive": True})
    assert rec.schema_version == SCHEMA_VERSION


def test_record_is_json_serializable() -> None:
    rec = classify_loop(ARBITER, {"source_status": "ok", "alive": True, "waiting_only": True})
    assert "loop-control/v1" in json.dumps(rec.to_dict())


def test_registry_complete_and_merge_arbiter_fault_distinction_retired() -> None:
    for kind in LoopKind:
        assert kind in LOOP_SPECS
    # The #7879 curated gap was retired by PR #8125: the arbiter breaker now
    # trips only on systemic operational faults, never on not-ready PRs.
    assert LOOP_SPECS[LoopKind.MERGE_ARBITER].guards.no_progress_distinguishes_fault is True
    arbiter_audit = audit_halt_readiness(LOOP_SPECS[LoopKind.MERGE_ARBITER].guards)
    assert arbiter_audit.verdict == "incomplete"  # budget ceiling still missing
    assert arbiter_audit.gaps == ["no dollar/budget ceiling (bounded by time/iterations only)"]


def test_source_paths_and_status_propagated() -> None:
    rec = classify_loop(ARBITER, {"source_status": "ok", "alive": True})
    assert rec.source_paths == list(ARBITER.source_paths)
    assert rec.source_status == "ok"
