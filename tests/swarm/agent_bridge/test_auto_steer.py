"""Tests for the auto-steer recommendation core (pure, deterministic)."""

from __future__ import annotations

from aragora.swarm.agent_bridge.auto_steer import SteerSignals, build_recommendation
from aragora.swarm.agent_bridge.codex_steer import STEERABLE_FORBIDDEN_ACTIONS

TS = "2026-06-16T00:00:00Z"


def test_no_conditions_recommends_nothing() -> None:
    rec = build_recommendation(SteerSignals(issued_at=TS, open_codex_prs=10, backlog_threshold=140))
    assert rec.directive is None
    assert rec.rationale


def test_backlog_over_threshold_forbids_create_pr() -> None:
    rec = build_recommendation(
        SteerSignals(issued_at=TS, open_codex_prs=156, backlog_threshold=140)
    )
    assert rec.directive is not None
    assert "create_pr" in rec.directive.add_forbidden_actions
    assert rec.directive.note and "DRAIN" in rec.directive.note.upper()


def test_backlog_at_threshold_is_inclusive() -> None:
    rec = build_recommendation(
        SteerSignals(issued_at=TS, open_codex_prs=140, backlog_threshold=140)
    )
    assert rec.directive is not None and "create_pr" in rec.directive.add_forbidden_actions


def test_stale_ledger_adds_prune_note_only() -> None:
    rec = build_recommendation(
        SteerSignals(issued_at=TS, open_codex_prs=5, stale_ledger_prs=(8424, 8406, 8406))
    )
    assert rec.directive is not None
    assert not rec.directive.add_forbidden_actions  # note-only, no forbidden action
    assert rec.directive.note and "prune" in rec.directive.note.lower()
    assert "8406" in rec.directive.note and "8424" in rec.directive.note


def test_claude_owned_pins_off_limits_deduped_sorted() -> None:
    rec = build_recommendation(
        SteerSignals(issued_at=TS, open_codex_prs=5, claude_owned_prs=(8459, 8458, 8459))
    )
    assert rec.directive is not None
    assert rec.directive.off_limits_prs == [8458, 8459]


def test_combined_signals_compose_one_directive() -> None:
    rec = build_recommendation(
        SteerSignals(
            issued_at=TS,
            open_codex_prs=160,
            backlog_threshold=140,
            stale_ledger_prs=(8424,),
            claude_owned_prs=(8460,),
        )
    )
    assert rec.directive is not None
    assert "create_pr" in rec.directive.add_forbidden_actions
    assert rec.directive.off_limits_prs == [8460]
    assert rec.directive.note and "8424" in rec.directive.note
    assert len(rec.rationale) >= 3


def test_recommendation_only_uses_steerable_vocabulary() -> None:
    # Whatever the recommender emits must be inside the safe vocabulary (it cannot
    # smuggle a permissive token) -- enforced by SteeringDirective construction.
    rec = build_recommendation(SteerSignals(issued_at=TS, open_codex_prs=200, backlog_threshold=10))
    assert rec.directive is not None
    assert set(rec.directive.add_forbidden_actions) <= STEERABLE_FORBIDDEN_ACTIONS


def test_negative_pr_numbers_filtered() -> None:
    rec = build_recommendation(
        SteerSignals(
            issued_at=TS,
            open_codex_prs=5,
            claude_owned_prs=(-1, 0, 8461),
            stale_ledger_prs=(-5, 8424),
        )
    )
    assert rec.directive is not None
    assert rec.directive.off_limits_prs == [8461]
    assert "8424" in (rec.directive.note or "") and "-5" not in (rec.directive.note or "")


def test_to_dict_shape() -> None:
    rec = build_recommendation(SteerSignals(issued_at=TS, open_codex_prs=200, backlog_threshold=10))
    d = rec.to_dict()
    assert set(d) == {"directive", "rationale"}
    assert isinstance(d["rationale"], list)
    assert d["directive"] is not None


def test_deterministic() -> None:
    s = SteerSignals(issued_at=TS, open_codex_prs=160, backlog_threshold=140, stale_ledger_prs=(1,))
    assert build_recommendation(s).to_dict() == build_recommendation(s).to_dict()
