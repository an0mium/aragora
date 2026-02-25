"""Tests for epistemic outcome ledger persistence."""

from __future__ import annotations

from aragora.debate.epistemic_outcomes import EpistemicOutcome, EpistemicOutcomeStore


def test_record_and_get_outcome_roundtrip(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    outcome = EpistemicOutcome(
        debate_id="debate-1",
        claim="System uptime will exceed 99.9%.",
        falsifier="Uptime drops below 99.9% during review horizon.",
        metric="30-day uptime percentage",
        review_horizon_days=30,
        resolver_type="oracle",
        initial_confidence=0.82,
        metadata={"source": "test"},
    )

    store.record_outcome(outcome)
    stored = store.get_outcome("debate-1")

    assert stored is not None
    assert stored.debate_id == "debate-1"
    assert stored.metric == "30-day uptime percentage"
    assert stored.metadata["source"] == "test"
    assert stored.status == "open"


def test_list_outcomes_supports_status_filter(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    store.record_outcome(
        EpistemicOutcome(
            debate_id="debate-open",
            claim="A",
            falsifier="B",
            metric="C",
            status="open",
        )
    )
    store.record_outcome(
        EpistemicOutcome(
            debate_id="debate-resolved",
            claim="X",
            falsifier="Y",
            metric="Z",
            status="resolved",
        )
    )

    open_items = store.list_outcomes(status="open")
    assert len(open_items) == 1
    assert open_items[0].debate_id == "debate-open"


def test_resolve_outcome_updates_resolution_fields(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    store.record_outcome(
        EpistemicOutcome(
            debate_id="debate-2",
            claim="Claim",
            falsifier="Falsifier",
            metric="Metric",
            status="open",
            initial_confidence=0.6,
        )
    )

    updated = store.resolve_outcome(
        "debate-2",
        resolved_truth=True,
        confidence_delta=0.2,
        resolver_type="human_panel",
        metadata={"review_id": "rvw-1"},
    )

    assert updated is True
    resolved = store.get_outcome("debate-2")
    assert resolved is not None
    assert resolved.status == "resolved"
    assert resolved.resolved_truth is True
    assert resolved.confidence_delta == 0.2
    assert resolved.resolver_type == "human_panel"
    assert resolved.metadata["review_id"] == "rvw-1"
    assert resolved.resolved_at is not None


def test_resolve_missing_outcome_returns_false(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    assert store.resolve_outcome("missing", resolved_truth=False) is False
