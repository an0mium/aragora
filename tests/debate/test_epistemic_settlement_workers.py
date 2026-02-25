"""Tests for epistemic settlement workers."""

from __future__ import annotations

from aragora.debate.epistemic_outcomes import EpistemicOutcome, EpistemicOutcomeStore
from aragora.debate.epistemic_settlement_workers import EpistemicSettlementCoordinator


def test_deterministic_worker_resolves_from_test_counts(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    store.record_outcome(
        EpistemicOutcome(
            debate_id="det-1",
            claim="Claim",
            falsifier="Falsifier",
            metric="Metric",
            status="pending_deterministic",
            resolver_type="deterministic",
            metadata={"tests_passed": 42, "tests_failed": 0},
        )
    )

    coordinator = EpistemicSettlementCoordinator(store)
    report = coordinator.run_once(limit_per_tier=20)

    assert report["deterministic"]["resolved"] == 1
    resolved = store.get_outcome("det-1")
    assert resolved is not None
    assert resolved.status == "resolved"
    assert resolved.resolved_truth is True
    assert resolved.resolver_type == "deterministic_worker"


def test_oracle_worker_resolves_from_oracle_signal(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    store.record_outcome(
        EpistemicOutcome(
            debate_id="oracle-1",
            claim="Claim",
            falsifier="Falsifier",
            metric="Metric",
            status="pending_oracle",
            resolver_type="oracle",
            metadata={"oracle_signal": "diverged"},
        )
    )

    coordinator = EpistemicSettlementCoordinator(store)
    report = coordinator.run_once(limit_per_tier=20)

    assert report["oracle"]["resolved"] == 1
    resolved = store.get_outcome("oracle-1")
    assert resolved is not None
    assert resolved.status == "resolved"
    assert resolved.resolved_truth is False
    assert resolved.resolver_type == "oracle_worker"


def test_workers_skip_when_no_resolution_signal(tmp_path):
    store = EpistemicOutcomeStore(db_path=tmp_path / "epistemic_outcomes.db")
    store.record_outcome(
        EpistemicOutcome(
            debate_id="det-skip",
            claim="Claim",
            falsifier="Falsifier",
            metric="Metric",
            status="pending_deterministic",
            resolver_type="deterministic",
            metadata={},
        )
    )
    store.record_outcome(
        EpistemicOutcome(
            debate_id="oracle-skip",
            claim="Claim",
            falsifier="Falsifier",
            metric="Metric",
            status="pending_oracle",
            resolver_type="oracle",
            metadata={},
        )
    )

    coordinator = EpistemicSettlementCoordinator(store)
    report = coordinator.run_once(limit_per_tier=20)

    assert report["total"]["resolved"] == 0
    assert report["total"]["skipped"] == 2
    assert store.get_outcome("det-skip").status == "pending_deterministic"  # type: ignore[union-attr]
    assert store.get_outcome("oracle-skip").status == "pending_oracle"  # type: ignore[union-attr]
