"""Tests for aragora.reputation.store (AGT-05 ReputationStore)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.reputation.settlement import settle_claim
from aragora.reputation.store import (
    AgentScore,
    ReputationStore,
    ReputationStoreError,
    enable_store,
    store_enabled,
)
from aragora.reputation.types import (
    DOMAIN_DEBATE_POSITION,
    DOMAIN_PREDICTION_MARKET,
    ReputationDelta,
    ResolvedClaim,
    StakeableClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(
    *,
    agent_id: str = "alice",
    probability: float | None = 0.8,
    position: str = "yes",
    stake: int = 100,
    domain: str = DOMAIN_PREDICTION_MARKET,
    resolution_id: str = "mkt_abc",
) -> StakeableClaim:
    return StakeableClaim.create(
        agent_id=agent_id,
        domain=domain,
        statement="PR #42 will be merged within 30 days",
        position=position,
        stake_units=stake,
        resolution_source="synthetic_github",
        resolution_id=resolution_id,
        predicted_probability=probability,
    )


def _resolved(claim: StakeableClaim, outcome: str = "yes") -> ResolvedClaim:
    return ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=outcome,  # type: ignore[arg-type]
        resolved_at="2026-04-24T12:00:00Z",
        resolution_source="synthetic_github",
    )


def _delta(
    *,
    agent_id: str = "alice",
    delta: float = 50.0,
    domain: str = DOMAIN_PREDICTION_MARKET,
    decay_half_life_days: float | None = None,
    applied_at: str | None = None,
    resolution_id: str = "mkt_abc",
) -> ReputationDelta:
    claim = _claim(agent_id=agent_id, domain=domain, resolution_id=resolution_id)
    resolved = _resolved(claim, "yes")
    d = settle_claim(claim, resolved, decay_half_life_days=decay_half_life_days)
    # Return a patched version with the desired delta value for determinism.
    return ReputationDelta(
        delta_id=d.delta_id,
        agent_id=agent_id,
        domain=domain,
        claim_id=d.claim_id,
        resolution_id=d.resolution_id,
        delta=delta,
        scoring_rule=d.scoring_rule,
        applied_at=applied_at or d.applied_at,
        decay_half_life_days=decay_half_life_days,
        reason=d.reason,
    )


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------


class TestFlagHelpers:
    def test_store_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
        assert store_enabled() is False

    def test_enable_store_sets_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
        enable_store()
        assert store_enabled() is True

    def test_store_enabled_via_env_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
        assert store_enabled() is True

    def test_store_enabled_via_true_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "true")
        assert store_enabled() is True


# ---------------------------------------------------------------------------
# record_delta and basic queries
# ---------------------------------------------------------------------------


class TestRecordAndQuery:
    def test_empty_store_len_zero(self) -> None:
        store = ReputationStore()
        assert len(store) == 0

    def test_empty_agent_score_zero(self) -> None:
        store = ReputationStore()
        assert store.get_score("ghost") == 0.0

    def test_record_single_delta(self) -> None:
        store = ReputationStore()
        d = _delta(delta=60.0)
        store.record_delta(d)
        assert len(store) == 1
        assert store.get_score("alice") == pytest.approx(60.0)

    def test_record_multiple_deltas_accumulate(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(agent_id="alice", delta=40.0, resolution_id="r1"))
        store.record_delta(_delta(agent_id="alice", delta=-20.0, resolution_id="r2"))
        assert len(store) == 2
        assert store.get_score("alice") == pytest.approx(20.0)

    def test_multiple_agents_isolated(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(agent_id="alice", delta=80.0))
        store.record_delta(_delta(agent_id="bob", delta=30.0))
        assert store.get_score("alice") == pytest.approx(80.0)
        assert store.get_score("bob") == pytest.approx(30.0)

    def test_agent_ids_sorted(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(agent_id="charlie", delta=10.0))
        store.record_delta(_delta(agent_id="alice", delta=10.0))
        store.record_delta(_delta(agent_id="bob", delta=10.0))
        assert store.agent_ids() == ["alice", "bob", "charlie"]

    def test_deltas_for_returns_copy(self) -> None:
        store = ReputationStore()
        d = _delta(delta=50.0)
        store.record_delta(d)
        history = store.deltas_for("alice")
        assert len(history) == 1
        history.clear()  # mutating the copy must not affect the store
        assert len(store.deltas_for("alice")) == 1

    def test_empty_agent_rejects(self) -> None:
        store = ReputationStore()
        d = _delta()
        # Craft a delta with empty agent_id by bypassing the factory
        bad = ReputationDelta(
            delta_id=d.delta_id,
            agent_id="  ",  # whitespace only
            domain=d.domain,
            claim_id=d.claim_id,
            resolution_id=d.resolution_id,
            delta=d.delta,
            scoring_rule=d.scoring_rule,
            applied_at=d.applied_at,
            decay_half_life_days=d.decay_half_life_days,
            reason=d.reason,
        )
        with pytest.raises(ReputationStoreError, match="agent_id"):
            store.record_delta(bad)


# ---------------------------------------------------------------------------
# AgentScore
# ---------------------------------------------------------------------------


class TestAgentScore:
    def test_agent_score_structure(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(agent_id="alice", delta=30.0, domain=DOMAIN_PREDICTION_MARKET))
        store.record_delta(_delta(agent_id="alice", delta=10.0, domain=DOMAIN_DEBATE_POSITION))
        score = store.agent_score("alice", apply_decay=False)
        assert score.agent_id == "alice"
        assert score.delta_count == 2
        assert score.running_score == pytest.approx(40.0)
        assert sorted(score.domains) == [DOMAIN_DEBATE_POSITION, DOMAIN_PREDICTION_MARKET]

    def test_all_scores_sorted_descending(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(agent_id="charlie", delta=10.0))
        store.record_delta(_delta(agent_id="alice", delta=80.0))
        store.record_delta(_delta(agent_id="bob", delta=40.0))
        scores = store.all_scores(apply_decay=False)
        assert [s.agent_id for s in scores] == ["alice", "bob", "charlie"]
        assert scores[0].running_score == pytest.approx(80.0)

    def test_agent_score_to_json(self) -> None:
        score = AgentScore(
            agent_id="alice",
            running_score=42.5,
            delta_count=3,
            domains=["prediction_market"],
        )
        j = score.to_json()
        assert j["agent_id"] == "alice"
        assert j["running_score"] == pytest.approx(42.5)
        assert j["delta_count"] == 3
        assert j["domains"] == ["prediction_market"]


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_no_decay_when_half_life_none(self) -> None:
        store = ReputationStore()
        d = _delta(delta=100.0, decay_half_life_days=None)
        store.record_delta(d)
        assert store.get_score("alice", apply_decay=True) == pytest.approx(100.0)

    def test_apply_decay_false_returns_raw_sum(self) -> None:
        store = ReputationStore()
        store.record_delta(_delta(delta=100.0, decay_half_life_days=30.0))
        # Raw sum ignores decay
        assert store.get_score("alice", apply_decay=False) == pytest.approx(100.0)

    def test_fresh_delta_barely_decays(self) -> None:
        store = ReputationStore()
        now_iso = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
        d = _delta(delta=100.0, decay_half_life_days=30.0, applied_at=now_iso)
        store.record_delta(d)
        score = store.get_score("alice", apply_decay=True)
        # Should be very close to 100 since it was just applied
        assert score == pytest.approx(100.0, rel=0.01)

    def test_one_half_life_old_delta_halves(self) -> None:
        store = ReputationStore()
        thirty_days_ago = (
            (datetime.now(tz=UTC) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
        )
        d = _delta(delta=100.0, decay_half_life_days=30.0, applied_at=thirty_days_ago)
        store.record_delta(d)
        score = store.get_score("alice", apply_decay=True)
        assert score == pytest.approx(50.0, rel=0.02)

    def test_very_old_delta_decays_near_zero(self) -> None:
        store = ReputationStore()
        old = (datetime.now(tz=UTC) - timedelta(days=365)).isoformat().replace("+00:00", "Z")
        d = _delta(delta=100.0, decay_half_life_days=30.0, applied_at=old)
        store.record_delta(d)
        score = store.get_score("alice", apply_decay=True)
        assert score < 1.0  # ~100 * 2^(-365/30) ≈ 0.00032


# ---------------------------------------------------------------------------
# Persistence (JSONL round-trip)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_file_created_on_record(self, tmp_path: Path) -> None:
        ledger = tmp_path / "rep.jsonl"
        store = ReputationStore(path=ledger)
        store.record_delta(_delta(delta=50.0))
        assert ledger.exists()

    def test_file_has_one_line_per_delta(self, tmp_path: Path) -> None:
        ledger = tmp_path / "rep.jsonl"
        store = ReputationStore(path=ledger)
        store.record_delta(_delta(delta=50.0, resolution_id="r1"))
        store.record_delta(_delta(delta=30.0, resolution_id="r2"))
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_load_from_file_round_trip(self, tmp_path: Path) -> None:
        ledger = tmp_path / "rep.jsonl"
        store = ReputationStore(path=ledger)
        store.record_delta(_delta(agent_id="alice", delta=70.0))
        store.record_delta(_delta(agent_id="bob", delta=20.0))

        reloaded = ReputationStore.load_from_file(ledger)
        assert reloaded.get_score("alice", apply_decay=False) == pytest.approx(70.0)
        assert reloaded.get_score("bob", apply_decay=False) == pytest.approx(20.0)

    def test_load_from_missing_file_returns_empty(self, tmp_path: Path) -> None:
        store = ReputationStore.load_from_file(tmp_path / "ghost.jsonl")
        assert len(store) == 0

    def test_load_skips_corrupt_lines(self, tmp_path: Path) -> None:
        ledger = tmp_path / "rep.jsonl"
        # Write one valid line and one invalid line
        good = _delta(delta=55.0)
        with ledger.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(good.to_json(), sort_keys=True) + "\n")
            fh.write("not valid json\n")
        store = ReputationStore.load_from_file(ledger)
        assert len(store) == 1
        assert store.get_score("alice", apply_decay=False) == pytest.approx(55.0)

    def test_no_path_no_file_written(self, tmp_path: Path) -> None:
        store = ReputationStore(path=None)
        store.record_delta(_delta(delta=10.0))
        assert not any(tmp_path.iterdir())

    def test_write_failure_raises_without_mutating_memory(self, tmp_path: Path) -> None:
        ledger = tmp_path / "rep.jsonl"
        store = ReputationStore(path=ledger)
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            with pytest.raises(ReputationStoreError, match="could not persist"):
                store.record_delta(_delta(delta=10.0))
        assert len(store) == 0
        assert store.get_score("alice", apply_decay=False) == 0.0
