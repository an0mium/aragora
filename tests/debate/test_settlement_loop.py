"""Tests for the epistemic settlement loop scaffolding.

Tests cover:
- SettlementMetadata creation, serialization, and lifecycle
- EpistemicSettlementTracker capture, retrieval, review, and scheduling
- InMemorySettlementStore and JsonFileSettlementStore
- Receipt integration with settlement metadata
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.debate.settlement import (
    EpistemicSettlementTracker,
    InMemorySettlementStore,
    JsonFileSettlementStore,
    SettlementMetadata,
    SettlementMetadataStatus,
    SettlementStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debate_result(
    debate_id: str = "debate-001",
    consensus_reached: bool = True,
    confidence: float = 0.85,
    winner: str = "claude",
    participants: list[str] | None = None,
    dissenting_views: list[str] | None = None,
    final_answer: str = "",
    unresolved_tensions: list | None = None,
    convergence_similarity: float = 0.8,
) -> SimpleNamespace:
    """Build a lightweight debate result for testing."""
    return SimpleNamespace(
        debate_id=debate_id,
        consensus_reached=consensus_reached,
        confidence=confidence,
        winner=winner,
        participants=participants or ["claude", "gpt4", "gemini"],
        dissenting_views=dissenting_views or [],
        final_answer=final_answer,
        unresolved_tensions=unresolved_tensions or [],
        convergence_similarity=convergence_similarity,
        messages=[],
        votes=[],
        rounds_used=3,
        duration_seconds=45.0,
    )


def _make_receipt(
    consensus_proof: SimpleNamespace | None = None,
    provenance_chain: list | None = None,
) -> SimpleNamespace:
    """Build a lightweight receipt for testing."""
    if consensus_proof is None:
        consensus_proof = SimpleNamespace(
            dissenting_agents=["gpt4"],
        )
    return SimpleNamespace(
        consensus_proof=consensus_proof,
        provenance_chain=provenance_chain or [],
    )


# ---------------------------------------------------------------------------
# SettlementMetadata
# ---------------------------------------------------------------------------


class TestSettlementMetadata:
    def test_defaults(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.85,
        )
        assert meta.status == "settled"
        assert meta.falsifiers == []
        assert meta.alternatives == []
        assert meta.cruxes == []
        assert meta.review_notes == []
        assert meta.reviewed_at is None
        assert meta.reviewed_by is None

    def test_to_dict(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.9,
            falsifiers=["if latency > 200ms"],
            alternatives=["use caching instead"],
            review_horizon="2026-02-01T00:00:00+00:00",
            cruxes=["latency vs throughput tradeoff"],
            status="settled",
        )
        d = meta.to_dict()
        assert d["debate_id"] == "d1"
        assert d["confidence"] == 0.9
        assert len(d["falsifiers"]) == 1
        assert d["status"] == "settled"

    def test_from_dict_roundtrip(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.75,
            falsifiers=["f1", "f2"],
            alternatives=["a1"],
            review_horizon="2026-03-01T00:00:00+00:00",
            cruxes=["c1"],
            status="confirmed",
            review_notes=["reviewed ok"],
            reviewed_at="2026-02-15T00:00:00+00:00",
            reviewed_by="admin",
        )
        d = meta.to_dict()
        restored = SettlementMetadata.from_dict(d)
        assert restored.debate_id == meta.debate_id
        assert restored.confidence == meta.confidence
        assert restored.falsifiers == meta.falsifiers
        assert restored.status == meta.status
        assert restored.review_notes == meta.review_notes
        assert restored.reviewed_by == meta.reviewed_by

    def test_to_dict_json_serializable(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.5,
        )
        serialized = json.dumps(meta.to_dict())
        assert "d1" in serialized

    def test_is_due_past_horizon(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon=past,
        )
        assert meta.is_due() is True

    def test_is_due_future_horizon(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon=future,
        )
        assert meta.is_due() is False

    def test_is_due_no_horizon(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon="",
        )
        assert meta.is_due() is False

    def test_is_due_confirmed_not_due(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon=past,
            status="confirmed",
        )
        assert meta.is_due() is False

    def test_is_due_invalidated_not_due(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon=past,
            status="invalidated",
        )
        assert meta.is_due() is False

    def test_is_due_with_explicit_as_of(self):
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            review_horizon="2026-03-01T00:00:00+00:00",
        )
        # Before the horizon
        before = datetime(2026, 2, 15, tzinfo=timezone.utc)
        assert meta.is_due(as_of=before) is False

        # After the horizon
        after = datetime(2026, 3, 15, tzinfo=timezone.utc)
        assert meta.is_due(as_of=after) is True


class TestSettlementMetadataStatus:
    def test_all_statuses(self):
        assert SettlementMetadataStatus.SETTLED.value == "settled"
        assert SettlementMetadataStatus.DUE_REVIEW.value == "due_review"
        assert SettlementMetadataStatus.INVALIDATED.value == "invalidated"
        assert SettlementMetadataStatus.CONFIRMED.value == "confirmed"

    def test_string_equality(self):
        assert SettlementMetadataStatus.SETTLED == "settled"
        assert SettlementMetadataStatus.CONFIRMED == "confirmed"


# ---------------------------------------------------------------------------
# InMemorySettlementStore
# ---------------------------------------------------------------------------


class TestInMemorySettlementStore:
    def test_save_and_get(self):
        store = InMemorySettlementStore()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
        )
        store.save(meta)
        retrieved = store.get("d1")
        assert retrieved is not None
        assert retrieved.confidence == 0.8

    def test_get_not_found(self):
        store = InMemorySettlementStore()
        assert store.get("nonexistent") is None

    def test_list_all(self):
        store = InMemorySettlementStore()
        for i in range(3):
            store.save(
                SettlementMetadata(
                    debate_id=f"d{i}",
                    settled_at="2026-01-01T00:00:00+00:00",
                    confidence=0.5 + i * 0.1,
                )
            )
        assert len(store.list_all()) == 3

    def test_delete(self):
        store = InMemorySettlementStore()
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
            )
        )
        assert store.delete("d1") is True
        assert store.get("d1") is None

    def test_delete_not_found(self):
        store = InMemorySettlementStore()
        assert store.delete("nonexistent") is False

    def test_update_existing(self):
        store = InMemorySettlementStore()
        meta = SettlementMetadata(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
        )
        store.save(meta)
        meta.status = "confirmed"
        store.save(meta)
        assert store.get("d1").status == "confirmed"


# ---------------------------------------------------------------------------
# JsonFileSettlementStore
# ---------------------------------------------------------------------------


class TestJsonFileSettlementStore:
    def test_creates_directory(self, tmp_path: Path):
        store = JsonFileSettlementStore(tmp_path)
        assert (tmp_path / "settlement").is_dir()

    def test_persistence_roundtrip(self, tmp_path: Path):
        store = JsonFileSettlementStore(tmp_path)
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.9,
                falsifiers=["f1"],
            )
        )
        store.save(
            SettlementMetadata(
                debate_id="d2",
                settled_at="2026-01-02T00:00:00+00:00",
                confidence=0.7,
            )
        )

        # Load into a new store instance
        store2 = JsonFileSettlementStore(tmp_path)
        assert store2.get("d1") is not None
        assert store2.get("d1").confidence == 0.9
        assert store2.get("d1").falsifiers == ["f1"]
        assert store2.get("d2") is not None

    def test_corrupt_json_handled(self, tmp_path: Path):
        settlement_dir = tmp_path / "settlement"
        settlement_dir.mkdir()
        (settlement_dir / "metadata.json").write_text("not valid json!!!")

        # Should not raise
        store = JsonFileSettlementStore(tmp_path)
        assert len(store.list_all()) == 0

    def test_delete_persists(self, tmp_path: Path):
        store = JsonFileSettlementStore(tmp_path)
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.5,
            )
        )
        store.delete("d1")

        store2 = JsonFileSettlementStore(tmp_path)
        assert store2.get("d1") is None


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: capture_settlement
# ---------------------------------------------------------------------------


class TestCaptureSettlement:
    def test_basic_capture(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result()
        meta = tracker.capture_settlement(result)

        assert meta.debate_id == "debate-001"
        assert meta.confidence == 0.85
        assert meta.status == "settled"
        assert meta.settled_at != ""
        assert meta.review_horizon != ""

    def test_capture_with_receipt(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result(dissenting_views=["gpt4: I disagree"])
        receipt = _make_receipt()
        meta = tracker.capture_settlement(result, receipt)

        # Should extract falsifiers from dissenting views and receipt
        assert len(meta.falsifiers) > 0

    def test_capture_extracts_alternatives(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result(
            winner="claude",
            participants=["claude", "gpt4", "gemini"],
            dissenting_views=["Use a different approach"],
        )
        meta = tracker.capture_settlement(result)

        # Should have alternatives from losers and dissenting views
        assert len(meta.alternatives) > 0

    def test_capture_extracts_cruxes_from_tensions(self):
        tracker = EpistemicSettlementTracker()
        tension = SimpleNamespace(
            description="Latency vs throughput tradeoff"
        )
        result = _make_debate_result(unresolved_tensions=[tension])
        meta = tracker.capture_settlement(result)

        assert len(meta.cruxes) > 0
        assert any("tension" in c.lower() for c in meta.cruxes)

    def test_capture_extracts_cruxes_from_low_confidence(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result(
            consensus_reached=True, confidence=0.55
        )
        meta = tracker.capture_settlement(result)

        assert any("low confidence" in c.lower() for c in meta.cruxes)

    def test_capture_extracts_cruxes_from_low_convergence(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result(convergence_similarity=0.3)
        meta = tracker.capture_settlement(result)

        assert any("convergence" in c.lower() for c in meta.cruxes)

    def test_capture_extracts_cruxes_from_final_answer_tradeoffs(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result(
            final_answer="We chose option A, however there is a trade-off between speed and accuracy."
        )
        meta = tracker.capture_settlement(result)

        assert any("trade-off" in c.lower() or "tradeoff" in c.lower() for c in meta.cruxes)

    def test_capture_custom_horizon(self):
        tracker = EpistemicSettlementTracker()
        result = _make_debate_result()
        meta = tracker.capture_settlement(result, review_horizon_days=7)

        horizon = datetime.fromisoformat(meta.review_horizon)
        settled = datetime.fromisoformat(meta.settled_at)
        delta = horizon - settled
        assert 6 <= delta.days <= 8  # Allow slight timing variance

    def test_capture_stores_in_store(self):
        store = InMemorySettlementStore()
        tracker = EpistemicSettlementTracker(store=store)
        result = _make_debate_result(debate_id="d42")
        tracker.capture_settlement(result)

        assert store.get("d42") is not None

    def test_capture_with_claims_kernel(self):
        tracker = EpistemicSettlementTracker()
        kernel = MagicMock()
        claim = MagicMock()
        claim.verification_criteria = "Check p99 latency"
        kernel.get_claims.return_value = [claim]

        result = _make_debate_result()
        result.claims_kernel = kernel
        meta = tracker.capture_settlement(result)

        assert any("Verifiable" in f for f in meta.falsifiers)

    def test_capture_generates_id_when_missing(self):
        tracker = EpistemicSettlementTracker()
        result = SimpleNamespace(
            confidence=0.7,
            consensus_reached=True,
            dissenting_views=[],
            final_answer="",
            messages=[],
            votes=[],
        )
        meta = tracker.capture_settlement(result)
        assert meta.debate_id.startswith("debate-")


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: get_due_settlements
# ---------------------------------------------------------------------------


class TestGetDueSettlements:
    def test_no_due(self):
        tracker = EpistemicSettlementTracker()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        store = tracker._store
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=future,
            )
        )
        assert len(tracker.get_due_settlements()) == 0

    def test_with_due(self):
        tracker = EpistemicSettlementTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        store = tracker._store
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=past,
            )
        )
        due = tracker.get_due_settlements()
        assert len(due) == 1
        assert due[0].debate_id == "d1"

    def test_mixed_due_and_not_due(self):
        tracker = EpistemicSettlementTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        store = tracker._store
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=past,
            )
        )
        store.save(
            SettlementMetadata(
                debate_id="d2",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.9,
                review_horizon=future,
            )
        )
        due = tracker.get_due_settlements()
        assert len(due) == 1

    def test_confirmed_excluded(self):
        tracker = EpistemicSettlementTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        store = tracker._store
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=past,
                status="confirmed",
            )
        )
        assert len(tracker.get_due_settlements()) == 0

    def test_custom_as_of(self):
        tracker = EpistemicSettlementTracker()
        store = tracker._store
        store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon="2026-02-01T00:00:00+00:00",
            )
        )
        # Before horizon
        before = datetime(2026, 1, 15, tzinfo=timezone.utc)
        assert len(tracker.get_due_settlements(as_of=before)) == 0

        # After horizon
        after = datetime(2026, 3, 1, tzinfo=timezone.utc)
        assert len(tracker.get_due_settlements(as_of=after)) == 1


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: mark_reviewed
# ---------------------------------------------------------------------------


class TestMarkReviewed:
    def _setup_tracker(self) -> EpistemicSettlementTracker:
        tracker = EpistemicSettlementTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        tracker._store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=past,
            )
        )
        return tracker

    def test_mark_confirmed(self):
        tracker = self._setup_tracker()
        result = tracker.mark_reviewed("d1", "confirmed", "Decision still valid")

        assert result is not None
        assert result.status == "confirmed"
        assert result.reviewed_at is not None
        assert result.reviewed_by == "manual"
        assert len(result.review_notes) == 1
        assert "Decision still valid" in result.review_notes[0]

    def test_mark_invalidated(self):
        tracker = self._setup_tracker()
        result = tracker.mark_reviewed(
            "d1", "invalidated", "New data contradicts decision",
            reviewed_by="auto-reviewer",
        )
        assert result.status == "invalidated"
        assert result.reviewed_by == "auto-reviewer"

    def test_mark_due_review(self):
        tracker = self._setup_tracker()
        result = tracker.mark_reviewed("d1", "due_review", "Needs more data")
        assert result.status == "due_review"

    def test_mark_not_found(self):
        tracker = self._setup_tracker()
        result = tracker.mark_reviewed("nonexistent", "confirmed")
        assert result is None

    def test_invalid_status_raises(self):
        tracker = self._setup_tracker()
        with pytest.raises(ValueError, match="Invalid status"):
            tracker.mark_reviewed("d1", "bogus_status")

    def test_multiple_reviews_append_notes(self):
        tracker = self._setup_tracker()
        tracker.mark_reviewed("d1", "due_review", "First review")
        tracker.mark_reviewed("d1", "confirmed", "Second review")

        meta = tracker.get_settlement("d1")
        assert len(meta.review_notes) == 2
        assert "First review" in meta.review_notes[0]
        assert "Second review" in meta.review_notes[1]

    def test_mark_without_notes(self):
        tracker = self._setup_tracker()
        result = tracker.mark_reviewed("d1", "confirmed")
        assert len(result.review_notes) == 0  # No notes added when empty


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: get_settlement
# ---------------------------------------------------------------------------


class TestGetSettlement:
    def test_found(self):
        tracker = EpistemicSettlementTracker()
        tracker._store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
            )
        )
        meta = tracker.get_settlement("d1")
        assert meta is not None
        assert meta.debate_id == "d1"

    def test_not_found(self):
        tracker = EpistemicSettlementTracker()
        assert tracker.get_settlement("nonexistent") is None


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: get_all_settlements
# ---------------------------------------------------------------------------


class TestGetAllSettlements:
    def test_empty(self):
        tracker = EpistemicSettlementTracker()
        assert tracker.get_all_settlements() == []

    def test_multiple(self):
        tracker = EpistemicSettlementTracker()
        for i in range(5):
            tracker._store.save(
                SettlementMetadata(
                    debate_id=f"d{i}",
                    settled_at="2026-01-01T00:00:00+00:00",
                    confidence=0.5 + i * 0.1,
                )
            )
        assert len(tracker.get_all_settlements()) == 5


# ---------------------------------------------------------------------------
# EpistemicSettlementTracker: get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_empty_summary(self):
        tracker = EpistemicSettlementTracker()
        summary = tracker.get_summary()
        assert summary["total"] == 0
        assert summary["due_for_review"] == 0
        assert summary["average_confidence"] == 0.0

    def test_populated_summary(self):
        tracker = EpistemicSettlementTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        tracker._store.save(
            SettlementMetadata(
                debate_id="d1",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.8,
                review_horizon=past,
                status="settled",
            )
        )
        tracker._store.save(
            SettlementMetadata(
                debate_id="d2",
                settled_at="2026-01-01T00:00:00+00:00",
                confidence=0.6,
                review_horizon=future,
                status="confirmed",
            )
        )

        summary = tracker.get_summary()
        assert summary["total"] == 2
        assert summary["by_status"]["settled"] == 1
        assert summary["by_status"]["confirmed"] == 1
        assert summary["due_for_review"] == 1
        assert summary["average_confidence"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Receipt integration
# ---------------------------------------------------------------------------


class TestReceiptIntegration:
    def test_receipt_has_settlement_metadata_field(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00+00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
        )
        assert receipt.settlement_metadata is None

    def test_receipt_with_settlement_metadata(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

        settlement_data = {
            "debate_id": "d1",
            "settled_at": "2026-01-01T00:00:00+00:00",
            "confidence": 0.85,
            "falsifiers": ["if latency > 200ms"],
            "alternatives": ["caching approach"],
            "review_horizon": "2026-02-01T00:00:00+00:00",
            "cruxes": ["latency vs throughput"],
            "status": "settled",
        }

        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00+00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            settlement_metadata=settlement_data,
        )
        assert receipt.settlement_metadata is not None
        assert receipt.settlement_metadata["debate_id"] == "d1"
        assert len(receipt.settlement_metadata["falsifiers"]) == 1

    def test_receipt_to_dict_includes_settlement_metadata(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

        settlement_data = {
            "debate_id": "d1",
            "confidence": 0.85,
            "falsifiers": ["f1"],
            "status": "settled",
        }

        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00+00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            settlement_metadata=settlement_data,
        )
        d = receipt.to_dict()
        assert "settlement_metadata" in d
        assert d["settlement_metadata"]["debate_id"] == "d1"

    def test_receipt_to_dict_without_settlement_metadata(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00+00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
        )
        d = receipt.to_dict()
        assert "settlement_metadata" in d
        assert d["settlement_metadata"] is None

    def test_receipt_from_dict_with_settlement_metadata(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

        data = {
            "receipt_id": "r1",
            "gauntlet_id": "g1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "input_summary": "test",
            "input_hash": "abc",
            "risk_summary": {},
            "attacks_attempted": 0,
            "attacks_successful": 0,
            "probes_run": 0,
            "vulnerabilities_found": 0,
            "verdict": "PASS",
            "confidence": 0.9,
            "robustness_score": 0.8,
            "settlement_metadata": {
                "debate_id": "d1",
                "falsifiers": ["f1"],
                "status": "settled",
            },
        }
        receipt = DecisionReceipt.from_dict(data)
        assert receipt.settlement_metadata is not None
        assert receipt.settlement_metadata["debate_id"] == "d1"

    def test_receipt_from_dict_without_settlement_metadata(self):
        """Backward compatibility: old receipts without settlement_metadata."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        data = {
            "receipt_id": "r1",
            "gauntlet_id": "g1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "input_summary": "test",
            "input_hash": "abc",
            "risk_summary": {},
            "attacks_attempted": 0,
            "attacks_successful": 0,
            "probes_run": 0,
            "vulnerabilities_found": 0,
            "verdict": "PASS",
            "confidence": 0.9,
            "robustness_score": 0.8,
        }
        receipt = DecisionReceipt.from_dict(data)
        assert receipt.settlement_metadata is None

    def test_from_debate_result_with_settlement_metadata(self):
        """from_debate_result accepts settlement_metadata parameter."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        result = _make_debate_result()
        settlement_data = {"debate_id": "d1", "status": "settled"}

        receipt = DecisionReceipt.from_debate_result(
            result,
            settlement_metadata=settlement_data,
        )
        assert receipt.settlement_metadata is not None
        assert receipt.settlement_metadata["debate_id"] == "d1"

    def test_from_debate_result_without_settlement_metadata(self):
        """from_debate_result works without settlement_metadata (backward compat)."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        result = _make_debate_result()
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.settlement_metadata is None


# ---------------------------------------------------------------------------
# Full lifecycle integration test
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_capture_review_confirm(self):
        """Full lifecycle: capture -> due -> review -> confirmed."""
        tracker = EpistemicSettlementTracker()

        # 1. Capture settlement with short horizon
        result = _make_debate_result(
            debate_id="lifecycle-test",
            confidence=0.85,
            dissenting_views=["Agent gpt4 disagrees with approach"],
        )
        meta = tracker.capture_settlement(
            result, review_horizon_days=0  # Immediately due
        )
        assert meta.status == "settled"

        # 2. Check due settlements
        due = tracker.get_due_settlements()
        assert len(due) == 1
        assert due[0].debate_id == "lifecycle-test"

        # 3. Review and confirm
        reviewed = tracker.mark_reviewed(
            "lifecycle-test",
            "confirmed",
            "Decision validated by production metrics",
            reviewed_by="sre-team",
        )
        assert reviewed.status == "confirmed"
        assert reviewed.reviewed_by == "sre-team"

        # 4. No longer due
        assert len(tracker.get_due_settlements()) == 0

        # 5. Summary reflects state
        summary = tracker.get_summary()
        assert summary["total"] == 1
        assert summary["by_status"]["confirmed"] == 1

    def test_capture_review_invalidate(self):
        """Full lifecycle: capture -> due -> review -> invalidated."""
        tracker = EpistemicSettlementTracker()

        result = _make_debate_result(
            debate_id="invalid-test",
            confidence=0.6,
        )
        tracker.capture_settlement(result, review_horizon_days=0)

        reviewed = tracker.mark_reviewed(
            "invalid-test",
            "invalidated",
            "New evidence contradicts the original decision",
        )
        assert reviewed.status == "invalidated"
        assert len(tracker.get_due_settlements()) == 0
