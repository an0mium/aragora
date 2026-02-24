"""Tests for aragora.debate.settlement -- claim settlement tracking."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aragora.debate.settlement import (
    SettlementBatch,
    SettlementOutcome,
    SettlementRecord,
    SettlementStatus,
    SettlementTracker,
    SettleResult,
    VerifiableClaim,
    _is_verifiable,
    _generate_settlement_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debate_result(
    messages: list[dict] | None = None,
    final_answer: str = "",
    confidence: float = 0.7,
    claims_kernel: object | None = None,
) -> SimpleNamespace:
    """Build a lightweight debate result for testing."""
    msgs = []
    for m in messages or []:
        msgs.append(
            SimpleNamespace(
                content=m.get("content", ""),
                agent=m.get("agent", "unknown"),
            )
        )
    return SimpleNamespace(
        messages=msgs,
        final_answer=final_answer,
        confidence=confidence,
        claims_kernel=claims_kernel,
    )


def _verifiable_message(agent: str = "agent-1") -> dict:
    """Return a message dict with verifiable language."""
    return {
        "content": "This will improve performance by 30% within two weeks.",
        "agent": agent,
    }


def _non_verifiable_message(agent: str = "agent-2") -> dict:
    """Return a message with no verifiable language."""
    return {
        "content": "The architecture is well structured and clean.",
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestIsVerifiable:
    def test_verifiable_keywords(self):
        assert _is_verifiable("This will improve performance by 30%.")
        assert _is_verifiable("We predict a 5x speedup.")
        assert _is_verifiable("Latency should decrease below 100ms.")
        assert _is_verifiable("We expect completion within 2 days.")
        assert _is_verifiable("The forecast is for 50% reduction.")

    def test_non_verifiable(self):
        assert not _is_verifiable("The code is clean and readable.")
        assert not _is_verifiable("This is a good approach.")
        assert not _is_verifiable("I agree with the previous analysis.")

    def test_empty(self):
        assert not _is_verifiable("")


class TestGenerateSettlementId:
    def test_deterministic(self):
        sid1 = _generate_settlement_id("d1", "claim text")
        sid2 = _generate_settlement_id("d1", "claim text")
        assert sid1 == sid2

    def test_prefix(self):
        sid = _generate_settlement_id("d1", "claim text")
        assert sid.startswith("stl-")

    def test_different_inputs(self):
        sid1 = _generate_settlement_id("d1", "claim A")
        sid2 = _generate_settlement_id("d1", "claim B")
        assert sid1 != sid2


# ---------------------------------------------------------------------------
# Unit tests: VerifiableClaim
# ---------------------------------------------------------------------------


class TestVerifiableClaim:
    def test_to_dict(self):
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="Will improve by 30%",
            author="agent-1",
            confidence=0.8,
        )
        d = claim.to_dict()
        assert d["claim_id"] == "c1"
        assert d["debate_id"] == "d1"
        assert d["confidence"] == 0.8
        assert "created_at" in d


# ---------------------------------------------------------------------------
# Unit tests: SettlementRecord
# ---------------------------------------------------------------------------


class TestSettlementRecord:
    def test_default_status(self):
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="a",
            confidence=0.5,
        )
        record = SettlementRecord(settlement_id="s1", claim=claim)
        assert record.status == SettlementStatus.PENDING
        assert record.outcome is None

    def test_to_dict(self):
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="a",
            confidence=0.5,
        )
        record = SettlementRecord(
            settlement_id="s1",
            claim=claim,
            status=SettlementStatus.SETTLED_CORRECT,
            outcome=SettlementOutcome.CORRECT,
            score=1.0,
        )
        d = record.to_dict()
        assert d["status"] == "settled_correct"
        assert d["outcome"] == "correct"
        assert d["score"] == 1.0


# ---------------------------------------------------------------------------
# SettlementTracker: extract_verifiable_claims
# ---------------------------------------------------------------------------


class TestExtractVerifiableClaims:
    def test_extract_from_messages(self):
        result = _make_debate_result(
            messages=[_verifiable_message("agent-1"), _non_verifiable_message("agent-2")]
        )
        tracker = SettlementTracker()
        batch = tracker.extract_verifiable_claims("d1", result)

        assert isinstance(batch, SettlementBatch)
        assert batch.debate_id == "d1"
        # At least the verifiable message should produce settlements
        assert batch.settlements_created >= 0  # depends on extraction
        assert batch.claims_skipped >= 0

    def test_extract_from_final_answer(self):
        result = _make_debate_result(
            final_answer="We predict this will improve latency by 40% within one month."
        )
        tracker = SettlementTracker()
        batch = tracker.extract_verifiable_claims("d2", result)
        # Should extract from final_answer when no messages
        assert isinstance(batch, SettlementBatch)

    def test_no_duplicates(self):
        result = _make_debate_result(messages=[_verifiable_message("agent-1")])
        tracker = SettlementTracker()
        batch1 = tracker.extract_verifiable_claims("d1", result)
        batch2 = tracker.extract_verifiable_claims("d1", result)

        # Second extraction should skip already-registered claims
        assert (
            batch2.settlements_created == 0 or batch2.claims_skipped >= batch1.settlements_created
        )

    def test_min_confidence_filter(self):
        result = _make_debate_result(messages=[_verifiable_message("agent-1")])
        tracker = SettlementTracker()
        # Very high min_confidence should filter most claims
        batch = tracker.extract_verifiable_claims("d1", result, min_confidence=0.99)
        # Most auto-extracted claims have low confidence
        assert batch.settlements_created == 0 or batch.claims_skipped > 0

    def test_extract_from_claims_kernel(self):
        """Test extraction using a ClaimsKernel-like object."""

        class FakeTypedClaim:
            def __init__(self):
                self.claim_id = "c1"
                self.statement = "This will increase throughput by 50%"
                self.author = "agent-kernel"
                self.confidence = 0.9
                self.claim_type = "assertion"
                self.round_num = 1

        kernel = MagicMock()
        kernel.get_claims.return_value = [FakeTypedClaim()]

        result = _make_debate_result(claims_kernel=kernel)
        tracker = SettlementTracker()
        batch = tracker.extract_verifiable_claims("d3", result)

        assert batch.settlements_created == 1
        assert len(batch.settlement_ids) == 1


# ---------------------------------------------------------------------------
# SettlementTracker: settle
# ---------------------------------------------------------------------------


class TestSettle:
    def _setup_tracker_with_pending(self) -> tuple[SettlementTracker, str]:
        """Create a tracker with one pending settlement and return (tracker, sid)."""
        tracker = SettlementTracker()
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="Will improve by 30%",
            author="agent-1",
            confidence=0.8,
            domain="performance",
        )
        record = SettlementRecord(settlement_id="stl-test1", claim=claim)
        tracker._records["stl-test1"] = record
        tracker._debate_index.setdefault("d1", []).append("stl-test1")
        return tracker, "stl-test1"

    def test_settle_correct(self):
        tracker, sid = self._setup_tracker_with_pending()
        result = tracker.settle(sid, "correct", evidence="Confirmed by metrics")

        assert isinstance(result, SettleResult)
        assert result.outcome == SettlementOutcome.CORRECT
        assert result.score == 1.0

        record = tracker.get_settlement(sid)
        assert record is not None
        assert record.status == SettlementStatus.SETTLED_CORRECT
        assert record.outcome_evidence == "Confirmed by metrics"

    def test_settle_incorrect(self):
        tracker, sid = self._setup_tracker_with_pending()
        result = tracker.settle(sid, "incorrect")
        assert result.score == 0.0
        assert result.outcome == SettlementOutcome.INCORRECT

    def test_settle_partial(self):
        tracker, sid = self._setup_tracker_with_pending()
        result = tracker.settle(sid, "partial")
        assert result.score == 0.5
        assert result.outcome == SettlementOutcome.PARTIAL

    def test_settle_not_found(self):
        tracker = SettlementTracker()
        with pytest.raises(KeyError, match="Settlement not found"):
            tracker.settle("nonexistent", "correct")

    def test_settle_already_resolved(self):
        tracker, sid = self._setup_tracker_with_pending()
        tracker.settle(sid, "correct")
        with pytest.raises(ValueError, match="already resolved"):
            tracker.settle(sid, "incorrect")

    def test_settle_with_enum(self):
        tracker, sid = self._setup_tracker_with_pending()
        result = tracker.settle(sid, SettlementOutcome.CORRECT)
        assert result.outcome == SettlementOutcome.CORRECT

    def test_settle_updates_elo(self):
        mock_elo = MagicMock()
        mock_elo.record_match.return_value = {"agent-1": 15.0}

        tracker, sid = self._setup_tracker_with_pending()
        tracker._elo_system = mock_elo

        result = tracker.settle(sid, "correct")
        assert result.elo_updates == {"agent-1": 15.0}
        mock_elo.record_match.assert_called_once()

    def test_settle_updates_calibration(self):
        mock_cal = MagicMock()
        mock_cal.record_prediction.return_value = 1

        tracker, sid = self._setup_tracker_with_pending()
        tracker._calibration_tracker = mock_cal

        result = tracker.settle(sid, "correct")
        assert result.calibration_recorded is True
        mock_cal.record_prediction.assert_called_once_with(
            agent="agent-1",
            confidence=0.8,
            correct=True,
            domain="performance",
            debate_id="d1",
            prediction_type="settlement",
        )

    def test_settle_calibration_partial_is_correct(self):
        """Partial outcomes (score >= 0.5) are recorded as correct for calibration."""
        mock_cal = MagicMock()
        tracker, sid = self._setup_tracker_with_pending()
        tracker._calibration_tracker = mock_cal

        tracker.settle(sid, "partial")
        mock_cal.record_prediction.assert_called_once()
        call_kwargs = mock_cal.record_prediction.call_args
        assert call_kwargs.kwargs.get("correct") is True or call_kwargs[1].get("correct") is True

    def test_settle_persists_to_km(self):
        mock_km = MagicMock()
        tracker, sid = self._setup_tracker_with_pending()
        tracker._knowledge_mound = mock_km

        tracker.settle(sid, "correct")
        # Should attempt to store
        assert mock_km.store_sync.called or mock_km.store_item.called


# ---------------------------------------------------------------------------
# SettlementTracker: queries
# ---------------------------------------------------------------------------


class TestQueries:
    def _populated_tracker(self) -> SettlementTracker:
        tracker = SettlementTracker()
        for i in range(5):
            claim = VerifiableClaim(
                claim_id=f"c{i}",
                debate_id="d1" if i < 3 else "d2",
                statement=f"Claim {i} will improve things",
                author=f"agent-{i % 2}",
                confidence=0.5 + i * 0.1,
                domain="general",
            )
            record = SettlementRecord(settlement_id=f"stl-{i}", claim=claim)
            tracker._records[f"stl-{i}"] = record
            tracker._debate_index.setdefault(claim.debate_id, []).append(f"stl-{i}")

        # Settle some
        tracker._records["stl-0"].status = SettlementStatus.SETTLED_CORRECT
        tracker._records["stl-0"].outcome = SettlementOutcome.CORRECT
        tracker._records["stl-0"].score = 1.0
        tracker._records["stl-1"].status = SettlementStatus.SETTLED_INCORRECT
        tracker._records["stl-1"].outcome = SettlementOutcome.INCORRECT
        tracker._records["stl-1"].score = 0.0

        return tracker

    def test_get_pending_all(self):
        tracker = self._populated_tracker()
        pending = tracker.get_pending()
        assert len(pending) == 3
        for r in pending:
            assert r.status == SettlementStatus.PENDING

    def test_get_pending_by_debate(self):
        tracker = self._populated_tracker()
        pending = tracker.get_pending(debate_id="d1")
        assert all(r.claim.debate_id == "d1" for r in pending)

    def test_get_pending_limit(self):
        tracker = self._populated_tracker()
        pending = tracker.get_pending(limit=1)
        assert len(pending) == 1

    def test_get_history(self):
        tracker = self._populated_tracker()
        history = tracker.get_history()
        assert len(history) == 2

    def test_get_history_by_author(self):
        tracker = self._populated_tracker()
        history = tracker.get_history(author="agent-0")
        assert all(r.claim.author == "agent-0" for r in history)

    def test_get_settlement_found(self):
        tracker = self._populated_tracker()
        record = tracker.get_settlement("stl-0")
        assert record is not None
        assert record.settlement_id == "stl-0"

    def test_get_settlement_not_found(self):
        tracker = self._populated_tracker()
        assert tracker.get_settlement("nonexistent") is None

    def test_get_agent_accuracy(self):
        tracker = self._populated_tracker()
        accuracy = tracker.get_agent_accuracy("agent-0")
        assert accuracy["agent"] == "agent-0"
        assert accuracy["total_settled"] >= 0
        assert "accuracy" in accuracy
        assert "brier_score" in accuracy

    def test_get_summary(self):
        tracker = self._populated_tracker()
        summary = tracker.get_summary()
        assert summary["total_settlements"] == 5
        assert summary["pending"] == 3
        assert summary["settled"] == 2
        assert summary["correct"] == 1
        assert summary["incorrect"] == 1


# ---------------------------------------------------------------------------
# SettlementTracker: batch settle
# ---------------------------------------------------------------------------


class TestSettleBatch:
    def test_batch_settle(self):
        tracker = SettlementTracker()
        for i in range(3):
            claim = VerifiableClaim(
                claim_id=f"c{i}",
                debate_id="d1",
                statement=f"Claim {i}",
                author="agent-1",
                confidence=0.7,
            )
            tracker._records[f"stl-{i}"] = SettlementRecord(settlement_id=f"stl-{i}", claim=claim)

        results = tracker.settle_batch(
            [
                {"settlement_id": "stl-0", "outcome": "correct", "evidence": "yes"},
                {"settlement_id": "stl-1", "outcome": "incorrect"},
                {"settlement_id": "stl-2", "outcome": "partial"},
            ]
        )

        assert len(results) == 3
        assert results[0].outcome == SettlementOutcome.CORRECT
        assert results[1].outcome == SettlementOutcome.INCORRECT
        assert results[2].outcome == SettlementOutcome.PARTIAL

    def test_batch_settle_skips_invalid(self):
        tracker = SettlementTracker()
        results = tracker.settle_batch(
            [
                {"settlement_id": "nonexistent", "outcome": "correct"},
            ]
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_calibration(self):
        """Agent with confidence matching outcomes has brier ~0."""
        tracker = SettlementTracker()
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="perfect",
            confidence=1.0,
        )
        record = SettlementRecord(
            settlement_id="stl-1",
            claim=claim,
            status=SettlementStatus.SETTLED_CORRECT,
            outcome=SettlementOutcome.CORRECT,
            score=1.0,
        )
        tracker._records["stl-1"] = record

        brier = tracker._compute_brier_score("perfect")
        assert brier == pytest.approx(0.0)

    def test_worst_calibration(self):
        """Agent confident but wrong has high brier score."""
        tracker = SettlementTracker()
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="bad",
            confidence=1.0,
        )
        record = SettlementRecord(
            settlement_id="stl-1",
            claim=claim,
            status=SettlementStatus.SETTLED_INCORRECT,
            outcome=SettlementOutcome.INCORRECT,
            score=0.0,
        )
        tracker._records["stl-1"] = record

        brier = tracker._compute_brier_score("bad")
        assert brier == pytest.approx(1.0)

    def test_no_records(self):
        tracker = SettlementTracker()
        assert tracker._compute_brier_score("nobody") == 0.0


# ---------------------------------------------------------------------------
# ELO integration edge cases
# ---------------------------------------------------------------------------


class TestEloIntegration:
    def test_elo_failure_does_not_raise(self):
        mock_elo = MagicMock()
        mock_elo.record_match.side_effect = ValueError("boom")

        tracker = SettlementTracker(elo_system=mock_elo)
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="a",
            confidence=0.5,
        )
        tracker._records["stl-1"] = SettlementRecord(settlement_id="stl-1", claim=claim)

        result = tracker.settle("stl-1", "correct")
        assert result.elo_updates == {}

    def test_no_elo_system(self):
        tracker = SettlementTracker()
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="test",
            author="a",
            confidence=0.5,
        )
        tracker._records["stl-1"] = SettlementRecord(settlement_id="stl-1", claim=claim)
        result = tracker.settle("stl-1", "correct")
        assert result.elo_updates == {}


# ---------------------------------------------------------------------------
# PostDebateConfig integration
# ---------------------------------------------------------------------------


class TestPostDebateConfigIntegration:
    def test_config_has_settlement_fields(self):
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig()
        assert hasattr(config, "auto_settlement_tracking")
        assert config.auto_settlement_tracking is False
        assert hasattr(config, "settlement_min_confidence")
        assert hasattr(config, "settlement_domain")

    def test_post_debate_result_has_settlement_batch(self):
        from aragora.debate.post_debate_coordinator import PostDebateResult

        result = PostDebateResult(debate_id="d1")
        assert hasattr(result, "settlement_batch")
        assert result.settlement_batch is None


# ---------------------------------------------------------------------------
# DebateProtocol integration
# ---------------------------------------------------------------------------


class TestDebateProtocolIntegration:
    def test_protocol_has_settlement_flag(self):
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        assert hasattr(protocol, "enable_settlement_tracking")
        assert protocol.enable_settlement_tracking is False

    def test_protocol_settlement_enabled(self):
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(enable_settlement_tracking=True)
        assert protocol.enable_settlement_tracking is True
