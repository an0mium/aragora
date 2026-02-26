"""Tests for settlement hook infrastructure.

Validates that:
- SettlementHookRegistry dispatches to registered hooks
- SettlementTracker fires hooks on extract and settle
- ERC8004SettlementHook pushes reputation and validation
- EventBusSettlementHook emits events
- Hook failures are logged but don't break the settlement flow
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.settlement import (
    SettlementBatch,
    SettlementOutcome,
    SettlementRecord,
    SettlementStatus,
    SettlementTracker,
    SettleResult,
    VerifiableClaim,
)
from aragora.debate.settlement_hooks import (
    ERC8004SettlementHook,
    EventBusSettlementHook,
    LoggingSettlementHook,
    SettlementHookRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_claim(
    debate_id: str = "d-1",
    statement: str = "Revenue will increase by 20%",
    author: str = "claude",
    confidence: float = 0.8,
) -> VerifiableClaim:
    return VerifiableClaim(
        claim_id="clm-1",
        debate_id=debate_id,
        statement=statement,
        author=author,
        confidence=confidence,
    )


def _make_record(claim: VerifiableClaim | None = None) -> SettlementRecord:
    claim = claim or _make_claim()
    return SettlementRecord(
        settlement_id="stl-abc123",
        claim=claim,
        status=SettlementStatus.SETTLED_CORRECT,
        outcome=SettlementOutcome.CORRECT,
        outcome_evidence="Revenue grew 25%",
        score=1.0,
        settled_at="2026-03-01T00:00:00Z",
        settled_by="manual",
    )


def _make_settle_result() -> SettleResult:
    return SettleResult(
        settlement_id="stl-abc123",
        outcome=SettlementOutcome.CORRECT,
        score=1.0,
        elo_updates={"claude": 12.5},
        calibration_recorded=True,
    )


def _make_batch() -> SettlementBatch:
    return SettlementBatch(
        debate_id="d-1",
        settlements_created=2,
        settlement_ids=["stl-abc123", "stl-def456"],
        claims_skipped=1,
    )


def _make_debate_result() -> MagicMock:
    """Minimal debate result with a predictive message."""
    result = MagicMock()
    result.debate_id = "d-1"
    result.confidence = 0.8
    result.final_answer = "Revenue will increase by 20% next quarter"
    result.messages = []
    result.claims_kernel = None
    return result


# ---------------------------------------------------------------------------
# SettlementHookRegistry
# ---------------------------------------------------------------------------


class TestSettlementHookRegistry:
    def test_empty_registry_does_not_error(self):
        registry = SettlementHookRegistry()
        registry.fire_claims_extracted(_make_batch())
        registry.fire_settled(_make_record(), _make_settle_result())

    def test_dispatches_to_all_hooks(self):
        hook1 = MagicMock()
        hook2 = MagicMock()
        registry = SettlementHookRegistry()
        registry.register(hook1)
        registry.register(hook2)

        batch = _make_batch()
        registry.fire_claims_extracted(batch)
        hook1.on_claims_extracted.assert_called_once_with(batch)
        hook2.on_claims_extracted.assert_called_once_with(batch)

        record = _make_record()
        result = _make_settle_result()
        registry.fire_settled(record, result)
        hook1.on_settled.assert_called_once_with(record, result)
        hook2.on_settled.assert_called_once_with(record, result)

    def test_hook_failure_does_not_break_dispatch(self):
        good_hook = MagicMock()
        bad_hook = MagicMock()
        bad_hook.on_claims_extracted.side_effect = RuntimeError("boom")
        bad_hook.on_settled.side_effect = ValueError("oops")

        registry = SettlementHookRegistry()
        registry.register(bad_hook)
        registry.register(good_hook)

        batch = _make_batch()
        registry.fire_claims_extracted(batch)
        # Good hook still called despite bad hook failing
        good_hook.on_claims_extracted.assert_called_once_with(batch)

        record = _make_record()
        result = _make_settle_result()
        registry.fire_settled(record, result)
        good_hook.on_settled.assert_called_once_with(record, result)

    def test_hook_count(self):
        registry = SettlementHookRegistry()
        assert registry.hook_count == 0
        registry.register(MagicMock())
        assert registry.hook_count == 1


# ---------------------------------------------------------------------------
# SettlementTracker integration with hooks
# ---------------------------------------------------------------------------


class TestSettlementTrackerHooks:
    def test_extract_fires_hook(self):
        hooks = SettlementHookRegistry()
        hook = MagicMock()
        hooks.register(hook)

        tracker = SettlementTracker(hooks=hooks)
        debate_result = _make_debate_result()

        with patch(
            "aragora.reasoning.claims.fast_extract_claims",
            return_value=[
                {"text": "Revenue will increase by 20%", "author": "claude", "confidence": 0.8}
            ],
        ):
            batch = tracker.extract_verifiable_claims("d-1", debate_result)

        assert batch.settlements_created >= 1
        hook.on_claims_extracted.assert_called_once()
        fired_batch = hook.on_claims_extracted.call_args[0][0]
        assert fired_batch.debate_id == "d-1"

    def test_extract_does_not_fire_hook_when_no_claims(self):
        hooks = SettlementHookRegistry()
        hook = MagicMock()
        hooks.register(hook)

        tracker = SettlementTracker(hooks=hooks)
        debate_result = MagicMock()
        debate_result.claims_kernel = None
        debate_result.messages = []
        debate_result.final_answer = ""  # Empty → no extraction attempted

        batch = tracker.extract_verifiable_claims("d-1", debate_result)
        assert batch.settlements_created == 0
        hook.on_claims_extracted.assert_not_called()

    def test_settle_fires_hook(self):
        hooks = SettlementHookRegistry()
        hook = MagicMock()
        hooks.register(hook)

        tracker = SettlementTracker(hooks=hooks)
        # Manually insert a record
        claim = _make_claim()
        record = SettlementRecord(settlement_id="stl-test", claim=claim)
        tracker._records["stl-test"] = record
        tracker._debate_index["d-1"] = ["stl-test"]

        result = tracker.settle("stl-test", "correct", evidence="it happened")

        hook.on_settled.assert_called_once()
        fired_record = hook.on_settled.call_args[0][0]
        fired_result = hook.on_settled.call_args[0][1]
        assert fired_record.settlement_id == "stl-test"
        assert fired_result.outcome == SettlementOutcome.CORRECT
        assert fired_result.score == 1.0

    def test_no_hooks_no_error(self):
        """Tracker works fine with hooks=None (default)."""
        tracker = SettlementTracker()
        claim = _make_claim()
        record = SettlementRecord(settlement_id="stl-nohook", claim=claim)
        tracker._records["stl-nohook"] = record
        result = tracker.settle("stl-nohook", "incorrect")
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# ERC8004SettlementHook
# ---------------------------------------------------------------------------


class TestERC8004SettlementHook:
    def test_on_settled_pushes_reputation(self):
        adapter = MagicMock()
        hook = ERC8004SettlementHook(adapter=adapter)

        record = _make_record()
        result = _make_settle_result()

        hook.on_settled(record, result)

        adapter.push_reputation.assert_called_once()
        call_kwargs = adapter.push_reputation.call_args[1]
        assert call_kwargs["agent_id"] == "claude"
        assert call_kwargs["domain"] == "settlement"
        assert "settlement_id" in call_kwargs["metadata"]
        assert "content_hash" in call_kwargs["metadata"]
        # Brier component for confidence=0.8, score=1.0: (0.8-1.0)^2 = 0.04
        assert call_kwargs["metadata"]["brier_component"] == pytest.approx(0.04, abs=0.01)
        # Reputation: (1.0 - 0.04) * 100 = 96
        assert call_kwargs["score"] == 96

    def test_on_settled_incorrect_low_reputation(self):
        adapter = MagicMock()
        hook = ERC8004SettlementHook(adapter=adapter)

        claim = _make_claim(confidence=0.9)
        record = SettlementRecord(
            settlement_id="stl-wrong",
            claim=claim,
            status=SettlementStatus.SETTLED_INCORRECT,
            outcome=SettlementOutcome.INCORRECT,
            score=0.0,
        )
        result = SettleResult(
            settlement_id="stl-wrong",
            outcome=SettlementOutcome.INCORRECT,
            score=0.0,
        )

        hook.on_settled(record, result)

        call_kwargs = adapter.push_reputation.call_args[1]
        # Brier component for confidence=0.9, score=0.0: (0.9-0.0)^2 = 0.81
        # Reputation: int((1.0 - 0.81) * 100) = int(19.0) but float precision → 18
        assert call_kwargs["score"] in (18, 19)

    def test_on_claims_extracted_is_noop(self):
        adapter = MagicMock()
        hook = ERC8004SettlementHook(adapter=adapter)
        hook.on_claims_extracted(_make_batch())
        adapter.push_reputation.assert_not_called()

    def test_no_adapter_gracefully_skips(self):
        hook = ERC8004SettlementHook(adapter=None)
        # Should not raise even without adapter
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters.erc8004_adapter": None}):
            hook.on_settled(_make_record(), _make_settle_result())


# ---------------------------------------------------------------------------
# EventBusSettlementHook
# ---------------------------------------------------------------------------


class TestEventBusSettlementHook:
    def test_on_claims_extracted_emits_event(self):
        bus = MagicMock()
        hook = EventBusSettlementHook(bus)

        batch = _make_batch()
        hook.on_claims_extracted(batch)

        bus.emit.assert_called_once_with(
            "settlement_claims_extracted",
            debate_id="d-1",
            settlements_created=2,
            settlement_ids=["stl-abc123", "stl-def456"],
            claims_skipped=1,
        )

    def test_on_settled_emits_event(self):
        bus = MagicMock()
        hook = EventBusSettlementHook(bus)

        record = _make_record()
        result = _make_settle_result()
        hook.on_settled(record, result)

        bus.emit.assert_called_once_with(
            "settlement_resolved",
            settlement_id="stl-abc123",
            debate_id="d-1",
            agent="claude",
            outcome="correct",
            score=1.0,
            elo_updates={"claude": 12.5},
            calibration_recorded=True,
        )

    def test_none_bus_does_not_error(self):
        hook = EventBusSettlementHook(None)
        hook.on_claims_extracted(_make_batch())
        hook.on_settled(_make_record(), _make_settle_result())


# ---------------------------------------------------------------------------
# LoggingSettlementHook
# ---------------------------------------------------------------------------


class TestLoggingSettlementHook:
    def test_logs_extraction(self, caplog):
        hook = LoggingSettlementHook()
        with caplog.at_level("INFO"):
            hook.on_claims_extracted(_make_batch())
        assert "2 claims extracted" in caplog.text

    def test_logs_settlement(self, caplog):
        hook = LoggingSettlementHook()
        with caplog.at_level("INFO"):
            hook.on_settled(_make_record(), _make_settle_result())
        assert "settled as correct" in caplog.text
