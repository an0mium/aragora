"""Tests for OutcomeVerifier — the feedback loop keystone."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.outcome_verifier import (
    AsyncOutcomeVerifier,
    OutcomeVerifier,
    PendingDecision,
    SignalType,
    VerificationResult,
)


@pytest.fixture
def verifier(tmp_path):
    """Create a fresh OutcomeVerifier with a temp database."""
    return OutcomeVerifier(db_path=tmp_path / "test_verifier.db")


class TestRecordDecision:
    """Tests for recording pending decisions."""

    def test_record_basic_decision(self, verifier):
        decision = verifier.record_decision(
            debate_id="d-1",
            agents=["claude", "gpt4"],
            consensus_confidence=0.85,
            consensus_text="Deploy with canary",
            domain="deployment",
        )
        assert decision.debate_id == "d-1"
        assert decision.agents == ["claude", "gpt4"]
        assert decision.consensus_confidence == 0.85
        assert decision.domain == "deployment"
        assert not decision.verified

    def test_record_multiple_decisions(self, verifier):
        for i in range(5):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["agent-a"],
                consensus_confidence=0.5 + i * 0.1,
                consensus_text=f"Decision {i}",
            )
        pending = verifier.get_pending_decisions()
        assert len(pending) == 5

    def test_record_replaces_existing(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.5,
            consensus_text="Original",
        )
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude", "gpt4"],
            consensus_confidence=0.9,
            consensus_text="Updated",
        )
        pending = verifier.get_pending_decisions()
        assert len(pending) == 1
        assert pending[0].consensus_confidence == 0.9

    def test_record_with_defaults(self, verifier):
        decision = verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.7,
            consensus_text="Test",
        )
        assert decision.domain == "general"
        assert decision.task == ""


class TestVerify:
    """Tests for verifying decisions with ground truth."""

    def test_verify_correct_decision(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude", "gpt4"],
            consensus_confidence=0.85,
            consensus_text="Deploy",
            domain="deployment",
        )
        result = verifier.verify(
            debate_id="d-1",
            outcome_correct=True,
            signal_type=SignalType.USER_FEEDBACK,
            signal_detail="Deployment succeeded",
        )
        assert result is not None
        assert result.outcome_correct is True
        assert result.debate_id == "d-1"
        assert len(result.agents_updated) == 2
        assert not result.overconfident

    def test_verify_incorrect_high_confidence(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.95,
            consensus_text="Safe to deploy",
        )
        result = verifier.verify(
            debate_id="d-1",
            outcome_correct=False,
            signal_type=SignalType.INCIDENT_REPORT,
            signal_detail="Production outage",
        )
        assert result is not None
        assert not result.outcome_correct
        assert result.overconfident  # 0.95 confidence but wrong

    def test_verify_nonexistent_debate_returns_none(self, verifier):
        result = verifier.verify(
            debate_id="nonexistent",
            outcome_correct=True,
        )
        assert result is None

    def test_verify_already_verified_returns_none(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.8,
            consensus_text="Test",
        )
        result1 = verifier.verify(debate_id="d-1", outcome_correct=True)
        assert result1 is not None

        result2 = verifier.verify(debate_id="d-1", outcome_correct=False)
        assert result2 is None

    def test_verify_marks_decision_as_verified(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.8,
            consensus_text="Test",
        )
        verifier.verify(debate_id="d-1", outcome_correct=True)
        pending = verifier.get_pending_decisions()
        assert len(pending) == 0  # No more unverified

    def test_verify_string_signal_type(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.8,
            consensus_text="Test",
        )
        result = verifier.verify(
            debate_id="d-1",
            outcome_correct=True,
            signal_type="test_result",
        )
        assert result is not None
        assert result.signal_type == SignalType.TEST_RESULT

    def test_brier_scores_computed_correctly(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.9,
            consensus_text="Test",
        )
        # Correct decision: Brier = (0.9 - 1.0)^2 = 0.01
        result = verifier.verify(debate_id="d-1", outcome_correct=True)
        assert abs(result.brier_scores["claude"] - 0.01) < 1e-6

        verifier.record_decision(
            debate_id="d-2",
            agents=["gpt4"],
            consensus_confidence=0.9,
            consensus_text="Test",
        )
        # Incorrect decision: Brier = (0.9 - 0.0)^2 = 0.81
        result2 = verifier.verify(debate_id="d-2", outcome_correct=False)
        assert abs(result2.brier_scores["gpt4"] - 0.81) < 1e-6


class TestCalibrationIntegration:
    """Tests for CalibrationTracker integration."""

    @patch("aragora.debate.outcome_verifier.OutcomeVerifier._update_calibration")
    def test_verify_calls_calibration_update(self, mock_update, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude", "gpt4"],
            consensus_confidence=0.85,
            consensus_text="Test",
            domain="security",
        )
        verifier.verify(debate_id="d-1", outcome_correct=True)

        mock_update.assert_called_once_with(
            ["claude", "gpt4"], 0.85, True, "security", "d-1"
        )

    @patch("aragora.debate.outcome_verifier.OutcomeVerifier._update_elo")
    def test_verify_calls_elo_update(self, mock_elo, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.7,
            consensus_text="Test",
        )
        verifier.verify(debate_id="d-1", outcome_correct=False)

        mock_elo.assert_called_once_with(["claude"], 0.7, False)


class TestSystematicErrors:
    """Tests for detecting systematic calibration errors."""

    def test_no_errors_with_few_verifications(self, verifier):
        verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.9,
            consensus_text="Test",
            domain="security",
        )
        verifier.verify(debate_id="d-1", outcome_correct=False)
        errors = verifier.get_systematic_errors(min_count=5)
        assert len(errors) == 0

    def test_detects_overconfidence(self, verifier):
        # Record many overconfident wrong decisions
        for i in range(10):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["claude"],
                consensus_confidence=0.9,
                consensus_text=f"Decision {i}",
                domain="security",
            )
            verifier.verify(debate_id=f"d-{i}", outcome_correct=i < 3)

        errors = verifier.get_systematic_errors(min_count=5)
        assert len(errors) > 0
        sec_error = next(e for e in errors if e["domain"] == "security")
        assert sec_error["overconfidence"] > 0.0
        assert sec_error["avg_confidence"] > sec_error["success_rate"]

    def test_domain_calibration(self, verifier):
        for i in range(6):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["claude"],
                consensus_confidence=0.8,
                consensus_text="Test",
                domain="medical",
            )
            verifier.verify(debate_id=f"d-{i}", outcome_correct=True)

        stats = verifier.get_domain_calibration("medical")
        assert stats["domain"] == "medical"
        assert stats["total_verifications"] == 6
        assert "claude" in stats["agents"]
        assert stats["agents"]["claude"]["success_rate"] == 1.0


class TestOverallStats:
    """Tests for aggregate statistics."""

    def test_empty_stats(self, verifier):
        stats = verifier.get_overall_stats()
        assert stats["total_verifications"] == 0
        assert stats["pending_decisions"] == 0

    def test_stats_after_verifications(self, verifier):
        for i in range(4):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["claude"],
                consensus_confidence=0.8,
                consensus_text="Test",
            )
            verifier.verify(debate_id=f"d-{i}", outcome_correct=i < 3)

        stats = verifier.get_overall_stats()
        assert stats["total_verifications"] == 4
        assert stats["correct_decisions"] == 3
        assert stats["accuracy"] == 0.75
        assert stats["pending_decisions"] == 0

    def test_pending_count(self, verifier):
        for i in range(3):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["claude"],
                consensus_confidence=0.8,
                consensus_text="Test",
            )
        verifier.verify(debate_id="d-0", outcome_correct=True)

        stats = verifier.get_overall_stats()
        assert stats["total_verifications"] == 1
        assert stats["pending_decisions"] == 2


class TestVerificationHistory:
    """Tests for verification history queries."""

    def test_history_empty(self, verifier):
        history = verifier.get_verification_history()
        assert len(history) == 0

    def test_history_records_verifications(self, verifier):
        for i in range(3):
            verifier.record_decision(
                debate_id=f"d-{i}",
                agents=["claude"],
                consensus_confidence=0.8,
                consensus_text="Test",
                domain="testing",
            )
            verifier.verify(debate_id=f"d-{i}", outcome_correct=True)

        history = verifier.get_verification_history()
        assert len(history) == 3

    def test_history_filter_by_domain(self, verifier):
        for domain in ["security", "medical", "security"]:
            did = f"d-{domain}-{id(domain)}"
            verifier.record_decision(
                debate_id=did,
                agents=["claude"],
                consensus_confidence=0.8,
                consensus_text="Test",
                domain=domain,
            )
            verifier.verify(debate_id=did, outcome_correct=True)

        history = verifier.get_verification_history(domain="security")
        assert len(history) == 2


class TestPendingDecision:
    """Tests for PendingDecision dataclass."""

    def test_to_dict_serializes_agents(self):
        pd = PendingDecision(
            debate_id="d-1",
            agents=["claude", "gpt4"],
            consensus_confidence=0.8,
            consensus_text="Test",
        )
        d = pd.to_dict()
        assert json.loads(d["agents"]) == ["claude", "gpt4"]
        assert d["verified"] == 0

    def test_signal_type_from_string(self):
        assert SignalType("user_feedback") == SignalType.USER_FEEDBACK
        assert SignalType("rollback") == SignalType.ROLLBACK


class TestAsyncOutcomeVerifier:
    """Tests for async wrapper."""

    @pytest.mark.asyncio
    async def test_async_record_and_verify(self, tmp_path):
        verifier = AsyncOutcomeVerifier(db_path=tmp_path / "async_test.db")
        await verifier.record_decision(
            debate_id="d-1",
            agents=["claude"],
            consensus_confidence=0.8,
            consensus_text="Test",
        )
        pending = await verifier.get_pending_decisions()
        assert len(pending) == 1

        result = await verifier.verify(debate_id="d-1", outcome_correct=True)
        assert result is not None
        assert result.outcome_correct

    @pytest.mark.asyncio
    async def test_async_stats(self, tmp_path):
        verifier = AsyncOutcomeVerifier(db_path=tmp_path / "async_stats.db")
        stats = await verifier.get_overall_stats()
        assert stats["total_verifications"] == 0
