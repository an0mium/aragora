"""Tests for bidirectional calibration feedback (Sprint 16B)."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from aragora.debate.phases.feedback_phase import FeedbackPhase


def _make_vote(agent: str, choice: str, confidence: float) -> MagicMock:
    """Create a mock vote."""
    vote = MagicMock()
    vote.agent = agent
    vote.choice = choice
    vote.confidence = confidence
    return vote


def _make_ctx(
    consensus_reached: bool = True,
    winner: str = "claude",
    votes: list | None = None,
    domain: str = "general",
) -> MagicMock:
    """Create a mock DebateContext."""
    ctx = MagicMock()
    ctx.debate_id = "test-debate-1"
    ctx.domain = domain
    ctx.choice_mapping = {}  # No remapping

    result = MagicMock()
    result.consensus_reached = consensus_reached
    result.winner = winner
    result.votes = votes or []
    result.debate_id = "test-debate-1"
    ctx.result = result

    return ctx


class TestCalibrationFeedback:
    """Test bidirectional calibration feedback in FeedbackPhase."""

    def test_correct_agent_records_positive_prediction(self):
        """Agent who voted correctly with high confidence gets positive record."""
        tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=tracker)

        votes = [_make_vote("claude", "claude", 0.9)]
        ctx = _make_ctx(consensus_reached=True, winner="claude", votes=votes)

        phase._apply_calibration_feedback(ctx)

        tracker.record_prediction.assert_called_once_with(
            agent="claude",
            confidence=0.9,
            correct=True,
            domain="general",
            debate_id="test-debate-1",
            prediction_type="consensus_feedback",
        )

    def test_incorrect_agent_records_negative_prediction(self):
        """Agent who voted incorrectly with high confidence gets negative record."""
        tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=tracker)

        votes = [_make_vote("gpt4", "gpt4", 0.85)]
        ctx = _make_ctx(consensus_reached=True, winner="claude", votes=votes)

        phase._apply_calibration_feedback(ctx)

        tracker.record_prediction.assert_called_once_with(
            agent="gpt4",
            confidence=0.85,
            correct=False,
            domain="general",
            debate_id="test-debate-1",
            prediction_type="consensus_feedback",
        )

    def test_skipped_when_no_consensus(self):
        """No calibration update when consensus not reached."""
        tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=tracker)

        votes = [_make_vote("claude", "claude", 0.9)]
        ctx = _make_ctx(consensus_reached=False, winner=None, votes=votes)

        phase._apply_calibration_feedback(ctx)

        tracker.record_prediction.assert_not_called()

    def test_skipped_when_no_calibration_tracker(self):
        """No error when calibration tracker is None."""
        phase = FeedbackPhase(calibration_tracker=None)
        ctx = _make_ctx()

        # Should not raise
        phase._apply_calibration_feedback(ctx)

    def test_low_confidence_votes_skipped(self):
        """Votes with confidence <= 0.7 are not recorded."""
        tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=tracker)

        votes = [
            _make_vote("claude", "claude", 0.5),  # Below threshold
            _make_vote("gpt4", "claude", 0.7),  # At threshold (excluded)
        ]
        ctx = _make_ctx(consensus_reached=True, winner="claude", votes=votes)

        phase._apply_calibration_feedback(ctx)

        tracker.record_prediction.assert_not_called()

    def test_multiple_agents_all_recorded(self):
        """Multiple high-confidence voters all get calibration records."""
        tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=tracker)

        votes = [
            _make_vote("claude", "claude", 0.9),
            _make_vote("gpt4", "gpt4", 0.85),
            _make_vote("gemini", "claude", 0.75),
        ]
        ctx = _make_ctx(consensus_reached=True, winner="claude", votes=votes)

        phase._apply_calibration_feedback(ctx)

        assert tracker.record_prediction.call_count == 3

        # claude voted correctly
        tracker.record_prediction.assert_any_call(
            agent="claude",
            confidence=0.9,
            correct=True,
            domain="general",
            debate_id="test-debate-1",
            prediction_type="consensus_feedback",
        )

        # gpt4 voted incorrectly
        tracker.record_prediction.assert_any_call(
            agent="gpt4",
            confidence=0.85,
            correct=False,
            domain="general",
            debate_id="test-debate-1",
            prediction_type="consensus_feedback",
        )
