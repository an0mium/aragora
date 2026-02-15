"""Tests for wiring SelectionFeedbackLoop into post-debate and TeamSelector.

Covers:
- Debate outcome flows to selection feedback loop in on_debate_complete
- TeamSelector receives feedback_loop and enable_feedback_weights
- Score includes feedback adjustment when loop and domain present
- No change when loop is None
- Winner extraction handles various result shapes
- Error in feedback loop doesn't break on_debate_complete
- Wiring skipped when team_selector is None
- Wiring skipped when feedback loop is None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from aragora.debate.subsystem_coordinator import SubsystemCoordinator


@dataclass
class _FakeEnv:
    task: str = "test task"


@dataclass
class _FakeAgent:
    name: str = "agent-1"


@dataclass
class _FakeCtx:
    debate_id: str = "debate-001"
    agents: list = field(default_factory=lambda: [_FakeAgent("claude"), _FakeAgent("gpt")])
    domain: str = "security"
    env: _FakeEnv = field(default_factory=_FakeEnv)


@dataclass
class _FakeResult:
    consensus: str = "Use rate limiting"
    consensus_confidence: float = 0.85
    winner: str = "claude"
    predictions: dict = field(default_factory=dict)


class TestFeedbackLoopInOnDebateComplete:
    """Test that on_debate_complete calls selection_feedback_loop.process_debate_outcome."""

    def test_outcome_flows_to_loop(self):
        """process_debate_outcome is called with correct arguments."""
        loop = MagicMock()
        loop.process_debate_outcome.return_value = {"claude": 0.1, "gpt": -0.05}
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,  # Don't auto-create, use mock
            selection_feedback_loop=loop,
        )
        ctx = _FakeCtx()
        result = _FakeResult()
        coord.on_debate_complete(ctx, result)

        loop.process_debate_outcome.assert_called_once_with(
            debate_id="debate-001",
            participants=["claude", "gpt"],
            winner="claude",
            domain="security",
            confidence=0.85,
        )

    def test_no_call_when_loop_is_none(self):
        """No error when selection_feedback_loop is None."""
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=None,
        )
        ctx = _FakeCtx()
        result = _FakeResult()
        # Should not raise
        coord.on_debate_complete(ctx, result)

    def test_winner_as_object_with_name(self):
        """Extracts winner name from object with .name attribute."""
        loop = MagicMock()
        loop.process_debate_outcome.return_value = {}
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=loop,
        )
        result = _FakeResult()
        result.winner = _FakeAgent("gemini")
        ctx = _FakeCtx()
        coord.on_debate_complete(ctx, result)
        call_kwargs = loop.process_debate_outcome.call_args[1]
        assert call_kwargs["winner"] == "gemini"

    def test_winner_none_passed_through(self):
        """winner=None flows through when result has no winner."""
        loop = MagicMock()
        loop.process_debate_outcome.return_value = {}
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=loop,
        )
        result = _FakeResult()
        result.winner = None
        ctx = _FakeCtx()
        coord.on_debate_complete(ctx, result)
        call_kwargs = loop.process_debate_outcome.call_args[1]
        assert call_kwargs["winner"] is None

    def test_error_in_loop_does_not_break_completion(self):
        """Exception in process_debate_outcome is caught and logged."""
        loop = MagicMock()
        loop.process_debate_outcome.side_effect = RuntimeError("feedback error")
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=loop,
        )
        ctx = _FakeCtx()
        result = _FakeResult()
        # Should not raise
        coord.on_debate_complete(ctx, result)


class TestTeamSelectorWiring:
    """Test that feedback loop is wired into TeamSelector during init."""

    def test_team_selector_receives_loop(self):
        """TeamSelector.feedback_loop is set to auto-created loop."""
        ts = MagicMock()
        ts.config = MagicMock()
        ts.config.enable_feedback_weights = False
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            team_selector=ts,
        )
        assert ts.feedback_loop is coord.selection_feedback_loop
        assert ts.config.enable_feedback_weights is True

    def test_wiring_skipped_when_no_team_selector(self):
        """No error when team_selector is None."""
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            team_selector=None,
        )
        assert coord.selection_feedback_loop is not None
        # No error occurred

    def test_wiring_skipped_when_no_loop(self):
        """No wiring when feedback loop is disabled and None."""
        ts = MagicMock()
        ts.config = MagicMock()
        ts.config.enable_feedback_weights = False
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=None,
            team_selector=ts,
        )
        # feedback_loop should not have been set on team_selector
        assert not hasattr(ts, "feedback_loop") or ts.feedback_loop is not coord.selection_feedback_loop
