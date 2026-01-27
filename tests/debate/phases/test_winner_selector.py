"""
Tests for winner selector module.

Tests cover:
- WinnerSelector class
- Majority winner determination
- Unanimous winner handling
- No-unanimity fallback
- Belief network analysis
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.winner_selector import WinnerSelector


@dataclass
class MockResult:
    """Mock debate result."""

    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.5
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    winner: str = ""
    votes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    id: str = "test-debate-123"


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "Test task"


@dataclass
class MockDebateContext:
    """Mock debate context."""

    result: MockResult = field(default_factory=MockResult)
    env: MockEnv = field(default_factory=MockEnv)
    proposals: dict = field(default_factory=dict)
    agents: list = field(default_factory=list)
    winner_agent: str = ""
    context_messages: list = field(default_factory=list)


@dataclass
class MockVote:
    """Mock vote."""

    agent: str
    choice: str
    confidence: float = 0.8


@dataclass
class MockAgent:
    """Mock agent."""

    name: str


class TestWinnerSelector:
    """Tests for WinnerSelector class."""

    def test_init_defaults(self):
        """Selector initializes with None defaults."""
        selector = WinnerSelector()

        assert selector.protocol is None
        assert selector.position_tracker is None
        assert selector.calibration_tracker is None
        assert selector.recorder is None

    def test_init_with_dependencies(self):
        """Selector stores injected dependencies."""
        protocol = MagicMock()
        position_tracker = MagicMock()
        notify_spectator = MagicMock()

        selector = WinnerSelector(
            protocol=protocol,
            position_tracker=position_tracker,
            notify_spectator=notify_spectator,
        )

        assert selector.protocol is protocol
        assert selector.position_tracker is position_tracker
        assert selector._notify_spectator is notify_spectator


class TestDetermineMajorityWinner:
    """Tests for determine_majority_winner method."""

    def normalize_choice(self, choice: str, agents: list, proposals: dict) -> str:
        """Helper to normalize choice."""
        if choice in proposals:
            return choice
        for agent in agents:
            if getattr(agent, "name", None) == choice:
                return choice
        return choice

    def test_empty_votes(self):
        """Empty votes falls back to first proposal."""
        ctx = MockDebateContext(proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"})
        selector = WinnerSelector()

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={},
            total_votes=0.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        # Should pick first proposal with zero confidence (no votes)
        assert ctx.result.final_answer in ctx.proposals.values()
        assert ctx.result.consensus_reached is False
        assert ctx.result.confidence == 0.0  # No votes = no confidence

    def test_majority_winner(self):
        """Majority winner is selected correctly."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
            agents=[MockAgent("agent1"), MockAgent("agent2")],
        )
        selector = WinnerSelector(protocol=MagicMock(consensus_threshold=0.5))

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 3.0, "agent2": 1.0},
            total_votes=4.0,
            choice_mapping={"agent1": "agent1", "agent2": "agent2"},
            normalize_choice=self.normalize_choice,
        )

        assert ctx.result.winner == "agent1"
        assert ctx.result.final_answer == "Proposal 1"
        assert ctx.result.consensus_reached is True  # 75% >= 50%
        assert ctx.result.confidence == 0.75

    def test_majority_threshold_not_met(self):
        """Consensus not reached if below threshold."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
            agents=[MockAgent("agent1"), MockAgent("agent2")],
        )
        selector = WinnerSelector(protocol=MagicMock(consensus_threshold=0.8))

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 3.0, "agent2": 2.0},
            total_votes=5.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        assert ctx.result.consensus_reached is False  # 60% < 80%
        assert ctx.result.confidence == 0.6

    def test_threshold_override(self):
        """Threshold override takes precedence."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1"},
            agents=[MockAgent("agent1")],
        )
        selector = WinnerSelector(protocol=MagicMock(consensus_threshold=0.9))

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 3.0},
            total_votes=5.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
            threshold_override=0.5,  # Lower threshold
        )

        assert ctx.result.consensus_reached is True  # 60% >= 50%

    def test_consensus_strength_unanimous(self):
        """Single choice has unanimous strength."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1"},
            agents=[MockAgent("agent1")],
        )
        selector = WinnerSelector()

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 5.0},
            total_votes=5.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        assert ctx.result.consensus_strength == "unanimous"
        assert ctx.result.consensus_variance == 0.0

    def test_consensus_strength_strong(self):
        """Low variance has strong strength."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
            agents=[MockAgent("agent1"), MockAgent("agent2")],
        )
        selector = WinnerSelector()

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 5.0, "agent2": 5.0},
            total_votes=10.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        assert ctx.result.consensus_strength == "strong"

    def test_consensus_strength_weak(self):
        """High variance has weak strength."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
            agents=[MockAgent("agent1"), MockAgent("agent2")],
        )
        selector = WinnerSelector()

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 9.0, "agent2": 1.0},
            total_votes=10.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        assert ctx.result.consensus_strength == "weak"

    def test_dissenting_views_tracked(self):
        """Non-winner proposals are tracked as dissenting."""
        ctx = MockDebateContext(
            proposals={
                "agent1": "Proposal 1",
                "agent2": "Proposal 2",
                "agent3": "Proposal 3",
            },
            agents=[],
        )
        selector = WinnerSelector()

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 3.0},
            total_votes=3.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        assert len(ctx.result.dissenting_views) == 2

    def test_spectator_notification(self):
        """Spectator is notified of consensus."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1"},
            agents=[],
        )
        notify = MagicMock()
        selector = WinnerSelector(notify_spectator=notify)

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 1.0},
            total_votes=1.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        notify.assert_called_once()
        assert "consensus" in str(notify.call_args)

    def test_recorder_called(self):
        """Recorder records phase change."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1"},
            agents=[],
        )
        recorder = MagicMock()
        selector = WinnerSelector(recorder=recorder)

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 1.0},
            total_votes=1.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        recorder.record_phase_change.assert_called()

    def test_position_tracker_finalization(self):
        """Position tracker is finalized on consensus."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1"},
            agents=[],
        )
        position_tracker = MagicMock()
        selector = WinnerSelector(position_tracker=position_tracker)

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"agent1": 1.0},
            total_votes=1.0,
            choice_mapping={},
            normalize_choice=self.normalize_choice,
        )

        position_tracker.finalize_debate.assert_called()

    def test_calibration_predictions_recorded(self):
        """Calibration predictions are recorded for votes."""
        vote = MockVote("agent1", "winner", 0.9)
        ctx = MockDebateContext(
            proposals={"winner": "Proposal 1"},
            agents=[],
        )
        ctx.result.votes = [vote]
        calibration = MagicMock()
        selector = WinnerSelector(
            calibration_tracker=calibration,
            extract_debate_domain=lambda: "general",
        )

        selector.determine_majority_winner(
            ctx=ctx,
            vote_counts={"winner": 1.0},
            total_votes=1.0,
            choice_mapping={"winner": "winner"},
            normalize_choice=self.normalize_choice,
        )

        calibration.record_prediction.assert_called()


class TestSetUnanimousWinner:
    """Tests for set_unanimous_winner method."""

    def test_unanimous_winner(self):
        """Sets result for unanimous consensus."""
        ctx = MockDebateContext(
            proposals={"agent1": "Unanimous Proposal"},
        )
        selector = WinnerSelector()

        selector.set_unanimous_winner(
            ctx=ctx,
            winner="agent1",
            unanimity_ratio=1.0,
            total_voters=3,
            count=3,
        )

        assert ctx.result.final_answer == "Unanimous Proposal"
        assert ctx.result.consensus_reached is True
        assert ctx.result.confidence == 1.0
        assert ctx.result.consensus_strength == "unanimous"
        assert ctx.result.consensus_variance == 0.0
        assert ctx.result.winner == "agent1"
        assert ctx.winner_agent == "agent1"

    def test_spectator_notified(self):
        """Spectator is notified of unanimous consensus."""
        ctx = MockDebateContext(proposals={"agent1": "P1"})
        notify = MagicMock()
        selector = WinnerSelector(notify_spectator=notify)

        selector.set_unanimous_winner(ctx, "agent1", 1.0, 3, 3)

        notify.assert_called()
        assert "Unanimous" in str(notify.call_args)

    def test_recorder_called(self):
        """Recorder records unanimous consensus."""
        ctx = MockDebateContext(proposals={"agent1": "P1"})
        recorder = MagicMock()
        selector = WinnerSelector(recorder=recorder)

        selector.set_unanimous_winner(ctx, "agent1", 1.0, 3, 3)

        recorder.record_phase_change.assert_called()

    def test_calibration_recorded(self):
        """Calibration predictions recorded for unanimous."""
        ctx = MockDebateContext(proposals={"agent1": "P1"})
        ctx.result.votes = [MockVote("v1", "agent1")]
        calibration = MagicMock()
        selector = WinnerSelector(
            calibration_tracker=calibration,
            extract_debate_domain=lambda: "testing",
        )

        selector.set_unanimous_winner(ctx, "agent1", 1.0, 3, 3)

        calibration.record_prediction.assert_called()


class TestSetNoUnanimity:
    """Tests for set_no_unanimity method."""

    def test_no_unanimity_message(self):
        """Sets appropriate message when no unanimity."""
        ctx = MockDebateContext(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
        )
        ctx.result.votes = [
            MockVote("v1", "agent1"),
            MockVote("v2", "agent2"),
        ]
        selector = WinnerSelector()

        selector.set_no_unanimity(
            ctx=ctx,
            winner="agent1",
            unanimity_ratio=0.6,
            total_voters=5,
            count=3,
            choice_mapping={"agent1": "agent1", "agent2": "agent2"},
        )

        assert "[No unanimous consensus reached]" in ctx.result.final_answer
        assert ctx.result.consensus_reached is False
        assert ctx.result.confidence == 0.6
        assert ctx.result.consensus_strength == "none"

    def test_dissenting_views_tracked(self):
        """All views tracked as dissenting when no unanimity."""
        ctx = MockDebateContext(
            proposals={"agent1": "P1", "agent2": "P2"},
        )
        ctx.result.votes = []
        selector = WinnerSelector()

        selector.set_no_unanimity(ctx, "agent1", 0.6, 5, 3, {})

        assert len(ctx.result.dissenting_views) == 2

    def test_spectator_notified(self):
        """Spectator notified of no unanimity."""
        ctx = MockDebateContext(proposals={"agent1": "P1"})
        ctx.result.votes = []
        notify = MagicMock()
        selector = WinnerSelector(notify_spectator=notify)

        selector.set_no_unanimity(ctx, "agent1", 0.6, 5, 3, {})

        notify.assert_called()
        assert "No unanimity" in str(notify.call_args)


class TestAnalyzeBeliefNetwork:
    """Tests for analyze_belief_network method."""

    def test_no_analyzer_returns_early(self):
        """No-op without belief analyzer."""
        ctx = MockDebateContext()
        selector = WinnerSelector()

        # Should not raise
        selector.analyze_belief_network(ctx)

    def test_empty_messages_returns_early(self):
        """No-op with empty messages."""
        ctx = MockDebateContext()
        ctx.result.messages = []
        selector = WinnerSelector(get_belief_analyzer=lambda: (None, None))

        selector.analyze_belief_network(ctx)

    @patch("aragora.reasoning.crux_detector.CruxDetector")
    def test_belief_analysis_called(self, mock_crux_detector_cls):
        """Belief network is analyzed when available."""
        mock_message = MagicMock()
        mock_message.role = "proposer"
        mock_message.content = "Test content"
        mock_message.agent = "agent1"

        ctx = MockDebateContext()
        ctx.result.messages = [mock_message]

        # Mock CruxDetector to return empty cruxes
        mock_analysis = MagicMock()
        mock_analysis.cruxes = []
        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = mock_analysis
        mock_crux_detector_cls.return_value = mock_detector

        mock_network = MagicMock()

        def get_analyzer():
            bn_class = MagicMock(return_value=mock_network)
            return (bn_class, MagicMock())

        selector = WinnerSelector(get_belief_analyzer=get_analyzer)

        selector.analyze_belief_network(ctx)

        mock_network.add_claim.assert_called()

    @patch("aragora.reasoning.crux_detector.CruxDetector")
    def test_cruxes_stored_in_result(self, mock_crux_detector_cls):
        """Identified cruxes are stored in result."""
        mock_message = MagicMock()
        mock_message.role = "critic"
        mock_message.content = "Critique content"
        mock_message.agent = "critic1"

        ctx = MockDebateContext()
        ctx.result.messages = [mock_message]

        # Create mock crux with correct attributes (statement, contesting_agents)
        mock_crux = MagicMock()
        mock_crux.statement = "Key disagreement"
        mock_crux.crux_score = 0.8
        mock_crux.contesting_agents = ["agent1", "agent2"]

        # Mock CruxDetector and its detect_cruxes method
        mock_analysis = MagicMock()
        mock_analysis.cruxes = [mock_crux]
        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = mock_analysis
        mock_crux_detector_cls.return_value = mock_detector

        mock_network = MagicMock()

        def get_analyzer():
            bn_class = MagicMock(return_value=mock_network)
            return (bn_class, MagicMock())

        selector = WinnerSelector(get_belief_analyzer=get_analyzer)

        selector.analyze_belief_network(ctx)

        assert hasattr(ctx.result, "cruxes")

    def test_belief_analysis_error_handled(self):
        """Errors in belief analysis are handled gracefully."""
        mock_message = MagicMock()
        mock_message.role = "proposer"
        mock_message.content = "Content"
        mock_message.agent = "agent1"

        ctx = MockDebateContext()
        ctx.result.messages = [mock_message]

        def get_analyzer():
            bn_class = MagicMock(side_effect=Exception("Belief error"))
            return (bn_class, MagicMock())

        selector = WinnerSelector(get_belief_analyzer=get_analyzer)

        # Should not raise
        selector.analyze_belief_network(ctx)
