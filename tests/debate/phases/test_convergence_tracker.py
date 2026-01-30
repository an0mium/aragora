"""
Tests for convergence tracker module.

Tests cover:
- ConvergenceResult dataclass
- DebateConvergenceTracker class initialization
- check_convergence method
- track_novelty method
- check_rlm_ready_quorum method
- Trickster integration for hollow consensus detection
- State management (reset, properties)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.convergence_tracker import (
    ConvergenceResult,
    DebateConvergenceTracker,
)
from aragora.debate.phases.ready_signal import (
    AgentReadinessSignal,
    CollectiveReadiness,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockEnv:
    """Mock environment for testing."""

    task: str = "What is the best approach to testing?"


@dataclass
class MockResult:
    """Mock debate result for testing."""

    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    final_answer: str = ""
    convergence_status: str = ""
    convergence_similarity: float = 0.0
    per_agent_similarity: dict = field(default_factory=dict)


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    result: MockResult = field(default_factory=MockResult)
    env: MockEnv = field(default_factory=MockEnv)
    proposals: dict = field(default_factory=dict)
    context_messages: list = field(default_factory=list)
    debate_id: str = "test-debate-123"
    per_agent_novelty: dict = field(default_factory=dict)
    avg_novelty: float = 0.0
    low_novelty_agents: list = field(default_factory=list)


@dataclass
class MockConvergenceResult:
    """Mock convergence result from detector."""

    converged: bool = False
    status: str = "refining"
    avg_similarity: float = 0.5
    per_agent_similarity: dict = field(default_factory=dict)


class MockInterventionType(Enum):
    """Mock intervention type enum."""

    EVIDENCE_CHALLENGE = "evidence_challenge"
    NOVELTY_CHALLENGE = "novelty_challenge"


@dataclass
class MockTricksterIntervention:
    """Mock trickster intervention."""

    intervention_type: MockInterventionType = MockInterventionType.EVIDENCE_CHALLENGE
    challenge_text: str = "Challenge the evidence quality"
    target_agents: list = field(default_factory=list)
    priority: float = 0.6
    evidence_gaps: dict = field(default_factory=dict)


@dataclass
class MockNoveltyResult:
    """Mock novelty result from tracker."""

    round_num: int = 1
    per_agent_novelty: dict = field(default_factory=dict)
    avg_novelty: float = 0.5
    min_novelty: float = 0.3
    max_novelty: float = 0.8
    low_novelty_agents: list = field(default_factory=list)

    def has_low_novelty(self) -> bool:
        """Check if any agent has low novelty."""
        return len(self.low_novelty_agents) > 0


# =============================================================================
# ConvergenceResult Tests
# =============================================================================


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_default_values(self):
        """ConvergenceResult has sensible default values."""
        result = ConvergenceResult()

        assert result.converged is False
        assert result.status == ""
        assert result.similarity == 0.0
        assert result.blocked_by_trickster is False

    def test_converged_result(self):
        """ConvergenceResult can represent converged state."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            similarity=0.92,
            blocked_by_trickster=False,
        )

        assert result.converged is True
        assert result.status == "converged"
        assert result.similarity == 0.92

    def test_blocked_by_trickster(self):
        """ConvergenceResult can indicate trickster blocked convergence."""
        result = ConvergenceResult(
            converged=False,
            status="blocked_hollow",
            similarity=0.85,
            blocked_by_trickster=True,
        )

        assert result.converged is False
        assert result.blocked_by_trickster is True

    def test_refining_state(self):
        """ConvergenceResult can represent refining state."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            similarity=0.65,
        )

        assert result.converged is False
        assert result.status == "refining"


# =============================================================================
# DebateConvergenceTracker Initialization Tests
# =============================================================================


class TestDebateConvergenceTrackerInit:
    """Tests for DebateConvergenceTracker initialization."""

    def test_init_with_no_arguments(self):
        """Tracker initializes with defaults when no arguments provided."""
        tracker = DebateConvergenceTracker()

        assert tracker.convergence_detector is None
        assert tracker.novelty_tracker is None
        assert tracker.trickster is None
        assert tracker.hooks == {}
        assert tracker.event_emitter is None
        assert tracker._notify_spectator is None
        assert tracker._inject_challenge is None

    def test_init_with_all_dependencies(self):
        """Tracker stores all injected dependencies."""
        convergence_detector = MagicMock()
        novelty_tracker = MagicMock()
        trickster = MagicMock()
        hooks = {"on_convergence_check": MagicMock()}
        event_emitter = MagicMock()
        notify_spectator = MagicMock()
        inject_challenge = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=convergence_detector,
            novelty_tracker=novelty_tracker,
            trickster=trickster,
            hooks=hooks,
            event_emitter=event_emitter,
            notify_spectator=notify_spectator,
            inject_challenge=inject_challenge,
        )

        assert tracker.convergence_detector is convergence_detector
        assert tracker.novelty_tracker is novelty_tracker
        assert tracker.trickster is trickster
        assert tracker.hooks is hooks
        assert tracker.event_emitter is event_emitter
        assert tracker._notify_spectator is notify_spectator
        assert tracker._inject_challenge is inject_challenge

    def test_hooks_defaults_to_empty_dict(self):
        """Hooks defaults to empty dict when None passed."""
        tracker = DebateConvergenceTracker(hooks=None)

        assert tracker.hooks == {}

    def test_internal_state_initialized(self):
        """Tracker initializes internal state correctly."""
        tracker = DebateConvergenceTracker()

        assert tracker._previous_round_responses == {}
        assert isinstance(tracker._collective_readiness, CollectiveReadiness)


# =============================================================================
# reset Method Tests
# =============================================================================


class TestDebateConvergenceTrackerReset:
    """Tests for reset method."""

    def test_reset_clears_previous_responses(self):
        """Reset clears previous round responses."""
        tracker = DebateConvergenceTracker()
        tracker._previous_round_responses = {"agent1": "response1"}

        tracker.reset()

        assert tracker._previous_round_responses == {}

    def test_reset_creates_new_collective_readiness(self):
        """Reset creates a new CollectiveReadiness instance."""
        tracker = DebateConvergenceTracker()
        original_readiness = tracker._collective_readiness
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="test", confidence=0.9, ready=True)
        )

        tracker.reset()

        assert tracker._collective_readiness is not original_readiness
        assert tracker._collective_readiness.total_count == 0


# =============================================================================
# check_convergence Method Tests
# =============================================================================


class TestCheckConvergence:
    """Tests for check_convergence method."""

    def test_returns_default_result_when_no_detector(self):
        """Returns default ConvergenceResult when no detector is configured."""
        tracker = DebateConvergenceTracker()
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        result = tracker.check_convergence(ctx, round_num=1)

        assert result.converged is False
        assert result.status == ""
        assert result.similarity == 0.0

    def test_returns_default_on_first_round(self):
        """Returns default result on first round (no previous responses)."""
        detector = MagicMock()
        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        result = tracker.check_convergence(ctx, round_num=1)

        assert result.converged is False
        # Detector should not be called on first round
        detector.check_convergence.assert_not_called()

    def test_stores_first_round_responses(self):
        """First round stores responses for comparison."""
        detector = MagicMock()
        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1", "agent2": "proposal 2"}

        tracker.check_convergence(ctx, round_num=1)

        assert tracker._previous_round_responses == ctx.proposals

    def test_calls_detector_on_subsequent_rounds(self):
        """Calls detector on rounds after the first."""
        mock_convergence = MockConvergenceResult(
            converged=False,
            status="refining",
            avg_similarity=0.6,
            per_agent_similarity={"agent1": 0.6},
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        detector.check_convergence.assert_called_once()

    def test_updates_context_result_on_convergence(self):
        """Updates context result with convergence information."""
        mock_convergence = MockConvergenceResult(
            converged=True,
            status="converged",
            avg_similarity=0.9,
            per_agent_similarity={"agent1": 0.9},
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        assert ctx.result.convergence_status == "converged"
        assert ctx.result.convergence_similarity == 0.9

    def test_notifies_spectator_on_convergence(self):
        """Notifies spectator about convergence check."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence
        notify_spectator = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            notify_spectator=notify_spectator,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        notify_spectator.assert_called_once()
        call_args = notify_spectator.call_args
        assert call_args[0][0] == "convergence"

    def test_calls_on_convergence_check_hook(self):
        """Calls on_convergence_check hook when present."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence
        hook = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            hooks={"on_convergence_check": hook},
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        hook.assert_called_once()

    def test_returns_converged_true_when_converged(self):
        """Returns converged=True when detector indicates convergence."""
        mock_convergence = MockConvergenceResult(
            converged=True, status="converged", avg_similarity=0.95
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        result = tracker.check_convergence(ctx, round_num=2)

        assert result.converged is True

    def test_returns_default_when_detector_returns_none(self):
        """Returns default result when detector returns None."""
        detector = MagicMock()
        detector.check_convergence.return_value = None

        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        result = tracker.check_convergence(ctx, round_num=2)

        assert result.converged is False
        assert result.status == ""


# =============================================================================
# Trickster Integration Tests
# =============================================================================


class TestTricksterIntegration:
    """Tests for trickster integration in check_convergence."""

    def test_trickster_called_when_similarity_above_threshold(self):
        """Trickster is called when similarity exceeds 0.5."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = None

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        trickster.check_and_intervene.assert_called_once()

    def test_trickster_not_called_when_similarity_below_threshold(self):
        """Trickster is not called when similarity is below 0.5."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="diverging", avg_similarity=0.3
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence
        trickster = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "very different proposal"}
        tracker.check_convergence(ctx, round_num=2)

        trickster.check_and_intervene.assert_not_called()

    def test_trickster_blocks_hollow_consensus(self):
        """High priority intervention blocks convergence."""
        mock_convergence = MockConvergenceResult(
            converged=True, status="converged", avg_similarity=0.9
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention(priority=0.7)
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        result = tracker.check_convergence(ctx, round_num=2)

        assert result.blocked_by_trickster is True
        assert result.converged is False

    def test_low_priority_intervention_does_not_block(self):
        """Low priority intervention does not block convergence."""
        mock_convergence = MockConvergenceResult(
            converged=True, status="converged", avg_similarity=0.9
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention(priority=0.3)
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        result = tracker.check_convergence(ctx, round_num=2)

        assert result.blocked_by_trickster is False
        assert result.converged is True

    def test_trickster_calls_inject_challenge(self):
        """Trickster intervention triggers inject_challenge callback."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention(challenge_text="Challenge the evidence quality")
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        inject_challenge = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
            inject_challenge=inject_challenge,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        inject_challenge.assert_called_once()

    def test_trickster_emits_events(self):
        """Trickster intervention emits events via event_emitter."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention()
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        event_emitter = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
            event_emitter=event_emitter,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        assert event_emitter.emit_sync.call_count == 2

    def test_trickster_calls_hooks(self):
        """Trickster intervention calls relevant hooks."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention()
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        on_hollow = MagicMock()
        on_intervention = MagicMock()

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
            hooks={
                "on_hollow_consensus": on_hollow,
                "on_trickster_intervention": on_intervention,
            },
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        on_hollow.assert_called_once()
        on_intervention.assert_called_once()


# =============================================================================
# track_novelty Method Tests
# =============================================================================


class TestTrackNovelty:
    """Tests for track_novelty method."""

    def test_does_nothing_without_tracker(self):
        """Does nothing when no novelty_tracker is configured."""
        tracker = DebateConvergenceTracker()
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        # Should not raise
        tracker.track_novelty(ctx, round_num=1)

    def test_does_nothing_with_empty_proposals(self):
        """Does nothing when proposals are empty."""
        novelty_tracker = MagicMock()
        tracker = DebateConvergenceTracker(novelty_tracker=novelty_tracker)
        ctx = MockDebateContext()
        ctx.proposals = {}

        tracker.track_novelty(ctx, round_num=1)

        novelty_tracker.compute_novelty.assert_not_called()

    def test_calls_novelty_tracker_compute(self):
        """Calls novelty tracker compute_novelty method."""
        novelty_result = MockNoveltyResult(
            per_agent_novelty={"agent1": 0.8},
            avg_novelty=0.8,
        )
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        tracker = DebateConvergenceTracker(novelty_tracker=novelty_tracker)
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        tracker.track_novelty(ctx, round_num=1)

        novelty_tracker.compute_novelty.assert_called_once()

    def test_updates_context_with_novelty(self):
        """Updates context with novelty scores."""
        novelty_result = MockNoveltyResult(
            per_agent_novelty={"agent1": 0.8, "agent2": 0.6},
            avg_novelty=0.7,
            low_novelty_agents=["agent2"],
        )
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        tracker = DebateConvergenceTracker(novelty_tracker=novelty_tracker)
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1", "agent2": "proposal 2"}

        tracker.track_novelty(ctx, round_num=1)

        assert ctx.avg_novelty == 0.7
        assert ctx.low_novelty_agents == ["agent2"]

    def test_adds_to_history_after_compute(self):
        """Adds proposals to history after computing novelty."""
        novelty_result = MockNoveltyResult()
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        tracker = DebateConvergenceTracker(novelty_tracker=novelty_tracker)
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        tracker.track_novelty(ctx, round_num=1)

        novelty_tracker.add_to_history.assert_called_once()

    def test_notifies_spectator_about_novelty(self):
        """Notifies spectator about novelty scores."""
        novelty_result = MockNoveltyResult(avg_novelty=0.75)
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        notify_spectator = MagicMock()

        tracker = DebateConvergenceTracker(
            novelty_tracker=novelty_tracker,
            notify_spectator=notify_spectator,
        )
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        tracker.track_novelty(ctx, round_num=1)

        notify_spectator.assert_called_once()
        call_args = notify_spectator.call_args
        assert call_args[0][0] == "novelty"

    def test_calls_on_novelty_check_hook(self):
        """Calls on_novelty_check hook when present."""
        novelty_result = MockNoveltyResult()
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        hook = MagicMock()

        tracker = DebateConvergenceTracker(
            novelty_tracker=novelty_tracker,
            hooks={"on_novelty_check": hook},
        )
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "proposal 1"}

        tracker.track_novelty(ctx, round_num=1)

        hook.assert_called_once()

    def test_triggers_trickster_on_low_novelty(self):
        """Triggers trickster novelty challenge on low novelty."""
        novelty_result = MockNoveltyResult(
            low_novelty_agents=["agent1"],
            per_agent_novelty={"agent1": 0.1},
        )
        novelty_tracker = MagicMock()
        novelty_tracker.compute_novelty.return_value = novelty_result

        trickster = MagicMock()
        trickster.create_novelty_challenge.return_value = MockTricksterIntervention(
            intervention_type=MockInterventionType.NOVELTY_CHALLENGE
        )

        tracker = DebateConvergenceTracker(
            novelty_tracker=novelty_tracker,
            trickster=trickster,
        )
        ctx = MockDebateContext()
        ctx.proposals = {"agent1": "same old proposal"}

        tracker.track_novelty(ctx, round_num=2)

        trickster.create_novelty_challenge.assert_called_once()


# =============================================================================
# check_rlm_ready_quorum Method Tests
# =============================================================================


class TestCheckRlmReadyQuorum:
    """Tests for check_rlm_ready_quorum method."""

    def test_parses_ready_signals_from_proposals(self):
        """Parses ready signals from proposal content."""
        tracker = DebateConvergenceTracker()
        ctx = MockDebateContext()
        ctx.proposals = {
            "agent1": 'My position is final. <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
        }

        with patch("aragora.debate.phases.convergence_tracker.parse_ready_signal") as mock_parse:
            mock_parse.return_value = AgentReadinessSignal(
                agent="agent1", confidence=0.9, ready=True
            )
            tracker.check_rlm_ready_quorum(ctx, round_num=1)

            mock_parse.assert_called_once()

    def test_returns_false_when_no_quorum(self):
        """Returns False when quorum is not reached."""
        tracker = DebateConvergenceTracker()
        ctx = MockDebateContext()
        ctx.proposals = {
            "agent1": "Still thinking...",
            "agent2": "Need more discussion.",
        }

        result = tracker.check_rlm_ready_quorum(ctx, round_num=1)

        assert result is False

    def test_returns_true_when_quorum_reached(self):
        """Returns True when enough agents signal ready."""
        tracker = DebateConvergenceTracker()
        # Pre-populate with ready signals
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent1", confidence=0.9, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent2", confidence=0.85, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent3", confidence=0.88, ready=True)
        )

        ctx = MockDebateContext()
        ctx.proposals = {
            "agent4": 'Final. <!-- READY_SIGNAL: {"confidence": 0.95, "ready": true} -->',
        }

        with patch("aragora.debate.phases.convergence_tracker.parse_ready_signal") as mock_parse:
            mock_parse.return_value = AgentReadinessSignal(
                agent="agent4", confidence=0.95, ready=True
            )
            result = tracker.check_rlm_ready_quorum(ctx, round_num=3)

        assert result is True

    def test_notifies_spectator_on_quorum(self):
        """Notifies spectator when quorum is reached."""
        notify_spectator = MagicMock()
        tracker = DebateConvergenceTracker(notify_spectator=notify_spectator)

        # Pre-populate with ready signals to reach quorum
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent1", confidence=0.9, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent2", confidence=0.85, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent3", confidence=0.88, ready=True)
        )

        ctx = MockDebateContext()
        ctx.proposals = {"agent4": "Final position."}

        with patch("aragora.debate.phases.convergence_tracker.parse_ready_signal") as mock_parse:
            mock_parse.return_value = AgentReadinessSignal(
                agent="agent4", confidence=0.95, ready=True
            )
            tracker.check_rlm_ready_quorum(ctx, round_num=3)

        notify_spectator.assert_called_once()
        call_args = notify_spectator.call_args
        assert call_args[0][0] == "rlm_ready"

    def test_calls_on_rlm_ready_quorum_hook(self):
        """Calls on_rlm_ready_quorum hook when quorum reached."""
        hook = MagicMock()
        tracker = DebateConvergenceTracker(hooks={"on_rlm_ready_quorum": hook})

        # Pre-populate with ready signals to reach quorum
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent1", confidence=0.9, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent2", confidence=0.85, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent3", confidence=0.88, ready=True)
        )

        ctx = MockDebateContext()
        ctx.proposals = {"agent4": "Final position."}

        with patch("aragora.debate.phases.convergence_tracker.parse_ready_signal") as mock_parse:
            mock_parse.return_value = AgentReadinessSignal(
                agent="agent4", confidence=0.95, ready=True
            )
            tracker.check_rlm_ready_quorum(ctx, round_num=3)

        hook.assert_called_once()


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for tracker properties."""

    def test_collective_readiness_property(self):
        """collective_readiness property returns internal state."""
        tracker = DebateConvergenceTracker()
        signal = AgentReadinessSignal(agent="test", confidence=0.9, ready=True)
        tracker._collective_readiness.update(signal)

        readiness = tracker.collective_readiness

        assert readiness.total_count == 1
        assert "test" in readiness.signals

    def test_previous_responses_property(self):
        """previous_responses property returns internal state."""
        tracker = DebateConvergenceTracker()
        tracker._previous_round_responses = {"agent1": "response1"}

        responses = tracker.previous_responses

        assert responses == {"agent1": "response1"}

    def test_previous_responses_property_is_reference(self):
        """previous_responses property returns the actual dict reference."""
        tracker = DebateConvergenceTracker()
        tracker._previous_round_responses = {"agent1": "response1"}

        responses = tracker.previous_responses
        responses["agent2"] = "response2"

        # Should affect the internal state
        assert "agent2" in tracker._previous_round_responses


# =============================================================================
# Metrics Recording Tests
# =============================================================================


class TestMetricsRecording:
    """Tests for metrics recording integration."""

    @patch("aragora.debate.phases.convergence_tracker.record_convergence_check")
    def test_records_convergence_check_metric(self, mock_record):
        """Records convergence check metric."""
        mock_convergence = MockConvergenceResult(
            converged=False, status="refining", avg_similarity=0.7
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        tracker = DebateConvergenceTracker(convergence_detector=detector)
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        mock_record.assert_called_once_with(status="refining", blocked=False)

    @patch("aragora.debate.phases.convergence_tracker.record_convergence_check")
    def test_records_blocked_convergence_metric(self, mock_record):
        """Records blocked convergence when trickster intervenes."""
        mock_convergence = MockConvergenceResult(
            converged=True, status="converged", avg_similarity=0.9
        )
        detector = MagicMock()
        detector.check_convergence.return_value = mock_convergence

        intervention = MockTricksterIntervention(priority=0.7)
        trickster = MagicMock()
        trickster.check_and_intervene.return_value = intervention

        tracker = DebateConvergenceTracker(
            convergence_detector=detector,
            trickster=trickster,
        )
        ctx = MockDebateContext()

        # First round
        ctx.proposals = {"agent1": "proposal 1"}
        tracker.check_convergence(ctx, round_num=1)

        # Second round
        ctx.proposals = {"agent1": "refined proposal 1"}
        tracker.check_convergence(ctx, round_num=2)

        # Should record both the initial check and the blocked check
        assert mock_record.call_count == 2

    @patch("aragora.debate.phases.convergence_tracker.record_rlm_ready_quorum")
    def test_records_rlm_ready_quorum_metric(self, mock_record):
        """Records RLM ready quorum metric when reached."""
        tracker = DebateConvergenceTracker()

        # Pre-populate with ready signals to reach quorum
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent1", confidence=0.9, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent2", confidence=0.85, ready=True)
        )
        tracker._collective_readiness.update(
            AgentReadinessSignal(agent="agent3", confidence=0.88, ready=True)
        )

        ctx = MockDebateContext()
        ctx.proposals = {"agent4": "Final position."}

        with patch("aragora.debate.phases.convergence_tracker.parse_ready_signal") as mock_parse:
            mock_parse.return_value = AgentReadinessSignal(
                agent="agent4", confidence=0.95, ready=True
            )
            tracker.check_rlm_ready_quorum(ctx, round_num=3)

        mock_record.assert_called_once()
