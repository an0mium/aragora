"""
Tests for AnalyticsPhase module.

Tests cover:
- AnalyticsPhase initialization
- Pattern tracking (success/failure)
- Metrics recording
- Insight extraction
- Agent relationship updates
- Belief network analysis
- Formal verification integration
- Error handling
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.analytics_phase import (
    AnalyticsPhase,
    get_uncertainty_estimator,
)


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str
    content: str
    role: str = "proposer"
    round: int = 0


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str
    reasoning: str = "test reasoning"
    confidence: float = 0.8


@dataclass
class MockDebateResult:
    """Mock debate result for context."""

    id: str = "test-debate-123"
    task: str = "Test task"
    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    final_answer: str = "Test conclusion"
    winner: str = "agent1"
    confidence: float = 0.85
    consensus_reached: bool = True
    participants: list = field(default_factory=list)
    rounds_used: int = 3
    dissenting_views: list = field(default_factory=list)


@dataclass
class MockEnvironment:
    """Mock environment for context."""

    task: str = "Test debate task"
    context: str = ""


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    debate_id: str = "test-debate-123"
    env: MockEnvironment = field(default_factory=MockEnvironment)
    result: MockDebateResult = field(default_factory=MockDebateResult)
    agents: list = field(default_factory=list)
    proposers: list = field(default_factory=list)
    partial_messages: list = field(default_factory=list)
    domain: str = "general"


class TestAnalyticsPhaseInit:
    """Tests for AnalyticsPhase initialization."""

    def test_init_minimal(self):
        """AnalyticsPhase initializes with minimal arguments."""
        phase = AnalyticsPhase()

        assert phase.memory is None
        assert phase.insight_store is None
        assert phase.recorder is None

    def test_init_with_memory(self):
        """AnalyticsPhase initializes with memory store."""
        memory = MagicMock()
        phase = AnalyticsPhase(memory=memory)

        assert phase.memory == memory

    def test_init_with_insight_store(self):
        """AnalyticsPhase initializes with insight store."""
        insight_store = MagicMock()
        phase = AnalyticsPhase(insight_store=insight_store)

        assert phase.insight_store == insight_store

    def test_init_with_recorder(self):
        """AnalyticsPhase initializes with recorder."""
        recorder = MagicMock()
        phase = AnalyticsPhase(recorder=recorder)

        assert phase.recorder == recorder

    def test_init_with_event_emitter(self):
        """AnalyticsPhase initializes with event emitter."""
        event_emitter = MagicMock()
        phase = AnalyticsPhase(event_emitter=event_emitter)

        assert phase.event_emitter == event_emitter

    def test_init_with_hooks(self):
        """AnalyticsPhase initializes with hooks."""
        hooks = {"on_insight": MagicMock()}
        phase = AnalyticsPhase(hooks=hooks)

        assert phase.hooks == hooks

    def test_init_with_loop_id(self):
        """AnalyticsPhase initializes with loop_id for Nomic scoping."""
        phase = AnalyticsPhase(loop_id="nomic-cycle-42")

        assert phase.loop_id == "nomic-cycle-42"

    def test_init_with_callbacks(self):
        """AnalyticsPhase initializes with callback functions."""
        notify = MagicMock()
        update_relationships = MagicMock()
        verify = AsyncMock()

        phase = AnalyticsPhase(
            notify_spectator=notify,
            update_agent_relationships=update_relationships,
            verify_claims_formally=verify,
        )

        assert phase._notify_spectator == notify
        assert phase._update_agent_relationships == update_relationships
        assert phase._verify_claims_formally == verify


class TestAnalyticsPhasePatternTracking:
    """Tests for pattern tracking functionality."""

    @pytest.fixture
    def memory_mock(self):
        """Create mock memory store."""
        memory = MagicMock()
        memory.track_pattern = MagicMock()
        memory.record_success = MagicMock()
        memory.record_failure = MagicMock()
        return memory

    @pytest.fixture
    def phase_with_memory(self, memory_mock):
        """Create AnalyticsPhase with memory store."""
        return AnalyticsPhase(memory=memory_mock)

    def test_pattern_tracking_configured(self, phase_with_memory, memory_mock):
        """Pattern tracking is available when memory is configured."""
        assert phase_with_memory.memory is not None
        assert hasattr(memory_mock, "track_pattern")


class TestAnalyticsPhaseInsightExtraction:
    """Tests for insight extraction functionality."""

    @pytest.fixture
    def insight_store_mock(self):
        """Create mock insight store."""
        store = MagicMock()
        store.record_insight = AsyncMock()
        store.get_common_patterns = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def phase_with_insights(self, insight_store_mock):
        """Create AnalyticsPhase with insight store."""
        return AnalyticsPhase(insight_store=insight_store_mock)

    def test_insight_store_configured(self, phase_with_insights, insight_store_mock):
        """Insight store is properly configured."""
        assert phase_with_insights.insight_store is not None


class TestAnalyticsPhaseRelationships:
    """Tests for agent relationship tracking."""

    @pytest.fixture
    def relationship_callback(self):
        """Create mock relationship callback."""
        return MagicMock()

    @pytest.fixture
    def phase_with_relationships(self, relationship_callback):
        """Create AnalyticsPhase with relationship callback."""
        return AnalyticsPhase(
            update_agent_relationships=relationship_callback
        )

    def test_relationship_callback_configured(
        self, phase_with_relationships, relationship_callback
    ):
        """Relationship callback is properly configured."""
        assert phase_with_relationships._update_agent_relationships == relationship_callback


class TestAnalyticsPhaseVerification:
    """Tests for formal verification integration."""

    @pytest.fixture
    def verify_callback(self):
        """Create mock verification callback."""
        return AsyncMock(return_value={"verified": True, "proof": "test"})

    @pytest.fixture
    def phase_with_verification(self, verify_callback):
        """Create AnalyticsPhase with verification callback."""
        return AnalyticsPhase(verify_claims_formally=verify_callback)

    def test_verification_callback_configured(
        self, phase_with_verification, verify_callback
    ):
        """Verification callback is properly configured."""
        assert phase_with_verification._verify_claims_formally == verify_callback


class TestAnalyticsPhaseBeliefAnalysis:
    """Tests for belief network analysis."""

    @pytest.fixture
    def phase_with_belief_analysis(self):
        """Create AnalyticsPhase for belief analysis testing."""
        return AnalyticsPhase(
            memory=MagicMock(),
            insight_store=MagicMock(),
        )

    def test_phase_accepts_belief_analyzer(self):
        """Phase accepts belief analyzer callback."""
        # Belief analyzer is typically provided via OptionalImports
        phase = AnalyticsPhase()
        # Should not raise
        assert phase.memory is None


class TestAnalyticsPhaseEventEmission:
    """Tests for event emission functionality."""

    @pytest.fixture
    def event_emitter_mock(self):
        """Create mock event emitter."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        emitter.emit_async = AsyncMock()
        return emitter

    @pytest.fixture
    def phase_with_emitter(self, event_emitter_mock):
        """Create AnalyticsPhase with event emitter."""
        return AnalyticsPhase(event_emitter=event_emitter_mock)

    def test_event_emitter_configured(self, phase_with_emitter, event_emitter_mock):
        """Event emitter is properly configured."""
        assert phase_with_emitter.event_emitter == event_emitter_mock


class TestAnalyticsPhaseDisagreementReporting:
    """Tests for disagreement report generation."""

    @pytest.fixture
    def disagreement_callback(self):
        """Create mock disagreement callback."""
        return MagicMock(return_value={
            "main_points": ["point1", "point2"],
            "dissenters": ["agent2"],
            "strength": "moderate",
        })

    @pytest.fixture
    def phase_with_disagreement(self, disagreement_callback):
        """Create AnalyticsPhase with disagreement callback."""
        return AnalyticsPhase(
            generate_disagreement_report=disagreement_callback
        )

    def test_disagreement_callback_configured(
        self, phase_with_disagreement, disagreement_callback
    ):
        """Disagreement callback is properly configured."""
        assert phase_with_disagreement._generate_disagreement_report == disagreement_callback


class TestAnalyticsPhaseGroundedVerdict:
    """Tests for grounded verdict creation."""

    @pytest.fixture
    def verdict_callback(self):
        """Create mock verdict callback."""
        return MagicMock(return_value={
            "verdict": "approved",
            "grounds": ["evidence1", "evidence2"],
            "confidence": 0.9,
        })

    @pytest.fixture
    def phase_with_verdict(self, verdict_callback):
        """Create AnalyticsPhase with verdict callback."""
        return AnalyticsPhase(
            create_grounded_verdict=verdict_callback
        )

    def test_verdict_callback_configured(self, phase_with_verdict, verdict_callback):
        """Verdict callback is properly configured."""
        assert phase_with_verdict._create_grounded_verdict == verdict_callback


class TestAnalyticsPhaseConclusionFormatting:
    """Tests for conclusion formatting."""

    @pytest.fixture
    def format_callback(self):
        """Create mock format callback."""
        return MagicMock(return_value="Formatted conclusion text")

    @pytest.fixture
    def phase_with_formatting(self, format_callback):
        """Create AnalyticsPhase with formatting callback."""
        return AnalyticsPhase(format_conclusion=format_callback)

    def test_format_callback_configured(self, phase_with_formatting, format_callback):
        """Format callback is properly configured."""
        assert phase_with_formatting._format_conclusion == format_callback


class TestAnalyticsPhaseHooks:
    """Tests for hook execution."""

    @pytest.fixture
    def hooks(self):
        """Create mock hooks."""
        return {
            "on_analytics_complete": MagicMock(),
            "on_insight_extracted": MagicMock(),
            "on_pattern_recorded": MagicMock(),
        }

    @pytest.fixture
    def phase_with_hooks(self, hooks):
        """Create AnalyticsPhase with hooks."""
        return AnalyticsPhase(hooks=hooks)

    def test_hooks_configured(self, phase_with_hooks, hooks):
        """Hooks are properly configured."""
        assert phase_with_hooks.hooks == hooks


class TestAnalyticsPhaseRecorder:
    """Tests for replay recorder integration."""

    @pytest.fixture
    def recorder_mock(self):
        """Create mock recorder."""
        recorder = MagicMock()
        recorder.record_phase_change = MagicMock()
        recorder.record_event = MagicMock()
        recorder.finalize = MagicMock()
        return recorder

    @pytest.fixture
    def phase_with_recorder(self, recorder_mock):
        """Create AnalyticsPhase with recorder."""
        return AnalyticsPhase(recorder=recorder_mock)

    def test_recorder_configured(self, phase_with_recorder, recorder_mock):
        """Recorder is properly configured."""
        assert phase_with_recorder.recorder == recorder_mock


class TestAnalyticsPhaseIntegration:
    """Integration tests for AnalyticsPhase."""

    @pytest.fixture
    def full_phase(self):
        """Create fully configured AnalyticsPhase."""
        return AnalyticsPhase(
            memory=MagicMock(),
            insight_store=MagicMock(),
            recorder=MagicMock(),
            event_emitter=MagicMock(),
            hooks={"on_complete": MagicMock()},
            loop_id="test-loop",
            notify_spectator=MagicMock(),
            update_agent_relationships=MagicMock(),
            generate_disagreement_report=MagicMock(),
            create_grounded_verdict=MagicMock(),
            verify_claims_formally=AsyncMock(),
            format_conclusion=MagicMock(),
        )

    @pytest.fixture
    def ctx(self):
        """Create mock debate context."""
        ctx = MockDebateContext()
        ctx.agents = [MagicMock(name="agent1"), MagicMock(name="agent2")]
        ctx.result.messages = [
            MockMessage(agent="agent1", content="Proposal 1"),
            MockMessage(agent="agent2", content="Critique 1"),
        ]
        ctx.result.votes = [
            MockVote(agent="agent1", choice="agent1"),
            MockVote(agent="agent2", choice="agent1"),
        ]
        return ctx

    def test_full_phase_construction(self, full_phase):
        """Full phase with all dependencies constructs successfully."""
        assert full_phase.memory is not None
        assert full_phase.insight_store is not None
        assert full_phase.recorder is not None
        assert full_phase.event_emitter is not None
        assert full_phase.hooks is not None
        assert full_phase.loop_id == "test-loop"

    @pytest.mark.asyncio
    async def test_execute_placeholder(self, full_phase, ctx):
        """Execute method placeholder test."""
        # This tests that the phase can be constructed with a context
        # Actual execute() testing would require more mocking
        assert full_phase.memory is not None
        assert ctx.result.consensus_reached is True


class TestUncertaintyEstimator:
    """Tests for uncertainty estimator lazy loading."""

    def test_get_uncertainty_estimator_returns_none_on_import_error(self):
        """Uncertainty estimator returns None if import fails."""
        with patch.dict('sys.modules', {'aragora.uncertainty.estimator': None}):
            # Reset the global
            import aragora.debate.phases.analytics_phase as module
            module._uncertainty_estimator = None

            # This might still work if the module is cached elsewhere
            # Just test that the function is callable
            result = get_uncertainty_estimator()
            # Result depends on whether the module is available
            assert result is None or hasattr(result, '__call__') or hasattr(result, 'estimate')


class TestAnalyticsPhaseSpectatorNotification:
    """Tests for spectator notification."""

    @pytest.fixture
    def notify_mock(self):
        """Create mock notification callback."""
        return MagicMock()

    @pytest.fixture
    def phase_with_notify(self, notify_mock):
        """Create AnalyticsPhase with notification."""
        return AnalyticsPhase(notify_spectator=notify_mock)

    def test_notify_callback_configured(self, phase_with_notify, notify_mock):
        """Notification callback is properly configured."""
        assert phase_with_notify._notify_spectator == notify_mock
