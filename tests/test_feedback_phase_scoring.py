"""
Feedback Phase Scoring Tests.

Tests for the post-debate feedback phase of debate orchestration.
Covers:
- ELO match recording
- Persona performance updates
- Calibration tracking
- Consensus memory storage
- Memory cleanup
- Event emission
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.phases.feedback_phase import FeedbackPhase


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    voter: str  # Sometimes used interchangeably with agent
    choice: str
    confidence: float = 0.8
    reasoning: str = "Test reasoning"


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    genome_id: Optional[str] = None
    prompt_version: Optional[int] = None


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str
    content: str
    role: str = "proposer"
    target_agent: Optional[str] = None


@dataclass
class MockEnvironment:
    """Mock environment."""

    task: str = "Test debate task"


@dataclass
class MockDebateResult:
    """Mock debate result."""

    id: str = "test-debate-123"
    debate_id: str = "test-debate-123"
    final_answer: str = "The final answer from the debate."
    consensus_reached: bool = True
    confidence: float = 0.85
    winner: Optional[str] = "claude"
    votes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    rounds_used: int = 3
    belief_cruxes: list = field(default_factory=list)
    duration_seconds: float = 120.0


@dataclass
class MockDebateContext:
    """Mock debate context."""

    result: MockDebateResult = field(default_factory=MockDebateResult)
    agents: list = field(default_factory=list)
    env: MockEnvironment = field(default_factory=MockEnvironment)
    debate_id: str = "test-debate-123"
    domain: str = "general"
    choice_mapping: dict = field(default_factory=dict)
    applied_insight_ids: list = field(default_factory=list)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    return [
        MockAgent("claude"),
        MockAgent("gpt4"),
        MockAgent("gemini"),
    ]


@pytest.fixture
def mock_result(mock_agents):
    """Create mock debate result."""
    return MockDebateResult(
        winner="claude",
        votes=[
            MockVote(agent="claude", voter="claude", choice="claude", confidence=0.9),
            MockVote(agent="gpt4", voter="gpt4", choice="claude", confidence=0.8),
            MockVote(agent="gemini", voter="gemini", choice="gpt4", confidence=0.7),
        ],
    )


@pytest.fixture
def mock_context(mock_agents, mock_result):
    """Create mock debate context."""
    return MockDebateContext(
        result=mock_result,
        agents=mock_agents,
        choice_mapping={"claude": "claude", "gpt4": "gpt4"},
    )


@pytest.fixture
def feedback_phase():
    """Create minimal feedback phase."""
    return FeedbackPhase()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestFeedbackPhaseInit:
    """Tests for FeedbackPhase initialization."""

    def test_init_with_no_dependencies(self):
        """Test initialization with no dependencies."""
        phase = FeedbackPhase()
        assert phase.elo_system is None
        assert phase.persona_manager is None
        assert phase.continuum_memory is None

    def test_init_with_dependencies(self):
        """Test initialization with all dependencies."""
        mock_elo = MagicMock()
        mock_persona = MagicMock()
        mock_memory = MagicMock()

        phase = FeedbackPhase(
            elo_system=mock_elo,
            persona_manager=mock_persona,
            continuum_memory=mock_memory,
            auto_evolve=False,
            breeding_threshold=0.9,
        )

        assert phase.elo_system is mock_elo
        assert phase.persona_manager is mock_persona
        assert phase.continuum_memory is mock_memory
        assert phase.auto_evolve is False
        assert phase.breeding_threshold == 0.9

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""

        def mock_callback():
            pass

        phase = FeedbackPhase(
            emit_moment_event=mock_callback,
            store_debate_outcome_as_memory=mock_callback,
        )

        assert phase._emit_moment_event is mock_callback
        assert phase._store_debate_outcome_as_memory is mock_callback


# =============================================================================
# Execute Tests
# =============================================================================


class TestFeedbackExecute:
    """Tests for main execute method."""

    @pytest.mark.asyncio
    async def test_execute_without_result_logs_warning(self, feedback_phase):
        """Test execute logs warning without result."""
        ctx = MockDebateContext(result=None)
        # Should not raise
        await feedback_phase.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_calls_all_methods(self, mock_context):
        """Test execute calls all feedback methods."""
        phase = FeedbackPhase()

        # Patch all methods - use delegated feedback objects for refactored methods
        with (
            patch.object(phase._elo_feedback, "record_elo_match") as mock_elo,
            patch.object(phase._persona_feedback, "update_persona_performance") as mock_persona,
            patch.object(phase, "_resolve_positions") as mock_positions,
            patch.object(phase, "_update_relationships") as mock_relationships,
            patch.object(phase, "_detect_moments") as mock_moments,
            patch.object(phase, "_index_debate", new_callable=AsyncMock) as mock_index,
            patch.object(phase, "_detect_flips") as mock_flips,
            patch.object(phase._consensus_storage, "store_consensus_outcome") as mock_consensus,
            patch.object(phase._consensus_storage, "store_cruxes") as mock_cruxes,
            patch.object(phase, "_store_memory") as mock_memory,
            patch.object(phase, "_update_memory_outcomes") as mock_outcomes,
            patch.object(phase, "_record_calibration") as mock_calibration,
        ):

            await phase.execute(mock_context)

            mock_elo.assert_called_once()
            mock_persona.assert_called_once()
            mock_positions.assert_called_once()


# =============================================================================
# ELO Recording Tests
# =============================================================================


class TestEloRecording:
    """Tests for ELO match recording."""

    def test_record_elo_match_without_system(self, mock_context, feedback_phase):
        """Test ELO recording skipped without system."""
        # Should not raise
        feedback_phase._record_elo_match(mock_context)

    def test_record_elo_match_without_winner(self, mock_context):
        """Test ELO recording skipped without winner."""
        mock_elo = MagicMock()
        phase = FeedbackPhase(elo_system=mock_elo)

        mock_context.result.winner = None
        phase._record_elo_match(mock_context)

        mock_elo.record_match.assert_not_called()

    def test_record_elo_match_with_winner(self, mock_context, mock_agents):
        """Test ELO match recorded with winner."""
        mock_elo = MagicMock()
        phase = FeedbackPhase(elo_system=mock_elo)

        mock_context.agents = mock_agents
        mock_context.result.winner = "claude"
        mock_context.result.consensus_reached = True

        phase._record_elo_match(mock_context)

        mock_elo.record_match.assert_called_once()
        call_args = mock_elo.record_match.call_args
        assert "claude" in call_args[0][1]  # participants
        assert call_args[0][2]["claude"] == 1.0  # winner gets 1.0

    def test_record_elo_match_scores_correct(self, mock_context, mock_agents):
        """Test ELO scores are calculated correctly."""
        mock_elo = MagicMock()
        phase = FeedbackPhase(elo_system=mock_elo)

        mock_context.agents = mock_agents
        mock_context.result.winner = "claude"
        mock_context.result.consensus_reached = True

        phase._record_elo_match(mock_context)

        call_args = mock_elo.record_match.call_args
        scores = call_args[0][2]

        assert scores["claude"] == 1.0  # Winner
        assert scores["gpt4"] == 0.5  # Non-winner in consensus
        assert scores["gemini"] == 0.5  # Non-winner in consensus


# =============================================================================
# Persona Performance Tests
# =============================================================================


class TestPersonaPerformance:
    """Tests for persona performance updates."""

    def test_update_persona_without_manager(self, mock_context, feedback_phase):
        """Test persona update skipped without manager."""
        # Should not raise
        feedback_phase._update_persona_performance(mock_context)

    def test_update_persona_records_performance(self, mock_context, mock_agents):
        """Test persona manager records performance."""
        mock_persona = MagicMock()
        phase = FeedbackPhase(persona_manager=mock_persona)

        mock_context.agents = mock_agents
        mock_context.result.winner = "claude"

        phase._update_persona_performance(mock_context)

        # Should record for each agent
        assert mock_persona.record_performance.call_count == 3

    def test_update_persona_winner_success(self, mock_context, mock_agents):
        """Test winner is recorded as success."""
        mock_persona = MagicMock()
        phase = FeedbackPhase(persona_manager=mock_persona)

        mock_context.agents = [MockAgent("claude")]
        mock_context.result.winner = "claude"
        mock_context.result.consensus_reached = True
        mock_context.result.confidence = 0.9

        phase._update_persona_performance(mock_context)

        call_args = mock_persona.record_performance.call_args
        assert call_args[1]["success"] is True


# =============================================================================
# Calibration Recording Tests
# =============================================================================


class TestCalibrationRecording:
    """Tests for calibration recording."""

    def test_record_calibration_without_tracker(self, mock_context, feedback_phase):
        """Test calibration skipped without tracker."""
        # Should not raise
        feedback_phase._record_calibration(mock_context)

    def test_record_calibration_without_result(self):
        """Test calibration skipped without result."""
        mock_tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=mock_tracker)

        ctx = MockDebateContext(result=None)
        phase._record_calibration(ctx)

        mock_tracker.record_prediction.assert_not_called()

    def test_record_calibration_records_predictions(self, mock_context):
        """Test calibration records predictions for each vote."""
        mock_tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=mock_tracker)

        phase._record_calibration(mock_context)

        # Should record for each vote with confidence
        assert mock_tracker.record_prediction.call_count == 3

    def test_record_calibration_correct_prediction(self, mock_context):
        """Test correct predictions are marked correctly."""
        mock_tracker = MagicMock()
        phase = FeedbackPhase(calibration_tracker=mock_tracker)

        # Claude voted for claude (winner), should be correct
        mock_context.result.winner = "claude"
        mock_context.result.votes = [
            MockVote(agent="claude", voter="claude", choice="claude", confidence=0.9),
        ]

        phase._record_calibration(mock_context)

        call_args = mock_tracker.record_prediction.call_args
        assert call_args[1]["correct"] is True


# =============================================================================
# Position Resolution Tests
# =============================================================================


class TestPositionResolution:
    """Tests for position ledger resolution."""

    def test_resolve_positions_without_ledger(self, mock_context, feedback_phase):
        """Test position resolution skipped without ledger."""
        # Should not raise
        feedback_phase._resolve_positions(mock_context)

    def test_resolve_positions_without_answer(self, mock_context):
        """Test position resolution skipped without final answer."""
        mock_ledger = MagicMock()
        phase = FeedbackPhase(position_ledger=mock_ledger)

        mock_context.result.final_answer = ""
        phase._resolve_positions(mock_context)

        mock_ledger.get_agent_positions.assert_not_called()


# =============================================================================
# Relationship Tracking Tests
# =============================================================================


class TestRelationshipTracking:
    """Tests for relationship tracking."""

    def test_update_relationships_without_tracker(self, mock_context, feedback_phase):
        """Test relationship update skipped without tracker."""
        # Should not raise
        feedback_phase._update_relationships(mock_context)

    def test_update_relationships_builds_vote_mapping(self, mock_context, mock_agents):
        """Test relationship update builds vote mapping."""
        mock_tracker = MagicMock()
        phase = FeedbackPhase(relationship_tracker=mock_tracker)

        mock_context.agents = mock_agents
        phase._update_relationships(mock_context)

        mock_tracker.update_from_debate.assert_called_once()
        call_args = mock_tracker.update_from_debate.call_args
        assert "participants" in call_args[1]
        assert "votes" in call_args[1]


# =============================================================================
# Moment Detection Tests
# =============================================================================


class TestMomentDetection:
    """Tests for narrative moment detection."""

    def test_detect_moments_without_detector(self, mock_context, feedback_phase):
        """Test moment detection skipped without detector."""
        # Should not raise
        feedback_phase._detect_moments(mock_context)

    def test_detect_moments_checks_upset_victory(self, mock_context, mock_agents):
        """Test moment detection checks for upset victories."""
        mock_detector = MagicMock()
        mock_detector.detect_upset_victory.return_value = None
        mock_elo = MagicMock()

        phase = FeedbackPhase(
            moment_detector=mock_detector,
            elo_system=mock_elo,
        )

        mock_context.agents = mock_agents
        mock_context.result.winner = "claude"

        phase._detect_moments(mock_context)

        # Should check for upset against losers
        assert mock_detector.detect_upset_victory.call_count >= 2


# =============================================================================
# Flip Detection Tests
# =============================================================================


class TestFlipDetection:
    """Tests for position flip detection."""

    def test_detect_flips_without_detector(self, mock_context, feedback_phase):
        """Test flip detection skipped without detector."""
        # Should not raise
        feedback_phase._detect_flips(mock_context)

    def test_detect_flips_for_each_agent(self, mock_context, mock_agents):
        """Test flip detection runs for each agent."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []

        phase = FeedbackPhase(flip_detector=mock_detector)
        mock_context.agents = mock_agents

        phase._detect_flips(mock_context)

        # Should check each agent
        assert mock_detector.detect_flips_for_agent.call_count == 3


# =============================================================================
# Consensus Memory Tests
# =============================================================================


class TestConsensusMemory:
    """Tests for consensus memory storage."""

    def test_store_consensus_without_memory(self, mock_context, feedback_phase):
        """Test consensus storage skipped without memory."""
        # Should not raise
        feedback_phase._store_consensus_outcome(mock_context)

    def test_store_consensus_without_result(self):
        """Test consensus storage skipped without result."""
        mock_memory = MagicMock()
        phase = FeedbackPhase(consensus_memory=mock_memory)

        ctx = MockDebateContext(result=None)
        phase._store_consensus_outcome(ctx)

        mock_memory.store_consensus.assert_not_called()

    def test_store_consensus_without_answer(self, mock_context):
        """Test consensus storage skipped without final answer."""
        mock_memory = MagicMock()
        phase = FeedbackPhase(consensus_memory=mock_memory)

        mock_context.result.final_answer = ""
        phase._store_consensus_outcome(mock_context)

        mock_memory.store_consensus.assert_not_called()


# =============================================================================
# Confidence to Strength Tests
# =============================================================================


class TestConfidenceToStrength:
    """Tests for confidence to strength conversion."""

    def test_confidence_unanimous(self):
        """Test high confidence returns unanimous."""
        phase = FeedbackPhase()
        strength = phase._confidence_to_strength(0.95)
        assert strength.value == "unanimous"

    def test_confidence_strong(self):
        """Test medium-high confidence returns strong."""
        phase = FeedbackPhase()
        strength = phase._confidence_to_strength(0.85)
        assert strength.value == "strong"

    def test_confidence_moderate(self):
        """Test medium confidence returns moderate."""
        phase = FeedbackPhase()
        strength = phase._confidence_to_strength(0.7)
        assert strength.value == "moderate"

    def test_confidence_weak(self):
        """Test low-medium confidence returns weak."""
        phase = FeedbackPhase()
        strength = phase._confidence_to_strength(0.55)
        assert strength.value == "weak"

    def test_confidence_split(self):
        """Test low confidence returns split."""
        phase = FeedbackPhase()
        strength = phase._confidence_to_strength(0.4)
        assert strength.value == "split"


# =============================================================================
# Memory Cleanup Tests
# =============================================================================


class TestMemoryCleanup:
    """Tests for memory cleanup."""

    def test_cleanup_without_memory(self, mock_context, feedback_phase):
        """Test cleanup skipped without memory."""
        # Should not raise
        feedback_phase._run_memory_cleanup(mock_context)

    def test_cleanup_calls_expired_cleanup(self, mock_context):
        """Test cleanup calls cleanup_expired_memories."""
        mock_memory = MagicMock()
        mock_memory.cleanup_expired_memories.return_value = 5

        phase = FeedbackPhase(continuum_memory=mock_memory)
        phase._run_memory_cleanup(mock_context)

        mock_memory.cleanup_expired_memories.assert_called_once()


# =============================================================================
# Genome Fitness Tests
# =============================================================================


class TestGenomeFitness:
    """Tests for genome fitness updates."""

    def test_update_fitness_without_manager(self, mock_context, feedback_phase):
        """Test fitness update skipped without manager."""
        # Should not raise
        feedback_phase._update_genome_fitness(mock_context)

    def test_update_fitness_without_result(self):
        """Test fitness update skipped without result."""
        mock_manager = MagicMock()
        phase = FeedbackPhase(population_manager=mock_manager)

        ctx = MockDebateContext(result=None)
        phase._update_genome_fitness(ctx)

        mock_manager.update_fitness.assert_not_called()

    def test_update_fitness_for_evolved_agent(self, mock_context):
        """Test fitness update for agent with genome_id."""
        mock_manager = MagicMock()
        phase = FeedbackPhase(population_manager=mock_manager)

        # Add agent with genome_id
        mock_context.agents = [MockAgent("claude", genome_id="genome-123")]
        mock_context.result.winner = "claude"

        phase._update_genome_fitness(mock_context)

        mock_manager.update_fitness.assert_called_once()


# =============================================================================
# Population Evolution Tests
# =============================================================================


class TestPopulationEvolution:
    """Tests for population evolution triggering."""

    @pytest.mark.asyncio
    async def test_evolution_without_manager(self, mock_context, feedback_phase):
        """Test evolution skipped without manager."""
        # Should not raise
        await feedback_phase._maybe_evolve_population(mock_context)

    @pytest.mark.asyncio
    async def test_evolution_disabled(self, mock_context):
        """Test evolution skipped when disabled."""
        mock_manager = MagicMock()
        phase = FeedbackPhase(
            population_manager=mock_manager,
            auto_evolve=False,
        )

        await phase._maybe_evolve_population(mock_context)

        mock_manager.get_or_create_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_evolution_below_threshold(self, mock_context):
        """Test evolution skipped below threshold."""
        mock_manager = MagicMock()
        phase = FeedbackPhase(
            population_manager=mock_manager,
            auto_evolve=True,
            breeding_threshold=0.9,
        )

        mock_context.result.confidence = 0.7  # Below threshold

        await phase._maybe_evolve_population(mock_context)

        mock_manager.get_or_create_population.assert_not_called()


# =============================================================================
# Pulse Outcome Tests
# =============================================================================


class TestPulseOutcome:
    """Tests for pulse outcome recording."""

    def test_pulse_without_manager(self, mock_context, feedback_phase):
        """Test pulse recording skipped without manager."""
        # Should not raise
        feedback_phase._record_pulse_outcome(mock_context)

    def test_pulse_without_trending_topic(self, mock_context):
        """Test pulse recording skipped without trending topic."""
        mock_pulse = MagicMock()
        phase = FeedbackPhase(pulse_manager=mock_pulse)

        phase._record_pulse_outcome(mock_context)

        mock_pulse.record_debate_outcome.assert_not_called()


# =============================================================================
# Risk Assessment Tests
# =============================================================================


class TestRiskAssessment:
    """Tests for risk assessment."""

    def test_assess_risks_without_emitter(self, mock_context, feedback_phase):
        """Test risk assessment skipped without emitter."""
        # Should not raise
        feedback_phase._assess_risks(mock_context)


# =============================================================================
# Insight Usage Tests
# =============================================================================


class TestInsightUsage:
    """Tests for insight usage recording."""

    @pytest.mark.asyncio
    async def test_insight_usage_without_store(self, mock_context, feedback_phase):
        """Test insight usage skipped without store."""
        # Should not raise
        await feedback_phase._record_insight_usage(mock_context)

    @pytest.mark.asyncio
    async def test_insight_usage_without_applied_ids(self, mock_context):
        """Test insight usage skipped without applied insight IDs."""
        mock_store = MagicMock()
        phase = FeedbackPhase(insight_store=mock_store)

        mock_context.applied_insight_ids = []
        await phase._record_insight_usage(mock_context)

        mock_store.record_insight_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_insight_usage_no_consensus(self, mock_context):
        """Test insight usage skipped without consensus."""
        mock_store = AsyncMock()
        phase = FeedbackPhase(insight_store=mock_store)

        mock_context.applied_insight_ids = ["insight-1"]
        mock_context.result.consensus_reached = False

        await phase._record_insight_usage(mock_context)

        mock_store.record_insight_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_insight_usage_records_success(self, mock_context):
        """Test insight usage records successful usage."""
        mock_store = AsyncMock()
        phase = FeedbackPhase(insight_store=mock_store)

        mock_context.applied_insight_ids = ["insight-1", "insight-2"]
        mock_context.result.consensus_reached = True
        mock_context.result.confidence = 0.9

        await phase._record_insight_usage(mock_context)

        assert mock_store.record_insight_usage.call_count == 2


# =============================================================================
# Evolution Pattern Tests
# =============================================================================


class TestEvolutionPatterns:
    """Tests for evolution pattern recording."""

    def test_patterns_without_evolver(self, mock_context, feedback_phase):
        """Test pattern recording skipped without evolver."""
        # Should not raise
        feedback_phase._record_evolution_patterns(mock_context)

    def test_patterns_low_confidence(self, mock_context):
        """Test pattern recording skipped for low confidence."""
        mock_evolver = MagicMock()
        phase = FeedbackPhase(prompt_evolver=mock_evolver)

        mock_context.result.confidence = 0.5  # Low confidence

        phase._record_evolution_patterns(mock_context)

        mock_evolver.extract_winning_patterns.assert_not_called()


# =============================================================================
# Agent Prediction Check Tests
# =============================================================================


class TestAgentPredictionCheck:
    """Tests for agent prediction checking."""

    def test_check_prediction_no_result(self):
        """Test prediction check returns False without result."""
        phase = FeedbackPhase()
        ctx = MockDebateContext(result=None)
        agent = MockAgent("claude")

        result = phase._check_agent_prediction(agent, ctx)
        assert result is False

    def test_check_prediction_no_votes(self, mock_context):
        """Test prediction check returns False without votes."""
        phase = FeedbackPhase()
        mock_context.result.votes = []

        agent = MockAgent("claude")
        result = phase._check_agent_prediction(agent, mock_context)
        assert result is False

    def test_check_prediction_correct(self, mock_context):
        """Test prediction check returns True for correct prediction."""
        phase = FeedbackPhase()
        mock_context.result.winner = "claude"
        mock_context.result.votes = [
            MockVote(agent="claude", voter="claude", choice="claude"),
        ]
        mock_context.choice_mapping = {"claude": "claude"}

        agent = MockAgent("claude")
        result = phase._check_agent_prediction(agent, mock_context)
        assert result is True

    def test_check_prediction_incorrect(self, mock_context):
        """Test prediction check returns False for incorrect prediction."""
        phase = FeedbackPhase()
        mock_context.result.winner = "gpt4"
        mock_context.result.votes = [
            MockVote(agent="claude", voter="claude", choice="claude"),
        ]
        mock_context.choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        agent = MockAgent("claude")
        result = phase._check_agent_prediction(agent, mock_context)
        assert result is False


# =============================================================================
# Trait Emergence Tests
# =============================================================================


class TestTraitEmergence:
    """Tests for trait emergence detection."""

    def test_check_trait_emergence_without_manager(self, mock_context, feedback_phase):
        """Test trait check skipped without manager."""
        # Should not raise
        feedback_phase._check_trait_emergence(mock_context)

    def test_detect_emerging_traits_without_stats(self, mock_context):
        """Test trait detection with no stats."""
        mock_persona = MagicMock()
        mock_persona.get_performance_stats.return_value = None

        phase = FeedbackPhase(persona_manager=mock_persona)
        traits = phase._detect_emerging_traits("claude", mock_context)

        assert traits == []


# =============================================================================
# Store Memory Tests
# =============================================================================


class TestStoreMemory:
    """Tests for memory storage."""

    def test_store_memory_without_continuum(self, mock_context, feedback_phase):
        """Test memory storage skipped without continuum."""
        # Should not raise
        feedback_phase._store_memory(mock_context)

    def test_store_memory_without_answer(self, mock_context):
        """Test memory storage skipped without final answer."""
        mock_memory = MagicMock()
        phase = FeedbackPhase(continuum_memory=mock_memory)

        mock_context.result.final_answer = ""
        phase._store_memory(mock_context)

    def test_store_memory_uses_callback(self, mock_context):
        """Test memory storage uses callback."""
        mock_memory = MagicMock()
        store_called = []

        def mock_store(result):
            store_called.append(result)

        phase = FeedbackPhase(
            continuum_memory=mock_memory,
            store_debate_outcome_as_memory=mock_store,
        )

        phase._store_memory(mock_context)

        assert len(store_called) == 1


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for WebSocket event emission."""

    def test_emit_match_recorded_without_emitter(self, mock_context):
        """Test match event skipped without emitter."""
        phase = FeedbackPhase()
        # Should not raise
        phase._emit_match_recorded_event(mock_context, ["claude", "gpt4"])

    def test_emit_flip_events_without_emitter(self, mock_context):
        """Test flip events skipped without emitter."""
        phase = FeedbackPhase()
        # Should not raise
        phase._emit_flip_events(mock_context, "claude", [])

    def test_emit_calibration_update_without_emitter(self, mock_context):
        """Test calibration event skipped without emitter."""
        phase = FeedbackPhase()
        # Should not raise
        phase._emit_calibration_update(mock_context, 5)


# =============================================================================
# Index Debate Tests
# =============================================================================


class TestIndexDebate:
    """Tests for debate indexing."""

    @pytest.mark.asyncio
    async def test_index_without_embeddings(self, mock_context, feedback_phase):
        """Test indexing skipped without embeddings."""
        # Should not raise
        await feedback_phase._index_debate(mock_context)

    @pytest.mark.asyncio
    async def test_index_builds_artifact(self, mock_context, mock_agents):
        """Test indexing builds debate artifact."""
        mock_embeddings = MagicMock()
        indexed_artifacts = []

        async def mock_index(artifact):
            indexed_artifacts.append(artifact)

        phase = FeedbackPhase(
            debate_embeddings=mock_embeddings,
            index_debate_async=mock_index,
        )

        mock_context.agents = mock_agents
        await phase._index_debate(mock_context)

        # Give async task time to start
        await asyncio.sleep(0.01)


# =============================================================================
# Crux Storage Tests
# =============================================================================


class TestCruxStorage:
    """Tests for belief crux storage."""

    def test_store_cruxes_without_memory(self, mock_context, feedback_phase):
        """Test crux storage skipped without memory."""
        # Should not raise
        feedback_phase._store_cruxes(mock_context)

    def test_store_cruxes_without_consensus_id(self, mock_context):
        """Test crux storage skipped without consensus ID."""
        mock_memory = MagicMock()
        phase = FeedbackPhase(consensus_memory=mock_memory)

        # No _last_consensus_id set
        phase._store_cruxes(mock_context)

        mock_memory.update_cruxes.assert_not_called()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeedbackIntegration:
    """Integration-style tests for feedback phase."""

    @pytest.mark.asyncio
    async def test_full_feedback_flow(self, mock_context, mock_agents):
        """Test complete feedback flow with all systems."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {}
        mock_persona = MagicMock()
        mock_calibration = MagicMock()

        phase = FeedbackPhase(
            elo_system=mock_elo,
            persona_manager=mock_persona,
            calibration_tracker=mock_calibration,
        )

        mock_context.agents = mock_agents
        await phase.execute(mock_context)

        # Verify all systems were called
        mock_elo.record_match.assert_called_once()
        assert mock_persona.record_performance.call_count == 3
        assert mock_calibration.record_prediction.call_count == 3

    @pytest.mark.asyncio
    async def test_feedback_handles_errors_gracefully(self, mock_context, mock_agents):
        """Test feedback handles errors without crashing."""
        mock_elo = MagicMock()
        mock_elo.record_match.side_effect = Exception("ELO error")

        phase = FeedbackPhase(elo_system=mock_elo)

        mock_context.agents = mock_agents

        # Should not raise
        await phase.execute(mock_context)


__all__ = [
    "TestFeedbackPhaseInit",
    "TestFeedbackExecute",
    "TestEloRecording",
    "TestPersonaPerformance",
    "TestCalibrationRecording",
    "TestPositionResolution",
    "TestRelationshipTracking",
    "TestMomentDetection",
    "TestFlipDetection",
    "TestConsensusMemory",
    "TestConfidenceToStrength",
    "TestMemoryCleanup",
    "TestGenomeFitness",
    "TestPopulationEvolution",
    "TestPulseOutcome",
    "TestRiskAssessment",
    "TestInsightUsage",
    "TestEvolutionPatterns",
    "TestAgentPredictionCheck",
    "TestTraitEmergence",
    "TestStoreMemory",
    "TestEventEmission",
    "TestIndexDebate",
    "TestCruxStorage",
    "TestFeedbackIntegration",
]
