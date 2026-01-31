"""
Tests for ML Integration module - ML-powered agent selection and quality gates.

Tests cover:
- MLIntegrationConfig initialization and defaults
- MLDelegationStrategy agent selection and scoring
- QualityGate response filtering and scoring
- ConsensusEstimator early termination detection
- MLEnhancedTeamSelector hybrid scoring
- DebateTrainingExporter export functionality
- Singleton getter functions
- Fallback behavior when ML modules unavailable
- Edge cases (cold start, timeout, empty inputs)
- Configuration validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.ml_integration import (
    ConsensusEstimator,
    DebateTrainingExporter,
    MLDelegationStrategy,
    MLEnhancedTeamSelector,
    MLIntegrationConfig,
    QualityGate,
    create_ml_team_selector,
    get_consensus_estimator,
    get_ml_delegation,
    get_quality_gate,
    get_training_exporter,
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, role: str = "proposer"):
        self.name = name
        self.role = role


class MockMessage:
    """Mock message for testing."""

    def __init__(self, agent: str, content: str):
        self.agent = agent
        self.content = content


@pytest.fixture
def mock_agents():
    """Create list of mock agents."""
    return [
        MockAgent("claude"),
        MockAgent("gpt4"),
        MockAgent("gemini"),
    ]


@pytest.fixture
def mock_router():
    """Create mock agent router."""
    router = MagicMock()

    @dataclass
    class MockRoutingDecision:
        task_type: MagicMock = MagicMock(value="reasoning")
        confidence: float = 0.85
        selected_agents: list = None
        agent_scores: dict = None

        def __post_init__(self):
            if self.selected_agents is None:
                self.selected_agents = ["claude", "gpt4", "gemini"]
            if self.agent_scores is None:
                self.agent_scores = {"claude": 0.9, "gpt4": 0.8, "gemini": 0.7}

    router.route = MagicMock(return_value=MockRoutingDecision())
    return router


@pytest.fixture
def mock_quality_scorer():
    """Create mock quality scorer."""
    scorer = MagicMock()

    @dataclass
    class MockQualityScore:
        overall: float = 0.75
        confidence: float = 0.8

    scorer.score = MagicMock(return_value=MockQualityScore())
    return scorer


@pytest.fixture
def mock_consensus_predictor():
    """Create mock consensus predictor."""
    predictor = MagicMock()

    @dataclass
    class MockConsensusPrediction:
        probability: float = 0.9
        confidence: float = 0.85
        early_termination_safe: bool = True
        needs_intervention: bool = False
        convergence_trend: str = "converging"
        estimated_rounds: int = 2
        key_factors: list = None
        features: dict = None

        def __post_init__(self):
            if self.key_factors is None:
                self.key_factors = ["similarity", "agreement"]
            if self.features is None:
                self.features = {"semantic_similarity": 0.88}

    predictor.predict = MagicMock(return_value=MockConsensusPrediction())
    predictor.record_outcome = MagicMock()
    return predictor


# ===========================================================================
# Test: MLIntegrationConfig
# ===========================================================================


class TestMLIntegrationConfig:
    """Tests for MLIntegrationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLIntegrationConfig()

        assert config.use_ml_routing is True
        assert config.ml_routing_weight == 0.4
        assert config.fallback_to_elo is True
        assert config.enable_quality_gates is True
        assert config.quality_threshold == 0.6
        assert config.min_confidence == 0.4
        assert config.enable_early_termination is True
        assert config.early_termination_threshold == 0.85
        assert config.min_rounds_before_termination == 2
        assert config.cache_routing_decisions is True
        assert config.cache_ttl_seconds == 300

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLIntegrationConfig(
            use_ml_routing=False,
            ml_routing_weight=0.6,
            quality_threshold=0.8,
            min_rounds_before_termination=3,
            cache_ttl_seconds=600,
        )

        assert config.use_ml_routing is False
        assert config.ml_routing_weight == 0.6
        assert config.quality_threshold == 0.8
        assert config.min_rounds_before_termination == 3
        assert config.cache_ttl_seconds == 600

    def test_config_immutability(self):
        """Test that config can be modified after creation."""
        config = MLIntegrationConfig()
        config.quality_threshold = 0.9

        assert config.quality_threshold == 0.9


# ===========================================================================
# Test: MLDelegationStrategy
# ===========================================================================


class TestMLDelegationStrategy:
    """Tests for MLDelegationStrategy class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        strategy = MLDelegationStrategy()

        assert strategy.config is not None
        assert strategy.config.use_ml_routing is True
        assert strategy._router is None
        assert strategy._cache == {}

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MLIntegrationConfig(ml_routing_weight=0.5)
        strategy = MLDelegationStrategy(config=config)

        assert strategy.config.ml_routing_weight == 0.5

    def test_init_with_ml_weight_override(self):
        """Test ml_weight parameter overrides config."""
        strategy = MLDelegationStrategy(ml_weight=0.7)

        assert strategy.config.ml_routing_weight == 0.7

    def test_select_agents_empty_list(self):
        """Test selecting from empty agent list."""
        strategy = MLDelegationStrategy()
        result = strategy.select_agents("Test task", [])

        assert result == []

    def test_select_agents_no_router_fallback(self, mock_agents):
        """Test fallback behavior when ML router unavailable."""
        strategy = MLDelegationStrategy()
        # Simulate router not available
        strategy._get_router = MagicMock(return_value=None)

        result = strategy.select_agents("Test task", mock_agents)

        assert result == list(mock_agents)

    def test_select_agents_with_max_agents(self, mock_agents):
        """Test selection with max_agents limit."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=None)

        result = strategy.select_agents("Test task", mock_agents, max_agents=2)

        assert len(result) == 2

    def test_select_agents_with_router(self, mock_agents, mock_router):
        """Test selection with ML router available."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Test task", mock_agents)

        assert len(result) == 3
        mock_router.route.assert_called_once()

    def test_select_agents_caching(self, mock_agents, mock_router):
        """Test that routing decisions are cached."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        # First call
        strategy.select_agents("Test task", mock_agents)
        # Second call with same params
        strategy.select_agents("Test task", mock_agents)

        # Router should only be called once due to caching
        assert mock_router.route.call_count == 1

    def test_select_agents_caching_disabled(self, mock_agents, mock_router):
        """Test selection with caching disabled."""
        config = MLIntegrationConfig(cache_routing_decisions=False)
        strategy = MLDelegationStrategy(config=config)
        strategy._get_router = MagicMock(return_value=mock_router)

        strategy.select_agents("Test task", mock_agents)
        strategy.select_agents("Test task", mock_agents)

        # Router should be called twice when caching disabled
        assert mock_router.route.call_count == 2

    def test_select_agents_handles_value_error(self, mock_agents, mock_router):
        """Test handling of ValueError from router."""
        strategy = MLDelegationStrategy()
        mock_router.route.side_effect = ValueError("Invalid task")
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Test task", mock_agents)

        # Should fall back to returning agents as-is
        assert result == list(mock_agents)

    def test_select_agents_handles_type_error(self, mock_agents, mock_router):
        """Test handling of TypeError from router."""
        strategy = MLDelegationStrategy()
        mock_router.route.side_effect = TypeError("Type error")
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Test task", mock_agents)

        assert result == list(mock_agents)

    def test_select_agents_handles_unexpected_error(self, mock_agents, mock_router):
        """Test handling of unexpected errors from router."""
        strategy = MLDelegationStrategy()
        mock_router.route.side_effect = RuntimeError("Unexpected error")
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Test task", mock_agents)

        assert result == list(mock_agents)

    def test_reorder_agents(self, mock_agents):
        """Test agent reordering according to ML decision."""
        strategy = MLDelegationStrategy()

        # Reverse order
        order = ["gemini", "gpt4", "claude"]
        result = strategy._reorder_agents(mock_agents, order)

        assert [a.name for a in result] == ["gemini", "gpt4", "claude"]

    def test_reorder_agents_missing_in_order(self, mock_agents):
        """Test reordering when some agents not in order list."""
        strategy = MLDelegationStrategy()

        # Partial order
        order = ["gpt4"]
        result = strategy._reorder_agents(mock_agents, order)

        # gpt4 first, then remaining agents
        assert result[0].name == "gpt4"
        assert len(result) == 3

    def test_score_agent_no_router(self, mock_agents):
        """Test scoring without router returns neutral score."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=None)

        score = strategy.score_agent(mock_agents[0], "Test task")

        assert score == 2.5

    def test_score_agent_with_router(self, mock_agents, mock_router):
        """Test scoring with router available."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        score = strategy.score_agent(mock_agents[0], "Test task")

        # Score should be agent_score * 5.0 = 0.9 * 5.0 = 4.5
        assert score == pytest.approx(4.5, rel=0.01)

    def test_score_agent_handles_errors(self, mock_agents, mock_router):
        """Test scoring error handling."""
        strategy = MLDelegationStrategy()
        mock_router.route.side_effect = ValueError("Error")
        strategy._get_router = MagicMock(return_value=mock_router)

        score = strategy.score_agent(mock_agents[0], "Test task")

        assert score == 2.5

    def test_get_cache_key(self, mock_agents):
        """Test cache key generation."""
        strategy = MLDelegationStrategy()

        key1 = strategy._get_cache_key("Task A", ["claude", "gpt4"])
        key2 = strategy._get_cache_key("Task A", ["gpt4", "claude"])
        key3 = strategy._get_cache_key("Task B", ["claude", "gpt4"])

        # Same agents in different order should produce same key
        assert key1 == key2
        # Different task should produce different key
        assert key1 != key3

    def test_lazy_router_loading(self):
        """Test lazy loading of ML router."""
        strategy = MLDelegationStrategy()

        # Router should be None initially
        assert strategy._router is None

        # When ML module not available, should return None
        with patch.dict("sys.modules", {"aragora.ml": None}):
            router = strategy._get_router()
            # Will either return None or raise depending on import behavior
            # The method handles this gracefully


# ===========================================================================
# Test: QualityGate
# ===========================================================================


class TestQualityGate:
    """Tests for QualityGate class."""

    def test_init_defaults(self):
        """Test default initialization."""
        gate = QualityGate()

        assert gate.threshold == 0.6
        assert gate.min_confidence == 0.4
        assert gate._scorer is None

    def test_init_custom(self):
        """Test custom initialization."""
        gate = QualityGate(threshold=0.8, min_confidence=0.5)

        assert gate.threshold == 0.8
        assert gate.min_confidence == 0.5

    def test_score_response_no_scorer(self):
        """Test scoring without scorer returns unknown."""
        gate = QualityGate()
        gate._get_scorer = MagicMock(return_value=None)

        quality, confidence = gate.score_response("Test text")

        assert quality == 0.5
        assert confidence == 0.0

    def test_score_response_with_scorer(self, mock_quality_scorer):
        """Test scoring with scorer available."""
        gate = QualityGate()
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        quality, confidence = gate.score_response("Test text", context="Task context")

        assert quality == 0.75
        assert confidence == 0.8
        mock_quality_scorer.score.assert_called_once_with("Test text", context="Task context")

    def test_score_response_handles_errors(self, mock_quality_scorer):
        """Test scoring error handling."""
        gate = QualityGate()
        mock_quality_scorer.score.side_effect = ValueError("Error")
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        quality, confidence = gate.score_response("Test text")

        assert quality == 0.5
        assert confidence == 0.0

    def test_passes_gate_low_confidence(self):
        """Test that low confidence responses pass through."""
        gate = QualityGate(threshold=0.7, min_confidence=0.5)
        gate.score_response = MagicMock(return_value=(0.3, 0.3))  # Low quality, low confidence

        result = gate.passes_gate("Low quality text")

        # Should pass because confidence < min_confidence
        assert result is True

    def test_passes_gate_high_quality(self):
        """Test high quality response passes."""
        gate = QualityGate(threshold=0.6)
        gate.score_response = MagicMock(return_value=(0.8, 0.9))

        result = gate.passes_gate("High quality text")

        assert result is True

    def test_passes_gate_low_quality(self):
        """Test low quality response fails with high confidence."""
        gate = QualityGate(threshold=0.6, min_confidence=0.4)
        gate.score_response = MagicMock(return_value=(0.4, 0.9))

        result = gate.passes_gate("Low quality text")

        assert result is False

    def test_filter_responses_empty(self):
        """Test filtering empty response list."""
        gate = QualityGate()

        result = gate.filter_responses([])

        assert result == []

    def test_filter_responses(self):
        """Test filtering responses by quality."""
        gate = QualityGate(threshold=0.6, min_confidence=0.5)

        # Mock varying quality scores
        scores = [(0.8, 0.9), (0.4, 0.9), (0.7, 0.9)]
        gate.score_response = MagicMock(side_effect=scores)

        responses = [
            ("agent1", "High quality"),
            ("agent2", "Low quality"),
            ("agent3", "Medium quality"),
        ]

        result = gate.filter_responses(responses)

        # agent2 should be filtered out
        assert len(result) == 2
        assert result[0][0] == "agent1"
        assert result[1][0] == "agent3"

    def test_filter_responses_low_confidence_passes(self):
        """Test that low confidence responses pass filter."""
        gate = QualityGate(threshold=0.7, min_confidence=0.5)

        scores = [(0.3, 0.3)]  # Low quality but low confidence
        gate.score_response = MagicMock(side_effect=scores)

        responses = [("agent1", "Uncertain quality")]

        result = gate.filter_responses(responses)

        # Should pass through
        assert len(result) == 1

    def test_filter_messages(self):
        """Test filtering Message objects."""
        gate = QualityGate(threshold=0.6, min_confidence=0.5)

        scores = [(0.8, 0.9), (0.4, 0.9)]
        gate.score_response = MagicMock(side_effect=scores)

        messages = [
            MockMessage("agent1", "High quality"),
            MockMessage("agent2", "Low quality"),
        ]

        result = gate.filter_messages(messages)

        assert len(result) == 1
        assert result[0][0].agent == "agent1"
        assert result[0][1] == 0.8

    def test_filter_messages_empty(self):
        """Test filtering empty message list."""
        gate = QualityGate()

        result = gate.filter_messages([])

        assert result == []


# ===========================================================================
# Test: ConsensusEstimator
# ===========================================================================


class TestConsensusEstimator:
    """Tests for ConsensusEstimator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        estimator = ConsensusEstimator()

        assert estimator.threshold == 0.85
        assert estimator.min_rounds == 2
        assert estimator._predictor is None
        assert estimator._similarity_history == []

    def test_init_custom(self):
        """Test custom initialization."""
        estimator = ConsensusEstimator(
            early_termination_threshold=0.9,
            min_rounds=3,
        )

        assert estimator.threshold == 0.9
        assert estimator.min_rounds == 3

    def test_estimate_consensus_no_predictor(self):
        """Test estimation without predictor."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=None)

        result = estimator.estimate_consensus([("agent1", "text")], current_round=1)

        assert result["probability"] == 0.5
        assert result["confidence"] == 0.0
        assert result["trend"] == "unknown"
        assert result["recommendation"] == "continue"

    def test_estimate_consensus_with_predictor(self, mock_consensus_predictor):
        """Test estimation with predictor available."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        responses = [("agent1", "text1"), ("agent2", "text2")]
        result = estimator.estimate_consensus(responses, current_round=2, total_rounds=3)

        assert result["probability"] == 0.9
        assert result["confidence"] == 0.85
        assert result["trend"] == "converging"
        assert result["recommendation"] == "terminate"

    def test_estimate_consensus_updates_history(self, mock_consensus_predictor):
        """Test that similarity history is updated."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        estimator.estimate_consensus([("agent1", "text")], current_round=1)

        assert len(estimator._similarity_history) == 1
        assert estimator._similarity_history[0] == 0.88

    def test_estimate_consensus_early_round_no_terminate(self, mock_consensus_predictor):
        """Test no termination recommendation for early rounds."""
        estimator = ConsensusEstimator(min_rounds=3)
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.estimate_consensus(
            [("agent1", "text")],
            current_round=2,  # Before min_rounds
            total_rounds=5,
        )

        # Even though early_termination_safe is True, round < min_rounds
        assert result["recommendation"] == "continue"

    def test_estimate_consensus_needs_intervention(self, mock_consensus_predictor):
        """Test intervention recommendation."""
        estimator = ConsensusEstimator()

        @dataclass
        class InterventionPrediction:
            probability: float = 0.3
            confidence: float = 0.8
            early_termination_safe: bool = False
            needs_intervention: bool = True
            convergence_trend: str = "diverging"
            estimated_rounds: int = 5
            key_factors: list = None
            features: dict = None

            def __post_init__(self):
                self.key_factors = []
                self.features = {}

        mock_consensus_predictor.predict.return_value = InterventionPrediction()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.estimate_consensus([("agent1", "text")], current_round=2)

        assert result["recommendation"] == "intervene"

    def test_estimate_consensus_handles_errors(self, mock_consensus_predictor):
        """Test error handling in estimation."""
        estimator = ConsensusEstimator()
        mock_consensus_predictor.predict.side_effect = ValueError("Error")
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.estimate_consensus([("agent1", "text")], current_round=1)

        assert result["probability"] == 0.5
        assert result["recommendation"] == "continue"

    def test_should_terminate_early_before_min_rounds(self):
        """Test no early termination before min_rounds."""
        estimator = ConsensusEstimator(min_rounds=3)

        result = estimator.should_terminate_early(
            responses=[("agent1", "text")],
            current_round=2,
            total_rounds=5,
        )

        assert result is False

    def test_should_terminate_early_after_min_rounds(self, mock_consensus_predictor):
        """Test early termination check after min_rounds."""
        estimator = ConsensusEstimator(min_rounds=2)
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.should_terminate_early(
            responses=[("agent1", "text")],
            current_round=3,
            total_rounds=5,
        )

        assert result is True

    def test_record_outcome(self, mock_consensus_predictor):
        """Test recording debate outcome."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        estimator.record_outcome("debate-123", reached_consensus=True)

        mock_consensus_predictor.record_outcome.assert_called_once_with("debate-123", True)

    def test_record_outcome_no_predictor(self):
        """Test recording outcome without predictor (no-op)."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=None)

        # Should not raise
        estimator.record_outcome("debate-123", reached_consensus=False)

    def test_record_outcome_handles_errors(self, mock_consensus_predictor):
        """Test recording outcome error handling."""
        estimator = ConsensusEstimator()
        mock_consensus_predictor.record_outcome.side_effect = ValueError("Error")
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        # Should not raise
        estimator.record_outcome("debate-123", reached_consensus=True)

    def test_reset_history(self):
        """Test resetting similarity history."""
        estimator = ConsensusEstimator()
        estimator._similarity_history = [0.5, 0.6, 0.7]

        estimator.reset_history()

        assert estimator._similarity_history == []


# ===========================================================================
# Test: MLEnhancedTeamSelector
# ===========================================================================


class TestMLEnhancedTeamSelector:
    """Tests for MLEnhancedTeamSelector class."""

    @pytest.fixture
    def mock_base_selector(self, mock_agents):
        """Create mock base team selector."""
        selector = MagicMock()
        selector.select = MagicMock(return_value=mock_agents)
        return selector

    def test_init(self, mock_base_selector):
        """Test initialization."""
        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_weight=0.4,
        )

        assert selector.base_selector is mock_base_selector
        assert selector.ml_weight == 0.4
        assert selector.ml_delegation is not None

    def test_init_with_ml_delegation(self, mock_base_selector):
        """Test initialization with custom ML delegation."""
        ml_delegation = MLDelegationStrategy()
        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_delegation=ml_delegation,
            ml_weight=0.3,
        )

        assert selector.ml_delegation is ml_delegation

    def test_select_no_task(self, mock_base_selector, mock_agents):
        """Test selection without task uses base selector only."""
        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_weight=0.3,
        )

        result = selector.select(agents=mock_agents, domain="general", task="")

        mock_base_selector.select.assert_called_once()
        assert result == mock_agents

    def test_select_with_task(self, mock_base_selector, mock_agents):
        """Test selection with task combines scores."""
        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_weight=0.3,
        )

        # ML delegation returns same agents
        selector.ml_delegation.select_agents = MagicMock(return_value=mock_agents)

        result = selector.select(
            agents=mock_agents,
            domain="general",
            task="Test task",
        )

        assert len(result) == 3

    def test_select_combines_scores(self, mock_base_selector, mock_agents):
        """Test that scores are properly combined."""
        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_weight=0.5,  # Equal weight
        )

        # Base selector: claude, gpt4, gemini
        mock_base_selector.select.return_value = mock_agents

        # ML delegation: gemini, gpt4, claude (reversed)
        reversed_agents = list(reversed(mock_agents))
        selector.ml_delegation.select_agents = MagicMock(return_value=reversed_agents)

        result = selector.select(
            agents=mock_agents,
            domain="general",
            task="Test task",
        )

        # With equal weights and opposite orderings,
        # scores should be balanced - middle agent might end up first
        assert len(result) == 3


# ===========================================================================
# Test: create_ml_team_selector factory
# ===========================================================================


class TestCreateMLTeamSelector:
    """Tests for create_ml_team_selector factory function."""

    def test_create_basic(self):
        """Test creating basic selector."""
        with patch("aragora.debate.team_selector.TeamSelector") as MockTeamSelector:
            MockTeamSelector.return_value = MagicMock()

            selector = create_ml_team_selector()

            assert isinstance(selector, MLEnhancedTeamSelector)
            MockTeamSelector.assert_called_once()

    def test_create_with_elo(self):
        """Test creating selector with ELO system."""
        mock_elo = MagicMock()

        with patch("aragora.debate.team_selector.TeamSelector") as MockTeamSelector:
            MockTeamSelector.return_value = MagicMock()

            selector = create_ml_team_selector(elo_system=mock_elo)

            # Check ELO was passed to TeamSelector
            call_kwargs = MockTeamSelector.call_args[1]
            assert call_kwargs["elo_system"] is mock_elo

    def test_create_with_all_options(self):
        """Test creating selector with all options."""
        mock_elo = MagicMock()
        mock_calibration = MagicMock()
        mock_breaker = MagicMock()

        with patch("aragora.debate.team_selector.TeamSelector") as MockTeamSelector:
            MockTeamSelector.return_value = MagicMock()

            selector = create_ml_team_selector(
                elo_system=mock_elo,
                calibration_tracker=mock_calibration,
                circuit_breaker=mock_breaker,
                ml_weight=0.5,
            )

            assert selector.ml_weight == 0.5

            call_kwargs = MockTeamSelector.call_args[1]
            assert call_kwargs["elo_system"] is mock_elo
            assert call_kwargs["calibration_tracker"] is mock_calibration
            assert call_kwargs["circuit_breaker"] is mock_breaker


# ===========================================================================
# Test: DebateTrainingExporter
# ===========================================================================


class TestDebateTrainingExporter:
    """Tests for DebateTrainingExporter class."""

    def test_init(self):
        """Test initialization."""
        exporter = DebateTrainingExporter()

        assert exporter._training_data is None

    def test_export_debate_no_ml_module(self):
        """Test export when ML module unavailable."""
        exporter = DebateTrainingExporter()
        exporter._get_training_data_class = MagicMock(return_value=None)

        result = exporter.export_debate(
            task="Test task",
            consensus_response="Consensus answer",
        )

        assert result is None

    def test_export_debate_with_ml_module(self):
        """Test export with ML module available."""
        exporter = DebateTrainingExporter()

        # Mock training data classes
        MockTrainingData = MagicMock()
        MockTrainingExample = MagicMock()
        MockTrainingExample.from_debate = MagicMock(return_value=MagicMock())

        mock_data_instance = MagicMock()
        MockTrainingData.return_value = mock_data_instance

        exporter._get_training_data_class = MagicMock(
            return_value=(MockTrainingData, MockTrainingExample)
        )

        result = exporter.export_debate(
            task="Test task",
            consensus_response="Consensus answer",
            rejected_responses=["Rejected 1"],
            context="Additional context",
        )

        assert result is mock_data_instance
        MockTrainingExample.from_debate.assert_called_once_with(
            task="Test task",
            winning_response="Consensus answer",
            losing_response="Rejected 1",
            context="Additional context",
        )

    def test_export_debate_no_rejected(self):
        """Test export without rejected responses."""
        exporter = DebateTrainingExporter()

        MockTrainingData = MagicMock()
        MockTrainingExample = MagicMock()
        MockTrainingExample.from_debate = MagicMock(return_value=MagicMock())
        MockTrainingData.return_value = MagicMock()

        exporter._get_training_data_class = MagicMock(
            return_value=(MockTrainingData, MockTrainingExample)
        )

        exporter.export_debate(
            task="Test task",
            consensus_response="Consensus answer",
        )

        # losing_response should be empty string
        call_kwargs = MockTrainingExample.from_debate.call_args[1]
        assert call_kwargs["losing_response"] == ""

    def test_export_debates_batch_no_ml_module(self):
        """Test batch export when ML module unavailable."""
        exporter = DebateTrainingExporter()
        exporter._get_training_data_class = MagicMock(return_value=None)

        result = exporter.export_debates_batch([{"task": "Task", "consensus": "Answer"}])

        assert result is None

    def test_export_debates_batch(self):
        """Test batch export."""
        exporter = DebateTrainingExporter()

        MockTrainingData = MagicMock()
        MockTrainingExample = MagicMock()
        MockTrainingExample.from_debate = MagicMock(return_value=MagicMock())

        mock_data_instance = MagicMock()
        MockTrainingData.return_value = mock_data_instance

        exporter._get_training_data_class = MagicMock(
            return_value=(MockTrainingData, MockTrainingExample)
        )

        debates = [
            {"task": "Task 1", "consensus": "Answer 1"},
            {"task": "Task 2", "consensus": "Answer 2", "rejected": ["Alt 2"]},
            {"task": "", "consensus": ""},  # Should be skipped
        ]

        result = exporter.export_debates_batch(debates)

        assert result is mock_data_instance
        # Only 2 debates should be added (third has empty task)
        assert mock_data_instance.add.call_count == 2


# ===========================================================================
# Test: Singleton Getters
# ===========================================================================


class TestSingletonGetters:
    """Tests for singleton getter functions."""

    def test_get_ml_delegation(self):
        """Test getting ML delegation singleton."""
        import aragora.debate.ml_integration as ml_mod

        # Reset singleton
        ml_mod._ml_delegation = None

        delegation = get_ml_delegation()

        assert isinstance(delegation, MLDelegationStrategy)

        # Second call should return same instance
        delegation2 = get_ml_delegation()
        assert delegation is delegation2

    def test_get_quality_gate(self):
        """Test getting quality gate singleton."""
        import aragora.debate.ml_integration as ml_mod

        ml_mod._quality_gate = None

        gate = get_quality_gate(threshold=0.7)

        assert isinstance(gate, QualityGate)
        assert gate.threshold == 0.7

        # Second call returns same instance (ignores new threshold)
        gate2 = get_quality_gate(threshold=0.9)
        assert gate is gate2

    def test_get_consensus_estimator(self):
        """Test getting consensus estimator singleton."""
        import aragora.debate.ml_integration as ml_mod

        ml_mod._consensus_estimator = None

        estimator = get_consensus_estimator()

        assert isinstance(estimator, ConsensusEstimator)

        estimator2 = get_consensus_estimator()
        assert estimator is estimator2

    def test_get_training_exporter(self):
        """Test getting training exporter singleton."""
        import aragora.debate.ml_integration as ml_mod

        ml_mod._training_exporter = None

        exporter = get_training_exporter()

        assert isinstance(exporter, DebateTrainingExporter)

        exporter2 = get_training_exporter()
        assert exporter is exporter2


# ===========================================================================
# Test: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ml_delegation_with_single_agent(self, mock_router):
        """Test ML delegation with single agent."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        agents = [MockAgent("solo")]
        result = strategy.select_agents("Task", agents)

        assert len(result) == 1

    def test_quality_gate_empty_text(self, mock_quality_scorer):
        """Test quality gate with empty text."""
        gate = QualityGate()
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        quality, confidence = gate.score_response("")

        mock_quality_scorer.score.assert_called_once_with("", context=None)

    def test_consensus_estimator_empty_responses(self, mock_consensus_predictor):
        """Test consensus estimator with empty responses."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.estimate_consensus([], current_round=1)

        # Should still call predictor
        mock_consensus_predictor.predict.assert_called_once()

    def test_ml_delegation_very_long_task(self, mock_router):
        """Test ML delegation with very long task string."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        long_task = "A" * 10000
        agents = [MockAgent("agent1")]

        # Should handle long task without error
        result = strategy.select_agents(long_task, agents)

        assert len(result) == 1

    def test_cache_key_truncation(self):
        """Test cache key truncates long tasks."""
        strategy = MLDelegationStrategy()

        long_task = "A" * 200
        key = strategy._get_cache_key(long_task, ["agent1"])

        # Task should be truncated to 100 chars
        assert len(key.split(":")[0]) == 100

    def test_ml_delegation_context_parameter(self, mock_agents, mock_router):
        """Test ML delegation passes context correctly."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        mock_context = MagicMock()

        result = strategy.select_agents("Task", mock_agents, context=mock_context)

        assert len(result) == 3

    def test_consensus_estimator_context_passed(self, mock_consensus_predictor):
        """Test consensus estimator passes context to predictor."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        responses = [("agent1", "text")]
        estimator.estimate_consensus(
            responses,
            context="Task context",
            current_round=1,
            total_rounds=3,
        )

        call_kwargs = mock_consensus_predictor.predict.call_args[1]
        assert call_kwargs["context"] == "Task context"

    def test_quality_gate_attribute_error_handling(self, mock_quality_scorer):
        """Test quality gate handles AttributeError."""
        gate = QualityGate()
        mock_quality_scorer.score.side_effect = AttributeError("Missing attribute")
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        quality, confidence = gate.score_response("Text")

        assert quality == 0.5
        assert confidence == 0.0

    def test_consensus_estimator_key_error_handling(self, mock_consensus_predictor):
        """Test consensus estimator handles KeyError."""
        estimator = ConsensusEstimator()
        mock_consensus_predictor.predict.side_effect = KeyError("Missing key")
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        result = estimator.estimate_consensus([("agent1", "text")], current_round=1)

        assert result["probability"] == 0.5
        assert result["recommendation"] == "continue"


# ===========================================================================
# Test: Cold Start Scenarios
# ===========================================================================


class TestColdStartScenarios:
    """Tests for cold start and initialization scenarios."""

    def test_ml_delegation_first_call_no_cache(self, mock_agents, mock_router):
        """Test first call without cache."""
        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=mock_router)

        assert len(strategy._cache) == 0

        result = strategy.select_agents("Task", mock_agents)

        assert len(strategy._cache) == 1
        assert len(result) == 3

    def test_consensus_estimator_first_call_no_history(self, mock_consensus_predictor):
        """Test first call without similarity history."""
        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        assert len(estimator._similarity_history) == 0

        estimator.estimate_consensus([("agent1", "text")], current_round=1)

        assert len(estimator._similarity_history) == 1

    def test_ml_delegation_with_uninitialized_elo(self, mock_agents, mock_router):
        """Test ML delegation with uninitialized ELO system."""
        strategy = MLDelegationStrategy(elo_system=None)
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Task", mock_agents)

        assert len(result) == 3

    def test_quality_gate_first_score(self, mock_quality_scorer):
        """Test quality gate first scoring call."""
        gate = QualityGate()
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        # First call
        quality, confidence = gate.score_response("Test")

        assert quality == 0.75
        assert confidence == 0.8


# ===========================================================================
# Test: Integration Between Components
# ===========================================================================


class TestComponentIntegration:
    """Tests for integration between ML components."""

    def test_quality_gate_with_consensus_estimator(
        self, mock_quality_scorer, mock_consensus_predictor
    ):
        """Test quality gate and consensus estimator work together."""
        gate = QualityGate(threshold=0.6)
        gate._get_scorer = MagicMock(return_value=mock_quality_scorer)

        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=mock_consensus_predictor)

        # First filter responses
        responses = [("agent1", "response1"), ("agent2", "response2")]
        filtered = gate.filter_responses(responses)

        # Then estimate consensus on filtered responses
        consensus_input = [(r[0], r[1]) for r in filtered]
        result = estimator.estimate_consensus(consensus_input, current_round=2)

        assert result["recommendation"] == "terminate"

    def test_ml_delegation_with_team_selector(self, mock_agents, mock_router):
        """Test ML delegation integrates with team selector."""
        ml_delegation = MLDelegationStrategy()
        ml_delegation._get_router = MagicMock(return_value=mock_router)

        mock_base_selector = MagicMock()
        mock_base_selector.select.return_value = mock_agents

        selector = MLEnhancedTeamSelector(
            base_selector=mock_base_selector,
            ml_delegation=ml_delegation,
            ml_weight=0.5,
        )

        result = selector.select(
            agents=mock_agents,
            domain="general",
            task="Test task",
        )

        assert len(result) == 3
        mock_router.route.assert_called()
