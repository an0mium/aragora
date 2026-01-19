"""
Tests for the ML Integration module.

Tests ML-powered agent selection, quality gates, consensus estimation,
and training data export.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class MockAgent:
    """Mock Agent for testing."""

    name: str
    role: str = "debater"


@dataclass
class MockMessage:
    """Mock Message for testing."""

    agent: str
    content: str


@dataclass
class MockRoutingDecision:
    """Mock ML routing decision."""

    task_type: MagicMock
    confidence: float
    selected_agents: List[str]
    agent_scores: Dict[str, float]


@dataclass
class MockQualityScore:
    """Mock quality score result."""

    overall: float
    confidence: float


@dataclass
class MockConsensusPrediction:
    """Mock consensus prediction result."""

    probability: float
    confidence: float
    convergence_trend: str
    early_termination_safe: bool
    needs_intervention: bool
    estimated_rounds: int
    key_factors: List[str]
    features: Dict[str, float]


class TestMLIntegrationConfig:
    """Tests for MLIntegrationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.debate.ml_integration import MLIntegrationConfig

        config = MLIntegrationConfig()

        assert config.use_ml_routing is True
        assert config.ml_routing_weight == 0.4
        assert config.fallback_to_elo is True
        assert config.enable_quality_gates is True
        assert config.quality_threshold == 0.6
        assert config.enable_early_termination is True
        assert config.early_termination_threshold == 0.85
        assert config.min_rounds_before_termination == 2

    def test_custom_values(self):
        """Should accept custom configuration."""
        from aragora.debate.ml_integration import MLIntegrationConfig

        config = MLIntegrationConfig(
            use_ml_routing=False,
            quality_threshold=0.8,
            early_termination_threshold=0.9,
        )

        assert config.use_ml_routing is False
        assert config.quality_threshold == 0.8
        assert config.early_termination_threshold == 0.9


class TestMLDelegationStrategy:
    """Tests for MLDelegationStrategy class."""

    def test_init_defaults(self):
        """Should initialize with default config."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()

        assert strategy.config is not None
        assert strategy._router is None
        assert strategy._cache == {}

    def test_select_agents_empty_list(self):
        """Should return empty list for empty agents."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()

        result = strategy.select_agents("Test task", [])

        assert result == []

    def test_select_agents_fallback_without_router(self):
        """Should return agents as-is when router unavailable."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]

        # Mock _get_router to return None
        strategy._get_router = MagicMock(return_value=None)

        result = strategy.select_agents("Test task", agents)

        assert len(result) == 3
        assert result[0].name == "alice"

    def test_select_agents_with_max_agents(self):
        """Should limit results when max_agents specified."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        strategy._get_router = MagicMock(return_value=None)

        result = strategy.select_agents("Test task", agents, max_agents=2)

        assert len(result) == 2

    def test_select_agents_uses_cache(self):
        """Should use cached routing decisions."""
        from aragora.debate.ml_integration import MLDelegationStrategy, MLIntegrationConfig

        config = MLIntegrationConfig(cache_routing_decisions=True)
        strategy = MLDelegationStrategy(config=config)
        agents = [MockAgent("alice"), MockAgent("bob")]

        # Pre-populate cache
        cache_key = strategy._get_cache_key("Test task", ["alice", "bob"])
        strategy._cache[cache_key] = (["bob", "alice"], 0.9)

        result = strategy.select_agents("Test task", agents)

        # Should return in cached order
        assert result[0].name == "bob"
        assert result[1].name == "alice"

    def test_select_agents_with_mock_router(self):
        """Should use ML router when available."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]

        mock_decision = MockRoutingDecision(
            task_type=MagicMock(value="coding"),
            confidence=0.85,
            selected_agents=["charlie", "alice", "bob"],
            agent_scores={"charlie": 0.9, "alice": 0.7, "bob": 0.5},
        )

        mock_router = MagicMock()
        mock_router.route.return_value = mock_decision
        strategy._get_router = MagicMock(return_value=mock_router)

        result = strategy.select_agents("Implement caching", agents)

        assert result[0].name == "charlie"
        assert result[1].name == "alice"
        assert result[2].name == "bob"

    def test_score_agent_without_router(self):
        """Should return neutral score without router."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()
        strategy._get_router = MagicMock(return_value=None)

        score = strategy.score_agent(MockAgent("alice"), "Test task")

        assert score == 2.5

    def test_score_agent_with_router(self):
        """Should return ML-based score with router."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()

        mock_decision = MockRoutingDecision(
            task_type=MagicMock(value="coding"),
            confidence=0.8,
            selected_agents=["alice"],
            agent_scores={"alice": 0.9},
        )

        mock_router = MagicMock()
        mock_router.route.return_value = mock_decision
        strategy._get_router = MagicMock(return_value=mock_router)

        score = strategy.score_agent(MockAgent("alice"), "Test task")

        # 0.9 * 5.0 = 4.5
        assert score == 4.5

    def test_reorder_agents(self):
        """Should reorder agents according to ML order."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]

        result = strategy._reorder_agents(agents, ["charlie", "alice"])

        assert result[0].name == "charlie"
        assert result[1].name == "alice"
        assert result[2].name == "bob"  # Not in order, appended

    def test_cache_key_generation(self):
        """Should generate consistent cache keys."""
        from aragora.debate.ml_integration import MLDelegationStrategy

        strategy = MLDelegationStrategy()

        key1 = strategy._get_cache_key("Test task", ["bob", "alice"])
        key2 = strategy._get_cache_key("Test task", ["alice", "bob"])

        # Should be same (sorted)
        assert key1 == key2


class TestQualityGate:
    """Tests for QualityGate class."""

    def test_init_defaults(self):
        """Should initialize with default thresholds."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate()

        assert gate.threshold == 0.6
        assert gate.min_confidence == 0.4

    def test_init_custom_thresholds(self):
        """Should accept custom thresholds."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.8, min_confidence=0.5)

        assert gate.threshold == 0.8
        assert gate.min_confidence == 0.5

    def test_score_response_without_scorer(self):
        """Should return unknown score without scorer."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate()
        gate._get_scorer = MagicMock(return_value=None)

        score, confidence = gate.score_response("Test response")

        assert score == 0.5
        assert confidence == 0.0

    def test_score_response_with_scorer(self):
        """Should return ML-based score with scorer."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate()

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MockQualityScore(overall=0.85, confidence=0.9)
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        score, confidence = gate.score_response("High quality response")

        assert score == 0.85
        assert confidence == 0.9

    def test_passes_gate_high_quality(self):
        """High quality response should pass gate."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.6, min_confidence=0.4)

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MockQualityScore(overall=0.8, confidence=0.9)
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        assert gate.passes_gate("Good response") is True

    def test_passes_gate_low_quality(self):
        """Low quality response should not pass gate."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.6, min_confidence=0.4)

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MockQualityScore(overall=0.3, confidence=0.9)
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        assert gate.passes_gate("Bad response") is False

    def test_passes_gate_low_confidence(self):
        """Low confidence responses should pass through (uncertain)."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.6, min_confidence=0.4)

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MockQualityScore(overall=0.3, confidence=0.2)
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        # Low quality but also low confidence - should pass
        assert gate.passes_gate("Uncertain response") is True

    def test_filter_responses(self):
        """Should filter out low quality responses."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.6, min_confidence=0.4)

        mock_scorer = MagicMock()

        def mock_score(text, context=None):
            if "good" in text.lower():
                return MockQualityScore(overall=0.8, confidence=0.9)
            return MockQualityScore(overall=0.3, confidence=0.9)

        mock_scorer.score.side_effect = mock_score
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        responses = [
            ("alice", "This is a good response"),
            ("bob", "This is a bad response"),
            ("charlie", "Another good response"),
        ]

        filtered = gate.filter_responses(responses)

        assert len(filtered) == 2
        assert filtered[0][0] == "alice"
        assert filtered[1][0] == "charlie"

    def test_filter_messages(self):
        """Should filter Message objects by quality."""
        from aragora.debate.ml_integration import QualityGate

        gate = QualityGate(threshold=0.6, min_confidence=0.4)

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MockQualityScore(overall=0.8, confidence=0.9)
        gate._get_scorer = MagicMock(return_value=mock_scorer)

        messages = [
            MockMessage("alice", "Good response"),
            MockMessage("bob", "Another response"),
        ]

        filtered = gate.filter_messages(messages)

        assert len(filtered) == 2
        assert filtered[0][1] == 0.8  # Score


class TestConsensusEstimator:
    """Tests for ConsensusEstimator class."""

    def test_init_defaults(self):
        """Should initialize with default thresholds."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator()

        assert estimator.threshold == 0.85
        assert estimator.min_rounds == 2
        assert estimator._similarity_history == []

    def test_estimate_consensus_without_predictor(self):
        """Should return unknown estimate without predictor."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator()
        estimator._get_predictor = MagicMock(return_value=None)

        responses = [("alice", "Response A"), ("bob", "Response B")]
        estimate = estimator.estimate_consensus(responses)

        assert estimate["probability"] == 0.5
        assert estimate["confidence"] == 0.0
        assert estimate["recommendation"] == "continue"

    def test_estimate_consensus_with_predictor(self):
        """Should return ML-based estimate with predictor."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator()

        mock_prediction = MockConsensusPrediction(
            probability=0.9,
            confidence=0.85,
            convergence_trend="converging",
            early_termination_safe=True,
            needs_intervention=False,
            estimated_rounds=1,
            key_factors=["semantic_similarity", "position_alignment"],
            features={"semantic_similarity": 0.85},
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_prediction
        estimator._get_predictor = MagicMock(return_value=mock_predictor)

        responses = [("alice", "Response A"), ("bob", "Response B")]
        estimate = estimator.estimate_consensus(responses, current_round=3)

        assert estimate["probability"] == 0.9
        assert estimate["confidence"] == 0.85
        assert estimate["recommendation"] == "terminate"

    def test_should_terminate_early_before_min_rounds(self):
        """Should not terminate before minimum rounds."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator(min_rounds=2)

        responses = [("alice", "Response A")]
        result = estimator.should_terminate_early(responses, current_round=1)

        assert result is False

    def test_should_terminate_early_after_min_rounds(self):
        """Should check termination after minimum rounds."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator(min_rounds=2)

        mock_prediction = MockConsensusPrediction(
            probability=0.9,
            confidence=0.9,
            convergence_trend="converging",
            early_termination_safe=True,
            needs_intervention=False,
            estimated_rounds=0,
            key_factors=[],
            features={},
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_prediction
        estimator._get_predictor = MagicMock(return_value=mock_predictor)

        responses = [("alice", "Response A")]
        result = estimator.should_terminate_early(responses, current_round=3)

        assert result is True

    def test_reset_history(self):
        """Should clear similarity history."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator()
        estimator._similarity_history = [0.5, 0.6, 0.7]

        estimator.reset_history()

        assert estimator._similarity_history == []

    def test_tracks_similarity_history(self):
        """Should track semantic similarity over rounds."""
        from aragora.debate.ml_integration import ConsensusEstimator

        estimator = ConsensusEstimator()

        mock_prediction = MockConsensusPrediction(
            probability=0.7,
            confidence=0.8,
            convergence_trend="converging",
            early_termination_safe=False,
            needs_intervention=False,
            estimated_rounds=2,
            key_factors=[],
            features={"semantic_similarity": 0.75},
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_prediction
        estimator._get_predictor = MagicMock(return_value=mock_predictor)

        responses = [("alice", "Response")]
        estimator.estimate_consensus(responses, current_round=1)

        assert 0.75 in estimator._similarity_history


class TestMLEnhancedTeamSelector:
    """Tests for MLEnhancedTeamSelector class."""

    def test_select_without_task(self):
        """Should use base selector when no task provided."""
        from aragora.debate.ml_integration import MLEnhancedTeamSelector

        mock_base = MagicMock()
        mock_base.select.return_value = [MockAgent("alice"), MockAgent("bob")]

        selector = MLEnhancedTeamSelector(base_selector=mock_base)
        agents = [MockAgent("alice"), MockAgent("bob")]

        result = selector.select(agents, domain="general", task="")

        mock_base.select.assert_called_once()
        assert len(result) == 2

    def test_select_combines_scores(self):
        """Should combine ML and base scores."""
        from aragora.debate.ml_integration import MLEnhancedTeamSelector, MLDelegationStrategy

        mock_base = MagicMock()
        mock_base.select.return_value = [MockAgent("alice"), MockAgent("bob")]

        mock_delegation = MagicMock(spec=MLDelegationStrategy)
        mock_delegation.select_agents.return_value = [MockAgent("bob"), MockAgent("alice")]

        selector = MLEnhancedTeamSelector(
            base_selector=mock_base,
            ml_delegation=mock_delegation,
            ml_weight=0.5,
        )
        agents = [MockAgent("alice"), MockAgent("bob")]

        result = selector.select(agents, domain="coding", task="Implement feature")

        assert len(result) == 2
        # Combined scoring should affect order


class TestDebateTrainingExporter:
    """Tests for DebateTrainingExporter class."""

    def test_export_debate_without_ml_module(self):
        """Should return None when ML module unavailable."""
        from aragora.debate.ml_integration import DebateTrainingExporter

        exporter = DebateTrainingExporter()
        exporter._get_training_data_class = MagicMock(return_value=None)

        result = exporter.export_debate(
            task="Test task",
            consensus_response="Test consensus",
        )

        assert result is None

    def test_export_debate_with_ml_module(self):
        """Should export training data when ML module available."""
        from aragora.debate.ml_integration import DebateTrainingExporter

        exporter = DebateTrainingExporter()

        mock_data = MagicMock()
        mock_example = MagicMock()
        mock_example_class = MagicMock()
        mock_example_class.from_debate.return_value = mock_example

        exporter._get_training_data_class = MagicMock(
            return_value=(MagicMock(return_value=mock_data), mock_example_class)
        )

        result = exporter.export_debate(
            task="Test task",
            consensus_response="Test consensus",
            rejected_responses=["Rejected response"],
        )

        assert result is not None

    def test_export_debates_batch(self):
        """Should export multiple debates."""
        from aragora.debate.ml_integration import DebateTrainingExporter

        exporter = DebateTrainingExporter()

        mock_data = MagicMock()
        mock_example_class = MagicMock()

        exporter._get_training_data_class = MagicMock(
            return_value=(MagicMock(return_value=mock_data), mock_example_class)
        )

        debates = [
            {"task": "Task 1", "consensus": "Consensus 1"},
            {"task": "Task 2", "consensus": "Consensus 2"},
        ]

        result = exporter.export_debates_batch(debates)

        assert result is not None
        assert mock_example_class.from_debate.call_count == 2


class TestSingletonGetters:
    """Tests for singleton getter functions."""

    def test_get_ml_delegation(self):
        """Should return singleton MLDelegationStrategy."""
        from aragora.debate.ml_integration import get_ml_delegation, MLDelegationStrategy
        import aragora.debate.ml_integration as ml_mod

        # Reset singleton
        ml_mod._ml_delegation = None

        result1 = get_ml_delegation()
        result2 = get_ml_delegation()

        assert isinstance(result1, MLDelegationStrategy)
        assert result1 is result2

    def test_get_quality_gate(self):
        """Should return singleton QualityGate."""
        from aragora.debate.ml_integration import get_quality_gate, QualityGate
        import aragora.debate.ml_integration as ml_mod

        # Reset singleton
        ml_mod._quality_gate = None

        result1 = get_quality_gate(threshold=0.7)
        result2 = get_quality_gate()

        assert isinstance(result1, QualityGate)
        assert result1 is result2

    def test_get_consensus_estimator(self):
        """Should return singleton ConsensusEstimator."""
        from aragora.debate.ml_integration import get_consensus_estimator, ConsensusEstimator
        import aragora.debate.ml_integration as ml_mod

        # Reset singleton
        ml_mod._consensus_estimator = None

        result1 = get_consensus_estimator()
        result2 = get_consensus_estimator()

        assert isinstance(result1, ConsensusEstimator)
        assert result1 is result2

    def test_get_training_exporter(self):
        """Should return singleton DebateTrainingExporter."""
        from aragora.debate.ml_integration import get_training_exporter, DebateTrainingExporter
        import aragora.debate.ml_integration as ml_mod

        # Reset singleton
        ml_mod._training_exporter = None

        result1 = get_training_exporter()
        result2 = get_training_exporter()

        assert isinstance(result1, DebateTrainingExporter)
        assert result1 is result2


class TestCreateMLTeamSelector:
    """Tests for create_ml_team_selector factory function."""

    def test_creates_selector(self):
        """Should create MLEnhancedTeamSelector."""
        from aragora.debate.ml_integration import create_ml_team_selector, MLEnhancedTeamSelector

        selector = create_ml_team_selector(ml_weight=0.4)

        assert isinstance(selector, MLEnhancedTeamSelector)
        assert selector.ml_weight == 0.4

    def test_creates_with_elo_system(self):
        """Should pass ELO system to base selector."""
        from aragora.debate.ml_integration import create_ml_team_selector

        mock_elo = MagicMock()
        selector = create_ml_team_selector(elo_system=mock_elo)

        assert selector.base_selector.elo_system is mock_elo
