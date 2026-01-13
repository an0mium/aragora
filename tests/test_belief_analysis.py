"""Tests for belief analysis phase functionality."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from aragora.debate.phases.belief_analysis import (
    DebateBeliefAnalyzer,
    BeliefAnalysisResult,
    _load_belief_classes,
)


@dataclass
class MockMessage:
    """Mock debate message."""

    agent: str
    content: str
    role: str = "proposer"


@dataclass
class MockClaim:
    """Mock grounded claim."""

    statement: str
    confidence: float = 0.5
    claim_id: str = ""

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = f"claim_{hash(self.statement[:20])}"


class TestBeliefAnalysisResult:
    """Tests for BeliefAnalysisResult dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        result = BeliefAnalysisResult()
        assert result.cruxes == []
        assert result.evidence_suggestions == []
        assert result.network_size == 0
        assert result.analysis_error is None


class TestDebateBeliefAnalyzer:
    """Tests for DebateBeliefAnalyzer."""

    def test_initialization(self):
        """Analyzer initializes with default parameters."""
        analyzer = DebateBeliefAnalyzer()
        assert analyzer.propagation_iterations == 3
        assert analyzer.crux_threshold == 0.6
        assert analyzer.max_claims == 20

    def test_custom_parameters(self):
        """Analyzer accepts custom parameters."""
        analyzer = DebateBeliefAnalyzer(
            propagation_iterations=5,
            crux_threshold=0.8,
            max_claims=50,
        )
        assert analyzer.propagation_iterations == 5
        assert analyzer.crux_threshold == 0.8
        assert analyzer.max_claims == 50

    def test_analyze_empty_messages(self):
        """Empty message list returns empty result."""
        analyzer = DebateBeliefAnalyzer()
        result = analyzer.analyze_messages([])
        assert result.cruxes == []
        assert result.network_size == 0

    def test_analyze_empty_claims(self):
        """Empty claims list returns empty result."""
        analyzer = DebateBeliefAnalyzer()
        result = analyzer.analyze_claims([])
        assert result.cruxes == []
        assert result.network_size == 0

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_analyze_messages_without_belief_module(self, mock_load):
        """Returns error when belief module not available."""
        mock_load.return_value = (None, None)
        analyzer = DebateBeliefAnalyzer()
        messages = [MockMessage("agent1", "Test claim content")]

        result = analyzer.analyze_messages(messages)

        assert result.analysis_error == "Belief module not available"

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_analyze_claims_without_belief_module(self, mock_load):
        """Returns error when belief module not available."""
        mock_load.return_value = (None, None)
        analyzer = DebateBeliefAnalyzer()
        claims = [MockClaim("Test statement")]

        result = analyzer.analyze_claims(claims)

        assert result.analysis_error == "Belief module not available"

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_analyze_messages_with_mock_belief(self, mock_load):
        """Analyzes messages when belief module is available."""
        # Setup mocks
        mock_network = MagicMock()
        mock_network.nodes = {"node1": {}, "node2": {}}
        mock_analyzer = MagicMock()
        mock_analyzer.identify_debate_cruxes.return_value = [
            {"claim": "Test crux", "uncertainty": 0.7}
        ]
        mock_analyzer.suggest_evidence_targets.return_value = ["Evidence 1", "Evidence 2"]

        mock_BN = MagicMock(return_value=mock_network)
        mock_BPA = MagicMock(return_value=mock_analyzer)
        mock_load.return_value = (mock_BN, mock_BPA)

        analyzer = DebateBeliefAnalyzer()
        messages = [
            MockMessage("agent1", "First claim about topic", "proposer"),
            MockMessage("agent2", "Critique of first claim", "critic"),
        ]

        result = analyzer.analyze_messages(messages)

        assert result.network_size == 2
        assert len(result.cruxes) == 1
        assert result.cruxes[0]["claim"] == "Test crux"
        assert len(result.evidence_suggestions) == 2
        assert result.analysis_error is None

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_analyze_claims_with_mock_belief(self, mock_load):
        """Analyzes grounded claims when belief module is available."""
        mock_analyzer = MagicMock()
        mock_analyzer.identify_debate_cruxes.return_value = [
            {"claim": "Crux claim", "uncertainty": 0.8}
        ]

        mock_BPA = MagicMock(return_value=mock_analyzer)
        mock_load.return_value = (None, mock_BPA)

        analyzer = DebateBeliefAnalyzer()
        claims = [
            MockClaim("Statement one", confidence=0.7),
            MockClaim("Statement two", confidence=0.3),
        ]

        result = analyzer.analyze_claims(claims)

        assert result.network_size == 2
        assert len(result.cruxes) == 1
        mock_analyzer.add_claim.call_count == 2

    def test_filters_non_debate_roles(self):
        """Only processes proposer and critic messages."""
        mock_network = MagicMock()
        mock_network.nodes = {}
        mock_analyzer = MagicMock()
        mock_analyzer.identify_debate_cruxes.return_value = []
        mock_analyzer.suggest_evidence_targets.return_value = []

        mock_BN = MagicMock(return_value=mock_network)
        mock_BPA = MagicMock(return_value=mock_analyzer)

        with patch("aragora.debate.phases.belief_analysis._load_belief_classes") as mock_load:
            mock_load.return_value = (mock_BN, mock_BPA)
            analyzer = DebateBeliefAnalyzer()

            messages = [
                MockMessage("agent1", "Valid claim", "proposer"),
                MockMessage("agent2", "System message", "system"),
                MockMessage("agent3", "Another valid", "critic"),
            ]

            result = analyzer.analyze_messages(messages)

            # Should only add_claim for proposer and critic
            assert mock_network.add_claim.call_count == 2

    def test_respects_max_claims_limit(self):
        """Stops adding claims after max_claims limit."""
        mock_network = MagicMock()
        mock_network.nodes = {}
        mock_analyzer = MagicMock()
        mock_analyzer.identify_debate_cruxes.return_value = []
        mock_analyzer.suggest_evidence_targets.return_value = []

        mock_BN = MagicMock(return_value=mock_network)
        mock_BPA = MagicMock(return_value=mock_analyzer)

        with patch("aragora.debate.phases.belief_analysis._load_belief_classes") as mock_load:
            mock_load.return_value = (mock_BN, mock_BPA)
            analyzer = DebateBeliefAnalyzer(max_claims=3)

            messages = [MockMessage(f"agent{i}", f"Claim {i}") for i in range(10)]

            analyzer.analyze_messages(messages)

            # Should only add 3 claims
            assert mock_network.add_claim.call_count == 3

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_handles_propagation_error(self, mock_load):
        """Handles errors during belief propagation."""
        mock_network = MagicMock()
        mock_network.nodes = {"node1": {}}
        mock_network.propagate.side_effect = ValueError("Propagation failed")

        mock_BN = MagicMock(return_value=mock_network)
        mock_load.return_value = (mock_BN, MagicMock())

        analyzer = DebateBeliefAnalyzer()
        messages = [MockMessage("agent1", "Test claim")]

        result = analyzer.analyze_messages(messages)

        assert result.analysis_error is not None
        assert "Propagation failed" in result.analysis_error

    @patch("aragora.debate.phases.belief_analysis._load_belief_classes")
    def test_handles_analyzer_error(self, mock_load):
        """Handles errors during crux identification."""
        mock_analyzer = MagicMock()
        mock_analyzer.identify_debate_cruxes.side_effect = RuntimeError("Analysis failed")

        mock_BPA = MagicMock(return_value=mock_analyzer)
        mock_load.return_value = (None, mock_BPA)

        analyzer = DebateBeliefAnalyzer()
        claims = [MockClaim("Test statement")]

        result = analyzer.analyze_claims(claims)

        assert result.analysis_error is not None
        assert "Analysis failed" in result.analysis_error
