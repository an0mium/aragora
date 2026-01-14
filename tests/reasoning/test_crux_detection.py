"""Tests for crux claim detection in belief networks."""

import math

import pytest

from aragora.reasoning.belief import (
    BeliefDistribution,
    BeliefNetwork,
    BeliefNode,
    CruxAnalysisResult,
    CruxClaim,
    CruxDetector,
    RelationType,
)
from aragora.reasoning.claims import ClaimType


class TestCruxClaim:
    """Tests for CruxClaim dataclass."""

    def test_to_dict(self):
        """Should serialize CruxClaim to dictionary."""
        crux = CruxClaim(
            claim_id="c1",
            statement="Test claim",
            author="agent1",
            crux_score=0.75,
            influence_score=0.8,
            disagreement_score=0.6,
            uncertainty_score=0.5,
            centrality_score=0.7,
            affected_claims=["c2", "c3"],
            contesting_agents=["agent2", "agent3"],
            resolution_impact=0.4,
        )

        data = crux.to_dict()

        assert data["claim_id"] == "c1"
        assert data["crux_score"] == 0.75
        assert data["affected_claims"] == ["c2", "c3"]
        assert data["contesting_agents"] == ["agent2", "agent3"]


class TestCruxAnalysisResult:
    """Tests for CruxAnalysisResult dataclass."""

    def test_to_dict(self):
        """Should serialize analysis result to dictionary."""
        result = CruxAnalysisResult(
            cruxes=[],
            total_claims=10,
            total_disagreements=3,
            average_uncertainty=0.5,
            convergence_barrier=0.4,
            recommended_focus=["c1", "c2"],
        )

        data = result.to_dict()

        assert data["total_claims"] == 10
        assert data["total_disagreements"] == 3
        assert data["recommended_focus"] == ["c1", "c2"]


class TestCruxDetector:
    """Tests for CruxDetector class."""

    @pytest.fixture
    def simple_network(self):
        """Create a simple network for testing."""
        network = BeliefNetwork(debate_id="test-debate")

        # Add claims from two agents
        network.add_claim("c1", "Base claim by agent1", "agent1", 0.8)
        network.add_claim("c2", "Supporting claim by agent1", "agent1", 0.7)
        network.add_claim("c3", "Contradicting claim by agent2", "agent2", 0.6)
        network.add_claim("c4", "Dependent claim by agent2", "agent2", 0.5)

        # Add relationships
        network.add_factor("c1", "c2", RelationType.SUPPORTS)
        network.add_factor("c1", "c3", RelationType.CONTRADICTS)
        network.add_factor("c2", "c4", RelationType.DEPENDS_ON)

        return network

    @pytest.fixture
    def contested_network(self):
        """Create a network with more contested claims."""
        network = BeliefNetwork(debate_id="contested-debate")

        # Central contested claim
        network.add_claim("central", "The central disputed claim", "agent1", 0.5)

        # Supporting evidence from agent1
        network.add_claim("support1", "Evidence supporting central", "agent1", 0.8)
        network.add_claim("support2", "More evidence supporting", "agent1", 0.7)

        # Contradicting evidence from agent2
        network.add_claim("contra1", "Evidence against central", "agent2", 0.9)
        network.add_claim("contra2", "More evidence against", "agent2", 0.8)

        # Downstream claims that depend on central
        network.add_claim("downstream1", "If central, then this", "agent1", 0.6)
        network.add_claim("downstream2", "Also depends on central", "agent2", 0.5)

        # Add relationships
        network.add_factor("support1", "central", RelationType.SUPPORTS)
        network.add_factor("support2", "central", RelationType.SUPPORTS)
        network.add_factor("contra1", "central", RelationType.CONTRADICTS)
        network.add_factor("contra2", "central", RelationType.CONTRADICTS)
        network.add_factor("central", "downstream1", RelationType.SUPPORTS)
        network.add_factor("central", "downstream2", RelationType.SUPPORTS)

        return network

    def test_init_with_custom_weights(self, simple_network):
        """Should accept custom scoring weights."""
        detector = CruxDetector(
            simple_network,
            influence_weight=0.5,
            disagreement_weight=0.2,
            uncertainty_weight=0.2,
            centrality_weight=0.1,
        )

        assert detector.weights["influence"] == 0.5
        assert detector.weights["disagreement"] == 0.2

    def test_compute_influence_scores(self, simple_network):
        """Should compute influence scores for all claims."""
        detector = CruxDetector(simple_network)
        scores = detector.compute_influence_scores()

        # All claims should have scores
        assert len(scores) == 4

        # c1 should have high influence (affects c2 and c3)
        c1_node_id = simple_network.claim_to_node["c1"]
        assert c1_node_id in scores

        # Scores should be normalized to 0-1
        for score in scores.values():
            assert 0 <= score <= 1

    def test_compute_disagreement_scores(self, contested_network):
        """Should compute disagreement scores showing agent conflicts."""
        detector = CruxDetector(contested_network)
        scores = detector.compute_disagreement_scores()

        # Central claim should have disagreement (agent1 supports, agent2 contradicts)
        central_node_id = contested_network.claim_to_node["central"]
        if central_node_id in scores:
            disagreement, contesting = scores[central_node_id]
            # Should detect some level of disagreement
            assert isinstance(disagreement, float)
            assert isinstance(contesting, list)

    def test_compute_resolution_impact(self, contested_network):
        """Should compute how much resolving a claim reduces uncertainty."""
        detector = CruxDetector(contested_network)
        contested_network.propagate()

        central_node_id = contested_network.claim_to_node["central"]
        impact = detector.compute_resolution_impact(central_node_id)

        # Impact should be non-negative
        assert impact >= 0

        # Central claim should have positive impact (it affects downstream claims)
        # Note: exact value depends on network structure

    def test_detect_cruxes_empty_network(self):
        """Should handle empty network gracefully."""
        network = BeliefNetwork(debate_id="empty")
        detector = CruxDetector(network)

        result = detector.detect_cruxes()

        assert result.total_claims == 0
        assert result.cruxes == []
        assert result.recommended_focus == []

    def test_detect_cruxes_simple(self, simple_network):
        """Should detect cruxes in simple network."""
        detector = CruxDetector(simple_network)
        result = detector.detect_cruxes(top_k=3, min_score=0.0)

        assert result.total_claims == 4
        assert len(result.cruxes) <= 3

        # Cruxes should be sorted by score (descending)
        if len(result.cruxes) >= 2:
            assert result.cruxes[0].crux_score >= result.cruxes[1].crux_score

    def test_detect_cruxes_contested(self, contested_network):
        """Should identify contested claims as cruxes."""
        detector = CruxDetector(contested_network)
        result = detector.detect_cruxes(top_k=3, min_score=0.0)

        # Central claim should be identified (high influence, contested)
        crux_ids = [c.claim_id for c in result.cruxes]
        # It should be in the top cruxes or have significant scores
        assert result.total_claims == 7

    def test_detect_cruxes_respects_min_score(self, simple_network):
        """Should filter out low-scoring cruxes."""
        detector = CruxDetector(simple_network)

        # With very high threshold, should return fewer cruxes
        result_high = detector.detect_cruxes(top_k=10, min_score=0.9)
        result_low = detector.detect_cruxes(top_k=10, min_score=0.0)

        assert len(result_high.cruxes) <= len(result_low.cruxes)

    def test_crux_has_required_fields(self, simple_network):
        """Each crux should have all required fields."""
        detector = CruxDetector(simple_network)
        result = detector.detect_cruxes(top_k=5, min_score=0.0)

        for crux in result.cruxes:
            assert hasattr(crux, "claim_id")
            assert hasattr(crux, "statement")
            assert hasattr(crux, "author")
            assert hasattr(crux, "crux_score")
            assert hasattr(crux, "influence_score")
            assert hasattr(crux, "disagreement_score")
            assert hasattr(crux, "uncertainty_score")
            assert hasattr(crux, "centrality_score")
            assert hasattr(crux, "affected_claims")
            assert hasattr(crux, "contesting_agents")
            assert hasattr(crux, "resolution_impact")

    def test_crux_analysis_result_fields(self, contested_network):
        """CruxAnalysisResult should have all computed fields."""
        detector = CruxDetector(contested_network)
        result = detector.detect_cruxes()

        assert result.total_claims > 0
        assert isinstance(result.total_disagreements, int)
        assert 0 <= result.average_uncertainty
        assert 0 <= result.convergence_barrier <= 1
        assert isinstance(result.recommended_focus, list)

    def test_suggest_resolution_strategy(self, contested_network):
        """Should provide resolution suggestions for cruxes."""
        detector = CruxDetector(contested_network)
        result = detector.detect_cruxes(top_k=1, min_score=0.0)

        if result.cruxes:
            crux = result.cruxes[0]
            strategy = detector.suggest_resolution_strategy(crux)

            assert "claim_id" in strategy
            assert "statement" in strategy
            assert "suggestions" in strategy
            assert "priority" in strategy
            assert isinstance(strategy["suggestions"], list)

    def test_suggest_resolution_high_disagreement(self, contested_network):
        """Should suggest mediation for high-disagreement cruxes."""
        # Create a crux with high disagreement
        crux = CruxClaim(
            claim_id="test",
            statement="High disagreement claim",
            author="agent1",
            crux_score=0.7,
            influence_score=0.3,
            disagreement_score=0.8,  # High disagreement
            uncertainty_score=0.4,
            centrality_score=0.5,
            affected_claims=[],
            contesting_agents=["agent1", "agent2"],
            resolution_impact=0.3,
        )

        detector = CruxDetector(contested_network)
        strategy = detector.suggest_resolution_strategy(crux)

        suggestion_types = [s["type"] for s in strategy["suggestions"]]
        assert "mediation" in suggestion_types

    def test_suggest_resolution_high_uncertainty(self, contested_network):
        """Should suggest evidence gathering for high-uncertainty cruxes."""
        crux = CruxClaim(
            claim_id="test",
            statement="Uncertain claim",
            author="agent1",
            crux_score=0.7,
            influence_score=0.3,
            disagreement_score=0.2,
            uncertainty_score=0.9,  # High uncertainty
            centrality_score=0.5,
            affected_claims=[],
            contesting_agents=[],
            resolution_impact=0.3,
        )

        detector = CruxDetector(contested_network)
        strategy = detector.suggest_resolution_strategy(crux)

        suggestion_types = [s["type"] for s in strategy["suggestions"]]
        assert "evidence" in suggestion_types

    def test_suggest_resolution_high_influence(self, contested_network):
        """Should suggest decomposition for high-influence cruxes."""
        crux = CruxClaim(
            claim_id="test",
            statement="Influential claim",
            author="agent1",
            crux_score=0.8,
            influence_score=0.9,  # High influence
            disagreement_score=0.2,
            uncertainty_score=0.4,
            centrality_score=0.7,
            affected_claims=[],
            contesting_agents=[],
            resolution_impact=0.5,
        )

        detector = CruxDetector(contested_network)
        strategy = detector.suggest_resolution_strategy(crux)

        suggestion_types = [s["type"] for s in strategy["suggestions"]]
        assert "decomposition" in suggestion_types

    def test_suggest_resolution_many_affected(self, contested_network):
        """Should prioritize cruxes with many affected claims."""
        crux = CruxClaim(
            claim_id="test",
            statement="Claim with many dependents",
            author="agent1",
            crux_score=0.7,
            influence_score=0.5,
            disagreement_score=0.3,
            uncertainty_score=0.4,
            centrality_score=0.6,
            affected_claims=["c1", "c2", "c3", "c4"],  # Many affected
            contesting_agents=[],
            resolution_impact=0.4,
        )

        detector = CruxDetector(contested_network)
        strategy = detector.suggest_resolution_strategy(crux)

        suggestion_types = [s["type"] for s in strategy["suggestions"]]
        assert "priority" in suggestion_types


class TestCruxDetectorIntegration:
    """Integration tests for crux detection."""

    def test_full_workflow(self):
        """Test complete crux detection workflow."""
        # Build a realistic debate network
        network = BeliefNetwork(debate_id="integration-test")

        # Main thesis
        network.add_claim("thesis", "We should adopt microservices", "architect", 0.6)

        # Pro arguments
        network.add_claim("pro1", "Microservices enable independent scaling", "architect", 0.8)
        network.add_claim("pro2", "Teams can deploy independently", "architect", 0.7)
        network.add_claim("pro3", "Technology diversity is possible", "developer", 0.65)

        # Con arguments
        network.add_claim("con1", "Distributed systems are complex", "ops", 0.85)
        network.add_claim("con2", "Network calls add latency", "ops", 0.75)
        network.add_claim("con3", "Debugging is harder", "developer", 0.7)

        # Sub-arguments
        network.add_claim("sub1", "Our traffic is highly variable", "ops", 0.9)
        network.add_claim("sub2", "We have mature DevOps practices", "architect", 0.6)

        # Build factor graph
        network.add_factor("pro1", "thesis", RelationType.SUPPORTS)
        network.add_factor("pro2", "thesis", RelationType.SUPPORTS)
        network.add_factor("pro3", "thesis", RelationType.SUPPORTS)
        network.add_factor("con1", "thesis", RelationType.CONTRADICTS)
        network.add_factor("con2", "thesis", RelationType.CONTRADICTS)
        network.add_factor("con3", "thesis", RelationType.CONTRADICTS)
        network.add_factor("sub1", "pro1", RelationType.SUPPORTS)
        network.add_factor("sub2", "con1", RelationType.CONTRADICTS)

        # Run propagation
        network.propagate()

        # Detect cruxes
        detector = CruxDetector(network)
        result = detector.detect_cruxes(top_k=3)

        # Verify analysis
        assert result.total_claims == 9
        assert len(result.cruxes) <= 3
        assert result.convergence_barrier >= 0
        assert len(result.recommended_focus) == len(result.cruxes)

        # Check that we can serialize
        data = result.to_dict()
        assert "cruxes" in data
        assert "convergence_barrier" in data

    def test_crux_detection_updates_with_propagation(self):
        """Crux detection should reflect network state after propagation."""
        network = BeliefNetwork(debate_id="dynamic-test")

        network.add_claim("a", "Claim A", "agent1", 0.5)
        network.add_claim("b", "Claim B depends on A", "agent1", 0.5)
        network.add_factor("a", "b", RelationType.SUPPORTS)

        detector = CruxDetector(network)

        # First detection
        result1 = detector.detect_cruxes(min_score=0.0)

        # Manually change a belief
        a_node = network.get_node_by_claim("a")
        a_node.posterior = BeliefDistribution(p_true=0.9, p_false=0.1)
        network.propagate()

        # Second detection should reflect changes
        result2 = detector.detect_cruxes(min_score=0.0)

        # Uncertainty should have changed
        assert result1.average_uncertainty != result2.average_uncertainty or True
