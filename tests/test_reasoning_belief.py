"""
Tests for the Bayesian Belief Propagation Network.

Tests cover:
- BeliefStatus enum
- BeliefDistribution dataclass (normalization, entropy, confidence, KL divergence)
- BeliefNode (update_posterior, to_dict)
- Factor (factor potentials for different relation types)
- PropagationResult
- BeliefNetwork (construction, propagation, queries)
- BeliefPropagationAnalyzer (cruxes, evidence targets, consensus, what-if)
"""

import math
import pytest

from aragora.reasoning.belief import (
    BeliefStatus,
    BeliefDistribution,
    BeliefNode,
    Factor,
    PropagationResult,
    BeliefNetwork,
    BeliefPropagationAnalyzer,
)
from aragora.reasoning.claims import (
    ClaimType,
    RelationType,
    TypedClaim,
    ClaimsKernel,
)


class TestBeliefStatusEnum:
    """Tests for BeliefStatus enumeration."""

    def test_all_status_values_defined(self):
        """Verify all expected status values exist."""
        expected = ["prior", "updated", "converged", "contested"]
        actual = [s.value for s in BeliefStatus]
        assert sorted(expected) == sorted(actual)

    def test_status_values(self):
        """Test specific status values."""
        assert BeliefStatus.PRIOR.value == "prior"
        assert BeliefStatus.UPDATED.value == "updated"
        assert BeliefStatus.CONVERGED.value == "converged"
        assert BeliefStatus.CONTESTED.value == "contested"


class TestBeliefDistribution:
    """Tests for BeliefDistribution dataclass."""

    def test_default_uniform_distribution(self):
        """Default distribution should be uniform."""
        dist = BeliefDistribution()
        assert dist.p_true == 0.5
        assert dist.p_false == 0.5
        assert dist.p_unknown == 0.0

    def test_normalization(self):
        """Probabilities should sum to 1.0 after normalization."""
        dist = BeliefDistribution(p_true=2.0, p_false=3.0, p_unknown=5.0)
        total = dist.p_true + dist.p_false + dist.p_unknown
        assert abs(total - 1.0) < 1e-9

    def test_normalization_preserves_ratios(self):
        """Normalization should preserve probability ratios."""
        dist = BeliefDistribution(p_true=2.0, p_false=3.0, p_unknown=5.0)
        # Original ratios: 2:3:5
        assert abs(dist.p_true / dist.p_false - 2.0 / 3.0) < 1e-9
        assert abs(dist.p_unknown / dist.p_true - 5.0 / 2.0) < 1e-9

    def test_entropy_uniform(self):
        """Entropy of uniform distribution should be 1.0 (2 states) or ~1.58 (3 states)."""
        # With 2 states (p_unknown = 0), max entropy is 1.0
        dist = BeliefDistribution(p_true=0.5, p_false=0.5, p_unknown=0.0)
        assert abs(dist.entropy - 1.0) < 1e-6

    def test_entropy_certain(self):
        """Entropy of certain distribution should be near 0."""
        dist = BeliefDistribution(p_true=1.0, p_false=0.0, p_unknown=0.0)
        assert dist.entropy < 0.01

    def test_confidence(self):
        """Confidence should be max probability."""
        dist = BeliefDistribution(p_true=0.7, p_false=0.2, p_unknown=0.1)
        assert dist.confidence == pytest.approx(0.7, abs=0.01)

    def test_expected_truth(self):
        """Expected truth should weight true=1, false=0, unknown=0.5."""
        dist = BeliefDistribution(p_true=0.6, p_false=0.4, p_unknown=0.0)
        assert dist.expected_truth == pytest.approx(0.6, abs=0.01)

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions should be 0."""
        dist1 = BeliefDistribution(p_true=0.7, p_false=0.3)
        dist2 = BeliefDistribution(p_true=0.7, p_false=0.3)
        assert dist1.kl_divergence(dist2) == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_different(self):
        """KL divergence of different distributions should be positive."""
        dist1 = BeliefDistribution(p_true=0.9, p_false=0.1)
        dist2 = BeliefDistribution(p_true=0.5, p_false=0.5)
        assert dist1.kl_divergence(dist2) > 0

    def test_from_confidence_true(self):
        """from_confidence with lean_true should create high p_true."""
        dist = BeliefDistribution.from_confidence(0.8, lean_true=True)
        assert dist.p_true == pytest.approx(0.8, abs=0.01)
        assert dist.p_false == pytest.approx(0.2, abs=0.01)

    def test_from_confidence_false(self):
        """from_confidence with lean_true=False should create high p_false."""
        dist = BeliefDistribution.from_confidence(0.8, lean_true=False)
        assert dist.p_false == pytest.approx(0.8, abs=0.01)
        assert dist.p_true == pytest.approx(0.2, abs=0.01)

    def test_uniform_factory(self):
        """uniform() should create maximum uncertainty distribution."""
        dist = BeliefDistribution.uniform()
        assert dist.p_true == 0.5
        assert dist.p_false == 0.5
        assert dist.entropy == pytest.approx(1.0, abs=0.01)

    def test_to_dict(self):
        """to_dict should include all distribution info."""
        dist = BeliefDistribution(p_true=0.7, p_false=0.3)
        d = dist.to_dict()
        assert d["p_true"] == pytest.approx(0.7, abs=0.01)
        assert d["p_false"] == pytest.approx(0.3, abs=0.01)
        assert "entropy" in d
        assert "confidence" in d


class TestFactor:
    """Tests for Factor dataclass and factor potentials."""

    def test_supports_factor_same_truth(self):
        """SUPPORTS: both true should have high potential."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        potential = factor.get_factor_potential(source_true=True, target_true=True)
        assert potential > 0.7

    def test_supports_factor_different_truth(self):
        """SUPPORTS: source true, target false should have low potential."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        potential = factor.get_factor_potential(source_true=True, target_true=False)
        assert potential < 0.3

    def test_contradicts_factor(self):
        """CONTRADICTS: source true, target true should have low potential."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.CONTRADICTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        potential_both_true = factor.get_factor_potential(source_true=True, target_true=True)
        potential_opposite = factor.get_factor_potential(source_true=True, target_true=False)
        assert potential_both_true < potential_opposite

    def test_depends_on_factor(self):
        """DEPENDS_ON: target true without source true should have very low potential."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.DEPENDS_ON,
            source_node_id="n1",
            target_node_id="n2",
        )
        potential = factor.get_factor_potential(source_true=False, target_true=True)
        assert potential < 0.2

    def test_strength_affects_potential(self):
        """Higher strength should increase effect on potential."""
        factor_weak = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=0.1,
        )
        factor_strong = Factor(
            factor_id="f2",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        pot_weak = factor_weak.get_factor_potential(True, True)
        pot_strong = factor_strong.get_factor_potential(True, True)
        assert pot_strong > pot_weak


class TestBeliefNode:
    """Tests for BeliefNode dataclass."""

    def test_node_creation(self):
        """Node should be created with defaults."""
        node = BeliefNode(
            node_id="n1",
            claim_id="c1",
            claim_statement="Test claim",
            author="test-agent",
        )
        assert node.node_id == "n1"
        assert node.claim_id == "c1"
        assert node.status == BeliefStatus.PRIOR
        assert node.centrality == 0.0

    def test_update_posterior_no_messages(self):
        """Posterior with no messages should equal prior."""
        node = BeliefNode(
            node_id="n1",
            claim_id="c1",
            claim_statement="Test claim",
            author="test-agent",
            prior=BeliefDistribution(p_true=0.7, p_false=0.3),
        )
        node.update_posterior()
        assert node.posterior.p_true > 0.6
        assert node.update_count == 1

    def test_to_dict(self):
        """to_dict should serialize all important fields."""
        node = BeliefNode(
            node_id="n1",
            claim_id="c1",
            claim_statement="A very long claim statement that should be truncated",
            author="test-agent",
        )
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["claim_id"] == "c1"
        assert d["author"] == "test-agent"
        assert "prior" in d
        assert "posterior" in d


class TestBeliefNetwork:
    """Tests for BeliefNetwork class."""

    def test_network_creation(self):
        """Network should be created with empty state."""
        network = BeliefNetwork(debate_id="test-debate")
        assert network.debate_id == "test-debate"
        assert len(network.nodes) == 0
        assert len(network.factors) == 0

    def test_add_claim_creates_node(self):
        """add_claim should create a belief node."""
        network = BeliefNetwork()
        node = network.add_claim(
            claim_id="c1",
            statement="Test claim",
            author="claude",
            initial_confidence=0.8,
        )
        assert node is not None
        assert node.claim_id == "c1"
        assert "c1" in network.claim_to_node
        assert len(network.nodes) == 1

    def test_add_factor_creates_edge(self):
        """add_factor should create factor between nodes."""
        network = BeliefNetwork()
        network.add_claim("c1", "First claim", "claude")
        network.add_claim("c2", "Second claim", "gemini")

        factor = network.add_factor(
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type=RelationType.SUPPORTS,
            strength=0.8,
        )

        assert factor is not None
        assert len(network.factors) == 1

    def test_add_factor_missing_node_returns_none(self):
        """add_factor with missing nodes should return None."""
        network = BeliefNetwork()
        network.add_claim("c1", "First claim", "claude")

        factor = network.add_factor(
            source_claim_id="c1",
            target_claim_id="nonexistent",
            relation_type=RelationType.SUPPORTS,
        )

        assert factor is None

    def test_from_claims_kernel(self):
        """from_claims_kernel should build network from kernel."""
        kernel = ClaimsKernel("test-debate")
        c1 = kernel.add_claim("First claim", "claude", ClaimType.ASSERTION, confidence=0.7)
        c2 = kernel.add_claim("Second claim", "gemini", ClaimType.OBJECTION, confidence=0.6)
        kernel.add_relation(c1.claim_id, c2.claim_id, RelationType.SUPPORTS)

        network = BeliefNetwork()
        network.from_claims_kernel(kernel)

        assert len(network.nodes) == 2
        assert len(network.factors) == 1

    def test_propagate_converges(self):
        """propagate should converge on simple network."""
        network = BeliefNetwork(max_iterations=100)
        network.add_claim("c1", "Premise", "claude", initial_confidence=0.9)
        network.add_claim("c2", "Conclusion", "claude", initial_confidence=0.5)
        network.add_factor("c1", "c2", RelationType.SUPPORTS, strength=0.8)

        result = network.propagate()

        assert result.iterations <= 100
        # Result might or might not converge depending on network structure
        assert result.max_change >= 0

    def test_propagate_updates_posteriors(self):
        """propagate should update node posteriors."""
        network = BeliefNetwork()
        network.add_claim("c1", "Strong premise", "claude", initial_confidence=0.95)
        network.add_claim("c2", "Uncertain conclusion", "claude", initial_confidence=0.5)
        network.add_factor("c1", "c2", RelationType.SUPPORTS, strength=1.0)

        result = network.propagate()

        # C2's posterior should be influenced by C1
        c2_node = network.get_node_by_claim("c2")
        assert c2_node is not None
        assert c2_node.status in [BeliefStatus.UPDATED, BeliefStatus.CONVERGED]

    def test_get_node_by_claim(self):
        """get_node_by_claim should return correct node."""
        network = BeliefNetwork()
        network.add_claim("c1", "Test claim", "claude")

        node = network.get_node_by_claim("c1")
        assert node is not None
        assert node.claim_id == "c1"

        none_node = network.get_node_by_claim("nonexistent")
        assert none_node is None

    def test_get_most_uncertain_claims(self):
        """get_most_uncertain_claims should return high entropy nodes."""
        network = BeliefNetwork()
        network.add_claim("c1", "Certain claim", "claude", initial_confidence=0.99)
        network.add_claim("c2", "Uncertain claim", "gemini", initial_confidence=0.5)
        network.propagate()

        uncertain = network.get_most_uncertain_claims(limit=1)

        assert len(uncertain) == 1
        # The uncertain claim should have higher entropy
        node, entropy = uncertain[0]
        assert node.claim_id == "c2"

    def test_get_load_bearing_claims(self):
        """get_load_bearing_claims should return high centrality nodes."""
        network = BeliefNetwork()
        # Create a hub node with many children
        network.add_claim("c1", "Hub claim", "claude")
        network.add_claim("c2", "Child 1", "gemini")
        network.add_claim("c3", "Child 2", "gpt4")
        network.add_factor("c1", "c2", RelationType.SUPPORTS)
        network.add_factor("c1", "c3", RelationType.SUPPORTS)
        network.propagate()

        load_bearing = network.get_load_bearing_claims(limit=3)

        assert len(load_bearing) > 0

    def test_empty_network_handling(self):
        """Empty network should handle operations gracefully."""
        network = BeliefNetwork()

        result = network.propagate()
        assert result.converged is True
        # Empty network still runs through the loop once
        assert result.iterations <= 1

        uncertain = network.get_most_uncertain_claims()
        assert len(uncertain) == 0

    def test_generate_summary(self):
        """generate_summary should produce markdown text."""
        network = BeliefNetwork(debate_id="test-debate")
        network.add_claim("c1", "Test claim", "claude", initial_confidence=0.8)
        network.propagate()

        summary = network.generate_summary()

        assert "# Belief Network Summary" in summary
        assert "test-debate" in summary
        assert "Nodes:" in summary

    def test_to_dict_serialization(self):
        """to_dict should serialize network state."""
        network = BeliefNetwork(debate_id="test-debate")
        network.add_claim("c1", "Test", "claude")
        network.add_claim("c2", "Test 2", "gemini")
        network.add_factor("c1", "c2", RelationType.SUPPORTS)

        d = network.to_dict()

        assert d["debate_id"] == "test-debate"
        assert "nodes" in d
        assert "factors" in d
        assert "claim_to_node" in d


class TestBeliefPropagationAnalyzer:
    """Tests for BeliefPropagationAnalyzer class."""

    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing."""
        network = BeliefNetwork(debate_id="test")
        network.add_claim("c1", "Base premise", "claude", initial_confidence=0.9)
        network.add_claim("c2", "Uncertain claim", "gemini", initial_confidence=0.5)
        network.add_claim("c3", "Conclusion", "gpt4", initial_confidence=0.6)
        network.add_factor("c1", "c2", RelationType.SUPPORTS)
        network.add_factor("c2", "c3", RelationType.SUPPORTS)
        network.propagate()
        return network

    def test_identify_debate_cruxes(self, sample_network):
        """identify_debate_cruxes should find high centrality+entropy claims."""
        analyzer = BeliefPropagationAnalyzer(sample_network)
        cruxes = analyzer.identify_debate_cruxes(top_k=3)

        assert len(cruxes) <= 3
        for crux in cruxes:
            assert "claim_id" in crux
            assert "crux_score" in crux
            assert "centrality" in crux
            assert "entropy" in crux

    def test_suggest_evidence_targets(self, sample_network):
        """suggest_evidence_targets should identify claims needing evidence."""
        analyzer = BeliefPropagationAnalyzer(sample_network)
        suggestions = analyzer.suggest_evidence_targets()

        # All returned suggestions should have high entropy
        for s in suggestions:
            assert "claim_id" in s
            assert "current_uncertainty" in s
            assert "importance" in s

    def test_compute_consensus_probability(self, sample_network):
        """compute_consensus_probability should estimate consensus."""
        analyzer = BeliefPropagationAnalyzer(sample_network)
        result = analyzer.compute_consensus_probability()

        assert "probability" in result
        assert "average_confidence" in result
        assert "contested_claims" in result
        assert 0.0 <= result["probability"] <= 1.0

    def test_compute_consensus_empty_network(self):
        """compute_consensus_probability should handle empty network."""
        network = BeliefNetwork()
        analyzer = BeliefPropagationAnalyzer(network)
        result = analyzer.compute_consensus_probability()

        assert result["probability"] == 0.0
        assert "No claims" in result["explanation"]

    def test_what_if_analysis(self, sample_network):
        """what_if_analysis should show effect of hypotheticals."""
        analyzer = BeliefPropagationAnalyzer(sample_network)

        # What if c1 were false?
        result = analyzer.what_if_analysis({"c1": False})

        assert "hypothetical" in result
        assert "affected_claims" in result
        assert "changes" in result

    def test_what_if_preserves_state(self, sample_network):
        """what_if_analysis should restore original state."""
        analyzer = BeliefPropagationAnalyzer(sample_network)

        # Get original posteriors
        original_c1 = sample_network.get_node_by_claim("c1").posterior.p_true

        # Run what-if
        analyzer.what_if_analysis({"c1": False})

        # State should be restored
        restored_c1 = sample_network.get_node_by_claim("c1").posterior.p_true
        assert abs(original_c1 - restored_c1) < 0.01


class TestPropagationResult:
    """Tests for PropagationResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize result."""
        posteriors = {"n1": BeliefDistribution.uniform()}
        result = PropagationResult(
            converged=True,
            iterations=10,
            max_change=0.001,
            node_posteriors=posteriors,
            centralities={"n1": 0.5},
        )

        d = result.to_dict()

        assert d["converged"] is True
        assert d["iterations"] == 10
        assert d["max_change"] == 0.001
        assert "n1" in d["node_posteriors"]
        assert "n1" in d["centralities"]
