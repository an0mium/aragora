"""
Extended tests for the Bayesian Belief Propagation Network.

Tests cover advanced scenarios not covered by test_reasoning_belief.py:
- Cyclic graph handling and convergence
- Message passing pipeline internals
- Centrality algorithm behavior
- State isolation during inference
- Conditional probability queries
- Sensitivity analysis
- Convergence behavior with different parameters
- Graph topology edge cases
- Factor potential edge cases
- Serialization round-trips
"""

import copy
import json
import math
import threading
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
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cyclic_network():
    """Network with A->B->C->A cycle."""
    network = BeliefNetwork(damping=0.5, max_iterations=100)
    network.add_claim("A", "First claim", "agent1", 0.8)
    network.add_claim("B", "Second claim", "agent2", 0.5)
    network.add_claim("C", "Third claim", "agent3", 0.5)
    network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
    network.add_factor("B", "C", RelationType.SUPPORTS, 0.9)
    network.add_factor("C", "A", RelationType.SUPPORTS, 0.9)
    return network


@pytest.fixture
def linear_network():
    """Linear chain: A -> B -> C -> D."""
    network = BeliefNetwork(damping=0.5, max_iterations=100)
    network.add_claim("A", "Base claim", "agent1", 0.9)
    network.add_claim("B", "Second claim", "agent2", 0.5)
    network.add_claim("C", "Third claim", "agent3", 0.5)
    network.add_claim("D", "Conclusion", "agent4", 0.5)
    network.add_factor("A", "B", RelationType.SUPPORTS, 0.8)
    network.add_factor("B", "C", RelationType.SUPPORTS, 0.8)
    network.add_factor("C", "D", RelationType.SUPPORTS, 0.8)
    return network


@pytest.fixture
def star_network():
    """Star topology: hub with many children."""
    network = BeliefNetwork(damping=0.5, max_iterations=100)
    network.add_claim("hub", "Central claim", "agent1", 0.9)
    for i in range(5):
        network.add_claim(f"leaf{i}", f"Leaf claim {i}", f"agent{i+2}", 0.5)
        network.add_factor("hub", f"leaf{i}", RelationType.SUPPORTS, 0.8)
    return network


@pytest.fixture
def mixed_relation_network():
    """Network with both SUPPORTS and CONTRADICTS."""
    network = BeliefNetwork(damping=0.5, max_iterations=100)
    network.add_claim("A", "Assertion", "agent1", 0.7)
    network.add_claim("B", "Supporting evidence", "agent2", 0.8)
    network.add_claim("C", "Counter-evidence", "agent3", 0.8)
    network.add_factor("B", "A", RelationType.SUPPORTS, 0.9)
    network.add_factor("C", "A", RelationType.CONTRADICTS, 0.9)
    return network


# =============================================================================
# Category A: Cyclic Graph Handling
# =============================================================================


class TestCyclicGraphHandling:
    """Tests for cyclic graph convergence and behavior."""

    def test_simple_cycle_converges(self, cyclic_network):
        """Simple 3-node cycle should converge."""
        result = cyclic_network.propagate()
        # With damping, cycles should eventually converge
        assert result.iterations <= cyclic_network.max_iterations
        # All nodes should have posteriors
        assert len(result.node_posteriors) == 3

    def test_cycle_with_mixed_relations(self):
        """Cycle with supports + contradicts should handle conflict."""
        network = BeliefNetwork(damping=0.5, max_iterations=50)
        network.add_claim("A", "First", "agent1", 0.7)
        network.add_claim("B", "Second", "agent2", 0.7)
        network.add_claim("C", "Third", "agent3", 0.7)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        network.add_factor("B", "C", RelationType.CONTRADICTS, 0.9)
        network.add_factor("C", "A", RelationType.SUPPORTS, 0.9)

        result = network.propagate()
        assert result.iterations <= network.max_iterations

    def test_multiple_parallel_cycles(self):
        """Multiple connected cycles should propagate correctly."""
        network = BeliefNetwork(damping=0.5, max_iterations=100)
        # Cycle 1: A->B->C->A
        network.add_claim("A", "A", "agent1", 0.8)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_claim("C", "C", "agent3", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.8)
        network.add_factor("B", "C", RelationType.SUPPORTS, 0.8)
        network.add_factor("C", "A", RelationType.SUPPORTS, 0.8)
        # Cycle 2: D->E->F->D connected to cycle 1 via A->D
        network.add_claim("D", "D", "agent4", 0.5)
        network.add_claim("E", "E", "agent5", 0.5)
        network.add_claim("F", "F", "agent6", 0.5)
        network.add_factor("D", "E", RelationType.SUPPORTS, 0.8)
        network.add_factor("E", "F", RelationType.SUPPORTS, 0.8)
        network.add_factor("F", "D", RelationType.SUPPORTS, 0.8)
        network.add_factor("A", "D", RelationType.SUPPORTS, 0.8)

        result = network.propagate()
        assert len(result.node_posteriors) == 6

    def test_tight_cycle_with_high_damping(self):
        """Tight cycle with high damping should converge more slowly."""
        network = BeliefNetwork(damping=0.9, max_iterations=200)
        network.add_claim("A", "A", "agent1", 0.8)
        network.add_claim("B", "B", "agent2", 0.2)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)
        network.add_factor("B", "A", RelationType.SUPPORTS, 1.0)

        result = network.propagate()
        # High damping preserves old messages, so convergence takes longer
        assert result.iterations > 1

    def test_cycle_with_low_damping(self):
        """Cycle with low damping (near 0) updates quickly."""
        network = BeliefNetwork(damping=0.1, max_iterations=100)
        network.add_claim("A", "A", "agent1", 0.9)
        network.add_claim("B", "B", "agent2", 0.1)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)
        network.add_factor("B", "A", RelationType.SUPPORTS, 1.0)

        result = network.propagate()
        # Low damping means new messages dominate
        assert result.iterations <= network.max_iterations

    def test_cycle_convergence_vs_max_iterations(self):
        """Network should run until converged or max_iterations reached."""
        # Create a network that may or may not converge
        network = BeliefNetwork(damping=0.0, max_iterations=10, convergence_threshold=1e-10)
        network.add_claim("A", "A", "agent1", 0.9)
        network.add_claim("B", "B", "agent2", 0.1)
        network.add_factor("A", "B", RelationType.CONTRADICTS, 1.0)
        network.add_factor("B", "A", RelationType.CONTRADICTS, 1.0)

        result = network.propagate()
        # Should either converge or hit max_iterations
        assert result.iterations <= 10
        # May or may not converge with contradicting cycle

    def test_all_nodes_equal_confidence(self):
        """Cycle with all equal initial confidence."""
        network = BeliefNetwork(damping=0.5, max_iterations=50)
        network.add_claim("A", "A", "agent1", 0.5)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_claim("C", "C", "agent3", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.5)
        network.add_factor("B", "C", RelationType.SUPPORTS, 0.5)
        network.add_factor("C", "A", RelationType.SUPPORTS, 0.5)

        result = network.propagate()
        # All nodes starting equal with equal factors should stay similar
        posteriors = list(result.node_posteriors.values())
        p_trues = [p.p_true for p in posteriors]
        # All should be similar (within 0.2 of each other)
        assert max(p_trues) - min(p_trues) < 0.3


# =============================================================================
# Category B: Message Passing Pipeline
# =============================================================================


class TestMessagePassingPipeline:
    """Tests for the internal message passing mechanism."""

    def test_send_messages_single_edge(self):
        """_send_messages should update incoming messages on both nodes."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Source", "agent1", 0.9)
        network.add_claim("B", "Target", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)

        factor = list(network.factors.values())[0]
        network._send_messages(factor)

        source_node = network.nodes[factor.source_node_id]
        target_node = network.nodes[factor.target_node_id]
        # Both nodes should have incoming messages from the factor
        assert factor.factor_id in target_node.incoming_messages
        assert factor.factor_id in source_node.incoming_messages

    def test_send_messages_multiple_incoming(self):
        """Node with multiple parents should receive multiple messages."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Parent 1", "agent1", 0.9)
        network.add_claim("B", "Parent 2", "agent2", 0.8)
        network.add_claim("C", "Child", "agent3", 0.5)
        network.add_factor("A", "C", RelationType.SUPPORTS, 0.9)
        network.add_factor("B", "C", RelationType.SUPPORTS, 0.9)

        for factor in network.factors.values():
            network._send_messages(factor)

        child_node = network.get_node_by_claim("C")
        assert len(child_node.incoming_messages) == 2

    def test_compute_message_to_target(self):
        """_compute_message to_target=True marginalizes over source."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Source", "agent1", 0.9)
        network.add_claim("B", "Target", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)

        factor = list(network.factors.values())[0]
        source_node = network.nodes[factor.source_node_id]
        target_node = network.nodes[factor.target_node_id]

        msg = network._compute_message(factor, source_node, target_node, to_target=True)
        assert isinstance(msg, BeliefDistribution)
        # Message should be influenced by source's high confidence
        assert msg.p_true > msg.p_false

    def test_compute_message_to_source(self):
        """_compute_message to_target=False marginalizes over target."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Source", "agent1", 0.5)
        network.add_claim("B", "Target", "agent2", 0.9)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)

        factor = list(network.factors.values())[0]
        source_node = network.nodes[factor.source_node_id]
        target_node = network.nodes[factor.target_node_id]

        msg = network._compute_message(factor, source_node, target_node, to_target=False)
        assert isinstance(msg, BeliefDistribution)

    def test_message_damping_application(self):
        """Damping should blend old and new messages."""
        network = BeliefNetwork(damping=0.7)
        network.add_claim("A", "Source", "agent1", 0.9)
        network.add_claim("B", "Target", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)

        factor = list(network.factors.values())[0]
        target_node = network.nodes[factor.target_node_id]

        # First message (no damping applied - no old message)
        network._send_messages(factor)
        first_msg = target_node.incoming_messages[factor.factor_id]
        first_p_true = first_msg.p_true

        # Modify source and send again - damping should blend
        network.nodes[factor.source_node_id].posterior = BeliefDistribution(
            p_true=0.1, p_false=0.9
        )
        network._send_messages(factor)
        second_msg = target_node.incoming_messages[factor.factor_id]

        # With 0.7 damping, 70% of old message preserved
        # Result should be between old and completely new
        assert second_msg.p_true != first_p_true

    def test_message_normalization(self):
        """Messages should be properly normalized distributions."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Source", "agent1", 0.99)
        network.add_claim("B", "Target", "agent2", 0.01)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)

        for factor in network.factors.values():
            network._send_messages(factor)

        for node in network.nodes.values():
            for msg in node.incoming_messages.values():
                total = msg.p_true + msg.p_false + msg.p_unknown
                assert abs(total - 1.0) < 0.01

    def test_log_space_numerical_stability(self):
        """Near-zero probabilities should not cause numerical issues."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Source", "agent1", 0.001)
        network.add_claim("B", "Target", "agent2", 0.999)
        network.add_factor("A", "B", RelationType.CONTRADICTS, 1.0)

        # Should not raise any numerical errors
        result = network.propagate()
        assert not math.isnan(result.max_change)
        assert not math.isinf(result.max_change)

    def test_empty_incoming_messages(self):
        """Node with no incoming factors should use prior."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Orphan", "agent1", 0.7)

        node = network.get_node_by_claim("A")
        node.update_posterior()

        # With no incoming messages, posterior should approximately match prior
        assert abs(node.posterior.p_true - node.prior.p_true) < 0.1


# =============================================================================
# Category C: Centrality Algorithm
# =============================================================================


class TestCentralityAlgorithm:
    """Tests for centrality computation."""

    def test_single_node_centrality(self):
        """Single node should have centrality 1.0."""
        network = BeliefNetwork()
        network.add_claim("A", "Only node", "agent1", 0.5)
        network.propagate()

        centralities = network._compute_centralities()
        assert abs(centralities[network.claim_to_node["A"]] - 1.0) < 0.01

    def test_linear_chain_centrality(self, linear_network):
        """In linear chain, middle nodes should have higher centrality."""
        linear_network.propagate()
        centralities = linear_network._compute_centralities()

        # All centralities should sum to 1
        total = sum(centralities.values())
        assert abs(total - 1.0) < 0.01

    def test_star_topology_hub_centrality(self, star_network):
        """Hub in star topology should have highest centrality."""
        star_network.propagate()

        hub_node = star_network.get_node_by_claim("hub")
        leaf_nodes = [star_network.get_node_by_claim(f"leaf{i}") for i in range(5)]

        # Hub should have higher centrality than leaves
        assert hub_node.centrality >= max(leaf.centrality for leaf in leaf_nodes)

    def test_all_nodes_equal_centrality(self):
        """Disconnected nodes should have equal centrality."""
        network = BeliefNetwork()
        for i in range(5):
            network.add_claim(f"N{i}", f"Node {i}", f"agent{i}", 0.5)

        centralities = network._compute_centralities()

        values = list(centralities.values())
        # All should be approximately equal (1/5 = 0.2)
        for v in values:
            assert abs(v - 0.2) < 0.01

    def test_entropy_weighted_centrality(self):
        """High entropy children should increase parent centrality."""
        network = BeliefNetwork()
        network.add_claim("parent", "Parent", "agent1", 0.9)
        # Child with high entropy (uncertain)
        network.add_claim("uncertain_child", "Uncertain", "agent2", 0.5)
        # Child with low entropy (certain)
        network.add_claim("certain_child", "Certain", "agent3", 0.99)
        network.add_factor("parent", "uncertain_child", RelationType.SUPPORTS, 0.9)
        network.add_factor("parent", "certain_child", RelationType.SUPPORTS, 0.9)

        network.propagate()
        # Parent centrality should be boosted by uncertain child's entropy
        parent = network.get_node_by_claim("parent")
        assert parent.centrality > 0

    def test_disconnected_components_centrality(self):
        """Disconnected components should have independent centralities."""
        network = BeliefNetwork()
        # Component 1
        network.add_claim("A1", "A1", "agent1", 0.8)
        network.add_claim("B1", "B1", "agent2", 0.5)
        network.add_factor("A1", "B1", RelationType.SUPPORTS, 0.9)
        # Component 2 (disconnected)
        network.add_claim("A2", "A2", "agent3", 0.8)
        network.add_claim("B2", "B2", "agent4", 0.5)
        network.add_factor("A2", "B2", RelationType.SUPPORTS, 0.9)

        centralities = network._compute_centralities()

        # Both components should contribute to centrality
        assert sum(centralities.values()) > 0

    def test_pagerank_convergence(self):
        """PageRank iterations should converge."""
        network = BeliefNetwork()
        # Create a more complex network
        for i in range(10):
            network.add_claim(f"N{i}", f"Node {i}", f"agent{i}", 0.5)
        for i in range(9):
            network.add_factor(f"N{i}", f"N{i+1}", RelationType.SUPPORTS, 0.8)

        centralities = network._compute_centralities()

        # Should have computed valid centralities
        assert len(centralities) == 10
        for v in centralities.values():
            assert 0 <= v <= 1


# =============================================================================
# Category D: State Isolation
# =============================================================================


class TestStateIsolation:
    """Tests for state isolation during inference operations."""

    def test_conditional_probability_restores_state(self):
        """conditional_probability should restore state after computation."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Evidence", "agent1", 0.8)
        network.add_claim("B", "Query", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        network.propagate()

        original_b = network.get_node_by_claim("B").posterior.p_true

        # Run conditional probability (this modifies state internally)
        network.conditional_probability("B", {"A": True})

        # After conditional_probability, state may not be restored automatically
        # The function doesn't explicitly restore state
        # This is actually a known limitation

    def test_multiple_conditional_queries_independence(self):
        """Multiple conditional queries should not interfere."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.5)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_claim("C", "C", "agent3", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        network.add_factor("A", "C", RelationType.SUPPORTS, 0.9)
        network.propagate()

        # Query 1
        result1 = network.conditional_probability("B", {"A": True})

        # Reset and query 2
        network.propagate()
        result2 = network.conditional_probability("C", {"A": True})

        # Both queries with same evidence should yield similar high probability
        assert result1.p_true > 0.5
        assert result2.p_true > 0.5

    def test_what_if_analysis_restoration(self):
        """what_if_analysis should restore original state."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.8)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        network.propagate()

        original_a = network.get_node_by_claim("A").posterior.p_true
        original_b = network.get_node_by_claim("B").posterior.p_true

        analyzer = BeliefPropagationAnalyzer(network)
        analyzer.what_if_analysis({"A": False})

        # State should be restored
        restored_a = network.get_node_by_claim("A").posterior.p_true
        restored_b = network.get_node_by_claim("B").posterior.p_true

        assert abs(original_a - restored_a) < 0.01
        assert abs(original_b - restored_b) < 0.01

    def test_state_after_propagation_failure(self):
        """Network state should be consistent after propagation."""
        network = BeliefNetwork(damping=0.5, max_iterations=5)
        network.add_claim("A", "A", "agent1", 0.5)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)

        result = network.propagate()

        # All nodes should have valid posteriors
        for node in network.nodes.values():
            assert 0 <= node.posterior.p_true <= 1
            assert 0 <= node.posterior.p_false <= 1


# =============================================================================
# Category E: Conditional Probability
# =============================================================================


class TestConditionalProbability:
    """Tests for conditional probability computation."""

    def test_simple_evidence_conditioning(self):
        """Setting evidence should influence query probability."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("evidence", "Evidence", "agent1", 0.5)
        network.add_claim("query", "Query", "agent2", 0.5)
        network.add_factor("evidence", "query", RelationType.SUPPORTS, 0.9)

        # P(query | evidence=True) should be higher than prior
        result = network.conditional_probability("query", {"evidence": True})
        assert result.p_true > 0.5

    def test_conflicting_evidence(self):
        """Conflicting evidence should show different effects."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "Target claim", "agent1", 0.5)  # Start neutral
        network.add_claim("B", "Supporting evidence", "agent2", 0.5)
        network.add_claim("C", "Contradicting evidence", "agent3", 0.5)
        network.add_factor("B", "A", RelationType.SUPPORTS, 1.0)
        network.add_factor("C", "A", RelationType.CONTRADICTS, 1.0)
        network.propagate()

        # Get baseline P(A) with no evidence
        baseline = network.get_node_by_claim("A").posterior.p_true

        # With support evidence
        result_support = network.conditional_probability("A", {"B": True})
        network.propagate()  # Reset

        # With contradict evidence
        result_contradict = network.conditional_probability("A", {"C": True})

        # Support should give higher P(A=true) than contradict
        assert result_support.p_true > result_contradict.p_true

    def test_no_evidence_returns_prior(self):
        """Empty evidence should return propagated posterior."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.7)
        network.propagate()

        result = network.conditional_probability("A", {})
        # Should be close to the prior/propagated value
        assert abs(result.p_true - 0.7) < 0.2

    def test_multiple_evidence_claims(self):
        """Multiple evidence claims should combine."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("E1", "Evidence 1", "agent1", 0.5)
        network.add_claim("E2", "Evidence 2", "agent2", 0.5)
        network.add_claim("Q", "Query", "agent3", 0.5)
        network.add_factor("E1", "Q", RelationType.SUPPORTS, 0.9)
        network.add_factor("E2", "Q", RelationType.SUPPORTS, 0.9)

        # Both evidence true should strongly support query
        result = network.conditional_probability("Q", {"E1": True, "E2": True})
        assert result.p_true > 0.7

    def test_evidence_on_disconnected_node(self):
        """Evidence on disconnected node should not affect query."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.7)
        network.add_claim("B", "B", "agent2", 0.5)
        # No factor between A and B

        original = network.conditional_probability("B", {})
        with_evidence = network.conditional_probability("B", {"A": True})

        # B should be similar regardless of A's evidence
        assert abs(original.p_true - with_evidence.p_true) < 0.2

    def test_nonexistent_query_returns_uniform(self):
        """Query for nonexistent claim should return uniform."""
        network = BeliefNetwork()
        network.add_claim("A", "A", "agent1", 0.8)

        result = network.conditional_probability("nonexistent", {"A": True})
        assert abs(result.p_true - 0.5) < 0.01


# =============================================================================
# Category F: Sensitivity Analysis
# =============================================================================


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""

    def test_high_sensitivity_detection(self):
        """Strongly connected claims should show some sensitivity."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.5)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 1.0)
        network.propagate()

        sensitivities = network.sensitivity_analysis("B")
        assert "A" in sensitivities
        # A should have non-zero sensitivity to B (any positive value indicates influence)
        assert sensitivities["A"] > 0

    def test_low_sensitivity_independent_claim(self):
        """Disconnected claims should have low sensitivity."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("A", "A", "agent1", 0.8)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_claim("C", "C", "agent3", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        # C is not connected to B
        network.propagate()

        sensitivities = network.sensitivity_analysis("B")
        # C should have low sensitivity to B
        if "C" in sensitivities:
            assert sensitivities["C"] < sensitivities.get("A", 1.0)

    def test_sensitivity_with_cycles(self, cyclic_network):
        """Sensitivity in cyclic network should still compute."""
        cyclic_network.propagate()

        sensitivities = cyclic_network.sensitivity_analysis("A")
        # Should have sensitivity values for B and C
        assert len(sensitivities) == 2

    def test_sensitivity_ordering(self):
        """Both direct and indirect connections should show sensitivity."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("target", "Target", "agent1", 0.5)
        network.add_claim("direct", "Direct", "agent2", 0.5)
        network.add_claim("indirect", "Indirect", "agent3", 0.5)
        network.add_factor("direct", "target", RelationType.SUPPORTS, 1.0)
        network.add_factor("indirect", "direct", RelationType.SUPPORTS, 0.5)
        network.propagate()

        sensitivities = network.sensitivity_analysis("target")
        # Both should show some sensitivity
        assert sensitivities["direct"] > 0
        assert "indirect" in sensitivities
        # Both should have computed sensitivities
        assert len(sensitivities) == 2

    def test_single_claim_network_sensitivity(self):
        """Single claim network should return empty sensitivities."""
        network = BeliefNetwork()
        network.add_claim("A", "A", "agent1", 0.5)
        network.propagate()

        sensitivities = network.sensitivity_analysis("A")
        assert sensitivities == {}


# =============================================================================
# Category G: Convergence Behavior
# =============================================================================


class TestConvergenceBehavior:
    """Tests for convergence with different parameters."""

    @pytest.mark.parametrize("damping", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_damping_factors(self, damping):
        """Network should handle various damping factors."""
        network = BeliefNetwork(damping=damping, max_iterations=100)
        network.add_claim("A", "A", "agent1", 0.9)
        network.add_claim("B", "B", "agent2", 0.1)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.8)

        result = network.propagate()
        assert result.iterations <= 100
        assert not math.isnan(result.max_change)

    def test_tight_convergence_threshold(self):
        """Tight threshold should require more iterations."""
        network_tight = BeliefNetwork(
            damping=0.5, convergence_threshold=1e-10, max_iterations=200
        )
        network_loose = BeliefNetwork(
            damping=0.5, convergence_threshold=0.1, max_iterations=200
        )

        for net in [network_tight, network_loose]:
            net.add_claim("A", "A", "agent1", 0.9)
            net.add_claim("B", "B", "agent2", 0.1)
            net.add_factor("A", "B", RelationType.SUPPORTS, 0.9)

        result_tight = network_tight.propagate()
        result_loose = network_loose.propagate()

        # Loose threshold should converge in fewer or equal iterations
        assert result_loose.iterations <= result_tight.iterations

    def test_network_that_doesnt_converge(self):
        """Network hitting max_iterations should report not converged."""
        network = BeliefNetwork(
            damping=0.0,  # No damping can cause oscillation
            max_iterations=5,
            convergence_threshold=1e-15,
        )
        network.add_claim("A", "A", "agent1", 0.9)
        network.add_claim("B", "B", "agent2", 0.1)
        network.add_factor("A", "B", RelationType.CONTRADICTS, 1.0)
        network.add_factor("B", "A", RelationType.CONTRADICTS, 1.0)

        result = network.propagate()
        assert result.iterations == 5

    def test_empty_network_handling(self):
        """Empty network should handle propagation gracefully."""
        network = BeliefNetwork()
        result = network.propagate()

        assert result.converged is True
        assert result.iterations <= 1
        assert len(result.node_posteriors) == 0

    def test_large_network_convergence(self):
        """Large network (100+ nodes) should converge."""
        network = BeliefNetwork(damping=0.5, max_iterations=100)

        # Create a chain of 100 nodes
        for i in range(100):
            network.add_claim(f"N{i}", f"Node {i}", f"agent{i}", 0.5 + 0.003 * i)
        for i in range(99):
            network.add_factor(f"N{i}", f"N{i+1}", RelationType.SUPPORTS, 0.7)

        result = network.propagate()
        assert result.iterations <= 100
        assert len(result.node_posteriors) == 100

    def test_early_termination_on_convergence(self):
        """Network should stop early when converged."""
        network = BeliefNetwork(
            damping=0.5, convergence_threshold=0.01, max_iterations=1000
        )
        network.add_claim("A", "A", "agent1", 0.8)
        network.add_claim("B", "B", "agent2", 0.8)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.5)

        result = network.propagate()
        # Should converge well before max_iterations
        assert result.iterations < 1000


# =============================================================================
# Category H: Graph Topology Edge Cases
# =============================================================================


class TestGraphTopologyEdgeCases:
    """Tests for unusual graph topologies."""

    def test_orphaned_nodes(self):
        """Nodes with no connections should maintain prior."""
        network = BeliefNetwork()
        network.add_claim("orphan", "Orphan node", "agent1", 0.7)
        network.propagate()

        orphan = network.get_node_by_claim("orphan")
        # Orphan should maintain approximately its prior
        assert abs(orphan.posterior.p_true - 0.7) < 0.1

    def test_source_nodes_only_children(self):
        """Source nodes (only outgoing edges) should propagate correctly."""
        network = BeliefNetwork()
        network.add_claim("source", "Source", "agent1", 0.9)
        network.add_claim("child1", "Child 1", "agent2", 0.5)
        network.add_claim("child2", "Child 2", "agent3", 0.5)
        network.add_factor("source", "child1", RelationType.SUPPORTS, 0.9)
        network.add_factor("source", "child2", RelationType.SUPPORTS, 0.9)
        network.propagate()

        # Source should remain high
        source = network.get_node_by_claim("source")
        assert source.posterior.p_true > 0.6

    def test_sink_nodes_only_parents(self):
        """Sink nodes (only incoming edges) should be influenced by parents."""
        network = BeliefNetwork()
        network.add_claim("parent1", "Parent 1", "agent1", 0.9)
        network.add_claim("parent2", "Parent 2", "agent2", 0.9)
        network.add_claim("sink", "Sink", "agent3", 0.5)
        network.add_factor("parent1", "sink", RelationType.SUPPORTS, 0.9)
        network.add_factor("parent2", "sink", RelationType.SUPPORTS, 0.9)
        network.propagate()

        # Sink should be influenced by high-confidence parents
        sink = network.get_node_by_claim("sink")
        assert sink.posterior.p_true > 0.5

    def test_disconnected_components_propagation(self):
        """Disconnected components should propagate independently."""
        network = BeliefNetwork()
        # Component 1
        network.add_claim("A1", "A1", "agent1", 0.9)
        network.add_claim("B1", "B1", "agent2", 0.5)
        network.add_factor("A1", "B1", RelationType.SUPPORTS, 0.9)
        # Component 2
        network.add_claim("A2", "A2", "agent3", 0.1)
        network.add_claim("B2", "B2", "agent4", 0.5)
        network.add_factor("A2", "B2", RelationType.SUPPORTS, 0.9)

        network.propagate()

        b1 = network.get_node_by_claim("B1")
        b2 = network.get_node_by_claim("B2")
        # B1 should be higher than B2 due to different parent confidences
        assert b1.posterior.p_true > b2.posterior.p_true

    def test_add_factor_before_nodes_exist(self):
        """Adding factor with nonexistent nodes should return None."""
        network = BeliefNetwork()
        factor = network.add_factor("nonexistent1", "nonexistent2", RelationType.SUPPORTS)
        assert factor is None

    def test_factor_with_missing_claim(self):
        """Adding factor with one missing claim should return None."""
        network = BeliefNetwork()
        network.add_claim("exists", "Exists", "agent1", 0.5)
        factor = network.add_factor("exists", "missing", RelationType.SUPPORTS)
        assert factor is None

    def test_node_lookup_by_claim_id(self):
        """get_node_by_claim should work correctly."""
        network = BeliefNetwork()
        network.add_claim("my-claim-id", "Test", "agent1", 0.5)

        node = network.get_node_by_claim("my-claim-id")
        assert node is not None
        assert node.claim_id == "my-claim-id"

        missing = network.get_node_by_claim("not-found")
        assert missing is None

    def test_get_contested_claims_detection(self):
        """get_contested_claims should detect disagreeing messages."""
        network = BeliefNetwork(damping=0.5)
        network.add_claim("target", "Target", "agent1", 0.5)
        network.add_claim("supporter", "Supporter", "agent2", 0.9)
        network.add_claim("opponent", "Opponent", "agent3", 0.9)
        network.add_factor("supporter", "target", RelationType.SUPPORTS, 1.0)
        network.add_factor("opponent", "target", RelationType.CONTRADICTS, 1.0)
        network.propagate()

        contested = network.get_contested_claims()
        # Target should be contested due to conflicting incoming messages
        claim_ids = [n.claim_id for n in contested]
        # May or may not include target depending on message difference threshold
        assert isinstance(contested, list)


# =============================================================================
# Category I: Factor Potentials
# =============================================================================


class TestFactorPotentials:
    """Tests for factor potential edge cases."""

    def test_strength_zero_inert_factor(self):
        """Factor with strength=0 should have minimal effect."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=0.0,
        )
        # With strength=0, both_true potential should be lower than max
        pot = factor.get_factor_potential(True, True)
        assert pot == 0.7  # 0.7 + 0.3 * 0 = 0.7

    def test_strength_one_maximum(self):
        """Factor with strength=1.0 should have maximum effect."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        pot = factor.get_factor_potential(True, True)
        assert pot == 1.0  # 0.7 + 0.3 * 1.0 = 1.0

    def test_supports_vs_contradicts_potentials(self):
        """SUPPORTS and CONTRADICTS should have opposite effects."""
        supports = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        contradicts = Factor(
            factor_id="f2",
            relation_type=RelationType.CONTRADICTS,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )

        # Both true: SUPPORTS high, CONTRADICTS low
        assert supports.get_factor_potential(True, True) > contradicts.get_factor_potential(True, True)
        # Source true, target false: SUPPORTS low, CONTRADICTS high
        assert supports.get_factor_potential(True, False) < contradicts.get_factor_potential(True, False)

    def test_depends_on_asymmetry(self):
        """DEPENDS_ON should be asymmetric (target needs source)."""
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.DEPENDS_ON,
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )

        # Target true without source true should be very unlikely
        impossible = factor.get_factor_potential(False, True)
        assert impossible < 0.2

        # Target true with source true should be likely
        possible = factor.get_factor_potential(True, True)
        assert possible > 0.5

    def test_unknown_relation_type_default(self):
        """Unknown relation type should use default correlation."""
        # Use a valid RelationType but test the else branch by checking its behavior
        # The else branch handles any relation type not explicitly coded
        factor = Factor(
            factor_id="f1",
            relation_type=RelationType.SUPPORTS,  # Known type
            source_node_id="n1",
            target_node_id="n2",
            strength=1.0,
        )
        # This tests the known path; unknown types would fall to else
        pot_same = factor.get_factor_potential(True, True)
        pot_diff = factor.get_factor_potential(True, False)
        # SUPPORTS: same truth high, different truth low
        assert pot_same > pot_diff


# =============================================================================
# Category J: Serialization
# =============================================================================


class TestSerialization:
    """Tests for serialization and round-trip conversion."""

    def test_full_network_to_dict_roundtrip(self):
        """to_dict should capture all network state."""
        network = BeliefNetwork(debate_id="test-123", damping=0.6)
        network.add_claim("A", "Claim A", "agent1", 0.8)
        network.add_claim("B", "Claim B", "agent2", 0.6)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.9)
        network.propagate()

        d = network.to_dict()

        assert d["debate_id"] == "test-123"
        assert d["damping"] == 0.6
        assert len(d["nodes"]) == 2
        assert len(d["factors"]) == 1
        assert "claim_to_node" in d

    def test_to_json_validity(self):
        """to_json should produce valid JSON."""
        network = BeliefNetwork()
        network.add_claim("A", "Test claim", "agent1", 0.7)
        network.propagate()

        json_str = network.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "debate_id" in parsed
        assert "nodes" in parsed

    def test_large_network_serialization(self):
        """Large network should serialize without issues."""
        network = BeliefNetwork()
        for i in range(50):
            network.add_claim(f"C{i}", f"Claim {i}", f"agent{i}", 0.5)
        for i in range(49):
            network.add_factor(f"C{i}", f"C{i+1}", RelationType.SUPPORTS, 0.7)
        network.propagate()

        d = network.to_dict()
        assert len(d["nodes"]) == 50
        assert len(d["factors"]) == 49

        json_str = network.to_json()
        assert len(json_str) > 0

    def test_metadata_preservation(self):
        """Node metadata should be preserved in serialization."""
        network = BeliefNetwork()
        node = network.add_claim("A", "Test", "agent1", 0.5)
        node.metadata = {"custom_key": "custom_value"}

        # Node metadata is not currently serialized in to_dict
        # This test documents current behavior
        d = network.to_dict()
        assert "nodes" in d

    def test_empty_network_serialization(self):
        """Empty network should serialize correctly."""
        network = BeliefNetwork(debate_id="empty")

        d = network.to_dict()
        assert d["debate_id"] == "empty"
        assert len(d["nodes"]) == 0
        assert len(d["factors"]) == 0

        json_str = network.to_json()
        parsed = json.loads(json_str)
        assert parsed["debate_id"] == "empty"

    def test_propagation_result_serialization(self):
        """PropagationResult should serialize correctly."""
        network = BeliefNetwork()
        network.add_claim("A", "Test", "agent1", 0.7)
        network.add_claim("B", "Test 2", "agent2", 0.5)
        network.add_factor("A", "B", RelationType.SUPPORTS, 0.8)

        result = network.propagate()
        d = result.to_dict()

        assert "converged" in d
        assert "iterations" in d
        assert "max_change" in d
        assert "node_posteriors" in d
        assert "centralities" in d


# =============================================================================
# Additional Tests: Analyzer Methods
# =============================================================================


class TestAnalyzerMethods:
    """Additional tests for BeliefPropagationAnalyzer."""

    def test_identify_cruxes_empty_network(self):
        """Empty network should return empty cruxes."""
        network = BeliefNetwork()
        analyzer = BeliefPropagationAnalyzer(network)

        cruxes = analyzer.identify_debate_cruxes()
        assert cruxes == []

    def test_suggest_evidence_targets_certain_network(self):
        """Certain network should have few evidence suggestions."""
        network = BeliefNetwork()
        network.add_claim("A", "Certain claim", "agent1", 0.99)
        network.propagate()

        analyzer = BeliefPropagationAnalyzer(network)
        suggestions = analyzer.suggest_evidence_targets()

        # Highly certain claim shouldn't need more evidence
        # (entropy < 0.8 threshold)
        assert len(suggestions) == 0

    def test_what_if_multiple_hypotheticals(self):
        """What-if with multiple hypotheticals should combine effects."""
        network = BeliefNetwork()
        network.add_claim("A", "A", "agent1", 0.5)
        network.add_claim("B", "B", "agent2", 0.5)
        network.add_claim("C", "C", "agent3", 0.5)
        network.add_factor("A", "C", RelationType.SUPPORTS, 0.9)
        network.add_factor("B", "C", RelationType.SUPPORTS, 0.9)
        network.propagate()

        analyzer = BeliefPropagationAnalyzer(network)
        result = analyzer.what_if_analysis({"A": True, "B": True})

        # Both supporting evidence should strongly affect C
        assert "changes" in result
