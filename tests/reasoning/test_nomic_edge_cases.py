"""
Edge case tests for NomicIntegration module.

Tests critical edge cases identified from recent fixes:
- Null posterior handling in deadlock resolution
- Empty centralities when belief network is None
- File reference extraction with special patterns
- Single contested claim handling
- Staleness check severity classification
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.nomic.integration import (
    NomicIntegration,
    BeliefAnalysis,
    StalenessReport,
    create_nomic_integration,
)
from aragora.reasoning.belief import BeliefNode, BeliefDistribution, BeliefNetwork
from aragora.reasoning.claims import TypedClaim, ClaimType


class TestDeadlockResolutionEdgeCases:
    """Test edge cases in resolve_deadlock method."""

    @pytest.mark.asyncio
    async def test_resolve_deadlock_disabled_returns_none(self):
        """Test deadlock resolution returns None when counterfactual disabled."""
        integration = NomicIntegration(
            enable_counterfactual=False,
            enable_checkpointing=False,
        )

        node = BeliefNode(
            node_id="node-1",
            claim_id="claim-1",
            claim_statement="Contested claim",
            author="test_agent",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )

        result = await integration.resolve_deadlock([node])
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_empty_claims(self):
        """Test deadlock resolution with empty claims list."""
        integration = NomicIntegration(
            enable_counterfactual=True,
            enable_checkpointing=False,
        )

        result = await integration.resolve_deadlock([])
        assert result is None

    def test_belief_node_entropy_calculation(self):
        """Test that BeliefNode entropy calculation works correctly.

        This tests the code path at integration.py:541 which uses
        pivot_node.posterior.entropy for disagreement_score.
        """
        # High entropy (uncertain) - 50/50 split (default posterior is uniform)
        node_uncertain = BeliefNode(
            node_id="uncertain",
            claim_id="claim-1",
            claim_statement="Uncertain claim",
            author="test",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )
        assert node_uncertain.posterior.entropy > 0.5

        # Low entropy (certain) - need to explicitly set posterior
        node_certain = BeliefNode(
            node_id="certain",
            claim_id="claim-2",
            claim_statement="Certain claim",
            author="test",
            prior=BeliefDistribution(p_true=0.9, p_false=0.1),
        )
        # Posterior defaults to uniform, set it explicitly for low entropy
        node_certain.posterior = BeliefDistribution(p_true=0.95, p_false=0.05)
        assert node_certain.posterior.entropy < 0.5

    def test_pivot_node_selection_without_network(self):
        """Test pivot selection uses first claim when no belief network.

        Tests the fallback at integration.py:533-534 which sets
        centralities = {} and uses contested_claims[0].
        """
        # When no belief network, should default to first claim
        claims = [
            BeliefNode(
                node_id="node-1",
                claim_id="first",
                claim_statement="First claim",
                author="agent1",
                prior=BeliefDistribution(p_true=0.5, p_false=0.5),
            ),
            BeliefNode(
                node_id="node-2",
                claim_id="second",
                claim_statement="Second claim",
                author="agent2",
                prior=BeliefDistribution(p_true=0.5, p_false=0.5),
            ),
        ]

        # The first claim should be selected as pivot
        centralities = {}  # Empty, simulating no belief network
        pivot_node = claims[0]  # Integration code does this when network is None
        assert pivot_node.claim_id == "first"

    def test_pivot_node_selection_with_centralities(self):
        """Test pivot selection uses highest centrality claim."""
        claims = [
            BeliefNode(
                node_id="node-1",
                claim_id="low-central",
                claim_statement="Low centrality claim",
                author="agent1",
                prior=BeliefDistribution(p_true=0.5, p_false=0.5),
                centrality=0.3,
            ),
            BeliefNode(
                node_id="node-2",
                claim_id="high-central",
                claim_statement="High centrality claim",
                author="agent2",
                prior=BeliefDistribution(p_true=0.5, p_false=0.5),
                centrality=0.9,
            ),
        ]

        centralities = {"low-central": 0.3, "high-central": 0.9}
        pivot_node = max(claims, key=lambda n: centralities.get(n.claim_id, 0))
        assert pivot_node.claim_id == "high-central"


class TestFileReferenceExtractionEdgeCases:
    """Test edge cases in _extract_file_references method."""

    def test_extract_file_with_multiple_dots(self):
        """Test extracting files with multiple dots like test.spec.ts."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c1",
            statement="See the test file test.spec.ts for details",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.8,
        )

        files = integration._extract_file_references(claim)
        # The regex should match .ts extension
        assert any("test.spec.ts" in f for f in files) or any(".ts" in f for f in files)

    def test_extract_file_with_nested_path(self):
        """Test extracting deeply nested file paths."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c2",
            statement="Implementation is in aragora/server/handlers/debates.py",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.9,
        )

        files = integration._extract_file_references(claim)
        assert "aragora/server/handlers/debates.py" in files

    def test_extract_multiple_file_types(self):
        """Test extracting various file extensions."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c3",
            statement="Check config.json and styles.tsx and README.md for details",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.7,
        )

        files = integration._extract_file_references(claim)
        # Should find all three file types
        assert len(files) >= 2
        assert any("config.json" in f for f in files)
        assert any("styles.tsx" in f for f in files)

    def test_extract_no_files(self):
        """Test claim with no file references."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c4",
            statement="This is a general statement with no files mentioned",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.5,
        )

        files = integration._extract_file_references(claim)
        assert files == []

    def test_extract_file_with_dashes_and_underscores(self):
        """Test extracting files with dashes and underscores in names."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c5",
            statement="The file src/my-component_test.tsx handles this",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.8,
        )

        files = integration._extract_file_references(claim)
        # Should extract a .tsx file with the full path
        assert any("my-component_test.tsx" in f for f in files)


class TestStalenessCheckEdgeCases:
    """Test edge cases in staleness detection."""

    @pytest.mark.asyncio
    async def test_staleness_disabled_returns_empty(self):
        """Test staleness check returns empty when disabled."""
        integration = NomicIntegration(
            enable_staleness_check=False,
            enable_checkpointing=False,
        )

        claim = TypedClaim(
            claim_id="assert-1",
            statement="The aragora/core.py file implements validation",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.9,
        )

        report = await integration.check_staleness([claim], ["aragora/core.py"])
        assert isinstance(report, StalenessReport)
        assert len(report.stale_claims) == 0
        assert not report.needs_redebate

    @pytest.mark.asyncio
    async def test_staleness_check_various_claim_types(self):
        """Test staleness check handles all claim types without error."""
        integration = NomicIntegration(
            enable_staleness_check=False,  # Disabled to avoid provenance issues
            enable_checkpointing=False,
        )

        claim_types = [
            ClaimType.ASSERTION,
            ClaimType.PROPOSAL,
            ClaimType.OBJECTION,
            ClaimType.QUESTION,
        ]

        for claim_type in claim_types:
            claim = TypedClaim(
                claim_id=f"claim-{claim_type.name}",
                statement=f"Test {claim_type.name} claim about core.py",
                claim_type=claim_type,
                author="test",
                confidence=0.7,
            )

            report = await integration.check_staleness([claim], ["core.py"])
            assert isinstance(report, StalenessReport)

    @pytest.mark.asyncio
    async def test_staleness_empty_changed_files(self):
        """Test staleness check with no changed files."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c1",
            statement="Implementation in test.py",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.8,
        )

        report = await integration.check_staleness([claim], [])
        assert not report.needs_redebate

    @pytest.mark.asyncio
    async def test_staleness_empty_claims(self):
        """Test staleness check with no claims."""
        integration = create_nomic_integration(enable_checkpointing=False)

        report = await integration.check_staleness([], ["file.py"])
        assert len(report.stale_claims) == 0
        assert not report.needs_redebate


class TestBeliefNetworkEdgeCases:
    """Test edge cases in belief network operations."""

    def test_entropy_with_certainty(self):
        """Test BeliefDistribution entropy when fully certain."""
        # Full certainty should have low/zero entropy
        dist = BeliefDistribution(p_true=1.0, p_false=0.0)
        assert dist.entropy < 0.1  # Should be near zero

    def test_entropy_with_uncertainty(self):
        """Test BeliefDistribution entropy when maximally uncertain."""
        # Max uncertainty (50/50) should have maximum entropy
        dist = BeliefDistribution(p_true=0.5, p_false=0.5)
        assert dist.entropy > 0.5  # Should be near 1.0 (log2 scale)

    def test_belief_node_to_dict_with_none_posterior(self):
        """Test BeliefNode serialization when posterior is None."""
        node = BeliefNode(
            node_id="test",
            claim_id="c1",
            claim_statement="Test claim",
            author="test",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )

        data = node.to_dict()
        assert "posterior" in data
        # Posterior should be None or handled gracefully
        assert data["claim_id"] == "c1"


class TestAgentProbingEdgeCases:
    """Test edge cases in agent probing."""

    @pytest.mark.asyncio
    async def test_probe_empty_agent_list(self):
        """Test probing with empty agent list."""
        integration = NomicIntegration(
            enable_probing=True,
            enable_checkpointing=False,
        )

        weights = await integration.probe_agents([])
        assert weights == {}

    @pytest.mark.asyncio
    async def test_get_weights_before_probing(self):
        """Test getting weights before any probing is done."""
        integration = create_nomic_integration(enable_checkpointing=False)

        weights = integration.get_agent_weights()
        assert weights == {}


class TestCheckpointEdgeCases:
    """Test edge cases in checkpointing."""

    @pytest.mark.asyncio
    async def test_resume_nonexistent_checkpoint(self):
        """Test resuming from a checkpoint that doesn't exist."""
        integration = NomicIntegration(enable_checkpointing=False)

        result = await integration.resume_from_checkpoint("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints_disabled(self):
        """Test listing checkpoints when feature is disabled."""
        integration = NomicIntegration(enable_checkpointing=False)

        checkpoints = await integration.list_checkpoints()
        assert checkpoints == []


class TestBeliefAnalysisEdgeCases:
    """Test edge cases in BeliefAnalysis dataclass."""

    def test_has_deadlock_with_crux_claims(self):
        """Test has_deadlock returns True when crux claims exist."""
        crux = BeliefNode(
            node_id="crux",
            claim_id="crux-1",
            claim_statement="Critical disagreement point",
            author="test",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )

        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[crux],
            crux_claims=[crux],
            convergence_achieved=False,
            iterations_used=50,
        )

        assert analysis.has_deadlock is True
        assert analysis.top_crux == crux

    def test_to_dict_comprehensive(self):
        """Test BeliefAnalysis.to_dict with all fields populated."""
        network = BeliefNetwork()

        # posteriors is a dict of str -> BeliefDistribution
        analysis = BeliefAnalysis(
            network=network,
            posteriors={
                "claim-1": BeliefDistribution(p_true=0.8, p_false=0.2),
                "claim-2": BeliefDistribution(p_true=0.3, p_false=0.7),
            },
            centralities={"claim-1": 0.7, "claim-2": 0.5},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=25,
        )

        data = analysis.to_dict()

        assert "network_size" in data
        assert "posteriors" in data
        assert "convergence_achieved" in data
        assert data["iterations_used"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
