"""
Gap tests for NomicIntegration.

Targets specific uncovered paths:
- Checkpoint resume edge cases
- Deadlock resolution paths
- Staleness report dataclass
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.nomic.integration import (
    NomicIntegration,
    create_nomic_integration,
    BeliefAnalysis,
    StalenessReport,
)
from aragora.reasoning.belief import (
    BeliefNetwork,
    BeliefNode,
    BeliefDistribution,
    BeliefStatus,
)
from aragora.debate.counterfactual import CounterfactualStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration_with_checkpoint():
    """Create integration with checkpointing enabled and mocked manager."""
    integration = NomicIntegration(
        enable_checkpointing=True,
        enable_counterfactual=True,
    )
    integration.checkpoint_mgr = MagicMock()
    integration.checkpoint_mgr.create_checkpoint = AsyncMock()
    integration.checkpoint_mgr.resume_from_checkpoint = AsyncMock()
    integration.checkpoint_mgr.list_debates_with_checkpoints = AsyncMock(return_value=[])
    return integration


@pytest.fixture
def sample_belief_network():
    """Create a sample belief network with nodes."""
    network = BeliefNetwork()
    network.add_claim(
        claim_id="claim-1",
        statement="Climate change is real",
        author="agent-1",
        initial_confidence=0.7,
    )
    network.add_claim(
        claim_id="claim-2",
        statement="Renewable energy is cost-effective",
        author="agent-2",
        initial_confidence=0.6,
    )
    from aragora.reasoning.claims import RelationType

    network.add_factor("claim-1", "claim-2", RelationType.SUPPORTS, strength=0.8)
    return network


@pytest.fixture
def contested_nodes():
    """Create contested belief nodes for deadlock testing."""
    node1 = BeliefNode(
        node_id="contested-1",
        claim_id="contested-claim-1",
        claim_statement="Implementation approach A is better",
        author="agent-1",
        prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        status=BeliefStatus.CONTESTED,
    )
    node2 = BeliefNode(
        node_id="contested-2",
        claim_id="contested-claim-2",
        claim_statement="We should use library X",
        author="agent-2",
        prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        status=BeliefStatus.CONTESTED,
    )
    return [node1, node2]


# =============================================================================
# Checkpoint Resume Tests
# =============================================================================


class TestCheckpointResume:
    """Tests for checkpoint resume functionality."""

    @pytest.mark.asyncio
    async def test_resume_restores_belief_network(
        self, integration_with_checkpoint, sample_belief_network
    ):
        """Test that resume restores belief network from checkpoint."""
        # Create mock resumed debate with belief network
        mock_checkpoint = MagicMock()
        mock_checkpoint.debate_id = "debate-456"
        mock_checkpoint.current_round = 3
        mock_checkpoint.phase = "debate"
        mock_checkpoint.belief_network_state = sample_belief_network.to_dict()

        mock_resumed = MagicMock()
        mock_resumed.checkpoint = mock_checkpoint
        integration_with_checkpoint.checkpoint_mgr.resume_from_checkpoint.return_value = (
            mock_resumed
        )

        result = await integration_with_checkpoint.resume_from_checkpoint("ckpt-123")

        assert result is not None
        assert result.phase == "debate"
        assert integration_with_checkpoint._belief_network is not None
        assert len(integration_with_checkpoint._belief_network.nodes) == 2

    @pytest.mark.asyncio
    async def test_resume_sets_debate_id_and_cycle(self, integration_with_checkpoint):
        """Test that resume sets debate_id and cycle from checkpoint."""
        mock_checkpoint = MagicMock()
        mock_checkpoint.debate_id = "debate-789"
        mock_checkpoint.current_round = 5
        mock_checkpoint.phase = "design"
        mock_checkpoint.belief_network_state = None

        mock_resumed = MagicMock()
        mock_resumed.checkpoint = mock_checkpoint
        integration_with_checkpoint.checkpoint_mgr.resume_from_checkpoint.return_value = (
            mock_resumed
        )

        await integration_with_checkpoint.resume_from_checkpoint("ckpt-456")

        assert integration_with_checkpoint._current_debate_id == "debate-789"
        assert integration_with_checkpoint._current_cycle == 5

    @pytest.mark.asyncio
    async def test_resume_nonexistent_checkpoint_returns_none(self, integration_with_checkpoint):
        """Test that resume with nonexistent checkpoint returns None."""
        integration_with_checkpoint.checkpoint_mgr.resume_from_checkpoint.return_value = None

        result = await integration_with_checkpoint.resume_from_checkpoint("nonexistent-ckpt")

        assert result is None

    @pytest.mark.asyncio
    async def test_resume_exception_returns_none(self, integration_with_checkpoint):
        """Test that resume gracefully handles exceptions."""
        integration_with_checkpoint.checkpoint_mgr.resume_from_checkpoint.side_effect = (
            RuntimeError("DB error")
        )

        result = await integration_with_checkpoint.resume_from_checkpoint("ckpt-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_resume_disabled_returns_none(self):
        """Test that resume with checkpointing disabled returns None."""
        integration = NomicIntegration(enable_checkpointing=False)

        result = await integration.resume_from_checkpoint("any-ckpt")

        assert result is None


# =============================================================================
# Deadlock Resolution Tests
# =============================================================================


class TestDeadlockResolutionExtended:
    """Extended tests for deadlock resolution."""

    @pytest.mark.asyncio
    async def test_resolve_deadlock_empty_claims_returns_none(self):
        """Test that empty contested claims returns None."""
        integration = NomicIntegration(
            enable_checkpointing=False,
            enable_counterfactual=True,
        )

        result = await integration.resolve_deadlock([])

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_disabled_returns_none(self, contested_nodes):
        """Test that disabled counterfactual returns None."""
        integration = NomicIntegration(
            enable_checkpointing=False,
            enable_counterfactual=False,
        )

        result = await integration.resolve_deadlock(contested_nodes)

        assert result is None


# =============================================================================
# Belief Analysis Tests
# =============================================================================


class TestBeliefAnalysisDataclass:
    """Tests for BeliefAnalysis dataclass methods."""

    def test_has_deadlock_with_crux_claims(self):
        """Test has_deadlock returns True with crux claims."""
        node = BeliefNode(
            node_id="crux",
            claim_id="crux-claim",
            claim_statement="Disputed claim",
            author="agent-1",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={"crux-claim": 0.9},
            contested_claims=[node],
            crux_claims=[node],
            convergence_achieved=False,
            iterations_used=50,
        )

        assert analysis.has_deadlock is True

    def test_has_deadlock_no_crux_claims(self):
        """Test has_deadlock returns False without crux claims."""
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=10,
        )

        assert analysis.has_deadlock is False

    def test_top_crux_returns_highest_centrality(self):
        """Test top_crux returns node with highest centrality."""
        node1 = BeliefNode(
            node_id="low",
            claim_id="low-centrality",
            claim_statement="Lower priority claim",
            author="agent-1",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )
        node2 = BeliefNode(
            node_id="high",
            claim_id="high-centrality",
            claim_statement="Higher priority claim",
            author="agent-2",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={"low-centrality": 0.3, "high-centrality": 0.9},
            contested_claims=[node1, node2],
            crux_claims=[node1, node2],
            convergence_achieved=False,
            iterations_used=50,
        )

        assert analysis.top_crux.claim_id == "high-centrality"

    def test_top_crux_no_crux_claims(self):
        """Test top_crux returns None without crux claims."""
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=10,
        )

        assert analysis.top_crux is None

    def test_to_dict_serialization(self):
        """Test to_dict returns expected structure."""
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={"claim-1": 0.5},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=10,
        )

        result = analysis.to_dict()

        assert "network_size" in result
        assert "contested_count" in result
        assert "crux_count" in result
        assert "convergence_achieved" in result
        assert "iterations_used" in result
        assert "posteriors" in result
        assert "centralities" in result
        assert result["convergence_achieved"] is True
        assert result["iterations_used"] == 10


# =============================================================================
# Staleness Report Tests
# =============================================================================


class TestStalenessReportDataclass:
    """Tests for StalenessReport dataclass."""

    def test_needs_redebate_high_severity(self):
        """Test needs_redebate returns True for high severity."""
        from aragora.reasoning.provenance_enhanced import RevalidationTrigger

        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[
                RevalidationTrigger(
                    trigger_id="trigger-1",
                    claim_id="test",
                    evidence_ids=["evidence-1"],
                    staleness_checks=[],
                    severity="high",
                    recommendation="Re-debate required",
                )
            ],
        )

        assert report.needs_redebate is True

    def test_needs_redebate_critical_severity(self):
        """Test needs_redebate returns True for critical severity."""
        from aragora.reasoning.provenance_enhanced import RevalidationTrigger

        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[
                RevalidationTrigger(
                    trigger_id="trigger-2",
                    claim_id="test",
                    evidence_ids=["evidence-1"],
                    staleness_checks=[],
                    severity="critical",
                    recommendation="Urgent re-debate",
                )
            ],
        )

        assert report.needs_redebate is True

    def test_needs_redebate_low_severity(self):
        """Test needs_redebate returns False for low severity."""
        from aragora.reasoning.provenance_enhanced import RevalidationTrigger

        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[
                RevalidationTrigger(
                    trigger_id="trigger-3",
                    claim_id="test",
                    evidence_ids=["evidence-1"],
                    staleness_checks=[],
                    severity="info",
                    recommendation="Optional update",
                )
            ],
        )

        assert report.needs_redebate is False

    def test_needs_redebate_empty_triggers(self):
        """Test needs_redebate returns False with no triggers."""
        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[],
        )

        assert report.needs_redebate is False

    def test_stale_claim_ids(self):
        """Test stale_claim_ids extracts claim IDs."""
        from aragora.reasoning.claims import TypedClaim, ClaimType

        claim1 = TypedClaim(
            claim_id="claim-1",
            claim_type=ClaimType.ASSERTION,
            statement="Test claim 1",
            author="test",
            confidence=0.8,
        )
        claim2 = TypedClaim(
            claim_id="claim-2",
            claim_type=ClaimType.ASSERTION,
            statement="Test claim 2",
            author="test",
            confidence=0.7,
        )
        report = StalenessReport(
            stale_claims=[claim1, claim2],
            staleness_checks={},
            revalidation_triggers=[],
        )

        assert report.stale_claim_ids == ["claim-1", "claim-2"]


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management methods."""

    def test_get_agent_weights_returns_copy(self):
        """Test that get_agent_weights returns a copy."""
        integration = create_nomic_integration(enable_checkpointing=False)
        integration._agent_weights = {"agent-1": 0.8}

        weights = integration.get_agent_weights()
        weights["agent-2"] = 0.5

        assert "agent-2" not in integration._agent_weights

    def test_set_cycle_updates_state(self):
        """Test that set_cycle updates internal state."""
        integration = create_nomic_integration(enable_checkpointing=False)

        integration.set_cycle(5)

        assert integration._current_cycle == 5

    def test_set_debate_id_updates_state(self):
        """Test that set_debate_id updates internal state."""
        integration = create_nomic_integration(enable_checkpointing=False)

        integration.set_debate_id("debate-123")

        assert integration._current_debate_id == "debate-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
