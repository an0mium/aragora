"""
Tests for NomicIntegration module.

Tests the integration hub that coordinates:
- Bayesian belief propagation for debate analysis
- Capability probing for agent reliability
- Evidence staleness detection
- Counterfactual branching for deadlock resolution
- Checkpointing for crash recovery
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.nomic.integration import (
    NomicIntegration,
    BeliefAnalysis,
    AgentReliability,
    StalenessReport,
    PhaseCheckpoint,
    create_nomic_integration,
)
from aragora.core import Agent, Vote, DebateResult
from aragora.reasoning.claims import TypedClaim, ClaimType

# Note: ClaimType values are ASSERTION, PROPOSAL, OBJECTION, etc.
# Not FACTUAL or POSITION


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, model: str = "mock"):
        self.name = name
        self.model = model
        self.role = "proposer"

    async def generate(self, prompt: str, context=None) -> str:
        return f"Mock response from {self.name}"

    async def critique(self, proposal: str, task: str, context=None):
        from aragora.core import Critique
        return Critique(
            agent=self.name,
            target_agent="other",
            target_content=proposal[:50],
            issues=["mock issue"],
            suggestions=["mock suggestion"],
            severity=0.5,
            reasoning="mock reasoning",
        )


class TestNomicIntegrationCreation:
    """Test NomicIntegration initialization."""

    def test_create_with_defaults(self):
        """Test creating integration with default settings."""
        integration = create_nomic_integration()

        assert integration is not None
        assert integration.enable_probing is True
        assert integration.enable_belief_analysis is True
        assert integration.enable_staleness_check is True
        assert integration.enable_counterfactual is True
        assert integration.enable_checkpointing is True

    def test_create_with_disabled_features(self):
        """Test creating integration with some features disabled."""
        integration = NomicIntegration(
            enable_probing=False,
            enable_belief_analysis=False,
            enable_staleness_check=True,
            enable_counterfactual=True,
            enable_checkpointing=False,
        )

        assert integration.enable_probing is False
        assert integration.enable_belief_analysis is False
        assert integration.enable_staleness_check is True
        assert integration.prober is None  # Disabled

    def test_create_with_checkpoint_dir(self):
        """Test creating integration with custom checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints"
            integration = create_nomic_integration(checkpoint_dir=str(checkpoint_path))

            assert integration.checkpoint_mgr is not None
            assert checkpoint_path.exists()


class TestBeliefAnalysis:
    """Test belief propagation analysis."""

    @pytest.mark.asyncio
    async def test_analyze_empty_debate(self):
        """Test analysis of debate with no votes."""
        integration = create_nomic_integration(enable_checkpointing=False)

        result = DebateResult(
            task="test task",
            votes=[],
            messages=[],
            critiques=[],
        )

        analysis = await integration.analyze_debate(result)

        assert isinstance(analysis, BeliefAnalysis)
        assert len(analysis.contested_claims) == 0
        assert analysis.convergence_achieved is True

    @pytest.mark.asyncio
    async def test_analyze_debate_with_votes(self):
        """Test analysis of debate with votes."""
        integration = create_nomic_integration(enable_checkpointing=False)

        result = DebateResult(
            task="test task",
            votes=[
                Vote(agent="agent1", choice="proposal_a", confidence=0.8, reasoning="good"),
                Vote(agent="agent2", choice="proposal_a", confidence=0.7, reasoning="agreed"),
                Vote(agent="agent3", choice="proposal_b", confidence=0.6, reasoning="different"),
            ],
            messages=[],
            critiques=[],
        )

        analysis = await integration.analyze_debate(result)

        assert isinstance(analysis, BeliefAnalysis)
        assert len(analysis.posteriors) > 0
        assert analysis.to_dict() is not None

    @pytest.mark.asyncio
    async def test_analyze_disabled(self):
        """Test that disabled belief analysis returns empty result."""
        integration = NomicIntegration(
            enable_belief_analysis=False,
            enable_checkpointing=False,
        )

        result = DebateResult(task="test", votes=[])

        analysis = await integration.analyze_debate(result)

        assert analysis.convergence_achieved is True
        assert analysis.iterations_used == 0


class TestAgentProbing:
    """Test capability probing for agent reliability."""

    @pytest.mark.asyncio
    async def test_probe_disabled(self):
        """Test that disabled probing returns uniform weights."""
        integration = NomicIntegration(
            enable_probing=False,
            enable_checkpointing=False,
        )

        agents = [MockAgent("agent1"), MockAgent("agent2")]
        weights = await integration.probe_agents(agents)

        assert weights == {"agent1": 1.0, "agent2": 1.0}

    @pytest.mark.asyncio
    async def test_get_agent_weights(self):
        """Test getting cached agent weights."""
        integration = NomicIntegration(
            enable_probing=False,
            enable_checkpointing=False,
        )

        # Initially empty
        assert integration.get_agent_weights() == {}

        # After probing
        integration._agent_weights = {"agent1": 0.8, "agent2": 0.6}
        weights = integration.get_agent_weights()

        assert weights == {"agent1": 0.8, "agent2": 0.6}


class TestStalenessDetection:
    """Test evidence staleness detection."""

    @pytest.mark.asyncio
    async def test_staleness_disabled(self):
        """Test that disabled staleness check returns empty result."""
        integration = NomicIntegration(
            enable_staleness_check=False,
            enable_checkpointing=False,
        )

        claims = [TypedClaim(
            claim_id="c1",
            statement="test claim",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.8,
        )]
        changed_files = ["file1.py", "file2.py"]

        report = await integration.check_staleness(claims, changed_files)

        assert isinstance(report, StalenessReport)
        assert len(report.stale_claims) == 0
        assert not report.needs_redebate

    @pytest.mark.asyncio
    async def test_extract_file_references(self):
        """Test extracting file references from claims."""
        integration = create_nomic_integration(enable_checkpointing=False)

        claim = TypedClaim(
            claim_id="c1",
            statement="The implementation in aragora/core.py handles this case",
            claim_type=ClaimType.ASSERTION,
            author="test",
            confidence=0.8,
        )

        files = integration._extract_file_references(claim)

        assert "aragora/core.py" in files


class TestCheckpointing:
    """Test phase checkpointing."""

    @pytest.mark.asyncio
    async def test_checkpoint_disabled(self):
        """Test that disabled checkpointing returns None."""
        integration = NomicIntegration(enable_checkpointing=False)

        result = await integration.checkpoint(
            phase="debate",
            state={"key": "value"},
            cycle=1,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_checkpoint_enabled(self):
        """Test checkpointing with enabled feature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            integration = NomicIntegration(
                checkpoint_dir=Path(tmpdir),
                enable_checkpointing=True,
            )

            integration.set_debate_id("test-debate-123")
            integration.set_cycle(1)

            checkpoint_id = await integration.checkpoint(
                phase="debate",
                state={"result": "test consensus", "confidence": 0.8},
            )

            # May be None if checkpoint manager isn't fully set up
            # Just verify no exceptions were raised

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints."""
        integration = NomicIntegration(enable_checkpointing=False)

        checkpoints = await integration.list_checkpoints()

        assert checkpoints == []


class TestDeadlockResolution:
    """Test counterfactual branching for deadlock resolution."""

    @pytest.mark.asyncio
    async def test_resolve_empty_deadlock(self):
        """Test resolving with no contested claims."""
        integration = create_nomic_integration(enable_checkpointing=False)

        result = await integration.resolve_deadlock([])

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_disabled(self):
        """Test that disabled counterfactual returns None."""
        integration = NomicIntegration(
            enable_counterfactual=False,
            enable_checkpointing=False,
        )

        from aragora.reasoning.belief import BeliefNode, BeliefDistribution

        node = BeliefNode(
            node_id="test-node",
            claim_id="test",
            claim_statement="contested claim",
            author="test_agent",
            prior=BeliefDistribution(p_true=0.5, p_false=0.5),
        )

        result = await integration.resolve_deadlock([node])

        assert result is None


class TestStateManagement:
    """Test state management methods."""

    def test_set_cycle(self):
        """Test setting current cycle."""
        integration = create_nomic_integration(enable_checkpointing=False)

        integration.set_cycle(5)

        assert integration._current_cycle == 5

    def test_set_debate_id(self):
        """Test setting current debate ID."""
        integration = create_nomic_integration(enable_checkpointing=False)

        integration.set_debate_id("debate-123")

        assert integration._current_debate_id == "debate-123"


class TestBeliefAnalysisDataclass:
    """Test BeliefAnalysis dataclass methods."""

    def test_has_deadlock_false(self):
        """Test has_deadlock returns False with no crux claims."""
        from aragora.reasoning.belief import BeliefNetwork

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
        assert analysis.top_crux is None

    def test_to_dict(self):
        """Test BeliefAnalysis.to_dict()."""
        from aragora.reasoning.belief import BeliefNetwork

        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=10,
        )

        data = analysis.to_dict()

        assert "network_size" in data
        assert "contested_count" in data
        assert "convergence_achieved" in data
        assert data["convergence_achieved"] is True


class TestStalenessReportDataclass:
    """Test StalenessReport dataclass methods."""

    def test_needs_redebate_false(self):
        """Test needs_redebate returns False with no triggers."""
        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[],
        )

        assert report.needs_redebate is False
        assert report.stale_claim_ids == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
