"""
Tests for NomicIntegration - the nomic loop feature integration hub.

This module tests:
1. Integration initialization with various feature flags
2. State synchronization between systems
3. Error handling during sync operations
4. Conflict resolution
5. Edge cases
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import uuid

from aragora.nomic.integration import (
    NomicIntegration,
    BeliefAnalysis,
    AgentReliability,
    StalenessReport,
    PhaseCheckpoint,
    create_nomic_integration,
)
from aragora.core import Agent, DebateResult, Vote, Message
from aragora.reasoning.belief import BeliefNetwork, BeliefNode, BeliefDistribution, BeliefStatus
from aragora.reasoning.claims import ClaimsKernel, ClaimType, TypedClaim
from aragora.reasoning.provenance_enhanced import (
    StalenessCheck,
    StalenessStatus,
    RevalidationTrigger,
)
from aragora.debate.counterfactual import CounterfactualBranch, PivotClaim, ConditionalConsensus
from aragora.debate.checkpoint import DebateCheckpoint, CheckpointStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()
    elo.get_rating = MagicMock(return_value=1500)
    elo.update_ratings = MagicMock()
    return elo


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def integration(tmp_checkpoint_dir):
    """Create a NomicIntegration instance with all features enabled."""
    return NomicIntegration(
        elo_system=None,
        checkpoint_dir=tmp_checkpoint_dir,
        enable_probing=True,
        enable_belief_analysis=True,
        enable_staleness_check=True,
        enable_counterfactual=True,
        enable_checkpointing=True,
    )


@pytest.fixture
def integration_minimal():
    """Create a NomicIntegration instance with all features disabled."""
    return NomicIntegration(
        elo_system=None,
        checkpoint_dir=None,
        enable_probing=False,
        enable_belief_analysis=False,
        enable_staleness_check=False,
        enable_counterfactual=False,
        enable_checkpointing=False,
    )


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult for testing."""
    return DebateResult(
        id="test-debate-001",
        debate_id="test-debate-001",
        task="Test task for debate",
        final_answer="The agreed solution is X",
        confidence=0.85,
        consensus_reached=True,
        rounds_used=3,
        participants=["agent1", "agent2", "agent3"],
        votes=[
            Vote(agent="agent1", choice="agent2", reasoning="Good solution", confidence=0.9),
            Vote(agent="agent2", choice="agent2", reasoning="I agree", confidence=0.8),
            Vote(agent="agent3", choice="agent1", reasoning="Different approach", confidence=0.7),
        ],
        messages=[
            Message(role="proposer", agent="agent1", content="Proposal A"),
            Message(role="proposer", agent="agent2", content="Proposal B"),
        ],
    )


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    agents = []
    for name in ["claude", "gemini", "grok"]:
        agent = MagicMock(spec=Agent)
        agent.name = name
        agent.model = f"{name}-model"
        agents.append(agent)
    return agents


@pytest.fixture
def mock_claims_kernel():
    """Create a mock ClaimsKernel with sample claims."""
    kernel = ClaimsKernel(debate_id="test-debate-001")

    # Use the add_claim method with its expected signature
    kernel.add_claim(
        statement="The system should use rate limiting (see aragora/server.py)",
        author="agent1",
        claim_type=ClaimType.ASSERTION,
        confidence=0.9,
    )

    kernel.add_claim(
        statement="We should implement caching in utils.py",
        author="agent2",
        claim_type=ClaimType.PROPOSAL,
        confidence=0.8,
    )

    return kernel


# =============================================================================
# Test: Integration Initialization
# =============================================================================


class TestIntegrationInitialization:
    """Tests for NomicIntegration initialization."""

    def test_init_with_defaults(self, tmp_checkpoint_dir):
        """Should initialize with default settings."""
        integration = NomicIntegration(checkpoint_dir=tmp_checkpoint_dir)

        assert integration.enable_probing is True
        assert integration.enable_belief_analysis is True
        assert integration.enable_staleness_check is True
        assert integration.enable_counterfactual is True
        assert integration.enable_checkpointing is True

    def test_init_all_features_disabled(self):
        """Should initialize with all features disabled."""
        integration = NomicIntegration(
            enable_probing=False,
            enable_belief_analysis=False,
            enable_staleness_check=False,
            enable_counterfactual=False,
            enable_checkpointing=False,
        )

        assert integration.prober is None
        assert integration.provenance is None
        assert integration.counterfactual is None
        assert integration.checkpoint_mgr is None

    def test_init_with_elo_system(self, mock_elo_system, tmp_checkpoint_dir):
        """Should initialize with ELO system for probing."""
        integration = NomicIntegration(
            elo_system=mock_elo_system,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        assert integration.elo_system is mock_elo_system

    def test_init_creates_checkpoint_dir(self, tmp_path):
        """Should create checkpoint directory if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoints"
        assert not new_dir.exists()

        integration = NomicIntegration(checkpoint_dir=new_dir)

        assert new_dir.exists()

    def test_init_state_tracking_defaults(self, integration):
        """Should initialize state tracking variables."""
        assert integration._current_debate_id is None
        assert integration._current_cycle == 0
        assert integration._belief_network is None
        assert integration._agent_weights == {}
        assert integration._phase_checkpoints == []


class TestCreateNomicIntegrationHelper:
    """Tests for create_nomic_integration convenience function."""

    def test_create_with_defaults(self, tmp_path):
        """Should create integration with defaults."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        integration = create_nomic_integration(checkpoint_dir=str(checkpoint_dir))

        assert isinstance(integration, NomicIntegration)
        assert integration.enable_probing is True

    def test_create_with_feature_flags(self, tmp_path):
        """Should accept feature flags."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        integration = create_nomic_integration(
            checkpoint_dir=str(checkpoint_dir),
            enable_probing=False,
            enable_belief_analysis=False,
        )

        assert integration.enable_probing is False
        assert integration.enable_belief_analysis is False

    def test_create_with_elo_system(self, mock_elo_system, tmp_path):
        """Should pass ELO system to integration."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        integration = create_nomic_integration(
            elo_system=mock_elo_system,
            checkpoint_dir=str(checkpoint_dir),
        )

        assert integration.elo_system is mock_elo_system


# =============================================================================
# Test: Belief Analysis
# =============================================================================


class TestBeliefAnalysis:
    """Tests for analyze_debate with belief propagation."""

    @pytest.mark.asyncio
    async def test_analyze_debate_basic(self, integration, mock_debate_result):
        """Should analyze debate and return BeliefAnalysis."""
        analysis = await integration.analyze_debate(mock_debate_result)

        assert isinstance(analysis, BeliefAnalysis)
        assert analysis.network is not None
        assert isinstance(analysis.posteriors, dict)
        assert isinstance(analysis.centralities, dict)

    @pytest.mark.asyncio
    async def test_analyze_debate_disabled(self, integration_minimal, mock_debate_result):
        """Should return empty analysis when belief analysis is disabled."""
        analysis = await integration_minimal.analyze_debate(mock_debate_result)

        assert isinstance(analysis, BeliefAnalysis)
        assert len(analysis.posteriors) == 0
        assert analysis.convergence_achieved is True
        assert analysis.iterations_used == 0

    @pytest.mark.asyncio
    async def test_analyze_debate_with_claims_kernel(
        self, integration, mock_debate_result, mock_claims_kernel
    ):
        """Should incorporate claims kernel into belief network."""
        analysis = await integration.analyze_debate(
            mock_debate_result,
            claims_kernel=mock_claims_kernel,
        )

        assert analysis.network is not None
        # Should have nodes for both vote proposals and kernel claims
        assert len(analysis.network.nodes) > 0

    @pytest.mark.asyncio
    async def test_analyze_debate_stores_network(self, integration, mock_debate_result):
        """Should store belief network for later use."""
        await integration.analyze_debate(mock_debate_result)

        assert integration._belief_network is not None
        assert integration._current_debate_id == mock_debate_result.id

    @pytest.mark.asyncio
    async def test_analyze_debate_identifies_contested_claims(
        self, integration, mock_debate_result
    ):
        """Should identify contested claims with high entropy."""
        analysis = await integration.analyze_debate(
            mock_debate_result,
            disagreement_threshold=0.3,  # Lower threshold to catch more
        )

        # Should have checked for contested claims
        assert isinstance(analysis.contested_claims, list)
        assert isinstance(analysis.crux_claims, list)

    @pytest.mark.asyncio
    async def test_analyze_debate_convergence_tracking(self, integration, mock_debate_result):
        """Should track convergence status."""
        analysis = await integration.analyze_debate(mock_debate_result)

        assert isinstance(analysis.convergence_achieved, bool)
        assert isinstance(analysis.iterations_used, int)


class TestBeliefAnalysisDataclass:
    """Tests for BeliefAnalysis dataclass."""

    def test_has_deadlock_property(self):
        """has_deadlock should be True when crux_claims exist."""
        analysis_no_deadlock = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=5,
        )
        assert analysis_no_deadlock.has_deadlock is False

        mock_crux = MagicMock(spec=BeliefNode)
        analysis_with_deadlock = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[mock_crux],
            convergence_achieved=False,
            iterations_used=10,
        )
        assert analysis_with_deadlock.has_deadlock is True

    def test_top_crux_property(self):
        """top_crux should return highest centrality crux claim."""
        mock_crux1 = MagicMock(spec=BeliefNode)
        mock_crux1.claim_id = "crux-1"
        mock_crux2 = MagicMock(spec=BeliefNode)
        mock_crux2.claim_id = "crux-2"

        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={"crux-1": 0.3, "crux-2": 0.7},
            contested_claims=[],
            crux_claims=[mock_crux1, mock_crux2],
            convergence_achieved=False,
            iterations_used=10,
        )

        assert analysis.top_crux is mock_crux2

    def test_top_crux_empty(self):
        """top_crux should return None when no crux claims."""
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=5,
        )

        assert analysis.top_crux is None

    def test_to_dict(self):
        """to_dict should serialize analysis correctly."""
        analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={"node-1": BeliefDistribution(p_true=0.8, p_false=0.2)},
            centralities={"node-1": 0.5},
            contested_claims=[],
            crux_claims=[],
            convergence_achieved=True,
            iterations_used=5,
        )

        result = analysis.to_dict()

        assert "network_size" in result
        assert "contested_count" in result
        assert "crux_count" in result
        assert "convergence_achieved" in result
        assert "posteriors" in result
        assert "timestamp" in result


# =============================================================================
# Test: Agent Probing
# =============================================================================


class TestAgentProbing:
    """Tests for probe_agents capability probing."""

    @pytest.mark.asyncio
    async def test_probe_agents_disabled(self, integration_minimal, mock_agents):
        """Should return uniform weights when probing disabled."""
        weights = await integration_minimal.probe_agents(mock_agents)

        assert all(w == 1.0 for w in weights.values())

    @pytest.mark.asyncio
    async def test_probe_agents_no_run_fn(self, integration, mock_agents):
        """Should return uniform weights when no run_agent_fn provided."""
        weights = await integration.probe_agents(mock_agents)

        assert all(w == 1.0 for w in weights.values())
        assert len(weights) == len(mock_agents)

    @pytest.mark.asyncio
    async def test_probe_agents_stores_weights(self, integration, mock_agents):
        """Should store weights for later retrieval after probing with run_fn."""

        async def mock_run_agent(agent, prompt):
            return "Mock response"

        mock_report = MagicMock()
        mock_report.vulnerability_rate = 0.2
        mock_report.critical_count = 0

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = mock_report

            await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
            )

        stored = integration.get_agent_weights()
        assert len(stored) == len(mock_agents)

    @pytest.mark.asyncio
    async def test_probe_agents_with_run_fn(self, integration, mock_agents):
        """Should use run_agent_fn to probe agents."""

        async def mock_run_agent(agent, prompt):
            return "Mock response"

        # Mock the prober to return controlled results
        mock_report = MagicMock()
        mock_report.vulnerability_rate = 0.2
        mock_report.critical_count = 0

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = mock_report

            weights = await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
                probe_count=2,
            )

        # Should have called probe for each agent
        assert mock_probe.call_count == len(mock_agents)
        # Weights should be 1.0 - vulnerability_rate = 0.8
        assert all(0.5 <= w <= 1.0 for w in weights.values())

    @pytest.mark.asyncio
    async def test_probe_agents_critical_vulnerability_penalty(self, integration, mock_agents):
        """Should reduce weight for critical vulnerabilities."""

        async def mock_run_agent(agent, prompt):
            return "Mock response"

        mock_report = MagicMock()
        mock_report.vulnerability_rate = 0.2
        mock_report.critical_count = 2  # Critical vulnerabilities

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = mock_report

            weights = await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
            )

        # Weight should be (1.0 - 0.2) * 0.5 = 0.4
        assert all(w == pytest.approx(0.4, rel=0.01) for w in weights.values())

    @pytest.mark.asyncio
    async def test_probe_agents_error_handling(self, integration, mock_agents):
        """Should use default weight on probing error."""

        async def mock_run_agent(agent, prompt):
            return "Mock response"

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.side_effect = RuntimeError("Probe failed")

            weights = await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
            )

        # Should use default weight of 0.75 on error
        assert all(w == 0.75 for w in weights.values())

    @pytest.mark.asyncio
    async def test_probe_agents_min_weight(self, integration, mock_agents):
        """Should respect minimum weight parameter."""

        async def mock_run_agent(agent, prompt):
            return "Mock response"

        mock_report = MagicMock()
        mock_report.vulnerability_rate = 0.9  # Very high vulnerability
        mock_report.critical_count = 0

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = mock_report

            weights = await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
                min_weight=0.6,  # Set high minimum
            )

        # Should not go below min_weight
        assert all(w >= 0.6 for w in weights.values())


class TestAgentReliabilityDataclass:
    """Tests for AgentReliability dataclass."""

    def test_is_reliable_property(self):
        """is_reliable should be True when weight >= 0.7."""
        reliable = AgentReliability(
            agent_name="claude",
            weight=0.85,
            vulnerability_report=None,
            probe_results=[],
        )
        assert reliable.is_reliable is True

        unreliable = AgentReliability(
            agent_name="weak_agent",
            weight=0.5,
            vulnerability_report=None,
            probe_results=[],
        )
        assert unreliable.is_reliable is False

    def test_critical_vulnerabilities_property(self):
        """Should return critical count from vulnerability report."""
        mock_report = MagicMock()
        mock_report.critical_count = 3

        reliability = AgentReliability(
            agent_name="test",
            weight=0.5,
            vulnerability_report=mock_report,
            probe_results=[],
        )

        assert reliability.critical_vulnerabilities == 3

    def test_critical_vulnerabilities_no_report(self):
        """Should return 0 when no vulnerability report."""
        reliability = AgentReliability(
            agent_name="test",
            weight=0.8,
            vulnerability_report=None,
            probe_results=[],
        )

        assert reliability.critical_vulnerabilities == 0


# =============================================================================
# Test: Staleness Check
# =============================================================================


class TestStalenessCheck:
    """Tests for check_staleness evidence staleness detection."""

    @pytest.mark.asyncio
    async def test_check_staleness_disabled(self, integration_minimal):
        """Should return empty report when staleness check disabled."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.ASSERTION,
                statement="Test claim",
                author="agent1",
                confidence=0.9,
            )
        ]

        report = await integration_minimal.check_staleness(claims, ["file.py"])

        assert isinstance(report, StalenessReport)
        assert len(report.stale_claims) == 0
        assert len(report.revalidation_triggers) == 0

    @pytest.mark.asyncio
    async def test_check_staleness_no_affected_files(self, integration):
        """Should not mark claims stale when no file overlap."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.ASSERTION,
                statement="See server.py for details",
                author="agent1",
                confidence=0.9,
            )
        ]

        report = await integration.check_staleness(claims, ["other_file.py"])

        assert len(report.stale_claims) == 0

    @pytest.mark.asyncio
    async def test_check_staleness_affected_files(self, integration):
        """Should mark claims stale when referencing changed files."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.ASSERTION,
                statement="Implementation in server.py uses rate limiting",
                author="agent1",
                confidence=0.9,
            )
        ]

        # Mock provenance manager
        mock_check = StalenessCheck(
            evidence_id="e1",
            status=StalenessStatus.STALE,
            checked_at=datetime.now().isoformat(),
            reason="File changed",
        )

        with patch.object(
            integration.provenance,
            "check_claim_evidence_staleness",
            return_value=[mock_check],
        ):
            report = await integration.check_staleness(claims, ["server.py"])

        assert len(report.stale_claims) == 1
        assert report.stale_claims[0].claim_id == "c1"

    @pytest.mark.asyncio
    async def test_check_staleness_creates_triggers(self, integration):
        """Should create revalidation triggers for stale claims."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.ASSERTION,
                statement="See config.yaml for settings",
                author="agent1",
                confidence=0.9,
            )
        ]

        mock_check = StalenessCheck(
            evidence_id="e1",
            status=StalenessStatus.STALE,
            checked_at=datetime.now().isoformat(),
            reason="File changed",
        )

        with patch.object(
            integration.provenance,
            "check_claim_evidence_staleness",
            return_value=[mock_check],
        ):
            report = await integration.check_staleness(claims, ["config.yaml"])

        assert len(report.revalidation_triggers) == 1
        trigger = report.revalidation_triggers[0]
        assert trigger.claim_id == "c1"
        assert trigger.severity == "high"  # ASSERTION claims get high severity

    @pytest.mark.asyncio
    async def test_check_staleness_file_extraction_patterns(self, integration):
        """Should extract various file patterns from claims."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.PROPOSAL,
                statement="Modify aragora/debate/orchestrator.py and tests/test_debate.py",
                author="agent1",
                confidence=0.9,
            )
        ]

        mock_check = StalenessCheck(
            evidence_id="e1",
            status=StalenessStatus.STALE,
            checked_at=datetime.now().isoformat(),
            reason="Files changed",
        )

        with patch.object(
            integration.provenance,
            "check_claim_evidence_staleness",
            return_value=[mock_check],
        ):
            report = await integration.check_staleness(
                claims,
                ["aragora/debate/orchestrator.py"],
            )

        assert len(report.stale_claims) == 1


class TestStalenessReportDataclass:
    """Tests for StalenessReport dataclass."""

    def test_needs_redebate_high_severity(self):
        """needs_redebate should be True for high/critical severity triggers."""
        trigger = RevalidationTrigger(
            trigger_id="t1",
            claim_id="c1",
            evidence_ids=["e1"],
            staleness_checks=[],
            severity="high",
            recommendation="Re-debate needed",
        )

        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[trigger],
        )

        assert report.needs_redebate is True

    def test_needs_redebate_low_severity(self):
        """needs_redebate should be False for low severity triggers."""
        trigger = RevalidationTrigger(
            trigger_id="t1",
            claim_id="c1",
            evidence_ids=["e1"],
            staleness_checks=[],
            severity="low",
            recommendation="Minor change",
        )

        report = StalenessReport(
            stale_claims=[],
            staleness_checks={},
            revalidation_triggers=[trigger],
        )

        assert report.needs_redebate is False

    def test_stale_claim_ids_property(self):
        """Should return list of stale claim IDs."""
        claim1 = TypedClaim(
            claim_id="c1",
            claim_type=ClaimType.ASSERTION,
            statement="Claim 1",
            author="agent1",
            confidence=0.9,
        )
        claim2 = TypedClaim(
            claim_id="c2",
            claim_type=ClaimType.PROPOSAL,
            statement="Claim 2",
            author="agent2",
            confidence=0.8,
        )

        report = StalenessReport(
            stale_claims=[claim1, claim2],
            staleness_checks={},
            revalidation_triggers=[],
        )

        assert report.stale_claim_ids == ["c1", "c2"]


# =============================================================================
# Test: Deadlock Resolution
# =============================================================================


class TestDeadlockResolution:
    """Tests for resolve_deadlock counterfactual branching."""

    @pytest.mark.asyncio
    async def test_resolve_deadlock_disabled(self, integration_minimal):
        """Should return None when counterfactual disabled."""
        mock_node = MagicMock(spec=BeliefNode)

        result = await integration_minimal.resolve_deadlock([mock_node])

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_empty_claims(self, integration):
        """Should return None for empty contested claims."""
        result = await integration.resolve_deadlock([])

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_creates_branches(self, integration):
        """Should create counterfactual branches."""
        # Create mock belief node
        mock_node = MagicMock(spec=BeliefNode)
        mock_node.claim_id = "claim-1"
        mock_node.claim_statement = "Test claim"
        mock_node.author = "agent1"
        mock_node.posterior = MagicMock()
        mock_node.posterior.entropy = 0.8
        mock_node.centrality = 0.5

        # Set up belief network for centrality computation
        integration._belief_network = MagicMock()
        integration._belief_network._compute_centralities.return_value = {"claim-1": 0.7}

        # Mock branch running
        mock_true_branch = MagicMock(spec=CounterfactualBranch)
        mock_true_branch.assumption = True
        mock_true_branch.conclusion = "True conclusion"
        mock_true_branch.confidence = 0.9

        mock_false_branch = MagicMock(spec=CounterfactualBranch)
        mock_false_branch.assumption = False
        mock_false_branch.conclusion = "False conclusion"
        mock_false_branch.confidence = 0.85

        mock_consensus = MagicMock(spec=ConditionalConsensus)

        with patch.object(
            integration.counterfactual,
            "create_and_run_branches",
            new_callable=AsyncMock,
            return_value=[mock_true_branch, mock_false_branch],
        ):
            with patch.object(
                integration.counterfactual,
                "synthesize_branches",
                return_value=mock_consensus,
            ):
                result = await integration.resolve_deadlock([mock_node])

        assert result is mock_consensus

    @pytest.mark.asyncio
    async def test_resolve_deadlock_with_run_branch_fn(self, integration):
        """Should use provided run_branch_fn."""
        mock_node = MagicMock(spec=BeliefNode)
        mock_node.claim_id = "claim-1"
        mock_node.claim_statement = "Test claim"
        mock_node.author = "agent1"
        mock_node.posterior = MagicMock()
        mock_node.posterior.entropy = 0.8
        mock_node.centrality = 0.5

        async def custom_run_branch(task, context, branch_id):
            return DebateResult(
                id=branch_id,
                task=task,
                final_answer="Custom branch result",
                confidence=0.95,
            )

        mock_true_branch = MagicMock()
        mock_true_branch.assumption = True
        mock_false_branch = MagicMock()
        mock_false_branch.assumption = False

        with patch.object(
            integration.counterfactual,
            "create_and_run_branches",
            new_callable=AsyncMock,
            return_value=[mock_true_branch, mock_false_branch],
        ):
            with patch.object(
                integration.counterfactual,
                "synthesize_branches",
                return_value=MagicMock(),
            ) as mock_synthesize:
                await integration.resolve_deadlock(
                    [mock_node],
                    run_branch_fn=custom_run_branch,
                )

        # Verify synthesis was called with branches
        mock_synthesize.assert_called_once()


# =============================================================================
# Test: Full Post-Debate Analysis
# =============================================================================


class TestFullPostDebateAnalysis:
    """Tests for full_post_debate_analysis unified entry point."""

    @pytest.mark.asyncio
    async def test_full_analysis_basic(self, integration, mock_debate_result):
        """Should run all analyses and return combined result."""
        result = await integration.full_post_debate_analysis(mock_debate_result)

        assert "belief" in result
        assert "conditional" in result
        assert "staleness" in result
        assert "summary" in result

        assert isinstance(result["belief"], BeliefAnalysis)

    @pytest.mark.asyncio
    async def test_full_analysis_summary_populated(self, integration, mock_debate_result):
        """Should populate summary with analysis results."""
        result = await integration.full_post_debate_analysis(mock_debate_result)

        summary = result["summary"]
        assert "has_deadlock" in summary
        assert "needs_redebate" in summary
        assert "contested_count" in summary
        assert "crux_count" in summary
        assert "stale_count" in summary

    @pytest.mark.asyncio
    async def test_full_analysis_with_claims_and_files(
        self, integration, mock_debate_result, mock_claims_kernel
    ):
        """Should check staleness when claims and files provided."""
        mock_check = StalenessCheck(
            evidence_id="e1",
            status=StalenessStatus.STALE,
            checked_at=datetime.now().isoformat(),
            reason="File changed",
        )

        with patch.object(
            integration.provenance,
            "check_claim_evidence_staleness",
            return_value=[mock_check],
        ):
            result = await integration.full_post_debate_analysis(
                mock_debate_result,
                claims_kernel=mock_claims_kernel,
                changed_files=["aragora/server.py"],
            )

        assert result["staleness"] is not None

    @pytest.mark.asyncio
    async def test_full_analysis_deadlock_resolution(self, integration, mock_debate_result):
        """Should attempt deadlock resolution when deadlock detected."""
        # Make analyze_debate return a result with deadlock
        mock_crux = MagicMock(spec=BeliefNode)
        mock_crux.claim_id = "crux-1"
        mock_crux.claim_statement = "Crux claim"
        mock_crux.author = "agent1"
        mock_crux.posterior = MagicMock()
        mock_crux.posterior.entropy = 0.9
        mock_crux.centrality = 0.8

        mock_analysis = BeliefAnalysis(
            network=BeliefNetwork(),
            posteriors={},
            centralities={"crux-1": 0.8},
            contested_claims=[mock_crux],
            crux_claims=[mock_crux],
            convergence_achieved=False,
            iterations_used=10,
        )

        with patch.object(integration, "analyze_debate", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = mock_analysis

            with patch.object(
                integration, "resolve_deadlock", new_callable=AsyncMock
            ) as mock_resolve:
                mock_resolve.return_value = MagicMock()

                result = await integration.full_post_debate_analysis(mock_debate_result)

        # Should have attempted deadlock resolution
        mock_resolve.assert_called_once()


# =============================================================================
# Test: Checkpointing
# =============================================================================


class TestCheckpointing:
    """Tests for checkpoint and resume functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_disabled(self, integration_minimal):
        """Should return None when checkpointing disabled."""
        result = await integration_minimal.checkpoint(
            phase="debate",
            state={"test": "data"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_checkpoint_creates_checkpoint(self, integration):
        """Should create and return checkpoint ID."""
        integration.set_debate_id("test-debate")
        integration.set_cycle(1)

        with patch.object(
            integration.checkpoint_mgr,
            "create_checkpoint",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_checkpoint = MagicMock()
            mock_checkpoint.checkpoint_id = "checkpoint-001"
            mock_create.return_value = mock_checkpoint

            checkpoint_id = await integration.checkpoint(
                phase="debate",
                state={"proposals": ["A", "B"]},
            )

        assert checkpoint_id == "checkpoint-001"
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_tracks_phase_checkpoints(self, integration):
        """Should track phase checkpoints internally."""
        integration.set_debate_id("test-debate")
        integration.set_cycle(2)

        with patch.object(
            integration.checkpoint_mgr,
            "create_checkpoint",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_checkpoint = MagicMock()
            mock_checkpoint.checkpoint_id = "checkpoint-001"
            mock_create.return_value = mock_checkpoint

            await integration.checkpoint(
                phase="implement",
                state={"code": "test"},
            )

        assert len(integration._phase_checkpoints) == 1
        assert integration._phase_checkpoints[0].phase == "implement"
        assert integration._phase_checkpoints[0].cycle == 2

    @pytest.mark.asyncio
    async def test_checkpoint_error_handling(self, integration):
        """Should return None on checkpoint error (not raise)."""
        with patch.object(
            integration.checkpoint_mgr,
            "create_checkpoint",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Storage error"),
        ):
            result = await integration.checkpoint(
                phase="verify",
                state={},
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_disabled(self, integration_minimal):
        """Should return None when checkpointing disabled."""
        result = await integration_minimal.resume_from_checkpoint("checkpoint-001")

        assert result is None

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, integration):
        """Should restore state from checkpoint."""
        mock_checkpoint = MagicMock(spec=DebateCheckpoint)
        mock_checkpoint.checkpoint_id = "checkpoint-001"
        mock_checkpoint.debate_id = "test-debate"
        mock_checkpoint.phase = "design"
        mock_checkpoint.current_round = 3
        mock_checkpoint.belief_network_state = None

        mock_resumed = MagicMock()
        mock_resumed.checkpoint = mock_checkpoint

        with patch.object(
            integration.checkpoint_mgr,
            "resume_from_checkpoint",
            new_callable=AsyncMock,
            return_value=mock_resumed,
        ):
            result = await integration.resume_from_checkpoint("checkpoint-001")

        assert isinstance(result, PhaseCheckpoint)
        assert result.phase == "design"
        assert integration._current_debate_id == "test-debate"
        assert integration._current_cycle == 3

    @pytest.mark.asyncio
    async def test_resume_restores_belief_network(self, integration):
        """Should restore belief network from checkpoint state."""
        mock_network = BeliefNetwork()

        mock_checkpoint = MagicMock(spec=DebateCheckpoint)
        mock_checkpoint.checkpoint_id = "checkpoint-001"
        mock_checkpoint.debate_id = "test-debate"
        mock_checkpoint.phase = "debate"
        mock_checkpoint.current_round = 2
        mock_checkpoint.belief_network_state = {"nodes": {}}

        mock_resumed = MagicMock()
        mock_resumed.checkpoint = mock_checkpoint

        with patch.object(
            integration.checkpoint_mgr,
            "resume_from_checkpoint",
            new_callable=AsyncMock,
            return_value=mock_resumed,
        ):
            with patch.object(
                BeliefNetwork,
                "from_dict",
                return_value=mock_network,
            ):
                result = await integration.resume_from_checkpoint("checkpoint-001")

        assert integration._belief_network is mock_network

    @pytest.mark.asyncio
    async def test_resume_error_handling(self, integration):
        """Should return None on resume error."""
        with patch.object(
            integration.checkpoint_mgr,
            "resume_from_checkpoint",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Not found"),
        ):
            result = await integration.resume_from_checkpoint("invalid-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, integration):
        """Should list available checkpoints."""
        integration.set_debate_id("test-debate")

        mock_checkpoints = [
            {"debate_id": "test-debate", "phase": "debate"},
            {"debate_id": "test-debate", "phase": "design"},
            {"debate_id": "other-debate", "phase": "implement"},
        ]

        with patch.object(
            integration.checkpoint_mgr,
            "list_debates_with_checkpoints",
            new_callable=AsyncMock,
            return_value=mock_checkpoints,
        ):
            result = await integration.list_checkpoints()

        # Should filter to current debate
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_checkpoints_no_filter(self, integration):
        """Should list all checkpoints when no debate_id set."""
        integration._current_debate_id = None

        mock_checkpoints = [
            {"debate_id": "debate-1", "phase": "debate"},
            {"debate_id": "debate-2", "phase": "design"},
        ]

        with patch.object(
            integration.checkpoint_mgr,
            "list_debates_with_checkpoints",
            new_callable=AsyncMock,
            return_value=mock_checkpoints,
        ):
            result = await integration.list_checkpoints()

        assert len(result) == 2


# =============================================================================
# Test: State Management
# =============================================================================


class TestStateManagement:
    """Tests for state tracking and management."""

    def test_set_cycle(self, integration):
        """Should set current cycle."""
        integration.set_cycle(5)

        assert integration._current_cycle == 5

    def test_set_debate_id(self, integration):
        """Should set current debate ID."""
        integration.set_debate_id("debate-123")

        assert integration._current_debate_id == "debate-123"

    def test_get_agent_weights_copy(self, integration):
        """Should return a copy of agent weights."""
        integration._agent_weights = {"claude": 0.9, "gemini": 0.8}

        weights = integration.get_agent_weights()
        weights["grok"] = 0.7

        # Original should not be modified
        assert "grok" not in integration._agent_weights


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_analyze_empty_debate_result(self, integration):
        """Should handle empty debate result gracefully."""
        empty_result = DebateResult(
            id="empty",
            task="",
            votes=[],
            messages=[],
        )

        analysis = await integration.analyze_debate(empty_result)

        assert isinstance(analysis, BeliefAnalysis)

    @pytest.mark.asyncio
    async def test_probe_empty_agents_list(self, integration):
        """Should handle empty agents list."""
        weights = await integration.probe_agents([])

        assert weights == {}

    @pytest.mark.asyncio
    async def test_check_staleness_empty_claims(self, integration):
        """Should handle empty claims list."""
        report = await integration.check_staleness([], ["file.py"])

        assert len(report.stale_claims) == 0

    @pytest.mark.asyncio
    async def test_check_staleness_empty_changed_files(self, integration):
        """Should handle empty changed files list."""
        claims = [
            TypedClaim(
                claim_id="c1",
                claim_type=ClaimType.ASSERTION,
                statement="Test claim referencing file.py",
                author="agent1",
                confidence=0.9,
            )
        ]

        report = await integration.check_staleness(claims, [])

        assert len(report.stale_claims) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_with_explicit_ids(self, integration):
        """Should use explicit debate_id and cycle if provided."""
        with patch.object(
            integration.checkpoint_mgr,
            "create_checkpoint",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_checkpoint = MagicMock()
            mock_checkpoint.checkpoint_id = "cp-explicit"
            mock_create.return_value = mock_checkpoint

            await integration.checkpoint(
                phase="debate",
                state={},
                debate_id="explicit-debate",
                cycle=99,
            )

        call_args = mock_create.call_args
        assert call_args.kwargs["debate_id"] == "explicit-debate"

    def test_file_extraction_various_extensions(self, integration):
        """Should extract files with various extensions."""
        claim = TypedClaim(
            claim_id="c1",
            claim_type=ClaimType.ASSERTION,
            statement="Check main.py, config.yaml, test.tsx, and data.json",
            author="agent1",
            confidence=0.9,
        )

        files = integration._extract_file_references(claim)

        assert "main.py" in files
        assert "config.yaml" in files
        assert "test.tsx" in files
        assert "data.json" in files

    def test_file_extraction_with_paths(self, integration):
        """Should extract file paths with directories."""
        claim = TypedClaim(
            claim_id="c1",
            claim_type=ClaimType.ASSERTION,
            statement="Modify aragora/server/handlers.py and tests/test_handlers.py",
            author="agent1",
            confidence=0.9,
        )

        files = integration._extract_file_references(claim)

        assert "aragora/server/handlers.py" in files
        assert "tests/test_handlers.py" in files

    @pytest.mark.asyncio
    async def test_full_analysis_no_claims_or_files(self, integration, mock_debate_result):
        """Should handle missing claims and files gracefully."""
        result = await integration.full_post_debate_analysis(
            mock_debate_result,
            claims_kernel=None,
            changed_files=None,
        )

        assert result["staleness"] is None

    @pytest.mark.asyncio
    async def test_resolve_deadlock_uses_debate_id_from_state(self, integration):
        """Should use current debate ID when resolving deadlock."""
        integration.set_debate_id("stored-debate-id")

        mock_node = MagicMock(spec=BeliefNode)
        mock_node.claim_id = "claim-1"
        mock_node.claim_statement = "Test"
        mock_node.author = "agent1"
        mock_node.posterior = MagicMock()
        mock_node.posterior.entropy = 0.8
        mock_node.centrality = 0.5

        with patch.object(
            integration.counterfactual,
            "create_and_run_branches",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_create:
            await integration.resolve_deadlock([mock_node])

        call_args = mock_create.call_args
        assert call_args.kwargs["debate_id"] == "stored-debate-id"


# =============================================================================
# Test: Conflict Resolution
# =============================================================================


class TestConflictResolution:
    """Tests for conflict resolution between systems."""

    @pytest.mark.asyncio
    async def test_analysis_updates_state_atomically(self, integration, mock_debate_result):
        """State updates should be atomic during analysis."""
        # Run multiple analyses
        analysis1 = await integration.analyze_debate(mock_debate_result)

        # Modify debate result
        modified_result = DebateResult(
            id="modified-debate",
            task="Different task",
            votes=[Vote(agent="a", choice="b", reasoning="r", confidence=0.5)],
            messages=[],
        )

        analysis2 = await integration.analyze_debate(modified_result)

        # State should reflect latest analysis
        assert integration._current_debate_id == "modified-debate"
        assert analysis1.network is not analysis2.network

    @pytest.mark.asyncio
    async def test_weight_updates_are_complete(self, integration, mock_agents):
        """Agent weights should be fully updated."""

        async def mock_run_agent(agent, prompt):
            return "Response"

        mock_report = MagicMock()
        mock_report.vulnerability_rate = 0.1
        mock_report.critical_count = 0

        with patch.object(integration.prober, "probe_agent", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = mock_report

            weights = await integration.probe_agents(
                mock_agents,
                run_agent_fn=mock_run_agent,
            )

        # All agents should have weights
        assert len(weights) == len(mock_agents)
        for agent in mock_agents:
            assert agent.name in weights

    @pytest.mark.asyncio
    async def test_checkpoint_phase_ordering(self, integration):
        """Checkpoints should maintain phase ordering."""
        integration.set_debate_id("test")

        phases = ["debate", "design", "implement", "verify"]

        with patch.object(
            integration.checkpoint_mgr,
            "create_checkpoint",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_checkpoint = MagicMock()
            mock_checkpoint.checkpoint_id = "cp"
            mock_create.return_value = mock_checkpoint

            for i, phase in enumerate(phases):
                integration.set_cycle(i + 1)
                await integration.checkpoint(phase=phase, state={})

        # Checkpoints should be in order
        assert len(integration._phase_checkpoints) == 4
        for i, pc in enumerate(integration._phase_checkpoints):
            assert pc.phase == phases[i]
            assert pc.cycle == i + 1


class TestPhaseCheckpointDataclass:
    """Tests for PhaseCheckpoint dataclass."""

    def test_phase_checkpoint_creation(self):
        """Should create phase checkpoint with all fields."""
        mock_checkpoint = MagicMock(spec=DebateCheckpoint)

        pc = PhaseCheckpoint(
            phase="implement",
            cycle=3,
            state={"code": "test"},
            checkpoint=mock_checkpoint,
        )

        assert pc.phase == "implement"
        assert pc.cycle == 3
        assert pc.state == {"code": "test"}
        assert pc.checkpoint is mock_checkpoint
        assert isinstance(pc.created_at, datetime)
