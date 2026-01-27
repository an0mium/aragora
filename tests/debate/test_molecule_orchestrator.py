"""Tests for molecule-based debate orchestration."""

import pytest

from aragora.debate.molecule_orchestrator import (
    AgentProfileWrapper,
    MoleculeExecutionResult,
    MoleculeOrchestrator,
    get_molecule_orchestrator,
    reset_molecule_orchestrator,
)
from aragora.debate.molecules import MoleculeStatus, MoleculeType


class SimpleAgent:
    """Simple agent for testing."""

    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model


class TestAgentProfileWrapper:
    """Tests for AgentProfileWrapper."""

    def test_create_from_basic_agent(self):
        """Test creating profile from basic agent."""
        agent = SimpleAgent("test_agent")
        profile = AgentProfileWrapper.from_agent(agent)

        assert profile.name == "test_agent"
        assert "reasoning" in profile.capabilities
        assert profile.elo_rating == 1200.0
        assert profile.availability == 1.0

    def test_claude_capabilities(self):
        """Test Claude agents get correct capabilities."""
        agent = SimpleAgent("claude", model="claude-3")
        profile = AgentProfileWrapper.from_agent(agent)

        assert "analysis" in profile.capabilities
        assert "synthesis" in profile.capabilities
        assert "creativity" in profile.capabilities

    def test_gpt_capabilities(self):
        """Test GPT agents get correct capabilities."""
        agent = SimpleAgent("gpt4", model="openai-gpt-4")
        profile = AgentProfileWrapper.from_agent(agent)

        assert "quality_assessment" in profile.capabilities
        assert "analysis" in profile.capabilities

    def test_gemini_capabilities(self):
        """Test Gemini agents get research capability."""
        agent = SimpleAgent("gemini", model="google-gemini")
        profile = AgentProfileWrapper.from_agent(agent)

        assert "research" in profile.capabilities
        assert "analysis" in profile.capabilities

    def test_grok_capabilities(self):
        """Test Grok agents get creativity capability."""
        agent = SimpleAgent("grok")
        profile = AgentProfileWrapper.from_agent(agent)

        assert "creativity" in profile.capabilities

    def test_deepseek_capabilities(self):
        """Test DeepSeek agents get correct capabilities."""
        agent = SimpleAgent("deepseek", model="deepseek-v2")
        profile = AgentProfileWrapper.from_agent(agent)

        assert "analysis" in profile.capabilities
        assert "synthesis" in profile.capabilities


class TestMoleculeOrchestrator:
    """Tests for MoleculeOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return MoleculeOrchestrator(max_attempts=3)

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            SimpleAgent("claude"),
            SimpleAgent("gpt4", model="openai-gpt-4"),
            SimpleAgent("gemini", model="google-gemini"),
        ]

    def test_register_agents(self, orchestrator, agents):
        """Test agent registration."""
        orchestrator.register_agents(agents)

        assert orchestrator.get_agent_profile("claude") is not None
        assert orchestrator.get_agent_profile("gpt4") is not None
        assert orchestrator.get_agent_profile("gemini") is not None
        assert orchestrator.get_agent_profile("nonexistent") is None

    @pytest.mark.asyncio
    async def test_create_round_molecules(self, orchestrator, agents):
        """Test creating molecules for a debate round."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Design a rate limiter",
            agents=agents,
        )

        # Should create:
        # - 3 proposals (one per agent)
        # - 6 critiques (each agent critiques 2 others)
        # - 1 synthesis
        assert len(molecules) == 10

        # Check types
        proposals = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL]
        critiques = [m for m in molecules if m.molecule_type == MoleculeType.CRITIQUE]
        synthesis = [m for m in molecules if m.molecule_type == MoleculeType.SYNTHESIS]

        assert len(proposals) == 3
        assert len(critiques) == 6
        assert len(synthesis) == 1

    @pytest.mark.asyncio
    async def test_create_round_molecules_without_synthesis(self, orchestrator, agents):
        """Test creating molecules without synthesis phase."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Design a rate limiter",
            agents=agents,
            include_synthesis=False,
        )

        # No synthesis molecule
        synthesis = [m for m in molecules if m.molecule_type == MoleculeType.SYNTHESIS]
        assert len(synthesis) == 0

    @pytest.mark.asyncio
    async def test_create_vote_molecules(self, orchestrator, agents):
        """Test creating vote molecules."""
        orchestrator.register_agents(agents)

        molecules = await orchestrator.create_vote_molecules(
            debate_id="test_debate",
            round_number=1,
            agents=agents,
        )

        assert len(molecules) == 3
        assert all(m.molecule_type == MoleculeType.VOTE for m in molecules)

    @pytest.mark.asyncio
    async def test_assign_molecule_by_name(self, orchestrator, agents):
        """Test assigning molecule to specific agent."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        proposal = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL][0]
        success = await orchestrator.assign_molecule(proposal.molecule_id, "claude")

        assert success
        mol = orchestrator.get_molecule(proposal.molecule_id)
        assert mol.assigned_agent == "claude"
        assert mol.status == MoleculeStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_assign_molecule_auto_select(self, orchestrator, agents):
        """Test auto-selecting best agent for molecule."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        proposal = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL][0]
        success = await orchestrator.assign_molecule(proposal.molecule_id)

        assert success
        mol = orchestrator.get_molecule(proposal.molecule_id)
        assert mol.assigned_agent is not None

    @pytest.mark.asyncio
    async def test_complete_molecule(self, orchestrator, agents):
        """Test completing a molecule."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        proposal = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL][0]
        await orchestrator.assign_molecule(proposal.molecule_id, "claude")
        await orchestrator.start_molecule(proposal.molecule_id)

        result = await orchestrator.complete_molecule(
            proposal.molecule_id,
            {"proposal": "Use token bucket algorithm"},
        )

        assert result.success
        assert result.output["proposal"] == "Use token bucket algorithm"
        assert result.agent == "claude"

        mol = orchestrator.get_molecule(proposal.molecule_id)
        assert mol.status == MoleculeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fail_molecule(self, orchestrator, agents):
        """Test failing a molecule."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        proposal = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL][0]
        await orchestrator.assign_molecule(proposal.molecule_id, "claude")
        await orchestrator.start_molecule(proposal.molecule_id)

        result = await orchestrator.fail_molecule(
            proposal.molecule_id,
            "Agent timeout",
        )

        assert not result.success
        assert result.error == "Agent timeout"

        mol = orchestrator.get_molecule(proposal.molecule_id)
        assert mol.status == MoleculeStatus.FAILED
        assert mol.can_retry()  # Still has attempts

    @pytest.mark.asyncio
    async def test_recover_failed_molecules(self, orchestrator, agents):
        """Test recovering failed molecules."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        # Use a VOTE molecule since all agents can handle it (only requires "reasoning")
        vote_molecules = await orchestrator.create_vote_molecules(
            debate_id="test_debate",
            round_number=1,
            agents=agents,
        )

        vote = vote_molecules[0]
        await orchestrator.assign_molecule(vote.molecule_id, "claude")
        await orchestrator.start_molecule(vote.molecule_id)
        await orchestrator.fail_molecule(vote.molecule_id, "Timeout")

        reassigned = await orchestrator.recover_failed_molecules("test_debate")

        # Should reassign to a different agent (gpt4 or gemini)
        vote_reassigned = [(m, a) for m, a in reassigned if m == vote.molecule_id]
        assert len(vote_reassigned) == 1
        mol_id, new_agent = vote_reassigned[0]
        assert mol_id == vote.molecule_id
        assert new_agent != "claude"

    @pytest.mark.asyncio
    async def test_get_progress(self, orchestrator, agents):
        """Test getting debate progress."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        # Complete some molecules
        proposals = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL]
        for p in proposals[:2]:
            await orchestrator.assign_molecule(p.molecule_id, "claude")
            await orchestrator.start_molecule(p.molecule_id)
            await orchestrator.complete_molecule(p.molecule_id, {"result": "done"})

        progress = orchestrator.get_progress("test_debate")

        assert progress["debate_id"] == "test_debate"
        assert progress["total"] == 10
        assert progress["completed"] == 2
        assert progress["progress"] == 0.2

    @pytest.mark.asyncio
    async def test_pending_molecules_respects_dependencies(self, orchestrator, agents):
        """Test that pending molecules respect dependencies."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        # Initially only proposals should be pending (no dependencies)
        pending = orchestrator.get_pending_molecules("test_debate")
        assert all(m.molecule_type == MoleculeType.PROPOSAL for m in pending)
        assert len(pending) == 3

        # Complete all proposals
        proposals = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL]
        for p in proposals:
            await orchestrator.assign_molecule(p.molecule_id, "claude")
            await orchestrator.start_molecule(p.molecule_id)
            await orchestrator.complete_molecule(p.molecule_id, {"result": "done"})

        # Now critiques should be pending
        pending = orchestrator.get_pending_molecules("test_debate")
        assert len(pending) == 6
        assert all(m.molecule_type == MoleculeType.CRITIQUE for m in pending)

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self, orchestrator, agents):
        """Test checkpointing and restoring state."""
        molecules = await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        # Complete some work
        proposals = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL]
        for p in proposals[:2]:
            await orchestrator.assign_molecule(p.molecule_id, "claude")
            await orchestrator.start_molecule(p.molecule_id)
            await orchestrator.complete_molecule(p.molecule_id, {"result": "done"})

        # Checkpoint
        state = orchestrator.to_checkpoint_state("test_debate")

        assert state["debate_id"] == "test_debate"
        assert len(state["molecules"]) == 10
        assert state["progress"]["completed"] == 2

        # Create new orchestrator and restore
        new_orchestrator = MoleculeOrchestrator()
        new_orchestrator.restore_from_checkpoint(state)

        progress = new_orchestrator.get_progress("test_debate")
        assert progress["completed"] == 2
        assert progress["total"] == 10

    @pytest.mark.asyncio
    async def test_clear_debate(self, orchestrator, agents):
        """Test clearing debate molecules."""
        await orchestrator.create_round_molecules(
            debate_id="test_debate",
            round_number=1,
            task="Test task",
            agents=agents,
        )

        assert len(orchestrator.get_debate_molecules("test_debate")) == 10

        orchestrator.clear_debate("test_debate")

        assert len(orchestrator.get_debate_molecules("test_debate")) == 0


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_molecule_orchestrator()

    def test_get_molecule_orchestrator(self):
        """Test singleton getter."""
        orch1 = get_molecule_orchestrator()
        orch2 = get_molecule_orchestrator()

        assert orch1 is orch2

    def test_reset_molecule_orchestrator(self):
        """Test singleton reset."""
        orch1 = get_molecule_orchestrator()
        reset_molecule_orchestrator()
        orch2 = get_molecule_orchestrator()

        assert orch1 is not orch2


class TestMoleculeExecutionResult:
    """Tests for MoleculeExecutionResult."""

    def test_success_result(self):
        """Test successful execution result."""
        result = MoleculeExecutionResult(
            molecule_id="mol-123",
            success=True,
            output={"data": "value"},
            agent="claude",
            duration_seconds=1.5,
        )

        assert result.molecule_id == "mol-123"
        assert result.success
        assert result.output["data"] == "value"
        assert result.agent == "claude"
        assert result.duration_seconds == 1.5
        assert result.error is None

    def test_failure_result(self):
        """Test failed execution result."""
        result = MoleculeExecutionResult(
            molecule_id="mol-123",
            success=False,
            error="Timeout after 30s",
        )

        assert not result.success
        assert result.error == "Timeout after 30s"
        assert result.output is None
