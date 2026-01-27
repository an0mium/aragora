"""Tests for molecule tracking integration with debate phases."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.molecules import (
    Molecule,
    MoleculeStatus,
    MoleculeTracker,
    MoleculeType,
    MOLECULE_CAPABILITIES,
    create_round_molecules,
)


class TestMolecule:
    """Tests for the Molecule dataclass."""

    def test_create_molecule(self):
        """Test creating a molecule with class method."""
        mol = Molecule.create(
            debate_id="debate-123",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
            input_data={"task": "Test task"},
        )

        assert mol.molecule_id.startswith("mol-")
        assert mol.debate_id == "debate-123"
        assert mol.molecule_type == MoleculeType.PROPOSAL
        assert mol.round_number == 0
        assert mol.status == MoleculeStatus.PENDING
        assert mol.input_data == {"task": "Test task"}

    def test_molecule_capabilities_auto_set(self):
        """Test that capabilities are automatically set based on type."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.CRITIQUE,
            round_number=1,
        )

        assert mol.required_capabilities == MOLECULE_CAPABILITIES[MoleculeType.CRITIQUE]

    def test_molecule_assign(self):
        """Test assigning a molecule to an agent."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )

        mol.assign("claude-opus")

        assert mol.assigned_agent == "claude-opus"
        assert mol.status == MoleculeStatus.ASSIGNED
        assert mol.attempts == 1
        assert "claude-opus" in mol.assignment_history

    def test_molecule_start(self):
        """Test starting a molecule."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )
        mol.assign("agent")
        mol.start()

        assert mol.status == MoleculeStatus.IN_PROGRESS
        assert mol.started_at is not None

    def test_molecule_complete(self):
        """Test completing a molecule."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )
        mol.assign("agent")
        mol.start()

        output = {"proposal": "Test proposal", "chars": 100}
        mol.complete(output)

        assert mol.status == MoleculeStatus.COMPLETED
        assert mol.output_data == output
        assert mol.completed_at is not None
        # Agent affinity should increase on success
        assert mol.agent_affinity.get("agent", 0) > 0.5

    def test_molecule_fail(self):
        """Test failing a molecule."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )
        mol.assign("agent")
        mol.start()

        mol.fail("Timeout error")

        assert mol.status == MoleculeStatus.FAILED
        assert mol.error_message == "Timeout error"
        # Agent affinity should decrease on failure
        assert mol.agent_affinity.get("agent", 0.5) < 0.5
        # Agent should be unassigned
        assert mol.assigned_agent is None

    def test_molecule_can_retry(self):
        """Test retry limit checking."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )

        # Should be able to retry with 0 attempts
        assert mol.can_retry() is True

        # Fail max_attempts times
        for _ in range(mol.max_attempts):
            mol.assign("agent")
            mol.start()
            mol.fail("error")

        # Should not be able to retry after max attempts
        assert mol.can_retry() is False

    def test_molecule_to_dict(self):
        """Test serialization to dictionary."""
        mol = Molecule.create(
            debate_id="test",
            molecule_type=MoleculeType.CRITIQUE,
            round_number=1,
            input_data={"target": "agent1"},
        )

        data = mol.to_dict()

        assert data["debate_id"] == "test"
        assert data["molecule_type"] == "critique"
        assert data["round_number"] == 1
        assert data["status"] == "pending"

    def test_molecule_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "molecule_id": "mol-test123",
            "debate_id": "test",
            "molecule_type": "revision",
            "round_number": 2,
            "status": "completed",
            "input_data": {"agent": "test"},
            "output_data": {"result": "success"},
        }

        mol = Molecule.from_dict(data)

        assert mol.molecule_id == "mol-test123"
        assert mol.molecule_type == MoleculeType.REVISION
        assert mol.status == MoleculeStatus.COMPLETED


class TestMoleculeTracker:
    """Tests for the MoleculeTracker class."""

    def test_create_molecule(self):
        """Test creating a molecule through tracker."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule(
            debate_id="debate-1",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
            input_data={"task": "Test"},
        )

        assert mol.molecule_id in tracker._molecules
        assert "debate-1" in tracker._debate_molecules

    def test_get_molecule(self):
        """Test retrieving a molecule by ID."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule(
            debate_id="test",
            molecule_type=MoleculeType.PROPOSAL,
            round_number=0,
        )

        retrieved = tracker.get_molecule(mol.molecule_id)
        assert retrieved == mol

    def test_get_molecule_not_found(self):
        """Test retrieving non-existent molecule."""
        tracker = MoleculeTracker()

        retrieved = tracker.get_molecule("nonexistent")
        assert retrieved is None

    def test_get_debate_molecules(self):
        """Test getting all molecules for a debate."""
        tracker = MoleculeTracker()

        mol1 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol2 = tracker.create_molecule("debate-1", MoleculeType.CRITIQUE, 0)
        tracker.create_molecule("debate-2", MoleculeType.PROPOSAL, 0)

        debate_mols = tracker.get_debate_molecules("debate-1")

        assert len(debate_mols) == 2
        assert mol1 in debate_mols
        assert mol2 in debate_mols

    def test_get_pending_molecules(self):
        """Test getting pending molecules."""
        tracker = MoleculeTracker()

        mol1 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol2 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        # Complete one molecule
        tracker._molecules[mol1.molecule_id].status = MoleculeStatus.COMPLETED

        pending = tracker.get_pending_molecules("debate-1")

        assert len(pending) == 1
        assert mol2 in pending

    def test_pending_molecules_dependency_check(self):
        """Test that pending molecules check dependencies."""
        tracker = MoleculeTracker()

        mol1 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol2 = tracker.create_molecule(
            "debate-1",
            MoleculeType.CRITIQUE,
            0,
            depends_on=[mol1.molecule_id],
        )

        # mol2 should be blocked because mol1 is not completed
        pending = tracker.get_pending_molecules("debate-1")

        assert mol1 in pending
        assert mol2 not in pending
        assert mol2.status == MoleculeStatus.BLOCKED

        # To complete mol1, we need to assign, start, and complete it
        agent = MagicMock()
        agent.name = "agent"
        agent.capabilities = {"reasoning", "creativity"}
        tracker.assign_molecule(mol1.molecule_id, agent)
        tracker.start_molecule(mol1.molecule_id)
        tracker.complete_molecule(mol1.molecule_id, {"result": "done"})

        # Now mol2 should be pending (unblocked)
        pending = tracker.get_pending_molecules("debate-1")
        assert mol2 in pending

    def test_assign_molecule(self):
        """Test assigning a molecule to an agent."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        # Create mock agent profile
        agent = MagicMock()
        agent.name = "claude"
        agent.capabilities = {"reasoning", "creativity", "analysis"}

        result = tracker.assign_molecule(mol.molecule_id, agent)

        assert result is True
        assert mol.assigned_agent == "claude"
        assert tracker._agent_workload["claude"] == 1

    def test_assign_molecule_missing_capabilities(self):
        """Test that assignment fails without required capabilities."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        # Agent without required capabilities
        agent = MagicMock()
        agent.name = "weak-agent"
        agent.capabilities = set()  # No capabilities

        result = tracker.assign_molecule(mol.molecule_id, agent)

        assert result is False
        assert mol.assigned_agent is None

    def test_find_best_agent(self):
        """Test finding the best agent for a molecule."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        # Create mock agents
        agent1 = MagicMock()
        agent1.name = "agent1"
        agent1.capabilities = {"reasoning", "creativity"}
        agent1.elo_rating = 1500
        agent1.availability = 1.0

        agent2 = MagicMock()
        agent2.name = "agent2"
        agent2.capabilities = {"reasoning", "creativity"}
        agent2.elo_rating = 1800
        agent2.availability = 1.0

        best = tracker.find_best_agent(mol, [agent1, agent2])

        # agent2 should be selected due to higher ELO
        assert best == agent2

    def test_start_molecule(self):
        """Test starting a molecule through tracker."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        agent = MagicMock()
        agent.name = "agent"
        agent.capabilities = {"reasoning", "creativity"}
        tracker.assign_molecule(mol.molecule_id, agent)

        result = tracker.start_molecule(mol.molecule_id)

        assert result is True
        assert mol.status == MoleculeStatus.IN_PROGRESS

    def test_complete_molecule(self):
        """Test completing a molecule through tracker."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        agent = MagicMock()
        agent.name = "agent"
        agent.capabilities = {"reasoning", "creativity"}
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)

        result = tracker.complete_molecule(mol.molecule_id, {"output": "done"})

        assert result is True
        assert mol.status == MoleculeStatus.COMPLETED
        # Workload should be decremented
        assert tracker._agent_workload.get("agent", 0) == 0

    def test_fail_molecule(self):
        """Test failing a molecule through tracker."""
        tracker = MoleculeTracker()

        mol = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)

        agent = MagicMock()
        agent.name = "agent"
        agent.capabilities = {"reasoning", "creativity"}
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)

        result = tracker.fail_molecule(mol.molecule_id, "Timeout")

        assert result is True
        assert mol.status == MoleculeStatus.FAILED
        assert tracker._agent_workload.get("agent", 0) == 0

    def test_get_progress(self):
        """Test getting progress summary."""
        tracker = MoleculeTracker()

        mol1 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol2 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol3 = tracker.create_molecule("debate-1", MoleculeType.CRITIQUE, 0)

        # Complete one
        agent = MagicMock()
        agent.name = "agent"
        agent.capabilities = {"reasoning", "creativity", "analysis", "quality_assessment"}
        tracker.assign_molecule(mol1.molecule_id, agent)
        tracker.start_molecule(mol1.molecule_id)
        tracker.complete_molecule(mol1.molecule_id, {})

        progress = tracker.get_progress("debate-1")

        assert progress["total"] == 3
        assert progress["completed"] == 1
        assert progress["progress"] == 1 / 3
        assert "by_status" in progress
        assert "by_type" in progress

    def test_clear_debate(self):
        """Test clearing molecules for a completed debate."""
        tracker = MoleculeTracker()

        mol1 = tracker.create_molecule("debate-1", MoleculeType.PROPOSAL, 0)
        mol2 = tracker.create_molecule("debate-2", MoleculeType.PROPOSAL, 0)

        tracker.clear_debate("debate-1")

        assert "debate-1" not in tracker._debate_molecules
        assert mol1.molecule_id not in tracker._molecules
        # debate-2 should still exist
        assert mol2.molecule_id in tracker._molecules


class TestCreateRoundMolecules:
    """Tests for the create_round_molecules helper function."""

    def test_creates_proposal_molecules(self):
        """Test that proposal molecules are created for each agent."""
        tracker = MoleculeTracker()

        molecules = create_round_molecules(
            tracker=tracker,
            debate_id="debate-1",
            round_number=0,
            agent_count=3,
            task="Test task",
        )

        proposals = [m for m in molecules if m.molecule_type == MoleculeType.PROPOSAL]
        assert len(proposals) == 3

    def test_creates_critique_molecules(self):
        """Test that critique molecules are created with dependencies."""
        tracker = MoleculeTracker()

        molecules = create_round_molecules(
            tracker=tracker,
            debate_id="debate-1",
            round_number=0,
            agent_count=3,
            task="Test task",
        )

        critiques = [m for m in molecules if m.molecule_type == MoleculeType.CRITIQUE]
        # 3 agents * 2 critiques each (each agent critiques 2 others)
        assert len(critiques) == 6

        # Each critique should depend on a proposal
        for critique in critiques:
            assert len(critique.depends_on) == 1

    def test_creates_synthesis_molecule(self):
        """Test that synthesis molecule is created."""
        tracker = MoleculeTracker()

        molecules = create_round_molecules(
            tracker=tracker,
            debate_id="debate-1",
            round_number=0,
            agent_count=3,
            task="Test task",
        )

        synthesis = [m for m in molecules if m.molecule_type == MoleculeType.SYNTHESIS]
        assert len(synthesis) == 1

        # Synthesis should depend on all critiques
        critique_ids = [
            m.molecule_id for m in molecules if m.molecule_type == MoleculeType.CRITIQUE
        ]
        assert set(synthesis[0].depends_on) == set(critique_ids)


class TestPhaseIntegration:
    """Tests for molecule integration with debate phases."""

    @pytest.mark.asyncio
    async def test_proposal_phase_molecule_tracking(self):
        """Test that ProposalPhase tracks molecules correctly."""
        from aragora.debate.phases.proposal_phase import ProposalPhase

        tracker = MoleculeTracker()

        # Create mock context
        ctx = MagicMock()
        ctx.debate_id = "test-debate"
        ctx.env = MagicMock()
        ctx.env.task = "Test task"
        ctx.proposals = {}
        ctx.proposers = []
        ctx.agents = []
        ctx.context_messages = []
        ctx.add_message = MagicMock()
        ctx.record_agent_failure = MagicMock()

        # Create mock proposer
        proposer = MagicMock()
        proposer.name = "claude"
        ctx.proposers = [proposer]
        ctx.agents = [proposer]

        # Create phase with molecule tracker
        phase = ProposalPhase(
            molecule_tracker=tracker,
            build_proposal_prompt=lambda agent: "Test prompt",
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=None,
        )

        # Execute should create and track molecules
        await phase.execute(ctx)

        # Check that molecules were created
        molecules = tracker.get_debate_molecules("test-debate")
        assert len(molecules) == 1
        assert molecules[0].molecule_type == MoleculeType.PROPOSAL

    @pytest.mark.asyncio
    async def test_critique_generator_molecule_tracking(self):
        """Test that CritiqueGenerator tracks molecules correctly."""
        from aragora.debate.phases.critique_generator import CritiqueGenerator

        tracker = MoleculeTracker()

        # Create mock context
        ctx = MagicMock()
        ctx.debate_id = "test-debate"
        ctx.env = MagicMock()
        ctx.env.task = "Test task"
        ctx.proposals = {"agent1": "Proposal 1"}
        ctx.result = MagicMock()
        ctx.result.critiques = []
        ctx.context_messages = []
        ctx.add_message = MagicMock()
        ctx.record_agent_failure = MagicMock()

        # Create mock critic
        critic = MagicMock()
        critic.name = "critic1"

        # Create mock critique result
        mock_critique = MagicMock()
        mock_critique.issues = ["Issue 1"]
        mock_critique.suggestions = ["Suggestion 1"]
        mock_critique.severity = 0.5
        mock_critique.to_prompt = MagicMock(return_value="Critique text")

        generator = CritiqueGenerator(
            molecule_tracker=tracker,
            critique_with_agent=AsyncMock(return_value=mock_critique),
            with_timeout=None,
        )

        # Execute should create and track molecules
        await generator.execute_critique_phase(
            ctx=ctx,
            critics=[critic],
            round_num=0,
            partial_messages=[],
            partial_critiques=[],
        )

        # Check that molecules were created
        molecules = tracker.get_debate_molecules("test-debate")
        assert len(molecules) == 1
        assert molecules[0].molecule_type == MoleculeType.CRITIQUE

    @pytest.mark.asyncio
    async def test_revision_generator_molecule_tracking(self):
        """Test that RevisionGenerator tracks molecules correctly."""
        from aragora.debate.phases.revision_phase import RevisionGenerator

        tracker = MoleculeTracker()

        # Create mock context
        ctx = MagicMock()
        ctx.debate_id = "test-debate"
        ctx.env = MagicMock()
        ctx.env.task = "Test task"
        ctx.proposals = {"agent1": "Original proposal"}
        ctx.proposers = []
        ctx.result = MagicMock()
        ctx.result.messages = []
        ctx.context_messages = []
        ctx.add_message = MagicMock()
        ctx.loop_id = None

        # Create mock proposer
        proposer = MagicMock()
        proposer.name = "agent1"
        ctx.proposers = [proposer]

        # Create mock critique
        critique = MagicMock()
        critique.target_agent = "agent1"

        generator = RevisionGenerator(
            molecule_tracker=tracker,
            generate_with_agent=AsyncMock(return_value="Revised proposal"),
            build_revision_prompt=lambda agent, proposal, critiques, round_num: "Prompt",
            with_timeout=None,
        )

        # Execute should create and track molecules
        await generator.execute_revision_phase(
            ctx=ctx,
            round_num=1,
            all_critiques=[critique],
            partial_messages=[],
        )

        # Check that molecules were created
        molecules = tracker.get_debate_molecules("test-debate")
        assert len(molecules) == 1
        assert molecules[0].molecule_type == MoleculeType.REVISION
