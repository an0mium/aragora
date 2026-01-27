"""Tests for Gastown-inspired patterns (Beads, Convoys, Hook Queue, Agent Roles, Molecules)."""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.nomic.beads import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
    reset_bead_store,
)
from aragora.nomic.convoys import (
    Convoy,
    ConvoyManager,
    ConvoyPriority,
    ConvoyStatus,
    reset_convoy_manager,
)
from aragora.nomic.hook_queue import (
    HookEntry,
    HookEntryStatus,
    HookQueue,
    HookQueueRegistry,
    reset_hook_queue_registry,
)
from aragora.nomic.agent_roles import (
    AgentHierarchy,
    AgentRole,
    RoleAssignment,
    RoleBasedRouter,
    RoleCapability,
    reset_agent_hierarchy,
)
from aragora.nomic.molecules import (
    Molecule,
    MoleculeEngine,
    MoleculeStatus,
    MoleculeStep,
    StepStatus,
    reset_molecule_engine,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_singletons():
    """Reset all singleton instances before and after each test."""
    reset_bead_store()
    reset_convoy_manager()
    reset_hook_queue_registry()
    reset_agent_hierarchy()
    reset_molecule_engine()
    yield
    reset_bead_store()
    reset_convoy_manager()
    reset_hook_queue_registry()
    reset_agent_hierarchy()
    reset_molecule_engine()


# =============================================================================
# Bead Tests
# =============================================================================


class TestBead:
    """Tests for Bead dataclass."""

    def test_create_bead(self):
        """Test creating a bead."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Test task",
            description="Test description",
        )
        assert bead.id is not None
        assert bead.bead_type == BeadType.TASK
        assert bead.status == BeadStatus.PENDING
        assert bead.title == "Test task"
        assert bead.description == "Test description"

    def test_bead_serialization(self):
        """Test bead serialization and deserialization."""
        bead = Bead.create(
            bead_type=BeadType.ISSUE,
            title="Bug report",
            priority=BeadPriority.HIGH,
        )
        data = bead.to_dict()
        restored = Bead.from_dict(data)

        assert restored.id == bead.id
        assert restored.bead_type == bead.bead_type
        assert restored.title == bead.title
        assert restored.priority == bead.priority

    def test_bead_dependencies(self):
        """Test bead dependency checking."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Dependent task",
            dependencies=["dep-1", "dep-2"],
        )
        assert not bead.can_start({"dep-1"})
        assert bead.can_start({"dep-1", "dep-2"})
        assert bead.can_start({"dep-1", "dep-2", "dep-3"})

    def test_bead_terminal_states(self):
        """Test terminal state detection."""
        bead = Bead.create(BeadType.TASK, "Test")
        assert not bead.is_terminal()

        bead.status = BeadStatus.COMPLETED
        assert bead.is_terminal()

        bead.status = BeadStatus.FAILED
        assert bead.is_terminal()


class TestBeadStore:
    """Tests for BeadStore."""

    @pytest.mark.asyncio
    async def test_create_and_get_bead(self, temp_dir, reset_singletons):
        """Test creating and retrieving a bead."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        bead = Bead.create(BeadType.TASK, "Test task")
        bead_id = await store.create(bead)

        retrieved = await store.get(bead_id)
        assert retrieved is not None
        assert retrieved.title == "Test task"

    @pytest.mark.asyncio
    async def test_claim_bead(self, temp_dir, reset_singletons):
        """Test claiming a bead."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        bead = Bead.create(BeadType.TASK, "Claimable task")
        await store.create(bead)

        success = await store.claim(bead.id, "agent-001")
        assert success

        updated = await store.get(bead.id)
        assert updated.status == BeadStatus.CLAIMED
        assert updated.claimed_by == "agent-001"

    @pytest.mark.asyncio
    async def test_list_by_status(self, temp_dir, reset_singletons):
        """Test listing beads by status."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        # Create multiple beads
        for i in range(3):
            bead = Bead.create(BeadType.TASK, f"Task {i}")
            await store.create(bead)

        pending = await store.list_by_status(BeadStatus.PENDING)
        assert len(pending) == 3


# =============================================================================
# Convoy Tests
# =============================================================================


class TestConvoy:
    """Tests for Convoy dataclass."""

    def test_create_convoy(self):
        """Test creating a convoy."""
        convoy = Convoy.create(
            title="Test convoy",
            bead_ids=["bead-1", "bead-2"],
            priority=ConvoyPriority.HIGH,
        )
        assert convoy.id is not None
        assert convoy.title == "Test convoy"
        assert len(convoy.bead_ids) == 2
        assert convoy.status == ConvoyStatus.PENDING

    def test_convoy_serialization(self):
        """Test convoy serialization."""
        convoy = Convoy.create("Test", ["bead-1"])
        data = convoy.to_dict()
        restored = Convoy.from_dict(data)

        assert restored.id == convoy.id
        assert restored.title == convoy.title


class TestConvoyManager:
    """Tests for ConvoyManager."""

    @pytest.mark.asyncio
    async def test_create_convoy(self, temp_dir, reset_singletons):
        """Test creating a convoy."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        # Create beads first
        bead1 = Bead.create(BeadType.TASK, "Task 1")
        bead2 = Bead.create(BeadType.TASK, "Task 2")
        await store.create(bead1)
        await store.create(bead2)

        manager = ConvoyManager(store, temp_dir)
        await manager.initialize()

        convoy = await manager.create_convoy(
            title="Test convoy",
            bead_ids=[bead1.id, bead2.id],
        )
        assert convoy.id is not None
        assert len(convoy.bead_ids) == 2

    @pytest.mark.asyncio
    async def test_assign_convoy(self, temp_dir, reset_singletons):
        """Test assigning a convoy to agents."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        bead = Bead.create(BeadType.TASK, "Task")
        await store.create(bead)

        manager = ConvoyManager(store, temp_dir)
        await manager.initialize()

        convoy = await manager.create_convoy("Test", [bead.id])
        success = await manager.assign_convoy(convoy.id, ["agent-1", "agent-2"])

        assert success
        updated = await manager.get_convoy(convoy.id)
        assert updated.status == ConvoyStatus.ACTIVE
        assert "agent-1" in updated.assigned_to


# =============================================================================
# Hook Queue Tests
# =============================================================================


class TestHookQueue:
    """Tests for HookQueue (GUPP pattern)."""

    @pytest.mark.asyncio
    async def test_push_and_pop(self, temp_dir, reset_singletons):
        """Test pushing and popping from hook queue."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        bead = Bead.create(BeadType.TASK, "Task")
        await store.create(bead)

        hook = HookQueue("agent-001", store, temp_dir / "hooks")
        await hook.initialize()

        entry = await hook.push(bead.id, priority=75)
        assert entry.status == HookEntryStatus.QUEUED

        popped = await hook.pop()
        assert popped is not None
        assert popped.id == bead.id

    @pytest.mark.asyncio
    async def test_gupp_recovery(self, temp_dir, reset_singletons):
        """Test GUPP recovery on startup."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        # Create beads
        bead1 = Bead.create(BeadType.TASK, "Task 1")
        bead2 = Bead.create(BeadType.TASK, "Task 2")
        await store.create(bead1)
        await store.create(bead2)

        # First session - push work
        hook = HookQueue("agent-001", store, temp_dir / "hooks")
        await hook.initialize()
        await hook.push(bead1.id)
        await hook.push(bead2.id)

        # Simulate restart - new hook queue
        hook2 = HookQueue("agent-001", store, temp_dir / "hooks")
        pending = await hook2.recover_on_startup()

        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_complete_and_fail(self, temp_dir, reset_singletons):
        """Test completing and failing beads in hook."""
        store = BeadStore(temp_dir, git_enabled=False)
        await store.initialize()

        bead1 = Bead.create(BeadType.TASK, "Success")
        bead2 = Bead.create(BeadType.TASK, "Failure")
        await store.create(bead1)
        await store.create(bead2)

        hook = HookQueue("agent-001", store, temp_dir / "hooks")
        await hook.initialize()

        await hook.push(bead1.id)
        await hook.push(bead2.id)

        await hook.pop()  # Get bead1
        await hook.complete(bead1.id)

        await hook.pop()  # Get bead2
        can_retry = await hook.fail(bead2.id, "Error occurred")

        assert can_retry  # First failure should allow retry


# =============================================================================
# Agent Roles Tests
# =============================================================================


class TestAgentRoles:
    """Tests for Agent Role hierarchy."""

    @pytest.mark.asyncio
    async def test_register_agent(self, temp_dir, reset_singletons):
        """Test registering an agent with a role."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        assignment = await hierarchy.register_agent(
            "mayor-001",
            AgentRole.MAYOR,
        )
        assert assignment.role == AgentRole.MAYOR
        assert RoleCapability.CREATE_CONVOY in assignment.capabilities

    @pytest.mark.asyncio
    async def test_supervision_hierarchy(self, temp_dir, reset_singletons):
        """Test supervision relationships."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        await hierarchy.register_agent("mayor-001", AgentRole.MAYOR)
        await hierarchy.register_agent(
            "worker-001",
            AgentRole.CREW,
            supervised_by="mayor-001",
        )

        supervisor = await hierarchy.get_supervisor("worker-001")
        assert supervisor is not None
        assert supervisor.agent_id == "mayor-001"

        supervised = await hierarchy.get_supervised("mayor-001")
        assert len(supervised) == 1
        assert supervised[0].agent_id == "worker-001"

    @pytest.mark.asyncio
    async def test_spawn_polecat(self, temp_dir, reset_singletons):
        """Test spawning ephemeral Polecat workers."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        await hierarchy.register_agent("mayor-001", AgentRole.MAYOR)
        polecat = await hierarchy.spawn_polecat(
            "mayor-001",
            "Fix bug in auth module",
        )

        assert polecat.role == AgentRole.POLECAT
        assert polecat.is_ephemeral
        assert polecat.supervised_by == "mayor-001"

    @pytest.mark.asyncio
    async def test_role_capabilities(self, temp_dir, reset_singletons):
        """Test role-based capabilities."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        mayor = await hierarchy.register_agent("mayor-001", AgentRole.MAYOR)
        crew = await hierarchy.register_agent("crew-001", AgentRole.CREW)

        assert mayor.has_capability(RoleCapability.CREATE_CONVOY)
        assert not crew.has_capability(RoleCapability.CREATE_CONVOY)
        assert crew.has_capability(RoleCapability.EXECUTE_TASK)


class TestRoleBasedRouter:
    """Tests for RoleBasedRouter."""

    @pytest.mark.asyncio
    async def test_route_coordination_task(self, temp_dir, reset_singletons):
        """Test routing coordination tasks to Mayor."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        await hierarchy.register_agent("mayor-001", AgentRole.MAYOR)
        await hierarchy.register_agent("crew-001", AgentRole.CREW)

        router = RoleBasedRouter(hierarchy)
        agent_id = await router.route_task("coordination")

        assert agent_id == "mayor-001"

    @pytest.mark.asyncio
    async def test_route_execution_task(self, temp_dir, reset_singletons):
        """Test routing execution tasks to Crew."""
        hierarchy = AgentHierarchy(temp_dir)
        await hierarchy.initialize()

        await hierarchy.register_agent("crew-001", AgentRole.CREW)

        router = RoleBasedRouter(hierarchy)
        agent_id = await router.route_task("execution", prefer_persistent=True)

        assert agent_id == "crew-001"


# =============================================================================
# Molecule Tests
# =============================================================================


class TestMolecule:
    """Tests for Molecule dataclass."""

    def test_create_molecule(self):
        """Test creating a molecule."""
        steps = [
            MoleculeStep.create("step1", "shell", {"command": "echo test"}),
            MoleculeStep.create("step2", "agent", {"task": "review"}),
        ]
        molecule = Molecule.create("test_workflow", steps)

        assert molecule.id is not None
        assert len(molecule.steps) == 2
        assert molecule.status == MoleculeStatus.PENDING

    def test_molecule_serialization(self):
        """Test molecule serialization."""
        steps = [MoleculeStep.create("step1", "shell")]
        molecule = Molecule.create("test", steps)

        data = molecule.to_dict()
        restored = Molecule.from_dict(data)

        assert restored.id == molecule.id
        assert len(restored.steps) == 1


class TestMoleculeEngine:
    """Tests for MoleculeEngine."""

    @pytest.mark.asyncio
    async def test_execute_shell_steps(self, temp_dir, reset_singletons):
        """Test executing shell steps."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [
            MoleculeStep.create("echo1", "shell", {"command": "echo hello"}),
            MoleculeStep.create("echo2", "shell", {"command": "echo world"}),
        ]
        molecule = Molecule.create("test_workflow", steps)

        result = await engine.execute(molecule)

        assert result.status == MoleculeStatus.COMPLETED
        assert result.completed_steps == 2
        assert result.failed_steps == 0

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume(self, temp_dir, reset_singletons):
        """Test checkpointing and resuming molecules."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [
            MoleculeStep.create("step1", "shell", {"command": "echo test"}),
        ]
        molecule = Molecule.create("resumable", steps)

        # Execute
        result = await engine.execute(molecule)
        assert result.success

        # Verify checkpoint exists
        checkpoint_file = temp_dir / f"{molecule.id}.json"
        assert checkpoint_file.exists()

        # Resume (should be a no-op since completed)
        result2 = await engine.resume(molecule.id)
        assert result2.success

    @pytest.mark.asyncio
    async def test_step_dependencies(self, temp_dir, reset_singletons):
        """Test step dependency ordering."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        step1 = MoleculeStep.create("first", "shell", {"command": "echo first"})
        step2 = MoleculeStep.create(
            "second",
            "shell",
            {"command": "echo second"},
            dependencies=[step1.id],
        )

        molecule = Molecule.create("ordered", [step2, step1])  # Out of order
        result = await engine.execute(molecule)

        assert result.success
        # step1 should complete before step2
        assert molecule.steps[1].status == StepStatus.COMPLETED  # step1
        assert molecule.steps[0].status == StepStatus.COMPLETED  # step2

    @pytest.mark.asyncio
    async def test_cancel_molecule(self, temp_dir, reset_singletons):
        """Test cancelling a molecule."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("step", "shell", {"command": "echo"})]
        molecule = Molecule.create("cancellable", steps)

        # Add to engine
        engine._molecules[molecule.id] = molecule

        success = await engine.cancel(molecule.id)
        assert success

        updated = await engine.get_molecule(molecule.id)
        assert updated.status == MoleculeStatus.CANCELLED


# =============================================================================
# Integration Tests
# =============================================================================


class TestGastownIntegration:
    """Integration tests for Gastown patterns working together."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_dir, reset_singletons):
        """Test a full workflow using all Gastown patterns."""
        # 1. Initialize stores
        store = BeadStore(temp_dir / "beads", git_enabled=False)
        await store.initialize()

        # 2. Set up agent hierarchy
        hierarchy = AgentHierarchy(temp_dir / "hierarchy")
        await hierarchy.initialize()
        await hierarchy.register_agent("mayor-001", AgentRole.MAYOR)
        await hierarchy.register_agent("worker-001", AgentRole.CREW, supervised_by="mayor-001")

        # 3. Create beads for work
        bead1 = Bead.create(BeadType.TASK, "Run tests")
        bead2 = Bead.create(BeadType.TASK, "Build", dependencies=[bead1.id])
        await store.create(bead1)
        await store.create(bead2)

        # 4. Create convoy grouping the beads
        convoy_manager = ConvoyManager(store, temp_dir / "convoys")
        await convoy_manager.initialize()
        convoy = await convoy_manager.create_convoy(
            "Build Pipeline",
            [bead1.id, bead2.id],
        )

        # 5. Assign convoy to worker
        await convoy_manager.assign_convoy(convoy.id, ["worker-001"])

        # 6. Worker processes via hook queue
        hook = HookQueue("worker-001", store, temp_dir / "hooks")
        await hook.initialize()
        await hook.push(bead1.id, priority=100)
        await hook.push(bead2.id, priority=50)

        # 7. Process work
        while await hook.has_work():
            bead = await hook.pop()
            if bead:
                await hook.complete(bead.id)

        # 8. Verify completion
        progress = await convoy_manager.get_convoy_progress(convoy.id)
        assert progress.completion_percentage == 100.0

    @pytest.mark.asyncio
    async def test_molecule_with_beads(self, temp_dir, reset_singletons):
        """Test molecules that create and track beads."""
        store = BeadStore(temp_dir / "beads", git_enabled=False)
        await store.initialize()

        engine = MoleculeEngine(store, temp_dir / "molecules")
        await engine.initialize()

        # Create molecule with shell steps
        steps = [
            MoleculeStep.create("setup", "shell", {"command": "echo setup"}),
            MoleculeStep.create("build", "shell", {"command": "echo build"}),
            MoleculeStep.create("test", "shell", {"command": "echo test"}),
        ]
        molecule = Molecule.create("ci_pipeline", steps)

        result = await engine.execute(molecule)

        assert result.success
        assert result.completed_steps == 3
        assert molecule.progress_percentage == 100.0
