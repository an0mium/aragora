"""Tests for Gastown Handoff Protocol - context transfer between beads and molecules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.nomic.gastown_handoff import (
    BeadHandoffContext,
    HandoffPriority,
    HandoffProtocol,
    HandoffStatus,
    HandoffStore,
    MoleculeHandoffContext,
    create_handoff_protocol,
    create_handoff_store,
)


@pytest.fixture
def tmp_handoff_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for handoff storage."""
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    return handoff_dir


@pytest.fixture
async def handoff_store(tmp_handoff_dir: Path) -> HandoffStore:
    """Create and initialize a handoff store for testing."""
    store = HandoffStore(tmp_handoff_dir)
    await store.initialize()
    return store


@pytest.fixture
async def handoff_protocol(handoff_store: HandoffStore) -> HandoffProtocol:
    """Create a handoff protocol for testing."""
    return HandoffProtocol(handoff_store)


class TestBeadHandoffContext:
    """Tests for BeadHandoffContext dataclass."""

    def test_create_basic(self):
        """Test creating a basic handoff context."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
        )
        assert ctx.source_bead_id == "bead-123"
        assert ctx.source_agent_id == "agent-001"
        assert ctx.status == HandoffStatus.PENDING
        assert ctx.priority == HandoffPriority.NORMAL
        assert ctx.id.startswith("handoff-")
        assert ctx.created_at is not None
        assert ctx.expires_at is not None

    def test_create_with_content(self):
        """Test creating handoff with findings and artifacts."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Finding 1", "Finding 2"],
            artifacts={"report.md": "/path/to/report.md"},
            decisions=["Chose approach A"],
            next_steps=["Implement solution"],
            open_questions=["What about edge cases?"],
            warnings=["High complexity detected"],
            execution_summary="Completed analysis phase",
        )
        assert ctx.findings == ["Finding 1", "Finding 2"]
        assert ctx.artifacts == {"report.md": "/path/to/report.md"}
        assert ctx.decisions == ["Chose approach A"]
        assert ctx.next_steps == ["Implement solution"]
        assert ctx.open_questions == ["What about edge cases?"]
        assert ctx.warnings == ["High complexity detected"]
        assert ctx.execution_summary == "Completed analysis phase"

    def test_create_with_target(self):
        """Test creating handoff with target bead and agent."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            target_bead_id="bead-456",
            target_agent_id="agent-002",
        )
        assert ctx.target_bead_id == "bead-456"
        assert ctx.target_agent_id == "agent-002"

    def test_create_with_priority(self):
        """Test creating handoff with custom priority."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            priority=HandoffPriority.CRITICAL,
        )
        assert ctx.priority == HandoffPriority.CRITICAL

    def test_create_with_ttl(self):
        """Test creating handoff with custom TTL."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            ttl_hours=48.0,
        )
        expected_expiry = ctx.created_at + timedelta(hours=48)
        assert ctx.expires_at is not None
        assert abs((ctx.expires_at - expected_expiry).total_seconds()) < 1

    def test_create_no_expiry(self):
        """Test creating handoff with no expiry."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            ttl_hours=0,
        )
        assert ctx.expires_at is None

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            target_bead_id="bead-456",
            findings=["Finding 1"],
            priority=HandoffPriority.HIGH,
        )
        data = original.to_dict()
        restored = BeadHandoffContext.from_dict(data)

        assert restored.id == original.id
        assert restored.source_bead_id == original.source_bead_id
        assert restored.source_agent_id == original.source_agent_id
        assert restored.target_bead_id == original.target_bead_id
        assert restored.findings == original.findings
        assert restored.priority == original.priority
        assert restored.status == original.status

    def test_is_expired_not_expired(self):
        """Test is_expired when not expired."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            ttl_hours=24.0,
        )
        assert ctx.is_expired() is False

    def test_is_expired_expired(self):
        """Test is_expired when expired."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            ttl_hours=0.001,  # Very short TTL
        )
        # Manually set expires_at to past
        ctx.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert ctx.is_expired() is True

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry set."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            ttl_hours=0,
        )
        assert ctx.is_expired() is False

    def test_format_for_prompt(self):
        """Test formatting handoff for LLM prompt."""
        ctx = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Key insight 1"],
            decisions=["Chose approach A"],
            next_steps=["Implement solution"],
            open_questions=["Edge case handling?"],
            warnings=["High complexity"],
            artifacts={"report.md": "/path/report.md"},
            execution_summary="Completed analysis",
        )
        prompt = ctx.format_for_prompt()

        assert "## Handoff Context" in prompt
        assert "agent-001" in prompt
        assert "bead-123" in prompt
        assert "Completed analysis" in prompt
        assert "Key insight 1" in prompt
        assert "Chose approach A" in prompt
        assert "Implement solution" in prompt
        assert "Edge case handling?" in prompt
        assert "High complexity" in prompt
        assert "report.md" in prompt


class TestMoleculeHandoffContext:
    """Tests for MoleculeHandoffContext dataclass."""

    def test_create_basic(self):
        """Test creating a basic molecule handoff."""
        ctx = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
        )
        assert ctx.molecule_id == "mol-123"
        assert ctx.source_step == "step-1"
        assert ctx.target_step == "step-2"
        assert ctx.id.startswith("mol-handoff-")
        assert ctx.step_success is True
        assert ctx.created_at is not None

    def test_create_with_output(self):
        """Test creating handoff with step output."""
        ctx = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
            step_output={"result": "data"},
            step_success=True,
        )
        assert ctx.step_output == {"result": "data"}
        assert ctx.step_success is True

    def test_create_with_failure(self):
        """Test creating handoff for failed step."""
        ctx = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
            step_success=False,
        )
        assert ctx.step_success is False

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
            step_output={"key": "value"},
        )
        original.accumulated_findings = ["Finding 1"]
        original.accumulated_artifacts = {"art.txt": "/path/art.txt"}
        original.checkpoint_data = {"checkpoint": "data"}

        data = original.to_dict()
        restored = MoleculeHandoffContext.from_dict(data)

        assert restored.id == original.id
        assert restored.molecule_id == original.molecule_id
        assert restored.source_step == original.source_step
        assert restored.target_step == original.target_step
        assert restored.step_output == original.step_output
        assert restored.accumulated_findings == original.accumulated_findings
        assert restored.accumulated_artifacts == original.accumulated_artifacts
        assert restored.checkpoint_data == original.checkpoint_data

    def test_skip_remaining(self):
        """Test skip_remaining control flow."""
        ctx = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
        )
        ctx.should_skip_remaining = True
        ctx.skip_reason = "Critical failure detected"

        assert ctx.should_skip_remaining is True
        assert ctx.skip_reason == "Critical failure detected"


class TestHandoffStore:
    """Tests for HandoffStore persistence."""

    @pytest.mark.asyncio
    async def test_save_and_get_bead_handoff(self, handoff_store: HandoffStore):
        """Test saving and retrieving a bead handoff."""
        handoff = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Finding 1"],
        )
        await handoff_store.save_bead_handoff(handoff)

        retrieved = await handoff_store.get_bead_handoff(handoff.id)
        assert retrieved is not None
        assert retrieved.id == handoff.id
        assert retrieved.findings == ["Finding 1"]

    @pytest.mark.asyncio
    async def test_save_and_get_molecule_handoff(self, handoff_store: HandoffStore):
        """Test saving and retrieving a molecule handoff."""
        handoff = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
        )
        await handoff_store.save_molecule_handoff(handoff)

        retrieved = await handoff_store.get_molecule_handoff(handoff.id)
        assert retrieved is not None
        assert retrieved.id == handoff.id
        assert retrieved.molecule_id == "mol-123"

    @pytest.mark.asyncio
    async def test_get_pending_for_agent(self, handoff_store: HandoffStore):
        """Test getting pending handoffs for an agent."""
        handoff1 = BeadHandoffContext.create(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            target_agent_id="agent-002",
        )
        handoff2 = BeadHandoffContext.create(
            source_bead_id="bead-2",
            source_agent_id="agent-001",
            target_agent_id="agent-003",
        )
        await handoff_store.save_bead_handoff(handoff1)
        await handoff_store.save_bead_handoff(handoff2)

        pending = await handoff_store.get_pending_for_agent("agent-002")
        assert len(pending) == 1
        assert pending[0].id == handoff1.id

    @pytest.mark.asyncio
    async def test_get_pending_for_bead(self, handoff_store: HandoffStore):
        """Test getting pending handoffs for a bead."""
        handoff = BeadHandoffContext.create(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            target_bead_id="bead-2",
        )
        await handoff_store.save_bead_handoff(handoff)

        pending = await handoff_store.get_pending_for_bead("bead-2")
        assert len(pending) == 1
        assert pending[0].id == handoff.id

    @pytest.mark.asyncio
    async def test_mark_delivered(self, handoff_store: HandoffStore):
        """Test marking a handoff as delivered."""
        handoff = BeadHandoffContext.create(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
        )
        await handoff_store.save_bead_handoff(handoff)

        await handoff_store.mark_delivered(handoff.id)

        retrieved = await handoff_store.get_bead_handoff(handoff.id)
        assert retrieved.status == HandoffStatus.DELIVERED
        assert retrieved.delivered_at is not None

    @pytest.mark.asyncio
    async def test_mark_acknowledged(self, handoff_store: HandoffStore):
        """Test marking a handoff as acknowledged."""
        handoff = BeadHandoffContext.create(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
        )
        await handoff_store.save_bead_handoff(handoff)

        await handoff_store.mark_acknowledged(handoff.id)

        retrieved = await handoff_store.get_bead_handoff(handoff.id)
        assert retrieved.status == HandoffStatus.ACKNOWLEDGED
        assert retrieved.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, handoff_store: HandoffStore):
        """Test cleaning up expired handoffs."""
        # Create expired handoff
        handoff = BeadHandoffContext.create(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            ttl_hours=0.001,
        )
        handoff.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await handoff_store.save_bead_handoff(handoff)

        count = await handoff_store.cleanup_expired()
        assert count == 1

        retrieved = await handoff_store.get_bead_handoff(handoff.id)
        assert retrieved.status == HandoffStatus.EXPIRED


class TestHandoffProtocol:
    """Tests for HandoffProtocol."""

    @pytest.mark.asyncio
    async def test_create_bead_handoff(self, handoff_protocol: HandoffProtocol):
        """Test creating a bead handoff via protocol."""
        handoff = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Key insight"],
        )
        assert handoff.source_bead_id == "bead-123"
        assert handoff.findings == ["Key insight"]

    @pytest.mark.asyncio
    async def test_create_molecule_handoff(self, handoff_protocol: HandoffProtocol):
        """Test creating a molecule handoff via protocol."""
        handoff = await handoff_protocol.create_molecule_handoff(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
        )
        assert handoff.molecule_id == "mol-123"
        assert handoff.source_step == "step-1"

    @pytest.mark.asyncio
    async def test_transfer_to_bead(self, handoff_protocol: HandoffProtocol):
        """Test transferring handoff to a specific bead."""
        handoff = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
        )

        updated = await handoff_protocol.transfer_to_bead(handoff.id, "bead-456")
        assert updated is not None
        assert updated.target_bead_id == "bead-456"

    @pytest.mark.asyncio
    async def test_transfer_to_agent(self, handoff_protocol: HandoffProtocol):
        """Test transferring handoff to a specific agent."""
        handoff = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
        )

        updated = await handoff_protocol.transfer_to_agent(handoff.id, "agent-002")
        assert updated is not None
        assert updated.target_agent_id == "agent-002"

    @pytest.mark.asyncio
    async def test_transfer_nonexistent(self, handoff_protocol: HandoffProtocol):
        """Test transferring non-existent handoff returns None."""
        result = await handoff_protocol.transfer_to_bead("nonexistent", "bead-456")
        assert result is None

    @pytest.mark.asyncio
    async def test_recover_for_agent(self, handoff_protocol: HandoffProtocol):
        """Test recovering pending handoffs for an agent."""
        # Create handoffs with different priorities
        handoff1 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            target_agent_id="agent-002",
            priority=HandoffPriority.NORMAL,
        )
        handoff2 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-2",
            source_agent_id="agent-001",
            target_agent_id="agent-002",
            priority=HandoffPriority.CRITICAL,
        )

        recovered = await handoff_protocol.recover_for_agent("agent-002")
        assert len(recovered) == 2
        # Should be sorted by priority (critical first)
        assert recovered[0].id == handoff2.id
        assert recovered[1].id == handoff1.id

    @pytest.mark.asyncio
    async def test_recover_for_bead(self, handoff_protocol: HandoffProtocol):
        """Test recovering pending handoffs for a bead."""
        await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            target_bead_id="bead-2",
        )

        recovered = await handoff_protocol.recover_for_bead("bead-2")
        assert len(recovered) == 1
        assert recovered[0].target_bead_id == "bead-2"

    @pytest.mark.asyncio
    async def test_acknowledge(self, handoff_protocol: HandoffProtocol):
        """Test acknowledging a handoff."""
        handoff = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
        )

        await handoff_protocol.acknowledge(handoff.id)

        retrieved = await handoff_protocol.store.get_bead_handoff(handoff.id)
        assert retrieved.status == HandoffStatus.ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_merge_contexts_single(self, handoff_protocol: HandoffProtocol):
        """Test merging single handoff (no-op)."""
        handoff = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Finding 1"],
        )

        merged = await handoff_protocol.merge_contexts([handoff])
        assert merged.findings == ["Finding 1"]

    @pytest.mark.asyncio
    async def test_merge_contexts_multiple(self, handoff_protocol: HandoffProtocol):
        """Test merging multiple handoffs."""
        handoff1 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            findings=["Finding 1", "Common finding"],
            decisions=["Decision A"],
            priority=HandoffPriority.HIGH,
        )
        handoff2 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-2",
            source_agent_id="agent-002",
            findings=["Finding 2", "Common finding"],
            decisions=["Decision B"],
            priority=HandoffPriority.NORMAL,
        )

        merged = await handoff_protocol.merge_contexts([handoff1, handoff2])

        # All unique findings should be present
        assert "Finding 1" in merged.findings
        assert "Finding 2" in merged.findings
        assert merged.findings.count("Common finding") == 1  # Deduped

        # All unique decisions should be present
        assert "Decision A" in merged.decisions
        assert "Decision B" in merged.decisions

        # Priority from highest-priority handoff
        assert merged.priority == HandoffPriority.HIGH

        # Source handoff IDs tracked
        assert handoff1.id in merged.metadata["source_handoff_ids"]
        assert handoff2.id in merged.metadata["source_handoff_ids"]

    @pytest.mark.asyncio
    async def test_merge_contexts_empty_raises(self, handoff_protocol: HandoffProtocol):
        """Test merging empty list raises error."""
        with pytest.raises(ValueError, match="Cannot merge empty"):
            await handoff_protocol.merge_contexts([])

    @pytest.mark.asyncio
    async def test_merge_contexts_artifacts(self, handoff_protocol: HandoffProtocol):
        """Test merging preserves artifacts without override."""
        handoff1 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-1",
            source_agent_id="agent-001",
            artifacts={"report.md": "/path1/report.md"},
            priority=HandoffPriority.CRITICAL,
        )
        handoff2 = await handoff_protocol.create_bead_handoff(
            source_bead_id="bead-2",
            source_agent_id="agent-002",
            artifacts={"report.md": "/path2/report.md", "data.json": "/path/data.json"},
            priority=HandoffPriority.NORMAL,
        )

        merged = await handoff_protocol.merge_contexts([handoff1, handoff2])

        # First handoff's artifact takes precedence
        assert merged.artifacts["report.md"] == "/path1/report.md"
        # New artifacts are added
        assert merged.artifacts["data.json"] == "/path/data.json"


class TestHandoffPersistence:
    """Tests for handoff persistence across restarts."""

    @pytest.mark.asyncio
    async def test_bead_handoffs_persist(self, tmp_handoff_dir: Path):
        """Test bead handoffs survive store restart."""
        # Create and save
        store1 = HandoffStore(tmp_handoff_dir)
        await store1.initialize()

        handoff = BeadHandoffContext.create(
            source_bead_id="bead-123",
            source_agent_id="agent-001",
            findings=["Persisted finding"],
        )
        await store1.save_bead_handoff(handoff)

        # Create new store (simulates restart)
        store2 = HandoffStore(tmp_handoff_dir)
        await store2.initialize()

        loaded = await store2.get_bead_handoff(handoff.id)
        assert loaded is not None
        assert loaded.findings == ["Persisted finding"]

    @pytest.mark.asyncio
    async def test_molecule_handoffs_persist(self, tmp_handoff_dir: Path):
        """Test molecule handoffs survive store restart."""
        store1 = HandoffStore(tmp_handoff_dir)
        await store1.initialize()

        handoff = MoleculeHandoffContext.create(
            molecule_id="mol-123",
            source_step="step-1",
            target_step="step-2",
            step_output={"key": "value"},
        )
        await store1.save_molecule_handoff(handoff)

        store2 = HandoffStore(tmp_handoff_dir)
        await store2.initialize()

        loaded = await store2.get_molecule_handoff(handoff.id)
        assert loaded is not None
        assert loaded.step_output == {"key": "value"}


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_create_handoff_store_default_path(self, tmp_path: Path, monkeypatch):
        """Test creating handoff store with default path."""
        # Change to temp directory to avoid polluting filesystem
        monkeypatch.chdir(tmp_path)
        store = await create_handoff_store()
        assert store.storage_path == Path(".handoffs")

    @pytest.mark.asyncio
    async def test_create_handoff_store_custom_path(self, tmp_path: Path):
        """Test creating handoff store with custom path."""
        custom_path = tmp_path / "custom_handoffs"
        store = await create_handoff_store(custom_path)
        assert store.storage_path == custom_path
        assert custom_path.exists()

    @pytest.mark.asyncio
    async def test_create_handoff_protocol(self, tmp_path: Path):
        """Test creating handoff protocol."""
        protocol = await create_handoff_protocol(tmp_path / "protocol_test")
        assert protocol.store is not None
