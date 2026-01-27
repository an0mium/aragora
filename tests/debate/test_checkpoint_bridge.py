"""Tests for Molecule-Checkpoint Bridge."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.checkpoint_bridge import (
    CheckpointBridge,
    UnifiedRecoveryState,
    create_checkpoint_bridge,
)


class TestUnifiedRecoveryState:
    """Tests for UnifiedRecoveryState dataclass."""

    def test_create_minimal(self):
        """Test creating state with minimal fields."""
        state = UnifiedRecoveryState(
            debate_id="debate_123",
            current_round=1,
            phase="proposal",
        )

        assert state.debate_id == "debate_123"
        assert state.current_round == 1
        assert state.phase == "proposal"
        assert state.pending_molecules == 0
        assert state.completed_molecules == 0
        assert state.failed_molecules == 0
        assert state.checkpoint_messages == []

    def test_create_full(self):
        """Test creating state with all fields."""
        state = UnifiedRecoveryState(
            debate_id="debate_123",
            current_round=2,
            phase="voting",
            molecule_state={"molecules": [{"id": "mol_1"}]},
            pending_molecules=3,
            completed_molecules=5,
            failed_molecules=1,
            checkpoint_id="cp_123",
            checkpoint_messages=[{"agent": "claude", "content": "test"}],
            checkpoint_critiques=[{"agent": "gpt4", "target": "claude"}],
            checkpoint_votes=[{"agent": "claude", "choice": "A"}],
            channel_history=[{"sender": "claude", "content": "hello"}],
        )

        assert state.molecule_state == {"molecules": [{"id": "mol_1"}]}
        assert state.pending_molecules == 3
        assert state.completed_molecules == 5
        assert state.failed_molecules == 1
        assert len(state.checkpoint_messages) == 1
        assert len(state.channel_history) == 1

    def test_to_dict(self):
        """Test serialization to dict."""
        state = UnifiedRecoveryState(
            debate_id="debate_123",
            current_round=1,
            phase="proposal",
            pending_molecules=2,
            checkpoint_messages=[{"agent": "test"}],
        )

        data = state.to_dict()

        assert data["debate_id"] == "debate_123"
        assert data["current_round"] == 1
        assert data["phase"] == "proposal"
        assert data["pending_molecules"] == 2
        assert data["checkpoint_messages"] == [{"agent": "test"}]
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "debate_id": "debate_456",
            "current_round": 3,
            "phase": "synthesis",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "pending_molecules": 5,
            "completed_molecules": 10,
            "failed_molecules": 2,
            "checkpoint_id": "cp_456",
            "checkpoint_messages": [{"test": "data"}],
            "checkpoint_critiques": [],
            "checkpoint_votes": [],
            "channel_history": [],
        }

        state = UnifiedRecoveryState.from_dict(data)

        assert state.debate_id == "debate_456"
        assert state.current_round == 3
        assert state.phase == "synthesis"
        assert state.pending_molecules == 5
        assert state.checkpoint_id == "cp_456"

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = UnifiedRecoveryState(
            debate_id="test_debate",
            current_round=5,
            phase="critique",
            molecule_state={"key": "value"},
            pending_molecules=1,
            completed_molecules=8,
            checkpoint_id="cp_test",
            checkpoint_messages=[{"msg": 1}],
            channel_history=[{"ch": 1}],
        )

        data = original.to_dict()
        restored = UnifiedRecoveryState.from_dict(data)

        assert restored.debate_id == original.debate_id
        assert restored.current_round == original.current_round
        assert restored.phase == original.phase
        assert restored.molecule_state == original.molecule_state
        assert restored.pending_molecules == original.pending_molecules
        assert restored.completed_molecules == original.completed_molecules


class TestCheckpointBridge:
    """Tests for CheckpointBridge."""

    @pytest.fixture
    def mock_molecule_orchestrator(self):
        """Create mock molecule orchestrator."""
        mock = MagicMock()
        mock.to_checkpoint_state.return_value = {
            "debate_id": "test_debate",
            "molecules": [{"id": "mol_1", "status": "completed"}],
        }
        mock.get_progress.return_value = {
            "total": 10,
            "pending": 2,
            "completed": 7,
            "failed": 1,
        }
        mock.restore_from_checkpoint = MagicMock()
        return mock

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Create mock checkpoint manager."""
        mock = AsyncMock()
        mock.save = AsyncMock(return_value="path/to/checkpoint")
        mock.load = AsyncMock(return_value=None)
        mock.get_latest = AsyncMock(return_value=None)
        return mock

    @pytest.fixture
    def bridge(self, mock_molecule_orchestrator, mock_checkpoint_manager):
        """Create bridge with mocks."""
        return CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=mock_checkpoint_manager,
        )

    def test_init(self, mock_molecule_orchestrator, mock_checkpoint_manager):
        """Test bridge initialization."""
        bridge = CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert bridge.has_molecules()
        assert bridge.has_checkpoints()

    def test_init_molecules_only(self, mock_molecule_orchestrator):
        """Test bridge with only molecules."""
        bridge = CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=None,
        )

        assert bridge.has_molecules()
        assert not bridge.has_checkpoints()

    def test_init_checkpoints_only(self, mock_checkpoint_manager):
        """Test bridge with only checkpoints."""
        bridge = CheckpointBridge(
            molecule_orchestrator=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert not bridge.has_molecules()
        assert bridge.has_checkpoints()

    def test_init_neither(self):
        """Test bridge with neither component."""
        bridge = CheckpointBridge(
            molecule_orchestrator=None,
            checkpoint_manager=None,
        )

        assert not bridge.has_molecules()
        assert not bridge.has_checkpoints()

    @pytest.mark.asyncio
    async def test_save_checkpoint_full(
        self, bridge, mock_molecule_orchestrator, mock_checkpoint_manager
    ):
        """Test saving checkpoint with all components."""
        state = await bridge.save_checkpoint(
            debate_id="test_debate",
            current_round=2,
            phase="critique",
            messages=[{"agent": "claude", "content": "test"}],
            critiques=[{"agent": "gpt4", "target": "claude"}],
            votes=[],
            channel_history=[{"sender": "claude"}],
        )

        assert state.debate_id == "test_debate"
        assert state.current_round == 2
        assert state.phase == "critique"
        assert len(state.checkpoint_messages) == 1
        assert len(state.checkpoint_critiques) == 1
        assert state.pending_molecules == 2
        assert state.completed_molecules == 7
        assert state.failed_molecules == 1

        # Verify molecule orchestrator called
        mock_molecule_orchestrator.to_checkpoint_state.assert_called_once_with("test_debate")
        mock_molecule_orchestrator.get_progress.assert_called_once_with("test_debate")

        # Verify checkpoint manager called
        mock_checkpoint_manager.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_molecules_only(self, mock_molecule_orchestrator):
        """Test saving with only molecule orchestrator."""
        bridge = CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=None,
        )

        state = await bridge.save_checkpoint(
            debate_id="test",
            current_round=1,
            phase="proposal",
        )

        assert state.molecule_state is not None
        assert state.checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_checkpoints_only(self, mock_checkpoint_manager):
        """Test saving with only checkpoint manager."""
        bridge = CheckpointBridge(
            molecule_orchestrator=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        state = await bridge.save_checkpoint(
            debate_id="test",
            current_round=1,
            phase="proposal",
        )

        assert state.molecule_state is None
        mock_checkpoint_manager.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_from_molecules(self, mock_molecule_orchestrator):
        """Test restoring from molecule state."""
        bridge = CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=None,
        )

        state = await bridge.restore_checkpoint("test_debate")

        assert state is not None
        assert state.debate_id == "test_debate"
        assert state.molecule_state is not None
        assert state.pending_molecules == 2
        assert state.completed_molecules == 7

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, mock_checkpoint_manager):
        """Test restoring from checkpoint."""
        # Setup mock checkpoint
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "cp_123"
        mock_checkpoint.current_round = 3
        mock_checkpoint.phase = "voting"
        mock_checkpoint.messages = [{"test": "msg"}]
        mock_checkpoint.critiques = []
        mock_checkpoint.votes = []
        mock_checkpoint.claims_kernel_state = {
            "molecule_state": {"molecules": []},
            "channel_history": [{"sender": "test"}],
        }
        mock_checkpoint_manager.get_latest.return_value = mock_checkpoint

        bridge = CheckpointBridge(
            molecule_orchestrator=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        state = await bridge.restore_checkpoint("test_debate")

        assert state is not None
        assert state.checkpoint_id == "cp_123"
        assert state.current_round == 3
        assert state.phase == "voting"
        assert len(state.checkpoint_messages) == 1
        assert len(state.channel_history) == 1

    @pytest.mark.asyncio
    async def test_restore_by_checkpoint_id(self, mock_checkpoint_manager):
        """Test restoring by specific checkpoint ID."""
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "specific_cp"
        mock_checkpoint.current_round = 5
        mock_checkpoint.phase = "synthesis"
        mock_checkpoint.messages = []
        mock_checkpoint.critiques = []
        mock_checkpoint.votes = []
        mock_checkpoint.claims_kernel_state = None
        mock_checkpoint_manager.load.return_value = mock_checkpoint

        bridge = CheckpointBridge(
            molecule_orchestrator=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        state = await bridge.restore_checkpoint("test_debate", checkpoint_id="specific_cp")

        mock_checkpoint_manager.load.assert_called_once_with("specific_cp")
        assert state.checkpoint_id == "specific_cp"

    @pytest.mark.asyncio
    async def test_restore_combined(self, mock_molecule_orchestrator, mock_checkpoint_manager):
        """Test restoring with both molecules and checkpoints."""
        # Setup mock checkpoint
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "cp_123"
        mock_checkpoint.current_round = 3
        mock_checkpoint.phase = "voting"
        mock_checkpoint.messages = [{"test": "msg"}]
        mock_checkpoint.critiques = []
        mock_checkpoint.votes = []
        mock_checkpoint.claims_kernel_state = None
        mock_checkpoint_manager.get_latest.return_value = mock_checkpoint

        bridge = CheckpointBridge(
            molecule_orchestrator=mock_molecule_orchestrator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        state = await bridge.restore_checkpoint("test_debate")

        # Should have both molecule and checkpoint data
        assert state.molecule_state is not None
        assert state.checkpoint_id == "cp_123"
        # Round/phase should be updated from checkpoint
        assert state.current_round == 3
        assert state.phase == "voting"

    @pytest.mark.asyncio
    async def test_restore_no_state(self, bridge):
        """Test restoring when no state exists."""
        bridge._molecules.to_checkpoint_state.return_value = {"molecules": []}

        state = await bridge.restore_checkpoint("nonexistent_debate")

        assert state is None

    @pytest.mark.asyncio
    async def test_recover_molecules_from_checkpoint(
        self, bridge, mock_checkpoint_manager, mock_molecule_orchestrator
    ):
        """Test recovering molecules from a stored checkpoint."""
        mock_checkpoint = MagicMock()
        mock_checkpoint.claims_kernel_state = {
            "molecule_state": {"molecules": [{"id": "mol_1"}]},
        }
        mock_checkpoint_manager.load.return_value = mock_checkpoint

        result = await bridge.recover_molecules_from_checkpoint("cp_123")

        assert result is True
        mock_molecule_orchestrator.restore_from_checkpoint.assert_called_once_with(
            {"molecules": [{"id": "mol_1"}]}
        )

    @pytest.mark.asyncio
    async def test_recover_molecules_no_checkpoint(self, bridge, mock_checkpoint_manager):
        """Test recovering when checkpoint doesn't exist."""
        mock_checkpoint_manager.load.return_value = None

        result = await bridge.recover_molecules_from_checkpoint("missing_cp")

        assert result is False

    @pytest.mark.asyncio
    async def test_recover_molecules_no_molecule_state(self, bridge, mock_checkpoint_manager):
        """Test recovering when checkpoint has no molecule state."""
        mock_checkpoint = MagicMock()
        mock_checkpoint.claims_kernel_state = None
        mock_checkpoint_manager.load.return_value = mock_checkpoint

        result = await bridge.recover_molecules_from_checkpoint("cp_no_mol")

        assert result is False

    def test_get_recovery_summary(self, bridge, mock_molecule_orchestrator):
        """Test getting recovery summary."""
        summary = bridge.get_recovery_summary("test_debate")

        assert summary["debate_id"] == "test_debate"
        assert summary["has_molecules"] is True
        assert summary["molecule_progress"]["total"] == 10
        assert summary["molecule_progress"]["pending"] == 2

    def test_get_recovery_summary_no_molecules(self):
        """Test recovery summary with no molecules."""
        bridge = CheckpointBridge(molecule_orchestrator=None, checkpoint_manager=None)

        summary = bridge.get_recovery_summary("test")

        assert summary["has_molecules"] is False
        assert summary["molecule_progress"] == {}


class TestCheckpointBridgeFactory:
    """Tests for factory function."""

    def test_create_checkpoint_bridge(self):
        """Test factory function."""
        mock_mol = MagicMock()
        mock_cp = MagicMock()

        bridge = create_checkpoint_bridge(
            molecule_orchestrator=mock_mol,
            checkpoint_manager=mock_cp,
        )

        assert bridge is not None
        assert bridge.has_molecules()
        assert bridge.has_checkpoints()

    def test_create_with_none(self):
        """Test factory with None arguments."""
        bridge = create_checkpoint_bridge()

        assert not bridge.has_molecules()
        assert not bridge.has_checkpoints()


class TestCheckpointBridgeEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_save_with_empty_lists(self):
        """Test saving with empty message lists."""
        mock_cp = AsyncMock()
        mock_cp.save = AsyncMock()

        bridge = CheckpointBridge(checkpoint_manager=mock_cp)

        state = await bridge.save_checkpoint(
            debate_id="test",
            current_round=1,
            phase="proposal",
            messages=None,
            critiques=None,
            votes=None,
        )

        assert state.checkpoint_messages == []
        assert state.checkpoint_critiques == []
        assert state.checkpoint_votes == []

    @pytest.mark.asyncio
    async def test_restore_molecules_missing_orchestrator(self):
        """Test molecule recovery without orchestrator."""
        mock_cp = AsyncMock()
        bridge = CheckpointBridge(checkpoint_manager=mock_cp)

        result = await bridge.recover_molecules_from_checkpoint("cp_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_restore_molecules_missing_manager(self):
        """Test molecule recovery without checkpoint manager."""
        mock_mol = MagicMock()
        bridge = CheckpointBridge(molecule_orchestrator=mock_mol)

        result = await bridge.recover_molecules_from_checkpoint("cp_123")

        assert result is False
