"""
Tests for Arena checkpoint integration.

Tests cover:
- Arena.save_checkpoint() method
- Arena.restore_from_checkpoint() method
- Arena.list_checkpoints() method
- Arena.cleanup_checkpoints() method
- Pre-consensus checkpoint creation via PhaseExecutor hooks
- Checkpoint cleanup after successful debate completion
- DebateProtocol checkpoint settings
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.core import Critique, Message, Vote


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock CheckpointManager."""
    manager = MagicMock()
    manager.store = MagicMock()

    # Mock create_checkpoint to return a checkpoint with an ID
    async def mock_create(*args, **kwargs):
        checkpoint = MagicMock()
        checkpoint.checkpoint_id = (
            f"cp-{kwargs.get('debate_id', 'test')}-{kwargs.get('current_round', 0):03d}"
        )
        return checkpoint

    manager.create_checkpoint = AsyncMock(side_effect=mock_create)

    # Mock resume_from_checkpoint
    manager.resume_from_checkpoint = AsyncMock(return_value=None)

    return manager


@pytest.fixture
def mock_checkpoint_store():
    """Create a mock CheckpointStore."""
    store = MagicMock()
    store.list_checkpoints = AsyncMock(return_value=[])
    store.delete = AsyncMock(return_value=True)
    return store


@pytest.fixture
def mock_arena(mock_checkpoint_manager, mock_checkpoint_store):
    """Create a minimal mock Arena with checkpoint support."""
    arena = MagicMock()
    arena.checkpoint_manager = mock_checkpoint_manager
    arena.checkpoint_manager.store = mock_checkpoint_store
    arena.env = MagicMock()
    arena.env.task = "Test debate task"
    arena.agents = []
    arena.protocol = MagicMock()
    arena.protocol.rounds = 5

    # Import the actual methods from Arena
    from aragora.debate.orchestrator import Arena

    # Bind the checkpoint methods to the mock
    arena.save_checkpoint = Arena.save_checkpoint.__get__(arena, type(arena))
    arena.restore_from_checkpoint = Arena.restore_from_checkpoint.__get__(arena, type(arena))
    arena.list_checkpoints = Arena.list_checkpoints.__get__(arena, type(arena))
    arena.cleanup_checkpoints = Arena.cleanup_checkpoints.__get__(arena, type(arena))

    return arena


@pytest.fixture
def sample_messages():
    """Create sample Message objects."""
    return [
        Message(
            role="proposer",
            agent="claude",
            content="Test proposal content",
            timestamp=datetime.now(),
            round=1,
        ),
    ]


@pytest.fixture
def sample_critiques():
    """Create sample Critique objects."""
    return [
        Critique(
            agent="gpt4",
            target_agent="claude",
            target_content="Test proposal",
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            severity=5.0,
            reasoning="Test reasoning",
        ),
    ]


@pytest.fixture
def sample_votes():
    """Create sample Vote objects."""
    return [
        Vote(
            agent="claude",
            choice="proposal_a",
            confidence=0.85,
            reasoning="Test reasoning",
            continue_debate=False,
        ),
    ]


# =============================================================================
# Arena.save_checkpoint() Tests
# =============================================================================


class TestArenaSaveCheckpoint:
    """Tests for Arena.save_checkpoint() method."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_success(self, mock_arena, sample_messages):
        """Test successful checkpoint creation."""
        checkpoint_id = await mock_arena.save_checkpoint(
            debate_id="test-debate-123",
            phase="mid-round",
            messages=sample_messages,
            current_round=3,
        )

        assert checkpoint_id is not None
        assert checkpoint_id.startswith("cp-")
        mock_arena.checkpoint_manager.create_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_no_manager(self, mock_arena):
        """Test returns None when no checkpoint manager configured."""
        mock_arena.checkpoint_manager = None

        checkpoint_id = await mock_arena.save_checkpoint(
            debate_id="test-debate-123",
            phase="mid-round",
        )

        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_all_params(
        self, mock_arena, sample_messages, sample_critiques, sample_votes
    ):
        """Test checkpoint creation with all parameters."""
        checkpoint_id = await mock_arena.save_checkpoint(
            debate_id="test-debate-456",
            phase="consensus",
            messages=sample_messages,
            critiques=sample_critiques,
            votes=sample_votes,
            current_round=5,
            current_consensus="We agree on approach X",
        )

        assert checkpoint_id is not None
        call_kwargs = mock_arena.checkpoint_manager.create_checkpoint.call_args.kwargs
        assert call_kwargs["debate_id"] == "test-debate-456"
        assert call_kwargs["phase"] == "consensus"
        assert call_kwargs["current_round"] == 5
        assert call_kwargs["current_consensus"] == "We agree on approach X"

    @pytest.mark.asyncio
    async def test_save_checkpoint_handles_error(self, mock_arena):
        """Test checkpoint creation handles errors gracefully."""
        mock_arena.checkpoint_manager.create_checkpoint = AsyncMock(
            side_effect=OSError("Storage error")
        )

        checkpoint_id = await mock_arena.save_checkpoint(
            debate_id="test-debate-123",
            phase="error-test",
        )

        # Should return None on error, not raise
        assert checkpoint_id is None


# =============================================================================
# Arena.restore_from_checkpoint() Tests
# =============================================================================


class TestArenaRestoreFromCheckpoint:
    """Tests for Arena.restore_from_checkpoint() method."""

    @pytest.mark.asyncio
    async def test_restore_checkpoint_not_found(self, mock_arena):
        """Test restore returns None when checkpoint not found."""
        mock_arena.checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=None)

        ctx = await mock_arena.restore_from_checkpoint("cp-nonexistent-001")

        assert ctx is None

    @pytest.mark.asyncio
    async def test_restore_checkpoint_no_manager(self, mock_arena):
        """Test returns None when no checkpoint manager configured."""
        mock_arena.checkpoint_manager = None

        ctx = await mock_arena.restore_from_checkpoint("cp-test-001")

        assert ctx is None

    @pytest.mark.asyncio
    async def test_restore_checkpoint_success(self, mock_arena):
        """Test successful checkpoint restoration."""
        # Setup mock resumed checkpoint
        resumed = MagicMock()
        resumed.original_debate_id = "debate-original-123"
        resumed.checkpoint = MagicMock()
        resumed.checkpoint.task = "Original task"
        resumed.checkpoint.current_round = 3
        resumed.checkpoint.consensus_confidence = 0.75
        resumed.checkpoint.current_consensus = "Previous consensus"
        resumed.checkpoint.critiques = []
        resumed.messages = []
        resumed.votes = []

        mock_arena.checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=resumed)
        mock_arena._extract_debate_domain = MagicMock(return_value="technology")
        mock_arena.hook_manager = None
        mock_arena.org_id = "test-org"

        ctx = await mock_arena.restore_from_checkpoint("cp-test-001")

        assert ctx is not None
        assert ctx.debate_id == "debate-original-123"
        assert ctx._restored_from_checkpoint == "cp-test-001"
        assert ctx._checkpoint_resume_round == 3

    @pytest.mark.asyncio
    async def test_restore_checkpoint_handles_error(self, mock_arena):
        """Test restore handles errors gracefully."""
        mock_arena.checkpoint_manager.resume_from_checkpoint = AsyncMock(
            side_effect=ValueError("Corrupted checkpoint")
        )

        ctx = await mock_arena.restore_from_checkpoint("cp-corrupted-001")

        assert ctx is None


# =============================================================================
# Arena.list_checkpoints() Tests
# =============================================================================


class TestArenaListCheckpoints:
    """Tests for Arena.list_checkpoints() method."""

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, mock_arena):
        """Test list returns empty when no checkpoints exist."""
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        checkpoints = await mock_arena.list_checkpoints()

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_no_manager(self, mock_arena):
        """Test returns empty list when no checkpoint manager."""
        mock_arena.checkpoint_manager = None

        checkpoints = await mock_arena.list_checkpoints()

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_results(self, mock_arena):
        """Test list returns checkpoints correctly."""
        mock_checkpoints = [
            {
                "checkpoint_id": "cp-test-001",
                "debate_id": "debate-123",
                "task": "Test task",
                "current_round": 3,
                "created_at": "2024-01-01T00:00:00Z",
                "status": "complete",
            },
            {
                "checkpoint_id": "cp-test-002",
                "debate_id": "debate-123",
                "task": "Test task",
                "current_round": 4,
                "created_at": "2024-01-01T00:01:00Z",
                "status": "complete",
            },
        ]
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(
            return_value=mock_checkpoints
        )

        checkpoints = await mock_arena.list_checkpoints(debate_id="debate-123")

        assert len(checkpoints) == 2
        mock_arena.checkpoint_manager.store.list_checkpoints.assert_called_once_with(
            debate_id="debate-123",
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, mock_arena):
        """Test list respects limit parameter."""
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        await mock_arena.list_checkpoints(limit=10)

        mock_arena.checkpoint_manager.store.list_checkpoints.assert_called_once_with(
            debate_id=None,
            limit=10,
        )


# =============================================================================
# Arena.cleanup_checkpoints() Tests
# =============================================================================


class TestArenaCleanupCheckpoints:
    """Tests for Arena.cleanup_checkpoints() method."""

    @pytest.mark.asyncio
    async def test_cleanup_no_manager(self, mock_arena):
        """Test cleanup returns 0 when no checkpoint manager."""
        mock_arena.checkpoint_manager = None

        deleted = await mock_arena.cleanup_checkpoints("debate-123")

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_no_checkpoints(self, mock_arena):
        """Test cleanup returns 0 when no checkpoints exist."""
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        deleted = await mock_arena.cleanup_checkpoints("debate-123")

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_deletes_all_when_keep_zero(self, mock_arena):
        """Test cleanup deletes all checkpoints when keep_latest=0."""
        mock_checkpoints = [
            {"checkpoint_id": f"cp-{i}", "created_at": f"2024-01-01T00:0{i}:00Z"} for i in range(3)
        ]
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(
            return_value=mock_checkpoints
        )
        mock_arena.checkpoint_manager.store.delete = AsyncMock(return_value=True)

        deleted = await mock_arena.cleanup_checkpoints("debate-123", keep_latest=0)

        assert deleted == 3
        assert mock_arena.checkpoint_manager.store.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_keeps_latest(self, mock_arena):
        """Test cleanup keeps the latest N checkpoints."""
        mock_checkpoints = [
            {"checkpoint_id": "cp-0", "created_at": "2024-01-01T00:00:00Z"},
            {"checkpoint_id": "cp-1", "created_at": "2024-01-01T00:01:00Z"},
            {"checkpoint_id": "cp-2", "created_at": "2024-01-01T00:02:00Z"},  # Newest
        ]
        mock_arena.checkpoint_manager.store.list_checkpoints = AsyncMock(
            return_value=mock_checkpoints
        )
        mock_arena.checkpoint_manager.store.delete = AsyncMock(return_value=True)

        deleted = await mock_arena.cleanup_checkpoints("debate-123", keep_latest=1)

        # Should keep 1 (newest) and delete 2 (oldest)
        assert deleted == 2


# =============================================================================
# DebateProtocol Checkpoint Settings Tests
# =============================================================================


class TestDebateProtocolCheckpointSettings:
    """Tests for DebateProtocol checkpoint configuration."""

    def test_protocol_has_checkpoint_settings(self):
        """Test DebateProtocol has checkpoint configuration fields."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()

        # Check default values
        assert hasattr(protocol, "checkpoint_after_rounds")
        assert hasattr(protocol, "checkpoint_before_consensus")
        assert hasattr(protocol, "checkpoint_interval_rounds")
        assert hasattr(protocol, "checkpoint_cleanup_on_success")
        assert hasattr(protocol, "checkpoint_keep_on_success")

    def test_protocol_checkpoint_defaults(self):
        """Test DebateProtocol checkpoint defaults."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()

        assert protocol.checkpoint_after_rounds is True
        assert protocol.checkpoint_before_consensus is True
        assert protocol.checkpoint_interval_rounds == 1
        assert protocol.checkpoint_cleanup_on_success is True
        assert protocol.checkpoint_keep_on_success == 0

    def test_protocol_checkpoint_custom_values(self):
        """Test DebateProtocol accepts custom checkpoint values."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(
            checkpoint_after_rounds=False,
            checkpoint_before_consensus=False,
            checkpoint_interval_rounds=2,
            checkpoint_cleanup_on_success=False,
            checkpoint_keep_on_success=3,
        )

        assert protocol.checkpoint_after_rounds is False
        assert protocol.checkpoint_before_consensus is False
        assert protocol.checkpoint_interval_rounds == 2
        assert protocol.checkpoint_cleanup_on_success is False
        assert protocol.checkpoint_keep_on_success == 3


# =============================================================================
# PhaseExecutor Checkpoint Callback Tests
# =============================================================================


class TestPhaseExecutorCheckpointCallbacks:
    """Tests for PhaseExecutor checkpoint callbacks."""

    def test_phase_config_has_callback_fields(self):
        """Test PhaseConfig has pre/post phase callback fields."""
        from aragora.debate.phase_executor import PhaseConfig

        config = PhaseConfig()

        assert hasattr(config, "pre_phase_callback")
        assert hasattr(config, "post_phase_callback")
        assert config.pre_phase_callback is None
        assert config.post_phase_callback is None

    @pytest.mark.asyncio
    async def test_pre_phase_callback_called(self):
        """Test pre_phase_callback is called before phase execution."""
        from aragora.debate.phase_executor import PhaseConfig, PhaseExecutor

        callback_calls = []

        async def mock_callback(phase_name, context):
            callback_calls.append(("pre", phase_name))

        mock_phase = MagicMock()
        mock_phase.name = "test_phase"
        mock_phase.execute = AsyncMock(return_value=None)

        executor = PhaseExecutor(
            phases={"test_phase": mock_phase},
            config=PhaseConfig(
                pre_phase_callback=mock_callback,
                phase_timeout_seconds=5.0,
            ),
        )

        mock_context = MagicMock()
        await executor.execute(mock_context, debate_id="test", phase_order=["test_phase"])

        assert ("pre", "test_phase") in callback_calls

    @pytest.mark.asyncio
    async def test_post_phase_callback_called(self):
        """Test post_phase_callback is called after phase execution."""
        from aragora.debate.phase_executor import PhaseConfig, PhaseExecutor

        callback_calls = []

        async def mock_callback(phase_name, context, result):
            callback_calls.append(("post", phase_name))

        mock_phase = MagicMock()
        mock_phase.name = "test_phase"
        mock_phase.execute = AsyncMock(return_value=None)

        executor = PhaseExecutor(
            phases={"test_phase": mock_phase},
            config=PhaseConfig(
                post_phase_callback=mock_callback,
                phase_timeout_seconds=5.0,
            ),
        )

        mock_context = MagicMock()
        await executor.execute(mock_context, debate_id="test", phase_order=["test_phase"])

        assert ("post", "test_phase") in callback_calls

    @pytest.mark.asyncio
    async def test_callback_error_does_not_fail_phase(self):
        """Test callback errors don't prevent phase execution."""
        from aragora.debate.phase_executor import PhaseConfig, PhaseExecutor, PhaseStatus

        async def failing_callback(phase_name, context):
            raise RuntimeError("Callback error")

        mock_phase = MagicMock()
        mock_phase.name = "test_phase"
        mock_phase.execute = AsyncMock(return_value="success")

        executor = PhaseExecutor(
            phases={"test_phase": mock_phase},
            config=PhaseConfig(
                pre_phase_callback=failing_callback,
                phase_timeout_seconds=5.0,
            ),
        )

        mock_context = MagicMock()
        result = await executor.execute(mock_context, debate_id="test", phase_order=["test_phase"])

        # Phase should still complete despite callback error
        assert result.success
        phase_result = result.get_phase_result("test_phase")
        assert phase_result.status == PhaseStatus.COMPLETED


# =============================================================================
# Pre-Consensus Checkpoint Creation Tests
# =============================================================================


class TestPreConsensusCheckpoint:
    """Tests for pre-consensus checkpoint creation via arena_phases."""

    def test_create_checkpoint_callbacks_returns_tuple(self):
        """Test _create_checkpoint_callbacks returns correct tuple."""
        from aragora.debate.arena_phases import _create_checkpoint_callbacks

        mock_arena = MagicMock()
        mock_arena.checkpoint_manager = None

        pre_cb, post_cb = _create_checkpoint_callbacks(mock_arena)

        # No manager means no callbacks
        assert pre_cb is None
        assert post_cb is None

    def test_create_checkpoint_callbacks_with_manager(self):
        """Test _create_checkpoint_callbacks returns callback when manager exists."""
        from aragora.debate.arena_phases import _create_checkpoint_callbacks

        mock_arena = MagicMock()
        mock_arena.checkpoint_manager = MagicMock()
        mock_arena.protocol = MagicMock()
        mock_arena.protocol.checkpoint_before_consensus = True
        mock_arena.save_checkpoint = AsyncMock(return_value="cp-test-001")

        pre_cb, post_cb = _create_checkpoint_callbacks(mock_arena)

        assert pre_cb is not None
        assert callable(pre_cb)
        # post_cb is currently None in implementation
        assert post_cb is None

    @pytest.mark.asyncio
    async def test_pre_consensus_callback_creates_checkpoint(self):
        """Test pre-phase callback creates checkpoint before consensus."""
        from aragora.debate.arena_phases import _create_checkpoint_callbacks

        mock_arena = MagicMock()
        mock_arena.checkpoint_manager = MagicMock()
        mock_arena.protocol = MagicMock()
        mock_arena.protocol.checkpoint_before_consensus = True
        mock_arena.save_checkpoint = AsyncMock(return_value="cp-preconsensus-001")

        pre_cb, _ = _create_checkpoint_callbacks(mock_arena)

        mock_context = MagicMock()
        mock_context.debate_id = "test-debate-123"
        mock_context.result = MagicMock()
        mock_context.result.rounds_used = 5
        mock_context.result.messages = []

        # Call pre-phase callback for consensus phase
        await pre_cb("consensus", mock_context)

        mock_arena.save_checkpoint.assert_called_once()
        call_kwargs = mock_arena.save_checkpoint.call_args.kwargs
        assert call_kwargs["debate_id"] == "test-debate-123"
        assert call_kwargs["phase"] == "pre_consensus"
        assert call_kwargs["current_round"] == 5

    @pytest.mark.asyncio
    async def test_pre_callback_skips_non_consensus_phases(self):
        """Test pre-phase callback doesn't checkpoint non-consensus phases."""
        from aragora.debate.arena_phases import _create_checkpoint_callbacks

        mock_arena = MagicMock()
        mock_arena.checkpoint_manager = MagicMock()
        mock_arena.protocol = MagicMock()
        mock_arena.protocol.checkpoint_before_consensus = True
        mock_arena.save_checkpoint = AsyncMock(return_value="cp-test-001")

        pre_cb, _ = _create_checkpoint_callbacks(mock_arena)

        mock_context = MagicMock()

        # Call for non-consensus phases
        await pre_cb("proposal", mock_context)
        await pre_cb("debate_rounds", mock_context)
        await pre_cb("analytics", mock_context)

        # Should not have called save_checkpoint
        mock_arena.save_checkpoint.assert_not_called()
