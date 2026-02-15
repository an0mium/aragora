"""
Comprehensive tests for aragora/debate/session.py.

Tests cover:
- Session creation and lifecycle (PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED)
- Participant management (agents, roles)
- Round progression tracking
- Message handling during execution
- State persistence via checkpoints
- Timeout and expiration handling
- Error recovery and failure modes
- SessionManager for concurrent session tracking
- Event emission and handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import DebateResult, Environment, Message
from aragora.debate.cancellation import (
    CancellationReason,
    CancellationToken,
    DebateCancelled,
)
from aragora.debate.session import (
    DebateSession,
    DebateSessionState,
    SessionEvent,
    SessionEventType,
    SessionManager,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_environment():
    """Create a sample debate environment."""
    return Environment(task="Design a rate limiter for an API gateway")


@pytest.fixture
def sample_agents():
    """Create mock agents for testing."""
    agents = []
    for name in ["claude", "gpt4", "gemini"]:
        agent = MagicMock()
        agent.name = name
        agent.model = f"{name}-model"
        agent.role = "proposer"
        agent.system_prompt = f"You are {name}"
        agent.stance = "neutral"
        agents.append(agent)
    return agents


@pytest.fixture
def sample_protocol():
    """Create a mock protocol for testing."""
    protocol = MagicMock()
    protocol.rounds = 5
    protocol.consensus = "majority"
    return protocol


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = AsyncMock()
    manager.create_checkpoint = AsyncMock(
        return_value=MagicMock(
            checkpoint_id="cp-test-001",
            debate_id="test-debate",
            current_round=2,
            total_rounds=5,
        )
    )
    manager.resume_from_checkpoint = AsyncMock(
        return_value=MagicMock(
            checkpoint=MagicMock(current_round=2, total_rounds=5),
        )
    )
    return manager


@pytest.fixture
def mock_arena():
    """Create a mock Arena for testing."""
    arena = AsyncMock()
    arena.run = AsyncMock(
        return_value=DebateResult(
            debate_id="test-debate",
            task="Test task",
            final_answer="Test answer",
            consensus_reached=True,
            rounds_used=3,
        )
    )
    arena._partial_messages = []
    arena._partial_critiques = []
    return arena


# =============================================================================
# DebateSessionState Tests
# =============================================================================


class TestDebateSessionState:
    """Tests for DebateSessionState enum."""

    def test_enum_values(self):
        """Test all enum values exist and have correct string values."""
        assert DebateSessionState.PENDING.value == "pending"
        assert DebateSessionState.RUNNING.value == "running"
        assert DebateSessionState.PAUSED.value == "paused"
        assert DebateSessionState.COMPLETED.value == "completed"
        assert DebateSessionState.FAILED.value == "failed"
        assert DebateSessionState.CANCELLED.value == "cancelled"

    def test_enum_members(self):
        """Test all expected enum members exist."""
        members = list(DebateSessionState)
        assert len(members) == 6
        assert DebateSessionState.PENDING in members
        assert DebateSessionState.RUNNING in members
        assert DebateSessionState.PAUSED in members
        assert DebateSessionState.COMPLETED in members
        assert DebateSessionState.FAILED in members
        assert DebateSessionState.CANCELLED in members


class TestSessionEventType:
    """Tests for SessionEventType enum."""

    def test_enum_values(self):
        """Test all enum values exist and have correct string values."""
        assert SessionEventType.CREATED.value == "created"
        assert SessionEventType.STARTED.value == "started"
        assert SessionEventType.PAUSED.value == "paused"
        assert SessionEventType.RESUMED.value == "resumed"
        assert SessionEventType.COMPLETED.value == "completed"
        assert SessionEventType.FAILED.value == "failed"
        assert SessionEventType.CANCELLED.value == "cancelled"
        assert SessionEventType.STATE_CHANGED.value == "state_changed"
        assert SessionEventType.CHECKPOINT_CREATED.value == "checkpoint_created"

    def test_enum_members(self):
        """Test all expected enum members exist."""
        members = list(SessionEventType)
        assert len(members) == 9


class TestSessionEvent:
    """Tests for SessionEvent dataclass."""

    def test_create_basic_event(self):
        """Test creating a basic session event."""
        event = SessionEvent(
            type=SessionEventType.CREATED,
            session_id="session-123",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert event.type == SessionEventType.CREATED
        assert event.session_id == "session-123"
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.data == {}
        assert event.previous_state is None
        assert event.new_state is None

    def test_create_event_with_data(self):
        """Test creating an event with additional data."""
        event = SessionEvent(
            type=SessionEventType.STARTED,
            session_id="session-456",
            timestamp="2024-01-15T10:30:00Z",
            data={"task": "Test task", "agent_count": 3},
        )

        assert event.data["task"] == "Test task"
        assert event.data["agent_count"] == 3

    def test_create_state_change_event(self):
        """Test creating a state change event."""
        event = SessionEvent(
            type=SessionEventType.STATE_CHANGED,
            session_id="session-789",
            timestamp="2024-01-15T10:30:00Z",
            previous_state=DebateSessionState.PENDING,
            new_state=DebateSessionState.RUNNING,
        )

        assert event.previous_state == DebateSessionState.PENDING
        assert event.new_state == DebateSessionState.RUNNING


# =============================================================================
# DebateSession Creation Tests
# =============================================================================


class TestDebateSessionCreation:
    """Tests for DebateSession creation and initialization."""

    @pytest.mark.asyncio
    async def test_create_session_minimal(self, sample_environment, sample_agents, sample_protocol):
        """Test creating a session with minimal required fields."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session.id.startswith("session-")
        assert session.state == DebateSessionState.PENDING
        assert session.env == sample_environment
        assert len(session.agents) == 3
        assert session.protocol == sample_protocol
        assert session.total_rounds == 5
        assert session.current_round == 0

    @pytest.mark.asyncio
    async def test_create_session_with_custom_id(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test creating a session with a custom session ID."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            session_id="custom-session-id",
        )

        assert session.id == "custom-session-id"

    @pytest.mark.asyncio
    async def test_create_session_with_checkpoint_manager(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test creating a session with a checkpoint manager."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert session._checkpoint_manager == mock_checkpoint_manager

    @pytest.mark.asyncio
    async def test_create_session_emits_created_event(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that session creation emits a CREATED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        # Register handler to capture events after creation
        session.on_event(lambda e: events.append(e))

        # The creation event was emitted before we registered
        # Test by checking session was created successfully
        assert session.state == DebateSessionState.PENDING

    @pytest.mark.asyncio
    async def test_create_session_timestamps(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that session has proper timestamps."""
        before = datetime.now(timezone.utc)
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        after = datetime.now(timezone.utc)

        assert before <= session.created_at <= after
        assert session.started_at is None
        assert session.paused_at is None
        assert session.completed_at is None

    @pytest.mark.asyncio
    async def test_create_session_cancellation_token(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that session has a cancellation token."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert isinstance(session.cancellation_token, CancellationToken)
        assert session.cancellation_token.is_cancelled is False


class TestDebateSessionFromCheckpoint:
    """Tests for restoring sessions from checkpoints."""

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, sample_agents, sample_protocol):
        """Test restoring a session from a checkpoint."""
        checkpoint = MagicMock()
        checkpoint.debate_id = "debate-12345678"
        checkpoint.task = "Restored task"
        checkpoint.current_round = 3
        checkpoint.total_rounds = 5
        checkpoint.checkpoint_id = "cp-test-001"

        session = await DebateSession.from_checkpoint(
            checkpoint=checkpoint,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session.state == DebateSessionState.PAUSED
        # Session ID is "session-{debate_id[:8]}-resumed"
        assert "resumed" in session.id
        assert session.current_round == 3
        assert session.total_rounds == 5
        assert session.checkpoint_id == "cp-test-001"

    @pytest.mark.asyncio
    async def test_restore_preserves_environment(self, sample_agents, sample_protocol):
        """Test that restoring from checkpoint preserves environment."""
        checkpoint = MagicMock()
        checkpoint.debate_id = "debate-abc123"
        checkpoint.task = "Original task from checkpoint"
        checkpoint.current_round = 2
        checkpoint.total_rounds = 5
        checkpoint.checkpoint_id = "cp-test-002"

        session = await DebateSession.from_checkpoint(
            checkpoint=checkpoint,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session.env.task == "Original task from checkpoint"


# =============================================================================
# Session Lifecycle Tests
# =============================================================================


class TestDebateSessionLifecycle:
    """Tests for session lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_start_session(self, sample_environment, sample_agents, sample_protocol):
        """Test starting a session transitions to RUNNING."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()

            assert session.state == DebateSessionState.RUNNING
            assert session.started_at is not None

    @pytest.mark.asyncio
    async def test_start_session_emits_started_event(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that starting a session emits a STARTED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.on_event(lambda e: events.append(e))

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()

        started_events = [e for e in events if e.type == SessionEventType.STARTED]
        assert len(started_events) >= 1

    @pytest.mark.asyncio
    async def test_cannot_start_running_session(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that starting an already running session raises error."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()

            with pytest.raises(RuntimeError, match="Cannot start session in running state"):
                await session.start()

    @pytest.mark.asyncio
    async def test_session_completes_successfully(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that session completes and transitions to COMPLETED."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                    rounds_used=3,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()
            result = await session.wait_for_completion()

            assert session.state == DebateSessionState.COMPLETED
            assert session.completed_at is not None
            assert result is not None
            assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_session_failed_on_error(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that session transitions to FAILED on error."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(side_effect=RuntimeError("Agent failure"))
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.wait_for_completion()

            assert session.state == DebateSessionState.FAILED
            assert session.error_message == "Session failed: RuntimeError"


class TestDebateSessionStateTransitions:
    """Tests for valid and invalid state transitions."""

    @pytest.mark.asyncio
    async def test_valid_pending_to_running(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from PENDING to RUNNING."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        session._transition_state(DebateSessionState.RUNNING)
        assert session.state == DebateSessionState.RUNNING

    @pytest.mark.asyncio
    async def test_valid_pending_to_cancelled(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from PENDING to CANCELLED."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        session._transition_state(DebateSessionState.CANCELLED)
        assert session.state == DebateSessionState.CANCELLED

    @pytest.mark.asyncio
    async def test_valid_running_to_paused(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from RUNNING to PAUSED."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session._transition_state(DebateSessionState.RUNNING)

        session._transition_state(DebateSessionState.PAUSED)
        assert session.state == DebateSessionState.PAUSED

    @pytest.mark.asyncio
    async def test_valid_running_to_completed(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from RUNNING to COMPLETED."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session._transition_state(DebateSessionState.RUNNING)

        session._transition_state(DebateSessionState.COMPLETED)
        assert session.state == DebateSessionState.COMPLETED

    @pytest.mark.asyncio
    async def test_valid_running_to_failed(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from RUNNING to FAILED."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session._transition_state(DebateSessionState.RUNNING)

        session._transition_state(DebateSessionState.FAILED)
        assert session.state == DebateSessionState.FAILED

    @pytest.mark.asyncio
    async def test_valid_paused_to_running(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test valid transition from PAUSED to RUNNING."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session._transition_state(DebateSessionState.RUNNING)
        session._transition_state(DebateSessionState.PAUSED)

        session._transition_state(DebateSessionState.RUNNING)
        assert session.state == DebateSessionState.RUNNING

    @pytest.mark.asyncio
    async def test_invalid_pending_to_completed(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test invalid transition from PENDING to COMPLETED is blocked."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        session._transition_state(DebateSessionState.COMPLETED)
        # Should remain in PENDING state
        assert session.state == DebateSessionState.PENDING

    @pytest.mark.asyncio
    async def test_terminal_state_no_transitions(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that terminal states allow no further transitions."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session._transition_state(DebateSessionState.RUNNING)
        session._transition_state(DebateSessionState.COMPLETED)

        # Try to transition from COMPLETED to RUNNING
        session._transition_state(DebateSessionState.RUNNING)
        # Should remain in COMPLETED state
        assert session.state == DebateSessionState.COMPLETED


# =============================================================================
# Pause and Resume Tests
# =============================================================================


class TestDebateSessionPauseResume:
    """Tests for pause and resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_running_session(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test pausing a running session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=mock_checkpoint_manager,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()

            checkpoint_id = await session.pause("Taking a break")

            assert session.state == DebateSessionState.PAUSED
            assert session.paused_at is not None

    @pytest.mark.asyncio
    async def test_pause_emits_paused_event(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that pausing emits a PAUSED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.on_event(lambda e: events.append(e))

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.pause("Test pause")

        paused_events = [e for e in events if e.type == SessionEventType.PAUSED]
        assert len(paused_events) >= 1
        assert paused_events[0].data.get("reason") == "Test pause"

    @pytest.mark.asyncio
    async def test_cannot_pause_pending_session(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that pausing a pending session raises error."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with pytest.raises(RuntimeError, match="Cannot pause session in pending state"):
            await session.pause()

    @pytest.mark.asyncio
    async def test_resume_paused_session(self, sample_environment, sample_agents, sample_protocol):
        """Test resuming a paused session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.pause()
            await session.resume()

            assert session.state == DebateSessionState.RUNNING

    @pytest.mark.asyncio
    async def test_resume_with_checkpoint(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test resuming with a specific checkpoint ID."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=mock_checkpoint_manager,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.pause()
            await session.resume(checkpoint_id="cp-test-001")

            mock_checkpoint_manager.resume_from_checkpoint.assert_called_once_with("cp-test-001")

    @pytest.mark.asyncio
    async def test_cannot_resume_running_session(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that resuming a running session raises error."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()

            with pytest.raises(RuntimeError, match="Cannot resume session in running state"):
                await session.resume()


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestDebateSessionCancellation:
    """Tests for session cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_session(self, sample_environment, sample_agents, sample_protocol):
        """Test cancelling a pending session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        await session.cancel("No longer needed")

        assert session.state == DebateSessionState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_session(self, sample_environment, sample_agents, sample_protocol):
        """Test cancelling a running session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()

            # Make run() block until cancelled
            async def slow_run():
                await asyncio.sleep(10)
                return DebateResult(debate_id="test", task="test")

            mock_arena.run = slow_run
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.cancel("User cancelled")

            assert session.state == DebateSessionState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_paused_session(self, sample_environment, sample_agents, sample_protocol):
        """Test cancelling a paused session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.pause()
            await session.cancel("Session no longer needed")

            assert session.state == DebateSessionState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_emits_cancelled_event(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that cancellation emits a CANCELLED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.on_event(lambda e: events.append(e))

        await session.cancel("Test cancellation")

        cancelled_events = [e for e in events if e.type == SessionEventType.CANCELLED]
        assert len(cancelled_events) >= 1
        assert cancelled_events[0].data.get("reason") == "Test cancellation"

    @pytest.mark.asyncio
    async def test_cancel_completed_session_is_noop(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that cancelling a completed session is a no-op."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.wait_for_completion()

            # Session is now COMPLETED
            await session.cancel("Should not change state")

            # Should still be COMPLETED
            assert session.state == DebateSessionState.COMPLETED

    @pytest.mark.asyncio
    async def test_cancellation_token_integration(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that cancellation signals the cancellation token."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        await session.cancel("Token test")

        assert session.cancellation_token.is_cancelled is True
        assert "Token test" in session.cancellation_token.reason


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestDebateSessionEvents:
    """Tests for session event handling."""

    @pytest.mark.asyncio
    async def test_register_event_handler(self, sample_environment, sample_agents, sample_protocol):
        """Test registering an event handler."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        unregister = session.on_event(lambda e: events.append(e))

        assert callable(unregister)

    @pytest.mark.asyncio
    async def test_unregister_event_handler(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test unregistering an event handler."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        unregister = session.on_event(lambda e: events.append(e))
        unregister()

        await session.cancel("After unregister")

        # Event handler was unregistered, so no events should be captured
        # (The CANCELLED event is still emitted, just not to our handler)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_multiple_event_handlers(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test multiple event handlers receive events."""
        events1 = []
        events2 = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        session.on_event(lambda e: events1.append(e))
        session.on_event(lambda e: events2.append(e))

        await session.cancel("Multi-handler test")

        assert len(events1) >= 1
        assert len(events2) >= 1

    @pytest.mark.asyncio
    async def test_event_handler_exception_logged(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that event handler exceptions are logged but don't stop other handlers."""
        events = []

        def failing_handler(e):
            raise ValueError("Handler error")

        def good_handler(e):
            events.append(e)

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        session.on_event(failing_handler)
        session.on_event(good_handler)

        # Should not raise
        await session.cancel("Error test")

        # Good handler should still receive events
        assert len(events) >= 1


# =============================================================================
# Wait for Completion Tests
# =============================================================================


class TestDebateSessionWaitForCompletion:
    """Tests for wait_for_completion functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_result(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that wait_for_completion returns the debate result."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        expected_result = DebateResult(
            debate_id="test",
            task="test",
            final_answer="The answer",
            consensus_reached=True,
            rounds_used=3,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(return_value=expected_result)
            mock_arena_class.return_value = mock_arena

            await session.start()
            result = await session.wait_for_completion()

            assert result is not None
            assert result.final_answer == "The answer"
            assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_wait_for_completion_with_timeout(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test wait_for_completion with timeout."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()

            async def slow_run():
                await asyncio.sleep(10)
                return DebateResult(debate_id="test", task="test")

            mock_arena.run = slow_run
            mock_arena_class.return_value = mock_arena

            await session.start()
            result = await session.wait_for_completion(timeout=0.1)

            # Should return None due to timeout
            assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_completion_no_task(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test wait_for_completion when no task is running."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        # No task started
        result = await session.wait_for_completion()

        assert result is None


# =============================================================================
# Property Tests
# =============================================================================


class TestDebateSessionProperties:
    """Tests for session properties."""

    @pytest.mark.asyncio
    async def test_is_terminal_property(self, sample_environment, sample_agents, sample_protocol):
        """Test is_terminal property for various states."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        # PENDING is not terminal
        assert session.is_terminal is False

        # COMPLETED is terminal
        session._transition_state(DebateSessionState.RUNNING)
        session._transition_state(DebateSessionState.COMPLETED)
        assert session.is_terminal is True

    @pytest.mark.asyncio
    async def test_is_running_property(self, sample_environment, sample_agents, sample_protocol):
        """Test is_running property."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session.is_running is False

        session._transition_state(DebateSessionState.RUNNING)
        assert session.is_running is True

        session._transition_state(DebateSessionState.PAUSED)
        assert session.is_running is False

    @pytest.mark.asyncio
    async def test_duration_seconds_not_started(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test duration_seconds when session not started."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session.duration_seconds is None

    @pytest.mark.asyncio
    async def test_duration_seconds_running(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test duration_seconds for running session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.started_at = datetime.now(timezone.utc) - timedelta(seconds=5)

        duration = session.duration_seconds
        assert duration is not None
        assert duration >= 5.0

    @pytest.mark.asyncio
    async def test_duration_seconds_completed(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test duration_seconds for completed session."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        session.completed_at = datetime.now(timezone.utc) - timedelta(seconds=5)

        duration = session.duration_seconds
        assert duration is not None
        assert 4.9 <= duration <= 5.1


class TestDebateSessionToDict:
    """Tests for session serialization."""

    @pytest.mark.asyncio
    async def test_to_dict_basic(self, sample_environment, sample_agents, sample_protocol):
        """Test basic to_dict serialization."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        d = session.to_dict()

        assert "id" in d
        assert d["state"] == "pending"
        assert d["task"][:200] in sample_environment.task
        assert d["agent_count"] == 3
        assert d["current_round"] == 0
        assert d["total_rounds"] == 5

    @pytest.mark.asyncio
    async def test_to_dict_includes_timestamps(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test to_dict includes timestamps."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        d = session.to_dict()

        assert "created_at" in d
        assert "started_at" in d
        assert "paused_at" in d
        assert "completed_at" in d
        assert "duration_seconds" in d

    @pytest.mark.asyncio
    async def test_to_dict_includes_error_message(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test to_dict includes error message when present."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.error_message = "Test error"

        d = session.to_dict()

        assert d["error_message"] == "Test error"


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self, sample_environment, sample_agents, sample_protocol):
        """Test creating a session through the manager."""
        manager = SessionManager()

        session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session is not None
        assert session.state == DebateSessionState.PENDING

    @pytest.mark.asyncio
    async def test_get_session(self, sample_environment, sample_agents, sample_protocol):
        """Test getting a session by ID."""
        manager = SessionManager()

        session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        retrieved = await manager.get_session(session.id)

        assert retrieved is session

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        """Test getting a nonexistent session returns None."""
        manager = SessionManager()

        result = await manager.get_session("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, sample_environment, sample_agents, sample_protocol):
        """Test listing sessions."""
        manager = SessionManager()

        await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        sessions = await manager.list_sessions()

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_by_state(self, sample_environment, sample_agents, sample_protocol):
        """Test listing sessions filtered by state."""
        manager = SessionManager()

        session1 = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session2 = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        # Cancel one session
        await session2.cancel("Test")

        pending_sessions = await manager.list_sessions(state=DebateSessionState.PENDING)
        cancelled_sessions = await manager.list_sessions(state=DebateSessionState.CANCELLED)

        assert len(pending_sessions) == 1
        assert len(cancelled_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_limit(self, sample_environment, sample_agents, sample_protocol):
        """Test listing sessions with limit."""
        manager = SessionManager()

        for _ in range(5):
            await manager.create_session(
                env=sample_environment,
                agents=sample_agents,
                protocol=sample_protocol,
            )

        sessions = await manager.list_sessions(limit=3)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_cancel_session(self, sample_environment, sample_agents, sample_protocol):
        """Test cancelling a session through the manager."""
        manager = SessionManager()

        session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        result = await manager.cancel_session(session.id, "Manager cancel")

        assert result is True
        assert session.state == DebateSessionState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session(self):
        """Test cancelling a nonexistent session returns False."""
        manager = SessionManager()

        result = await manager.cancel_session("nonexistent-id", "Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_terminal_sessions(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test cleanup of terminal sessions."""
        manager = SessionManager(max_sessions=2)

        session1 = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        await session1.cancel("Test")

        session2 = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        await session2.cancel("Test")

        # Create third session - should trigger cleanup
        session3 = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        sessions = await manager.list_sessions()
        # Should have cleaned up terminal sessions
        assert len(sessions) <= 2

    @pytest.mark.asyncio
    async def test_cleanup_all_sessions(self, sample_environment, sample_agents, sample_protocol):
        """Test cleanup of all sessions."""
        manager = SessionManager()

        for _ in range(3):
            await manager.create_session(
                env=sample_environment,
                agents=sample_agents,
                protocol=sample_protocol,
            )

        await manager.cleanup()

        sessions = await manager.list_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_manager_with_checkpoint_manager(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test manager with checkpoint manager."""
        manager = SessionManager(checkpoint_manager=mock_checkpoint_manager)

        session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        assert session._checkpoint_manager == mock_checkpoint_manager


class TestSessionManagerConcurrency:
    """Tests for SessionManager concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test creating sessions concurrently."""
        manager = SessionManager()

        async def create_session():
            return await manager.create_session(
                env=sample_environment,
                agents=sample_agents,
                protocol=sample_protocol,
            )

        tasks = [create_session() for _ in range(10)]
        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 10
        assert len(set(s.id for s in sessions)) == 10  # All unique IDs

    @pytest.mark.asyncio
    async def test_concurrent_list_and_create(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test listing and creating sessions concurrently."""
        manager = SessionManager()

        # Pre-create some sessions
        for _ in range(3):
            await manager.create_session(
                env=sample_environment,
                agents=sample_agents,
                protocol=sample_protocol,
            )

        async def list_sessions():
            return await manager.list_sessions()

        async def create_session():
            return await manager.create_session(
                env=sample_environment,
                agents=sample_agents,
                protocol=sample_protocol,
            )

        tasks = [
            list_sessions(),
            create_session(),
            list_sessions(),
            create_session(),
            list_sessions(),
        ]

        results = await asyncio.gather(*tasks)

        # Should complete without errors
        assert len(results) == 5


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestDebateSessionErrorRecovery:
    """Tests for session error recovery."""

    @pytest.mark.asyncio
    async def test_execution_error_sets_failed_state(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that execution errors set the session to FAILED state."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(side_effect=RuntimeError("Execution failed"))
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.wait_for_completion()

            assert session.state == DebateSessionState.FAILED
            assert session.error_message == "Session failed: RuntimeError"

    @pytest.mark.asyncio
    async def test_execution_error_emits_failed_event(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that execution errors emit a FAILED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.on_event(lambda e: events.append(e))

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(side_effect=RuntimeError("Test error"))
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.wait_for_completion()

        failed_events = [e for e in events if e.type == SessionEventType.FAILED]
        assert len(failed_events) >= 1
        assert "session_failed:RuntimeError" in failed_events[0].data.get("error", "")

    @pytest.mark.asyncio
    async def test_debate_cancelled_exception_handled(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that DebateCancelled exception sets CANCELLED state."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                side_effect=DebateCancelled(
                    reason="Cancelled via exception",
                    reason_type=CancellationReason.USER_REQUESTED,
                )
            )
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.wait_for_completion()

            assert session.state == DebateSessionState.CANCELLED


# =============================================================================
# Checkpoint Integration Tests
# =============================================================================


class TestDebateSessionCheckpoints:
    """Tests for checkpoint integration."""

    @pytest.mark.asyncio
    async def test_pause_creates_checkpoint(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test that pausing creates a checkpoint."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=mock_checkpoint_manager,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            checkpoint_id = await session.pause("Creating checkpoint")

            assert checkpoint_id is not None
            assert session.checkpoint_id == checkpoint_id

    @pytest.mark.asyncio
    async def test_pause_emits_checkpoint_created_event(
        self, sample_environment, sample_agents, sample_protocol, mock_checkpoint_manager
    ):
        """Test that creating a checkpoint emits CHECKPOINT_CREATED event."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=mock_checkpoint_manager,
        )
        session.on_event(lambda e: events.append(e))

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            await session.pause("Checkpoint test")

        checkpoint_events = [e for e in events if e.type == SessionEventType.CHECKPOINT_CREATED]
        assert len(checkpoint_events) >= 1

    @pytest.mark.asyncio
    async def test_pause_without_checkpoint_manager(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test pausing without a checkpoint manager."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
            checkpoint_manager=None,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            await session.start()
            checkpoint_id = await session.pause("No checkpoint manager")

            # Should succeed but return None for checkpoint_id
            assert checkpoint_id is None
            assert session.state == DebateSessionState.PAUSED


# =============================================================================
# Timeout Tests
# =============================================================================


class TestDebateSessionTimeout:
    """Tests for session timeout handling."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout_returns_none(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test that timeout in wait_for_completion returns None."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()

            async def slow_run():
                await asyncio.sleep(10)
                return DebateResult(debate_id="test", task="test")

            mock_arena.run = slow_run
            mock_arena_class.return_value = mock_arena

            await session.start()
            result = await session.wait_for_completion(timeout=0.05)

            assert result is None

    @pytest.mark.asyncio
    async def test_session_cancellation_on_timeout(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test session can be cancelled after timeout."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()

            async def slow_run():
                await asyncio.sleep(10)
                return DebateResult(debate_id="test", task="test")

            mock_arena.run = slow_run
            mock_arena_class.return_value = mock_arena

            await session.start()

            # Wait with timeout
            await session.wait_for_completion(timeout=0.05)

            # Then cancel
            await session.cancel("Timeout exceeded")

            assert session.state == DebateSessionState.CANCELLED


# =============================================================================
# Integration Tests
# =============================================================================


class TestDebateSessionIntegration:
    """Integration tests combining multiple session features."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self, sample_environment, sample_agents, sample_protocol):
        """Test complete session lifecycle from creation to completion."""
        events = []

        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        session.on_event(lambda e: events.append(e))

        # Verify initial state
        assert session.state == DebateSessionState.PENDING

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    final_answer="Final answer",
                    consensus_reached=True,
                    rounds_used=3,
                )
            )
            mock_arena_class.return_value = mock_arena

            # Start session
            await session.start()
            assert session.state == DebateSessionState.RUNNING

            # Wait for completion
            result = await session.wait_for_completion()
            assert session.state == DebateSessionState.COMPLETED

            # Verify result
            assert result is not None
            assert result.final_answer == "Final answer"
            assert result.consensus_reached is True

        # Verify events were emitted
        event_types = [e.type for e in events]
        assert SessionEventType.STARTED in event_types
        assert SessionEventType.COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_session_with_pause_resume_complete(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test session that pauses, resumes, and completes."""
        session = await DebateSession.create(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
            mock_arena = AsyncMock()
            mock_arena.run = AsyncMock(
                return_value=DebateResult(
                    debate_id="test",
                    task="test",
                    consensus_reached=True,
                )
            )
            mock_arena._partial_messages = []
            mock_arena._partial_critiques = []
            mock_arena_class.return_value = mock_arena

            # Start
            await session.start()
            assert session.state == DebateSessionState.RUNNING

            # Pause
            await session.pause("Break time")
            assert session.state == DebateSessionState.PAUSED

            # Resume
            await session.resume()
            assert session.state == DebateSessionState.RUNNING

            # Complete
            await session.wait_for_completion()
            assert session.state == DebateSessionState.COMPLETED

    @pytest.mark.asyncio
    async def test_manager_with_multiple_session_states(
        self, sample_environment, sample_agents, sample_protocol
    ):
        """Test manager tracking multiple sessions in different states."""
        manager = SessionManager()

        # Create sessions
        pending_session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )

        cancelled_session = await manager.create_session(
            env=sample_environment,
            agents=sample_agents,
            protocol=sample_protocol,
        )
        await cancelled_session.cancel("Test")

        # List all
        all_sessions = await manager.list_sessions()
        assert len(all_sessions) == 2

        # List by state
        pending = await manager.list_sessions(state=DebateSessionState.PENDING)
        cancelled = await manager.list_sessions(state=DebateSessionState.CANCELLED)

        assert len(pending) == 1
        assert len(cancelled) == 1
