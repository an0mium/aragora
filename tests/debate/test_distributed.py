"""
Tests for distributed debate coordinator.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aragora.debate.distributed import DistributedDebateCoordinator, DistributedDebateResult
from aragora.debate.distributed_events import (
    DistributedDebateEventType,
    DistributedDebateEvent,
    DistributedDebateState,
    AgentProposal,
)


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.publish = AsyncMock(return_value=True)
    bus.subscribe = AsyncMock()
    bus.local_region = "test-region"
    return bus


@pytest.fixture
def mock_local_registry():
    """Create a mock local registry."""
    registry = MagicMock()
    registry.list_available = AsyncMock(
        return_value=[
            MagicMock(agent_id="claude-3", model="claude-3", capabilities=["debate"]),
            MagicMock(agent_id="gpt-4", model="gpt-4", capabilities=["debate"]),
        ]
    )
    return registry


@pytest.fixture
def coordinator(mock_event_bus, mock_local_registry):
    """Create a distributed debate coordinator."""
    return DistributedDebateCoordinator(
        instance_id="test-instance",
        event_bus=mock_event_bus,
        local_registry=mock_local_registry,
    )


class TestDistributedDebateCoordinatorInit:
    """Tests for coordinator initialization."""

    def test_coordinator_creation(self, mock_event_bus, mock_local_registry):
        """Test creating a coordinator."""
        coordinator = DistributedDebateCoordinator(
            instance_id="my-instance",
            event_bus=mock_event_bus,
            local_registry=mock_local_registry,
        )

        assert coordinator._instance_id == "my-instance"
        assert coordinator._event_bus == mock_event_bus
        assert coordinator._connected is False

    def test_coordinator_without_event_bus(self, mock_local_registry):
        """Test coordinator works without event bus (local only)."""
        coordinator = DistributedDebateCoordinator(
            instance_id="local-instance",
            local_registry=mock_local_registry,
        )

        assert coordinator._event_bus is None

    def test_coordinator_generates_instance_id(self, mock_event_bus, mock_local_registry):
        """Test coordinator generates instance ID if not provided."""
        coordinator = DistributedDebateCoordinator(
            event_bus=mock_event_bus,
            local_registry=mock_local_registry,
        )

        assert coordinator._instance_id is not None
        assert len(coordinator._instance_id) > 0


class TestDistributedDebateCoordinatorLifecycle:
    """Tests for coordinator lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self, coordinator, mock_event_bus):
        """Test connecting the coordinator."""
        await coordinator.connect()

        assert coordinator._connected is True
        mock_event_bus.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, coordinator, mock_event_bus):
        """Test connecting twice is idempotent."""
        await coordinator.connect()
        await coordinator.connect()

        # Should only subscribe once
        assert mock_event_bus.subscribe.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect(self, coordinator):
        """Test disconnecting the coordinator."""
        await coordinator.connect()
        await coordinator.disconnect()

        assert coordinator._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_without_connect(self, coordinator):
        """Test disconnecting without connecting is safe."""
        await coordinator.disconnect()
        assert coordinator._connected is False


class TestDistributedDebateCoordinatorDebates:
    """Tests for running distributed debates."""

    @pytest.mark.asyncio
    async def test_start_debate_not_connected(self, coordinator):
        """Test starting debate when not connected."""
        result = await coordinator.start_debate(
            task="What database should we use?",
        )

        # Should still work in local-only mode
        assert result is not None

    @pytest.mark.asyncio
    async def test_start_debate_publishes_event(self, coordinator, mock_event_bus):
        """Test starting debate publishes event."""
        await coordinator.connect()

        # Mock the debate execution
        with patch.object(coordinator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = DistributedDebateResult(
                debate_id="test-debate",
                consensus_reached=True,
                final_answer="PostgreSQL",
                confidence=0.85,
                rounds_completed=3,
            )

            result = await coordinator.start_debate(
                task="What database should we use?",
            )

            # Verify event was published
            assert mock_event_bus.publish.called

    @pytest.mark.asyncio
    async def test_start_debate_with_agents(self, coordinator, mock_event_bus):
        """Test starting debate with specific agents."""
        await coordinator.connect()

        with patch.object(coordinator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = DistributedDebateResult(
                debate_id="test-debate",
                consensus_reached=True,
                final_answer="Use microservices",
                confidence=0.9,
                rounds_completed=2,
            )

            result = await coordinator.start_debate(
                task="Architecture decision",
                agents=["claude-3", "gpt-4", "gemini"],
            )

            # Verify agents were passed
            call_args = mock_run.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_start_debate_with_context(self, coordinator, mock_event_bus):
        """Test starting debate with context."""
        await coordinator.connect()

        with patch.object(coordinator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = DistributedDebateResult(
                debate_id="test-debate",
                consensus_reached=False,
                final_answer=None,
                confidence=0.0,
                rounds_completed=5,
            )

            result = await coordinator.start_debate(
                task="Design review",
                context={"codebase": "python", "team_size": 5},
            )

            assert result.rounds_completed == 5


class TestDistributedDebateCoordinatorEventHandling:
    """Tests for event handling."""

    @pytest.mark.asyncio
    async def test_handle_debate_event(self, coordinator):
        """Test handling debate events."""
        await coordinator.connect()

        # Create a mock event
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.AGENT_PROPOSAL,
            debate_id="debate-123",
            source_instance="other-instance",
            agent_id="claude-3",
            data={"content": "My proposal"},
        )

        # Should not raise
        await coordinator._handle_event(event)

    @pytest.mark.asyncio
    async def test_handle_event_from_self(self, coordinator):
        """Test ignoring events from self."""
        await coordinator.connect()

        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.AGENT_PROPOSAL,
            debate_id="debate-123",
            source_instance=coordinator._instance_id,  # Same instance
            agent_id="claude-3",
        )

        # Should be ignored (no error)
        await coordinator._handle_event(event)


class TestDistributedDebateCoordinatorState:
    """Tests for debate state management."""

    @pytest.mark.asyncio
    async def test_get_debate_state(self, coordinator):
        """Test getting debate state."""
        # Create a debate first
        coordinator._active_debates["debate-123"] = DistributedDebateState(
            debate_id="debate-123",
            task="Test task",
            coordinator_instance=coordinator._instance_id,
            status="running",
            current_round=2,
        )

        state = coordinator.get_debate_state("debate-123")

        assert state is not None
        assert state.debate_id == "debate-123"
        assert state.current_round == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_debate_state(self, coordinator):
        """Test getting state for nonexistent debate."""
        state = coordinator.get_debate_state("nonexistent")

        assert state is None

    @pytest.mark.asyncio
    async def test_list_active_debates(self, coordinator):
        """Test listing active debates."""
        coordinator._active_debates["debate-1"] = DistributedDebateState(
            debate_id="debate-1",
            task="Task 1",
            coordinator_instance=coordinator._instance_id,
        )
        coordinator._active_debates["debate-2"] = DistributedDebateState(
            debate_id="debate-2",
            task="Task 2",
            coordinator_instance=coordinator._instance_id,
        )

        debates = coordinator.list_active_debates()

        assert len(debates) == 2
        assert "debate-1" in debates
        assert "debate-2" in debates


class TestDistributedDebateCoordinatorConsensus:
    """Tests for consensus detection."""

    @pytest.mark.asyncio
    async def test_check_consensus_with_votes(self, coordinator):
        """Test checking consensus with votes."""
        from aragora.debate.distributed_events import ConsensusVote

        state = DistributedDebateState(
            debate_id="debate-123",
            task="Test",
            coordinator_instance=coordinator._instance_id,
            votes=[
                ConsensusVote(
                    agent_id="agent-1",
                    instance_id="i1",
                    proposal_agent_id="claude-3",
                    vote="support",
                    round_number=1,
                    confidence=0.9,
                ),
                ConsensusVote(
                    agent_id="agent-2",
                    instance_id="i1",
                    proposal_agent_id="claude-3",
                    vote="support",
                    round_number=1,
                    confidence=0.85,
                ),
                ConsensusVote(
                    agent_id="agent-3",
                    instance_id="i1",
                    proposal_agent_id="claude-3",
                    vote="support",
                    round_number=1,
                    confidence=0.8,
                ),
            ],
        )

        reached, confidence = coordinator._check_consensus(state)

        # With 3/3 support, consensus should be reached
        assert reached is True
        assert confidence > 0.8


class TestDistributedDebateResult:
    """Tests for DistributedDebateResult."""

    def test_result_creation(self):
        """Test creating a debate result."""
        result = DistributedDebateResult(
            debate_id="debate-123",
            consensus_reached=True,
            final_answer="Use PostgreSQL",
            confidence=0.92,
            rounds_completed=3,
            participating_agents=["claude-3", "gpt-4"],
            participating_instances=["instance-1", "instance-2"],
        )

        assert result.debate_id == "debate-123"
        assert result.consensus_reached is True
        assert result.final_answer == "Use PostgreSQL"
        assert result.confidence == 0.92
        assert result.rounds_completed == 3
        assert len(result.participating_agents) == 2

    def test_result_no_consensus(self):
        """Test result when no consensus reached."""
        result = DistributedDebateResult(
            debate_id="debate-456",
            consensus_reached=False,
            final_answer=None,
            confidence=0.0,
            rounds_completed=5,
        )

        assert result.consensus_reached is False
        assert result.final_answer is None
        assert result.confidence == 0.0

    def test_result_to_dict(self):
        """Test result serialization."""
        result = DistributedDebateResult(
            debate_id="debate-789",
            consensus_reached=True,
            final_answer="Microservices",
            confidence=0.88,
            rounds_completed=4,
        )

        data = result.to_dict()

        assert data["debate_id"] == "debate-789"
        assert data["consensus_reached"] is True
        assert data["final_answer"] == "Microservices"
        assert data["confidence"] == 0.88
