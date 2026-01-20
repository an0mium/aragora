"""
Tests for Control Plane Integration module.

Tests the IntegratedControlPlane that bridges ControlPlaneCoordinator
with SharedControlPlaneState.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.registry import AgentCapability, AgentInfo, AgentStatus
from aragora.control_plane.scheduler import Task, TaskPriority, TaskStatus


class TestIntegratedControlPlane:
    """Tests for IntegratedControlPlane."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator."""
        coordinator = MagicMock()
        coordinator.register_agent = AsyncMock()
        coordinator.unregister_agent = AsyncMock(return_value=True)
        coordinator.heartbeat = AsyncMock(return_value=True)
        coordinator.list_agents = AsyncMock(return_value=[])
        coordinator.get_agent = AsyncMock(return_value=None)
        coordinator.submit_task = AsyncMock(return_value="task-123")
        coordinator.complete_task = AsyncMock(return_value=True)
        coordinator.fail_task = AsyncMock(return_value=True)
        coordinator.get_task = AsyncMock(return_value=None)
        coordinator.wait_for_result = AsyncMock(return_value=None)
        coordinator.get_stats = AsyncMock(return_value={})
        coordinator.shutdown = AsyncMock()
        return coordinator

    @pytest.fixture
    def mock_shared_state(self):
        """Create mock shared state."""
        shared = MagicMock()
        shared.register_agent = AsyncMock()
        shared.update_agent_status = AsyncMock(return_value={})
        shared.list_agents = AsyncMock(return_value=[])
        shared.add_task = AsyncMock()
        shared.update_task_priority = AsyncMock(return_value={})
        shared.record_agent_activity = AsyncMock()
        shared.get_metrics = AsyncMock(return_value={})
        shared._broadcast_event = AsyncMock()
        shared.is_persistent = True
        shared.close = AsyncMock()
        return shared

    @pytest.fixture
    def integrated(self, mock_coordinator, mock_shared_state):
        """Create IntegratedControlPlane instance."""
        from aragora.control_plane.integration import IntegratedControlPlane

        return IntegratedControlPlane(
            coordinator=mock_coordinator,
            shared_state=mock_shared_state,
            sync_interval=1.0,
        )

    @pytest.mark.asyncio
    async def test_register_agent_syncs_to_shared_state(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that registering agent syncs to shared state."""
        # Set up mock agent
        mock_agent = MagicMock(spec=AgentInfo)
        mock_agent.agent_id = "test-agent"
        mock_agent.model = "test-model"
        mock_agent.provider = "test-provider"
        mock_agent.status = AgentStatus.READY
        mock_agent.capabilities = [AgentCapability.DEBATE]
        mock_agent.tasks_completed = 5
        mock_agent.avg_latency_ms = 100.0
        mock_agent.success_rate = 0.95
        mock_agent.last_heartbeat = datetime.utcnow()
        mock_agent.metadata = {}

        mock_coordinator.register_agent.return_value = mock_agent

        # Register agent
        result = await integrated.register_agent(
            agent_id="test-agent",
            capabilities=["debate"],
            model="test-model",
            provider="test-provider",
        )

        # Verify coordinator was called
        mock_coordinator.register_agent.assert_called_once()

        # Verify shared state was synced
        mock_shared_state.register_agent.assert_called()
        mock_shared_state._broadcast_event.assert_called()

    @pytest.mark.asyncio
    async def test_unregister_agent_updates_shared_state(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that unregistering agent updates shared state."""
        result = await integrated.unregister_agent("test-agent")

        assert result is True
        mock_coordinator.unregister_agent.assert_called_once_with("test-agent")
        mock_shared_state.update_agent_status.assert_called_with("test-agent", "offline")
        mock_shared_state._broadcast_event.assert_called()

    @pytest.mark.asyncio
    async def test_pause_agent(self, integrated, mock_coordinator, mock_shared_state):
        """Test pausing an agent."""
        result = await integrated.pause_agent("test-agent")

        assert result is True
        mock_coordinator.heartbeat.assert_called_with(
            "test-agent", status=AgentStatus.DRAINING
        )
        mock_shared_state.update_agent_status.assert_called_with("test-agent", "paused")

    @pytest.mark.asyncio
    async def test_resume_agent(self, integrated, mock_coordinator, mock_shared_state):
        """Test resuming an agent."""
        result = await integrated.resume_agent("test-agent")

        assert result is True
        mock_coordinator.heartbeat.assert_called_with(
            "test-agent", status=AgentStatus.READY
        )
        mock_shared_state.update_agent_status.assert_called_with("test-agent", "active")

    @pytest.mark.asyncio
    async def test_submit_task_syncs_to_shared_state(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that submitting task syncs to shared state."""
        task_id = await integrated.submit_task(
            task_type="debate",
            payload={"question": "test"},
            priority=TaskPriority.HIGH,
        )

        assert task_id == "task-123"
        mock_coordinator.submit_task.assert_called_once()
        mock_shared_state.add_task.assert_called_once()

        # Verify task data
        call_args = mock_shared_state.add_task.call_args[0][0]
        assert call_args["id"] == "task-123"
        assert call_args["type"] == "debate"
        assert call_args["priority"] == "high"
        assert call_args["status"] == "pending"

    @pytest.mark.asyncio
    async def test_complete_task_updates_shared_state(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that completing task updates shared state."""
        result = await integrated.complete_task(
            task_id="task-123",
            result={"answer": "test"},
            agent_id="agent-1",
            latency_ms=150.0,
        )

        assert result is True
        mock_coordinator.complete_task.assert_called_once()
        mock_shared_state.record_agent_activity.assert_called_with(
            "agent-1", tasks_completed=1, response_time_ms=150.0
        )
        mock_shared_state._broadcast_event.assert_called()

    @pytest.mark.asyncio
    async def test_fail_task_records_error(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that failing task records error in shared state."""
        result = await integrated.fail_task(
            task_id="task-123",
            error="Test error",
            agent_id="agent-1",
            latency_ms=50.0,
        )

        assert result is True
        mock_shared_state.record_agent_activity.assert_called_with(
            "agent-1", response_time_ms=50.0, error=True
        )

    @pytest.mark.asyncio
    async def test_get_stats_combines_both_systems(
        self, integrated, mock_coordinator, mock_shared_state
    ):
        """Test that stats combine coordinator and shared state."""
        mock_coordinator.get_stats.return_value = {"registry": {}}
        mock_shared_state.get_metrics.return_value = {"agents": {}}

        stats = await integrated.get_stats()

        assert "coordinator" in stats
        assert "shared_state" in stats
        assert "integrated" in stats
        assert stats["integrated"]["persistent_backend"] is True

    @pytest.mark.asyncio
    async def test_start_and_stop(self, integrated):
        """Test starting and stopping the integration."""
        await integrated.start()
        assert integrated._running is True
        assert integrated._sync_task is not None

        await integrated.stop()
        assert integrated._running is False

    @pytest.mark.asyncio
    async def test_agent_status_mapping(self, integrated):
        """Test agent status mapping from coordinator to shared state."""
        assert integrated._map_agent_status(AgentStatus.STARTING) == "idle"
        assert integrated._map_agent_status(AgentStatus.READY) == "active"
        assert integrated._map_agent_status(AgentStatus.BUSY) == "active"
        assert integrated._map_agent_status(AgentStatus.DRAINING) == "paused"
        assert integrated._map_agent_status(AgentStatus.OFFLINE) == "offline"
        assert integrated._map_agent_status(AgentStatus.FAILED) == "offline"


class TestSetupIntegration:
    """Tests for setup_control_plane_integration function."""

    @pytest.mark.asyncio
    async def test_setup_creates_integrated_instance(self):
        """Test that setup creates and returns integrated instance."""
        with patch(
            "aragora.control_plane.integration.ControlPlaneCoordinator"
        ) as mock_coordinator_class, patch(
            "aragora.control_plane.integration.SharedControlPlaneState"
        ) as mock_shared_class, patch(
            "aragora.control_plane.integration.set_shared_state"
        ):
            # Set up mocks
            mock_coordinator = MagicMock()
            mock_coordinator_class.create = AsyncMock(return_value=mock_coordinator)

            mock_shared = MagicMock()
            mock_shared.connect = AsyncMock(return_value=True)
            mock_shared.is_persistent = True
            mock_shared_class.return_value = mock_shared

            # Reset module state
            import aragora.control_plane.integration as integration_module
            integration_module._integrated = None

            from aragora.control_plane.integration import setup_control_plane_integration

            result = await setup_control_plane_integration()

            assert result is not None
            mock_coordinator_class.create.assert_called_once()
            mock_shared.connect.assert_called_once()

            # Clean up
            integration_module._integrated = None

    @pytest.mark.asyncio
    async def test_get_integrated_control_plane_returns_singleton(self):
        """Test that get_integrated_control_plane returns singleton."""
        import aragora.control_plane.integration as integration_module
        from aragora.control_plane.integration import (
            get_integrated_control_plane,
            IntegratedControlPlane,
        )

        # Initially None
        integration_module._integrated = None
        assert get_integrated_control_plane() is None

        # After setting, returns instance
        mock_integrated = MagicMock(spec=IntegratedControlPlane)
        integration_module._integrated = mock_integrated
        assert get_integrated_control_plane() is mock_integrated

        # Clean up
        integration_module._integrated = None


class TestSyncLoop:
    """Tests for the sync loop functionality."""

    @pytest.mark.asyncio
    async def test_sync_loop_syncs_agents(self):
        """Test that sync loop syncs agents from coordinator to shared state."""
        from aragora.control_plane.integration import IntegratedControlPlane

        mock_coordinator = MagicMock()
        mock_shared = MagicMock()

        # Set up mock agent
        mock_agent = MagicMock(spec=AgentInfo)
        mock_agent.agent_id = "sync-agent"
        mock_agent.model = "model"
        mock_agent.provider = "provider"
        mock_agent.status = AgentStatus.READY
        mock_agent.capabilities = []
        mock_agent.tasks_completed = 0
        mock_agent.avg_latency_ms = 0.0
        mock_agent.success_rate = 1.0
        mock_agent.last_heartbeat = datetime.utcnow()
        mock_agent.metadata = {}

        mock_coordinator.list_agents = AsyncMock(return_value=[mock_agent])
        mock_shared.register_agent = AsyncMock()

        integrated = IntegratedControlPlane(
            coordinator=mock_coordinator,
            shared_state=mock_shared,
            sync_interval=0.1,
        )

        # Run sync manually
        await integrated._sync_agents()

        # Verify agent was synced
        mock_coordinator.list_agents.assert_called_once_with(only_available=False)
        mock_shared.register_agent.assert_called_once()

        # Verify agent data
        call_args = mock_shared.register_agent.call_args[0][0]
        assert call_args["id"] == "sync-agent"
        assert call_args["status"] == "active"
