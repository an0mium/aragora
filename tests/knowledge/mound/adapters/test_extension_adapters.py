"""
Tests for Extension Module Adapters.

Tests the Knowledge Mound adapters for extension modules:
- FabricAdapter (Agent Fabric)
- WorkspaceAdapter (Workspace Manager)
- ComputerUseAdapter (Computer Use Orchestrator)
- GatewayAdapter (Local Gateway)
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters import (
    FabricAdapter,
    PoolSnapshot,
    TaskSchedulingOutcome,
    BudgetUsageSnapshot,
    PolicyDecisionRecord,
    WorkspaceAdapter,
    RigSnapshot,
    ConvoyOutcome,
    MergeOutcome,
    ComputerUseAdapter,
    TaskExecutionRecord,
    ActionPerformanceRecord,
    PolicyBlockRecord,
    GatewayAdapter,
    MessageRoutingRecord,
    ChannelPerformanceSnapshot,
    DeviceRegistrationRecord,
    RoutingDecisionRecord,
)
from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS


class TestAdapterRegistration:
    """Test that extension adapters are properly registered."""

    def test_fabric_adapter_registered(self):
        """FabricAdapter should be registered in ADAPTER_SPECS."""
        assert "fabric" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["fabric"]
        assert spec.forward_method == "sync_from_fabric"
        assert spec.reverse_method == "get_pool_recommendations"
        assert spec.priority == 35

    def test_workspace_adapter_registered(self):
        """WorkspaceAdapter should be registered in ADAPTER_SPECS."""
        assert "workspace" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["workspace"]
        assert spec.forward_method == "sync_from_workspace"
        assert spec.reverse_method == "get_rig_recommendations"
        assert spec.priority == 34

    def test_computer_use_adapter_registered(self):
        """ComputerUseAdapter should be registered in ADAPTER_SPECS."""
        assert "computer_use" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["computer_use"]
        assert spec.forward_method == "sync_from_orchestrator"
        assert spec.reverse_method == "get_similar_tasks"
        assert spec.priority == 33

    def test_gateway_adapter_registered(self):
        """GatewayAdapter should be registered in ADAPTER_SPECS."""
        assert "gateway" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["gateway"]
        assert spec.forward_method == "sync_from_gateway"
        assert spec.reverse_method == "get_routing_recommendations"
        assert spec.priority == 32


class TestFabricAdapter:
    """Test FabricAdapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter without dependencies."""
        return FabricAdapter(workspace_id="test-workspace")

    def test_initialization(self, adapter):
        """Should initialize with correct defaults."""
        assert adapter.adapter_name == "fabric"
        assert adapter._workspace_id == "test-workspace"
        assert adapter._knowledge_mound is None
        assert adapter._fabric is None

    def test_stats_initialization(self, adapter):
        """Should initialize stats correctly."""
        stats = adapter.get_stats()
        assert stats["pool_snapshots_stored"] == 0
        assert stats["task_outcomes_stored"] == 0
        assert stats["budget_snapshots_stored"] == 0
        assert stats["workspace_id"] == "test-workspace"

    @pytest.mark.asyncio
    async def test_store_pool_snapshot_without_km(self, adapter):
        """Should return None when no KM configured."""
        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Test Pool",
            model="claude-3-opus",
            current_agents=5,
            min_agents=2,
            max_agents=10,
        )
        result = await adapter.store_pool_snapshot(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_store_task_outcome_without_km(self, adapter):
        """Should return None when no KM configured."""
        outcome = TaskSchedulingOutcome(
            task_id="task-1",
            task_type="debate",
            agent_id="agent-1",
            pool_id="pool-1",
            priority=2,
            scheduled_at=time.time(),
            success=True,
            duration_seconds=10.0,
        )
        result = await adapter.store_task_outcome(outcome)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_pool_recommendations_empty(self, adapter):
        """Should return empty list when no KM configured."""
        result = await adapter.get_pool_recommendations("debate")
        assert result == []

    def test_clear_cache(self, adapter):
        """Should clear caches correctly."""
        adapter._pool_performance_cache["pool-1"] = []
        adapter._task_patterns_cache["debate"] = []
        count = adapter.clear_cache()
        assert count == 2
        assert len(adapter._pool_performance_cache) == 0


class TestWorkspaceAdapter:
    """Test WorkspaceAdapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter without dependencies."""
        return WorkspaceAdapter(workspace_id="test-workspace")

    def test_initialization(self, adapter):
        """Should initialize with correct defaults."""
        assert adapter.adapter_name == "workspace"
        assert adapter._workspace_id == "test-workspace"

    def test_stats_initialization(self, adapter):
        """Should initialize stats correctly."""
        stats = adapter.get_stats()
        assert stats["rig_snapshots_stored"] == 0
        assert stats["convoy_outcomes_stored"] == 0
        assert stats["merge_outcomes_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_without_km(self, adapter):
        """Should return None when no KM configured."""
        snapshot = RigSnapshot(
            rig_id="rig-1",
            name="Backend Rig",
            workspace_id="test-workspace",
            status="active",
            assigned_agents=3,
            max_agents=10,
        )
        result = await adapter.store_rig_snapshot(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_without_km(self, adapter):
        """Should return None when no KM configured."""
        outcome = ConvoyOutcome(
            convoy_id="conv-1",
            workspace_id="test-workspace",
            rig_id="rig-1",
            name="Feature Convoy",
            status="done",
            total_beads=5,
            completed_beads=5,
            duration_seconds=300.0,
        )
        result = await adapter.store_convoy_outcome(outcome)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_empty(self, adapter):
        """Should return empty list when no KM configured."""
        result = await adapter.get_rig_recommendations("backend")
        assert result == []


class TestComputerUseAdapter:
    """Test ComputerUseAdapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter without dependencies."""
        return ComputerUseAdapter(workspace_id="test-workspace")

    def test_initialization(self, adapter):
        """Should initialize with correct defaults."""
        assert adapter.adapter_name == "computer_use"
        assert adapter._workspace_id == "test-workspace"

    def test_stats_initialization(self, adapter):
        """Should initialize stats correctly."""
        stats = adapter.get_stats()
        assert stats["task_records_stored"] == 0
        assert stats["action_records_stored"] == 0
        assert stats["policy_blocks_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_task_execution_record_without_km(self, adapter):
        """Should return None when no KM configured."""
        record = TaskExecutionRecord(
            task_id="task-1",
            goal="Open settings and enable dark mode",
            status="completed",
            total_steps=5,
            successful_steps=5,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=30.0,
        )
        result = await adapter.store_task_execution_record(record)
        assert result is None

    @pytest.mark.asyncio
    async def test_store_action_performance_without_km(self, adapter):
        """Should return None when no KM configured."""
        record = ActionPerformanceRecord(
            action_type="click",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            avg_duration_ms=150.0,
        )
        result = await adapter.store_action_performance(record)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_similar_tasks_empty(self, adapter):
        """Should return empty list when no KM configured."""
        result = await adapter.get_similar_tasks("open settings")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_task_recommendations_no_history(self, adapter):
        """Should return no-history recommendation when no similar tasks."""
        result = await adapter.get_task_recommendations("new task")
        assert len(result) == 1
        assert result[0]["confidence"] == 0.0


class TestGatewayAdapter:
    """Test GatewayAdapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter without dependencies."""
        return GatewayAdapter(workspace_id="test-workspace")

    def test_initialization(self, adapter):
        """Should initialize with correct defaults."""
        assert adapter.adapter_name == "gateway"
        assert adapter._workspace_id == "test-workspace"

    def test_stats_initialization(self, adapter):
        """Should initialize stats correctly."""
        stats = adapter.get_stats()
        assert stats["routing_records_stored"] == 0
        assert stats["channel_snapshots_stored"] == 0
        assert stats["device_records_stored"] == 0

    @pytest.mark.asyncio
    async def test_store_routing_record_without_km(self, adapter):
        """Should return None when no KM configured."""
        record = MessageRoutingRecord(
            message_id="msg-1",
            channel="slack",
            sender="user@example.com",
            agent_id="support-agent",
            routing_rule="support-rule",
            success=True,
            latency_ms=50.0,
        )
        result = await adapter.store_routing_record(record)
        assert result is None

    @pytest.mark.asyncio
    async def test_store_channel_snapshot_without_km(self, adapter):
        """Should return None when no KM configured."""
        snapshot = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=1000,
            messages_routed=980,
            messages_failed=20,
            avg_latency_ms=45.0,
            active_threads=50,
            unique_senders=200,
        )
        result = await adapter.store_channel_snapshot(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_store_device_registration_without_km(self, adapter):
        """Should return None when no KM configured."""
        record = DeviceRegistrationRecord(
            device_id="device-1",
            device_name="MacBook Pro",
            device_type="laptop",
            status="online",
            capabilities=["browser", "terminal", "camera"],
            registered_at=time.time(),
            last_seen=time.time(),
        )
        result = await adapter.store_device_registration(record)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_routing_recommendations_no_history(self, adapter):
        """Should return no-history recommendation when no patterns."""
        result = await adapter.get_routing_recommendations("slack")
        assert len(result) == 1
        assert "No routing history" in result[0]["recommendation"]


class TestDataclassRecords:
    """Test dataclass record structures."""

    def test_pool_snapshot_defaults(self):
        """PoolSnapshot should have correct defaults."""
        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Test",
            model="opus",
            current_agents=5,
            min_agents=1,
            max_agents=10,
        )
        assert snapshot.tasks_pending == 0
        assert snapshot.tasks_completed == 0
        assert snapshot.avg_task_duration_seconds == 0.0
        assert snapshot.workspace_id == "default"

    def test_rig_snapshot_defaults(self):
        """RigSnapshot should have correct defaults."""
        snapshot = RigSnapshot(
            rig_id="rig-1",
            name="Test",
            workspace_id="ws-1",
            status="ready",
        )
        assert snapshot.repo_url == ""
        assert snapshot.branch == "main"
        assert snapshot.assigned_agents == 0
        assert snapshot.max_agents == 10

    def test_task_execution_record_defaults(self):
        """TaskExecutionRecord should have correct defaults."""
        record = TaskExecutionRecord(
            task_id="task-1",
            goal="Test goal",
            status="completed",
            total_steps=5,
            successful_steps=5,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=30.0,
        )
        assert record.agent_id is None
        assert record.error_message is None
        assert record.workspace_id == "default"

    def test_message_routing_record_defaults(self):
        """MessageRoutingRecord should have correct defaults."""
        record = MessageRoutingRecord(
            message_id="msg-1",
            channel="slack",
            sender="user",
            agent_id=None,
            routing_rule=None,
            success=True,
            latency_ms=50.0,
        )
        assert record.priority == "normal"
        assert record.thread_id is None
        assert record.error_message is None


class TestAdapterCaching:
    """Test adapter caching behavior."""

    @pytest.fixture
    def fabric_adapter(self):
        """Create FabricAdapter."""
        return FabricAdapter(workspace_id="test")

    @pytest.fixture
    def workspace_adapter(self):
        """Create WorkspaceAdapter."""
        return WorkspaceAdapter(workspace_id="test")

    def test_fabric_cache_clear(self, fabric_adapter):
        """Should clear all caches correctly."""
        fabric_adapter._pool_performance_cache["pool-1"] = [
            PoolSnapshot("p1", "Pool 1", "opus", 5, 1, 10)
        ]
        fabric_adapter._task_patterns_cache["debate"] = []
        fabric_adapter._cache_times["pool-1"] = time.time()

        count = fabric_adapter.clear_cache()
        assert count == 2
        assert len(fabric_adapter._pool_performance_cache) == 0
        assert len(fabric_adapter._task_patterns_cache) == 0
        assert len(fabric_adapter._cache_times) == 0

    def test_workspace_cache_clear(self, workspace_adapter):
        """Should clear all caches correctly."""
        workspace_adapter._rig_performance_cache["rig-1"] = []
        workspace_adapter._convoy_patterns_cache["rig-1"] = []

        count = workspace_adapter.clear_cache()
        assert count == 2


class TestHealthAndStats:
    """Test adapter health check and stats."""

    def test_fabric_health_check(self):
        """FabricAdapter should provide health status."""
        adapter = FabricAdapter()
        health = adapter.health_check()
        assert "adapter" in health
        assert health["adapter"] == "fabric"
        assert "healthy" in health

    def test_workspace_health_check(self):
        """WorkspaceAdapter should provide health status."""
        adapter = WorkspaceAdapter()
        health = adapter.health_check()
        assert health["adapter"] == "workspace"

    def test_computer_use_health_check(self):
        """ComputerUseAdapter should provide health status."""
        adapter = ComputerUseAdapter()
        health = adapter.health_check()
        assert health["adapter"] == "computer_use"

    def test_gateway_health_check(self):
        """GatewayAdapter should provide health status."""
        adapter = GatewayAdapter()
        health = adapter.health_check()
        assert health["adapter"] == "gateway"
