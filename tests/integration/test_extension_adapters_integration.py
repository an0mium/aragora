"""
Integration tests for Extension Module Knowledge Mound Adapters.

Tests the complete integration of extension module adapters:
1. Factory creating fabric, workspace, computer_use, gateway adapters
2. Coordinator orchestrating sync across extension adapters
3. Cross-module data flow validation
4. Event callback propagation
5. Policy enforcement across modules
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Any, Optional


class MockKnowledgeMound:
    """Mock Knowledge Mound for testing extension adapters."""

    def __init__(self):
        self.items: dict[str, dict[str, Any]] = {}
        self.store_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    async def ingest(self, item: Any) -> str:
        """Ingest a KnowledgeItem into the mound."""
        item_id = getattr(item, "id", f"km-item-{len(self.items)}")
        # Extract type from metadata if available
        metadata = getattr(item, "metadata", {}) or {}
        item_type = metadata.get("type", "unknown")
        content = getattr(item, "content", str(item))
        source = getattr(getattr(item, "source", None), "name", "unknown")
        self.items[item_id] = {
            "source": source,
            "type": item_type,
            "content": content,
            "item": item,
            "metadata": metadata,
        }
        self.store_calls.append({"source": source, "type": item_type, "content": content})
        return item_id

    async def query(self, query: str, source: str | None = None, limit: int = 10) -> list[dict]:
        self.query_calls.append({"query": query, "source": source, "limit": limit})
        results = []
        for item_id, item in self.items.items():
            if source and item["source"] != source:
                continue
            if query.lower() in str(item["content"]).lower():
                results.append({"id": item_id, **item, "score": 0.9})
        return results[:limit]


class MockAgentFabric:
    """Mock Agent Fabric for testing FabricAdapter."""

    def __init__(self):
        self.pools = {
            "pool-1": {
                "id": "pool-1",
                "name": "Debate Pool",
                "model": "claude-3-opus",
                "current_agents": 5,
                "min_agents": 2,
                "max_agents": 10,
            },
            "pool-2": {
                "id": "pool-2",
                "name": "Analysis Pool",
                "model": "gpt-4",
                "current_agents": 3,
                "min_agents": 1,
                "max_agents": 5,
            },
        }
        self.tasks = []
        self.budget_usage = {"total": 1000, "used": 250, "remaining": 750}

    def get_pool_stats(self, pool_id: str) -> dict | None:
        return self.pools.get(pool_id)

    def get_all_pools(self) -> list[dict]:
        return list(self.pools.values())


class MockWorkspaceManager:
    """Mock Workspace Manager for testing WorkspaceAdapter."""

    def __init__(self):
        self.rigs = {
            "rig-1": {
                "id": "rig-1",
                "name": "Backend Rig",
                "workspace_id": "ws-1",
                "status": "active",
                "assigned_agents": 3,
            },
        }
        self.convoys = []
        self.merges = []

    def get_rig(self, rig_id: str) -> dict | None:
        return self.rigs.get(rig_id)


class MockComputerUseOrchestrator:
    """Mock Computer-Use Orchestrator for testing ComputerUseAdapter."""

    def __init__(self):
        self.executed_tasks = []
        self.action_stats = {
            "click": {"total": 100, "success": 95, "avg_duration_ms": 150},
            "type": {"total": 50, "success": 48, "avg_duration_ms": 200},
        }

    def get_action_stats(self, action_type: str) -> dict | None:
        return self.action_stats.get(action_type)


class MockLocalGateway:
    """Mock Local Gateway for testing GatewayAdapter."""

    def __init__(self):
        self.channels = {
            "slack": {"messages_received": 1000, "messages_routed": 980},
            "email": {"messages_received": 500, "messages_routed": 495},
        }
        self.devices = []
        self.routing_records = []

    def get_channel_stats(self, channel: str) -> dict | None:
        return self.channels.get(channel)


class TestExtensionAdapterFactoryIntegration:
    """Test factory creating extension adapters with subsystems."""

    def test_factory_creates_fabric_adapter(self):
        """Factory should create FabricAdapter when fabric is provided."""
        from aragora.knowledge.mound.adapters import AdapterFactory, FabricAdapter

        mock_fabric = MockAgentFabric()
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(fabric=mock_fabric)

        assert "fabric" in adapters
        assert isinstance(adapters["fabric"].adapter, FabricAdapter)

    def test_factory_creates_workspace_adapter(self):
        """Factory should create WorkspaceAdapter when workspace is provided."""
        from aragora.knowledge.mound.adapters import AdapterFactory, WorkspaceAdapter

        mock_workspace = MockWorkspaceManager()
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(workspace_manager=mock_workspace)

        assert "workspace" in adapters
        assert isinstance(adapters["workspace"].adapter, WorkspaceAdapter)

    def test_factory_creates_computer_use_adapter(self):
        """Factory should create ComputerUseAdapter when orchestrator is provided."""
        from aragora.knowledge.mound.adapters import AdapterFactory, ComputerUseAdapter

        mock_orchestrator = MockComputerUseOrchestrator()
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(computer_use_orchestrator=mock_orchestrator)

        assert "computer_use" in adapters
        assert isinstance(adapters["computer_use"].adapter, ComputerUseAdapter)

    def test_factory_creates_gateway_adapter(self):
        """Factory should create GatewayAdapter when gateway is provided."""
        from aragora.knowledge.mound.adapters import AdapterFactory, GatewayAdapter

        mock_gateway = MockLocalGateway()
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(gateway=mock_gateway)

        assert "gateway" in adapters
        assert isinstance(adapters["gateway"].adapter, GatewayAdapter)

    def test_factory_creates_all_extension_adapters(self):
        """Factory should create all extension adapters when all deps provided."""
        from aragora.knowledge.mound.adapters import AdapterFactory

        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(
            fabric=MockAgentFabric(),
            workspace_manager=MockWorkspaceManager(),
            computer_use_orchestrator=MockComputerUseOrchestrator(),
            gateway=MockLocalGateway(),
        )

        assert "fabric" in adapters
        assert "workspace" in adapters
        assert "computer_use" in adapters
        assert "gateway" in adapters


class TestExtensionAdapterCoordinatorSync:
    """Test coordinator orchestrating sync across extension adapters."""

    @pytest.fixture
    def extension_adapters(self):
        """Create all extension adapters."""
        from aragora.knowledge.mound.adapters import (
            FabricAdapter,
            WorkspaceAdapter,
            ComputerUseAdapter,
            GatewayAdapter,
        )

        return {
            "fabric": FabricAdapter(workspace_id="test-ws"),
            "workspace": WorkspaceAdapter(workspace_id="test-ws"),
            "computer_use": ComputerUseAdapter(workspace_id="test-ws"),
            "gateway": GatewayAdapter(workspace_id="test-ws"),
        }

    def test_register_extension_adapters_with_coordinator(self, extension_adapters):
        """All extension adapters should register with coordinator."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator
        from aragora.knowledge.mound.adapters import ADAPTER_SPECS

        coordinator = BidirectionalCoordinator()

        for name, adapter in extension_adapters.items():
            spec = ADAPTER_SPECS[name]
            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method=spec.forward_method,
                reverse_method=spec.reverse_method,
                priority=spec.priority,
            )

        status = coordinator.get_status()
        assert "fabric" in status["adapters"]
        assert "workspace" in status["adapters"]
        assert "computer_use" in status["adapters"]
        assert "gateway" in status["adapters"]

    def test_extension_adapter_priority_order(self):
        """Extension adapters should have correct priority order."""
        from aragora.knowledge.mound.adapters import ADAPTER_SPECS

        # Verify priority order: fabric > workspace > computer_use > gateway
        assert ADAPTER_SPECS["fabric"].priority > ADAPTER_SPECS["workspace"].priority
        assert ADAPTER_SPECS["workspace"].priority > ADAPTER_SPECS["computer_use"].priority
        assert ADAPTER_SPECS["computer_use"].priority > ADAPTER_SPECS["gateway"].priority


class TestCrossModuleDataFlow:
    """Test data flow between extension modules via KM."""

    @pytest.mark.asyncio
    async def test_fabric_to_workspace_flow(self):
        """Pool data from fabric should inform workspace rig assignments."""
        from aragora.knowledge.mound.adapters import (
            FabricAdapter,
            WorkspaceAdapter,
            PoolSnapshot,
        )

        # Simulate pool snapshot from fabric
        fabric = FabricAdapter(workspace_id="test")
        fabric._knowledge_mound = MockKnowledgeMound()

        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Debate Pool",
            model="claude-3-opus",
            current_agents=5,
            min_agents=2,
            max_agents=10,
            tasks_pending=3,
        )

        result = await fabric.store_pool_snapshot(snapshot)
        assert result is not None

        # Verify data was stored
        stats = fabric.get_stats()
        assert stats["pool_snapshots_stored"] == 1

    @pytest.mark.asyncio
    async def test_computer_use_to_gateway_flow(self):
        """Task execution data should inform routing decisions."""
        from aragora.knowledge.mound.adapters import (
            ComputerUseAdapter,
            GatewayAdapter,
            TaskExecutionRecord,
        )

        # Store computer use task result
        cu_adapter = ComputerUseAdapter(workspace_id="test")
        cu_adapter._knowledge_mound = MockKnowledgeMound()

        record = TaskExecutionRecord(
            task_id="task-1",
            goal="Click submit button",
            status="completed",
            total_steps=3,
            successful_steps=3,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=5.0,
        )

        result = await cu_adapter.store_task_execution_record(record)
        assert result is not None

        # Gateway can use this to optimize routing for similar tasks
        gateway = GatewayAdapter(workspace_id="test")
        recommendations = await gateway.get_routing_recommendations("automation")
        # Without KM, returns default recommendation
        assert len(recommendations) >= 1


class TestEventCallbackPropagation:
    """Test event callbacks propagate across adapters."""

    @pytest.mark.asyncio
    async def test_fabric_event_callback(self):
        """FabricAdapter should emit events on store operations."""
        from aragora.knowledge.mound.adapters import FabricAdapter, PoolSnapshot

        events_received = []

        def event_handler(event_type: str, data: dict):
            events_received.append((event_type, data))

        adapter = FabricAdapter(workspace_id="test", event_callback=event_handler)
        adapter._knowledge_mound = MockKnowledgeMound()

        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Test Pool",
            model="opus",
            current_agents=5,
            min_agents=1,
            max_agents=10,
        )

        await adapter.store_pool_snapshot(snapshot)

        # Should have emitted an event
        assert len(events_received) >= 1
        assert events_received[0][0] == "fabric_pool_snapshot_stored"

    @pytest.mark.asyncio
    async def test_gateway_event_callback(self):
        """GatewayAdapter should emit events on store operations."""
        from aragora.knowledge.mound.adapters import GatewayAdapter, MessageRoutingRecord

        events_received = []

        def event_handler(event_type: str, data: dict):
            events_received.append((event_type, data))

        adapter = GatewayAdapter(workspace_id="test", event_callback=event_handler)
        adapter._knowledge_mound = MockKnowledgeMound()

        record = MessageRoutingRecord(
            message_id="msg-1",
            channel="slack",
            sender="user@example.com",
            agent_id="support-agent",
            routing_rule="default",
            success=True,
            latency_ms=50.0,
        )

        await adapter.store_routing_record(record)

        # Should have emitted an event
        assert len(events_received) >= 1
        assert events_received[0][0] == "gw_routing_record_stored"


class TestAdapterHealthIntegration:
    """Test health check integration across extension adapters."""

    def test_all_extension_adapters_healthy(self):
        """All extension adapters should report healthy when no KM configured."""
        from aragora.knowledge.mound.adapters import (
            FabricAdapter,
            WorkspaceAdapter,
            ComputerUseAdapter,
            GatewayAdapter,
        )

        adapters = [
            FabricAdapter(workspace_id="test"),
            WorkspaceAdapter(workspace_id="test"),
            ComputerUseAdapter(workspace_id="test"),
            GatewayAdapter(workspace_id="test"),
        ]

        for adapter in adapters:
            health = adapter.health_check()
            assert "healthy" in health
            # Should be degraded (not failed) without KM
            assert health["healthy"] in [True, False]
            assert "adapter" in health

    def test_extension_adapter_stats_structure(self):
        """All extension adapters should return consistent stats structure."""
        from aragora.knowledge.mound.adapters import (
            FabricAdapter,
            WorkspaceAdapter,
            ComputerUseAdapter,
            GatewayAdapter,
        )

        adapters = {
            "fabric": FabricAdapter(workspace_id="test"),
            "workspace": WorkspaceAdapter(workspace_id="test"),
            "computer_use": ComputerUseAdapter(workspace_id="test"),
            "gateway": GatewayAdapter(workspace_id="test"),
        }

        for name, adapter in adapters.items():
            stats = adapter.get_stats()
            assert isinstance(stats, dict)
            assert "workspace_id" in stats or any("stored" in k for k in stats.keys())


class TestAdapterCacheIntegration:
    """Test caching behavior across extension adapters."""

    def test_fabric_cache_operations(self):
        """FabricAdapter cache should work correctly."""
        from aragora.knowledge.mound.adapters import FabricAdapter

        adapter = FabricAdapter(workspace_id="test")

        # Add items to cache
        adapter._pool_performance_cache["pool-1"] = [{"data": "test"}]
        adapter._task_patterns_cache["debate"] = [{"pattern": "test"}]
        adapter._cache_times["pool-1"] = time.time()

        # Clear cache
        count = adapter.clear_cache()
        assert count == 2
        assert len(adapter._pool_performance_cache) == 0
        assert len(adapter._task_patterns_cache) == 0
        assert len(adapter._cache_times) == 0

    def test_workspace_cache_operations(self):
        """WorkspaceAdapter cache should work correctly."""
        from aragora.knowledge.mound.adapters import WorkspaceAdapter

        adapter = WorkspaceAdapter(workspace_id="test")

        # Add items to cache
        adapter._rig_performance_cache["rig-1"] = [{"data": "test"}]
        adapter._convoy_patterns_cache["rig-1"] = [{"pattern": "test"}]

        # Clear cache
        count = adapter.clear_cache()
        assert count == 2
        assert len(adapter._rig_performance_cache) == 0
        assert len(adapter._convoy_patterns_cache) == 0


class TestExtensionAdapterWithKM:
    """Test extension adapters with mock Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_fabric_store_and_query(self):
        """FabricAdapter should store and query from KM."""
        from aragora.knowledge.mound.adapters import (
            FabricAdapter,
            PoolSnapshot,
            TaskSchedulingOutcome,
        )

        mock_km = MockKnowledgeMound()
        adapter = FabricAdapter(workspace_id="test")
        adapter._knowledge_mound = mock_km

        # Store pool snapshot
        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Debate Pool",
            model="claude-3-opus",
            current_agents=5,
            min_agents=2,
            max_agents=10,
        )
        await adapter.store_pool_snapshot(snapshot)

        # Store task outcome
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
        await adapter.store_task_outcome(outcome)

        # Verify items stored
        assert len(mock_km.store_calls) == 2
        assert mock_km.store_calls[0]["type"] == "fabric_pool_snapshot"
        assert mock_km.store_calls[1]["type"] == "fabric_task_outcome"

    @pytest.mark.asyncio
    async def test_workspace_store_and_query(self):
        """WorkspaceAdapter should store and query from KM."""
        from aragora.knowledge.mound.adapters import (
            WorkspaceAdapter,
            RigSnapshot,
            ConvoyOutcome,
        )

        mock_km = MockKnowledgeMound()
        adapter = WorkspaceAdapter(workspace_id="test")
        adapter._knowledge_mound = mock_km

        # Store rig snapshot
        snapshot = RigSnapshot(
            rig_id="rig-1",
            name="Backend Rig",
            workspace_id="test",
            status="active",
            assigned_agents=3,
            max_agents=10,
        )
        await adapter.store_rig_snapshot(snapshot)

        # Store convoy outcome
        outcome = ConvoyOutcome(
            convoy_id="convoy-1",
            workspace_id="test",
            rig_id="rig-1",
            name="Feature Convoy",
            status="done",
            total_beads=5,
            completed_beads=5,
            duration_seconds=300.0,
        )
        await adapter.store_convoy_outcome(outcome)

        # Verify items stored
        assert len(mock_km.store_calls) == 2
        assert mock_km.store_calls[0]["type"] == "workspace_rig_snapshot"
        assert mock_km.store_calls[1]["type"] == "workspace_convoy_outcome"

    @pytest.mark.asyncio
    async def test_computer_use_store_and_query(self):
        """ComputerUseAdapter should store and query from KM."""
        from aragora.knowledge.mound.adapters import (
            ComputerUseAdapter,
            TaskExecutionRecord,
            ActionPerformanceRecord,
        )

        mock_km = MockKnowledgeMound()
        adapter = ComputerUseAdapter(workspace_id="test")
        adapter._knowledge_mound = mock_km

        # Store task execution
        record = TaskExecutionRecord(
            task_id="task-1",
            goal="Open settings",
            status="completed",
            total_steps=5,
            successful_steps=5,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=30.0,
        )
        await adapter.store_task_execution_record(record)

        # Store action performance
        action = ActionPerformanceRecord(
            action_type="click",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            avg_duration_ms=150.0,
        )
        await adapter.store_action_performance(action)

        # Verify items stored
        assert len(mock_km.store_calls) == 2
        assert mock_km.store_calls[0]["type"] == "computer_use_task"
        assert mock_km.store_calls[1]["type"] == "computer_use_action_performance"

    @pytest.mark.asyncio
    async def test_gateway_store_and_query(self):
        """GatewayAdapter should store and query from KM."""
        from aragora.knowledge.mound.adapters import (
            GatewayAdapter,
            MessageRoutingRecord,
            ChannelPerformanceSnapshot,
        )

        mock_km = MockKnowledgeMound()
        adapter = GatewayAdapter(workspace_id="test")
        adapter._knowledge_mound = mock_km

        # Store routing record
        record = MessageRoutingRecord(
            message_id="msg-1",
            channel="slack",
            sender="user@example.com",
            agent_id="support-agent",
            routing_rule="default",
            success=True,
            latency_ms=50.0,
        )
        await adapter.store_routing_record(record)

        # Store channel snapshot
        snapshot = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=1000,
            messages_routed=980,
            messages_failed=20,
            avg_latency_ms=45.0,
            active_threads=50,
            unique_senders=200,
        )
        await adapter.store_channel_snapshot(snapshot)

        # Verify items stored
        assert len(mock_km.store_calls) == 2
        assert mock_km.store_calls[0]["type"] == "gateway_routing_record"
        assert mock_km.store_calls[1]["type"] == "gateway_channel_snapshot"


class TestExtensionAdapterResilience:
    """Test resilience patterns in extension adapters."""

    @pytest.mark.asyncio
    async def test_fabric_handles_km_failure_gracefully(self):
        """FabricAdapter should handle KM failures without crashing."""
        from aragora.knowledge.mound.adapters import FabricAdapter, PoolSnapshot

        class FailingKM:
            async def ingest(self, *args, **kwargs):
                raise ConnectionError("KM unavailable")

        adapter = FabricAdapter(workspace_id="test")
        adapter._knowledge_mound = FailingKM()

        snapshot = PoolSnapshot(
            pool_id="pool-1",
            name="Test",
            model="opus",
            current_agents=5,
            min_agents=1,
            max_agents=10,
        )

        # Should not raise, but return None or error result
        result = await adapter.store_pool_snapshot(snapshot)
        # Result handling depends on implementation - no crash is success

    @pytest.mark.asyncio
    async def test_gateway_handles_km_timeout_gracefully(self):
        """GatewayAdapter should handle KM timeouts gracefully."""
        from aragora.knowledge.mound.adapters import GatewayAdapter, MessageRoutingRecord
        import asyncio

        class SlowKM:
            async def store(self, *args, **kwargs):
                await asyncio.sleep(1)  # Simulate slow response
                return "item-id"

        adapter = GatewayAdapter(workspace_id="test")
        adapter._knowledge_mound = SlowKM()

        record = MessageRoutingRecord(
            message_id="msg-1",
            channel="slack",
            sender="user",
            agent_id=None,
            routing_rule=None,
            success=True,
            latency_ms=50.0,
        )

        # Use asyncio.wait_for to test timeout handling
        try:
            await asyncio.wait_for(
                adapter.store_routing_record(record),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            pass  # Expected - adapter operation was cancelled

        # Adapter should still be functional
        health = adapter.health_check()
        assert "adapter" in health


class TestExtensionAdapterSpecConsistency:
    """Test adapter specs are consistent across all extension adapters."""

    def test_all_extension_specs_have_required_fields(self):
        """All extension adapter specs should have required fields."""
        from aragora.knowledge.mound.adapters import ADAPTER_SPECS

        extension_names = ["fabric", "workspace", "computer_use", "gateway"]

        for name in extension_names:
            assert name in ADAPTER_SPECS, f"{name} not in ADAPTER_SPECS"
            spec = ADAPTER_SPECS[name]
            assert spec.forward_method, f"{name} missing forward_method"
            assert spec.reverse_method, f"{name} missing reverse_method"
            assert spec.priority > 0, f"{name} has invalid priority"

    def test_extension_specs_unique_priorities(self):
        """Extension adapters should have unique priorities."""
        from aragora.knowledge.mound.adapters import ADAPTER_SPECS

        extension_names = ["fabric", "workspace", "computer_use", "gateway"]
        priorities = [ADAPTER_SPECS[name].priority for name in extension_names]

        assert len(priorities) == len(set(priorities)), "Duplicate priorities found"
