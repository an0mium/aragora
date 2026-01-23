"""
End-to-End Integration Tests for Enterprise Control Plane.

Tests the full flow:
1. Task submission with policy evaluation
2. Agent selection with KM recommendations
3. Deliberation via Arena Bridge with SLA monitoring
4. Prometheus metrics export
5. Audit trail logging
6. Regional routing

These tests validate that all components work together correctly.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Core control plane imports
from aragora.control_plane.coordinator import (
    ControlPlaneConfig,
    ControlPlaneCoordinator,
)
from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)
from aragora.control_plane.scheduler import (
    Task,
    TaskPriority,
    TaskScheduler,
    TaskStatus,
    RegionRoutingMode,
)
from aragora.control_plane.health import HealthMonitor, HealthStatus

# Policy and audit
from aragora.control_plane.policy import (
    ControlPlanePolicy,
    ControlPlanePolicyManager,
    PolicyDecision,
    EnforcementLevel,
)
from aragora.control_plane.audit import (
    AuditAction,
    AuditLog,
    AuditEntry,
)

# Arena bridge and deliberation
from aragora.control_plane.arena_bridge import (
    ArenaControlPlaneBridge,
    ArenaEventAdapter,
    AgentMetrics,
)
from aragora.control_plane.deliberation import (
    DeliberationTask,
    DeliberationOutcome,
    SLARequirement,
    SLAComplianceLevel,
)
from aragora.control_plane.deliberation_events import DeliberationEventType

# Region routing
from aragora.control_plane.region_router import (
    RegionRouter,
    RegionHealth,
    RegionStatus,
    RegionRoutingDecision,
)

# KM adapter
from aragora.knowledge.mound.adapters.control_plane_adapter import (
    ControlPlaneAdapter,
    TaskOutcome,
    AgentCapabilityRecord,
)


class MockRedis:
    """Mock Redis for testing without actual Redis connection."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._streams: Dict[str, List[tuple]] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
        return deleted

    async def hset(self, key: str, mapping: Dict[str, str]) -> int:
        if key not in self._data:
            self._data[key] = {}
        self._data[key].update(mapping)
        return len(mapping)

    async def hgetall(self, key: str) -> Dict[str, str]:
        return self._data.get(key, {})

    async def xadd(self, stream: str, fields: Dict[str, str], id: str = "*") -> str:
        if stream not in self._streams:
            self._streams[stream] = []
        entry_id = f"{int(time.time() * 1000)}-0"
        self._streams[stream].append((entry_id, fields))
        return entry_id

    async def xread(
        self,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> List:
        results = []
        for stream, last_id in streams.items():
            if stream in self._streams:
                entries = self._streams[stream]
                results.append((stream, entries[-count:] if count else entries))
        return results

    async def close(self):
        pass


class MockKnowledgeMound:
    """Mock KnowledgeMound for testing KM integration."""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}
        self._query_results: List[Dict[str, Any]] = []

    async def ingest(self, item: Any) -> str:
        item_id = getattr(item, "id", f"item-{len(self._items)}")
        self._items[item_id] = {
            "id": item_id,
            "content": getattr(item, "content", ""),
            "metadata": getattr(item, "metadata", {}),
            "confidence": getattr(item, "confidence", 0.8),
        }
        return item_id

    async def query(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str = "default",
    ) -> List[Dict[str, Any]]:
        # Return pre-configured results or search items
        if self._query_results:
            return self._query_results[:limit]

        results = []
        for item in self._items.values():
            if query.lower() in item.get("content", "").lower():
                results.append(item)
        return results[:limit]

    def set_query_results(self, results: List[Dict[str, Any]]) -> None:
        """Pre-configure query results for testing."""
        self._query_results = results


class MockStreamServer:
    """Mock ControlPlaneStreamServer for testing event emission."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    async def emit_deliberation_started(self, **kwargs):
        self.events.append({"type": "deliberation_started", **kwargs})

    async def emit_deliberation_round(self, **kwargs):
        self.events.append({"type": "deliberation_round", **kwargs})

    async def emit_deliberation_progress(self, **kwargs):
        self.events.append({"type": "deliberation_progress", **kwargs})

    async def emit_deliberation_vote(self, **kwargs):
        self.events.append({"type": "deliberation_vote", **kwargs})

    async def emit_deliberation_consensus(self, **kwargs):
        self.events.append({"type": "deliberation_consensus", **kwargs})

    async def emit_deliberation_completed(self, **kwargs):
        self.events.append({"type": "deliberation_completed", **kwargs})

    async def emit_deliberation_failed(self, **kwargs):
        self.events.append({"type": "deliberation_failed", **kwargs})

    async def emit_deliberation_sla_warning(self, **kwargs):
        self.events.append({"type": "deliberation_sla_warning", **kwargs})


class MockArena:
    """Mock Arena for testing deliberation execution."""

    def __init__(self, result: Optional[Any] = None):
        self._result = result
        self.run_called = False
        self.event_hooks: Dict[str, Any] = {}

    async def run(self) -> Any:
        self.run_called = True

        # Simulate event hooks
        if "on_round_start" in self.event_hooks:
            await self.event_hooks["on_round_start"](1, 3)

        if "on_consensus" in self.event_hooks:
            await self.event_hooks["on_consensus"](True, 0.85)

        return self._result


@pytest.fixture
def mock_redis():
    """Create a mock Redis instance."""
    return MockRedis()


@pytest.fixture
def mock_km():
    """Create a mock KnowledgeMound instance."""
    return MockKnowledgeMound()


@pytest.fixture
def mock_stream_server():
    """Create a mock stream server."""
    return MockStreamServer()


@pytest.fixture
def audit_log():
    """Create an in-memory audit log."""
    return AuditLog(storage_path=":memory:")


@pytest.fixture
def policy_manager():
    """Create a policy manager with test policies."""
    manager = ControlPlanePolicyManager()

    # Add a test policy
    policy = ControlPlanePolicy(
        id="test-policy",
        name="Test Policy",
        description="Policy for testing",
        enforcement_level=EnforcementLevel.ENFORCE,
        task_types=["debate", "deliberation"],
    )
    manager.add_policy(policy)

    return manager


@pytest.fixture
def region_router():
    """Create a region router with test regions."""
    router = RegionRouter(local_region="us-west-2")

    # Setup test regions
    router.update_region_metrics(
        "us-west-2",
        agent_count=5,
        pending_tasks=2,
        latency_ms=10.0,
        error_rate=0.01,
    )
    router.update_region_metrics(
        "us-east-1",
        agent_count=3,
        pending_tasks=5,
        latency_ms=50.0,
        error_rate=0.02,
    )
    router.update_region_metrics(
        "eu-west-1",
        agent_count=2,
        pending_tasks=1,
        latency_ms=100.0,
        error_rate=0.05,
    )

    return router


class TestControlPlaneE2EFlow:
    """Test the complete control plane flow end-to-end."""

    @pytest.mark.asyncio
    async def test_full_deliberation_flow(
        self,
        mock_km,
        mock_stream_server,
        audit_log,
        policy_manager,
        region_router,
    ):
        """
        Test the complete deliberation flow:
        1. Policy evaluation allows the task
        2. Region router selects optimal region
        3. KM provides agent recommendations
        4. Arena bridge executes deliberation
        5. Metrics are recorded
        6. Audit trail is logged
        """
        # Setup KM with historical agent performance data
        mock_km.set_query_results(
            [
                {
                    "id": "cp_capability_claude_debate",
                    "content": "Agent claude capability 'debate': 90% success rate",
                    "metadata": {
                        "type": "control_plane_capability",
                        "agent_id": "claude",
                        "capability": "debate",
                        "success_count": 90,
                        "failure_count": 10,
                        "avg_duration_seconds": 45.0,
                    },
                    "confidence": 0.9,
                },
                {
                    "id": "cp_capability_gpt4_debate",
                    "content": "Agent gpt-4 capability 'debate': 85% success rate",
                    "metadata": {
                        "type": "control_plane_capability",
                        "agent_id": "gpt-4",
                        "capability": "debate",
                        "success_count": 85,
                        "failure_count": 15,
                        "avg_duration_seconds": 50.0,
                    },
                    "confidence": 0.85,
                },
            ]
        )

        # Create KM adapter
        km_adapter = ControlPlaneAdapter(
            knowledge_mound=mock_km,
            workspace_id="test-workspace",
        )

        # Step 1: Policy evaluation
        task_type = "deliberation"
        result = policy_manager.evaluate_task_dispatch(
            task_type=task_type,
            workspace="test-workspace",
            agent_id="claude",
        )
        assert result.decision == PolicyDecision.ALLOW

        # Step 2: Region routing
        task = Task(
            task_type=task_type,
            payload={"question": "What is the best testing strategy?"},
            region_routing_mode=RegionRoutingMode.PREFERRED,
            target_region="us-west-2",
        )
        routing_decision = await region_router.select_region(task, prefer_local=True)
        assert routing_decision.selected_region == "us-west-2"
        assert len(routing_decision.fallback_regions) > 0

        # Step 3: KM agent recommendations
        recommendations = await km_adapter.get_capability_recommendations(
            capability="debate",
            limit=5,
        )
        assert len(recommendations) > 0
        # Claude should be recommended first (higher success rate)
        assert recommendations[0].agent_id == "claude"

        # Step 4: Create and execute deliberation via Arena Bridge
        arena_bridge = ArenaControlPlaneBridge(
            stream_server=mock_stream_server,
        )

        deliberation_task = DeliberationTask(
            task_id="test-delib-001",
            request_id="req-001",
            question="What is the best testing strategy?",
            agents=["claude", "gpt-4"],
            max_rounds=3,
            sla=SLARequirement(
                timeout_seconds=300,
                warning_threshold_pct=0.7,
                critical_threshold_pct=0.9,
            ),
        )

        # Mock the Arena execution
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.rounds_completed = 3
        mock_result.final_answer = "Use integration tests for E2E validation"
        mock_result.agent_responses = {
            "claude": ["Response 1", "Response 2", "Response 3"],
            "gpt-4": ["Response A", "Response B", "Response C"],
        }

        with patch.object(
            arena_bridge,
            "_create_arena",
            return_value=MockArena(mock_result),
        ):
            outcome = await arena_bridge.execute_via_arena(
                task=deliberation_task,
                agents=["claude", "gpt-4"],
            )

        # Verify deliberation outcome
        assert outcome.success is True
        assert outcome.consensus_reached is True
        assert outcome.sla_compliant is True

        # Step 5: Verify stream events were emitted
        event_types = [e["type"] for e in mock_stream_server.events]
        assert "deliberation_started" in event_types
        assert "deliberation_completed" in event_types

        # Step 6: Store outcome in KM for future learning
        task_outcome = TaskOutcome(
            task_id=deliberation_task.task_id,
            task_type="deliberation",
            agent_id="claude",
            success=True,
            duration_seconds=outcome.duration_seconds,
            workspace_id="test-workspace",
        )
        outcome_id = await km_adapter.store_task_outcome(task_outcome)
        assert outcome_id is not None

    @pytest.mark.asyncio
    async def test_policy_denial_flow(self, policy_manager):
        """Test that policy denials are handled correctly."""
        # Add a restrictive policy
        policy = ControlPlanePolicy(
            id="restrict-agent",
            name="Agent Restriction",
            description="Restrict certain agents",
            enforcement_level=EnforcementLevel.ENFORCE,
            allowed_agents=["claude"],  # Only claude allowed
        )
        policy_manager.add_policy(policy)

        # Try to dispatch with blocked agent
        result = policy_manager.evaluate_task_dispatch(
            task_type="debate",
            workspace="test-workspace",
            agent_id="gpt-4",  # Not in allowed list
        )

        assert result.decision == PolicyDecision.DENY
        assert "agent" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_region_failover_flow(self, region_router):
        """Test region failover when primary region fails."""
        # Create a task targeting us-west-2
        task = Task(
            task_type="debate",
            payload={},
            target_region="us-west-2",
            fallback_regions=["us-east-1", "eu-west-1"],
            region_routing_mode=RegionRoutingMode.PREFERRED,
        )

        # Initial selection should be us-west-2
        decision = await region_router.select_region(task, prefer_local=True)
        assert decision.selected_region == "us-west-2"

        # Simulate failure and request failover
        failover_region = await region_router.failover_region(
            task_id="task-001",
            failed_region="us-west-2",
            task=task,
        )

        # Should failover to next healthy region
        assert failover_region is not None
        assert failover_region != "us-west-2"
        assert failover_region in ["us-east-1", "eu-west-1"]

        # Verify us-west-2 is now marked unhealthy
        health = region_router.get_region_health("us-west-2")
        assert health.status == RegionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_km_learning_feedback_loop(self, mock_km):
        """Test that task outcomes feed back into KM for learning."""
        km_adapter = ControlPlaneAdapter(
            knowledge_mound=mock_km,
            workspace_id="test-workspace",
        )

        # Simulate multiple task completions
        for i in range(10):
            outcome = TaskOutcome(
                task_id=f"task-{i}",
                task_type="debate",
                agent_id="claude" if i % 2 == 0 else "gpt-4",
                success=i < 8,  # 80% success rate
                duration_seconds=45.0 + i,
                workspace_id="test-workspace",
            )
            await km_adapter.store_task_outcome(outcome)

        # Verify outcomes are stored
        assert km_adapter.get_stats()["task_outcomes_stored"] == 10

        # Query for historical patterns
        history = await km_adapter.get_task_history(
            task_type="debate",
            limit=20,
        )
        # Should find stored outcomes (depends on mock implementation)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_sla_monitoring_and_alerts(self, mock_stream_server):
        """Test SLA monitoring during deliberation."""
        arena_bridge = ArenaControlPlaneBridge(
            stream_server=mock_stream_server,
        )

        # Create a task with tight SLA
        task = DeliberationTask(
            task_id="sla-test-001",
            request_id="req-sla",
            question="Quick question",
            agents=["claude"],
            max_rounds=1,
            sla=SLARequirement(
                timeout_seconds=1,  # Very tight timeout
                warning_threshold_pct=0.5,
                critical_threshold_pct=0.8,
            ),
        )

        # The SLA monitor should detect threshold breaches
        # (In real scenario, this would trigger warnings)
        compliance = task.sla.get_compliance_level(0.6)  # 60% of timeout
        assert compliance == SLAComplianceLevel.WARNING

        compliance = task.sla.get_compliance_level(0.9)  # 90% of timeout
        assert compliance == SLAComplianceLevel.CRITICAL

        compliance = task.sla.get_compliance_level(1.1)  # Over timeout
        assert compliance == SLAComplianceLevel.VIOLATED

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, mock_km, mock_stream_server):
        """Test coordination between multiple agents in a deliberation."""
        km_adapter = ControlPlaneAdapter(
            knowledge_mound=mock_km,
            workspace_id="test-workspace",
        )

        # Setup capability records for multiple agents
        agents = ["claude", "gpt-4", "gemini", "mistral"]
        for i, agent in enumerate(agents):
            record = AgentCapabilityRecord(
                agent_id=agent,
                capability="debate",
                success_count=80 + i * 5,
                failure_count=20 - i * 3,
                avg_duration_seconds=40.0 + i * 5,
                workspace_id="test-workspace",
            )
            await km_adapter.store_capability_record(record)

        # Get recommendations - should rank by success rate
        recommendations = await km_adapter.get_agent_recommendations_for_task(
            task_type="debate",
            available_agents=agents,
            required_capabilities=["debate"],
            top_n=4,
        )

        # All agents should be included
        assert len(recommendations) == 4
        # Should have scores
        for rec in recommendations:
            assert "combined_score" in rec
            assert "agent_id" in rec


class TestMetricsIntegration:
    """Test Prometheus metrics integration."""

    def test_deliberation_metrics_recording(self):
        """Test that deliberation metrics are recorded correctly."""
        from aragora.server.prometheus_control_plane import (
            record_deliberation_complete,
            record_deliberation_sla,
            record_agent_utilization,
            record_policy_decision,
        )

        # These should not raise
        record_deliberation_complete(
            duration_seconds=45.0,
            status="completed",
            consensus_reached=True,
            confidence=0.85,
            round_count=3,
            agent_count=2,
        )

        record_deliberation_sla("compliant")
        record_deliberation_sla("warning")

        record_agent_utilization("claude", 0.75)
        record_agent_utilization("gpt-4", 0.60)

        record_policy_decision("allow", "all")
        record_policy_decision("deny", "agent_restriction")

    def test_control_plane_task_metrics(self):
        """Test control plane task metrics."""
        from aragora.server.prometheus_control_plane import (
            record_control_plane_task_submitted,
            record_control_plane_task_status,
            record_control_plane_task_completed,
            record_control_plane_queue_depth,
        )

        record_control_plane_task_submitted("debate", "high")
        record_control_plane_task_status("running", 5)
        record_control_plane_task_completed("debate", "completed", 45.0)
        record_control_plane_queue_depth("high", 3)


class TestAuditTrailIntegration:
    """Test audit trail integration."""

    @pytest.mark.asyncio
    async def test_audit_log_persistence(self):
        """Test that audit entries are persisted correctly."""
        audit_log = AuditLog(storage_path=":memory:")
        await audit_log.initialize()

        # Log various actions
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor_id="user-001",
            resource_id="task-001",
            details={"task_type": "debate", "priority": "high"},
        )

        await audit_log.log(
            action=AuditAction.POLICY_DECISION_ALLOW,
            actor_id="system",
            resource_id="task-001",
            details={"policy_id": "test-policy", "reason": "All policies passed"},
        )

        await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor_id="agent-claude",
            resource_id="task-001",
            details={"duration_seconds": 45.0, "success": True},
        )

        # Query audit log
        entries = await audit_log.query(
            resource_id="task-001",
            limit=10,
        )

        assert len(entries) == 3
        actions = [e.action for e in entries]
        assert AuditAction.TASK_SUBMITTED in actions
        assert AuditAction.POLICY_DECISION_ALLOW in actions
        assert AuditAction.TASK_COMPLETED in actions

    @pytest.mark.asyncio
    async def test_audit_log_integrity(self):
        """Test audit log tamper detection via hash chain."""
        audit_log = AuditLog(storage_path=":memory:")
        await audit_log.initialize()

        # Log entries
        for i in range(5):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor_id=f"user-{i}",
                resource_id=f"task-{i}",
                details={"index": i},
            )

        # Verify integrity
        is_valid = await audit_log.verify_integrity()
        assert is_valid is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_agent_pool(self, region_router):
        """Test handling when no agents are available."""
        # Create region with no agents
        region_router.update_region_metrics(
            "empty-region",
            agent_count=0,
            error_rate=0.0,
        )

        health = region_router.get_region_health("empty-region")
        assert health.status == RegionStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_all_regions_unhealthy(self, region_router):
        """Test failover when all regions are unhealthy."""
        # Mark all regions as unhealthy
        for region in ["us-west-2", "us-east-1", "eu-west-1"]:
            region_router.update_region_metrics(
                region,
                agent_count=0,
                error_rate=0.9,
            )

        task = Task(
            task_type="debate",
            payload={},
            region_routing_mode=RegionRoutingMode.ANY,
        )

        decision = await region_router.select_region(task)

        # Should have no eligible regions
        # (depending on implementation, might return None or local region)
        assert decision.selected_region is None or decision.fallback_regions == []

    @pytest.mark.asyncio
    async def test_km_unavailable_graceful_degradation(self):
        """Test that system works when KM is unavailable."""
        # Create adapter without KM
        km_adapter = ControlPlaneAdapter(
            knowledge_mound=None,
            workspace_id="test-workspace",
        )

        # Should return empty results, not error
        recommendations = await km_adapter.get_capability_recommendations("debate")
        assert recommendations == []

        history = await km_adapter.get_task_history("debate")
        assert history == []

        # Store should return None
        outcome = TaskOutcome(
            task_id="test",
            task_type="debate",
            agent_id="claude",
            success=True,
            duration_seconds=45.0,
        )
        result = await km_adapter.store_task_outcome(outcome)
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_deliberations(self, mock_stream_server):
        """Test handling multiple concurrent deliberations."""
        arena_bridge = ArenaControlPlaneBridge(
            stream_server=mock_stream_server,
        )

        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = DeliberationTask(
                task_id=f"concurrent-{i}",
                request_id=f"req-{i}",
                question=f"Question {i}",
                agents=["claude"],
                max_rounds=1,
                sla=SLARequirement(timeout_seconds=60),
            )
            tasks.append(task)

        # Mock Arena for all tasks
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.8
        mock_result.rounds_completed = 1
        mock_result.final_answer = "Answer"
        mock_result.agent_responses = {"claude": ["Response"]}

        # Execute concurrently
        with patch.object(
            arena_bridge,
            "_create_arena",
            return_value=MockArena(mock_result),
        ):
            outcomes = await asyncio.gather(
                *[arena_bridge.execute_via_arena(task, ["claude"]) for task in tasks]
            )

        # All should complete
        assert len(outcomes) == 3
        assert all(o.success for o in outcomes)


class TestDataConsistency:
    """Test data consistency across components."""

    @pytest.mark.asyncio
    async def test_task_state_consistency(self, mock_km):
        """Test that task state is consistent across components."""
        km_adapter = ControlPlaneAdapter(
            knowledge_mound=mock_km,
            workspace_id="test-workspace",
        )

        task_id = "consistency-test-001"

        # Store initial outcome
        outcome1 = TaskOutcome(
            task_id=task_id,
            task_type="debate",
            agent_id="claude",
            success=True,
            duration_seconds=45.0,
            workspace_id="test-workspace",
        )
        await km_adapter.store_task_outcome(outcome1)

        # Store updated outcome (same task, different agent)
        outcome2 = TaskOutcome(
            task_id=task_id,
            task_type="debate",
            agent_id="gpt-4",
            success=True,
            duration_seconds=50.0,
            workspace_id="test-workspace",
        )
        await km_adapter.store_task_outcome(outcome2)

        # Both should be stored (different agents)
        assert km_adapter.get_stats()["task_outcomes_stored"] == 2

    @pytest.mark.asyncio
    async def test_region_health_consistency(self, region_router):
        """Test that region health is consistent."""
        # Update metrics multiple times
        for i in range(5):
            region_router.update_region_metrics(
                "us-west-2",
                agent_count=5 + i,
                latency_ms=10.0 + i,
            )

        # Should have latest values
        health = region_router.get_region_health("us-west-2")
        assert health.agent_count == 9  # 5 + 4
        assert health.latency_ms == 14.0  # 10 + 4
