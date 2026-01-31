"""
Tests for FabricAdapter - Agent Fabric to Knowledge Mound integration.

Tests cover:
1. Adapter initialization and configuration
2. Knowledge item CRUD operations (Pool, Task, Budget, Policy)
3. Sync operations with external fabric systems
4. Query and search functionality (recommendations)
5. Batch processing (cache operations)
6. Error handling and recovery
7. Metrics and health reporting
"""

import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.fabric_adapter import (
    FabricAdapter,
    PoolSnapshot,
    TaskSchedulingOutcome,
    BudgetUsageSnapshot,
    PolicyDecisionRecord,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.ingest = AsyncMock(return_value="km_test_id")
    mound.query = AsyncMock(return_value=[])
    return mound


@pytest.fixture
def mock_fabric():
    """Create a mock AgentFabric."""
    fabric = MagicMock()

    # Mock pool
    pool = MagicMock()
    pool.id = "pool_123"
    pool.name = "debate_pool"
    pool.model = "claude-3"
    pool.current_agents = ["agent1", "agent2"]
    pool.min_agents = 1
    pool.max_agents = 5

    fabric.list_pools = AsyncMock(return_value=[pool])

    # Mock scheduler
    scheduler = MagicMock()
    scheduler.get_stats = AsyncMock(return_value={"tasks_pending": 3, "tasks_completed": 25})
    fabric.scheduler = scheduler

    # Mock budget
    budget = MagicMock()
    budget.get_stats = AsyncMock(return_value={"tracked_entities": ["agent_1", "agent_2"]})
    fabric.budget = budget

    # Mock usage report
    usage_report = MagicMock()
    usage_report.total_tokens = 50000
    usage_report.tokens_limit = 100000
    usage_report.total_cost_usd = 2.50
    usage_report.cost_limit_usd = 10.00
    usage_report.period_start = datetime.now()
    usage_report.period_end = datetime.now()
    usage_report.alerts_count = 1

    fabric.get_usage_report = AsyncMock(return_value=usage_report)

    return fabric


@pytest.fixture
def adapter(mock_knowledge_mound):
    """Create a FabricAdapter with mock KM."""
    return FabricAdapter(
        fabric=None,
        knowledge_mound=mock_knowledge_mound,
        workspace_id="test_workspace",
    )


@pytest.fixture
def adapter_with_fabric(mock_knowledge_mound, mock_fabric):
    """Create a FabricAdapter with mock KM and fabric."""
    return FabricAdapter(
        fabric=mock_fabric,
        knowledge_mound=mock_knowledge_mound,
        workspace_id="test_workspace",
    )


@pytest.fixture
def pool_snapshot():
    """Create a sample PoolSnapshot."""
    return PoolSnapshot(
        pool_id="pool_123",
        name="debate_pool",
        model="claude-3",
        current_agents=3,
        min_agents=1,
        max_agents=5,
        tasks_pending=2,
        tasks_completed=15,
        avg_task_duration_seconds=12.5,
        workspace_id="test_workspace",
        metadata={"region": "us-west"},
    )


@pytest.fixture
def task_outcome():
    """Create a sample TaskSchedulingOutcome."""
    return TaskSchedulingOutcome(
        task_id="task_456",
        task_type="debate",
        agent_id="claude-3-agent",
        pool_id="pool_123",
        priority=1,
        scheduled_at=time.time() - 100,
        completed_at=time.time(),
        success=True,
        duration_seconds=15.5,
        error_message=None,
        workspace_id="test_workspace",
    )


@pytest.fixture
def budget_snapshot():
    """Create a sample BudgetUsageSnapshot."""
    return BudgetUsageSnapshot(
        entity_id="agent_123",
        entity_type="agent",
        tokens_used=50000,
        tokens_limit=100000,
        cost_used_usd=2.50,
        cost_limit_usd=10.00,
        period_start=time.time() - 86400,
        period_end=time.time(),
        alerts_triggered=1,
        workspace_id="test_workspace",
    )


@pytest.fixture
def policy_decision():
    """Create a sample PolicyDecisionRecord."""
    return PolicyDecisionRecord(
        decision_id="decision_789",
        agent_id="claude-3-agent",
        action="execute_tool",
        allowed=True,
        policy_id="policy_001",
        reason="Agent has required permissions",
        timestamp=time.time(),
        context={"tool": "file_write", "path": "/tmp/test.txt"},
        workspace_id="test_workspace",
    )


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestAdapterInitialization:
    """Tests for adapter initialization and configuration."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        adapter = FabricAdapter()

        assert adapter._fabric is None
        assert adapter._knowledge_mound is None
        assert adapter._workspace_id == "default"
        assert adapter._min_confidence_threshold == 0.6
        assert adapter._enable_dual_write is False

    def test_init_with_knowledge_mound(self, mock_knowledge_mound):
        """Should accept knowledge mound."""
        adapter = FabricAdapter(knowledge_mound=mock_knowledge_mound)

        assert adapter._knowledge_mound is mock_knowledge_mound

    def test_init_with_fabric(self, mock_fabric):
        """Should accept fabric."""
        adapter = FabricAdapter(fabric=mock_fabric)

        assert adapter._fabric is mock_fabric

    def test_init_with_workspace_id(self):
        """Should accept workspace_id."""
        adapter = FabricAdapter(workspace_id="custom_workspace")

        assert adapter._workspace_id == "custom_workspace"

    def test_init_with_event_callback(self):
        """Should accept event_callback."""
        callback = MagicMock()
        adapter = FabricAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_init_with_custom_confidence_threshold(self):
        """Should accept min_confidence_threshold."""
        adapter = FabricAdapter(min_confidence_threshold=0.8)

        assert adapter._min_confidence_threshold == 0.8

    def test_init_with_dual_write_enabled(self):
        """Should accept enable_dual_write."""
        adapter = FabricAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True

    def test_init_caches_empty(self):
        """Should initialize empty caches."""
        adapter = FabricAdapter()

        assert len(adapter._pool_performance_cache) == 0
        assert len(adapter._task_patterns_cache) == 0
        assert len(adapter._cache_times) == 0

    def test_init_stats_empty(self):
        """Should initialize stats to zero."""
        adapter = FabricAdapter()

        assert adapter._stats["pool_snapshots_stored"] == 0
        assert adapter._stats["task_outcomes_stored"] == 0
        assert adapter._stats["budget_snapshots_stored"] == 0
        assert adapter._stats["policy_decisions_stored"] == 0
        assert adapter._stats["pool_queries"] == 0
        assert adapter._stats["task_pattern_queries"] == 0

    def test_adapter_name(self):
        """Should have correct adapter_name."""
        adapter = FabricAdapter()

        assert adapter.adapter_name == "fabric"


# =============================================================================
# Pool Snapshot Storage Tests
# =============================================================================


class TestPoolSnapshotStorage:
    """Tests for pool snapshot storage."""

    @pytest.mark.asyncio
    async def test_stores_pool_snapshot(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should store pool snapshot in KM."""
        result = await adapter.store_pool_snapshot(pool_snapshot)

        assert result == "km_test_id"
        mock_knowledge_mound.ingest.assert_called_once()

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "fabric_pool_" in call_args.id
        assert pool_snapshot.name in call_args.content
        assert pool_snapshot.model in call_args.content

    @pytest.mark.asyncio
    async def test_pool_snapshot_metadata(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should include correct metadata."""
        await adapter.store_pool_snapshot(pool_snapshot)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        metadata = call_args.metadata

        assert metadata["type"] == "fabric_pool_snapshot"
        assert metadata["pool_id"] == pool_snapshot.pool_id
        assert metadata["pool_name"] == pool_snapshot.name
        assert metadata["model"] == pool_snapshot.model
        assert metadata["current_agents"] == pool_snapshot.current_agents
        assert metadata["tasks_completed"] == pool_snapshot.tasks_completed
        assert metadata["workspace_id"] == pool_snapshot.workspace_id
        assert metadata["region"] == "us-west"

    @pytest.mark.asyncio
    async def test_pool_snapshot_utilization(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should calculate utilization correctly."""
        await adapter.store_pool_snapshot(pool_snapshot)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        utilization = call_args.metadata["utilization"]

        expected = pool_snapshot.current_agents / pool_snapshot.max_agents
        assert utilization == expected

    @pytest.mark.asyncio
    async def test_no_km_returns_none(self, pool_snapshot):
        """Should return None when no KM configured."""
        adapter = FabricAdapter(knowledge_mound=None)

        result = await adapter.store_pool_snapshot(pool_snapshot)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, pool_snapshot):
        """Should update stats on storage."""
        assert adapter._stats["pool_snapshots_stored"] == 0

        await adapter.store_pool_snapshot(pool_snapshot)

        assert adapter._stats["pool_snapshots_stored"] == 1

    @pytest.mark.asyncio
    async def test_invalidates_cache(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should invalidate pool cache on storage."""
        # Pre-populate cache
        adapter._pool_performance_cache[pool_snapshot.pool_id] = [pool_snapshot]

        await adapter.store_pool_snapshot(pool_snapshot)

        assert pool_snapshot.pool_id not in adapter._pool_performance_cache

    @pytest.mark.asyncio
    async def test_emits_event(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should emit event on storage."""
        events = []

        def callback(event_type, data):
            events.append((event_type, data))

        adapter.set_event_callback(callback)

        await adapter.store_pool_snapshot(pool_snapshot)

        assert len(events) == 1
        assert events[0][0] == "fabric_pool_snapshot_stored"
        assert events[0][1]["pool_id"] == pool_snapshot.pool_id

    @pytest.mark.asyncio
    async def test_handles_ingest_error(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should handle ingest errors gracefully."""
        mock_knowledge_mound.ingest.side_effect = Exception("Ingest failed")

        result = await adapter.store_pool_snapshot(pool_snapshot)

        assert result is None


# =============================================================================
# Task Outcome Storage Tests
# =============================================================================


class TestTaskOutcomeStorage:
    """Tests for task outcome storage."""

    @pytest.mark.asyncio
    async def test_stores_successful_task(self, adapter, task_outcome, mock_knowledge_mound):
        """Should store successful task outcomes."""
        result = await adapter.store_task_outcome(task_outcome)

        assert result == "km_test_id"
        mock_knowledge_mound.ingest.assert_called_once()

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "fabric_task_" in call_args.id
        assert task_outcome.agent_id in call_args.content
        assert "success" in call_args.content

    @pytest.mark.asyncio
    async def test_stores_failed_task(self, adapter, mock_knowledge_mound):
        """Should store failed task outcomes."""
        outcome = TaskSchedulingOutcome(
            task_id="task_failed",
            task_type="code_review",
            agent_id="gpt-4",
            pool_id="pool_1",
            priority=2,
            scheduled_at=time.time() - 60,
            completed_at=time.time(),
            success=False,
            duration_seconds=30.0,
            error_message="Timeout exceeded",
        )

        result = await adapter.store_task_outcome(outcome)

        assert result == "km_test_id"
        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "failed" in call_args.content

    @pytest.mark.asyncio
    async def test_task_outcome_metadata(self, adapter, task_outcome, mock_knowledge_mound):
        """Should include correct metadata."""
        await adapter.store_task_outcome(task_outcome)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        metadata = call_args.metadata

        assert metadata["type"] == "fabric_task_outcome"
        assert metadata["task_id"] == task_outcome.task_id
        assert metadata["task_type"] == task_outcome.task_type
        assert metadata["agent_id"] == task_outcome.agent_id
        assert metadata["pool_id"] == task_outcome.pool_id
        assert metadata["success"] is True
        assert metadata["duration_seconds"] == task_outcome.duration_seconds

    @pytest.mark.asyncio
    async def test_no_km_returns_none(self, task_outcome):
        """Should return None when no KM configured."""
        adapter = FabricAdapter(knowledge_mound=None)

        result = await adapter.store_task_outcome(task_outcome)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, task_outcome):
        """Should update stats on storage."""
        assert adapter._stats["task_outcomes_stored"] == 0

        await adapter.store_task_outcome(task_outcome)

        assert adapter._stats["task_outcomes_stored"] == 1

    @pytest.mark.asyncio
    async def test_invalidates_cache(self, adapter, task_outcome, mock_knowledge_mound):
        """Should invalidate task cache on storage."""
        cache_key = task_outcome.task_type.lower()
        adapter._task_patterns_cache[cache_key] = [task_outcome]

        await adapter.store_task_outcome(task_outcome)

        assert cache_key not in adapter._task_patterns_cache


# =============================================================================
# Budget Snapshot Storage Tests
# =============================================================================


class TestBudgetSnapshotStorage:
    """Tests for budget snapshot storage."""

    @pytest.mark.asyncio
    async def test_stores_budget_snapshot(self, adapter, budget_snapshot, mock_knowledge_mound):
        """Should store budget snapshot in KM."""
        result = await adapter.store_budget_snapshot(budget_snapshot)

        assert result == "km_test_id"
        mock_knowledge_mound.ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_budget_snapshot_content(self, adapter, budget_snapshot, mock_knowledge_mound):
        """Should format budget content correctly."""
        await adapter.store_budget_snapshot(budget_snapshot)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        content = call_args.content

        assert budget_snapshot.entity_id in content
        assert budget_snapshot.entity_type in content
        assert str(budget_snapshot.tokens_used) in content.replace(",", "")

    @pytest.mark.asyncio
    async def test_budget_snapshot_metadata(self, adapter, budget_snapshot, mock_knowledge_mound):
        """Should include correct metadata."""
        await adapter.store_budget_snapshot(budget_snapshot)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        metadata = call_args.metadata

        assert metadata["type"] == "fabric_budget_snapshot"
        assert metadata["entity_id"] == budget_snapshot.entity_id
        assert metadata["entity_type"] == budget_snapshot.entity_type
        assert metadata["tokens_used"] == budget_snapshot.tokens_used
        assert metadata["tokens_limit"] == budget_snapshot.tokens_limit
        assert metadata["cost_used_usd"] == budget_snapshot.cost_used_usd
        assert metadata["cost_limit_usd"] == budget_snapshot.cost_limit_usd

    @pytest.mark.asyncio
    async def test_utilization_percentages(self, adapter, budget_snapshot, mock_knowledge_mound):
        """Should calculate utilization percentages."""
        await adapter.store_budget_snapshot(budget_snapshot)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        metadata = call_args.metadata

        expected_token_pct = (50000 / 100000) * 100
        expected_cost_pct = (2.50 / 10.00) * 100

        assert metadata["token_utilization_pct"] == expected_token_pct
        assert metadata["cost_utilization_pct"] == expected_cost_pct

    @pytest.mark.asyncio
    async def test_no_km_returns_none(self, budget_snapshot):
        """Should return None when no KM configured."""
        adapter = FabricAdapter(knowledge_mound=None)

        result = await adapter.store_budget_snapshot(budget_snapshot)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, budget_snapshot):
        """Should update stats on storage."""
        assert adapter._stats["budget_snapshots_stored"] == 0

        await adapter.store_budget_snapshot(budget_snapshot)

        assert adapter._stats["budget_snapshots_stored"] == 1


# =============================================================================
# Policy Decision Storage Tests
# =============================================================================


class TestPolicyDecisionStorage:
    """Tests for policy decision storage."""

    @pytest.mark.asyncio
    async def test_stores_allowed_decision(self, adapter, policy_decision, mock_knowledge_mound):
        """Should store allowed policy decision."""
        result = await adapter.store_policy_decision(policy_decision)

        assert result == "km_test_id"
        mock_knowledge_mound.ingest.assert_called_once()

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "allowed" in call_args.content

    @pytest.mark.asyncio
    async def test_stores_denied_decision(self, adapter, mock_knowledge_mound):
        """Should store denied policy decision."""
        decision = PolicyDecisionRecord(
            decision_id="decision_denied",
            agent_id="rogue-agent",
            action="delete_file",
            allowed=False,
            policy_id="policy_secure",
            reason="Insufficient permissions",
            timestamp=time.time(),
            context={"path": "/critical/file.txt"},
        )

        await adapter.store_policy_decision(decision)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "denied" in call_args.content

    @pytest.mark.asyncio
    async def test_policy_decision_metadata(self, adapter, policy_decision, mock_knowledge_mound):
        """Should include correct metadata."""
        await adapter.store_policy_decision(policy_decision)

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        metadata = call_args.metadata

        assert metadata["type"] == "fabric_policy_decision"
        assert metadata["decision_id"] == policy_decision.decision_id
        assert metadata["agent_id"] == policy_decision.agent_id
        assert metadata["action"] == policy_decision.action
        assert metadata["allowed"] is True
        assert metadata["policy_id"] == policy_decision.policy_id
        assert metadata["reason"] == policy_decision.reason

    @pytest.mark.asyncio
    async def test_no_km_returns_none(self, policy_decision):
        """Should return None when no KM configured."""
        adapter = FabricAdapter(knowledge_mound=None)

        result = await adapter.store_policy_decision(policy_decision)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, policy_decision):
        """Should update stats on storage."""
        assert adapter._stats["policy_decisions_stored"] == 0

        await adapter.store_policy_decision(policy_decision)

        assert adapter._stats["policy_decisions_stored"] == 1


# =============================================================================
# Pool Performance History Tests
# =============================================================================


class TestPoolPerformanceHistory:
    """Tests for pool performance history retrieval."""

    @pytest.mark.asyncio
    async def test_queries_km(self, adapter, mock_knowledge_mound):
        """Should query KM for pool history."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_pool_performance_history("pool_123", limit=10)

        mock_knowledge_mound.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_snapshots(self, adapter, mock_knowledge_mound):
        """Should return parsed PoolSnapshots."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Pool test",
                "metadata": {
                    "type": "fabric_pool_snapshot",
                    "pool_id": "pool_123",
                    "pool_name": "test_pool",
                    "model": "claude-3",
                    "current_agents": 3,
                    "min_agents": 1,
                    "max_agents": 5,
                    "tasks_pending": 2,
                    "tasks_completed": 15,
                    "avg_task_duration_seconds": 12.5,
                },
            }
        ]

        results = await adapter.get_pool_performance_history("pool_123")

        assert len(results) == 1
        assert results[0].pool_id == "pool_123"
        assert results[0].name == "test_pool"
        assert results[0].current_agents == 3

    @pytest.mark.asyncio
    async def test_filters_by_pool_id(self, adapter, mock_knowledge_mound):
        """Should filter results by pool_id."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Pool 1",
                "metadata": {
                    "type": "fabric_pool_snapshot",
                    "pool_id": "pool_123",
                    "pool_name": "pool1",
                    "model": "claude",
                },
            },
            {
                "content": "Pool 2",
                "metadata": {
                    "type": "fabric_pool_snapshot",
                    "pool_id": "pool_other",
                    "pool_name": "pool2",
                    "model": "gpt",
                },
            },
        ]

        results = await adapter.get_pool_performance_history("pool_123")

        assert len(results) == 1
        assert results[0].pool_id == "pool_123"

    @pytest.mark.asyncio
    async def test_uses_cache(self, adapter, mock_knowledge_mound):
        """Should use cache on subsequent calls."""
        mock_knowledge_mound.query.return_value = []

        # First call
        await adapter.get_pool_performance_history("pool_123")

        # Second call should use cache
        await adapter.get_pool_performance_history("pool_123", use_cache=True)

        assert mock_knowledge_mound.query.call_count == 1

    @pytest.mark.asyncio
    async def test_bypasses_cache(self, adapter, mock_knowledge_mound):
        """Should bypass cache when use_cache=False."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_pool_performance_history("pool_123")
        await adapter.get_pool_performance_history("pool_123", use_cache=False)

        assert mock_knowledge_mound.query.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_cache_ttl(self, adapter, mock_knowledge_mound):
        """Should refresh cache after TTL expires."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_pool_performance_history("pool_123")

        # Expire cache
        adapter._cache_times["pool_123"] = time.time() - adapter._cache_ttl - 1

        await adapter.get_pool_performance_history("pool_123", use_cache=True)

        assert mock_knowledge_mound.query.call_count == 2

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, mock_knowledge_mound):
        """Should update query stats."""
        mock_knowledge_mound.query.return_value = []

        assert adapter._stats["pool_queries"] == 0

        await adapter.get_pool_performance_history("pool_123")

        assert adapter._stats["pool_queries"] == 1

    @pytest.mark.asyncio
    async def test_no_km_returns_empty(self):
        """Should return empty list without KM."""
        adapter = FabricAdapter(knowledge_mound=None)

        results = await adapter.get_pool_performance_history("pool_123")

        assert results == []


# =============================================================================
# Task Pattern Tests
# =============================================================================


class TestTaskPatterns:
    """Tests for task pattern retrieval."""

    @pytest.mark.asyncio
    async def test_queries_km(self, adapter, mock_knowledge_mound):
        """Should query KM for task patterns."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_task_patterns("debate", limit=50)

        mock_knowledge_mound.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_outcomes(self, adapter, mock_knowledge_mound):
        """Should return parsed TaskSchedulingOutcomes."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Task outcome",
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "task_1",
                    "task_type": "debate",
                    "agent_id": "claude-3",
                    "pool_id": "pool_1",
                    "priority": 1,
                    "scheduled_at": time.time() - 100,
                    "completed_at": time.time(),
                    "success": True,
                    "duration_seconds": 15.5,
                },
            }
        ]

        results = await adapter.get_task_patterns("debate")

        assert len(results) == 1
        assert results[0].task_id == "task_1"
        assert results[0].task_type == "debate"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_filters_by_task_type(self, adapter, mock_knowledge_mound):
        """Should filter results by task_type."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Task 1",
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "a1",
                    "scheduled_at": 100,
                },
            },
            {
                "content": "Task 2",
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t2",
                    "task_type": "code_review",
                    "agent_id": "a2",
                    "scheduled_at": 200,
                },
            },
        ]

        results = await adapter.get_task_patterns("debate")

        assert len(results) == 1
        assert results[0].task_type == "debate"

    @pytest.mark.asyncio
    async def test_uses_cache(self, adapter, mock_knowledge_mound):
        """Should use cache on subsequent calls."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_task_patterns("debate")
        await adapter.get_task_patterns("debate", use_cache=True)

        assert mock_knowledge_mound.query.call_count == 1

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, mock_knowledge_mound):
        """Should update query stats."""
        mock_knowledge_mound.query.return_value = []

        assert adapter._stats["task_pattern_queries"] == 0

        await adapter.get_task_patterns("debate")

        assert adapter._stats["task_pattern_queries"] == 1


# =============================================================================
# Pool Recommendations Tests
# =============================================================================


class TestPoolRecommendations:
    """Tests for pool recommendations."""

    @pytest.mark.asyncio
    async def test_returns_recommendations(self, adapter, mock_knowledge_mound):
        """Should return pool recommendations."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "a1",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": 100,
                },
            },
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t2",
                    "task_type": "debate",
                    "agent_id": "a2",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 15.0,
                    "scheduled_at": 200,
                },
            },
        ]

        results = await adapter.get_pool_recommendations("debate")

        assert len(results) == 1
        assert results[0]["pool_id"] == "pool_1"
        assert results[0]["success_rate"] == 1.0
        assert "combined_score" in results[0]
        assert "sample_size" in results[0]

    @pytest.mark.asyncio
    async def test_filters_by_available_pools(self, adapter, mock_knowledge_mound):
        """Should filter by available pools."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "a1",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": 100,
                },
            },
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t2",
                    "task_type": "debate",
                    "agent_id": "a2",
                    "pool_id": "pool_2",
                    "success": True,
                    "duration_seconds": 15.0,
                    "scheduled_at": 200,
                },
            },
        ]

        results = await adapter.get_pool_recommendations("debate", available_pools=["pool_1"])

        assert len(results) == 1
        assert results[0]["pool_id"] == "pool_1"

    @pytest.mark.asyncio
    async def test_respects_top_n(self, adapter, mock_knowledge_mound):
        """Should return at most top_n results."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": f"t{i}",
                    "task_type": "debate",
                    "agent_id": "a1",
                    "pool_id": f"pool_{i}",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": i * 100,
                },
            }
            for i in range(10)
        ]

        results = await adapter.get_pool_recommendations("debate", top_n=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_calculates_combined_score(self, adapter, mock_knowledge_mound):
        """Should calculate combined score correctly."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "a1",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 0.0,  # Fastest possible
                    "scheduled_at": 100,
                },
            },
        ]

        results = await adapter.get_pool_recommendations("debate")

        # With 100% success and 0 duration: 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert results[0]["combined_score"] == 1.0


# =============================================================================
# Agent Recommendations Tests
# =============================================================================


class TestAgentRecommendations:
    """Tests for agent recommendations within a pool."""

    @pytest.mark.asyncio
    async def test_returns_recommendations(self, adapter, mock_knowledge_mound):
        """Should return agent recommendations."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "agent_1",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": 100,
                },
            },
        ]

        results = await adapter.get_agent_recommendations_for_pool("pool_1", "debate")

        assert len(results) == 1
        assert results[0]["agent_id"] == "agent_1"
        assert results[0]["pool_id"] == "pool_1"
        assert results[0]["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_filters_by_pool(self, adapter, mock_knowledge_mound):
        """Should filter by pool_id."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t1",
                    "task_type": "debate",
                    "agent_id": "agent_1",
                    "pool_id": "pool_1",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": 100,
                },
            },
            {
                "metadata": {
                    "type": "fabric_task_outcome",
                    "task_id": "t2",
                    "task_type": "debate",
                    "agent_id": "agent_2",
                    "pool_id": "pool_other",
                    "success": True,
                    "duration_seconds": 10.0,
                    "scheduled_at": 200,
                },
            },
        ]

        results = await adapter.get_agent_recommendations_for_pool("pool_1", "debate")

        assert len(results) == 1
        assert results[0]["pool_id"] == "pool_1"


# =============================================================================
# Budget Forecast Tests
# =============================================================================


class TestBudgetForecast:
    """Tests for budget forecasting."""

    @pytest.mark.asyncio
    async def test_returns_forecast(self, adapter, mock_knowledge_mound):
        """Should return budget forecast."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_budget_snapshot",
                    "entity_id": "agent_1",
                    "tokens_used": 1000,
                    "tokens_limit": 10000,
                    "cost_used_usd": 0.10,
                    "period_end": time.time() - 86400 * i,
                },
            }
            for i in range(7)
        ]

        result = await adapter.get_budget_forecast("agent_1", forecast_days=7)

        assert result["forecast_available"] is True
        assert result["entity_id"] == "agent_1"
        assert "daily_avg_tokens" in result
        assert "projected_tokens" in result

    @pytest.mark.asyncio
    async def test_insufficient_data(self, adapter, mock_knowledge_mound):
        """Should report insufficient data."""
        mock_knowledge_mound.query.return_value = [
            {
                "metadata": {
                    "type": "fabric_budget_snapshot",
                    "entity_id": "agent_1",
                    "tokens_used": 1000,
                    "period_end": time.time(),
                },
            }
        ]

        result = await adapter.get_budget_forecast("agent_1")

        assert result["forecast_available"] is False
        assert "Insufficient" in result["reason"]

    @pytest.mark.asyncio
    async def test_no_km_returns_unavailable(self):
        """Should return unavailable without KM."""
        adapter = FabricAdapter(knowledge_mound=None)

        result = await adapter.get_budget_forecast("agent_1")

        assert result["forecast_available"] is False


# =============================================================================
# Sync from Fabric Tests
# =============================================================================


class TestSyncFromFabric:
    """Tests for sync from fabric."""

    @pytest.mark.asyncio
    async def test_syncs_pools(self, adapter_with_fabric, mock_knowledge_mound):
        """Should sync pool snapshots."""
        result = await adapter_with_fabric.sync_from_fabric()

        assert result["pools"] == 1

    @pytest.mark.asyncio
    async def test_syncs_budgets(self, adapter_with_fabric, mock_knowledge_mound):
        """Should sync budget snapshots."""
        result = await adapter_with_fabric.sync_from_fabric()

        assert result["budgets"] == 2  # Two tracked entities in mock

    @pytest.mark.asyncio
    async def test_no_fabric_returns_error(self, adapter):
        """Should return error when no fabric configured."""
        result = await adapter.sync_from_fabric()

        assert "error" in result
        assert "No fabric configured" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_sync_errors(self, adapter_with_fabric, mock_fabric):
        """Should handle sync errors gracefully."""
        mock_fabric.list_pools.side_effect = Exception("Connection failed")

        result = await adapter_with_fabric.sync_from_fabric()

        assert "error" in result


# =============================================================================
# Stats and Health Tests
# =============================================================================


class TestStatsAndHealth:
    """Tests for stats and health reporting."""

    def test_get_stats(self, adapter):
        """Should return stats dict."""
        stats = adapter.get_stats()

        assert "pool_snapshots_stored" in stats
        assert "task_outcomes_stored" in stats
        assert "budget_snapshots_stored" in stats
        assert "policy_decisions_stored" in stats
        assert "pool_queries" in stats
        assert "task_pattern_queries" in stats
        assert stats["workspace_id"] == "test_workspace"
        assert stats["has_knowledge_mound"] is True

    def test_get_stats_cache_sizes(self, adapter):
        """Should include cache sizes in stats."""
        adapter._pool_performance_cache["p1"] = []
        adapter._task_patterns_cache["t1"] = []

        stats = adapter.get_stats()

        assert stats["pool_cache_size"] == 1
        assert stats["task_cache_size"] == 1

    def test_clear_cache(self, adapter):
        """Should clear cache and return count."""
        adapter._pool_performance_cache["p1"] = []
        adapter._pool_performance_cache["p2"] = []
        adapter._task_patterns_cache["t1"] = []
        adapter._cache_times["p1"] = time.time()

        count = adapter.clear_cache()

        assert count == 3
        assert len(adapter._pool_performance_cache) == 0
        assert len(adapter._task_patterns_cache) == 0
        assert len(adapter._cache_times) == 0


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Tests for adapter dataclasses."""

    def test_pool_snapshot_defaults(self):
        """Should have correct defaults."""
        snapshot = PoolSnapshot(
            pool_id="p1",
            name="pool",
            model="claude",
            current_agents=1,
            min_agents=1,
            max_agents=5,
        )

        assert snapshot.tasks_pending == 0
        assert snapshot.tasks_completed == 0
        assert snapshot.avg_task_duration_seconds == 0.0
        assert snapshot.workspace_id == "default"
        assert snapshot.metadata == {}

    def test_pool_snapshot_with_metadata(self):
        """Should accept metadata."""
        snapshot = PoolSnapshot(
            pool_id="p1",
            name="pool",
            model="claude",
            current_agents=1,
            min_agents=1,
            max_agents=5,
            metadata={"region": "us-west"},
        )

        assert snapshot.metadata == {"region": "us-west"}

    def test_task_outcome_defaults(self):
        """Should have correct defaults."""
        outcome = TaskSchedulingOutcome(
            task_id="t1",
            task_type="test",
            agent_id="agent",
            pool_id=None,
            priority=1,
            scheduled_at=100.0,
        )

        assert outcome.completed_at is None
        assert outcome.success is False
        assert outcome.duration_seconds == 0.0
        assert outcome.error_message is None
        assert outcome.workspace_id == "default"

    def test_budget_snapshot_defaults(self):
        """Should have correct defaults."""
        snapshot = BudgetUsageSnapshot(
            entity_id="e1",
            entity_type="agent",
            tokens_used=1000,
            tokens_limit=10000,
            cost_used_usd=0.10,
            cost_limit_usd=1.00,
            period_start=100.0,
            period_end=200.0,
        )

        assert snapshot.alerts_triggered == 0
        assert snapshot.workspace_id == "default"

    def test_policy_decision_defaults(self):
        """Should have correct defaults."""
        decision = PolicyDecisionRecord(
            decision_id="d1",
            agent_id="agent",
            action="test",
            allowed=True,
            policy_id=None,
            reason="Allowed",
            timestamp=100.0,
        )

        assert decision.context == {}
        assert decision.workspace_id == "default"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_handles_ingest_exception(self, adapter, pool_snapshot, mock_knowledge_mound):
        """Should handle ingest exceptions gracefully."""
        mock_knowledge_mound.ingest.side_effect = Exception("Ingest failed")

        result = await adapter.store_pool_snapshot(pool_snapshot)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_query_exception(self, adapter, mock_knowledge_mound):
        """Should handle query exceptions gracefully."""
        mock_knowledge_mound.query.side_effect = Exception("Query failed")

        results = await adapter.get_pool_performance_history("pool_1")

        assert results == []

    def test_event_callback_exception_handled(self, adapter):
        """Should handle event callback exceptions."""
        callback = MagicMock(side_effect=Exception("Callback failed"))
        adapter.set_event_callback(callback)

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_handles_forecast_exception(self, adapter, mock_knowledge_mound):
        """Should handle forecast query exceptions."""
        mock_knowledge_mound.query.side_effect = Exception("Query failed")

        result = await adapter.get_budget_forecast("entity_1")

        assert result["forecast_available"] is False
        assert "error" in result


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_emits_task_outcome_event(self, adapter, task_outcome, mock_knowledge_mound):
        """Should emit event on task outcome storage."""
        events = []
        adapter.set_event_callback(lambda t, d: events.append((t, d)))

        await adapter.store_task_outcome(task_outcome)

        assert len(events) == 1
        assert events[0][0] == "fabric_task_outcome_stored"

    @pytest.mark.asyncio
    async def test_emits_budget_snapshot_event(
        self, adapter, budget_snapshot, mock_knowledge_mound
    ):
        """Should emit event on budget snapshot storage."""
        events = []
        adapter.set_event_callback(lambda t, d: events.append((t, d)))

        await adapter.store_budget_snapshot(budget_snapshot)

        assert len(events) == 1
        assert events[0][0] == "fabric_budget_snapshot_stored"

    @pytest.mark.asyncio
    async def test_emits_policy_decision_event(
        self, adapter, policy_decision, mock_knowledge_mound
    ):
        """Should emit event on policy decision storage."""
        events = []
        adapter.set_event_callback(lambda t, d: events.append((t, d)))

        await adapter.store_policy_decision(policy_decision)

        assert len(events) == 1
        assert events[0][0] == "fabric_policy_decision_stored"
