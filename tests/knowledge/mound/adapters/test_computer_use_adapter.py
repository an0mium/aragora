"""
Tests for ComputerUseAdapter - Bridges Computer-Use Orchestrator to Knowledge Mound.

Tests cover:
- TaskExecutionRecord, ActionPerformanceRecord, PolicyBlockRecord dataclasses
- Adapter initialization
- Task result storage
- Action performance storage
- Policy block storage
- Similar task queries
- Task recommendations
- Statistics
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from aragora.knowledge.mound.adapters.computer_use_adapter import (
    ComputerUseAdapter,
    TaskExecutionRecord,
    ActionPerformanceRecord,
    PolicyBlockRecord,
)


# =============================================================================
# TaskExecutionRecord Dataclass Tests
# =============================================================================


class TestTaskExecutionRecord:
    """Tests for TaskExecutionRecord dataclass."""

    def test_create_record(self):
        """Should create a task execution record."""
        record = TaskExecutionRecord(
            task_id="task-001",
            goal="Open the settings menu",
            status="completed",
            total_steps=5,
            successful_steps=5,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=12.5,
        )

        assert record.task_id == "task-001"
        assert record.goal == "Open the settings menu"
        assert record.status == "completed"
        assert record.successful_steps == 5

    def test_record_defaults(self):
        """Should use default values."""
        record = TaskExecutionRecord(
            task_id="task-002",
            goal="Test goal",
            status="failed",
            total_steps=3,
            successful_steps=1,
            failed_steps=2,
            blocked_steps=0,
            duration_seconds=5.0,
        )

        assert record.agent_id is None
        assert record.error_message is None
        assert record.workspace_id == "default"
        assert record.metadata == {}

    def test_record_with_all_fields(self):
        """Should accept all fields."""
        record = TaskExecutionRecord(
            task_id="task-003",
            goal="Complex task",
            status="blocked",
            total_steps=10,
            successful_steps=5,
            failed_steps=2,
            blocked_steps=3,
            duration_seconds=30.0,
            agent_id="agent-123",
            error_message="Policy blocked",
            workspace_id="ws-custom",
            metadata={"action_types": ["click", "type"]},
        )

        assert record.agent_id == "agent-123"
        assert record.error_message == "Policy blocked"
        assert record.metadata["action_types"] == ["click", "type"]


# =============================================================================
# ActionPerformanceRecord Dataclass Tests
# =============================================================================


class TestActionPerformanceRecord:
    """Tests for ActionPerformanceRecord dataclass."""

    def test_create_record(self):
        """Should create an action performance record."""
        record = ActionPerformanceRecord(
            action_type="click",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            avg_duration_ms=50.5,
        )

        assert record.action_type == "click"
        assert record.total_executions == 100
        assert record.successful_executions == 95

    def test_record_defaults(self):
        """Should use default values."""
        record = ActionPerformanceRecord(
            action_type="type",
            total_executions=50,
            successful_executions=48,
            failed_executions=2,
            avg_duration_ms=120.0,
        )

        assert record.policy_blocked_count == 0
        assert record.workspace_id == "default"


# =============================================================================
# PolicyBlockRecord Dataclass Tests
# =============================================================================


class TestPolicyBlockRecord:
    """Tests for PolicyBlockRecord dataclass."""

    def test_create_record(self):
        """Should create a policy block record."""
        record = PolicyBlockRecord(
            block_id="block-001",
            task_id="task-001",
            action_type="click",
            element_selector="#admin-panel",
            domain="admin.example.com",
            policy_rule="no_admin_access",
            reason="Admin panels are restricted",
            timestamp=time.time(),
        )

        assert record.block_id == "block-001"
        assert record.action_type == "click"
        assert record.policy_rule == "no_admin_access"

    def test_record_defaults(self):
        """Should use default workspace."""
        record = PolicyBlockRecord(
            block_id="block-002",
            task_id="task-002",
            action_type="navigate",
            element_selector=None,
            domain="blocked.com",
            policy_rule="domain_blocklist",
            reason="Domain is blocked",
            timestamp=time.time(),
        )

        assert record.workspace_id == "default"


# =============================================================================
# ComputerUseAdapter Initialization Tests
# =============================================================================


class TestComputerUseAdapterInit:
    """Tests for ComputerUseAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = ComputerUseAdapter()

        assert adapter._orchestrator is None
        assert adapter._knowledge_mound is None
        assert adapter._workspace_id == "default"
        assert adapter._min_confidence_threshold == 0.6
        assert adapter.adapter_name == "computer_use"

    def test_init_with_orchestrator(self):
        """Should accept orchestrator instance."""
        mock_orchestrator = MagicMock()
        adapter = ComputerUseAdapter(orchestrator=mock_orchestrator)

        assert adapter._orchestrator is mock_orchestrator

    def test_init_with_knowledge_mound(self):
        """Should accept knowledge mound."""
        mock_km = MagicMock()
        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        assert adapter._knowledge_mound is mock_km

    def test_init_with_workspace(self):
        """Should accept workspace ID."""
        adapter = ComputerUseAdapter(workspace_id="ws-custom")

        assert adapter._workspace_id == "ws-custom"

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = ComputerUseAdapter(event_callback=callback)

        assert adapter._event_callback is callback


# =============================================================================
# Task Execution Record Storage Tests
# =============================================================================


class TestStoreTaskExecutionRecord:
    """Tests for storing task execution records."""

    @pytest.mark.asyncio
    async def test_store_record(self):
        """Should store task execution record."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        record = TaskExecutionRecord(
            task_id="task-001",
            goal="Open settings",
            status="completed",
            total_steps=5,
            successful_steps=5,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=10.0,
        )

        result = await adapter.store_task_execution_record(record)

        assert result == "item-001"
        mock_km.ingest.assert_called_once()
        assert adapter._stats["task_records_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = ComputerUseAdapter()

        record = TaskExecutionRecord(
            task_id="task-002",
            goal="Test goal",
            status="completed",
            total_steps=3,
            successful_steps=3,
            failed_steps=0,
            blocked_steps=0,
            duration_seconds=5.0,
        )

        result = await adapter.store_task_execution_record(record)

        assert result is None


# =============================================================================
# Action Performance Storage Tests
# =============================================================================


class TestStoreActionPerformance:
    """Tests for storing action performance."""

    @pytest.mark.asyncio
    async def test_store_performance(self):
        """Should store action performance record."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        record = ActionPerformanceRecord(
            action_type="click",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            avg_duration_ms=50.0,
        )

        result = await adapter.store_action_performance(record)

        assert result == "item-001"
        assert adapter._stats["action_records_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = ComputerUseAdapter()

        record = ActionPerformanceRecord(
            action_type="type",
            total_executions=50,
            successful_executions=48,
            failed_executions=2,
            avg_duration_ms=120.0,
        )

        result = await adapter.store_action_performance(record)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidates_cache(self):
        """Should invalidate action stats cache."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)
        adapter._action_stats_cache["click"] = MagicMock()

        record = ActionPerformanceRecord(
            action_type="click",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            avg_duration_ms=50.0,
        )

        await adapter.store_action_performance(record)

        assert "click" not in adapter._action_stats_cache


# =============================================================================
# Policy Block Storage Tests
# =============================================================================


class TestStorePolicyBlock:
    """Tests for storing policy blocks."""

    @pytest.mark.asyncio
    async def test_store_block(self):
        """Should store policy block record."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        record = PolicyBlockRecord(
            block_id="block-001",
            task_id="task-001",
            action_type="click",
            element_selector="#admin-panel",
            domain="admin.example.com",
            policy_rule="no_admin_access",
            reason="Admin panels are restricted",
            timestamp=time.time(),
        )

        result = await adapter.store_policy_block(record)

        assert result == "item-001"
        assert adapter._stats["policy_blocks_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = ComputerUseAdapter()

        record = PolicyBlockRecord(
            block_id="block-002",
            task_id="task-002",
            action_type="navigate",
            element_selector=None,
            domain="blocked.com",
            policy_rule="domain_blocklist",
            reason="Domain is blocked",
            timestamp=time.time(),
        )

        result = await adapter.store_policy_block(record)

        assert result is None


# =============================================================================
# Similar Tasks Query Tests
# =============================================================================


class TestGetSimilarTasks:
    """Tests for finding similar tasks."""

    @pytest.mark.asyncio
    async def test_query_no_km(self):
        """Should return empty list without KM."""
        adapter = ComputerUseAdapter()

        result = await adapter.get_similar_tasks("open settings")

        assert result == []
        assert adapter._stats["task_queries"] == 1

    @pytest.mark.asyncio
    async def test_query_with_results(self):
        """Should return matching tasks from KM."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_task",
                        "task_id": "task-001",
                        "goal": "Open settings menu",
                        "status": "completed",
                        "total_steps": 5,
                        "successful_steps": 5,
                        "failed_steps": 0,
                        "blocked_steps": 0,
                        "duration_seconds": 10.0,
                    },
                    "score": 0.9,
                }
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_similar_tasks("open settings")

        assert len(result) == 1
        assert result[0].task_id == "task-001"

    @pytest.mark.asyncio
    async def test_query_success_only(self):
        """Should filter for successful tasks only."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_task",
                        "task_id": "task-success",
                        "goal": "Open settings",
                        "status": "completed",
                        "total_steps": 5,
                        "successful_steps": 5,
                        "failed_steps": 0,
                        "blocked_steps": 0,
                        "duration_seconds": 10.0,
                    },
                    "score": 0.9,
                },
                {
                    "metadata": {
                        "type": "computer_use_task",
                        "task_id": "task-failed",
                        "goal": "Open settings",
                        "status": "failed",
                        "total_steps": 5,
                        "successful_steps": 2,
                        "failed_steps": 3,
                        "blocked_steps": 0,
                        "duration_seconds": 15.0,
                    },
                    "score": 0.85,
                },
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_similar_tasks("open settings", success_only=True)

        assert len(result) == 1
        assert result[0].task_id == "task-success"


# =============================================================================
# Action Statistics Query Tests
# =============================================================================


class TestGetActionStatistics:
    """Tests for querying action statistics."""

    @pytest.mark.asyncio
    async def test_query_no_km(self):
        """Should return empty dict without KM."""
        adapter = ComputerUseAdapter()

        result = await adapter.get_action_statistics()

        assert result == {}
        assert adapter._stats["action_queries"] == 1

    @pytest.mark.asyncio
    async def test_query_with_results(self):
        """Should return action stats from KM."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_action_performance",
                        "action_type": "click",
                        "total_executions": 100,
                        "successful_executions": 95,
                        "failed_executions": 5,
                        "avg_duration_ms": 50.0,
                        "policy_blocked_count": 2,
                    }
                }
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_action_statistics()

        assert "click" in result
        assert result["click"].total_executions == 100

    @pytest.mark.asyncio
    async def test_query_specific_action(self):
        """Should filter by action type."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_action_performance",
                        "action_type": "click",
                        "total_executions": 100,
                        "successful_executions": 95,
                        "failed_executions": 5,
                        "avg_duration_ms": 50.0,
                        "policy_blocked_count": 2,
                    }
                },
                {
                    "metadata": {
                        "type": "computer_use_action_performance",
                        "action_type": "type",
                        "total_executions": 50,
                        "successful_executions": 48,
                        "failed_executions": 2,
                        "avg_duration_ms": 120.0,
                        "policy_blocked_count": 0,
                    }
                },
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_action_statistics(action_type="click")

        assert "click" in result
        assert "type" not in result


# =============================================================================
# Task Recommendations Tests
# =============================================================================


class TestGetTaskRecommendations:
    """Tests for task recommendations."""

    @pytest.mark.asyncio
    async def test_no_similar_tasks(self):
        """Should return no tasks message."""
        adapter = ComputerUseAdapter()

        result = await adapter.get_task_recommendations("unique task")

        assert len(result) >= 1
        assert "No similar tasks found" in result[0]["recommendation"]

    @pytest.mark.asyncio
    async def test_recommendations_from_history(self):
        """Should generate recommendations from similar tasks."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_task",
                        "task_id": f"task-{i}",
                        "goal": "Open settings",
                        "status": "completed",
                        "total_steps": 5,
                        "successful_steps": 5,
                        "failed_steps": 0,
                        "blocked_steps": 0,
                        "duration_seconds": 10.0,
                        "agent_id": "agent-best" if i % 2 == 0 else None,
                    },
                    "score": 0.9 - i * 0.01,
                }
                for i in range(5)
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_task_recommendations("open settings")

        assert len(result) >= 1
        # Should have steps and duration recommendations
        types = [r["type"] for r in result]
        assert "steps" in types
        assert "duration" in types


# =============================================================================
# Policy Block Analysis Tests
# =============================================================================


class TestGetPolicyBlockAnalysis:
    """Tests for policy block analysis."""

    @pytest.mark.asyncio
    async def test_analysis_no_km(self):
        """Should return unavailable without KM."""
        adapter = ComputerUseAdapter()

        result = await adapter.get_policy_block_analysis()

        assert result["analysis_available"] is False

    @pytest.mark.asyncio
    async def test_analysis_with_data(self):
        """Should analyze policy blocks from KM."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "computer_use_policy_block",
                        "block_id": f"block-{i}",
                        "action_type": "click" if i % 2 == 0 else "navigate",
                        "policy_rule": "admin_restriction",
                        "domain": "admin.example.com",
                        "timestamp": time.time(),
                    }
                }
                for i in range(10)
            ]
        )

        adapter = ComputerUseAdapter(knowledge_mound=mock_km)

        result = await adapter.get_policy_block_analysis()

        assert result["analysis_available"] is True
        assert result["total_blocks"] == 10
        assert "blocks_by_rule" in result
        assert "blocks_by_action" in result

    def test_generate_recommendations_no_blocks(self):
        """Should handle no blocks."""
        adapter = ComputerUseAdapter()

        result = adapter._generate_policy_recommendations({}, {}, 0)

        assert "No policy blocks" in result[0]

    def test_generate_recommendations_high_frequency_rule(self):
        """Should recommend reviewing high-frequency rules."""
        adapter = ComputerUseAdapter()

        result = adapter._generate_policy_recommendations(
            {"admin_restriction": 40},
            {"click": 40},
            100,
        )

        assert any("admin_restriction" in r for r in result)


# =============================================================================
# Stats and Cache Tests
# =============================================================================


class TestStatsAndCache:
    """Tests for statistics and cache operations."""

    def test_get_stats(self):
        """Should return adapter stats."""
        adapter = ComputerUseAdapter(workspace_id="ws-test")

        stats = adapter.get_stats()

        assert stats["task_records_stored"] == 0
        assert stats["action_records_stored"] == 0
        assert stats["policy_blocks_stored"] == 0
        assert stats["workspace_id"] == "ws-test"
        assert stats["has_knowledge_mound"] is False
        assert stats["has_orchestrator"] is False

    def test_clear_cache(self):
        """Should clear all caches."""
        adapter = ComputerUseAdapter()

        # Populate caches
        adapter._task_patterns_cache["test"] = [MagicMock()]
        adapter._action_stats_cache["click"] = MagicMock()
        adapter._cache_times["test"] = time.time()

        count = adapter.clear_cache()

        assert count == 2
        assert len(adapter._task_patterns_cache) == 0
        assert len(adapter._action_stats_cache) == 0
        assert len(adapter._cache_times) == 0


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Tests for sync operations."""

    @pytest.mark.asyncio
    async def test_sync_no_orchestrator(self):
        """Should return error without orchestrator."""
        adapter = ComputerUseAdapter()

        result = await adapter.sync_from_orchestrator()

        assert "error" in result
        assert "No orchestrator configured" in result["error"]
