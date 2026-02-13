"""
End-to-End Integration Tests: Control Plane <-> Knowledge Mound Learning Loop.

Tests the complete bidirectional flow between Control Plane and KM:
1. Task completion -> KM storage
2. KM recommendations -> Agent selection
3. Cross-workspace insight sharing
4. Full learning loop verification
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.coordinator import (
    ControlPlaneCoordinator,
    ControlPlaneConfig,
)
from aragora.control_plane.scheduler import Task, TaskStatus, TaskPriority
from aragora.knowledge.mound.adapters.control_plane_adapter import (
    ControlPlaneAdapter,
    TaskOutcome,
    AgentCapabilityRecord,
    CrossWorkspaceInsight,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_km_item(content, confidence, metadata):
    """Create a mock KnowledgeItem-like object with attribute access.

    The adapter code uses ``getattr(result, "metadata", {})`` and similar
    attribute access, so plain dicts won't work -- we need objects.
    """
    item = MagicMock()
    item.content = content
    item.confidence = confidence
    item.metadata = metadata
    return item


def _make_query_result(items):
    """Wrap a list of items in an object with an ``.items`` attribute.

    The adapter code does:
        query_result.items if hasattr(query_result, "items") else []
    so the mock query must return an object with ``.items``, not a plain list.
    """
    qr = MagicMock()
    qr.items = items
    return qr


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock KnowledgeMound with storage."""
    mound = MagicMock()
    mound._items = {}
    mound._item_counter = 0

    async def mock_ingest(item):
        mound._item_counter += 1
        item_id = f"km_{mound._item_counter:03d}"
        mound._items[item_id] = {
            "id": item.id,
            "content": item.content,
            "source": item.source.value if hasattr(item.source, "value") else str(item.source),
            "metadata": item.metadata,
            "confidence": item.confidence.value
            if hasattr(item.confidence, "value")
            else str(item.confidence),
        }
        return item_id

    async def mock_query(query, limit=10, **kwargs):
        results = []
        for item_id, item in mound._items.items():
            if query.lower() in item["content"].lower():
                results.append(
                    {
                        "id": item_id,
                        "content": item["content"],
                        "confidence": 0.9,
                        "metadata": item["metadata"],
                    }
                )
        return results[:limit]

    mound.ingest = AsyncMock(side_effect=mock_ingest)
    mound.query = AsyncMock(side_effect=mock_query)

    return mound


@pytest.fixture
def mock_registry():
    """Create a mock AgentRegistry."""
    registry = MagicMock()
    registry.connect = AsyncMock()
    registry.close = AsyncMock()
    registry.register = AsyncMock()
    registry.heartbeat = AsyncMock(return_value=True)
    registry.record_task_completion = AsyncMock()
    registry.get_stats = AsyncMock(return_value={"agents": 3})
    return registry


@pytest.fixture
def mock_scheduler():
    """Create a mock TaskScheduler."""
    scheduler = MagicMock()
    scheduler._tasks = {}
    scheduler.connect = AsyncMock()
    scheduler.close = AsyncMock()

    async def mock_submit(task_type, payload, **kwargs):
        task_id = f"task_{len(scheduler._tasks) + 1:03d}"
        scheduler._tasks[task_id] = Task(
            id=task_id,
            task_type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            priority=kwargs.get("priority", TaskPriority.NORMAL),
            created_at=datetime.now(),
            timeout_seconds=kwargs.get("timeout_seconds", 300),
            metadata=kwargs.get("metadata", {}),
        )
        return task_id

    async def mock_get(task_id):
        return scheduler._tasks.get(task_id)

    async def mock_complete(task_id, result=None):
        if task_id in scheduler._tasks:
            scheduler._tasks[task_id] = Task(
                id=task_id,
                task_type=scheduler._tasks[task_id].task_type,
                payload=scheduler._tasks[task_id].payload,
                status=TaskStatus.COMPLETED,
                priority=scheduler._tasks[task_id].priority,
                created_at=scheduler._tasks[task_id].created_at,
                timeout_seconds=scheduler._tasks[task_id].timeout_seconds,
                result=result,
                metadata=scheduler._tasks[task_id].metadata,
            )
            return True
        return False

    async def mock_fail(task_id, error, requeue=True):
        if task_id in scheduler._tasks:
            scheduler._tasks[task_id] = Task(
                id=task_id,
                task_type=scheduler._tasks[task_id].task_type,
                payload=scheduler._tasks[task_id].payload,
                status=TaskStatus.FAILED,
                priority=scheduler._tasks[task_id].priority,
                created_at=scheduler._tasks[task_id].created_at,
                timeout_seconds=scheduler._tasks[task_id].timeout_seconds,
                error=error,
                metadata=scheduler._tasks[task_id].metadata,
            )
            return True
        return False

    scheduler.submit = AsyncMock(side_effect=mock_submit)
    scheduler.get = AsyncMock(side_effect=mock_get)
    scheduler.complete = AsyncMock(side_effect=mock_complete)
    scheduler.fail = AsyncMock(side_effect=mock_fail)
    scheduler.get_stats = AsyncMock(return_value={"tasks": 0})

    return scheduler


@pytest.fixture
def mock_health_monitor():
    """Create a mock HealthMonitor."""
    monitor = MagicMock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.get_stats = MagicMock(return_value={"healthy": True})
    monitor._health_checks = {}
    return monitor


@pytest.fixture
def coordinator(mock_registry, mock_scheduler, mock_health_monitor, mock_knowledge_mound):
    """Create a ControlPlaneCoordinator with KM integration."""
    config = ControlPlaneConfig(
        enable_km_integration=True,
        km_workspace_id="test_workspace",
    )

    coord = ControlPlaneCoordinator(
        config=config,
        registry=mock_registry,
        scheduler=mock_scheduler,
        health_monitor=mock_health_monitor,
        knowledge_mound=mock_knowledge_mound,
    )

    return coord


# ============================================================================
# Task Outcome -> KM Storage Tests
# ============================================================================


class TestTaskOutcomeStorage:
    """Tests for task outcome storage in KM."""

    @pytest.mark.asyncio
    async def test_successful_task_stored_in_km(self, coordinator, mock_knowledge_mound):
        """Completing a task should store outcome in KM."""
        # Submit and complete a task
        task_id = await coordinator.submit_task(
            task_type="debate",
            payload={"topic": "test"},
            metadata={"domain": "ai"},
        )

        await coordinator.complete_task(
            task_id=task_id,
            result={"consensus": True},
            agent_id="claude-3",
            latency_ms=5000,
        )

        # Verify KM received the outcome
        assert mock_knowledge_mound.ingest.called
        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "cp_task_" in call_args.id
        assert "debate" in call_args.content
        assert "claude-3" in call_args.content
        assert "success" in call_args.content

    @pytest.mark.asyncio
    async def test_failed_task_stored_in_km(self, coordinator, mock_knowledge_mound):
        """Permanently failed tasks should store failure outcomes."""
        # Lower the confidence threshold to allow failed tasks
        coordinator._km_adapter._min_task_confidence = 0.4

        task_id = await coordinator.submit_task(
            task_type="code_review",
            payload={"file": "test.py"},
        )

        await coordinator.fail_task(
            task_id=task_id,
            error="Timeout exceeded",
            agent_id="gpt-4",
            latency_ms=30000,
            requeue=False,  # Permanent failure
        )

        # Verify KM received the failure
        assert mock_knowledge_mound.ingest.called
        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "failure" in call_args.content
        assert call_args.metadata["error_message"] == "Timeout exceeded"

    @pytest.mark.asyncio
    async def test_requeued_task_not_stored(self, coordinator, mock_knowledge_mound):
        """Requeued tasks should NOT store outcomes (will retry)."""
        task_id = await coordinator.submit_task(
            task_type="analysis",
            payload={"data": "test"},
        )

        await coordinator.fail_task(
            task_id=task_id,
            error="Temporary error",
            agent_id="claude-3",
            requeue=True,  # Will retry
        )

        # Should NOT call ingest for requeued tasks
        assert not mock_knowledge_mound.ingest.called


# ============================================================================
# KM Recommendations -> Agent Selection Tests
# ============================================================================


class TestKMRecommendations:
    """Tests for KM-based agent recommendations."""

    @pytest.mark.asyncio
    async def test_get_recommendations_from_km(self, coordinator, mock_knowledge_mound):
        """Should query KM for agent recommendations."""
        # The adapter's get_capability_recommendations expects query() to return
        # an object with .items attribute, where each item has .metadata,
        # .confidence, and .content as attributes (KnowledgeItem-like objects).
        mock_knowledge_mound.query.side_effect = None

        item1 = _make_km_item(
            content="Agent claude-3 capability 'debate': 95% success",
            confidence=0.9,
            metadata={
                "type": "control_plane_capability",
                "agent_id": "claude-3",
                "capability": "debate",
                "success_count": 95,
                "failure_count": 5,
                "avg_duration_seconds": 12.5,
            },
        )
        item2 = _make_km_item(
            content="Agent gpt-4 capability 'debate': 85% success",
            confidence=0.8,
            metadata={
                "type": "control_plane_capability",
                "agent_id": "gpt-4",
                "capability": "debate",
                "success_count": 85,
                "failure_count": 15,
                "avg_duration_seconds": 15.0,
            },
        )

        mock_knowledge_mound.query.return_value = _make_query_result([item1, item2])

        recommendations = await coordinator.get_agent_recommendations("debate")

        assert len(recommendations) == 2
        assert recommendations[0]["agent_id"] == "claude-3"
        assert recommendations[0]["success_rate"] == 0.95
        assert recommendations[1]["agent_id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_recommendations_empty_without_km(self):
        """Should return empty list when KM not configured."""
        config = ControlPlaneConfig(enable_km_integration=False)
        coord = ControlPlaneCoordinator(config=config)

        recommendations = await coord.get_agent_recommendations("debate")

        assert recommendations == []


# ============================================================================
# Cross-Workspace Insight Sharing Tests
# ============================================================================


class TestCrossWorkspaceInsights:
    """Tests for cross-workspace knowledge sharing."""

    @pytest.mark.asyncio
    async def test_share_insight_across_workspaces(self, mock_knowledge_mound):
        """Should share insights via KM."""
        adapter = ControlPlaneAdapter(
            knowledge_mound=mock_knowledge_mound,
            workspace_id="workspace_a",
        )

        insight = CrossWorkspaceInsight(
            insight_id="insight_001",
            source_workspace="workspace_a",
            target_workspaces=["workspace_b", "workspace_c"],
            task_type="debate",
            content="Structured debates with 3 rounds work best",
            confidence=0.85,
            created_at=datetime.now().isoformat(),
        )

        result = await adapter.share_insight_cross_workspace(insight)

        assert result is True
        assert mock_knowledge_mound.ingest.called
        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "cp_insight_" in call_args.id
        assert "Structured debates" in call_args.content

    @pytest.mark.asyncio
    async def test_get_insights_filters_own_workspace(self, mock_knowledge_mound):
        """Should filter out insights from own workspace."""
        adapter = ControlPlaneAdapter(
            knowledge_mound=mock_knowledge_mound,
            workspace_id="workspace_a",
        )

        # The adapter's get_cross_workspace_insights expects query() to return
        # an object with .items attribute, where each item has .metadata,
        # .confidence, and .content as attributes (KnowledgeItem-like objects).
        mock_knowledge_mound.query.side_effect = None

        item1 = _make_km_item(
            content="Own insight",
            confidence=0.8,
            metadata={
                "type": "cross_workspace_insight",
                "insight_id": "insight_1",
                "source_workspace": "workspace_a",  # Same as adapter
                "target_workspaces": [],
                "task_type": "debate",
            },
        )
        item2 = _make_km_item(
            content="Other workspace insight",
            confidence=0.9,
            metadata={
                "type": "cross_workspace_insight",
                "insight_id": "insight_2",
                "source_workspace": "workspace_b",  # Different
                "target_workspaces": ["workspace_a"],
                "task_type": "debate",
            },
        )

        mock_knowledge_mound.query.return_value = _make_query_result([item1, item2])

        insights = await adapter.get_cross_workspace_insights("debate")

        assert len(insights) == 1
        assert insights[0].source_workspace == "workspace_b"


# ============================================================================
# Full Learning Loop Tests
# ============================================================================


class TestFullLearningLoop:
    """Tests for the complete learning loop."""

    @pytest.mark.asyncio
    async def test_learning_loop_complete_flow(self, coordinator, mock_knowledge_mound):
        """Test complete flow: task -> KM -> recommendations."""
        # Step 1: Complete several tasks
        for i in range(5):
            task_id = await coordinator.submit_task(
                task_type="debate",
                payload={"topic": f"topic_{i}"},
            )
            await coordinator.complete_task(
                task_id=task_id,
                result={"consensus": True},
                agent_id="claude-3",
                latency_ms=5000 + i * 100,
            )

        # Verify outcomes were stored
        assert mock_knowledge_mound.ingest.call_count == 5

        # Step 2: Query for recommendations (clear side_effect first)
        # The adapter expects query() to return an object with .items,
        # containing KnowledgeItem-like objects with attribute access.
        mock_knowledge_mound.query.side_effect = None

        item = _make_km_item(
            content="Agent claude-3 capability 'debate': 100% success",
            confidence=0.95,
            metadata={
                "type": "control_plane_capability",
                "agent_id": "claude-3",
                "capability": "debate",
                "success_count": 5,
                "failure_count": 0,
                "avg_duration_seconds": 5.2,
            },
        )

        mock_knowledge_mound.query.return_value = _make_query_result([item])

        recommendations = await coordinator.get_agent_recommendations("debate")

        # Step 3: Verify learning worked
        assert len(recommendations) == 1
        assert recommendations[0]["agent_id"] == "claude-3"
        assert recommendations[0]["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_include_km_data(self, coordinator, mock_knowledge_mound):
        """Stats should include KM adapter data."""
        # Complete a task to trigger KM storage
        task_id = await coordinator.submit_task(
            task_type="test",
            payload={},
        )
        await coordinator.complete_task(
            task_id=task_id,
            agent_id="test-agent",
            latency_ms=1000,
        )

        stats = await coordinator.get_stats()

        assert "knowledge_mound" in stats
        assert stats["knowledge_mound"]["task_outcomes_stored"] == 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_km_failure_does_not_block_task(self, coordinator, mock_knowledge_mound):
        """KM storage failure should not block task completion."""
        mock_knowledge_mound.ingest.side_effect = Exception("KM unavailable")

        task_id = await coordinator.submit_task(
            task_type="test",
            payload={},
        )

        # Should succeed despite KM failure
        result = await coordinator.complete_task(
            task_id=task_id,
            agent_id="test-agent",
            latency_ms=1000,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_km_adapter_can_be_set_after_init(
        self, mock_registry, mock_scheduler, mock_health_monitor, mock_knowledge_mound
    ):
        """KM adapter can be configured after coordinator creation."""
        config = ControlPlaneConfig(enable_km_integration=True)
        coord = ControlPlaneCoordinator(
            config=config,
            registry=mock_registry,
            scheduler=mock_scheduler,
            health_monitor=mock_health_monitor,
        )

        # Initially no adapter
        assert coord.km_adapter is None

        # Set adapter
        adapter = ControlPlaneAdapter(
            knowledge_mound=mock_knowledge_mound,
            workspace_id="test",
        )
        coord.set_km_adapter(adapter)

        assert coord.km_adapter is not None

    @pytest.mark.asyncio
    async def test_capability_record_aggregation(self, mock_knowledge_mound):
        """Should aggregate capability records correctly."""
        adapter = ControlPlaneAdapter(
            knowledge_mound=mock_knowledge_mound,
            workspace_id="test",
            min_capability_sample_size=3,
        )

        # Store multiple task outcomes
        for i in range(5):
            outcome = TaskOutcome(
                task_id=f"task_{i}",
                task_type="debate",
                agent_id="claude-3",
                success=i < 4,  # 4 successes, 1 failure
                duration_seconds=10.0 + i,
            )
            await adapter.store_task_outcome(outcome)

        # Create capability record from aggregated data
        record = AgentCapabilityRecord(
            agent_id="claude-3",
            capability="debate",
            success_count=4,
            failure_count=1,
            avg_duration_seconds=12.0,
            confidence=0.9,
        )

        result = await adapter.store_capability_record(record)

        assert result is not None
        # Verify the stored record has correct success rate
        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "80.0% success rate" in call_args.content
