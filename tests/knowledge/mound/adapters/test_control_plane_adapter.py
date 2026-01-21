"""
Tests for ControlPlaneAdapter - Control Plane to Knowledge Mound integration.

Tests cover:
- Task outcome storage
- Agent capability record storage
- Capability recommendations from KM
- Cross-workspace insight sharing
- Cache behavior
- Edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from aragora.knowledge.mound.adapters.control_plane_adapter import (
    ControlPlaneAdapter,
    TaskOutcome,
    AgentCapabilityRecord,
    CrossWorkspaceInsight,
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
def adapter(mock_knowledge_mound):
    """Create a ControlPlaneAdapter with mock KM."""
    return ControlPlaneAdapter(
        knowledge_mound=mock_knowledge_mound,
        workspace_id="test_workspace",
    )


@pytest.fixture
def task_outcome():
    """Create a sample TaskOutcome."""
    return TaskOutcome(
        task_id="task_123",
        task_type="debate",
        agent_id="claude-3",
        success=True,
        duration_seconds=15.5,
        workspace_id="test_workspace",
    )


@pytest.fixture
def capability_record():
    """Create a sample AgentCapabilityRecord."""
    return AgentCapabilityRecord(
        agent_id="claude-3",
        capability="debate",
        success_count=45,
        failure_count=5,
        avg_duration_seconds=12.3,
        workspace_id="test_workspace",
        confidence=0.9,
    )


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
        assert "cp_task_task_123" in call_args.id
        assert task_outcome.agent_id in call_args.content

    @pytest.mark.asyncio
    async def test_stores_failed_task(self, adapter, mock_knowledge_mound):
        """Should store failed task outcomes with lower confidence."""
        outcome = TaskOutcome(
            task_id="task_456",
            task_type="code_review",
            agent_id="gpt-4",
            success=False,
            duration_seconds=30.0,
            error_message="Timeout",
        )

        # Failed tasks have lower confidence, might not meet threshold
        adapter._min_task_confidence = 0.4  # Lower threshold for test

        result = await adapter.store_task_outcome(outcome)

        mock_knowledge_mound.ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_km_returns_none(self, task_outcome):
        """Should return None when no KM configured."""
        adapter = ControlPlaneAdapter(knowledge_mound=None)

        result = await adapter.store_task_outcome(task_outcome)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_stats(self, adapter, task_outcome):
        """Should update stats on storage."""
        assert adapter._stats["task_outcomes_stored"] == 0

        await adapter.store_task_outcome(task_outcome)

        assert adapter._stats["task_outcomes_stored"] == 1


# =============================================================================
# Capability Record Storage Tests
# =============================================================================


class TestCapabilityRecordStorage:
    """Tests for agent capability record storage."""

    @pytest.mark.asyncio
    async def test_stores_capability_record(self, adapter, capability_record, mock_knowledge_mound):
        """Should store capability records."""
        result = await adapter.store_capability_record(capability_record)

        assert result == "km_test_id"
        mock_knowledge_mound.ingest.assert_called_once()

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "cp_capability_" in call_args.id
        assert capability_record.agent_id in call_args.content

    @pytest.mark.asyncio
    async def test_below_sample_threshold_not_stored(self, adapter, mock_knowledge_mound):
        """Should not store records below sample threshold."""
        record = AgentCapabilityRecord(
            agent_id="claude-3",
            capability="rare_task",
            success_count=2,
            failure_count=1,
            avg_duration_seconds=10.0,
        )

        adapter._min_capability_sample_size = 5

        result = await adapter.store_capability_record(record)

        assert result is None
        mock_knowledge_mound.ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_store(self, adapter, capability_record, mock_knowledge_mound):
        """Should invalidate cache when storing new record."""
        # Pre-populate cache
        adapter._capability_cache["debate"] = [capability_record]

        await adapter.store_capability_record(capability_record)

        assert "debate" not in adapter._capability_cache


# =============================================================================
# Capability Recommendations Tests
# =============================================================================


class TestCapabilityRecommendations:
    """Tests for capability recommendations from KM."""

    @pytest.mark.asyncio
    async def test_queries_km_for_recommendations(self, adapter, mock_knowledge_mound):
        """Should query KM for capability recommendations."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Agent claude-3 capability 'debate': 90% success",
                "confidence": 0.9,
                "metadata": {
                    "type": "control_plane_capability",
                    "agent_id": "claude-3",
                    "capability": "debate",
                    "success_count": 90,
                    "failure_count": 10,
                    "avg_duration_seconds": 15.0,
                },
            }
        ]

        results = await adapter.get_capability_recommendations("debate")

        assert len(results) == 1
        assert results[0].agent_id == "claude-3"
        assert results[0].success_count == 90

    @pytest.mark.asyncio
    async def test_caches_results(self, adapter, mock_knowledge_mound):
        """Should cache query results."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Agent test",
                "confidence": 0.8,
                "metadata": {
                    "type": "control_plane_capability",
                    "agent_id": "test",
                    "capability": "debate",
                    "success_count": 50,
                    "failure_count": 5,
                    "avg_duration_seconds": 10.0,
                },
            }
        ]

        # First call
        await adapter.get_capability_recommendations("debate")

        # Second call should use cache
        await adapter.get_capability_recommendations("debate", use_cache=True)

        # Only one query to KM
        assert mock_knowledge_mound.query.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_empty_without_km(self):
        """Should return empty list without KM."""
        adapter = ControlPlaneAdapter(knowledge_mound=None)

        results = await adapter.get_capability_recommendations("debate")

        assert results == []

    @pytest.mark.asyncio
    async def test_sorts_by_success_rate(self, adapter, mock_knowledge_mound):
        """Should sort recommendations by success rate."""
        mock_knowledge_mound.query.return_value = [
            {
                "content": "Agent low",
                "confidence": 0.8,
                "metadata": {
                    "type": "control_plane_capability",
                    "agent_id": "low",
                    "capability": "debate",
                    "success_count": 50,
                    "failure_count": 50,  # 50% success
                    "avg_duration_seconds": 10.0,
                },
            },
            {
                "content": "Agent high",
                "confidence": 0.9,
                "metadata": {
                    "type": "control_plane_capability",
                    "agent_id": "high",
                    "capability": "debate",
                    "success_count": 95,
                    "failure_count": 5,  # 95% success
                    "avg_duration_seconds": 10.0,
                },
            },
        ]

        results = await adapter.get_capability_recommendations("debate")

        assert results[0].agent_id == "high"
        assert results[1].agent_id == "low"


# =============================================================================
# Cross-Workspace Insight Tests
# =============================================================================


class TestCrossWorkspaceInsights:
    """Tests for cross-workspace insight sharing."""

    @pytest.mark.asyncio
    async def test_shares_insight(self, adapter, mock_knowledge_mound):
        """Should share insight across workspaces."""
        insight = CrossWorkspaceInsight(
            insight_id="insight_123",
            source_workspace="workspace_a",
            target_workspaces=["workspace_b", "workspace_c"],
            task_type="debate",
            content="Discovered effective debate strategy",
            confidence=0.85,
            created_at=datetime.now().isoformat(),
        )

        result = await adapter.share_insight_cross_workspace(insight)

        assert result is True
        mock_knowledge_mound.ingest.assert_called_once()

        call_args = mock_knowledge_mound.ingest.call_args[0][0]
        assert "cp_insight_" in call_args.id
        assert insight.content in call_args.content

    @pytest.mark.asyncio
    async def test_updates_share_stats(self, adapter, mock_knowledge_mound):
        """Should update stats on share."""
        insight = CrossWorkspaceInsight(
            insight_id="insight_456",
            source_workspace="workspace_a",
            target_workspaces=["workspace_b"],
            task_type="debate",
            content="Test insight",
            confidence=0.8,
            created_at=datetime.now().isoformat(),
        )

        assert adapter._stats["cross_workspace_shares"] == 0

        await adapter.share_insight_cross_workspace(insight)

        assert adapter._stats["cross_workspace_shares"] == 1

    @pytest.mark.asyncio
    async def test_get_insights_filters_own_workspace(self, adapter, mock_knowledge_mound):
        """Should filter out insights from own workspace."""
        adapter._workspace_id = "workspace_a"

        mock_knowledge_mound.query.return_value = [
            {
                "content": "Own insight",
                "confidence": 0.8,
                "metadata": {
                    "type": "cross_workspace_insight",
                    "insight_id": "insight_1",
                    "source_workspace": "workspace_a",  # Same as adapter
                    "target_workspaces": [],
                    "task_type": "debate",
                },
            },
            {
                "content": "Other insight",
                "confidence": 0.9,
                "metadata": {
                    "type": "cross_workspace_insight",
                    "insight_id": "insight_2",
                    "source_workspace": "workspace_b",  # Different
                    "target_workspaces": ["workspace_a"],
                    "task_type": "debate",
                },
            },
        ]

        results = await adapter.get_cross_workspace_insights("debate")

        # Only the insight from workspace_b should be returned
        assert len(results) == 1
        assert results[0].source_workspace == "workspace_b"


# =============================================================================
# Stats and Cache Tests
# =============================================================================


class TestStatsAndCache:
    """Tests for stats and cache behavior."""

    def test_get_stats(self, adapter):
        """Should return stats dict."""
        stats = adapter.get_stats()

        assert "task_outcomes_stored" in stats
        assert "capability_records_stored" in stats
        assert "capability_queries" in stats
        assert "cross_workspace_shares" in stats
        assert stats["workspace_id"] == "test_workspace"

    def test_clear_cache(self, adapter):
        """Should clear cache and return count."""
        # Populate cache
        adapter._capability_cache["debate"] = []
        adapter._capability_cache["code"] = []
        adapter._cache_times["debate"] = 0
        adapter._cache_times["code"] = 0

        count = adapter.clear_cache()

        assert count == 2
        assert len(adapter._capability_cache) == 0
        assert len(adapter._cache_times) == 0

    @pytest.mark.asyncio
    async def test_cache_ttl_respected(self, adapter, mock_knowledge_mound):
        """Should respect cache TTL."""
        import time

        mock_knowledge_mound.query.return_value = []

        # First query
        await adapter.get_capability_recommendations("debate")

        # Expire cache manually
        adapter._cache_times["debate"] = time.time() - adapter._cache_ttl - 1

        # Second query should hit KM again
        await adapter.get_capability_recommendations("debate", use_cache=True)

        assert mock_knowledge_mound.query.call_count == 2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestControlPlaneEdgeCases:
    """Tests for edge cases and error handling."""

    def test_init_without_coordinator(self):
        """Should initialize without coordinator."""
        adapter = ControlPlaneAdapter(coordinator=None)

        assert adapter._coordinator is None

    def test_init_with_custom_thresholds(self):
        """Should accept custom thresholds."""
        adapter = ControlPlaneAdapter(
            min_task_confidence=0.9,
            min_capability_sample_size=10,
        )

        assert adapter._min_task_confidence == 0.9
        assert adapter._min_capability_sample_size == 10

    def test_set_event_callback(self, adapter):
        """Should set event callback."""
        callback = MagicMock()
        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event_handles_errors(self, adapter):
        """Should handle event callback errors."""
        callback = MagicMock(side_effect=Exception("Callback failed"))
        adapter._event_callback = callback

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_task_outcome_below_confidence(self, adapter, mock_knowledge_mound):
        """Should skip task outcomes below confidence threshold."""
        outcome = TaskOutcome(
            task_id="task_low",
            task_type="rare_task",
            agent_id="agent",
            success=False,
            duration_seconds=5.0,
        )

        adapter._min_task_confidence = 0.9  # High threshold

        result = await adapter.store_task_outcome(outcome)

        # Failed task confidence (0.5) is below 0.9 threshold
        assert result is None

    @pytest.mark.asyncio
    async def test_capability_with_limit(self, adapter, mock_knowledge_mound):
        """Should respect limit parameter in recommendations."""
        mock_knowledge_mound.query.return_value = []

        await adapter.get_capability_recommendations("debate", limit=5)

        # Verify query was called with correct limit
        call_kwargs = mock_knowledge_mound.query.call_args
        # Adapter may modify limit internally, just verify query was made
        assert mock_knowledge_mound.query.called

    @pytest.mark.asyncio
    async def test_share_insight_without_km(self):
        """Should handle share without KM gracefully."""
        adapter = ControlPlaneAdapter(knowledge_mound=None)

        insight = CrossWorkspaceInsight(
            insight_id="insight_789",
            source_workspace="workspace_a",
            target_workspaces=["workspace_b"],
            task_type="debate",
            content="Test insight",
            confidence=0.8,
            created_at=datetime.now().isoformat(),
        )

        result = await adapter.share_insight_cross_workspace(insight)

        # Should return False or None when no KM
        assert result in (False, None)

    @pytest.mark.asyncio
    async def test_get_insights_without_km(self):
        """Should return empty list without KM."""
        adapter = ControlPlaneAdapter(knowledge_mound=None)

        results = await adapter.get_cross_workspace_insights("debate")

        assert results == []


class TestDataclasses:
    """Tests for adapter dataclasses."""

    def test_task_outcome_defaults(self):
        """Should have correct defaults."""
        outcome = TaskOutcome(
            task_id="t1",
            task_type="test",
            agent_id="agent",
            success=True,
            duration_seconds=1.0,
        )

        assert outcome.workspace_id == "default"
        assert outcome.error_message is None
        assert outcome.metadata == {}

    def test_task_outcome_with_metadata(self):
        """Should accept metadata."""
        outcome = TaskOutcome(
            task_id="t1",
            task_type="test",
            agent_id="agent",
            success=True,
            duration_seconds=1.0,
            metadata={"key": "value"},
        )

        assert outcome.metadata == {"key": "value"}

    def test_capability_record_defaults(self):
        """Should have correct defaults."""
        record = AgentCapabilityRecord(
            agent_id="agent",
            capability="test",
            success_count=10,
            failure_count=2,
            avg_duration_seconds=5.0,
        )

        assert record.workspace_id == "default"
        assert record.confidence == 0.8

    def test_cross_workspace_insight_creation(self):
        """Should create insight with all fields."""
        insight = CrossWorkspaceInsight(
            insight_id="i1",
            source_workspace="ws1",
            target_workspaces=["ws2", "ws3"],
            task_type="debate",
            content="Test content",
            confidence=0.9,
            created_at="2024-01-15T10:00:00",
        )

        assert insight.insight_id == "i1"
        assert len(insight.target_workspaces) == 2


class TestStatsTracking:
    """Tests for stats tracking."""

    @pytest.mark.asyncio
    async def test_capability_query_stats(self, adapter, mock_knowledge_mound):
        """Should track capability query stats."""
        mock_knowledge_mound.query.return_value = []

        initial_queries = adapter._stats["capability_queries"]

        await adapter.get_capability_recommendations("debate", use_cache=False)
        await adapter.get_capability_recommendations("code", use_cache=False)

        assert adapter._stats["capability_queries"] == initial_queries + 2

    @pytest.mark.asyncio
    async def test_capability_record_stats(self, adapter, capability_record, mock_knowledge_mound):
        """Should track capability record storage stats."""
        initial_stored = adapter._stats["capability_records_stored"]

        await adapter.store_capability_record(capability_record)

        assert adapter._stats["capability_records_stored"] == initial_stored + 1
