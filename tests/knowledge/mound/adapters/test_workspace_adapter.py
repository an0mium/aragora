"""
Tests for WorkspaceAdapter - Bridges Workspace Manager to Knowledge Mound.

Tests cover:
- RigSnapshot, ConvoyOutcome, MergeOutcome dataclasses
- Forward sync: storing snapshots and outcomes to KM
- Reverse flow: querying history and recommendations from KM
- Caching behavior
- Statistics tracking
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.workspace_adapter import (
    WorkspaceAdapter,
    RigSnapshot,
    ConvoyOutcome,
    MergeOutcome,
)


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestRigSnapshot:
    """Tests for RigSnapshot dataclass."""

    def test_create_rig_snapshot(self):
        """Should create RigSnapshot with all fields."""
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Backend Rig",
            workspace_id="ws-456",
            status="active",
            repo_url="https://github.com/example/repo",
            branch="main",
            assigned_agents=3,
            max_agents=10,
            active_convoys=2,
            tasks_completed=50,
            tasks_failed=5,
        )

        assert snapshot.rig_id == "rig-123"
        assert snapshot.name == "Backend Rig"
        assert snapshot.workspace_id == "ws-456"
        assert snapshot.status == "active"
        assert snapshot.repo_url == "https://github.com/example/repo"
        assert snapshot.branch == "main"
        assert snapshot.assigned_agents == 3
        assert snapshot.max_agents == 10
        assert snapshot.active_convoys == 2
        assert snapshot.tasks_completed == 50
        assert snapshot.tasks_failed == 5

    def test_rig_snapshot_defaults(self):
        """Should use default values for optional fields."""
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test Rig",
            workspace_id="ws-456",
            status="idle",
        )

        assert snapshot.repo_url == ""
        assert snapshot.branch == "main"
        assert snapshot.assigned_agents == 0
        assert snapshot.max_agents == 10
        assert snapshot.active_convoys == 0
        assert snapshot.tasks_completed == 0
        assert snapshot.tasks_failed == 0
        assert snapshot.metadata == {}


class TestConvoyOutcome:
    """Tests for ConvoyOutcome dataclass."""

    def test_create_convoy_outcome(self):
        """Should create ConvoyOutcome with all fields."""
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Feature Implementation",
            status="completed",
            total_beads=10,
            completed_beads=10,
            assigned_agents=3,
            duration_seconds=3600.0,
            merge_success=True,
        )

        assert outcome.convoy_id == "conv-789"
        assert outcome.workspace_id == "ws-456"
        assert outcome.rig_id == "rig-123"
        assert outcome.name == "Feature Implementation"
        assert outcome.status == "completed"
        assert outcome.total_beads == 10
        assert outcome.completed_beads == 10
        assert outcome.assigned_agents == 3
        assert outcome.duration_seconds == 3600.0
        assert outcome.merge_success is True

    def test_convoy_outcome_defaults(self):
        """Should use default values for optional fields."""
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test Convoy",
            status="pending",
            total_beads=5,
        )

        assert outcome.completed_beads == 0
        assert outcome.assigned_agents == 0
        assert outcome.duration_seconds == 0.0
        assert outcome.merge_success is None
        assert outcome.error_message is None
        assert outcome.metadata == {}


class TestMergeOutcome:
    """Tests for MergeOutcome dataclass."""

    def test_create_merge_outcome(self):
        """Should create MergeOutcome with all fields."""
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
            conflicts_resolved=2,
            files_changed=15,
            tests_passed=True,
            review_approved=True,
            duration_seconds=120.0,
        )

        assert outcome.merge_id == "merge-001"
        assert outcome.convoy_id == "conv-789"
        assert outcome.rig_id == "rig-123"
        assert outcome.workspace_id == "ws-456"
        assert outcome.success is True
        assert outcome.conflicts_resolved == 2
        assert outcome.files_changed == 15
        assert outcome.tests_passed is True
        assert outcome.review_approved is True
        assert outcome.duration_seconds == 120.0

    def test_merge_outcome_failed(self):
        """Should handle failed merge outcome."""
        outcome = MergeOutcome(
            merge_id="merge-002",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=False,
            error_message="Merge conflict in src/main.py",
        )

        assert outcome.success is False
        assert outcome.error_message == "Merge conflict in src/main.py"
        assert outcome.tests_passed is False
        assert outcome.review_approved is False


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestWorkspaceAdapterInit:
    """Tests for WorkspaceAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = WorkspaceAdapter()

        assert adapter._workspace_manager is None
        assert adapter._knowledge_mound is None
        assert adapter._workspace_id == "default"
        assert adapter._min_confidence_threshold == 0.6
        assert adapter._enable_dual_write is False
        assert adapter._cache_ttl == 300

    def test_init_with_workspace_manager(self):
        """Should accept WorkspaceManager."""
        mock_ws = MagicMock()
        adapter = WorkspaceAdapter(workspace_manager=mock_ws)

        assert adapter._workspace_manager is mock_ws

    def test_init_with_knowledge_mound(self):
        """Should accept KnowledgeMound."""
        mock_km = MagicMock()
        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        assert adapter._knowledge_mound is mock_km

    def test_init_with_custom_workspace_id(self):
        """Should accept custom workspace ID."""
        adapter = WorkspaceAdapter(workspace_id="ws-custom")

        assert adapter._workspace_id == "ws-custom"

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = WorkspaceAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_init_with_dual_write(self):
        """Should enable dual write mode."""
        adapter = WorkspaceAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True


# =============================================================================
# Forward Sync Tests (Workspace → KM)
# =============================================================================


class TestStoreRigSnapshot:
    """Tests for storing rig snapshots."""

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_no_km(self):
        """Should return None if no KM configured."""
        adapter = WorkspaceAdapter()
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
        )

        result = await adapter.store_rig_snapshot(snapshot)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_success(self):
        """Should store snapshot and return item ID."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Backend Rig",
            workspace_id="ws-456",
            status="active",
            tasks_completed=10,
            tasks_failed=2,
        )

        result = await adapter.store_rig_snapshot(snapshot)

        assert result == "item-001"
        mock_km.ingest.assert_called_once()
        assert adapter._stats["rig_snapshots_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_below_threshold(self):
        """Should skip storage if below confidence threshold."""
        mock_km = AsyncMock()
        adapter = WorkspaceAdapter(
            knowledge_mound=mock_km,
            min_confidence_threshold=0.9,  # High threshold
        )
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=0,  # Low confidence
        )

        result = await adapter.store_rig_snapshot(snapshot)

        assert result is None
        mock_km.ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_invalidates_cache(self):
        """Should invalidate cache after storing."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        adapter._rig_performance_cache["rig-123"] = [MagicMock()]

        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=5,
        )

        await adapter.store_rig_snapshot(snapshot)

        assert "rig-123" not in adapter._rig_performance_cache


class TestStoreConvoyOutcome:
    """Tests for storing convoy outcomes."""

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_no_km(self):
        """Should return None if no KM configured."""
        adapter = WorkspaceAdapter()
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test",
            status="completed",
            total_beads=5,
        )

        result = await adapter.store_convoy_outcome(outcome)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_success(self):
        """Should store outcome and return item ID."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Feature Work",
            status="completed",
            total_beads=10,
            completed_beads=10,
        )

        result = await adapter.store_convoy_outcome(outcome)

        assert result == "item-002"
        mock_km.ingest.assert_called_once()
        assert adapter._stats["convoy_outcomes_stored"] == 1


class TestStoreMergeOutcome:
    """Tests for storing merge outcomes."""

    @pytest.mark.asyncio
    async def test_store_merge_outcome_success(self):
        """Should store merge outcome."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
            files_changed=10,
            tests_passed=True,
        )

        result = await adapter.store_merge_outcome(outcome)

        assert result == "item-003"
        assert adapter._stats["merge_outcomes_stored"] == 1


# =============================================================================
# Reverse Flow Tests (KM → Workspace)
# =============================================================================


class TestGetRigPerformanceHistory:
    """Tests for retrieving rig performance history."""

    @pytest.mark.asyncio
    async def test_get_rig_performance_no_km(self):
        """Should return empty list if no KM configured."""
        adapter = WorkspaceAdapter()

        result = await adapter.get_rig_performance_history("rig-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_rig_performance_from_cache(self):
        """Should return cached results if valid."""
        adapter = WorkspaceAdapter()
        cached_snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Cached",
            workspace_id="ws-456",
            status="active",
        )
        adapter._rig_performance_cache["rig-123"] = [cached_snapshot]
        adapter._cache_times["rig_rig-123"] = time.time()

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        assert result[0].name == "Cached"

    @pytest.mark.asyncio
    async def test_get_rig_performance_from_km(self):
        """Should query KM if cache miss."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": "Test Rig",
                        "workspace_id": "ws-456",
                        "status": "active",
                        "tasks_completed": 10,
                    }
                }
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        assert result[0].name == "Test Rig"
        assert adapter._stats["rig_queries"] == 1


class TestGetConvoyPatterns:
    """Tests for retrieving convoy patterns."""

    @pytest.mark.asyncio
    async def test_get_convoy_patterns_no_km(self):
        """Should return empty list if no KM configured."""
        adapter = WorkspaceAdapter()

        result = await adapter.get_convoy_patterns("rig-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_convoy_patterns_from_km(self):
        """Should query KM for convoy patterns."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_convoy_outcome",
                        "convoy_id": "conv-789",
                        "convoy_name": "Feature Work",
                        "workspace_id": "ws-456",
                        "rig_id": "rig-123",
                        "status": "completed",
                        "total_beads": 5,
                        "completed_beads": 5,
                    }
                }
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_convoy_patterns("rig-123")

        assert len(result) == 1
        assert result[0].name == "Feature Work"
        assert adapter._stats["convoy_pattern_queries"] == 1


class TestGetRigRecommendations:
    """Tests for rig recommendations."""

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_no_km(self):
        """Should return empty list if no KM configured."""
        adapter = WorkspaceAdapter()

        result = await adapter.get_rig_recommendations("backend")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_with_data(self):
        """Should return recommendations based on historical data."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-001",
                        "rig_name": "High Performer",
                        "tasks_completed": 80,
                        "tasks_failed": 5,
                        "active_convoys": 3,
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-002",
                        "rig_name": "Low Performer",
                        "tasks_completed": 20,
                        "tasks_failed": 30,
                        "active_convoys": 1,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_recommendations("backend", top_n=2)

        assert len(result) <= 2
        # High performer should rank higher
        if len(result) == 2:
            assert result[0]["rig_id"] == "rig-001"


class TestGetOptimalAgentCount:
    """Tests for optimal agent count recommendations."""

    @pytest.mark.asyncio
    async def test_get_optimal_agent_count_insufficient_data(self):
        """Should indicate when data is insufficient."""
        adapter = WorkspaceAdapter()

        result = await adapter.get_optimal_agent_count("rig-123", convoy_size=10)

        assert result["recommendation_available"] is False

    @pytest.mark.asyncio
    async def test_get_optimal_agent_count_with_patterns(self):
        """Should calculate optimal agent count from patterns."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_convoy_outcome",
                        "convoy_id": f"conv-{i}",
                        "convoy_name": f"Convoy {i}",
                        "workspace_id": "ws-456",
                        "rig_id": "rig-123",
                        "status": "completed",
                        "total_beads": 10,
                        "completed_beads": 10,
                        "assigned_agents": 5,
                        "duration_seconds": 1000.0,
                    }
                }
                for i in range(5)
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_optimal_agent_count("rig-123", convoy_size=20)

        assert result["recommendation_available"] is True
        assert result["recommended_agents"] > 0
        assert "optimal_ratio" in result


# =============================================================================
# Stats and Cache Tests
# =============================================================================


class TestStatsAndCache:
    """Tests for statistics and caching."""

    def test_get_stats(self):
        """Should return current statistics."""
        adapter = WorkspaceAdapter(workspace_id="ws-test")

        stats = adapter.get_stats()

        assert stats["workspace_id"] == "ws-test"
        assert stats["rig_snapshots_stored"] == 0
        assert stats["convoy_outcomes_stored"] == 0
        assert stats["merge_outcomes_stored"] == 0
        assert stats["rig_queries"] == 0
        assert stats["convoy_pattern_queries"] == 0
        assert stats["rig_cache_size"] == 0
        assert stats["convoy_cache_size"] == 0

    def test_clear_cache(self):
        """Should clear all caches."""
        adapter = WorkspaceAdapter()
        adapter._rig_performance_cache["rig-1"] = [MagicMock()]
        adapter._rig_performance_cache["rig-2"] = [MagicMock()]
        adapter._convoy_patterns_cache["rig-1"] = [MagicMock()]
        adapter._cache_times["rig_rig-1"] = time.time()

        count = adapter.clear_cache()

        assert count == 3  # 2 rig caches + 1 convoy cache
        assert len(adapter._rig_performance_cache) == 0
        assert len(adapter._convoy_patterns_cache) == 0
        assert len(adapter._cache_times) == 0


class TestEventCallback:
    """Tests for event callback functionality."""

    @pytest.mark.asyncio
    async def test_event_callback_on_rig_snapshot(self):
        """Should emit event when rig snapshot is stored."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        callback = MagicMock()
        adapter = WorkspaceAdapter(knowledge_mound=mock_km, event_callback=callback)

        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=10,
        )

        await adapter.store_rig_snapshot(snapshot)

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == "workspace_rig_snapshot_stored"


class TestSyncFromWorkspace:
    """Tests for syncing from WorkspaceManager."""

    @pytest.mark.asyncio
    async def test_sync_no_workspace_manager(self):
        """Should return error if no workspace manager."""
        adapter = WorkspaceAdapter()

        result = await adapter.sync_from_workspace()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_sync_with_workspace_manager(self):
        """Should sync rigs and convoys from workspace manager."""
        mock_ws = AsyncMock()

        # Mock rig
        mock_rig = MagicMock()
        mock_rig.rig_id = "rig-123"
        mock_rig.name = "Test Rig"
        mock_rig.workspace_id = "ws-456"
        mock_rig.status = MagicMock()
        mock_rig.status.value = "active"
        mock_rig.config = MagicMock()
        mock_rig.config.repo_url = "https://github.com/test"
        mock_rig.config.branch = "main"
        mock_rig.config.max_agents = 10
        mock_rig.assigned_agents = []
        mock_rig.active_convoys = []
        mock_rig.tasks_completed = 5
        mock_rig.tasks_failed = 1

        mock_ws.list_rigs = AsyncMock(return_value=[mock_rig])
        mock_ws.list_convoys = AsyncMock(return_value=[])

        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(
            workspace_manager=mock_ws,
            knowledge_mound=mock_km,
        )

        result = await adapter.sync_from_workspace()

        assert result["rigs"] == 1
        assert result["convoys"] == 0
