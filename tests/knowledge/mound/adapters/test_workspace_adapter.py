"""
Tests for WorkspaceAdapter - Bridges Workspace Manager to Knowledge Mound.

Tests cover:
- RigSnapshot, ConvoyOutcome, MergeOutcome dataclasses
- Forward sync: storing snapshots and outcomes to KM
- Reverse flow: querying history and recommendations from KM
- Caching behavior and expiration
- Statistics tracking
- Error handling and edge cases
- Access control and permissions
- Query and search within workspaces
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.knowledge.mound.adapters.workspace_adapter import (
    WorkspaceAdapter,
    RigSnapshot,
    ConvoyOutcome,
    MergeOutcome,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.ingest = AsyncMock(return_value="km_test_id")
    mound.query = AsyncMock(return_value=[])
    return mound


@pytest.fixture
def mock_workspace_manager():
    """Create a mock WorkspaceManager."""
    ws = AsyncMock()
    ws.list_rigs = AsyncMock(return_value=[])
    ws.list_convoys = AsyncMock(return_value=[])
    return ws


@pytest.fixture
def adapter(mock_knowledge_mound):
    """Create a WorkspaceAdapter with mock KM."""
    return WorkspaceAdapter(
        knowledge_mound=mock_knowledge_mound,
        workspace_id="test_workspace",
    )


@pytest.fixture
def sample_rig_snapshot():
    """Create a sample RigSnapshot."""
    return RigSnapshot(
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


@pytest.fixture
def sample_convoy_outcome():
    """Create a sample ConvoyOutcome."""
    return ConvoyOutcome(
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


@pytest.fixture
def sample_merge_outcome():
    """Create a sample MergeOutcome."""
    return MergeOutcome(
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

    def test_rig_snapshot_with_metadata(self):
        """Should accept custom metadata."""
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test Rig",
            workspace_id="ws-456",
            status="active",
            metadata={"custom_key": "custom_value", "priority": "high"},
        )

        assert snapshot.metadata == {"custom_key": "custom_value", "priority": "high"}

    def test_rig_snapshot_zero_tasks(self):
        """Should handle zero tasks completed and failed."""
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="New Rig",
            workspace_id="ws-456",
            status="initializing",
            tasks_completed=0,
            tasks_failed=0,
        )

        assert snapshot.tasks_completed == 0
        assert snapshot.tasks_failed == 0


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

    def test_convoy_outcome_with_error(self):
        """Should handle convoy with error message."""
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Failed Convoy",
            status="failed",
            total_beads=5,
            completed_beads=2,
            error_message="Build failed: syntax error in main.py",
        )

        assert outcome.status == "failed"
        assert outcome.error_message == "Build failed: syntax error in main.py"

    def test_convoy_outcome_partial_completion(self):
        """Should handle partially completed convoy."""
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Partial Convoy",
            status="cancelled",
            total_beads=10,
            completed_beads=6,
        )

        assert outcome.completed_beads == 6
        assert outcome.total_beads == 10


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

    def test_merge_outcome_defaults(self):
        """Should use default values for optional fields."""
        outcome = MergeOutcome(
            merge_id="merge-003",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
        )

        assert outcome.conflicts_resolved == 0
        assert outcome.files_changed == 0
        assert outcome.tests_passed is False
        assert outcome.review_approved is False
        assert outcome.duration_seconds == 0.0
        assert outcome.error_message is None


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

    def test_init_with_custom_confidence_threshold(self):
        """Should accept custom confidence threshold."""
        adapter = WorkspaceAdapter(min_confidence_threshold=0.8)

        assert adapter._min_confidence_threshold == 0.8

    def test_init_with_all_options(self):
        """Should accept all configuration options."""
        mock_ws = MagicMock()
        mock_km = MagicMock()
        callback = MagicMock()

        adapter = WorkspaceAdapter(
            workspace_manager=mock_ws,
            knowledge_mound=mock_km,
            workspace_id="ws-all",
            event_callback=callback,
            min_confidence_threshold=0.75,
            enable_dual_write=True,
        )

        assert adapter._workspace_manager is mock_ws
        assert adapter._knowledge_mound is mock_km
        assert adapter._workspace_id == "ws-all"
        assert adapter._event_callback is callback
        assert adapter._min_confidence_threshold == 0.75
        assert adapter._enable_dual_write is True

    def test_init_stats_structure(self):
        """Should initialize stats with correct structure."""
        adapter = WorkspaceAdapter()

        assert "rig_snapshots_stored" in adapter._stats
        assert "convoy_outcomes_stored" in adapter._stats
        assert "merge_outcomes_stored" in adapter._stats
        assert "rig_queries" in adapter._stats
        assert "convoy_pattern_queries" in adapter._stats
        assert adapter._stats["rig_snapshots_stored"] == 0

    def test_adapter_name(self):
        """Should have correct adapter name."""
        adapter = WorkspaceAdapter()

        assert adapter.adapter_name == "workspace"


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

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_emits_event(self):
        """Should emit event when snapshot stored."""
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
        assert callback.call_args[0][0] == "workspace_rig_snapshot_stored"

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_calculates_success_rate(self):
        """Should calculate correct success rate."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=80,
            tasks_failed=20,
        )

        await adapter.store_rig_snapshot(snapshot)

        call_args = mock_km.ingest.call_args[0][0]
        assert call_args.metadata["success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_handles_exception(self):
        """Should handle ingest exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(side_effect=Exception("Ingest failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=10,
        )

        result = await adapter.store_rig_snapshot(snapshot)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_rig_snapshot_high_confidence(self):
        """Should use HIGH confidence when tasks completed > 0."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Test",
            workspace_id="ws-456",
            status="active",
            tasks_completed=10,
        )

        await adapter.store_rig_snapshot(snapshot)

        # Verify confidence is HIGH (value 0.8+)
        call_args = mock_km.ingest.call_args[0][0]
        from aragora.knowledge.unified.types import ConfidenceLevel

        assert call_args.confidence == ConfidenceLevel.HIGH


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

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_invalidates_cache(self):
        """Should invalidate convoy cache after storing."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        adapter._convoy_patterns_cache["rig-123"] = [MagicMock()]

        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test",
            status="completed",
            total_beads=5,
        )

        await adapter.store_convoy_outcome(outcome)

        assert "rig-123" not in adapter._convoy_patterns_cache

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_emits_event(self):
        """Should emit event when outcome stored."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")
        callback = MagicMock()

        adapter = WorkspaceAdapter(knowledge_mound=mock_km, event_callback=callback)
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test",
            status="completed",
            total_beads=5,
        )

        await adapter.store_convoy_outcome(outcome)

        callback.assert_called_once()
        assert callback.call_args[0][0] == "workspace_convoy_outcome_stored"

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_calculates_completion_pct(self):
        """Should calculate correct completion percentage."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test",
            status="done",
            total_beads=10,
            completed_beads=8,
        )

        await adapter.store_convoy_outcome(outcome)

        call_args = mock_km.ingest.call_args[0][0]
        assert call_args.metadata["completion_pct"] == 80.0

    @pytest.mark.asyncio
    async def test_store_convoy_outcome_handles_exception(self):
        """Should handle ingest exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(side_effect=Exception("Ingest failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
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
    async def test_store_convoy_outcome_zero_beads(self):
        """Should handle convoy with zero beads."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Empty Convoy",
            status="completed",
            total_beads=0,
        )

        await adapter.store_convoy_outcome(outcome)

        call_args = mock_km.ingest.call_args[0][0]
        assert call_args.metadata["completion_pct"] == 0.0


class TestStoreMergeOutcome:
    """Tests for storing merge outcomes."""

    @pytest.mark.asyncio
    async def test_store_merge_outcome_no_km(self):
        """Should return None if no KM configured."""
        adapter = WorkspaceAdapter()
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
        )

        result = await adapter.store_merge_outcome(outcome)

        assert result is None

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

    @pytest.mark.asyncio
    async def test_store_merge_outcome_emits_event(self):
        """Should emit event when merge outcome stored."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")
        callback = MagicMock()

        adapter = WorkspaceAdapter(knowledge_mound=mock_km, event_callback=callback)
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
        )

        await adapter.store_merge_outcome(outcome)

        callback.assert_called_once()
        assert callback.call_args[0][0] == "workspace_merge_outcome_stored"

    @pytest.mark.asyncio
    async def test_store_merge_outcome_failed(self):
        """Should store failed merge outcome."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=False,
            error_message="Merge conflict",
        )

        result = await adapter.store_merge_outcome(outcome)

        assert result == "item-003"

    @pytest.mark.asyncio
    async def test_store_merge_outcome_handles_exception(self):
        """Should handle ingest exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(side_effect=Exception("Ingest failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
        )

        result = await adapter.store_merge_outcome(outcome)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_merge_outcome_content_format(self):
        """Should format content correctly for successful merge."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
            files_changed=5,
            tests_passed=True,
            review_approved=True,
        )

        await adapter.store_merge_outcome(outcome)

        call_args = mock_km.ingest.call_args[0][0]
        assert "success" in call_args.content
        assert "5 files" in call_args.content
        assert "tests passed" in call_args.content


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

    @pytest.mark.asyncio
    async def test_get_rig_performance_cache_expiry(self):
        """Should refresh cache when expired."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": "Fresh",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                }
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        adapter._cache_ttl = 0.1  # Short TTL

        cached_snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Stale",
            workspace_id="ws-456",
            status="active",
        )
        adapter._rig_performance_cache["rig-123"] = [cached_snapshot]
        adapter._cache_times["rig_rig-123"] = time.time() - 1  # Expired

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        assert result[0].name == "Fresh"

    @pytest.mark.asyncio
    async def test_get_rig_performance_respects_limit(self):
        """Should respect limit parameter."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": f"Rig {i}",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                }
                for i in range(10)
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123", limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_rig_performance_skip_cache(self):
        """Should skip cache when use_cache=False."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(return_value=[])

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        cached_snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Cached",
            workspace_id="ws-456",
            status="active",
        )
        adapter._rig_performance_cache["rig-123"] = [cached_snapshot]
        adapter._cache_times["rig_rig-123"] = time.time()

        result = await adapter.get_rig_performance_history("rig-123", use_cache=False)

        assert result == []
        mock_km.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_rig_performance_filters_type(self):
        """Should filter by workspace_rig_snapshot type."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_convoy_outcome",  # Wrong type
                        "rig_id": "rig-123",
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": "Correct",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        assert result[0].name == "Correct"

    @pytest.mark.asyncio
    async def test_get_rig_performance_filters_rig_id(self):
        """Should filter by rig_id."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "other-rig",  # Wrong rig
                        "rig_name": "Wrong",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": "Correct",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        assert result[0].name == "Correct"

    @pytest.mark.asyncio
    async def test_get_rig_performance_handles_exception(self):
        """Should handle query exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(side_effect=Exception("Query failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert result == []


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

    @pytest.mark.asyncio
    async def test_get_convoy_patterns_from_cache(self):
        """Should return cached results if valid."""
        adapter = WorkspaceAdapter()
        cached_outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Cached Convoy",
            status="completed",
            total_beads=5,
        )
        adapter._convoy_patterns_cache["rig-123"] = [cached_outcome]
        adapter._cache_times["convoy_rig-123"] = time.time()

        result = await adapter.get_convoy_patterns("rig-123")

        assert len(result) == 1
        assert result[0].name == "Cached Convoy"

    @pytest.mark.asyncio
    async def test_get_convoy_patterns_skip_empty_convoy_id(self):
        """Should skip results with empty convoy_id."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_convoy_outcome",
                        "convoy_id": "",  # Empty
                        "rig_id": "rig-123",
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_convoy_outcome",
                        "convoy_id": "conv-789",
                        "convoy_name": "Valid",
                        "workspace_id": "ws-456",
                        "rig_id": "rig-123",
                        "status": "completed",
                        "total_beads": 5,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_convoy_patterns("rig-123")

        assert len(result) == 1
        assert result[0].name == "Valid"

    @pytest.mark.asyncio
    async def test_get_convoy_patterns_handles_exception(self):
        """Should handle query exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(side_effect=Exception("Query failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_convoy_patterns("rig-123")

        assert result == []


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

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_filters_available_rigs(self):
        """Should filter by available_rigs."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-001",
                        "rig_name": "Available",
                        "tasks_completed": 80,
                        "tasks_failed": 5,
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-002",
                        "rig_name": "Not Available",
                        "tasks_completed": 90,
                        "tasks_failed": 2,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_recommendations(
            "backend",
            available_rigs=["rig-001"],
        )

        assert len(result) == 1
        assert result[0]["rig_id"] == "rig-001"

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_handles_zero_tasks(self):
        """Should skip rigs with zero total tasks."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-001",
                        "rig_name": "Empty",
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_recommendations("backend")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_calculates_scores(self):
        """Should calculate combined scores correctly."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-001",
                        "rig_name": "Balanced",
                        "tasks_completed": 50,
                        "tasks_failed": 50,  # 50% success
                        "active_convoys": 5,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_recommendations("backend")

        assert len(result) == 1
        assert result[0]["success_rate"] == 0.5
        assert "combined_score" in result[0]
        assert "confidence" in result[0]

    @pytest.mark.asyncio
    async def test_get_rig_recommendations_handles_exception(self):
        """Should handle query exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(side_effect=Exception("Query failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_recommendations("backend")

        assert result == []


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

    @pytest.mark.asyncio
    async def test_get_optimal_agent_count_no_successful_convoys(self):
        """Should handle case with no successful convoys."""
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
                        "status": "failed",  # Not successful
                        "total_beads": 10,
                    }
                }
                for i in range(5)
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_optimal_agent_count("rig-123", convoy_size=20)

        assert result["recommendation_available"] is False
        assert "No successful convoys" in result["reason"]

    @pytest.mark.asyncio
    async def test_get_optimal_agent_count_no_agent_data(self):
        """Should handle convoys without agent/bead counts."""
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
                        "total_beads": 0,  # No beads
                        "assigned_agents": 0,  # No agents
                    }
                }
                for i in range(5)
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_optimal_agent_count("rig-123", convoy_size=20)

        assert result["recommendation_available"] is False

    @pytest.mark.asyncio
    async def test_get_optimal_agent_count_minimum_one_agent(self):
        """Should recommend at least one agent."""
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
                        "status": "done",
                        "total_beads": 100,  # Large convoy
                        "completed_beads": 100,
                        "assigned_agents": 1,  # Small ratio
                        "duration_seconds": 100.0,
                    }
                }
                for i in range(5)
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_optimal_agent_count("rig-123", convoy_size=1)

        assert result["recommended_agents"] >= 1


class TestGetMergeSuccessFactors:
    """Tests for merge success factor analysis."""

    @pytest.mark.asyncio
    async def test_get_merge_success_factors_no_km(self):
        """Should return unavailable if no KM."""
        adapter = WorkspaceAdapter()

        result = await adapter.get_merge_success_factors("rig-123")

        assert result["analysis_available"] is False

    @pytest.mark.asyncio
    async def test_get_merge_success_factors_no_data(self):
        """Should handle no merge data."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(return_value=[])

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_merge_success_factors("rig-123")

        assert result["analysis_available"] is False
        assert result["reason"] == "No merge data"

    @pytest.mark.asyncio
    async def test_get_merge_success_factors_with_data(self):
        """Should analyze success factors."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_merge_outcome",
                        "rig_id": "rig-123",
                        "success": True,
                        "tests_passed": True,
                        "review_approved": True,
                        "files_changed": 10,
                        "conflicts_resolved": 1,
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_merge_outcome",
                        "rig_id": "rig-123",
                        "success": False,
                        "tests_passed": False,
                        "review_approved": False,
                        "files_changed": 5,
                        "conflicts_resolved": 0,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_merge_success_factors("rig-123")

        assert result["analysis_available"] is True
        assert result["success_rate"] == 0.5
        assert result["successful_count"] == 1
        assert result["failed_count"] == 1
        assert "factors" in result

    @pytest.mark.asyncio
    async def test_get_merge_success_factors_handles_exception(self):
        """Should handle query exceptions gracefully."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(side_effect=Exception("Query failed"))

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_merge_success_factors("rig-123")

        assert result["analysis_available"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_merge_success_factors_filters_rig_id(self):
        """Should filter by rig_id."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_merge_outcome",
                        "rig_id": "other-rig",  # Different rig
                        "success": True,
                    }
                },
                {
                    "metadata": {
                        "type": "workspace_merge_outcome",
                        "rig_id": "rig-123",
                        "success": True,
                        "tests_passed": True,
                        "review_approved": True,
                        "files_changed": 10,
                        "conflicts_resolved": 0,
                    }
                },
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_merge_success_factors("rig-123")

        assert result["successful_count"] == 1
        assert result["failed_count"] == 0


# =============================================================================
# Sync Tests
# =============================================================================


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

    @pytest.mark.asyncio
    async def test_sync_with_convoys(self):
        """Should sync convoys from workspace manager."""
        mock_ws = AsyncMock()

        # Mock convoy
        mock_convoy = MagicMock()
        mock_convoy.convoy_id = "conv-789"
        mock_convoy.workspace_id = "ws-456"
        mock_convoy.rig_id = "rig-123"
        mock_convoy.name = "Test Convoy"
        mock_convoy.status = MagicMock()
        mock_convoy.status.value = "completed"
        mock_convoy.total_beads = 5
        mock_convoy.bead_ids = ["b1", "b2", "b3"]
        mock_convoy.assigned_agents = ["agent-1"]
        mock_convoy.created_at = time.time() - 100
        mock_convoy.started_at = time.time() - 50
        mock_convoy.completed_at = time.time()
        mock_convoy.error = None

        mock_ws.list_rigs = AsyncMock(return_value=[])
        mock_ws.list_convoys = AsyncMock(return_value=[mock_convoy])

        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(
            workspace_manager=mock_ws,
            knowledge_mound=mock_km,
        )

        result = await adapter.sync_from_workspace()

        assert result["convoys"] == 1

    @pytest.mark.asyncio
    async def test_sync_handles_exception(self):
        """Should handle sync exceptions gracefully."""
        mock_ws = AsyncMock()
        mock_ws.list_rigs = AsyncMock(side_effect=Exception("List failed"))

        adapter = WorkspaceAdapter(workspace_manager=mock_ws)

        result = await adapter.sync_from_workspace()

        assert "error" in result


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

    def test_get_stats_with_data(self):
        """Should return accurate statistics with data."""
        adapter = WorkspaceAdapter(workspace_id="ws-test")
        adapter._stats["rig_snapshots_stored"] = 5
        adapter._stats["convoy_outcomes_stored"] = 10
        adapter._rig_performance_cache["rig-1"] = []
        adapter._rig_performance_cache["rig-2"] = []

        stats = adapter.get_stats()

        assert stats["rig_snapshots_stored"] == 5
        assert stats["convoy_outcomes_stored"] == 10
        assert stats["rig_cache_size"] == 2

    def test_get_stats_includes_km_status(self):
        """Should include KM and WS manager status."""
        adapter = WorkspaceAdapter()

        stats = adapter.get_stats()

        assert stats["has_knowledge_mound"] is False
        assert stats["has_workspace_manager"] is False

        mock_km = MagicMock()
        adapter._knowledge_mound = mock_km

        stats = adapter.get_stats()

        assert stats["has_knowledge_mound"] is True

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

    def test_clear_empty_cache(self):
        """Should handle clearing empty cache."""
        adapter = WorkspaceAdapter()

        count = adapter.clear_cache()

        assert count == 0


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

    @pytest.mark.asyncio
    async def test_event_callback_on_convoy_outcome(self):
        """Should emit event when convoy outcome is stored."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        callback = MagicMock()
        adapter = WorkspaceAdapter(knowledge_mound=mock_km, event_callback=callback)

        outcome = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Test",
            status="completed",
            total_beads=5,
        )

        await adapter.store_convoy_outcome(outcome)

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == "workspace_convoy_outcome_stored"

    @pytest.mark.asyncio
    async def test_event_callback_on_merge_outcome(self):
        """Should emit event when merge outcome is stored."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")

        callback = MagicMock()
        adapter = WorkspaceAdapter(knowledge_mound=mock_km, event_callback=callback)

        outcome = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
        )

        await adapter.store_merge_outcome(outcome)

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == "workspace_merge_outcome_stored"

    def test_set_event_callback(self):
        """Should allow setting event callback."""
        adapter = WorkspaceAdapter()
        callback = MagicMock()

        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_adapter_inherits_base_class(self):
        """Should inherit from KnowledgeMoundAdapter."""
        from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter

        adapter = WorkspaceAdapter()

        assert isinstance(adapter, KnowledgeMoundAdapter)

    @pytest.mark.asyncio
    async def test_store_snapshot_with_zero_success_rate(self):
        """Should handle 100% failure rate."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)
        snapshot = RigSnapshot(
            rig_id="rig-123",
            name="Failed Rig",
            workspace_id="ws-456",
            status="error",
            tasks_completed=0,
            tasks_failed=10,
        )

        result = await adapter.store_rig_snapshot(snapshot)

        # May be None due to confidence threshold
        # The important thing is it doesn't crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_query_results(self):
        """Should handle empty query results."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(return_value=[])

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_query_results(self):
        """Should handle malformed query results."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {},  # No metadata
                {"metadata": {}},  # Empty metadata
                {"metadata": {"type": "wrong_type"}},  # Wrong type
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert result == []

    def test_health_check(self):
        """Should return health status."""
        adapter = WorkspaceAdapter()

        health = adapter.health_check()

        assert "adapter" in health
        assert health["adapter"] == "workspace"
        assert "healthy" in health

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Should handle concurrent cache access."""
        import asyncio

        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        "rig_name": "Test",
                        "workspace_id": "ws-456",
                        "status": "active",
                    }
                }
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        # Concurrent access
        tasks = [adapter.get_rig_performance_history("rig-123") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for result in results:
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_missing_metadata_fields(self):
        """Should handle missing metadata fields gracefully."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-123",
                        # Missing many fields
                    }
                }
            ]
        )

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        result = await adapter.get_rig_performance_history("rig-123")

        assert len(result) == 1
        # Should use defaults
        assert result[0].branch == "main"
        assert result[0].max_agents == 10


# =============================================================================
# Access Control Tests
# =============================================================================


class TestAccessControl:
    """Tests for access control and permissions."""

    @pytest.mark.asyncio
    async def test_workspace_isolation(self):
        """Should query within workspace context."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(return_value=[])

        adapter = WorkspaceAdapter(
            knowledge_mound=mock_km,
            workspace_id="isolated-ws",
        )

        await adapter.get_rig_performance_history("rig-123")

        # Verify workspace_id is passed
        call_kwargs = mock_km.query.call_args[1]
        assert call_kwargs["workspace_id"] == "isolated-ws"

    @pytest.mark.asyncio
    async def test_different_workspace_ids(self):
        """Should use correct workspace ID for queries."""
        mock_km = AsyncMock()
        mock_km.query = AsyncMock(return_value=[])

        adapter1 = WorkspaceAdapter(knowledge_mound=mock_km, workspace_id="ws-1")
        adapter2 = WorkspaceAdapter(knowledge_mound=mock_km, workspace_id="ws-2")

        await adapter1.get_rig_recommendations("backend")
        await adapter2.get_rig_recommendations("backend")

        calls = mock_km.query.call_args_list
        assert calls[0][1]["workspace_id"] == "ws-1"
        assert calls[1][1]["workspace_id"] == "ws-2"


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestIntegrationScenarios:
    """Tests for integration-like scenarios."""

    @pytest.mark.asyncio
    async def test_full_rig_lifecycle(self):
        """Should handle full rig lifecycle."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")
        mock_km.query = AsyncMock(return_value=[])

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        # Store initial snapshot
        snapshot1 = RigSnapshot(
            rig_id="rig-123",
            name="New Rig",
            workspace_id="ws-456",
            status="initializing",
            tasks_completed=0,
            tasks_failed=0,
        )
        await adapter.store_rig_snapshot(snapshot1)

        # Store active snapshot
        snapshot2 = RigSnapshot(
            rig_id="rig-123",
            name="Active Rig",
            workspace_id="ws-456",
            status="active",
            tasks_completed=50,
            tasks_failed=5,
        )
        await adapter.store_rig_snapshot(snapshot2)

        # Verify stats
        stats = adapter.get_stats()
        assert stats["rig_snapshots_stored"] == 2

    @pytest.mark.asyncio
    async def test_convoy_to_merge_workflow(self):
        """Should handle convoy to merge workflow."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        # Store convoy outcome
        convoy = ConvoyOutcome(
            convoy_id="conv-789",
            workspace_id="ws-456",
            rig_id="rig-123",
            name="Feature Work",
            status="completed",
            total_beads=10,
            completed_beads=10,
        )
        await adapter.store_convoy_outcome(convoy)

        # Store merge outcome
        merge = MergeOutcome(
            merge_id="merge-001",
            convoy_id="conv-789",
            rig_id="rig-123",
            workspace_id="ws-456",
            success=True,
            tests_passed=True,
        )
        await adapter.store_merge_outcome(merge)

        stats = adapter.get_stats()
        assert stats["convoy_outcomes_stored"] == 1
        assert stats["merge_outcomes_stored"] == 1

    @pytest.mark.asyncio
    async def test_recommendation_based_on_history(self):
        """Should provide recommendations based on stored history."""
        mock_km = AsyncMock()

        # First store some data
        mock_km.ingest = AsyncMock(return_value="item-001")
        adapter = WorkspaceAdapter(knowledge_mound=mock_km)

        # Then query for recommendations
        mock_km.query = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "type": "workspace_rig_snapshot",
                        "rig_id": "rig-best",
                        "rig_name": "Best Rig",
                        "tasks_completed": 100,
                        "tasks_failed": 5,
                    }
                },
            ]
        )

        recommendations = await adapter.get_rig_recommendations("backend")

        assert len(recommendations) == 1
        assert recommendations[0]["rig_id"] == "rig-best"
        assert recommendations[0]["success_rate"] > 0.9
