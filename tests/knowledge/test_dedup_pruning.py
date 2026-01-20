"""Tests for Knowledge Mound Phase 3: Deduplication and Pruning."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.dedup import (
    DedupOperationsMixin,
    DuplicateCluster,
    DuplicateMatch,
    DedupReport,
    MergeResult,
)
from aragora.knowledge.mound.ops.pruning import (
    PruningOperationsMixin,
    PruningPolicy,
    PrunableItem,
    PruneResult,
    PruneHistory,
    PruningAction,
)


class TestPruningPolicy:
    """Tests for PruningPolicy dataclass."""

    def test_default_policy(self):
        """Should have sensible defaults."""
        policy = PruningPolicy(
            policy_id="test-1",
            workspace_id="ws-123",
            name="Test Policy",
        )
        assert policy.enabled is True
        assert policy.staleness_threshold == 0.9
        assert policy.min_age_days == 30
        assert policy.action == PruningAction.ARCHIVE
        assert "glacial" in policy.tier_exceptions

    def test_custom_policy(self):
        """Should accept custom settings."""
        policy = PruningPolicy(
            policy_id="custom-1",
            workspace_id="ws-123",
            name="Aggressive Pruning",
            staleness_threshold=0.7,
            min_age_days=7,
            action=PruningAction.DELETE,
            tier_exceptions=["glacial", "slow"],
            auto_prune=True,
        )
        assert policy.staleness_threshold == 0.7
        assert policy.min_age_days == 7
        assert policy.action == PruningAction.DELETE
        assert policy.auto_prune is True


class TestPrunableItem:
    """Tests for PrunableItem dataclass."""

    def test_item_structure(self):
        """Should hold all pruning metadata."""
        item = PrunableItem(
            node_id="node-123",
            content_preview="Test content...",
            staleness_score=0.95,
            confidence=0.3,
            retrieval_count=0,
            last_retrieved_at=None,
            tier="medium",
            created_at=datetime.now() - timedelta(days=60),
            prune_reason="staleness=0.95, low_confidence=0.30",
            recommended_action=PruningAction.ARCHIVE,
        )
        assert item.staleness_score == 0.95
        assert item.recommended_action == PruningAction.ARCHIVE


class TestPruneResult:
    """Tests for PruneResult dataclass."""

    def test_result_counts(self):
        """Should track pruning action counts."""
        result = PruneResult(
            workspace_id="ws-123",
            executed_at=datetime.now(),
            policy_id="policy-1",
            items_analyzed=100,
            items_pruned=25,
            items_archived=20,
            items_deleted=0,
            items_demoted=5,
            items_flagged=0,
            pruned_item_ids=["id1", "id2"],
        )
        assert result.items_pruned == 25
        assert result.items_archived == 20
        assert result.items_demoted == 5
        assert len(result.pruned_item_ids) == 2


class TestPruneHistory:
    """Tests for PruneHistory dataclass."""

    def test_history_entry(self):
        """Should record pruning history."""
        history = PruneHistory(
            history_id="prune_12345",
            workspace_id="ws-123",
            executed_at=datetime.now(),
            policy_id="policy-1",
            action=PruningAction.ARCHIVE,
            items_pruned=10,
            pruned_item_ids=["id1", "id2"],
            reason="auto_prune",
            executed_by="system",
        )
        assert history.action == PruningAction.ARCHIVE
        assert history.items_pruned == 10


class TestDuplicateMatch:
    """Tests for DuplicateMatch dataclass."""

    def test_match_structure(self):
        """Should hold duplicate match data."""
        match = DuplicateMatch(
            node_id="dup-1",
            similarity=0.92,
            content_preview="Similar content...",
            created_at=datetime.now(),
            tier="medium",
            confidence=0.8,
        )
        assert match.similarity == 0.92
        assert match.tier == "medium"


class TestDuplicateCluster:
    """Tests for DuplicateCluster dataclass."""

    def test_cluster_recommendations(self):
        """Should recommend actions based on similarity."""
        high_sim_cluster = DuplicateCluster(
            cluster_id="cluster-1",
            primary_node_id="primary-1",
            duplicates=[
                DuplicateMatch(
                    node_id="dup-1",
                    similarity=0.97,
                    content_preview="...",
                    created_at=datetime.now(),
                    tier="medium",
                    confidence=0.8,
                )
            ],
            avg_similarity=0.97,
            recommended_action="merge",
        )
        assert high_sim_cluster.recommended_action == "merge"

        low_sim_cluster = DuplicateCluster(
            cluster_id="cluster-2",
            primary_node_id="primary-2",
            duplicates=[],
            avg_similarity=0.82,
            recommended_action="keep_separate",
        )
        assert low_sim_cluster.recommended_action == "keep_separate"


class TestDedupReport:
    """Tests for DedupReport dataclass."""

    def test_report_structure(self):
        """Should summarize dedup analysis."""
        report = DedupReport(
            workspace_id="ws-123",
            generated_at=datetime.now(),
            total_nodes_analyzed=1000,
            duplicate_clusters_found=50,
            clusters=[],
            estimated_reduction_percent=5.0,
        )
        assert report.total_nodes_analyzed == 1000
        assert report.duplicate_clusters_found == 50
        assert report.estimated_reduction_percent == 5.0


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_merge_result(self):
        """Should track merge operations."""
        result = MergeResult(
            kept_node_id="primary-1",
            merged_node_ids=["dup-1", "dup-2", "dup-3"],
            archived_count=3,
            updated_relationships=5,
        )
        assert result.kept_node_id == "primary-1"
        assert len(result.merged_node_ids) == 3
        assert result.archived_count == 3


class TestPruningAction:
    """Tests for PruningAction enum."""

    def test_action_values(self):
        """Should have expected action values."""
        assert PruningAction.ARCHIVE.value == "archive"
        assert PruningAction.DELETE.value == "delete"
        assert PruningAction.DEMOTE.value == "demote"
        assert PruningAction.FLAG.value == "flag"

    def test_action_comparison(self):
        """Should support string comparison."""
        assert PruningAction.ARCHIVE == "archive"
        assert PruningAction.DELETE == "delete"


class TestPruningOperationsMixin:
    """Tests for PruningOperationsMixin methods."""

    @pytest.fixture
    def mock_mound(self):
        """Create mock Knowledge Mound with pruning mixin."""

        class MockMound(PruningOperationsMixin):
            def __init__(self):
                self._store = MagicMock()
                self._store.get_prune_history = AsyncMock(return_value=[])
                self._store.save_prune_history = AsyncMock()
                self._store.archive_node = AsyncMock()
                self._store.delete_node = AsyncMock()
                self._store.update_node = AsyncMock()
                self._store.get_node = AsyncMock(
                    return_value=MagicMock(tier="medium", confidence=0.5)
                )
                self._store.restore_archived_node = AsyncMock(return_value=True)
                self._store.get_nodes_for_workspace = AsyncMock(return_value=[])

                self._staleness_detector = MagicMock()
                self._staleness_detector.get_stale_nodes = AsyncMock(return_value=[])

        return MockMound()

    @pytest.mark.asyncio
    async def test_get_prunable_items_empty(self, mock_mound):
        """Should return empty list when no stale items."""
        items = await mock_mound.get_prunable_items("ws-123")
        assert items == []

    @pytest.mark.asyncio
    async def test_prune_items_archive(self, mock_mound):
        """Should archive items."""
        result = await mock_mound.prune_items(
            workspace_id="ws-123",
            item_ids=["id1", "id2"],
            action=PruningAction.ARCHIVE,
            reason="test",
        )
        assert result.items_archived == 2
        assert result.items_pruned == 2
        assert mock_mound._store.archive_node.call_count == 2

    @pytest.mark.asyncio
    async def test_prune_items_delete(self, mock_mound):
        """Should delete items."""
        result = await mock_mound.prune_items(
            workspace_id="ws-123",
            item_ids=["id1"],
            action=PruningAction.DELETE,
            reason="test",
        )
        assert result.items_deleted == 1
        mock_mound._store.delete_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_prune_items_demote(self, mock_mound):
        """Should demote items to lower tier."""
        result = await mock_mound.prune_items(
            workspace_id="ws-123",
            item_ids=["id1"],
            action=PruningAction.DEMOTE,
            reason="test",
        )
        assert result.items_demoted == 1
        mock_mound._store.update_node.assert_called()

    @pytest.mark.asyncio
    async def test_prune_items_flag(self, mock_mound):
        """Should flag items for review."""
        result = await mock_mound.prune_items(
            workspace_id="ws-123",
            item_ids=["id1"],
            action=PruningAction.FLAG,
            reason="test",
        )
        assert result.items_flagged == 1

    @pytest.mark.asyncio
    async def test_auto_prune_disabled_policy(self, mock_mound):
        """Should not prune when policy is disabled."""
        policy = PruningPolicy(
            policy_id="p1",
            workspace_id="ws-123",
            name="Test",
            enabled=False,
        )
        result = await mock_mound.auto_prune("ws-123", policy, dry_run=False)
        assert result.items_pruned == 0
        assert "disabled" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_get_prune_history(self, mock_mound):
        """Should return pruning history."""
        history = await mock_mound.get_prune_history("ws-123")
        assert history == []
        mock_mound._store.get_prune_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_pruned_item(self, mock_mound):
        """Should restore archived item."""
        result = await mock_mound.restore_pruned_item("ws-123", "node-1")
        assert result is True
        mock_mound._store.restore_archived_node.assert_called_once()


class TestDedupOperationsMixin:
    """Tests for DedupOperationsMixin methods."""

    @pytest.fixture
    def mock_mound(self):
        """Create mock Knowledge Mound with dedup mixin."""

        class MockMound(DedupOperationsMixin):
            def __init__(self):
                self._store = MagicMock()
                self._store.get_nodes_for_workspace = AsyncMock(return_value=[])
                self._store.search_similar = AsyncMock(return_value=[])
                self._store.count_nodes = AsyncMock(return_value=0)
                self._store.get_node_relationships = AsyncMock(return_value=[])
                self._store.create_relationship = AsyncMock()
                self._store.archive_node = AsyncMock()
                self._store.delete_node = AsyncMock()
                self._store.get_nodes_by_content_hash = AsyncMock(return_value={})
                self._store.get_node = AsyncMock()

        return MockMound()

    @pytest.mark.asyncio
    async def test_find_duplicates_empty(self, mock_mound):
        """Should return empty list when no nodes."""
        clusters = await mock_mound.find_duplicates("ws-123")
        assert clusters == []

    @pytest.mark.asyncio
    async def test_generate_dedup_report(self, mock_mound):
        """Should generate dedup report."""
        report = await mock_mound.generate_dedup_report("ws-123")
        assert report.workspace_id == "ws-123"
        assert report.total_nodes_analyzed == 0
        assert report.duplicate_clusters_found == 0

    @pytest.mark.asyncio
    async def test_auto_merge_exact_duplicates_dry_run(self, mock_mound):
        """Should report duplicates in dry run mode."""
        mock_mound._store.get_nodes_by_content_hash = AsyncMock(
            return_value={
                "hash1": ["id1", "id2", "id3"],  # 2 duplicates
                "hash2": ["id4"],  # No duplicates
            }
        )
        result = await mock_mound.auto_merge_exact_duplicates("ws-123", dry_run=True)
        assert result["duplicates_found"] == 2
        assert result["merges_performed"] == 0
        assert result["dry_run"] is True


class TestPruningIntegration:
    """Integration tests for pruning workflows."""

    def test_policy_with_confidence_decay(self):
        """Should support confidence decay configuration."""
        policy = PruningPolicy(
            policy_id="decay-1",
            workspace_id="ws-123",
            name="Confidence Decay",
            confidence_decay_rate=0.01,  # 1% per day
            min_confidence=0.1,
        )
        assert policy.confidence_decay_rate == 0.01

    def test_policy_with_usage_based_pruning(self):
        """Should support usage-based pruning."""
        policy = PruningPolicy(
            policy_id="usage-1",
            workspace_id="ws-123",
            name="Usage Based",
            min_retrieval_count=5,  # Prune if never retrieved
            usage_window_days=90,
        )
        assert policy.min_retrieval_count == 5
