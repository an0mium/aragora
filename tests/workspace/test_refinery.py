"""
Tests for Workspace Refinery (merge queue).

Tests the merge queue functionality for integrating convoy work.
"""

from __future__ import annotations

import pytest

from aragora.workspace.refinery import (
    ConflictResolution,
    MergeRequest,
    MergeStatus,
    Refinery,
    RefineryConfig,
)


# =============================================================================
# MergeRequest Tests
# =============================================================================


class TestMergeRequest:
    """Test MergeRequest dataclass."""

    def test_merge_request_creation(self):
        req = MergeRequest(
            convoy_id="convoy-1",
            rig_id="rig-1",
            source_branch="feature/test",
        )
        assert req.convoy_id == "convoy-1"
        assert req.rig_id == "rig-1"
        assert req.source_branch == "feature/test"
        assert req.target_branch == "main"
        assert req.status == MergeStatus.QUEUED
        assert req.request_id.startswith("merge-")
        assert req.queued_at > 0

    def test_merge_request_with_priority(self):
        req = MergeRequest(
            convoy_id="c1",
            rig_id="r1",
            source_branch="branch",
            priority=10,
        )
        assert req.priority == 10

    def test_merge_request_to_dict(self):
        req = MergeRequest(
            convoy_id="c1",
            rig_id="r1",
            source_branch="branch",
            request_id="merge-test",
        )
        data = req.to_dict()
        assert data["convoy_id"] == "c1"
        assert data["request_id"] == "merge-test"
        assert data["status"] == "queued"

    def test_merge_request_from_dict(self):
        data = {
            "convoy_id": "c1",
            "rig_id": "r1",
            "source_branch": "branch",
            "request_id": "merge-test",
            "status": "merging",
            "priority": 5,
        }
        req = MergeRequest.from_dict(data)
        assert req.convoy_id == "c1"
        assert req.status == MergeStatus.MERGING
        assert req.priority == 5

    def test_merge_request_roundtrip(self):
        original = MergeRequest(
            convoy_id="convoy-x",
            rig_id="rig-y",
            source_branch="feature/z",
            priority=7,
            metadata={"author": "test"},
        )
        data = original.to_dict()
        restored = MergeRequest.from_dict(data)
        assert restored.convoy_id == original.convoy_id
        assert restored.priority == original.priority
        assert restored.metadata == original.metadata


class TestMergeStatus:
    """Test MergeStatus enum."""

    def test_all_statuses_exist(self):
        statuses = [
            MergeStatus.QUEUED,
            MergeStatus.VALIDATING,
            MergeStatus.REBASING,
            MergeStatus.MERGING,
            MergeStatus.MERGED,
            MergeStatus.CONFLICT,
            MergeStatus.FAILED,
            MergeStatus.ROLLED_BACK,
        ]
        assert len(statuses) == 8


class TestRefineryConfig:
    """Test RefineryConfig defaults."""

    def test_defaults(self):
        config = RefineryConfig()
        assert config.max_concurrent_merges == 1
        assert config.auto_rebase is True
        assert config.require_tests is True
        assert config.require_review is False
        assert config.conflict_resolution == ConflictResolution.AUTO_REBASE

    def test_custom_config(self):
        config = RefineryConfig(
            max_concurrent_merges=3,
            require_review=True,
            conflict_resolution=ConflictResolution.MANUAL,
        )
        assert config.max_concurrent_merges == 3
        assert config.require_review is True
        assert config.conflict_resolution == ConflictResolution.MANUAL


# =============================================================================
# Refinery Tests
# =============================================================================


class TestRefineryInit:
    """Test Refinery initialization."""

    def test_default_init(self):
        refinery = Refinery()
        assert refinery.config.max_concurrent_merges == 1
        assert len(refinery._queue) == 0
        assert len(refinery._active) == 0

    def test_init_with_config(self):
        config = RefineryConfig(max_concurrent_merges=2)
        refinery = Refinery(config=config)
        assert refinery.config.max_concurrent_merges == 2


class TestRefineryQueue:
    """Test queue operations."""

    @pytest.fixture
    def refinery(self):
        return Refinery()

    @pytest.mark.asyncio
    async def test_queue_for_merge(self, refinery):
        req = await refinery.queue_for_merge(
            convoy_id="convoy-1",
            rig_id="rig-1",
            source_branch="feature/x",
        )
        assert req.convoy_id == "convoy-1"
        assert req.status == MergeStatus.QUEUED

        queue = await refinery.get_queue()
        assert len(queue) == 1

    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1", priority=0)
        await refinery.queue_for_merge("c2", "r1", "b2", priority=10)
        await refinery.queue_for_merge("c3", "r1", "b3", priority=5)

        queue = await refinery.get_queue()
        assert queue[0].convoy_id == "c2"  # Highest priority
        assert queue[1].convoy_id == "c3"
        assert queue[2].convoy_id == "c1"

    @pytest.mark.asyncio
    async def test_get_request(self, refinery):
        req = await refinery.queue_for_merge("c1", "r1", "b1")
        found = await refinery.get_request(req.request_id)
        assert found is not None
        assert found.convoy_id == "c1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_request(self, refinery):
        found = await refinery.get_request("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_cancel_request(self, refinery):
        req = await refinery.queue_for_merge("c1", "r1", "b1")
        cancelled = await refinery.cancel_request(req.request_id)
        assert cancelled is True
        queue = await refinery.get_queue()
        assert len(queue) == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, refinery):
        cancelled = await refinery.cancel_request("nonexistent")
        assert cancelled is False


class TestRefineryProcess:
    """Test merge processing."""

    @pytest.fixture
    def refinery(self):
        return Refinery()

    @pytest.mark.asyncio
    async def test_process_next_empty_queue(self, refinery):
        result = await refinery.process_next()
        assert result is None

    @pytest.mark.asyncio
    async def test_process_next_success(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        result = await refinery.process_next()

        assert result is not None
        assert result.status == MergeStatus.MERGED
        assert result.merge_commit is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_process_queue_all(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.queue_for_merge("c2", "r1", "b2")
        await refinery.queue_for_merge("c3", "r1", "b3")

        processed = await refinery.process_queue()
        assert len(processed) == 3
        assert all(r.status == MergeStatus.MERGED for r in processed)

    @pytest.mark.asyncio
    async def test_max_concurrent_merges(self):
        # With max_concurrent=1, only one at a time
        config = RefineryConfig(max_concurrent_merges=1)
        refinery = Refinery(config=config)

        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.queue_for_merge("c2", "r1", "b2")

        # First should process
        result1 = await refinery.process_next()
        assert result1.status == MergeStatus.MERGED

        # Queue should now have 1 left
        queue = await refinery.get_queue()
        assert len(queue) == 1

    @pytest.mark.asyncio
    async def test_get_active(self, refinery):
        # Initially empty
        active = await refinery.get_active()
        assert len(active) == 0


class TestRefineryHistory:
    """Test history tracking."""

    @pytest.fixture
    def refinery(self):
        return Refinery()

    @pytest.mark.asyncio
    async def test_get_history(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.process_next()

        history = await refinery.get_history()
        assert len(history) == 1
        assert history[0].status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_get_history_by_status(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.queue_for_merge("c2", "r1", "b2")
        await refinery.process_queue()

        merged = await refinery.get_history(status=MergeStatus.MERGED)
        assert len(merged) == 2

        failed = await refinery.get_history(status=MergeStatus.FAILED)
        assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_history_limit(self, refinery):
        for i in range(5):
            await refinery.queue_for_merge(f"c{i}", "r1", f"b{i}")
        await refinery.process_queue()

        history = await refinery.get_history(limit=3)
        assert len(history) == 3


class TestRefineryRollback:
    """Test rollback functionality."""

    @pytest.fixture
    def refinery(self):
        return Refinery()

    @pytest.mark.asyncio
    async def test_rollback_merged(self, refinery):
        req = await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.process_next()

        result = await refinery.rollback(req.request_id)
        assert result is True

        found = await refinery.get_request(req.request_id)
        assert found.status == MergeStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_nonexistent(self, refinery):
        result = await refinery.rollback("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_queued_fails(self, refinery):
        req = await refinery.queue_for_merge("c1", "r1", "b1")
        # Not processed yet
        result = await refinery.rollback(req.request_id)
        assert result is False


class TestRefineryStats:
    """Test statistics."""

    @pytest.fixture
    def refinery(self):
        return Refinery()

    @pytest.mark.asyncio
    async def test_get_stats(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.queue_for_merge("c2", "r1", "b2")

        stats = await refinery.get_stats()
        assert stats["queued"] == 2
        assert stats["active"] == 0
        assert stats["total_merged"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_process(self, refinery):
        await refinery.queue_for_merge("c1", "r1", "b1")
        await refinery.queue_for_merge("c2", "r1", "b2")
        await refinery.process_queue()

        stats = await refinery.get_stats()
        assert stats["queued"] == 0
        assert stats["total_merged"] == 2
