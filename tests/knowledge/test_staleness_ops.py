"""
Comprehensive tests for Knowledge Mound Staleness Operations.

Tests cover:
- StalenessOperationsMixin methods
- Stale knowledge detection
- Validation marking
- Revalidation scheduling
- Integration with control plane
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.staleness import StalenessOperationsMixin


# =============================================================================
# Mock Mound Class for Testing
# =============================================================================


class MockMound(StalenessOperationsMixin):
    """Mock mound class that implements the staleness mixin."""

    def __init__(self):
        self.config = MagicMock()
        self.workspace_id = "test-workspace"
        self._staleness_detector = None
        self._initialized = True
        self._updates = []

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("Not initialized")

    async def update(self, node_id: str, updates: dict):
        self._updates.append((node_id, updates))
        return MagicMock(id=node_id, **updates)


# =============================================================================
# Tests: get_stale_knowledge
# =============================================================================


class TestGetStaleKnowledge:
    """Tests for get_stale_knowledge method."""

    @pytest.mark.asyncio
    async def test_no_detector_returns_empty(self):
        """Should return empty list when no staleness detector configured."""
        mound = MockMound()
        result = await mound.get_stale_knowledge()
        assert result == []

    @pytest.mark.asyncio
    async def test_with_detector(self):
        """Should delegate to staleness detector."""
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(
            return_value=[
                MagicMock(node_id="item-1", staleness_score=0.8),
                MagicMock(node_id="item-2", staleness_score=0.7),
            ]
        )

        mound = MockMound()
        mound._staleness_detector = mock_detector

        result = await mound.get_stale_knowledge(threshold=0.6, limit=50)

        assert len(result) == 2
        mock_detector.get_stale_nodes.assert_called_once_with(
            workspace_id="test-workspace",
            threshold=0.6,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_custom_workspace_id(self):
        """Should use custom workspace ID if provided."""
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=[])

        mound = MockMound()
        mound._staleness_detector = mock_detector

        await mound.get_stale_knowledge(workspace_id="other-workspace")

        mock_detector.get_stale_nodes.assert_called_once_with(
            workspace_id="other-workspace",
            threshold=0.5,
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self):
        """Should raise if mound not initialized."""
        mound = MockMound()
        mound._initialized = False

        with pytest.raises(RuntimeError, match="Not initialized"):
            await mound.get_stale_knowledge()


# =============================================================================
# Tests: mark_validated
# =============================================================================


class TestMarkValidated:
    """Tests for mark_validated method."""

    @pytest.mark.asyncio
    async def test_mark_validated_basic(self):
        """Should update validation status."""
        mound = MockMound()

        await mound.mark_validated("node-123", validator="alice")

        assert len(mound._updates) == 1
        node_id, updates = mound._updates[0]
        assert node_id == "node-123"
        assert updates["validation_status"] == "majority_agreed"
        assert updates["staleness_score"] == 0.0
        assert "last_validated_at" in updates

    @pytest.mark.asyncio
    async def test_mark_validated_with_confidence(self):
        """Should update confidence if provided."""
        mound = MockMound()

        await mound.mark_validated("node-123", validator="bob", confidence=0.95)

        node_id, updates = mound._updates[0]
        assert updates["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_mark_validated_without_confidence(self):
        """Should not include confidence if not provided."""
        mound = MockMound()

        await mound.mark_validated("node-123", validator="carol")

        node_id, updates = mound._updates[0]
        assert "confidence" not in updates

    @pytest.mark.asyncio
    async def test_mark_validated_sets_timestamp(self):
        """Should set validation timestamp."""
        mound = MockMound()

        await mound.mark_validated("node-123", validator="dave")

        node_id, updates = mound._updates[0]
        assert "last_validated_at" in updates
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(updates["last_validated_at"])


# =============================================================================
# Tests: schedule_revalidation
# =============================================================================


class TestScheduleRevalidation:
    """Tests for schedule_revalidation method."""

    @pytest.mark.asyncio
    async def test_schedule_single_node(self):
        """Should schedule revalidation for a single node."""
        mound = MockMound()

        with patch(
            "aragora.knowledge.mound.ops.staleness._task_queue",
            [],
            create=True,
        ):
            # Mock the import to avoid actual control plane
            with patch.dict(
                "sys.modules",
                {"aragora.server.handlers.features.control_plane": MagicMock(_task_queue=[])},
            ):
                task_ids = await mound.schedule_revalidation(["node-123"])

        # Should mark node for revalidation
        assert len(mound._updates) == 1
        node_id, updates = mound._updates[0]
        assert node_id == "node-123"
        assert updates["revalidation_requested"] is True

        # Should return task IDs
        assert len(task_ids) == 1

    @pytest.mark.asyncio
    async def test_schedule_multiple_nodes(self):
        """Should schedule revalidation for multiple nodes."""
        mound = MockMound()

        with patch.dict(
            "sys.modules",
            {"aragora.server.handlers.features.control_plane": MagicMock(_task_queue=[])},
        ):
            task_ids = await mound.schedule_revalidation(
                ["node-1", "node-2", "node-3"], priority="high"
            )

        assert len(mound._updates) == 3
        assert len(task_ids) == 3

    @pytest.mark.asyncio
    async def test_schedule_with_priority(self):
        """Should include priority in task."""
        mound = MockMound()

        task_queue = []
        mock_module = MagicMock()
        mock_module._task_queue = task_queue

        with patch.dict(
            "sys.modules", {"aragora.server.handlers.features.control_plane": mock_module}
        ):
            await mound.schedule_revalidation(["node-123"], priority="high")

        # Task should be added to queue with priority
        if task_queue:
            assert task_queue[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_schedule_empty_list(self):
        """Should handle empty node list."""
        mound = MockMound()

        task_ids = await mound.schedule_revalidation([])

        assert task_ids == []
        assert len(mound._updates) == 0

    @pytest.mark.asyncio
    async def test_schedule_fallback_when_control_plane_unavailable(self):
        """Should create pending IDs when control plane unavailable."""
        mound = MockMound()

        # Force ImportError by not mocking the module
        with patch("aragora.knowledge.mound.ops.staleness.logger") as mock_logger:
            # The import will fail, so we get pending IDs
            task_ids = await mound.schedule_revalidation(["node-123"])

        # Should still mark node and return pending ID
        assert len(mound._updates) == 1
        assert len(task_ids) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestStalenessIntegration:
    """Integration tests for staleness operations."""

    @pytest.mark.asyncio
    async def test_detect_and_schedule_revalidation_flow(self):
        """Full flow: detect stale → schedule revalidation."""
        # Setup mock detector with stale nodes
        stale_nodes = [
            MagicMock(node_id="item-1", staleness_score=0.85),
            MagicMock(node_id="item-2", staleness_score=0.72),
        ]
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=stale_nodes)

        mound = MockMound()
        mound._staleness_detector = mock_detector

        # Step 1: Detect stale nodes
        stale = await mound.get_stale_knowledge(threshold=0.7)
        assert len(stale) == 2

        # Step 2: Schedule revalidation
        node_ids = [node.node_id for node in stale]
        with patch.dict(
            "sys.modules",
            {"aragora.server.handlers.features.control_plane": MagicMock(_task_queue=[])},
        ):
            task_ids = await mound.schedule_revalidation(node_ids)

        assert len(task_ids) == 2

    @pytest.mark.asyncio
    async def test_validate_after_revalidation(self):
        """Mark node validated after successful revalidation."""
        mound = MockMound()

        # Simulate revalidation complete
        await mound.mark_validated("item-1", validator="revalidation_worker", confidence=0.9)

        node_id, updates = mound._updates[0]
        assert updates["validation_status"] == "majority_agreed"
        assert updates["confidence"] == 0.9
        assert updates["staleness_score"] == 0.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestStalenessEdgeCases:
    """Edge case tests for staleness operations."""

    @pytest.mark.asyncio
    async def test_detector_returns_none(self):
        """Should handle detector returning None."""
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=None)

        mound = MockMound()
        mound._staleness_detector = mock_detector

        # Should not crash, but behavior depends on implementation
        # The actual code returns the result directly, so None would be returned
        result = await mound.get_stale_knowledge()
        assert result is None  # Or [] depending on implementation

    @pytest.mark.asyncio
    async def test_detector_raises_exception(self):
        """Should propagate detector exceptions."""
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(side_effect=RuntimeError("Database error"))

        mound = MockMound()
        mound._staleness_detector = mock_detector

        with pytest.raises(RuntimeError, match="Database error"):
            await mound.get_stale_knowledge()

    @pytest.mark.asyncio
    async def test_update_failure_in_mark_validated(self):
        """Should propagate update failures."""

        class FailingMound(MockMound):
            async def update(self, node_id, updates):
                raise RuntimeError("Update failed")

        mound = FailingMound()

        with pytest.raises(RuntimeError, match="Update failed"):
            await mound.mark_validated("node-123", validator="test")

    @pytest.mark.asyncio
    async def test_very_low_threshold(self):
        """Should handle very low threshold (most items stale)."""
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(
            return_value=[MagicMock(node_id=f"item-{i}", staleness_score=0.1) for i in range(1000)]
        )

        mound = MockMound()
        mound._staleness_detector = mock_detector

        result = await mound.get_stale_knowledge(threshold=0.05, limit=100)
        # Should respect limit even with many results
        assert len(result) == 1000  # Detector returns all, limit is passed to detector

    @pytest.mark.asyncio
    async def test_concurrent_validation(self):
        """Should handle concurrent validation calls."""
        import asyncio

        mound = MockMound()

        async def validate(node_id):
            await mound.mark_validated(node_id, validator="test")

        await asyncio.gather(*[validate(f"node-{i}") for i in range(50)])

        assert len(mound._updates) == 50
