"""Tests for DedupOperationsMixin (aragora/server/handlers/knowledge_base/mound/dedup.py).

Covers all four async methods on the mixin:
- get_duplicate_clusters      (GET  /api/knowledge/mound/dedup/clusters)
- get_dedup_report            (GET  /api/knowledge/mound/dedup/report)
- merge_duplicate_cluster     (POST /api/knowledge/mound/dedup/merge)
- auto_merge_exact_duplicates (POST /api/knowledge/mound/dedup/auto-merge)

Each method is tested for:
- Success with valid inputs and response field verification
- Mound not available (503)
- Missing required parameters (400)
- All four caught exception types: KeyError, ValueError, OSError, TypeError (500)
- Edge cases: empty results, default parameters, boundary values, multiple items
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.handlers.knowledge_base.mound.dedup import (
    DedupOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


@dataclass
class MockDuplicate:
    """Mock duplicate node entry within a cluster."""

    node_id: str = "dup-001"
    similarity: float = 0.95
    content_preview: str = "Some duplicate content..."
    tier: str = "warm"
    confidence: float = 0.8


@dataclass
class MockDuplicateCluster:
    """Mock duplicate cluster returned by mound.find_duplicates."""

    cluster_id: str = "cluster-001"
    primary_node_id: str = "primary-001"
    duplicates: list[MockDuplicate] = field(
        default_factory=lambda: [MockDuplicate()]
    )
    avg_similarity: float = 0.92
    recommended_action: str = "merge"


@dataclass
class MockDedupReport:
    """Mock dedup report returned by mound.generate_dedup_report."""

    workspace_id: str = "default"
    generated_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    total_nodes_analyzed: int = 500
    duplicate_clusters_found: int = 12
    estimated_reduction_percent: float = 8.5
    clusters: list[MockDuplicateCluster] = field(default_factory=list)


@dataclass
class MockMergeResult:
    """Mock merge result returned by mound.merge_duplicates."""

    kept_node_id: str = "primary-001"
    merged_node_ids: list[str] = field(
        default_factory=lambda: ["dup-001", "dup-002"]
    )
    archived_count: int = 2
    updated_relationships: int = 5


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class DedupTestHandler(DedupOperationsMixin):
    """Concrete handler for testing the dedup mixin."""

    def __init__(self, mound=None):
        self._mound = mound
        self.ctx: dict[str, Any] = {}

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with AsyncMock methods for dedup."""
    mound = MagicMock()
    mound.find_duplicates = AsyncMock(return_value=[])
    mound.generate_dedup_report = AsyncMock(return_value=MockDedupReport())
    mound.merge_duplicates = AsyncMock(return_value=MockMergeResult())
    mound.auto_merge_exact_duplicates = AsyncMock(
        return_value={
            "dry_run": True,
            "duplicates_found": 0,
            "merges_performed": 0,
            "details": [],
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a DedupTestHandler with a mock mound."""
    return DedupTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a DedupTestHandler with no mound (returns None)."""
    return DedupTestHandler(mound=None)


# ============================================================================
# Tests: get_duplicate_clusters
# ============================================================================


class TestGetDuplicateClusters:
    """Test the get_duplicate_clusters method."""

    @pytest.mark.asyncio
    async def test_success_returns_clusters(self, handler, mock_mound):
        """Successful call returns clusters list with metadata."""
        clusters = [
            MockDuplicateCluster(
                cluster_id="c1",
                primary_node_id="p1",
                duplicates=[
                    MockDuplicate(node_id="d1", similarity=0.95, tier="warm"),
                    MockDuplicate(node_id="d2", similarity=0.91, tier="cold"),
                ],
                avg_similarity=0.93,
                recommended_action="merge",
            ),
        ]
        mock_mound.find_duplicates = AsyncMock(return_value=clusters)

        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)

        assert _status(result) == 200
        assert body["workspace_id"] == "ws-1"
        assert body["clusters_found"] == 1
        assert len(body["clusters"]) == 1
        c0 = body["clusters"][0]
        assert c0["cluster_id"] == "c1"
        assert c0["primary_node_id"] == "p1"
        assert c0["duplicate_count"] == 2
        assert c0["avg_similarity"] == 0.93
        assert c0["recommended_action"] == "merge"

    @pytest.mark.asyncio
    async def test_success_empty_clusters(self, handler, mock_mound):
        """No duplicate clusters returns empty list with count 0."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["clusters_found"] == 0
        assert body["clusters"] == []

    @pytest.mark.asyncio
    async def test_cluster_duplicate_fields(self, handler, mock_mound):
        """All expected duplicate fields are present in the response."""
        dup = MockDuplicate(
            node_id="d-check",
            similarity=0.97,
            content_preview="preview text here",
            tier="hot",
            confidence=0.85,
        )
        clusters = [
            MockDuplicateCluster(
                cluster_id="c-check",
                primary_node_id="p-check",
                duplicates=[dup],
                avg_similarity=0.97,
                recommended_action="auto_merge",
            ),
        ]
        mock_mound.find_duplicates = AsyncMock(return_value=clusters)
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        d0 = body["clusters"][0]["duplicates"][0]

        expected_fields = ["node_id", "similarity", "content_preview", "tier", "confidence"]
        for f in expected_fields:
            assert f in d0, f"Missing field: {f}"

        assert d0["node_id"] == "d-check"
        assert d0["similarity"] == 0.97
        assert d0["content_preview"] == "preview text here"
        assert d0["tier"] == "hot"
        assert d0["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_default_similarity_threshold_in_response(self, handler, mock_mound):
        """Default similarity_threshold is reflected in the response."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert body["similarity_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_custom_similarity_threshold(self, handler, mock_mound):
        """Custom similarity_threshold is forwarded and reflected."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(
            workspace_id="ws-1",
            similarity_threshold=0.75,
        )
        body = _body(result)
        assert body["similarity_threshold"] == 0.75
        mock_mound.find_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            similarity_threshold=0.75,
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to mound."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(
            workspace_id="ws-1",
            limit=25,
        )
        body = _body(result)
        assert _status(result) == 200
        mock_mound.find_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            similarity_threshold=0.9,
            limit=25,
        )

    @pytest.mark.asyncio
    async def test_custom_threshold_and_limit(self, handler, mock_mound):
        """Custom similarity_threshold and limit are both forwarded."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        await handler.get_duplicate_clusters(
            workspace_id="ws-2",
            similarity_threshold=0.5,
            limit=10,
        )
        mock_mound.find_duplicates.assert_awaited_once_with(
            workspace_id="ws-2",
            similarity_threshold=0.5,
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_multiple_clusters(self, handler, mock_mound):
        """Multiple clusters are correctly serialized."""
        clusters = [
            MockDuplicateCluster(cluster_id="c1", duplicates=[MockDuplicate(node_id="d1")]),
            MockDuplicateCluster(cluster_id="c2", duplicates=[MockDuplicate(node_id="d2"), MockDuplicate(node_id="d3")]),
            MockDuplicateCluster(cluster_id="c3", duplicates=[]),
        ]
        mock_mound.find_duplicates = AsyncMock(return_value=clusters)
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)

        assert body["clusters_found"] == 3
        assert body["clusters"][0]["cluster_id"] == "c1"
        assert body["clusters"][0]["duplicate_count"] == 1
        assert body["clusters"][1]["cluster_id"] == "c2"
        assert body["clusters"][1]["duplicate_count"] == 2
        assert body["clusters"][2]["cluster_id"] == "c3"
        assert body["clusters"][2]["duplicate_count"] == 0

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.get_duplicate_clusters(workspace_id="")
        assert _status(result) == 400
        assert "workspace_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.find_duplicates = AsyncMock(side_effect=ValueError("bad query"))
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.find_duplicates = AsyncMock(side_effect=KeyError("missing"))
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.find_duplicates = AsyncMock(side_effect=OSError("disk fail"))
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.find_duplicates = AsyncMock(side_effect=TypeError("type mismatch"))
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_cluster_with_many_duplicates(self, handler, mock_mound):
        """Cluster with many duplicates serializes all of them."""
        dups = [MockDuplicate(node_id=f"d-{i}", similarity=0.9 + i * 0.01) for i in range(10)]
        clusters = [MockDuplicateCluster(cluster_id="big", duplicates=dups)]
        mock_mound.find_duplicates = AsyncMock(return_value=clusters)
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)

        assert body["clusters"][0]["duplicate_count"] == 10
        assert len(body["clusters"][0]["duplicates"]) == 10
        assert body["clusters"][0]["duplicates"][0]["node_id"] == "d-0"
        assert body["clusters"][0]["duplicates"][9]["node_id"] == "d-9"

    @pytest.mark.asyncio
    async def test_threshold_boundary_zero(self, handler, mock_mound):
        """Similarity threshold of 0.0 is forwarded."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(
            workspace_id="ws-1",
            similarity_threshold=0.0,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["similarity_threshold"] == 0.0

    @pytest.mark.asyncio
    async def test_threshold_boundary_one(self, handler, mock_mound):
        """Similarity threshold of 1.0 is forwarded."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(
            workspace_id="ws-1",
            similarity_threshold=1.0,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["similarity_threshold"] == 1.0

    @pytest.mark.asyncio
    async def test_response_contains_workspace_id(self, handler, mock_mound):
        """Response always includes the requested workspace_id."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(workspace_id="my-ws-123")
        body = _body(result)
        assert body["workspace_id"] == "my-ws-123"


# ============================================================================
# Tests: get_dedup_report
# ============================================================================


class TestGetDedupReport:
    """Test the get_dedup_report method."""

    @pytest.mark.asyncio
    async def test_success_returns_report(self, handler, mock_mound):
        """Successful call returns report with statistics."""
        report = MockDedupReport(
            workspace_id="ws-1",
            generated_at=datetime(2026, 2, 15, 10, 30, 0, tzinfo=timezone.utc),
            total_nodes_analyzed=1000,
            duplicate_clusters_found=25,
            estimated_reduction_percent=12.5,
            clusters=[MockDuplicateCluster(), MockDuplicateCluster()],
        )
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)

        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)

        assert _status(result) == 200
        assert body["workspace_id"] == "ws-1"
        assert body["generated_at"] == "2026-02-15T10:30:00+00:00"
        assert body["total_nodes_analyzed"] == 1000
        assert body["duplicate_clusters_found"] == 25
        assert body["estimated_reduction_percent"] == 12.5
        assert body["cluster_count"] == 2

    @pytest.mark.asyncio
    async def test_success_empty_clusters(self, handler, mock_mound):
        """Report with no clusters returns cluster_count 0."""
        report = MockDedupReport(clusters=[])
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["cluster_count"] == 0

    @pytest.mark.asyncio
    async def test_default_similarity_threshold(self, handler, mock_mound):
        """Default similarity_threshold is 0.9."""
        report = MockDedupReport()
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        await handler.get_dedup_report(workspace_id="ws-1")
        mock_mound.generate_dedup_report.assert_awaited_once_with(
            workspace_id="ws-1",
            similarity_threshold=0.9,
        )

    @pytest.mark.asyncio
    async def test_custom_similarity_threshold(self, handler, mock_mound):
        """Custom similarity_threshold is forwarded."""
        report = MockDedupReport()
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        await handler.get_dedup_report(
            workspace_id="ws-1",
            similarity_threshold=0.7,
        )
        mock_mound.generate_dedup_report.assert_awaited_once_with(
            workspace_id="ws-1",
            similarity_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_dedup_report(workspace_id="ws-1")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.get_dedup_report(workspace_id="")
        assert _status(result) == 400
        assert "workspace_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.generate_dedup_report = AsyncMock(side_effect=ValueError("bad data"))
        result = await handler.get_dedup_report(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.generate_dedup_report = AsyncMock(side_effect=KeyError("missing key"))
        result = await handler.get_dedup_report(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.generate_dedup_report = AsyncMock(side_effect=OSError("storage error"))
        result = await handler.get_dedup_report(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.generate_dedup_report = AsyncMock(side_effect=TypeError("wrong type"))
        result = await handler.get_dedup_report(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_report_fields_present(self, handler, mock_mound):
        """All expected report fields are present in the response."""
        report = MockDedupReport()
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)

        expected_fields = [
            "workspace_id", "generated_at", "total_nodes_analyzed",
            "duplicate_clusters_found", "estimated_reduction_percent",
            "cluster_count",
        ]
        for f in expected_fields:
            assert f in body, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_generated_at_iso_format(self, handler, mock_mound):
        """generated_at is in ISO format."""
        ts = datetime(2026, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        report = MockDedupReport(generated_at=ts)
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)
        assert body["generated_at"] == "2026-06-15T08:00:00+00:00"

    @pytest.mark.asyncio
    async def test_zero_reduction_percent(self, handler, mock_mound):
        """Report with 0% reduction is valid."""
        report = MockDedupReport(estimated_reduction_percent=0.0)
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["estimated_reduction_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_large_cluster_count(self, handler, mock_mound):
        """Report with many clusters returns correct count."""
        clusters = [MockDuplicateCluster(cluster_id=f"c-{i}") for i in range(50)]
        report = MockDedupReport(clusters=clusters, duplicate_clusters_found=50)
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)
        assert body["cluster_count"] == 50
        assert body["duplicate_clusters_found"] == 50

    @pytest.mark.asyncio
    async def test_workspace_id_in_response(self, handler, mock_mound):
        """Response uses workspace_id from the report object."""
        report = MockDedupReport(workspace_id="from-report")
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="from-report")
        body = _body(result)
        assert body["workspace_id"] == "from-report"


# ============================================================================
# Tests: merge_duplicate_cluster
# ============================================================================


class TestMergeDuplicateCluster:
    """Test the merge_duplicate_cluster method."""

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_mound):
        """Successful merge returns result details."""
        merge_result = MockMergeResult(
            kept_node_id="primary-001",
            merged_node_ids=["dup-001", "dup-002", "dup-003"],
            archived_count=3,
            updated_relationships=8,
        )
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)

        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="cluster-abc",
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["success"] is True
        assert body["kept_node_id"] == "primary-001"
        assert body["merged_node_ids"] == ["dup-001", "dup-002", "dup-003"]
        assert body["archived_count"] == 3
        assert body["updated_relationships"] == 8

    @pytest.mark.asyncio
    async def test_success_with_primary_node_id(self, handler, mock_mound):
        """Custom primary_node_id is forwarded."""
        merge_result = MockMergeResult(kept_node_id="custom-primary")
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)

        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
            primary_node_id="custom-primary",
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["kept_node_id"] == "custom-primary"
        mock_mound.merge_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            cluster_id="c-1",
            primary_node_id="custom-primary",
            archive=True,
        )

    @pytest.mark.asyncio
    async def test_default_archive_true(self, handler, mock_mound):
        """Default archive is True."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)

        await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        call_kwargs = mock_mound.merge_duplicates.call_args.kwargs
        assert call_kwargs["archive"] is True

    @pytest.mark.asyncio
    async def test_archive_false(self, handler, mock_mound):
        """archive=False (delete duplicates) is forwarded."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)

        await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
            archive=False,
        )
        call_kwargs = mock_mound.merge_duplicates.call_args.kwargs
        assert call_kwargs["archive"] is False

    @pytest.mark.asyncio
    async def test_default_primary_node_id_none(self, handler, mock_mound):
        """Default primary_node_id is None (auto-selected)."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)

        await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        call_kwargs = mock_mound.merge_duplicates.call_args.kwargs
        assert call_kwargs["primary_node_id"] is None

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.merge_duplicate_cluster(
            workspace_id="",
            cluster_id="c-1",
        )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_cluster_id_returns_400(self, handler):
        """Empty cluster_id returns 400."""
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="",
        )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_both_empty_returns_400(self, handler):
        """Both workspace_id and cluster_id empty returns 400."""
        result = await handler.merge_duplicate_cluster(
            workspace_id="",
            cluster_id="",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.merge_duplicates = AsyncMock(side_effect=ValueError("invalid merge"))
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.merge_duplicates = AsyncMock(side_effect=KeyError("cluster not found"))
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.merge_duplicates = AsyncMock(side_effect=OSError("write failed"))
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.merge_duplicates = AsyncMock(side_effect=TypeError("unexpected"))
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_result_fields_present(self, handler, mock_mound):
        """All expected result fields are present in the response."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)

        expected_fields = [
            "success", "kept_node_id", "merged_node_ids",
            "archived_count", "updated_relationships",
        ]
        for f in expected_fields:
            assert f in body, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_zero_archived_count(self, handler, mock_mound):
        """Zero archived_count (e.g., archive=False scenario) is valid."""
        merge_result = MockMergeResult(archived_count=0)
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["archived_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_merged_node_ids(self, handler, mock_mound):
        """Empty merged_node_ids list is valid."""
        merge_result = MockMergeResult(merged_node_ids=[])
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["merged_node_ids"] == []

    @pytest.mark.asyncio
    async def test_mound_called_with_correct_args(self, handler, mock_mound):
        """Mound is called with the exact arguments provided."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        await handler.merge_duplicate_cluster(
            workspace_id="ws-x",
            cluster_id="c-y",
            primary_node_id="p-z",
            archive=False,
        )
        mock_mound.merge_duplicates.assert_awaited_once_with(
            workspace_id="ws-x",
            cluster_id="c-y",
            primary_node_id="p-z",
            archive=False,
        )

    @pytest.mark.asyncio
    async def test_large_merged_node_ids_list(self, handler, mock_mound):
        """Many merged_node_ids are serialized correctly."""
        ids = [f"node-{i}" for i in range(100)]
        merge_result = MockMergeResult(merged_node_ids=ids, archived_count=100)
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)
        assert len(body["merged_node_ids"]) == 100
        assert body["archived_count"] == 100


# ============================================================================
# Tests: auto_merge_exact_duplicates
# ============================================================================


class TestAutoMergeExactDuplicates:
    """Test the auto_merge_exact_duplicates method."""

    @pytest.mark.asyncio
    async def test_success_dry_run(self, handler, mock_mound):
        """Successful dry run returns summary without performing merges."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": True,
                "duplicates_found": 15,
                "merges_performed": 0,
                "details": [
                    {"cluster_id": "c1", "nodes": ["n1", "n2"]},
                ],
            }
        )

        result = await handler.auto_merge_exact_duplicates(
            workspace_id="ws-1",
            dry_run=True,
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["workspace_id"] == "ws-1"
        assert body["dry_run"] is True
        assert body["duplicates_found"] == 15
        assert body["merges_performed"] == 0
        assert len(body["details"]) == 1

    @pytest.mark.asyncio
    async def test_success_actual_merge(self, handler, mock_mound):
        """Actual merge returns counts with dry_run=False."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": False,
                "duplicates_found": 10,
                "merges_performed": 10,
                "details": [],
            }
        )

        result = await handler.auto_merge_exact_duplicates(
            workspace_id="ws-1",
            dry_run=False,
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["dry_run"] is False
        assert body["duplicates_found"] == 10
        assert body["merges_performed"] == 10

    @pytest.mark.asyncio
    async def test_default_dry_run_true(self, handler, mock_mound):
        """Default dry_run is True."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={"dry_run": True, "duplicates_found": 0, "merges_performed": 0, "details": []}
        )
        await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        mock_mound.auto_merge_exact_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.auto_merge_exact_duplicates(workspace_id="ws-1")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.auto_merge_exact_duplicates(workspace_id="")
        assert _status(result) == 400
        assert "workspace_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(side_effect=ValueError("bad"))
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(side_effect=KeyError("oops"))
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(side_effect=OSError("fs error"))
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(side_effect=TypeError("nope"))
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_zero_duplicates(self, handler, mock_mound):
        """No duplicates found returns zero counts."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": True,
                "duplicates_found": 0,
                "merges_performed": 0,
                "details": [],
            }
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["duplicates_found"] == 0
        assert body["merges_performed"] == 0
        assert body["details"] == []

    @pytest.mark.asyncio
    async def test_response_fields_present(self, handler, mock_mound):
        """All expected response fields are present."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={"dry_run": True, "duplicates_found": 5, "merges_performed": 0, "details": []}
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        body = _body(result)

        expected_fields = [
            "workspace_id", "dry_run", "duplicates_found",
            "merges_performed", "details",
        ]
        for f in expected_fields:
            assert f in body, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_workspace_id_in_response(self, handler, mock_mound):
        """Response includes the requested workspace_id."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={"dry_run": True}
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="my-ws")
        body = _body(result)
        assert body["workspace_id"] == "my-ws"

    @pytest.mark.asyncio
    async def test_result_missing_keys_use_defaults(self, handler, mock_mound):
        """Result dict missing keys uses defaults from .get()."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={}
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        # Defaults via .get()
        assert body["duplicates_found"] == 0
        assert body["merges_performed"] == 0
        assert body["details"] == []

    @pytest.mark.asyncio
    async def test_dry_run_from_result_dict(self, handler, mock_mound):
        """dry_run value comes from the result dict if present."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={"dry_run": False, "duplicates_found": 3, "merges_performed": 3, "details": []}
        )
        result = await handler.auto_merge_exact_duplicates(
            workspace_id="ws-1",
            dry_run=False,
        )
        body = _body(result)
        assert body["dry_run"] is False

    @pytest.mark.asyncio
    async def test_dry_run_fallback_to_param(self, handler, mock_mound):
        """When result dict lacks dry_run key, falls back to the parameter value."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={"duplicates_found": 2, "merges_performed": 2}
        )
        result = await handler.auto_merge_exact_duplicates(
            workspace_id="ws-1",
            dry_run=False,
        )
        body = _body(result)
        assert body["dry_run"] is False

    @pytest.mark.asyncio
    async def test_mound_called_with_dry_run_true(self, handler, mock_mound):
        """Mound is called with dry_run=True."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(return_value={})
        await handler.auto_merge_exact_duplicates(workspace_id="ws-1", dry_run=True)
        mock_mound.auto_merge_exact_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_mound_called_with_dry_run_false(self, handler, mock_mound):
        """Mound is called with dry_run=False."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(return_value={})
        await handler.auto_merge_exact_duplicates(workspace_id="ws-1", dry_run=False)
        mock_mound.auto_merge_exact_duplicates.assert_awaited_once_with(
            workspace_id="ws-1",
            dry_run=False,
        )

    @pytest.mark.asyncio
    async def test_details_with_multiple_entries(self, handler, mock_mound):
        """Multiple detail entries are serialized correctly."""
        details = [
            {"cluster_id": f"c-{i}", "nodes": [f"n-{i}a", f"n-{i}b"]}
            for i in range(5)
        ]
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": False,
                "duplicates_found": 10,
                "merges_performed": 5,
                "details": details,
            }
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1", dry_run=False)
        body = _body(result)
        assert len(body["details"]) == 5
        assert body["details"][0]["cluster_id"] == "c-0"
        assert body["details"][4]["cluster_id"] == "c-4"


# ============================================================================
# Tests: Cross-cutting concerns
# ============================================================================


class TestCrossCuttingConcerns:
    """Test cross-cutting behavior across all methods."""

    @pytest.mark.asyncio
    async def test_all_methods_return_503_when_no_mound(self, handler_no_mound):
        """All methods return 503 when mound is unavailable."""
        r1 = await handler_no_mound.get_duplicate_clusters(workspace_id="ws")
        r2 = await handler_no_mound.get_dedup_report(workspace_id="ws")
        r3 = await handler_no_mound.merge_duplicate_cluster(workspace_id="ws", cluster_id="c")
        r4 = await handler_no_mound.auto_merge_exact_duplicates(workspace_id="ws")

        assert _status(r1) == 503
        assert _status(r2) == 503
        assert _status(r3) == 503
        assert _status(r4) == 503

    @pytest.mark.asyncio
    async def test_all_methods_return_400_for_empty_workspace(self, handler):
        """All methods return 400 for empty workspace_id."""
        r1 = await handler.get_duplicate_clusters(workspace_id="")
        r2 = await handler.get_dedup_report(workspace_id="")
        r3 = await handler.merge_duplicate_cluster(workspace_id="", cluster_id="c")
        r4 = await handler.auto_merge_exact_duplicates(workspace_id="")

        assert _status(r1) == 400
        assert _status(r2) == 400
        assert _status(r3) == 400
        assert _status(r4) == 400

    @pytest.mark.asyncio
    async def test_mound_not_called_when_validation_fails(self, handler, mock_mound):
        """Mound methods are not called when validation fails."""
        await handler.get_duplicate_clusters(workspace_id="")
        await handler.get_dedup_report(workspace_id="")
        await handler.merge_duplicate_cluster(workspace_id="", cluster_id="c")
        await handler.auto_merge_exact_duplicates(workspace_id="")

        mock_mound.find_duplicates.assert_not_awaited()
        mock_mound.generate_dedup_report.assert_not_awaited()
        mock_mound.merge_duplicates.assert_not_awaited()
        mock_mound.auto_merge_exact_duplicates.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_503_error_message_consistency(self, handler_no_mound):
        """All 503 responses have consistent error messages."""
        results = [
            await handler_no_mound.get_duplicate_clusters(workspace_id="ws"),
            await handler_no_mound.get_dedup_report(workspace_id="ws"),
            await handler_no_mound.merge_duplicate_cluster(workspace_id="ws", cluster_id="c"),
            await handler_no_mound.auto_merge_exact_duplicates(workspace_id="ws"),
        ]
        for r in results:
            body = _body(r)
            assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_success_responses_are_200(self, handler, mock_mound):
        """All successful operations return 200."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        mock_mound.generate_dedup_report = AsyncMock(return_value=MockDedupReport())
        mock_mound.merge_duplicates = AsyncMock(return_value=MockMergeResult())
        mock_mound.auto_merge_exact_duplicates = AsyncMock(return_value={})

        r1 = await handler.get_duplicate_clusters(workspace_id="ws")
        r2 = await handler.get_dedup_report(workspace_id="ws")
        r3 = await handler.merge_duplicate_cluster(workspace_id="ws", cluster_id="c")
        r4 = await handler.auto_merge_exact_duplicates(workspace_id="ws")

        assert _status(r1) == 200
        assert _status(r2) == 200
        assert _status(r3) == 200
        assert _status(r4) == 200

    @pytest.mark.asyncio
    async def test_error_sanitization(self, handler, mock_mound):
        """Error messages from exceptions are sanitized via safe_error_message."""
        # The handler uses safe_error_message which strips sensitive info.
        # We just verify that 500 is returned and an error is in the body.
        mock_mound.find_duplicates = AsyncMock(
            side_effect=ValueError("sensitive info: password=abc123")
        )
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        assert _status(result) == 500
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_handler_with_mound_returning_false_for_merge_validation(self, handler):
        """Both workspace_id and cluster_id must be non-empty for merge."""
        # workspace present, cluster empty
        result = await handler.merge_duplicate_cluster(workspace_id="ws-1", cluster_id="")
        assert _status(result) == 400

        # workspace empty, cluster present
        result2 = await handler.merge_duplicate_cluster(workspace_id="", cluster_id="c-1")
        assert _status(result2) == 400


# ============================================================================
# Tests: Edge cases and additional coverage
# ============================================================================


class TestEdgeCases:
    """Additional edge case tests across all dedup methods."""

    @pytest.mark.asyncio
    async def test_clusters_workspace_with_special_chars(self, handler, mock_mound):
        """Workspace IDs with special characters are preserved."""
        mock_mound.find_duplicates = AsyncMock(return_value=[])
        result = await handler.get_duplicate_clusters(workspace_id="ws/foo-bar_123")
        body = _body(result)
        assert body["workspace_id"] == "ws/foo-bar_123"

    @pytest.mark.asyncio
    async def test_report_workspace_with_special_chars(self, handler, mock_mound):
        """Report preserves workspace ID from the report object."""
        report = MockDedupReport(workspace_id="org::team::project")
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="org::team::project")
        body = _body(result)
        assert body["workspace_id"] == "org::team::project"

    @pytest.mark.asyncio
    async def test_merge_success_flag_always_true(self, handler, mock_mound):
        """On successful merge, success is always True."""
        merge_result = MockMergeResult()
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_auto_merge_empty_details_list(self, handler, mock_mound):
        """Empty details list is valid in auto-merge."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": True,
                "duplicates_found": 0,
                "merges_performed": 0,
                "details": [],
            }
        )
        result = await handler.auto_merge_exact_duplicates(workspace_id="ws-1")
        body = _body(result)
        assert body["details"] == []

    @pytest.mark.asyncio
    async def test_cluster_duplicate_with_zero_similarity(self, handler, mock_mound):
        """Duplicate with 0.0 similarity is serialized correctly."""
        dup = MockDuplicate(node_id="d-zero", similarity=0.0)
        cluster = MockDuplicateCluster(cluster_id="c-zero", duplicates=[dup])
        mock_mound.find_duplicates = AsyncMock(return_value=[cluster])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert body["clusters"][0]["duplicates"][0]["similarity"] == 0.0

    @pytest.mark.asyncio
    async def test_cluster_duplicate_with_perfect_similarity(self, handler, mock_mound):
        """Duplicate with 1.0 (perfect) similarity is serialized correctly."""
        dup = MockDuplicate(node_id="d-perfect", similarity=1.0)
        cluster = MockDuplicateCluster(cluster_id="c-perfect", duplicates=[dup])
        mock_mound.find_duplicates = AsyncMock(return_value=[cluster])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert body["clusters"][0]["duplicates"][0]["similarity"] == 1.0

    @pytest.mark.asyncio
    async def test_report_high_reduction_percent(self, handler, mock_mound):
        """Report with very high reduction percent is valid."""
        report = MockDedupReport(estimated_reduction_percent=95.7)
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-1")
        body = _body(result)
        assert body["estimated_reduction_percent"] == 95.7

    @pytest.mark.asyncio
    async def test_report_zero_nodes_analyzed(self, handler, mock_mound):
        """Report with zero nodes analyzed is valid (empty workspace)."""
        report = MockDedupReport(total_nodes_analyzed=0, duplicate_clusters_found=0)
        mock_mound.generate_dedup_report = AsyncMock(return_value=report)
        result = await handler.get_dedup_report(workspace_id="ws-empty")
        body = _body(result)
        assert _status(result) == 200
        assert body["total_nodes_analyzed"] == 0
        assert body["duplicate_clusters_found"] == 0

    @pytest.mark.asyncio
    async def test_merge_updated_relationships_zero(self, handler, mock_mound):
        """Merge with zero updated relationships is valid."""
        merge_result = MockMergeResult(updated_relationships=0)
        mock_mound.merge_duplicates = AsyncMock(return_value=merge_result)
        result = await handler.merge_duplicate_cluster(
            workspace_id="ws-1",
            cluster_id="c-1",
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["updated_relationships"] == 0

    @pytest.mark.asyncio
    async def test_auto_merge_large_duplicates_found(self, handler, mock_mound):
        """Auto-merge with large number of duplicates found."""
        mock_mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": False,
                "duplicates_found": 10000,
                "merges_performed": 10000,
                "details": [],
            }
        )
        result = await handler.auto_merge_exact_duplicates(
            workspace_id="ws-1",
            dry_run=False,
        )
        body = _body(result)
        assert body["duplicates_found"] == 10000
        assert body["merges_performed"] == 10000

    @pytest.mark.asyncio
    async def test_duplicate_cluster_with_empty_content_preview(self, handler, mock_mound):
        """Duplicate with empty content_preview is valid."""
        dup = MockDuplicate(node_id="d-empty", content_preview="")
        cluster = MockDuplicateCluster(cluster_id="c-empty", duplicates=[dup])
        mock_mound.find_duplicates = AsyncMock(return_value=[cluster])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert body["clusters"][0]["duplicates"][0]["content_preview"] == ""

    @pytest.mark.asyncio
    async def test_duplicate_cluster_with_zero_confidence(self, handler, mock_mound):
        """Duplicate with zero confidence is valid."""
        dup = MockDuplicate(node_id="d-no-conf", confidence=0.0)
        cluster = MockDuplicateCluster(cluster_id="c-low", duplicates=[dup])
        mock_mound.find_duplicates = AsyncMock(return_value=[cluster])
        result = await handler.get_duplicate_clusters(workspace_id="ws-1")
        body = _body(result)
        assert body["clusters"][0]["duplicates"][0]["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_handler_ctx_attribute_exists(self, handler):
        """Handler has ctx attribute as required by protocol."""
        assert hasattr(handler, "ctx")
        assert isinstance(handler.ctx, dict)

    @pytest.mark.asyncio
    async def test_handler_get_mound_returns_mound(self, handler, mock_mound):
        """_get_mound returns the mound instance."""
        assert handler._get_mound() is mock_mound

    @pytest.mark.asyncio
    async def test_handler_no_mound_get_mound_returns_none(self, handler_no_mound):
        """_get_mound returns None when no mound is configured."""
        assert handler_no_mound._get_mound() is None
