"""
Tests for Knowledge Mound Deduplication handler endpoints.

Tests dedup operations:
- GET /api/knowledge/mound/dedup/clusters - Find duplicate clusters
- GET /api/knowledge/mound/dedup/report - Generate dedup report
- POST /api/knowledge/mound/dedup/merge - Merge a duplicate cluster
- POST /api/knowledge/mound/dedup/auto-merge - Auto-merge exact duplicates
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.dedup import (
    DedupOperationsMixin,
)


@dataclass
class MockDuplicateNode:
    """Mock duplicate node for testing."""

    node_id: str = "node-dup-1"
    similarity: float = 0.95
    content_preview: str = "Some duplicate content..."
    tier: str = "warm"
    confidence: float = 0.8


@dataclass
class MockDuplicateCluster:
    """Mock duplicate cluster for testing."""

    cluster_id: str = "cluster-123"
    primary_node_id: str = "node-primary"
    avg_similarity: float = 0.93
    recommended_action: str = "merge"
    duplicates: list = field(default_factory=lambda: [MockDuplicateNode()])


@dataclass
class MockDedupReport:
    """Mock dedup report for testing."""

    workspace_id: str = "workspace-123"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_nodes_analyzed: int = 1000
    duplicate_clusters_found: int = 25
    estimated_reduction_percent: float = 5.2
    clusters: list = field(default_factory=list)


@dataclass
class MockMergeResult:
    """Mock merge result for testing."""

    kept_node_id: str = "node-primary"
    merged_node_ids: list = field(default_factory=lambda: ["node-dup-1", "node-dup-2"])
    archived_count: int = 2
    updated_relationships: int = 5


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self.find_duplicates = AsyncMock(
            return_value=[MockDuplicateCluster(), MockDuplicateCluster(cluster_id="c-456")]
        )
        self.generate_dedup_report = AsyncMock(return_value=MockDedupReport())
        self.merge_duplicates = AsyncMock(return_value=MockMergeResult())
        self.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": True,
                "duplicates_found": 10,
                "merges_performed": 0,
                "details": [],
            }
        )


class MockDedupHandler(DedupOperationsMixin):
    """Handler for testing DedupOperationsMixin."""

    def __init__(self):
        self.mound = MockMound()
        self.ctx = {"user_id": "test-user", "org_id": "test-org"}

    def _get_mound(self):
        return self.mound


class MockDedupHandlerNoMound(DedupOperationsMixin):
    """Handler with no mound available."""

    def __init__(self):
        self.ctx = {}

    def _get_mound(self):
        return None


def parse_json_response(result):
    """Parse JSON response from HandlerResult dataclass."""
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


@pytest.fixture
def handler():
    """Create test handler with mocked mound."""
    return MockDedupHandler()


@pytest.fixture
def handler_no_mound():
    """Create test handler without mound."""
    return MockDedupHandlerNoMound()


# Mock the decorators to bypass RBAC and rate limiting for tests
@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock RBAC and rate limit decorators."""
    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.dedup.require_permission",
            lambda perm: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.dedup.rate_limit",
            lambda **kwargs: lambda fn: fn,
        ),
    ):
        yield


class TestGetDuplicateClusters:
    """Tests for get_duplicate_clusters endpoint."""

    @pytest.mark.asyncio
    async def test_get_clusters_success(self, handler):
        """Test successful retrieval of duplicate clusters."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_duplicate_clusters(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert data["clusters_found"] == 2
        assert len(data["clusters"]) == 2

    @pytest.mark.asyncio
    async def test_get_clusters_with_threshold(self, handler):
        """Test retrieval with custom similarity threshold."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_duplicate_clusters(
            workspace_id="workspace-123",
            similarity_threshold=0.85,
        )

        data = parse_json_response(result)
        assert data["similarity_threshold"] == 0.85

        test_handler.mound.find_duplicates.assert_called_once_with(
            workspace_id="workspace-123",
            similarity_threshold=0.85,
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_get_clusters_with_limit(self, handler):
        """Test retrieval with custom limit."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_duplicate_clusters(
            workspace_id="workspace-123",
            limit=50,
        )

        data = parse_json_response(result)
        test_handler.mound.find_duplicates.assert_called_once_with(
            workspace_id="workspace-123",
            similarity_threshold=0.9,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_get_clusters_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_duplicate_clusters(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_clusters_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.get_duplicate_clusters(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_clusters_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockDedupHandler()
        test_handler.mound.find_duplicates = AsyncMock(side_effect=Exception("Cluster error"))

        result = await test_handler.get_duplicate_clusters(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestGetDedupReport:
    """Tests for get_dedup_report endpoint."""

    @pytest.mark.asyncio
    async def test_get_report_success(self, handler):
        """Test successful report generation."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_dedup_report(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert data["total_nodes_analyzed"] == 1000
        assert data["duplicate_clusters_found"] == 25
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_get_report_with_threshold(self, handler):
        """Test report with custom similarity threshold."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_dedup_report(
            workspace_id="workspace-123",
            similarity_threshold=0.8,
        )

        test_handler.mound.generate_dedup_report.assert_called_once_with(
            workspace_id="workspace-123",
            similarity_threshold=0.8,
        )

    @pytest.mark.asyncio
    async def test_get_report_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockDedupHandler()
        result = await test_handler.get_dedup_report(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_report_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.get_dedup_report(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_report_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockDedupHandler()
        test_handler.mound.generate_dedup_report = AsyncMock(side_effect=Exception("Report error"))

        result = await test_handler.get_dedup_report(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestMergeDuplicateCluster:
    """Tests for merge_duplicate_cluster endpoint."""

    @pytest.mark.asyncio
    async def test_merge_success(self, handler):
        """Test successful cluster merge."""
        test_handler = MockDedupHandler()
        result = await test_handler.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
        )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["kept_node_id"] == "node-primary"
        assert data["archived_count"] == 2
        assert len(data["merged_node_ids"]) == 2

    @pytest.mark.asyncio
    async def test_merge_with_primary_node(self, handler):
        """Test merge with specific primary node."""
        test_handler = MockDedupHandler()
        result = await test_handler.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
            primary_node_id="custom-primary",
        )

        test_handler.mound.merge_duplicates.assert_called_once_with(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
            primary_node_id="custom-primary",
            archive=True,
        )

    @pytest.mark.asyncio
    async def test_merge_with_delete(self, handler):
        """Test merge with delete instead of archive."""
        test_handler = MockDedupHandler()
        result = await test_handler.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
            archive=False,
        )

        test_handler.mound.merge_duplicates.assert_called_once_with(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
            primary_node_id=None,
            archive=False,
        )

    @pytest.mark.asyncio
    async def test_merge_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockDedupHandler()
        result = await test_handler.merge_duplicate_cluster(
            workspace_id="",
            cluster_id="cluster-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and cluster_id are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_merge_missing_cluster_id(self, handler):
        """Test fails without cluster_id."""
        test_handler = MockDedupHandler()
        result = await test_handler.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and cluster_id are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_merge_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_merge_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockDedupHandler()
        test_handler.mound.merge_duplicates = AsyncMock(side_effect=Exception("Merge error"))

        result = await test_handler.merge_duplicate_cluster(
            workspace_id="workspace-123",
            cluster_id="cluster-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestAutoMergeExactDuplicates:
    """Tests for auto_merge_exact_duplicates endpoint."""

    @pytest.mark.asyncio
    async def test_auto_merge_dry_run(self, handler):
        """Test auto-merge in dry run mode."""
        test_handler = MockDedupHandler()
        result = await test_handler.auto_merge_exact_duplicates(
            workspace_id="workspace-123",
            dry_run=True,
        )

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert data["dry_run"] is True
        assert data["duplicates_found"] == 10
        assert data["merges_performed"] == 0

    @pytest.mark.asyncio
    async def test_auto_merge_execute(self, handler):
        """Test auto-merge with actual execution."""
        test_handler = MockDedupHandler()
        test_handler.mound.auto_merge_exact_duplicates = AsyncMock(
            return_value={
                "dry_run": False,
                "duplicates_found": 10,
                "merges_performed": 5,
                "details": [{"merged": "node-1"}],
            }
        )

        result = await test_handler.auto_merge_exact_duplicates(
            workspace_id="workspace-123",
            dry_run=False,
        )

        data = parse_json_response(result)
        assert data["dry_run"] is False
        assert data["merges_performed"] == 5

    @pytest.mark.asyncio
    async def test_auto_merge_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockDedupHandler()
        result = await test_handler.auto_merge_exact_duplicates(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_auto_merge_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.auto_merge_exact_duplicates(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_auto_merge_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockDedupHandler()
        test_handler.mound.auto_merge_exact_duplicates = AsyncMock(
            side_effect=Exception("Auto-merge error")
        )

        result = await test_handler.auto_merge_exact_duplicates(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data

    @pytest.mark.asyncio
    async def test_auto_merge_default_dry_run(self, handler):
        """Test default dry_run is True."""
        test_handler = MockDedupHandler()
        await test_handler.auto_merge_exact_duplicates(workspace_id="workspace-123")

        test_handler.mound.auto_merge_exact_duplicates.assert_called_once_with(
            workspace_id="workspace-123",
            dry_run=True,
        )
