"""
Tests for Knowledge API resource.

Tests cover:
- KnowledgeAPI search and semantic query
- KnowledgeAPI CRUD operations (create, get, update, delete nodes)
- KnowledgeAPI statistics and analytics
- KnowledgeAPI health and dashboard
- KnowledgeAPI contradiction detection and resolution
- KnowledgeAPI knowledge extraction
- Dataclass models (KnowledgeNode, KnowledgeSearchResult, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.knowledge import (
    ContradictionResult,
    CoverageReport,
    KnowledgeAPI,
    KnowledgeNode,
    KnowledgeSearchResult,
    KnowledgeStats,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def knowledge_api(mock_client: AragoraClient) -> KnowledgeAPI:
    """Create a KnowledgeAPI with mock client."""
    return KnowledgeAPI(mock_client)


@pytest.fixture
def sample_timestamp() -> str:
    """Sample ISO timestamp for tests."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# KnowledgeNode Dataclass Tests
# ============================================================================


class TestKnowledgeNodeDataclass:
    """Tests for KnowledgeNode dataclass."""

    def test_node_minimal(self):
        """Test KnowledgeNode with required fields."""
        node = KnowledgeNode(
            id="node-123",
            content="Water boils at 100C",
            node_type="fact",
        )
        assert node.id == "node-123"
        assert node.content == "Water boils at 100C"
        assert node.node_type == "fact"
        assert node.confidence == 0.8  # default

    def test_node_full(self):
        """Test KnowledgeNode with all fields."""
        created = datetime.now(timezone.utc)
        node = KnowledgeNode(
            id="node-456",
            content="Python is interpreted",
            node_type="claim",
            confidence=0.95,
            workspace_id="ws-123",
            domain="programming",
            source_debate_id="deb-789",
            metadata={"source": "debate", "verified": True},
            created_at=created,
            updated_at=created,
        )
        assert node.confidence == 0.95
        assert node.workspace_id == "ws-123"
        assert node.domain == "programming"
        assert node.metadata["verified"] is True


# ============================================================================
# KnowledgeSearchResult Dataclass Tests
# ============================================================================


class TestKnowledgeSearchResultDataclass:
    """Tests for KnowledgeSearchResult dataclass."""

    def test_search_result_minimal(self):
        """Test KnowledgeSearchResult with required fields."""
        result = KnowledgeSearchResult(
            node_id="node-1",
            content="Test content",
            score=0.85,
            node_type="fact",
            confidence=0.9,
        )
        assert result.node_id == "node-1"
        assert result.score == 0.85
        assert result.domain is None
        assert result.metadata == {}

    def test_search_result_full(self):
        """Test KnowledgeSearchResult with all fields."""
        result = KnowledgeSearchResult(
            node_id="node-2",
            content="Full result",
            score=0.95,
            node_type="evidence",
            confidence=0.92,
            domain="science",
            metadata={"highlight": "matching text"},
        )
        assert result.domain == "science"
        assert result.metadata["highlight"] == "matching text"


# ============================================================================
# KnowledgeStats Dataclass Tests
# ============================================================================


class TestKnowledgeStatsDataclass:
    """Tests for KnowledgeStats dataclass."""

    def test_stats(self):
        """Test KnowledgeStats with all fields."""
        stats = KnowledgeStats(
            total_nodes=1000,
            nodes_by_type={"fact": 500, "claim": 300, "evidence": 200},
            nodes_by_tier={"fast": 100, "medium": 400, "slow": 500},
            total_relationships=2500,
            average_confidence=0.85,
            stale_nodes_count=50,
        )
        assert stats.total_nodes == 1000
        assert stats.nodes_by_type["fact"] == 500
        assert stats.average_confidence == 0.85


# ============================================================================
# CoverageReport Dataclass Tests
# ============================================================================


class TestCoverageReportDataclass:
    """Tests for CoverageReport dataclass."""

    def test_coverage_report(self):
        """Test CoverageReport with all fields."""
        report = CoverageReport(
            domains={"programming": 100, "science": 50, "business": 25},
            coverage_score=0.75,
            gaps=["finance", "legal"],
            recommendations=["Add more finance knowledge", "Verify legal claims"],
        )
        assert report.coverage_score == 0.75
        assert len(report.gaps) == 2
        assert "finance" in report.gaps


# ============================================================================
# ContradictionResult Dataclass Tests
# ============================================================================


class TestContradictionResultDataclass:
    """Tests for ContradictionResult dataclass."""

    def test_contradiction_minimal(self):
        """Test ContradictionResult with required fields."""
        contradiction = ContradictionResult(
            id="contra-1",
            node_a_id="node-1",
            node_b_id="node-2",
            contradiction_type="logical",
            severity="high",
            description="Nodes claim opposite facts",
        )
        assert contradiction.id == "contra-1"
        assert contradiction.severity == "high"
        assert contradiction.suggested_resolution is None

    def test_contradiction_full(self):
        """Test ContradictionResult with all fields."""
        contradiction = ContradictionResult(
            id="contra-2",
            node_a_id="node-3",
            node_b_id="node-4",
            contradiction_type="factual",
            severity="medium",
            description="Conflicting data values",
            suggested_resolution="Keep newer node",
        )
        assert contradiction.suggested_resolution == "Keep newer node"


# ============================================================================
# KnowledgeAPI.search() Tests
# ============================================================================


class TestKnowledgeAPISearch:
    """Tests for KnowledgeAPI.search() method."""

    def test_search_basic(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test basic search() call."""
        mock_client._get.return_value = {
            "results": [
                {
                    "node_id": "n1",
                    "content": "Result 1",
                    "score": 0.9,
                    "node_type": "fact",
                    "confidence": 0.85,
                },
                {
                    "node_id": "n2",
                    "content": "Result 2",
                    "score": 0.8,
                    "node_type": "claim",
                    "confidence": 0.75,
                },
            ]
        }

        results = knowledge_api.search("test query")

        assert len(results) == 2
        assert results[0].node_id == "n1"
        assert results[0].score == 0.9
        mock_client._get.assert_called_once()

    def test_search_with_filters(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test search() with all filters."""
        mock_client._get.return_value = {"results": []}

        knowledge_api.search(
            query="filtered query",
            limit=20,
            workspace_id="ws-123",
            domain="science",
            min_confidence=0.7,
        )

        call_args = mock_client._get.call_args
        params = call_args[0][1]
        assert params["query"] == "filtered query"
        assert params["limit"] == 20
        assert params["workspace_id"] == "ws-123"
        assert params["domain"] == "science"
        assert params["min_confidence"] == 0.7


class TestKnowledgeAPISearchAsync:
    """Tests for KnowledgeAPI.search_async() method."""

    @pytest.mark.asyncio
    async def test_search_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test search_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "results": [
                    {
                        "node_id": "async-n1",
                        "content": "Async result",
                        "score": 0.95,
                        "node_type": "fact",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        results = await knowledge_api.search_async("async query")

        assert len(results) == 1
        assert results[0].node_id == "async-n1"


# ============================================================================
# KnowledgeAPI.semantic_query() Tests
# ============================================================================


class TestKnowledgeAPISemanticQuery:
    """Tests for KnowledgeAPI.semantic_query() method."""

    def test_semantic_query_basic(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test semantic_query() basic call."""
        mock_client._post.return_value = {
            "results": [{"id": "n1", "content": "Semantic match", "similarity": 0.92}]
        }

        results = knowledge_api.semantic_query("What is water made of?")

        assert len(results) == 1
        assert results[0]["similarity"] == 0.92
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/query"

    def test_semantic_query_with_workspace(
        self, knowledge_api: KnowledgeAPI, mock_client: MagicMock
    ):
        """Test semantic_query() with workspace."""
        mock_client._post.return_value = {"results": []}

        knowledge_api.semantic_query("query", limit=5, workspace_id="ws-456")

        call_args = mock_client._post.call_args
        body = call_args[0][1]
        assert body["limit"] == 5
        assert body["workspace_id"] == "ws-456"


class TestKnowledgeAPISemanticQueryAsync:
    """Tests for KnowledgeAPI.semantic_query_async() method."""

    @pytest.mark.asyncio
    async def test_semantic_query_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test semantic_query_async() call."""
        mock_client._post_async = AsyncMock(return_value={"results": []})

        results = await knowledge_api.semantic_query_async("async semantic")

        assert results == []


# ============================================================================
# KnowledgeAPI CRUD Tests
# ============================================================================


class TestKnowledgeAPICreateNode:
    """Tests for KnowledgeAPI.create_node() method."""

    def test_create_node_basic(
        self, knowledge_api: KnowledgeAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test create_node() basic call."""
        mock_client._post.return_value = {
            "id": "new-node",
            "content": "New knowledge",
            "node_type": "fact",
            "confidence": 0.8,
            "created_at": sample_timestamp,
        }

        node = knowledge_api.create_node("New knowledge")

        assert node.id == "new-node"
        assert node.content == "New knowledge"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/nodes"

    def test_create_node_full(
        self, knowledge_api: KnowledgeAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test create_node() with all options."""
        mock_client._post.return_value = {
            "id": "full-node",
            "content": "Full node",
            "node_type": "evidence",
            "confidence": 0.95,
            "workspace_id": "ws-1",
            "domain": "science",
            "metadata": {"source": "research"},
            "created_at": sample_timestamp,
        }

        node = knowledge_api.create_node(
            content="Full node",
            node_type="evidence",
            confidence=0.95,
            workspace_id="ws-1",
            domain="science",
            metadata={"source": "research"},
        )

        assert node.confidence == 0.95
        assert node.domain == "science"


class TestKnowledgeAPICreateNodeAsync:
    """Tests for KnowledgeAPI.create_node_async() method."""

    @pytest.mark.asyncio
    async def test_create_node_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test create_node_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "id": "async-node",
                "content": "Async content",
                "node_type": "fact",
                "confidence": 0.8,
            }
        )

        node = await knowledge_api.create_node_async("Async content")

        assert node.id == "async-node"


class TestKnowledgeAPIGetNode:
    """Tests for KnowledgeAPI.get_node() method."""

    def test_get_node(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_node() retrieves a node."""
        mock_client._get.return_value = {
            "id": "get-node",
            "content": "Retrieved content",
            "node_type": "fact",
            "confidence": 0.85,
        }

        node = knowledge_api.get_node("get-node")

        assert node.id == "get-node"
        assert node.content == "Retrieved content"
        mock_client._get.assert_called_once_with("/api/v1/knowledge/get-node")


class TestKnowledgeAPIGetNodeAsync:
    """Tests for KnowledgeAPI.get_node_async() method."""

    @pytest.mark.asyncio
    async def test_get_node_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_node_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "id": "async-get",
                "content": "Async",
                "node_type": "claim",
                "confidence": 0.7,
            }
        )

        node = await knowledge_api.get_node_async("async-get")

        assert node.id == "async-get"


class TestKnowledgeAPIUpdateNode:
    """Tests for KnowledgeAPI.update_node() method."""

    def test_update_node_content(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test update_node() changes content."""
        mock_client._patch.return_value = {
            "id": "update-node",
            "content": "Updated content",
            "node_type": "fact",
            "confidence": 0.8,
        }

        node = knowledge_api.update_node("update-node", content="Updated content")

        assert node.content == "Updated content"
        call_args = mock_client._patch.call_args
        assert call_args[0][0] == "/api/v1/knowledge/update-node"
        body = call_args[0][1]
        assert body["content"] == "Updated content"

    def test_update_node_multiple_fields(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test update_node() with multiple fields."""
        mock_client._patch.return_value = {
            "id": "multi-update",
            "content": "New content",
            "node_type": "fact",
            "confidence": 0.95,
            "metadata": {"updated": True},
        }

        node = knowledge_api.update_node(
            "multi-update",
            content="New content",
            confidence=0.95,
            metadata={"updated": True},
        )

        call_args = mock_client._patch.call_args
        body = call_args[0][1]
        assert body["content"] == "New content"
        assert body["confidence"] == 0.95
        assert body["metadata"] == {"updated": True}


class TestKnowledgeAPIUpdateNodeAsync:
    """Tests for KnowledgeAPI.update_node_async() method."""

    @pytest.mark.asyncio
    async def test_update_node_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test update_node_async() call."""
        mock_client._patch_async = AsyncMock(
            return_value={
                "id": "async-update",
                "content": "Async updated",
                "node_type": "fact",
                "confidence": 0.9,
            }
        )

        node = await knowledge_api.update_node_async("async-update", confidence=0.9)

        assert node.confidence == 0.9


class TestKnowledgeAPIDeleteNode:
    """Tests for KnowledgeAPI.delete_node() method."""

    def test_delete_node_success(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test delete_node() returns True on success."""
        mock_client._delete.return_value = {"deleted": True}

        result = knowledge_api.delete_node("delete-node")

        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/knowledge/delete-node")

    def test_delete_node_not_found(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test delete_node() returns False when not found."""
        mock_client._delete.return_value = {"deleted": False}

        result = knowledge_api.delete_node("missing-node")

        assert result is False


class TestKnowledgeAPIDeleteNodeAsync:
    """Tests for KnowledgeAPI.delete_node_async() method."""

    @pytest.mark.asyncio
    async def test_delete_node_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test delete_node_async() call."""
        mock_client._delete_async = AsyncMock(return_value={"deleted": True})

        result = await knowledge_api.delete_node_async("async-delete")

        assert result is True


# ============================================================================
# KnowledgeAPI Statistics Tests
# ============================================================================


class TestKnowledgeAPIGetStats:
    """Tests for KnowledgeAPI.get_stats() method."""

    def test_get_stats_basic(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_stats() basic call."""
        mock_client._get.return_value = {
            "total_nodes": 500,
            "nodes_by_type": {"fact": 300, "claim": 200},
            "nodes_by_tier": {"fast": 100, "slow": 400},
            "total_relationships": 1000,
            "average_confidence": 0.82,
            "stale_nodes_count": 25,
        }

        stats = knowledge_api.get_stats()

        assert stats.total_nodes == 500
        assert stats.average_confidence == 0.82
        assert stats.stale_nodes_count == 25

    def test_get_stats_with_workspace(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_stats() with workspace filter."""
        mock_client._get.return_value = {
            "total_nodes": 100,
            "nodes_by_type": {},
            "nodes_by_tier": {},
            "total_relationships": 50,
            "average_confidence": 0.9,
            "stale_nodes_count": 5,
        }

        knowledge_api.get_stats(workspace_id="ws-filter")

        call_args = mock_client._get.call_args
        params = call_args[0][1]
        assert params["workspace_id"] == "ws-filter"


class TestKnowledgeAPIGetStatsAsync:
    """Tests for KnowledgeAPI.get_stats_async() method."""

    @pytest.mark.asyncio
    async def test_get_stats_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_stats_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "total_nodes": 200,
                "nodes_by_type": {},
                "nodes_by_tier": {},
                "total_relationships": 100,
                "average_confidence": 0.85,
                "stale_nodes_count": 10,
            }
        )

        stats = await knowledge_api.get_stats_async()

        assert stats.total_nodes == 200


class TestKnowledgeAPIGetCoverageAnalytics:
    """Tests for KnowledgeAPI.get_coverage_analytics() method."""

    def test_get_coverage(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_coverage_analytics() call."""
        mock_client._get.return_value = {
            "domains": {"programming": 100, "science": 50},
            "coverage_score": 0.65,
            "gaps": ["finance"],
            "recommendations": ["Add finance knowledge"],
        }

        report = knowledge_api.get_coverage_analytics()

        assert report.coverage_score == 0.65
        assert "finance" in report.gaps


class TestKnowledgeAPIGetUsageAnalytics:
    """Tests for KnowledgeAPI.get_usage_analytics() method."""

    def test_get_usage(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_usage_analytics() call."""
        mock_client._get.return_value = {
            "searches_per_day": 500,
            "nodes_created": 50,
            "nodes_deleted": 5,
        }

        result = knowledge_api.get_usage_analytics(days=7)

        assert result["searches_per_day"] == 500
        call_args = mock_client._get.call_args
        params = call_args[0][1]
        assert params["days"] == 7


class TestKnowledgeAPIGetQualitySnapshot:
    """Tests for KnowledgeAPI.get_quality_snapshot() method."""

    def test_get_quality(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_quality_snapshot() call."""
        mock_client._get.return_value = {
            "quality_score": 0.88,
            "completeness": 0.92,
            "accuracy": 0.85,
        }

        result = knowledge_api.get_quality_snapshot()

        assert result["quality_score"] == 0.88


# ============================================================================
# KnowledgeAPI Health and Dashboard Tests
# ============================================================================


class TestKnowledgeAPIGetHealth:
    """Tests for KnowledgeAPI.get_health() method."""

    def test_get_health(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_health() call."""
        mock_client._get.return_value = {
            "status": "healthy",
            "components": {"postgres": "up", "redis": "up"},
        }

        result = knowledge_api.get_health()

        assert result["status"] == "healthy"
        mock_client._get.assert_called_once_with("/api/v1/knowledge/mound/dashboard/health")


class TestKnowledgeAPIGetHealthAsync:
    """Tests for KnowledgeAPI.get_health_async() method."""

    @pytest.mark.asyncio
    async def test_get_health_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_health_async() call."""
        mock_client._get_async = AsyncMock(return_value={"status": "healthy"})

        result = await knowledge_api.get_health_async()

        assert result["status"] == "healthy"


class TestKnowledgeAPIGetMetrics:
    """Tests for KnowledgeAPI.get_metrics() method."""

    def test_get_metrics(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_metrics() call."""
        mock_client._get.return_value = {
            "query_latency_p50": 15,
            "query_latency_p99": 150,
            "cache_hit_rate": 0.85,
        }

        result = knowledge_api.get_metrics()

        assert result["cache_hit_rate"] == 0.85
        mock_client._get.assert_called_once_with("/api/v1/knowledge/mound/dashboard/metrics")


class TestKnowledgeAPIGetAdapters:
    """Tests for KnowledgeAPI.get_adapters() method."""

    def test_get_adapters(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_adapters() call."""
        mock_client._get.return_value = {
            "adapters": [
                {"name": "continuum", "status": "active", "items": 100},
                {"name": "consensus", "status": "active", "items": 50},
            ]
        }

        result = knowledge_api.get_adapters()

        assert len(result) == 2
        assert result[0]["name"] == "continuum"


class TestKnowledgeAPIGetAdaptersAsync:
    """Tests for KnowledgeAPI.get_adapters_async() method."""

    @pytest.mark.asyncio
    async def test_get_adapters_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test get_adapters_async() call."""
        mock_client._get_async = AsyncMock(return_value={"adapters": [{"name": "async-adapter"}]})

        result = await knowledge_api.get_adapters_async()

        assert len(result) == 1


# ============================================================================
# KnowledgeAPI Contradiction Detection Tests
# ============================================================================


class TestKnowledgeAPIDetectContradictions:
    """Tests for KnowledgeAPI.detect_contradictions() method."""

    def test_detect_contradictions_basic(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test detect_contradictions() basic call."""
        mock_client._post.return_value = {
            "contradictions": [
                {
                    "id": "contra-1",
                    "node_a_id": "n1",
                    "node_b_id": "n2",
                    "contradiction_type": "logical",
                    "severity": "high",
                    "description": "Opposite claims",
                },
            ]
        }

        results = knowledge_api.detect_contradictions()

        assert len(results) == 1
        assert results[0].severity == "high"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/contradictions/detect"

    def test_detect_contradictions_with_options(
        self, knowledge_api: KnowledgeAPI, mock_client: MagicMock
    ):
        """Test detect_contradictions() with options."""
        mock_client._post.return_value = {"contradictions": []}

        knowledge_api.detect_contradictions(workspace_id="ws-123", threshold=0.9)

        call_args = mock_client._post.call_args
        body = call_args[0][1]
        assert body["workspace_id"] == "ws-123"
        assert body["threshold"] == 0.9


class TestKnowledgeAPIDetectContradictionsAsync:
    """Tests for KnowledgeAPI.detect_contradictions_async() method."""

    @pytest.mark.asyncio
    async def test_detect_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test detect_contradictions_async() call."""
        mock_client._post_async = AsyncMock(return_value={"contradictions": []})

        results = await knowledge_api.detect_contradictions_async()

        assert results == []


class TestKnowledgeAPIResolveContradiction:
    """Tests for KnowledgeAPI.resolve_contradiction() method."""

    def test_resolve_keep_a(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test resolve_contradiction() with keep_a strategy."""
        mock_client._post.return_value = {
            "resolved": True,
            "resolution": "keep_a",
            "kept_node": "n1",
        }

        result = knowledge_api.resolve_contradiction(
            "contra-1", resolution="keep_a", keep_node_id="n1"
        )

        assert result["resolved"] is True
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/contradictions/contra-1/resolve"
        body = call_args[0][1]
        assert body["resolution"] == "keep_a"
        assert body["keep_node_id"] == "n1"

    def test_resolve_merge(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test resolve_contradiction() with merge strategy."""
        mock_client._post.return_value = {
            "resolved": True,
            "resolution": "merge",
            "new_node_id": "merged-123",
        }

        result = knowledge_api.resolve_contradiction("contra-2", resolution="merge")

        assert result["new_node_id"] == "merged-123"


class TestKnowledgeAPIResolveContradictionAsync:
    """Tests for KnowledgeAPI.resolve_contradiction_async() method."""

    @pytest.mark.asyncio
    async def test_resolve_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test resolve_contradiction_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={"resolved": True, "resolution": "deprecate_both"}
        )

        result = await knowledge_api.resolve_contradiction_async(
            "contra-async", resolution="deprecate_both"
        )

        assert result["resolved"] is True


# ============================================================================
# KnowledgeAPI Knowledge Extraction Tests
# ============================================================================


class TestKnowledgeAPIExtractFromDebate:
    """Tests for KnowledgeAPI.extract_from_debate() method."""

    def test_extract_basic(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test extract_from_debate() basic call."""
        mock_client._post.return_value = {
            "debate_id": "deb-123",
            "extracted_count": 5,
            "extraction_ids": ["ext-1", "ext-2", "ext-3", "ext-4", "ext-5"],
        }

        result = knowledge_api.extract_from_debate("deb-123")

        assert result["extracted_count"] == 5
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/extraction/debate"
        body = call_args[0][1]
        assert body["debate_id"] == "deb-123"
        assert body["auto_promote"] is False

    def test_extract_with_auto_promote(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test extract_from_debate() with auto_promote."""
        mock_client._post.return_value = {
            "debate_id": "deb-456",
            "extracted_count": 3,
            "promoted_count": 3,
        }

        result = knowledge_api.extract_from_debate("deb-456", auto_promote=True)

        assert result["promoted_count"] == 3
        call_args = mock_client._post.call_args
        body = call_args[0][1]
        assert body["auto_promote"] is True


class TestKnowledgeAPIExtractFromDebateAsync:
    """Tests for KnowledgeAPI.extract_from_debate_async() method."""

    @pytest.mark.asyncio
    async def test_extract_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test extract_from_debate_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={"debate_id": "deb-async", "extracted_count": 2}
        )

        result = await knowledge_api.extract_from_debate_async("deb-async")

        assert result["extracted_count"] == 2


class TestKnowledgeAPIPromoteExtracted:
    """Tests for KnowledgeAPI.promote_extracted() method."""

    def test_promote(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test promote_extracted() call."""
        mock_client._post.return_value = {
            "promoted_count": 3,
            "promoted_ids": ["n1", "n2", "n3"],
        }

        result = knowledge_api.promote_extracted(["ext-1", "ext-2", "ext-3"])

        assert result["promoted_count"] == 3
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/knowledge/mound/extraction/promote"
        body = call_args[0][1]
        assert body["extraction_ids"] == ["ext-1", "ext-2", "ext-3"]


class TestKnowledgeAPIPromoteExtractedAsync:
    """Tests for KnowledgeAPI.promote_extracted_async() method."""

    @pytest.mark.asyncio
    async def test_promote_async(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test promote_extracted_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={"promoted_count": 1, "promoted_ids": ["n-async"]}
        )

        result = await knowledge_api.promote_extracted_async(["ext-async"])

        assert result["promoted_count"] == 1


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestKnowledgeAPIIntegration:
    """Integration-like tests for KnowledgeAPI."""

    def test_full_knowledge_workflow(
        self, knowledge_api: KnowledgeAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test full knowledge workflow: create -> search -> update -> delete."""
        # Create node
        mock_client._post.return_value = {
            "id": "new-knowledge",
            "content": "Python is a programming language",
            "node_type": "fact",
            "confidence": 0.95,
            "domain": "programming",
            "created_at": sample_timestamp,
        }
        node = knowledge_api.create_node(
            content="Python is a programming language",
            confidence=0.95,
            domain="programming",
        )
        assert node.id == "new-knowledge"

        # Search for it
        mock_client._get.return_value = {
            "results": [
                {
                    "node_id": "new-knowledge",
                    "content": "Python is a programming language",
                    "score": 0.98,
                    "node_type": "fact",
                    "confidence": 0.95,
                }
            ]
        }
        results = knowledge_api.search("Python programming")
        assert len(results) == 1
        assert results[0].node_id == "new-knowledge"

        # Update confidence
        mock_client._patch.return_value = {
            "id": "new-knowledge",
            "content": "Python is a programming language",
            "node_type": "fact",
            "confidence": 0.99,
        }
        updated = knowledge_api.update_node("new-knowledge", confidence=0.99)
        assert updated.confidence == 0.99

        # Delete
        mock_client._delete.return_value = {"deleted": True}
        deleted = knowledge_api.delete_node("new-knowledge")
        assert deleted is True

    def test_contradiction_workflow(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test contradiction detection and resolution workflow."""
        # Detect contradictions
        mock_client._post.return_value = {
            "contradictions": [
                {
                    "id": "contra-workflow",
                    "node_a_id": "fact-1",
                    "node_b_id": "fact-2",
                    "contradiction_type": "factual",
                    "severity": "high",
                    "description": "Conflicting facts about Python",
                    "suggested_resolution": "Keep newer",
                }
            ]
        }
        contradictions = knowledge_api.detect_contradictions()
        assert len(contradictions) == 1
        assert contradictions[0].severity == "high"

        # Resolve contradiction
        mock_client._post.return_value = {
            "resolved": True,
            "resolution": "keep_a",
            "kept_node": "fact-1",
            "deprecated_node": "fact-2",
        }
        resolution = knowledge_api.resolve_contradiction(
            "contra-workflow",
            resolution="keep_a",
            keep_node_id="fact-1",
        )
        assert resolution["resolved"] is True
        assert resolution["kept_node"] == "fact-1"

    def test_extraction_workflow(self, knowledge_api: KnowledgeAPI, mock_client: MagicMock):
        """Test knowledge extraction from debate workflow."""
        # Extract from debate
        mock_client._post.return_value = {
            "debate_id": "deb-extract",
            "extracted_count": 5,
            "extraction_ids": ["ext-1", "ext-2", "ext-3", "ext-4", "ext-5"],
        }
        extraction = knowledge_api.extract_from_debate("deb-extract")
        assert extraction["extracted_count"] == 5

        # Promote selected extractions
        mock_client._post.return_value = {
            "promoted_count": 3,
            "promoted_ids": ["node-1", "node-2", "node-3"],
        }
        promotion = knowledge_api.promote_extracted(["ext-1", "ext-2", "ext-3"])
        assert promotion["promoted_count"] == 3
