"""Tests for Memory namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# ===========================================================================
# Core CRUD Operations Tests
# ===========================================================================


class TestMemoryStore:
    """Tests for memory store operations."""

    def test_store_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"stored": True, "tier": "fast"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.store("my-key", {"data": "value"})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory",
                json={"key": "my-key", "value": {"data": "value"}},
            )
            assert result["stored"] is True
            client.close()

    def test_store_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"stored": True, "tier": "slow"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.store(
                "my-key",
                {"data": "value"},
                tier="slow",
                importance=0.8,
                tags=["important", "config"],
                ttl_seconds=3600,
                metadata={"source": "test"},
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory",
                json={
                    "key": "my-key",
                    "value": {"data": "value"},
                    "tier": "slow",
                    "importance": 0.8,
                    "tags": ["important", "config"],
                    "ttl_seconds": 3600,
                    "metadata": {"source": "test"},
                },
            )
            assert result["tier"] == "slow"
            client.close()


class TestMemoryRetrieve:
    """Tests for memory retrieve operations."""

    def test_retrieve_by_key(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"value": {"data": "test"}, "tier": "fast"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.retrieve("my-key")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/my-key",
                params={},
            )
            assert result["value"]["data"] == "test"
            client.close()

    def test_retrieve_with_tier(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"value": "data", "tier": "slow"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.retrieve("my-key", tier="slow")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/my-key",
                params={"tier": "slow"},
            )
            client.close()

    def test_retrieve_url_encodes_key(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"value": "data"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.retrieve("key/with/slashes")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/key%2Fwith%2Fslashes",
                params={},
            )
            client.close()


class TestMemoryUpdate:
    """Tests for memory update operations."""

    def test_update_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"updated": True, "tier": "fast"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.update("my-key", {"new": "value"})
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/memory/my-key",
                json={"value": {"new": "value"}},
            )
            assert result["updated"] is True
            client.close()

    def test_update_with_merge(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"updated": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.update(
                "my-key",
                {"field": "value"},
                merge=True,
                tags=["updated"],
            )
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/memory/my-key",
                json={"value": {"field": "value"}, "merge": True, "tags": ["updated"]},
            )
            client.close()


class TestMemoryDelete:
    """Tests for memory delete operations."""

    def test_delete_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.delete("my-key")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/memory/my-key",
                params={},
            )
            assert result["deleted"] is True
            client.close()

    def test_delete_from_tier(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.delete("my-key", tier="glacial")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/memory/my-key",
                params={"tier": "glacial"},
            )
            client.close()


# ===========================================================================
# Search and Query Tests
# ===========================================================================


class TestMemorySearch:
    """Tests for memory search operations."""

    def test_search_with_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"id": "r1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.search("consensus algorithm")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/search",
                params={
                    "q": "consensus algorithm",
                    "limit": 20,
                    "min_importance": 0.0,
                    "sort": "relevance",
                },
            )
            assert result["total"] == 1
            client.close()

    def test_search_with_tier_filter_and_sort(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.search(
                "debate outcomes",
                tier=["slow", "glacial"],
                limit=10,
                min_importance=0.8,
                sort="timestamp",
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/search",
                params={
                    "q": "debate outcomes",
                    "limit": 10,
                    "min_importance": 0.8,
                    "sort": "timestamp",
                    "tier": "slow,glacial",
                },
            )
            client.close()


class TestMemoryQuery:
    """Tests for advanced memory query operations."""

    def test_query_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.query()
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/query",
                json={"limit": 20, "offset": 0, "include_metadata": True},
            )
            assert result["total"] == 0
            client.close()

    def test_query_with_filter_and_sort(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [{"id": "e1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.query(
                filter={"tags": ["important"], "tier": "slow"},
                sort_by="created_at",
                sort_order="asc",
                limit=50,
                offset=10,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/query",
                json={
                    "filter": {"tags": ["important"], "tier": "slow"},
                    "sort_by": "created_at",
                    "sort_order": "asc",
                    "limit": 50,
                    "offset": 10,
                    "include_metadata": True,
                },
            )
            client.close()


class TestSemanticSearch:
    """Tests for semantic search operations."""

    def test_semantic_search_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.semantic_search("rate limiting patterns")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/semantic-search",
                json={
                    "query": "rate limiting patterns",
                    "limit": 10,
                    "min_similarity": 0.7,
                    "include_embeddings": False,
                },
            )
            client.close()

    def test_semantic_search_with_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [{"similarity": 0.95}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.semantic_search(
                "consensus mechanisms",
                tiers=["slow", "glacial"],
                limit=5,
                min_similarity=0.85,
                include_embeddings=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/semantic-search",
                json={
                    "query": "consensus mechanisms",
                    "tiers": ["slow", "glacial"],
                    "limit": 5,
                    "min_similarity": 0.85,
                    "include_embeddings": True,
                },
            )
            client.close()


# ===========================================================================
# Statistics and Monitoring Tests
# ===========================================================================


class TestMemoryStats:
    """Tests for tier stats, archive stats, pressure, and tier listing."""

    def test_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_entries": 1000, "total_size_bytes": 5000000}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/stats")
            assert result["total_entries"] == 1000
            client.close()

    def test_get_tier_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"fast": {"count": 120}, "medium": {"count": 45}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_tier_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/tier-stats")
            assert result["fast"]["count"] == 120
            client.close()

    def test_get_archive_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_archived": 500, "size_mb": 12.3}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_archive_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/archive-stats")
            assert result["total_archived"] == 500
            client.close()

    def test_get_pressure(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"utilization": 0.72, "pressure": "moderate"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_pressure()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
            assert result["pressure"] == "moderate"
            client.close()

    def test_get_analytics(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": [{"timestamp": "2024-01-01", "entries": 100}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_analytics(
                start_time="2024-01-01T00:00:00Z",
                end_time="2024-01-02T00:00:00Z",
                granularity="day",
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/analytics",
                params={
                    "granularity": "day",
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-01-02T00:00:00Z",
                },
            )
            client.close()


# ===========================================================================
# Tier Operations Tests
# ===========================================================================


class TestTierOperations:
    """Tests for tier management operations."""

    def test_list_tiers(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tiers": [
                    {"name": "fast", "ttl": 60},
                    {"name": "glacial", "ttl": 604800},
                ],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.list_tiers()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/tiers")
            assert len(result["tiers"]) == 2
            client.close()

    def test_tiers_alias(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"tiers": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.tiers()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/tiers")
            client.close()

    def test_get_tier(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [{"key": "k1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_tier("fast", limit=100, offset=50)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/tier/fast",
                params={"limit": 100, "offset": 50},
            )
            assert result["total"] == 1
            client.close()

    def test_move_tier(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "moved": True,
                "key": "my-key",
                "from_tier": "fast",
                "to_tier": "slow",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.move_tier("my-key", "fast", "slow")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/my-key/move",
                json={"from_tier": "fast", "to_tier": "slow"},
            )
            assert result["moved"] is True
            client.close()

    def test_promote(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"promoted": True, "new_tier": "fast"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.promote("my-key", reason="high access frequency")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/my-key/promote",
                json={"reason": "high access frequency"},
            )
            assert result["promoted"] is True
            client.close()

    def test_demote(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"demoted": True, "new_tier": "glacial"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.demote("my-key", reason="low importance")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/my-key/demote",
                json={"reason": "low importance"},
            )
            assert result["demoted"] is True
            client.close()


# ===========================================================================
# Continuum Operations Tests
# ===========================================================================


class TestContinuumOperations:
    """Tests for continuum memory operations."""

    def test_store_to_continuum(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "entry-123", "tier": "medium"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.store_to_continuum(
                "Important debate outcome about rate limiting",
                importance=0.9,
                tags=["debate", "rate-limiting"],
                source="debate-456",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/continuum",
                json={
                    "content": "Important debate outcome about rate limiting",
                    "importance": 0.9,
                    "tags": ["debate", "rate-limiting"],
                    "source": "debate-456",
                },
            )
            assert result["id"] == "entry-123"
            client.close()

    def test_retrieve_continuum_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"memories": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.retrieve_continuum()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/continuum/retrieve",
                params={"query": "", "limit": 10, "min_importance": 0.0},
            )
            assert result["total"] == 0
            client.close()

    def test_retrieve_continuum_with_query_and_tiers(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"memories": [{"id": "m1", "tier": "fast"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.retrieve_continuum(
                query="rate limiter",
                tiers=["fast", "medium"],
                limit=5,
                min_importance=0.5,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/continuum/retrieve",
                params={
                    "query": "rate limiter",
                    "limit": 5,
                    "min_importance": 0.5,
                    "tiers": "fast,medium",
                },
            )
            assert result["memories"][0]["tier"] == "fast"
            client.close()

    def test_retrieve_from_continuum_alias(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"memories": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.retrieve_from_continuum("test query", tiers=["slow"])
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/continuum/retrieve",
                params={
                    "query": "test query",
                    "limit": 10,
                    "min_importance": 0.0,
                    "tiers": "slow",
                },
            )
            client.close()

    def test_continuum_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_entries": 500,
                "by_tier": {"fast": 100, "slow": 400},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.continuum_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/continuum/stats")
            assert result["total_entries"] == 500
            client.close()

    def test_consolidate(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "entries_archived": 50}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.consolidate()
            mock_request.assert_called_once_with("POST", "/api/v1/memory/consolidate", json={})
            assert result["success"] is True
            client.close()


# ===========================================================================
# Critique Operations Tests
# ===========================================================================


class TestCritiqueOperations:
    """Tests for critique store operations."""

    def test_list_critiques_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"critiques": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.list_critiques()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_critiques_with_agent_filter(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"critiques": [{"id": "c1", "agent": "claude"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.list_critiques(agent="claude", limit=5, offset=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 5, "offset": 10, "agent": "claude"},
            )
            assert result["critiques"][0]["agent"] == "claude"
            client.close()

    def test_critiques_alias(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"critiques": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.critiques(agent="gemini", limit=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 10, "offset": 0, "agent": "gemini"},
            )
            client.close()

    def test_store_critique(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "critique-123"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.store_critique(
                "The proposal lacks consideration for edge cases",
                agent="claude",
                debate_id="debate-456",
                target_agent="gpt4",
                score=0.85,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/critiques",
                json={
                    "critique": "The proposal lacks consideration for edge cases",
                    "agent": "claude",
                    "debate_id": "debate-456",
                    "target_agent": "gpt4",
                    "score": 0.85,
                },
            )
            assert result["id"] == "critique-123"
            client.close()


# ===========================================================================
# Context Management Tests
# ===========================================================================


class TestContextManagement:
    """Tests for context management operations."""

    def test_get_context_default(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"context_id": "ctx-123", "data": {"user_id": "u1"}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_context()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/context",
                params={},
            )
            assert result["context_id"] == "ctx-123"
            client.close()

    def test_get_context_with_id(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"context_id": "ctx-456", "data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.get_context(context_id="ctx-456")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/context",
                params={"context_id": "ctx-456"},
            )
            client.close()

    def test_set_context(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"context_id": "ctx-789", "data": {"session": "s1"}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.set_context(
                {"session": "s1", "user": "u1"},
                context_id="ctx-789",
                ttl_seconds=3600,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/context",
                json={
                    "data": {"session": "s1", "user": "u1"},
                    "context_id": "ctx-789",
                    "ttl_seconds": 3600,
                },
            )
            client.close()

    def test_clear_context(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cleared": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.clear_context(context_id="ctx-123")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/memory/context",
                params={"context_id": "ctx-123"},
            )
            assert result["cleared"] is True
            client.close()


# ===========================================================================
# Cross-Debate Memory Tests
# ===========================================================================


class TestCrossDebateMemory:
    """Tests for cross-debate institutional knowledge operations."""

    def test_get_cross_debate(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [{"id": "e1", "topic": "rate-limiting"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.get_cross_debate(
                topic="rate-limiting",
                limit=5,
                min_relevance=0.7,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/cross-debate",
                params={"limit": 5, "min_relevance": 0.7, "topic": "rate-limiting"},
            )
            assert result["entries"][0]["topic"] == "rate-limiting"
            client.close()

    def test_store_cross_debate(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "xd-123"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.store_cross_debate(
                "Rate limiting should use token bucket algorithm",
                debate_id="debate-789",
                topic="rate-limiting",
                conclusion="consensus",
                confidence=0.95,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/cross-debate",
                json={
                    "content": "Rate limiting should use token bucket algorithm",
                    "debate_id": "debate-789",
                    "topic": "rate-limiting",
                    "conclusion": "consensus",
                    "confidence": 0.95,
                },
            )
            assert result["id"] == "xd-123"
            client.close()

    def test_inject_institutional(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"injected_count": 3}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.inject_institutional(
                "debate-999",
                topic="architecture",
                max_entries=5,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/cross-debate/inject",
                json={
                    "debate_id": "debate-999",
                    "max_entries": 5,
                    "topic": "architecture",
                },
            )
            assert result["injected_count"] == 3
            client.close()


# ===========================================================================
# Export/Import Tests
# ===========================================================================


class TestExportImport:
    """Tests for memory export/import operations."""

    def test_export_memory(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "https://storage/backup.json"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.export_memory(
                tiers=["slow", "glacial"],
                tags=["important"],
                format="json",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/export",
                json={
                    "format": "json",
                    "include_metadata": True,
                    "tiers": ["slow", "glacial"],
                    "tags": ["important"],
                },
            )
            client.close()

    def test_import_memory(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"imported": 100, "skipped": 5}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.import_memory(
                [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}],
                overwrite=True,
                target_tier="slow",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/import",
                json={
                    "data": [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}],
                    "overwrite": True,
                    "target_tier": "slow",
                },
            )
            assert result["imported"] == 100
            client.close()


# ===========================================================================
# Snapshot Tests
# ===========================================================================


class TestSnapshots:
    """Tests for snapshot management operations."""

    def test_create_snapshot(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "snap-123", "created_at": "2024-01-01T00:00:00Z"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.create_snapshot(
                name="pre-migration",
                description="Snapshot before schema migration",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/snapshots",
                json={"name": "pre-migration", "description": "Snapshot before schema migration"},
            )
            assert result["id"] == "snap-123"
            client.close()

    def test_list_snapshots(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"snapshots": [{"id": "s1"}, {"id": "s2"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.list_snapshots(limit=10, offset=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/snapshots",
                params={"limit": 10, "offset": 5},
            )
            assert len(result["snapshots"]) == 2
            client.close()

    def test_restore_snapshot(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"restored": True, "entries_restored": 500}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.restore_snapshot("snap-123", overwrite=True)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/snapshots/snap-123/restore",
                json={"overwrite": True},
            )
            assert result["restored"] is True
            client.close()

    def test_delete_snapshot(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.delete_snapshot("snap-123")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/memory/snapshots/snap-123",
            )
            assert result["deleted"] is True
            client.close()


# ===========================================================================
# Maintenance Operations Tests
# ===========================================================================


class TestMaintenanceOperations:
    """Tests for maintenance operations."""

    def test_prune(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"pruned_count": 150, "freed_bytes": 1024000}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.prune(
                older_than_days=30,
                min_importance=0.1,
                tiers=["fast", "medium"],
                dry_run=False,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/prune",
                json={
                    "dry_run": False,
                    "older_than_days": 30,
                    "min_importance": 0.1,
                    "tiers": ["fast", "medium"],
                },
            )
            assert result["pruned_count"] == 150
            client.close()

    def test_compact(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "compacted": True,
                "entries_merged": 50,
                "space_saved_bytes": 512000,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.compact(tier="slow", merge_threshold=0.85)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/compact",
                json={"merge_threshold": 0.85, "tier": "slow"},
            )
            assert result["compacted"] is True
            client.close()

    def test_sync(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "synced": True,
                "entries_synced": 200,
                "conflicts_resolved": 5,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.sync(
                target="all",
                conflict_resolution="merge",
                tiers=["slow", "glacial"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/sync",
                json={
                    "conflict_resolution": "merge",
                    "target": "all",
                    "tiers": ["slow", "glacial"],
                },
            )
            assert result["synced"] is True
            client.close()

    def test_vacuum(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"vacuumed": True, "space_reclaimed_bytes": 2048000}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.vacuum()
            mock_request.assert_called_once_with("POST", "/api/v1/memory/vacuum", json={})
            assert result["vacuumed"] is True
            client.close()

    def test_rebuild_index(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rebuilt": True, "entries_indexed": 1000}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = client.memory.rebuild_index(tier="slow")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/rebuild-index",
                json={"tier": "slow"},
            )
            assert result["rebuilt"] is True
            client.close()


# ===========================================================================
# Async Memory Tests
# ===========================================================================


class TestAsyncMemory:
    """Tests for async memory methods."""

    @pytest.mark.asyncio
    async def test_store(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"stored": True, "tier": "fast"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.store("key", {"value": 1}, tier="fast")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory",
                json={"key": "key", "value": {"value": 1}, "tier": "fast"},
            )
            assert result["stored"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_retrieve(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"value": "data", "tier": "fast"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.retrieve("key")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/key",
                params={},
            )
            assert result["value"] == "data"
            await client.close()

    @pytest.mark.asyncio
    async def test_update(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"updated": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.update("key", {"new": "value"}, merge=True)
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/memory/key",
                json={"value": {"new": "value"}, "merge": True},
            )
            assert result["updated"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.delete("key")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/memory/key",
                params={},
            )
            assert result["deleted"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.search("fallback strategy", sort="timestamp")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/search",
                params={
                    "q": "fallback strategy",
                    "limit": 20,
                    "min_importance": 0.0,
                    "sort": "timestamp",
                },
            )
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_query(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.query(filter={"tags": ["test"]})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/query",
                json={
                    "filter": {"tags": ["test"]},
                    "limit": 20,
                    "offset": 0,
                    "include_metadata": True,
                },
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_entries": 1000}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/stats")
            assert result["total_entries"] == 1000
            await client.close()

    @pytest.mark.asyncio
    async def test_get_pressure(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"utilization": 0.95, "pressure": "high"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.get_pressure()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
            assert result["pressure"] == "high"
            await client.close()

    @pytest.mark.asyncio
    async def test_retrieve_continuum(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"memories": [{"id": "m1"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.retrieve_continuum(
                query="design patterns",
                tiers=["fast"],
                limit=3,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/continuum/retrieve",
                params={
                    "query": "design patterns",
                    "limit": 3,
                    "min_importance": 0.0,
                    "tiers": "fast",
                },
            )
            assert len(result["memories"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_list_critiques(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"critiques": [{"id": "c1"}], "total": 1}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.list_critiques(agent="gemini", limit=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 10, "offset": 0, "agent": "gemini"},
            )
            assert result["total"] == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_get_context(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"context_id": "ctx-123", "data": {}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.get_context()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/context",
                params={},
            )
            assert result["context_id"] == "ctx-123"
            await client.close()

    @pytest.mark.asyncio
    async def test_set_context(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"context_id": "ctx-123", "data": {"key": "value"}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.set_context({"key": "value"}, ttl_seconds=3600)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/context",
                json={"data": {"key": "value"}, "ttl_seconds": 3600},
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_prune(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"pruned_count": 100}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.prune(older_than_days=30)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/prune",
                json={"dry_run": False, "older_than_days": 30},
            )
            assert result["pruned_count"] == 100
            await client.close()

    @pytest.mark.asyncio
    async def test_compact(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"compacted": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.compact(tier="slow")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/compact",
                json={"merge_threshold": 0.9, "tier": "slow"},
            )
            assert result["compacted"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_sync(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"synced": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.sync(conflict_resolution="merge")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/sync",
                json={"conflict_resolution": "merge"},
            )
            assert result["synced"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_get_tier(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.get_tier("fast")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/tier/fast",
                params={"limit": 50, "offset": 0},
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_move_tier(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"moved": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.move_tier("key", "fast", "slow")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/key/move",
                json={"from_tier": "fast", "to_tier": "slow"},
            )
            assert result["moved"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_store_to_continuum(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "entry-123"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.store_to_continuum("content", importance=0.9)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/continuum",
                json={"content": "content", "importance": 0.9},
            )
            assert result["id"] == "entry-123"
            await client.close()

    @pytest.mark.asyncio
    async def test_export_memory(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"download_url": "https://storage/backup.json"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.export_memory(tiers=["slow"])
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/export",
                json={"format": "json", "include_metadata": True, "tiers": ["slow"]},
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_import_memory(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"imported": 50}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.import_memory([{"key": "k1", "value": "v1"}])
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/import",
                json={"data": [{"key": "k1", "value": "v1"}], "overwrite": False},
            )
            assert result["imported"] == 50
            await client.close()

    @pytest.mark.asyncio
    async def test_create_snapshot(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "snap-123"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.create_snapshot(name="test-snap")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/snapshots",
                json={"name": "test-snap"},
            )
            assert result["id"] == "snap-123"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_cross_debate(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entries": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.get_cross_debate(topic="architecture")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/cross-debate",
                params={"limit": 10, "min_relevance": 0.5, "topic": "architecture"},
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_vacuum(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"vacuumed": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.vacuum()
            mock_request.assert_called_once_with("POST", "/api/v1/memory/vacuum", json={})
            assert result["vacuumed"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_rebuild_index(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"rebuilt": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            _ = await client.memory.rebuild_index()
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/memory/rebuild-index",
                json={},
            )
            assert result["rebuilt"] is True
            await client.close()
