"""Tests for Memory namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestMemoryRetrieve:
    """Tests for memory continuum retrieval."""

    def test_retrieve_continuum_default(self, client: AragoraClient, mock_request) -> None:
        """Retrieve memories with default parameters."""
        mock_request.return_value = {"memories": [], "total": 0}

        client.memory.retrieve_continuum()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/memory/continuum/retrieve",
            params={"query": "", "limit": 10, "min_importance": 0.0},
        )

    def test_retrieve_continuum_with_query(self, client: AragoraClient, mock_request) -> None:
        """Retrieve memories matching a query."""
        mock_request.return_value = {
            "memories": [{"content": "Previous debate about APIs", "importance": 0.8}]
        }

        result = client.memory.retrieve_continuum(query="API design", limit=5)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["query"] == "API design"
        assert call_params["limit"] == 5
        assert len(result["memories"]) == 1

    def test_retrieve_continuum_with_tiers(self, client: AragoraClient, mock_request) -> None:
        """Retrieve memories from specific tiers."""
        mock_request.return_value = {"memories": []}

        client.memory.retrieve_continuum(tiers=["fast", "medium"])

        call_params = mock_request.call_args[1]["params"]
        assert call_params["tiers"] == "fast,medium"

    def test_retrieve_continuum_with_min_importance(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Retrieve memories above a minimum importance threshold."""
        mock_request.return_value = {"memories": []}

        client.memory.retrieve_continuum(min_importance=0.7)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["min_importance"] == 0.7


class TestMemorySearch:
    """Tests for memory search."""

    def test_search_default(self, client: AragoraClient, mock_request) -> None:
        """Search memories with default parameters."""
        mock_request.return_value = {"results": [], "total": 0}

        client.memory.search("microservices debate")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/memory/search",
            params={
                "q": "microservices debate",
                "limit": 20,
                "min_importance": 0.0,
                "sort": "relevance",
            },
        )

    def test_search_with_tier_filter(self, client: AragoraClient, mock_request) -> None:
        """Search memories filtered by tier."""
        mock_request.return_value = {"results": []}

        client.memory.search("query", tier=["slow", "glacial"])

        call_params = mock_request.call_args[1]["params"]
        assert call_params["tier"] == "slow,glacial"

    def test_search_custom_sort(self, client: AragoraClient, mock_request) -> None:
        """Search memories with custom sort order."""
        mock_request.return_value = {"results": []}

        client.memory.search("query", sort="timestamp", limit=5)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["sort"] == "timestamp"
        assert call_params["limit"] == 5


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_get_tier_stats(self, client: AragoraClient, mock_request) -> None:
        """Get tier statistics."""
        mock_request.return_value = {
            "fast": {"count": 100, "size_mb": 2.5},
            "medium": {"count": 500, "size_mb": 15.0},
            "slow": {"count": 2000, "size_mb": 80.0},
            "glacial": {"count": 10000, "size_mb": 500.0},
        }

        result = client.memory.get_tier_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/memory/tier-stats")
        assert result["fast"]["count"] == 100

    def test_get_archive_stats(self, client: AragoraClient, mock_request) -> None:
        """Get archive statistics."""
        mock_request.return_value = {"total_archived": 5000, "size_gb": 1.2}

        result = client.memory.get_archive_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/memory/archive-stats")
        assert result["total_archived"] == 5000

    def test_get_pressure(self, client: AragoraClient, mock_request) -> None:
        """Get memory pressure metrics."""
        mock_request.return_value = {
            "utilization": 0.65,
            "eviction_rate": 0.02,
            "pressure_level": "normal",
        }

        result = client.memory.get_pressure()

        mock_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
        assert result["pressure_level"] == "normal"

    def test_list_tiers(self, client: AragoraClient, mock_request) -> None:
        """List all memory tiers with detailed stats."""
        mock_request.return_value = {
            "tiers": [
                {"name": "fast", "ttl": 60, "count": 100},
                {"name": "medium", "ttl": 3600, "count": 500},
            ]
        }

        result = client.memory.list_tiers()

        mock_request.assert_called_once_with("GET", "/api/v1/memory/tiers")
        assert len(result["tiers"]) == 2


class TestMemoryCritiques:
    """Tests for critique store browsing."""

    def test_list_critiques_default(self, client: AragoraClient, mock_request) -> None:
        """List critiques with default pagination."""
        mock_request.return_value = {"critiques": [], "total": 0}

        client.memory.list_critiques()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/memory/critiques",
            params={"limit": 20, "offset": 0},
        )

    def test_list_critiques_filtered(self, client: AragoraClient, mock_request) -> None:
        """List critiques filtered by agent."""
        mock_request.return_value = {"critiques": [{"agent": "claude", "content": "Good point"}]}

        result = client.memory.list_critiques(agent="claude", limit=10, offset=5)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["agent"] == "claude"
        assert call_params["limit"] == 10
        assert call_params["offset"] == 5
        assert result["critiques"][0]["agent"] == "claude"


class TestAsyncMemory:
    """Tests for async memory API."""

    @pytest.mark.asyncio
    async def test_async_retrieve_continuum(self, mock_async_request) -> None:
        """Retrieve memories asynchronously."""
        mock_async_request.return_value = {"memories": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.memory.retrieve_continuum(query="test")

            call_params = mock_async_request.call_args[1]["params"]
            assert call_params["query"] == "test"

    @pytest.mark.asyncio
    async def test_async_search(self, mock_async_request) -> None:
        """Search memories asynchronously."""
        mock_async_request.return_value = {"results": [{"content": "found"}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.memory.search("test query")

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/search",
                params={
                    "q": "test query",
                    "limit": 20,
                    "min_importance": 0.0,
                    "sort": "relevance",
                },
            )
            assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_async_get_pressure(self, mock_async_request) -> None:
        """Get memory pressure asynchronously."""
        mock_async_request.return_value = {"pressure_level": "high"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.memory.get_pressure()

            mock_async_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
            assert result["pressure_level"] == "high"

    @pytest.mark.asyncio
    async def test_async_list_critiques(self, mock_async_request) -> None:
        """List critiques asynchronously."""
        mock_async_request.return_value = {"critiques": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.memory.list_critiques(agent="gpt-4")

            call_params = mock_async_request.call_args[1]["params"]
            assert call_params["agent"] == "gpt-4"
