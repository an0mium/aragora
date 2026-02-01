"""Tests for Memory namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestMemoryContinuum:
    """Tests for continuum retrieval."""

    def test_retrieve_continuum_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"memories": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.retrieve_continuum()
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
            result = client.memory.retrieve_continuum(
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

    def test_retrieve_continuum_without_tiers_omits_param(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"memories": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.memory.retrieve_continuum(query="test")
            call_params = mock_request.call_args[1]["params"]
            assert "tiers" not in call_params
            client.close()


class TestMemorySearch:
    """Tests for memory search operations."""

    def test_search_with_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"id": "r1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.search("consensus algorithm")
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


class TestMemoryStats:
    """Tests for tier stats, archive stats, pressure, and tier listing."""

    def test_get_tier_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"fast": {"count": 120}, "medium": {"count": 45}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.get_tier_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/tier-stats")
            assert result["fast"]["count"] == 120
            client.close()

    def test_get_archive_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_archived": 500, "size_mb": 12.3}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.get_archive_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/archive-stats")
            assert result["total_archived"] == 500
            client.close()

    def test_get_pressure(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"utilization": 0.72, "pressure": "moderate"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.get_pressure()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
            assert result["pressure"] == "moderate"
            client.close()

    def test_list_tiers(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tiers": [
                    {"name": "fast", "ttl": 60},
                    {"name": "glacial", "ttl": 604800},
                ],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.list_tiers()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/tiers")
            assert len(result["tiers"]) == 2
            client.close()


class TestMemoryCritiques:
    """Tests for critique store browsing."""

    def test_list_critiques_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"critiques": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.memory.list_critiques()
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
            result = client.memory.list_critiques(agent="claude", limit=5, offset=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 5, "offset": 10, "agent": "claude"},
            )
            assert result["critiques"][0]["agent"] == "claude"
            client.close()


class TestAsyncMemory:
    """Tests for async memory methods."""

    @pytest.mark.asyncio
    async def test_retrieve_continuum(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"memories": [{"id": "m1"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.memory.retrieve_continuum(
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
    async def test_search(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.memory.search("fallback strategy", sort="timestamp")
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
    async def test_get_pressure(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"utilization": 0.95, "pressure": "high"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.memory.get_pressure()
            mock_request.assert_called_once_with("GET", "/api/v1/memory/pressure")
            assert result["pressure"] == "high"
            await client.close()

    @pytest.mark.asyncio
    async def test_list_critiques(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"critiques": [{"id": "c1"}], "total": 1}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.memory.list_critiques(agent="gemini", limit=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/memory/critiques",
                params={"limit": 10, "offset": 0, "agent": "gemini"},
            )
            assert result["total"] == 1
            await client.close()
