"""Tests for Benchmarks namespace API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora_sdk.client import AragoraClient
from aragora_sdk.namespaces.benchmarks import AsyncBenchmarksAPI


class TestBenchmarks:
    """Tests for benchmark endpoint bindings."""

    def test_list_categories(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"categories": ["finance", "healthcare"], "count": 2}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.benchmarks.categories()

            mock_request.assert_called_once_with("GET", "/api/v1/benchmarks/categories")
            assert result["count"] == 2
            client.close()

    @pytest.mark.asyncio
    async def test_async_list_categories(self) -> None:
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"categories": ["finance"], "count": 1})

        api = AsyncBenchmarksAPI(mock_client)
        result = await api.categories()

        mock_client.request.assert_awaited_once_with("GET", "/api/v1/benchmarks/categories")
        assert result["count"] == 1
