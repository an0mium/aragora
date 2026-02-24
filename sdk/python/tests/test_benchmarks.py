"""Tests for Benchmarks namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


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
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"categories": ["finance"], "count": 1}

            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.benchmarks.categories()

            mock_request.assert_called_once_with("GET", "/api/v1/benchmarks/categories")
            assert result["count"] == 1
            await client.close()
