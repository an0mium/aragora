"""Tests for docs introspection helpers in the Python OpenAPI namespace."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestOpenApiDocsRoutes:
    def test_get_docs_routes_uses_v1_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total": 12}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.openapi.request_get_api_v1_docs_routes({"tag": "chat"})

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/docs/routes",
                params={"tag": "chat"},
            )
            assert result["total"] == 12
            client.close()

    @pytest.mark.asyncio
    async def test_async_get_docs_stats_uses_v1_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_endpoints": 1797}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.openapi.request_get_api_v1_docs_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/docs/stats", params=None)
            assert result["total_endpoints"] == 1797
