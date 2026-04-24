from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.services import AsyncServicesAPI, ServicesAPI


class TestServicesWiring:
    def test_sync_client_has_services(self) -> None:
        client = AragoraClient(base_url="http://localhost")
        assert isinstance(client.services, ServicesAPI)
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_has_services(self) -> None:
        async with AragoraAsyncClient(base_url="http://localhost") as client:
            assert isinstance(client.services, AsyncServicesAPI)


class TestServicesNamespace:
    def test_list_services(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"services": [{"id": "svc-1"}], "count": 1}

            client = AragoraClient(base_url="http://localhost")
            result = client.services.list(status="healthy")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/services",
                params={"status": "healthy"},
            )
            assert result["count"] == 1
            client.close()

    def test_get_service(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"service": {"id": "svc-1", "name": "API"}}

            client = AragoraClient(base_url="http://localhost")
            result = client.services.get("svc-1")

            mock_request.assert_called_once_with("GET", "/api/v1/services/svc-1")
            assert result["service"]["id"] == "svc-1"
            client.close()

    def test_namespace_no_longer_exposes_unimplemented_service_operations(self) -> None:
        client = AragoraClient(base_url="http://localhost")

        assert not hasattr(client.services, "get_health")
        assert not hasattr(client.services, "get_metrics")
        assert not hasattr(client.services, "register")
        assert not hasattr(client.services, "deregister")
        assert not hasattr(client.services, "get_dependencies")

        client.close()
