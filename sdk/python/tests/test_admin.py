"""Tests for Admin namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestAdmin:
    """Tests for admin methods."""

    def test_get_mfa_compliance_uses_runtime_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_admins": 1, "compliance_pct": 100.0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = client.admin.get_mfa_compliance()

            mock_request.assert_called_once_with("GET", "/api/v1/admin/mfa/compliance")
            assert result["compliance_pct"] == 100.0
            client.close()

    @pytest.mark.parametrize(
        ("method_name", "expected_path"),
        [
            ("get_system_health_circuit_breakers", "/api/v1/admin/system-health/circuit-breakers"),
            ("get_system_health_slos", "/api/v1/admin/system-health/slos"),
            ("get_system_health_adapters", "/api/v1/admin/system-health/adapters"),
            ("get_system_health_agents", "/api/v1/admin/system-health/agents"),
            ("get_system_health_budget", "/api/v1/admin/system-health/budget"),
        ],
    )
    def test_system_health_helpers_use_runtime_routes(
        self, method_name: str, expected_path: str
    ) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = getattr(client.admin, method_name)()

            mock_request.assert_called_once_with("GET", expected_path)
            assert result["ok"] is True
            client.close()

    def test_system_health_component_dispatches_to_documented_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"component": "circuit-breakers"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = client.admin.get_system_health_component("circuit_breakers")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/admin/system-health/circuit-breakers"
            )
            assert result["component"] == "circuit-breakers"
            client.close()

    def test_system_health_component_rejects_unknown_component(self) -> None:
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

        with pytest.raises(ValueError, match="Unsupported system health component"):
            client.admin.get_system_health_component("overview")

        client.close()


class TestAsyncAdmin:
    """Tests for async admin methods."""

    @pytest.mark.asyncio
    async def test_get_mfa_compliance_uses_runtime_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_admins": 1, "compliance_pct": 100.0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = await client.admin.get_mfa_compliance()

            mock_request.assert_called_once_with("GET", "/api/v1/admin/mfa/compliance")
            assert result["compliance_pct"] == 100.0
            await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "expected_path"),
        [
            ("get_system_health_circuit_breakers", "/api/v1/admin/system-health/circuit-breakers"),
            ("get_system_health_slos", "/api/v1/admin/system-health/slos"),
            ("get_system_health_adapters", "/api/v1/admin/system-health/adapters"),
            ("get_system_health_agents", "/api/v1/admin/system-health/agents"),
            ("get_system_health_budget", "/api/v1/admin/system-health/budget"),
        ],
    )
    async def test_async_system_health_helpers_use_runtime_routes(
        self, method_name: str, expected_path: str
    ) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = await getattr(client.admin, method_name)()

            mock_request.assert_called_once_with("GET", expected_path)
            assert result["ok"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_async_system_health_component_dispatches_to_documented_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"component": "budget"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = await client.admin.get_system_health_component("budget")

            mock_request.assert_called_once_with("GET", "/api/v1/admin/system-health/budget")
            assert result["component"] == "budget"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_system_health_component_rejects_unknown_component(self) -> None:
        client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

        with pytest.raises(ValueError, match="Unsupported system health component"):
            await client.admin.get_system_health_component("overview")

        await client.close()
