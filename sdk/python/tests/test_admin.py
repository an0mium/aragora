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
