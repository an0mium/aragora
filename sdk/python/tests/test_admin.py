"""Tests for the Admin namespace API."""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestAdminMFACompliance:
    """Tests for admin MFA compliance operations."""

    def test_get_mfa_compliance_uses_runtime_slash_route(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Get MFA compliance through the server-supported slash route."""
        mock_request.return_value = {
            "total_admins": 4,
            "mfa_enabled_count": 3,
            "mfa_disabled_count": 1,
        }

        result = client.admin.get_mfa_compliance()

        mock_request.assert_called_once_with("GET", "/api/v1/admin/mfa/compliance")
        assert result["total_admins"] == 4

    @pytest.mark.asyncio
    async def test_async_get_mfa_compliance_uses_runtime_slash_route(
        self, mock_async_request
    ) -> None:
        """Get MFA compliance asynchronously through the server-supported slash route."""
        mock_async_request.return_value = {"total_admins": 2}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.admin.get_mfa_compliance()

        mock_async_request.assert_called_once_with("GET", "/api/v1/admin/mfa/compliance")
        assert result["total_admins"] == 2
