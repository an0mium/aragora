"""Tests for Security namespace API."""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestSecurityStatus:
    """Tests for security status operations."""

    def test_get_status(self, client: AragoraClient, mock_request) -> None:
        """Get overall security status."""
        mock_request.return_value = {
            "overall": "healthy",
            "encryption_enabled": True,
            "audit_logging_enabled": True,
            "mfa_enabled": True,
            "active_threats": 0,
        }

        result = client.security.get_status()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/status",
            params=None,
            json=None,
            headers=None,
        )
        assert result["overall"] == "healthy"
        assert result["encryption_enabled"] is True


class TestSecurityHealthChecks:
    """Tests for security health check operations."""

    def test_get_health_checks(self, client: AragoraClient, mock_request) -> None:
        """Get security health checks."""
        mock_request.return_value = {
            "checks": [
                {
                    "component": "encryption",
                    "status": "ok",
                    "last_checked": "2024-01-15T10:00:00Z",
                },
                {
                    "component": "auth",
                    "status": "ok",
                    "last_checked": "2024-01-15T10:00:00Z",
                },
            ]
        }

        result = client.security.get_health_checks()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/health",
            params=None,
            json=None,
            headers=None,
        )
        assert len(result["checks"]) == 2
        assert result["checks"][0]["status"] == "ok"


class TestSecurityKeys:
    """Tests for key management operations."""

    def test_list_keys(self, client: AragoraClient, mock_request) -> None:
        """List security keys."""
        mock_request.return_value = {
            "keys": [
                {
                    "key_id": "key_1",
                    "version": 1,
                    "is_active": True,
                    "created_at": "2024-01-15T10:00:00Z",
                    "age_days": 3,
                },
            ]
        }

        result = client.security.list_keys()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/keys",
            params=None,
            json=None,
            headers=None,
        )
        assert len(result["keys"]) == 1
        assert result["keys"][0]["is_active"] is True

    def test_rotate_key(self, client: AragoraClient, mock_request) -> None:
        """Rotate a key."""
        mock_request.return_value = {
            "success": True,
            "new_key_id": "key_new",
            "old_key_id": "key_old",
            "rotated_at": "2024-01-15T10:00:00Z",
        }

        result = client.security.rotate_key(
            key_id="key_old",
            algorithm="AES-256-GCM",
            reason="Scheduled rotation",
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["key_id"] == "key_old"
        assert call_json["reason"] == "Scheduled rotation"
        assert result["success"] is True


class TestAsyncSecurity:
    """Tests for async security API."""

    @pytest.mark.asyncio
    async def test_async_get_status(self, mock_async_request) -> None:
        """Get status asynchronously."""
        mock_async_request.return_value = {"overall": "healthy"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.security.get_status()

            assert result["overall"] == "healthy"

    @pytest.mark.asyncio
    async def test_async_get_health_checks(self, mock_async_request) -> None:
        """Get health checks asynchronously."""
        mock_async_request.return_value = {"checks": [{"status": "ok"}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.security.get_health_checks()

            assert len(result["checks"]) == 1

    @pytest.mark.asyncio
    async def test_async_list_keys(self, mock_async_request) -> None:
        """List keys asynchronously."""
        mock_async_request.return_value = {"keys": [{"id": "key_1"}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.security.list_keys()

            assert len(result["keys"]) == 1

    @pytest.mark.asyncio
    async def test_async_rotate_key(self, mock_async_request) -> None:
        """Rotate key asynchronously."""
        mock_async_request.return_value = {"success": True, "new_key_id": "key_new"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.security.rotate_key(key_id="key_old")

            assert result["success"] is True
