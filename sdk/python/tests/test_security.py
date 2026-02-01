"""Tests for Security namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


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

    def test_run_security_scan(self, client: AragoraClient, mock_request) -> None:
        """Trigger a security scan."""
        mock_request.return_value = {
            "scan_id": "scan_123",
            "status": "in_progress",
        }

        result = client.security.run_security_scan()

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/admin/security/scan",
            params=None,
            json=None,
            headers=None,
        )
        assert result["status"] == "in_progress"

    def test_get_scan_status(self, client: AragoraClient, mock_request) -> None:
        """Get security scan status."""
        mock_request.return_value = {
            "scan_id": "scan_123",
            "status": "completed",
            "findings": [],
            "progress": 100,
        }

        result = client.security.get_scan_status("scan_123")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/scan/scan_123",
            params=None,
            json=None,
            headers=None,
        )
        assert result["status"] == "completed"


class TestSecurityKeys:
    """Tests for key management operations."""

    def test_list_keys(self, client: AragoraClient, mock_request) -> None:
        """List security keys."""
        mock_request.return_value = {
            "keys": [
                {
                    "id": "key_1",
                    "name": "primary",
                    "algorithm": "AES-256-GCM",
                    "status": "active",
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
        assert result["keys"][0]["status"] == "active"

    def test_get_key(self, client: AragoraClient, mock_request) -> None:
        """Get key details."""
        mock_request.return_value = {
            "id": "key_123",
            "name": "encryption_key",
            "algorithm": "AES-256-GCM",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "active",
        }

        result = client.security.get_key("key_123")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/keys/key_123",
            params=None,
            json=None,
            headers=None,
        )
        assert result["algorithm"] == "AES-256-GCM"

    def test_create_key(self, client: AragoraClient, mock_request) -> None:
        """Create a new key."""
        mock_request.return_value = {
            "id": "key_new",
            "name": "backup_key",
            "status": "active",
        }

        result = client.security.create_key(
            name="backup_key",
            algorithm="AES-256-GCM",
            expires_in_days=365,
            metadata={"purpose": "backup"},
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["name"] == "backup_key"
        assert call_json["algorithm"] == "AES-256-GCM"
        assert call_json["expires_in_days"] == 365
        assert result["status"] == "active"

    def test_revoke_key(self, client: AragoraClient, mock_request) -> None:
        """Revoke a key."""
        mock_request.return_value = {"revoked": True}

        result = client.security.revoke_key("key_123", reason="Compromised")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/admin/security/keys/key_123/revoke",
            params=None,
            json={"reason": "Compromised"},
            headers=None,
        )
        assert result["revoked"] is True

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


class TestSecurityAudit:
    """Tests for audit and compliance operations."""

    def test_get_audit_log(self, client: AragoraClient, mock_request) -> None:
        """Get audit log entries."""
        mock_request.return_value = {
            "entries": [
                {"event_type": "key_rotation", "timestamp": "2024-01-15T10:00:00Z"},
            ],
            "total": 1,
        }

        result = client.security.get_audit_log(
            limit=10,
            event_type="key_rotation",
            since="2024-01-01T00:00:00Z",
        )

        call_kwargs = mock_request.call_args[1]
        call_params = call_kwargs["params"]
        assert call_params["limit"] == 10
        assert call_params["event_type"] == "key_rotation"
        assert len(result["entries"]) == 1

    def test_get_compliance_status(self, client: AragoraClient, mock_request) -> None:
        """Get compliance status."""
        mock_request.return_value = {
            "soc2": {"status": "compliant", "last_audit": "2024-01-01"},
            "gdpr": {"status": "compliant", "last_audit": "2024-01-01"},
        }

        result = client.security.get_compliance_status()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/admin/security/compliance",
            params=None,
            json=None,
            headers=None,
        )
        assert result["soc2"]["status"] == "compliant"


class TestSecurityThreats:
    """Tests for threat detection operations."""

    def test_list_threats(self, client: AragoraClient, mock_request) -> None:
        """List detected threats."""
        mock_request.return_value = {
            "threats": [
                {"id": "threat_1", "type": "brute_force", "status": "active"},
            ]
        }

        result = client.security.list_threats(limit=20, status="active")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["limit"] == 20
        assert call_kwargs["params"]["status"] == "active"
        assert len(result["threats"]) == 1

    def test_resolve_threat(self, client: AragoraClient, mock_request) -> None:
        """Resolve a threat."""
        mock_request.return_value = {"resolved": True}

        result = client.security.resolve_threat(
            "threat_123",
            resolution="IP blocked and account locked",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/admin/security/threats/threat_123/resolve",
            params=None,
            json={"resolution": "IP blocked and account locked"},
            headers=None,
        )
        assert result["resolved"] is True


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
