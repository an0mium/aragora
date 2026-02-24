"""Tests for RBAC namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Bulk Operations
# =========================================================================


class TestRBACWorkspaceRoles:
    """Tests for workspace role management."""

    def test_list_profiles(self) -> None:
        """List RBAC profiles."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"profiles": ["lite", "standard", "enterprise"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.list_profiles()

            mock_request.assert_called_once_with("GET", "/api/v1/workspaces/profiles")
            assert len(result["profiles"]) == 3
            client.close()


# =========================================================================
# Audit Trail
# =========================================================================


class TestRBACAudit:
    """Tests for audit trail methods."""

    def test_query_audit_defaults(self) -> None:
        """Query audit with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.query_audit()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/entries", params={"limit": 50, "offset": 0}
            )
            client.close()

    def test_query_audit_with_filters(self) -> None:
        """Query audit with all filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.query_audit(
                action="login", user_id="u1", since="2025-01-01", limit=10, offset=5
            )

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/audit/entries",
                params={
                    "limit": 10,
                    "offset": 5,
                    "action": "login",
                    "user_id": "u1",
                    "since": "2025-01-01",
                },
            )
            client.close()

    def test_get_audit_report(self) -> None:
        """Generate audit report."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"report": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.get_audit_report(framework="soc2")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/report", params={"framework": "soc2"}
            )
            client.close()

    def test_verify_audit_integrity(self) -> None:
        """Verify audit log integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.verify_audit_integrity()

            mock_request.assert_called_once_with("GET", "/api/v1/audit/verify")
            assert result["valid"] is True
            client.close()

    def test_get_denied_access(self) -> None:
        """Get denied access attempts."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"denied": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.get_denied_access(limit=10)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/audit/denied", params={"limit": 10, "offset": 0}
            )
            client.close()


# =========================================================================
# API Keys
# =========================================================================


class TestRBACApiKeys:
    """Tests for API key management."""

    def test_generate_api_key_name_only(self) -> None:
        """Generate API key with name only."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"key": "ak_xxx"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.generate_api_key("my-key")

            mock_request.assert_called_once_with(
                "POST", "/api/auth/api-key", json={"name": "my-key"}
            )
            assert result["key"] == "ak_xxx"
            client.close()

    def test_generate_api_key_with_options(self) -> None:
        """Generate API key with permissions and expiry."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"key": "ak_xxx"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.generate_api_key(
                "my-key",
                permissions=["debates:read", "agents:read"],
                expires_at="2026-12-31",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/auth/api-key",
                json={
                    "name": "my-key",
                    "permissions": ["debates:read", "agents:read"],
                    "expires_at": "2026-12-31",
                },
            )
            client.close()

    def test_list_api_keys(self) -> None:
        """List API keys."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"keys": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.list_api_keys()

            mock_request.assert_called_once_with("GET", "/api/keys")
            client.close()

    def test_revoke_api_key(self) -> None:
        """Revoke an API key."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"revoked": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.revoke_api_key("key_123")

            mock_request.assert_called_once_with("DELETE", "/api/keys/key_123")
            client.close()


# =========================================================================
# Sessions
# =========================================================================


class TestRBACSessions:
    """Tests for session management."""

    def test_list_sessions(self) -> None:
        """List active sessions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sessions": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.list_sessions()

            mock_request.assert_called_once_with("GET", "/api/auth/sessions")
            client.close()

    def test_revoke_session(self) -> None:
        """Revoke a session."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"revoked": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.revoke_session("sess_abc")

            mock_request.assert_called_once_with("DELETE", "/api/auth/sessions/sess_abc")
            client.close()

    def test_logout_all(self) -> None:
        """Logout from all devices."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"logged_out": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.logout_all()

            mock_request.assert_called_once_with("POST", "/api/auth/logout-all", json={})
            client.close()


# =========================================================================
# MFA
# =========================================================================


class TestRBACMFA:
    """Tests for MFA methods."""

    def test_setup_mfa(self) -> None:
        """Setup MFA."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"secret": "xxx", "qr_code": "data:..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.setup_mfa()

            mock_request.assert_called_once_with("POST", "/api/auth/mfa/setup", json={})
            assert "secret" in result
            client.close()

    def test_enable_mfa(self) -> None:
        """Enable MFA with verification code."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"enabled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.enable_mfa("123456")

            mock_request.assert_called_once_with(
                "POST", "/api/auth/mfa/enable", json={"code": "123456"}
            )
            client.close()

    def test_disable_mfa(self) -> None:
        """Disable MFA."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"disabled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.disable_mfa("123456")

            mock_request.assert_called_once_with(
                "POST", "/api/auth/mfa/disable", json={"code": "123456"}
            )
            client.close()

    def test_verify_mfa(self) -> None:
        """Verify MFA code."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"verified": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.verify_mfa("654321")

            mock_request.assert_called_once_with(
                "POST", "/api/auth/mfa/verify", json={"code": "654321"}
            )
            assert result["verified"] is True
            client.close()

    def test_regenerate_backup_codes(self) -> None:
        """Regenerate MFA backup codes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"codes": ["abc", "def"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.regenerate_backup_codes("123456")

            mock_request.assert_called_once_with(
                "POST", "/api/auth/mfa/backup-codes", json={"code": "123456"}
            )
            assert len(result["codes"]) == 2
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncRBAC:
    """Tests for async RBAC API."""

    @pytest.mark.asyncio
    async def test_async_query_audit(self) -> None:
        """Query audit asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entries": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.query_audit(action="create")

                call_args = mock_request.call_args
                assert call_args[1]["params"]["action"] == "create"

    @pytest.mark.asyncio
    async def test_async_setup_mfa(self) -> None:
        """Setup MFA asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"secret": "xxx"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.rbac.setup_mfa()

                mock_request.assert_called_once_with("POST", "/api/auth/mfa/setup", json={})
                assert "secret" in result

    @pytest.mark.asyncio
    async def test_async_generate_api_key(self) -> None:
        """Generate API key asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"key": "ak_xxx"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.rbac.generate_api_key("test-key")

                mock_request.assert_called_once_with(
                    "POST", "/api/auth/api-key", json={"name": "test-key"}
                )
                assert result["key"] == "ak_xxx"

    @pytest.mark.asyncio
    async def test_async_list_sessions(self) -> None:
        """List sessions asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"sessions": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.list_sessions()

                mock_request.assert_called_once_with("GET", "/api/auth/sessions")
