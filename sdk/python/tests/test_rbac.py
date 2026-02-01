"""Tests for RBAC namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Bulk Operations
# =========================================================================


class TestRBACBulkOperations:
    """Tests for bulk RBAC operations."""

    def test_bulk_assign(self) -> None:
        """Bulk assign roles to users."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"assigned": 3}

            client = AragoraClient(base_url="https://api.aragora.ai")
            assignments = [
                {"user_id": "u1", "role_id": "r1"},
                {"user_id": "u2", "role_id": "r2"},
                {"user_id": "u3", "role_id": "r1"},
            ]
            result = client.rbac.bulk_assign(assignments)

            mock_request.assert_called_once_with(
                "POST", "/api/v1/rbac/bulk-assign", json={"assignments": assignments}
            )
            assert result["assigned"] == 3
            client.close()


# =========================================================================
# User Management
# =========================================================================


class TestRBACUserManagement:
    """Tests for user management methods."""

    def test_list_users_defaults(self) -> None:
        """List users with default pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"users": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.list_users()

            mock_request.assert_called_once_with(
                "GET", "/api/users", params={"limit": 100, "offset": 0}
            )
            client.close()

    def test_list_users_custom_pagination(self) -> None:
        """List users with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"users": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.list_users(limit=10, offset=20)

            mock_request.assert_called_once_with(
                "GET", "/api/users", params={"limit": 10, "offset": 20}
            )
            client.close()

    def test_invite_user_email_only(self) -> None:
        """Invite user with email only."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"invited": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.invite_user("user@example.com")

            mock_request.assert_called_once_with(
                "POST", "/api/users/invite", json={"email": "user@example.com"}
            )
            client.close()

    def test_invite_user_with_role(self) -> None:
        """Invite user with a pre-assigned role."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"invited": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.invite_user("user@example.com", role="analyst")

            mock_request.assert_called_once_with(
                "POST",
                "/api/users/invite",
                json={"email": "user@example.com", "role": "analyst"},
            )
            client.close()

    def test_remove_user(self) -> None:
        """Remove user from organization."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"removed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.remove_user("user_123")

            mock_request.assert_called_once_with("DELETE", "/api/users/user_123")
            client.close()

    def test_change_user_role(self) -> None:
        """Change user's role."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"updated": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.change_user_role("user_123", "admin")

            mock_request.assert_called_once_with(
                "PUT", "/api/users/user_123/role", json={"role": "admin"}
            )
            client.close()


# =========================================================================
# Workspace Roles
# =========================================================================


class TestRBACWorkspaceRoles:
    """Tests for workspace role management."""

    def test_get_workspace_roles(self) -> None:
        """Get workspace roles."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"roles": ["admin", "member"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.rbac.get_workspace_roles("ws_1")

            mock_request.assert_called_once_with("GET", "/api/v1/workspaces/ws_1/roles")
            assert "roles" in result
            client.close()

    def test_add_workspace_member(self) -> None:
        """Add member to workspace."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"added": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.add_workspace_member("ws_1", "user_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workspaces/ws_1/members",
                json={"user_id": "user_1"},
            )
            client.close()

    def test_add_workspace_member_with_role(self) -> None:
        """Add member to workspace with a specific role."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"added": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.add_workspace_member("ws_1", "user_1", role="editor")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workspaces/ws_1/members",
                json={"user_id": "user_1", "role": "editor"},
            )
            client.close()

    def test_remove_workspace_member(self) -> None:
        """Remove member from workspace."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"removed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.remove_workspace_member("ws_1", "user_1")

            mock_request.assert_called_once_with("DELETE", "/api/v1/workspaces/ws_1/members/user_1")
            client.close()

    def test_update_member_role(self) -> None:
        """Update member's role in workspace."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"updated": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.update_member_role("ws_1", "user_1", "admin")

            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/workspaces/ws_1/members/user_1/role",
                json={"role": "admin"},
            )
            client.close()

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

    def test_get_user_activity_history(self) -> None:
        """Get user activity history."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.get_user_activity_history("user_1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/audit/actor/user_1/history",
                params={"limit": 50, "offset": 0},
            )
            client.close()

    def test_get_resource_history(self) -> None:
        """Get resource access history."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.rbac.get_resource_history("debate", "deb_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/audit/resource/debate/deb_123/history",
                params={"limit": 50, "offset": 0},
            )
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
    async def test_async_bulk_assign(self) -> None:
        """Bulk assign asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"assigned": 2}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                assignments = [{"user_id": "u1", "role_id": "r1"}]
                result = await client.rbac.bulk_assign(assignments)

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/rbac/bulk-assign", json={"assignments": assignments}
                )
                assert result["assigned"] == 2

    @pytest.mark.asyncio
    async def test_async_list_users(self) -> None:
        """List users asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"users": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.list_users()

                mock_request.assert_called_once_with(
                    "GET", "/api/users", params={"limit": 100, "offset": 0}
                )

    @pytest.mark.asyncio
    async def test_async_invite_user(self) -> None:
        """Invite user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"invited": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.invite_user("u@example.com", role="viewer")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/users/invite",
                    json={"email": "u@example.com", "role": "viewer"},
                )

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
    async def test_async_get_workspace_roles(self) -> None:
        """Get workspace roles asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"roles": ["admin"]}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.get_workspace_roles("ws_1")

                mock_request.assert_called_once_with("GET", "/api/v1/workspaces/ws_1/roles")

    @pytest.mark.asyncio
    async def test_async_list_sessions(self) -> None:
        """List sessions asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"sessions": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.rbac.list_sessions()

                mock_request.assert_called_once_with("GET", "/api/auth/sessions")
