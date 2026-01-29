"""
Tests for AuthAPI resource.

Tests cover:
- Authentication (login, logout, token refresh)
- User profile (get, update, change password)
- MFA (setup, verify, enable, disable, backup codes)
- Sessions (list, revoke, revoke all)
- API Keys (list, create, revoke)
- Dataclass parsing (User, Session, APIKey, MFASetupResult)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.auth import (
    APIKey,
    AuthAPI,
    MFASetupResult,
    Session,
    User,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create a basic client for testing."""
    return AragoraClient(base_url="http://test.example.com", api_key="test-key")


@pytest.fixture
def auth_api(client):
    """Create an AuthAPI instance for testing."""
    return client.auth


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestUserDataclass:
    """Tests for User dataclass."""

    def test_user_required_fields(self):
        """Test User with required fields only."""
        user = User(id="user123", email="test@example.com")
        assert user.id == "user123"
        assert user.email == "test@example.com"
        assert user.name is None
        assert user.mfa_enabled is False
        assert user.roles == []

    def test_user_all_fields(self):
        """Test User with all fields."""
        now = datetime.now()
        user = User(
            id="user123",
            email="test@example.com",
            name="Test User",
            mfa_enabled=True,
            created_at=now,
            last_login=now,
            roles=["admin", "user"],
        )
        assert user.name == "Test User"
        assert user.mfa_enabled is True
        assert user.roles == ["admin", "user"]


class TestSessionDataclass:
    """Tests for Session dataclass."""

    def test_session_required_fields(self):
        """Test Session with required fields."""
        now = datetime.now()
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=now,
            created_at=now,
        )
        assert session.session_id == "sess123"
        assert session.user_id == "user123"
        assert session.ip_address is None

    def test_session_all_fields(self):
        """Test Session with all fields."""
        now = datetime.now()
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=now,
            created_at=now,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"


class TestAPIKeyDataclass:
    """Tests for APIKey dataclass."""

    def test_api_key_required_fields(self):
        """Test APIKey with required fields."""
        now = datetime.now()
        key = APIKey(
            id="key123",
            name="My Key",
            key_prefix="ak_",
            created_at=now,
        )
        assert key.id == "key123"
        assert key.name == "My Key"
        assert key.scopes == []

    def test_api_key_all_fields(self):
        """Test APIKey with all fields."""
        now = datetime.now()
        key = APIKey(
            id="key123",
            name="My Key",
            key_prefix="ak_",
            created_at=now,
            last_used=now,
            expires_at=now,
            scopes=["read", "write"],
        )
        assert key.scopes == ["read", "write"]


class TestMFASetupResultDataclass:
    """Tests for MFASetupResult dataclass."""

    def test_mfa_setup_result_required_fields(self):
        """Test MFASetupResult with required fields."""
        result = MFASetupResult(
            secret="JBSWY3DPEHPK3PXP",
            qr_code_url="otpauth://totp/Example:user?secret=JBSWY3DPEHPK3PXP",
        )
        assert result.secret == "JBSWY3DPEHPK3PXP"
        assert result.backup_codes == []

    def test_mfa_setup_result_with_backup_codes(self):
        """Test MFASetupResult with backup codes."""
        result = MFASetupResult(
            secret="JBSWY3DPEHPK3PXP",
            qr_code_url="otpauth://totp/Example:user?secret=JBSWY3DPEHPK3PXP",
            backup_codes=["code1", "code2", "code3"],
        )
        assert len(result.backup_codes) == 3


# ============================================================================
# Authentication Tests
# ============================================================================


class TestLogin:
    """Tests for login methods."""

    def test_login_basic(self, auth_api, client):
        """Test basic login."""
        client._post = MagicMock(
            return_value={
                "access_token": "token123",
                "refresh_token": "refresh123",
                "expires_in": 3600,
            }
        )

        result = auth_api.login("test@example.com", "password123")

        client._post.assert_called_once_with(
            "/api/v1/auth/login",
            {"email": "test@example.com", "password": "password123"},
        )
        assert result["access_token"] == "token123"

    def test_login_with_mfa(self, auth_api, client):
        """Test login with MFA code."""
        client._post = MagicMock(return_value={"access_token": "token123"})

        auth_api.login("test@example.com", "password123", mfa_code="123456")

        call_args = client._post.call_args[0]
        assert call_args[1]["mfa_code"] == "123456"

    def test_login_without_mfa_code(self, auth_api, client):
        """Test login body doesn't include mfa_code when not provided."""
        client._post = MagicMock(return_value={"access_token": "token123"})

        auth_api.login("test@example.com", "password123")

        call_args = client._post.call_args[0]
        assert "mfa_code" not in call_args[1]


class TestLogout:
    """Tests for logout methods."""

    def test_logout(self, auth_api, client):
        """Test logout."""
        client._post = MagicMock(return_value={})

        result = auth_api.logout()

        client._post.assert_called_once_with("/api/v1/auth/logout", {})
        assert result is True


class TestTokenRefresh:
    """Tests for token refresh methods."""

    def test_refresh_token(self, auth_api, client):
        """Test token refresh."""
        client._post = MagicMock(
            return_value={
                "access_token": "new_token",
                "expires_in": 3600,
            }
        )

        result = auth_api.refresh_token("old_refresh_token")

        client._post.assert_called_once_with(
            "/api/v1/auth/refresh",
            {"refresh_token": "old_refresh_token"},
        )
        assert result["access_token"] == "new_token"


# ============================================================================
# User Profile Tests
# ============================================================================


class TestGetCurrentUser:
    """Tests for get_current_user methods."""

    def test_get_current_user(self, auth_api, client):
        """Test getting current user."""
        client._get = MagicMock(
            return_value={
                "id": "user123",
                "email": "test@example.com",
                "name": "Test User",
                "mfa_enabled": True,
                "roles": ["user"],
            }
        )

        result = auth_api.get_current_user()

        client._get.assert_called_once_with("/api/v1/auth/me")
        assert isinstance(result, User)
        assert result.id == "user123"
        assert result.email == "test@example.com"
        assert result.name == "Test User"
        assert result.mfa_enabled is True


class TestUpdateProfile:
    """Tests for update_profile methods."""

    def test_update_profile_name(self, auth_api, client):
        """Test updating profile name."""
        client._patch = MagicMock(
            return_value={
                "id": "user123",
                "email": "test@example.com",
                "name": "New Name",
            }
        )

        result = auth_api.update_profile(name="New Name")

        client._patch.assert_called_once_with(
            "/api/v1/auth/me",
            {"name": "New Name"},
        )
        assert isinstance(result, User)
        assert result.name == "New Name"

    def test_update_profile_email(self, auth_api, client):
        """Test updating profile email."""
        client._patch = MagicMock(
            return_value={
                "id": "user123",
                "email": "new@example.com",
            }
        )

        result = auth_api.update_profile(email="new@example.com")

        call_args = client._patch.call_args[0]
        assert call_args[1]["email"] == "new@example.com"

    def test_update_profile_both(self, auth_api, client):
        """Test updating both name and email."""
        client._patch = MagicMock(
            return_value={
                "id": "user123",
                "email": "new@example.com",
                "name": "New Name",
            }
        )

        auth_api.update_profile(name="New Name", email="new@example.com")

        call_args = client._patch.call_args[0]
        assert call_args[1]["name"] == "New Name"
        assert call_args[1]["email"] == "new@example.com"


class TestChangePassword:
    """Tests for change_password methods."""

    def test_change_password(self, auth_api, client):
        """Test changing password."""
        client._post = MagicMock(return_value={})

        result = auth_api.change_password("old_pass", "new_pass")

        client._post.assert_called_once_with(
            "/api/v1/auth/change-password",
            {"current_password": "old_pass", "new_password": "new_pass"},
        )
        assert result is True


# ============================================================================
# MFA Tests
# ============================================================================


class TestMFASetup:
    """Tests for MFA setup methods."""

    def test_setup_mfa(self, auth_api, client):
        """Test MFA setup initiation."""
        client._post = MagicMock(
            return_value={
                "secret": "JBSWY3DPEHPK3PXP",
                "qr_code_url": "otpauth://totp/Example",
                "backup_codes": ["code1", "code2"],
            }
        )

        result = auth_api.setup_mfa()

        client._post.assert_called_once_with("/api/v1/auth/mfa/setup", {})
        assert isinstance(result, MFASetupResult)
        assert result.secret == "JBSWY3DPEHPK3PXP"
        assert len(result.backup_codes) == 2

    def test_setup_mfa_with_qr_uri_fallback(self, auth_api, client):
        """Test MFA setup with qr_uri instead of qr_code_url."""
        client._post = MagicMock(
            return_value={
                "secret": "JBSWY3DPEHPK3PXP",
                "qr_uri": "otpauth://totp/Example",
            }
        )

        result = auth_api.setup_mfa()

        assert result.qr_code_url == "otpauth://totp/Example"


class TestMFAVerify:
    """Tests for MFA verify methods."""

    def test_verify_mfa_setup_success(self, auth_api, client):
        """Test MFA verification success."""
        client._post = MagicMock(return_value={"verified": True})

        result = auth_api.verify_mfa_setup("123456")

        client._post.assert_called_once_with(
            "/api/v1/auth/mfa/verify",
            {"code": "123456"},
        )
        assert result is True

    def test_verify_mfa_setup_failure(self, auth_api, client):
        """Test MFA verification failure."""
        client._post = MagicMock(return_value={"verified": False})

        result = auth_api.verify_mfa_setup("000000")

        assert result is False


class TestMFAEnable:
    """Tests for MFA enable methods."""

    def test_enable_mfa_success(self, auth_api, client):
        """Test enabling MFA."""
        client._post = MagicMock(return_value={"enabled": True})

        result = auth_api.enable_mfa("123456")

        client._post.assert_called_once_with(
            "/api/v1/auth/mfa/enable",
            {"code": "123456"},
        )
        assert result is True

    def test_enable_mfa_failure(self, auth_api, client):
        """Test enabling MFA failure."""
        client._post = MagicMock(return_value={"enabled": False})

        result = auth_api.enable_mfa("000000")

        assert result is False


class TestMFADisable:
    """Tests for MFA disable methods."""

    def test_disable_mfa(self, auth_api, client):
        """Test disabling MFA."""
        client._post = MagicMock(return_value={})

        result = auth_api.disable_mfa("123456")

        client._post.assert_called_once_with(
            "/api/v1/auth/mfa/disable",
            {"code": "123456"},
        )
        assert result is True


class TestBackupCodes:
    """Tests for backup codes methods."""

    def test_get_backup_codes(self, auth_api, client):
        """Test generating backup codes."""
        client._post = MagicMock(
            return_value={
                "codes": ["code1", "code2", "code3", "code4", "code5"],
            }
        )

        result = auth_api.get_backup_codes()

        client._post.assert_called_once_with("/api/v1/auth/mfa/backup-codes", {})
        assert len(result) == 5

    def test_get_backup_codes_empty(self, auth_api, client):
        """Test empty backup codes response."""
        client._post = MagicMock(return_value={})

        result = auth_api.get_backup_codes()

        assert result == []


class TestMFAStatus:
    """Tests for MFA status methods."""

    def test_get_mfa_status(self, auth_api, client):
        """Test getting MFA status."""
        client._get = MagicMock(
            return_value={
                "enabled": True,
                "verified": True,
                "backup_codes_remaining": 3,
            }
        )

        result = auth_api.get_mfa_status()

        client._get.assert_called_once_with("/api/v1/auth/mfa/status")
        assert result["enabled"] is True
        assert result["backup_codes_remaining"] == 3


# ============================================================================
# Session Tests
# ============================================================================


class TestListSessions:
    """Tests for list_sessions methods."""

    def test_list_sessions(self, auth_api, client):
        """Test listing sessions."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "sessions": [
                    {
                        "session_id": "sess1",
                        "user_id": "user123",
                        "expires_at": now,
                        "created_at": now,
                        "ip_address": "192.168.1.1",
                    },
                    {
                        "session_id": "sess2",
                        "user_id": "user123",
                        "expires_at": now,
                        "created_at": now,
                    },
                ],
            }
        )

        result = auth_api.list_sessions()

        client._get.assert_called_once_with("/api/v1/auth/sessions")
        assert len(result) == 2
        assert all(isinstance(s, Session) for s in result)
        assert result[0].session_id == "sess1"
        assert result[0].ip_address == "192.168.1.1"

    def test_list_sessions_empty(self, auth_api, client):
        """Test listing sessions when empty."""
        client._get = MagicMock(return_value={"sessions": []})

        result = auth_api.list_sessions()

        assert result == []


class TestRevokeSession:
    """Tests for revoke_session methods."""

    def test_revoke_session(self, auth_api, client):
        """Test revoking a specific session."""
        client._delete = MagicMock(return_value={})

        result = auth_api.revoke_session("sess123")

        client._delete.assert_called_once_with("/api/v1/auth/sessions/sess123")
        assert result is True


class TestRevokeAllSessions:
    """Tests for revoke_all_sessions methods."""

    def test_revoke_all_sessions_except_current(self, auth_api, client):
        """Test revoking all sessions except current."""
        client._delete = MagicMock(return_value={"revoked_count": 5})

        result = auth_api.revoke_all_sessions(except_current=True)

        client._delete.assert_called_once_with(
            "/api/v1/auth/sessions",
            {"except_current": True},
        )
        assert result == 5

    def test_revoke_all_sessions_including_current(self, auth_api, client):
        """Test revoking all sessions including current."""
        client._delete = MagicMock(return_value={"revoked_count": 6})

        result = auth_api.revoke_all_sessions(except_current=False)

        call_args = client._delete.call_args[0]
        assert call_args[1]["except_current"] is False
        assert result == 6


# ============================================================================
# API Key Tests
# ============================================================================


class TestListAPIKeys:
    """Tests for list_api_keys methods."""

    def test_list_api_keys(self, auth_api, client):
        """Test listing API keys."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "keys": [
                    {
                        "id": "key1",
                        "name": "Production Key",
                        "key_prefix": "ak_prod",
                        "created_at": now,
                        "scopes": ["read", "write"],
                    },
                    {
                        "id": "key2",
                        "name": "Dev Key",
                        "key_prefix": "ak_dev",
                        "created_at": now,
                    },
                ],
            }
        )

        result = auth_api.list_api_keys()

        client._get.assert_called_once_with("/api/v1/auth/api-keys")
        assert len(result) == 2
        assert all(isinstance(k, APIKey) for k in result)
        assert result[0].name == "Production Key"
        assert result[0].scopes == ["read", "write"]

    def test_list_api_keys_with_api_keys_key(self, auth_api, client):
        """Test listing API keys with 'api_keys' response key."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "api_keys": [
                    {
                        "id": "key1",
                        "name": "Key",
                        "key_prefix": "ak_",
                        "created_at": now,
                    },
                ],
            }
        )

        result = auth_api.list_api_keys()

        assert len(result) == 1


class TestCreateAPIKey:
    """Tests for create_api_key methods."""

    def test_create_api_key_basic(self, auth_api, client):
        """Test creating API key with name only."""
        client._post = MagicMock(
            return_value={
                "id": "key123",
                "name": "My Key",
                "key": "ak_full_secret_key_here",
                "key_prefix": "ak_",
            }
        )

        result = auth_api.create_api_key("My Key")

        client._post.assert_called_once_with(
            "/api/v1/auth/api-keys",
            {"name": "My Key"},
        )
        assert result["key"] == "ak_full_secret_key_here"

    def test_create_api_key_with_scopes(self, auth_api, client):
        """Test creating API key with scopes."""
        client._post = MagicMock(return_value={"id": "key123"})

        auth_api.create_api_key("My Key", scopes=["read", "write"])

        call_args = client._post.call_args[0]
        assert call_args[1]["scopes"] == ["read", "write"]

    def test_create_api_key_with_expiration(self, auth_api, client):
        """Test creating API key with expiration."""
        client._post = MagicMock(return_value={"id": "key123"})

        auth_api.create_api_key("My Key", expires_in_days=30)

        call_args = client._post.call_args[0]
        assert call_args[1]["expires_in_days"] == 30

    def test_create_api_key_all_options(self, auth_api, client):
        """Test creating API key with all options."""
        client._post = MagicMock(return_value={"id": "key123"})

        auth_api.create_api_key(
            "My Key",
            scopes=["read"],
            expires_in_days=90,
        )

        call_args = client._post.call_args[0]
        assert call_args[1]["name"] == "My Key"
        assert call_args[1]["scopes"] == ["read"]
        assert call_args[1]["expires_in_days"] == 90


class TestRevokeAPIKey:
    """Tests for revoke_api_key methods."""

    def test_revoke_api_key(self, auth_api, client):
        """Test revoking an API key."""
        client._delete = MagicMock(return_value={})

        result = auth_api.revoke_api_key("key123")

        client._delete.assert_called_once_with("/api/v1/auth/api-keys/key123")
        assert result is True


# ============================================================================
# Client Integration Tests
# ============================================================================


class TestAuthAPIIntegration:
    """Tests for AuthAPI integration with AragoraClient."""

    def test_auth_accessible_from_client(self):
        """Test auth API is accessible from client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert hasattr(client, "auth")
        assert isinstance(client.auth, AuthAPI)

    def test_auth_shares_client(self):
        """Test auth API shares the same client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert client.auth._client is client
