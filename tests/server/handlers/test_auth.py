"""
Tests for aragora.server.handlers.auth - User authentication handler.

Tests cover:
- Email/password validation functions
- Registration flows
- Login flows (including lockout, MFA)
- Token refresh
- Logout (single and all devices)
- User info (get/update)
- Password change
- API key generation/revocation
- MFA setup/enable/disable/verify
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.auth import (
    AuthHandler,
    InMemoryUserStore,
    validate_email,
    validate_password,
    MIN_PASSWORD_LENGTH,
    MAX_PASSWORD_LENGTH,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user object for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = None
    role: str = "user"
    is_active: bool = True
    password_hash: str = "hashed"
    password_salt: str = "salt"
    api_key: str | None = None
    api_key_hash: str | None = None
    api_key_prefix: str | None = None
    api_key_created_at: datetime | None = None
    api_key_expires_at: datetime | None = None
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    last_login_at: datetime | None = None

    def verify_password(self, password: str) -> bool:
        """Mock password verification."""
        return password == "correct_password"

    def verify_api_key(self, key: str) -> bool:
        """Mock API key verification."""
        # Support both hashed keys (new) and plain keys (legacy)
        if self.api_key_hash is not None and key.startswith("ara_"):
            return True
        # Legacy: direct comparison for plain text keys
        return self.api_key is not None and self.api_key == key

    def generate_api_key(self, expires_days: int = 365) -> str:
        """Mock API key generation."""
        self.api_key_prefix = "ara_test"
        self.api_key_hash = "hashed_key"
        self.api_key_created_at = datetime.utcnow()
        self.api_key_expires_at = datetime.utcnow() + timedelta(days=expires_days)
        return "ara_test_full_key_12345"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    owner_id: str = "user-123"
    limits: Any = field(default_factory=lambda: MagicMock(api_access=True))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "owner_id": self.owner_id}


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = None
    role: str = "user"


@dataclass
class MockTokenPair:
    """Mock token pair."""

    access_token: str = "access_token_123"
    refresh_token: str = "refresh_token_123"
    expires_in: int = 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_in": self.expires_in,
        }


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.users_by_email: dict[str, str] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self.token_versions: dict[str, int] = {}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> MockUser | None:
        user_id = self.users_by_email.get(email.lower())
        return self.users.get(user_id) if user_id else None

    def create_user(
        self, email: str, password_hash: str, password_salt: str, name: str
    ) -> MockUser:
        user = MockUser(
            id=f"user-{len(self.users) + 1}",
            email=email.lower(),
            name=name,
            password_hash=password_hash,
            password_salt=password_salt,
        )
        self.users[user.id] = user
        self.users_by_email[user.email] = user.id
        return user

    def update_user(self, user_id: str, **kwargs) -> None:
        user = self.users.get(user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

    def create_organization(self, name: str, owner_id: str) -> MockOrganization:
        org = MockOrganization(id=f"org-{len(self.orgs) + 1}", name=name, owner_id=owner_id)
        self.orgs[org.id] = org
        # Update user's org_id
        user = self.users.get(owner_id)
        if user:
            user.org_id = org.id
        return org

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def increment_token_version(self, user_id: str) -> int:
        self.token_versions[user_id] = self.token_versions.get(user_id, 0) + 1
        return self.token_versions[user_id]


def make_mock_handler(body: dict | None = None, method: str = "POST", headers: dict | None = None):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiter state before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    # Clear each limiter's buckets, not just the dict
    for limiter in _limiters.values():
        limiter.clear()
    yield
    for limiter in _limiters.values():
        limiter.clear()


@pytest.fixture
def auth_handler():
    """Create AuthHandler with mock context."""
    ctx = {"user_store": MockUserStore()}
    return AuthHandler(ctx)


@pytest.fixture
def user_store():
    """Create a mock user store."""
    return MockUserStore()


# ===========================================================================
# Test Email Validation
# ===========================================================================


class TestValidateEmail:
    """Tests for validate_email function."""

    def test_valid_email(self):
        valid, err = validate_email("test@example.com")
        assert valid is True
        assert err == ""

    def test_valid_email_with_subdomain(self):
        valid, err = validate_email("test@mail.example.com")
        assert valid is True
        assert err == ""

    def test_valid_email_with_plus(self):
        valid, err = validate_email("test+label@example.com")
        assert valid is True
        assert err == ""

    def test_empty_email(self):
        valid, err = validate_email("")
        assert valid is False
        assert "required" in err.lower()

    def test_email_too_long(self):
        long_email = "a" * 250 + "@example.com"
        valid, err = validate_email(long_email)
        assert valid is False
        assert "too long" in err.lower()

    def test_invalid_format_no_at(self):
        valid, err = validate_email("testexample.com")
        assert valid is False
        assert "invalid" in err.lower()

    def test_invalid_format_no_domain(self):
        valid, err = validate_email("test@")
        assert valid is False
        assert "invalid" in err.lower()

    def test_invalid_format_no_tld(self):
        valid, err = validate_email("test@example")
        assert valid is False
        assert "invalid" in err.lower()


# ===========================================================================
# Test Password Validation
# ===========================================================================


class TestValidatePassword:
    """Tests for validate_password function."""

    def test_valid_password(self):
        valid, err = validate_password("securepass123")
        assert valid is True
        assert err == ""

    def test_empty_password(self):
        valid, err = validate_password("")
        assert valid is False
        assert "required" in err.lower()

    def test_password_too_short(self):
        short = "a" * (MIN_PASSWORD_LENGTH - 1)
        valid, err = validate_password(short)
        assert valid is False
        assert str(MIN_PASSWORD_LENGTH) in err

    def test_password_too_long(self):
        long = "a" * (MAX_PASSWORD_LENGTH + 1)
        valid, err = validate_password(long)
        assert valid is False
        assert str(MAX_PASSWORD_LENGTH) in err

    def test_password_exact_min_length(self):
        exact = "a" * MIN_PASSWORD_LENGTH
        valid, err = validate_password(exact)
        assert valid is True
        assert err == ""

    def test_password_exact_max_length(self):
        exact = "a" * MAX_PASSWORD_LENGTH
        valid, err = validate_password(exact)
        assert valid is True
        assert err == ""


# ===========================================================================
# Test InMemoryUserStore
# ===========================================================================


class TestInMemoryUserStore:
    """Tests for InMemoryUserStore."""

    def test_save_and_get_user_by_id(self):
        store = InMemoryUserStore()
        user = MockUser(id="user-1", email="test@example.com")
        store.save_user(user)

        result = store.get_user_by_id("user-1")
        assert result is not None
        assert result.id == "user-1"

    def test_get_user_by_email(self):
        store = InMemoryUserStore()
        user = MockUser(id="user-1", email="Test@Example.com")
        store.save_user(user)

        # Should find by lowercase
        result = store.get_user_by_email("test@example.com")
        assert result is not None
        assert result.email == "Test@Example.com"

    def test_get_nonexistent_user(self):
        store = InMemoryUserStore()
        assert store.get_user_by_id("nonexistent") is None
        assert store.get_user_by_email("nonexistent@test.com") is None

    def test_api_key_lookup_legacy(self):
        store = InMemoryUserStore()
        user = MockUser(id="user-1", email="test@example.com", api_key="legacy_key_123")
        store.save_user(user)

        result = store.get_user_by_api_key("legacy_key_123")
        assert result is not None
        assert result.id == "user-1"

    def test_organization_operations(self):
        store = InMemoryUserStore()
        org = MockOrganization(id="org-1", name="Test Org", owner_id="user-1")
        store.save_organization(org)

        result = store.get_organization_by_id("org-1")
        assert result is not None
        assert result.name == "Test Org"


# ===========================================================================
# Test AuthHandler Routing
# ===========================================================================


class TestAuthHandlerRouting:
    """Tests for AuthHandler routing."""

    def test_can_handle_register(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/register") is True

    def test_can_handle_login(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/login") is True

    def test_can_handle_logout(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/logout") is True

    def test_can_handle_me(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/me") is True

    def test_can_handle_mfa_endpoints(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/mfa/setup") is True
        assert auth_handler.can_handle("/api/auth/mfa/enable") is True
        assert auth_handler.can_handle("/api/auth/mfa/verify") is True

    def test_cannot_handle_unknown(self, auth_handler):
        assert auth_handler.can_handle("/api/other/endpoint") is False


# ===========================================================================
# Test Registration
# ===========================================================================


class TestAuthHandlerRegistration:
    """Tests for user registration endpoint."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_success(self, mock_hash, mock_tokens, auth_handler):
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {
                "email": "new@example.com",
                "password": "securepass123",
                "name": "New User",
            }
        )

        result = auth_handler.handle("/api/auth/register", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_invalid_email(self, auth_handler):
        handler = make_mock_handler(
            {
                "email": "invalid",
                "password": "securepass123",
            }
        )

        result = auth_handler.handle("/api/auth/register", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body.decode())
        assert "error" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_weak_password(self, auth_handler):
        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "short",
            }
        )

        result = auth_handler.handle("/api/auth/register", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body.decode())
        assert "error" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_invalid_json(self, auth_handler):
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "5"}
        handler.rfile = BytesIO(b"invalid")
        handler.client_address = ("127.0.0.1", 12345)

        result = auth_handler.handle("/api/auth/register", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Login
# ===========================================================================


class TestAuthHandlerLogin:
    """Tests for user login endpoint."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_success(self, mock_tokens, mock_lockout, auth_handler):
        # Setup user
        user = MockUser(email="test@example.com")
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].users_by_email["test@example.com"] = "user-123"

        mock_tokens.return_value = MockTokenPair()
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "correct_password",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_wrong_password(self, mock_lockout, auth_handler):
        # Setup user
        user = MockUser(email="test@example.com")
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (1, None)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "wrong_password",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_user_not_found(self, mock_lockout, auth_handler):
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "nonexistent@example.com",
                "password": "anypassword",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        # Should return 401, not 404, to prevent email enumeration
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_account_locked(self, mock_lockout, auth_handler):
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = True
        mock_lockout_instance.get_remaining_time.return_value = 300
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "anypassword",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 429

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_inactive_account(self, mock_lockout, auth_handler):
        # Setup inactive user
        user = MockUser(email="test@example.com", is_active=False)
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "correct_password",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        # 401: disabled accounts cannot authenticate (per handler implementation)
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_mfa_pending_token")
    def test_login_mfa_required(self, mock_pending, mock_lockout, auth_handler):
        # Setup user with MFA enabled
        user = MockUser(email="test@example.com", mfa_enabled=True, mfa_secret="JBSWY3DPEHPK3PXP")
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance
        mock_pending.return_value = "pending_token_123"

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "correct_password",
            }
        )

        result = auth_handler.handle("/api/auth/login", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data.get("mfa_required") is True
        assert "pending_token" in data


# ===========================================================================
# Test Token Refresh
# ===========================================================================


class TestAuthHandlerRefresh:
    """Tests for token refresh endpoint."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_refresh_success(
        self, mock_revoke, mock_blacklist, mock_tokens, mock_validate, auth_handler
    ):
        # Setup user
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")
        mock_tokens.return_value = MockTokenPair()
        mock_blacklist_instance = MagicMock()
        mock_blacklist.return_value = mock_blacklist_instance

        handler = make_mock_handler({"refresh_token": "valid_refresh_token"})

        result = auth_handler.handle("/api/auth/refresh", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_invalid_token(self, mock_validate, auth_handler):
        mock_validate.return_value = None

        handler = make_mock_handler({"refresh_token": "invalid_token"})

        result = auth_handler.handle("/api/auth/refresh", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_refresh_missing_token(self, auth_handler):
        handler = make_mock_handler({"refresh_token": ""})

        result = auth_handler.handle("/api/auth/refresh", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Logout
# ===========================================================================


class TestAuthHandlerLogout:
    """Tests for logout endpoints."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_success(
        self, mock_revoke, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_revoke.return_value = True
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist.return_value = mock_blacklist_instance

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/logout", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "logged out" in data.get("message", "").lower()

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_logout_not_authenticated(self, mock_auth, auth_handler):
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/logout", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_success(
        self, mock_revoke, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        # Setup user store with increment_token_version
        user_store = auth_handler.ctx["user_store"]
        user_store.users["user-123"] = MockUser()

        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/logout-all", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data.get("sessions_invalidated") is True


# ===========================================================================
# Test Get/Update User Info
# ===========================================================================


class TestAuthHandlerUserInfo:
    """Tests for user info endpoints."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_success(self, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="GET")

        result = auth_handler.handle("/api/auth/me", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "user" in data

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_with_org(self, mock_auth, auth_handler):
        org = MockOrganization()
        user = MockUser(org_id="org-123")
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].orgs["org-123"] = org

        mock_auth.return_value = MockAuthContext(org_id="org-123")

        handler = make_mock_handler(method="GET")

        result = auth_handler.handle("/api/auth/me", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "organization" in data
        assert data["organization"] is not None

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_not_authenticated(self, mock_auth, auth_handler):
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler(method="GET")

        result = auth_handler.handle("/api/auth/me", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_update_me_success(self, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"name": "Updated Name"}, method="PUT")

        result = auth_handler.handle("/api/auth/me", {}, handler, "PUT")

        assert result is not None
        assert result.status_code == 200
        # Check that user was updated
        assert auth_handler.ctx["user_store"].users["user-123"].name == "Updated Name"


# ===========================================================================
# Test Password Change
# ===========================================================================


class TestAuthHandlerPasswordChange:
    """Tests for password change endpoint."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.models.hash_password")
    def test_change_password_success(self, mock_hash, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()
        mock_hash.return_value = ("new_hash", "new_salt")

        handler = make_mock_handler(
            {
                "current_password": "correct_password",
                "new_password": "new_secure_password",
            }
        )

        result = auth_handler.handle("/api/auth/password", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "success" in data.get("message", "").lower()

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_wrong_current(self, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(
            {
                "current_password": "wrong_password",
                "new_password": "new_secure_password",
            }
        )

        result = auth_handler.handle("/api/auth/password", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_weak_new(self, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(
            {
                "current_password": "correct_password",
                "new_password": "short",
            }
        )

        result = auth_handler.handle("/api/auth/password", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test API Key Operations
# ===========================================================================


class TestAuthHandlerApiKey:
    """Tests for API key endpoints."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_generate_api_key_success(self, mock_auth, auth_handler):
        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/api-key", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "api_key" in data
        assert "prefix" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_generate_api_key_tier_restricted(self, mock_auth, auth_handler):
        # User in org without API access
        org = MockOrganization()
        org.limits.api_access = False
        user = MockUser(org_id="org-123")
        auth_handler.ctx["user_store"].users["user-123"] = user
        auth_handler.ctx["user_store"].orgs["org-123"] = org

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/api-key", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 403

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_revoke_api_key_success(self, mock_auth, auth_handler):
        user = MockUser(api_key="existing_key")
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({}, method="DELETE")

        result = auth_handler.handle("/api/auth/api-key", {}, handler, "DELETE")

        assert result is not None
        assert result.status_code == 200
        # Check key was cleared
        assert auth_handler.ctx["user_store"].users["user-123"].api_key is None


# ===========================================================================
# Test MFA Operations
# ===========================================================================


class TestAuthHandlerMFA:
    """Tests for MFA endpoints."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_success(self, mock_auth, auth_handler):
        pytest.importorskip("pyotp")

        user = MockUser()
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "secret" in data
        assert "provisioning_uri" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_already_enabled(self, mock_auth, auth_handler):
        pytest.importorskip("pyotp")

        user = MockUser(mfa_enabled=True)
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_success(self, mock_auth, auth_handler):
        pyotp = pytest.importorskip("pyotp")

        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_secret=secret)
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        # Generate valid TOTP code
        totp = pyotp.TOTP(secret)
        code = totp.now()

        handler = make_mock_handler({"code": code})

        result = auth_handler.handle("/api/auth/mfa/enable", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_invalid_code(self, mock_auth, auth_handler):
        pytest.importorskip("pyotp")

        user = MockUser(mfa_secret="JBSWY3DPEHPK3PXP")
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"code": "000000"})

        result = auth_handler.handle("/api/auth/mfa/enable", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_disable_with_password(self, mock_auth, auth_handler):
        pytest.importorskip("pyotp")

        user = MockUser(mfa_enabled=True, mfa_secret="JBSWY3DPEHPK3PXP")
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"password": "correct_password"})

        result = auth_handler.handle("/api/auth/mfa/disable", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        # MFA should be disabled
        assert auth_handler.ctx["user_store"].users["user-123"].mfa_enabled is False

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_mfa_verify_success(self, mock_blacklist, mock_tokens, mock_pending, auth_handler):
        pyotp = pytest.importorskip("pyotp")

        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        auth_handler.ctx["user_store"].users["user-123"] = user

        mock_pending.return_value = MagicMock(sub="user-123")
        mock_tokens.return_value = MockTokenPair()
        mock_blacklist.return_value = MagicMock()

        totp = pyotp.TOTP(secret)
        code = totp.now()

        handler = make_mock_handler(
            {
                "code": code,
                "pending_token": "pending_token_123",
            }
        )

        result = auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_mfa_verify_invalid_pending_token(self, mock_pending, auth_handler):
        pytest.importorskip("pyotp")

        mock_pending.return_value = None

        handler = make_mock_handler(
            {
                "code": "123456",
                "pending_token": "invalid_token",
            }
        )

        result = auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Token Revocation
# ===========================================================================


class TestAuthHandlerRevokeToken:
    """Tests for token revocation endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_current_token(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist_instance.size.return_value = 1
        mock_blacklist.return_value = mock_blacklist_instance
        mock_revoke_persistent.return_value = True

        handler = make_mock_handler({})

        result = auth_handler.handle("/api/auth/revoke", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_specific_token(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        mock_auth.return_value = MockAuthContext()
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist_instance.size.return_value = 1
        mock_blacklist.return_value = mock_blacklist_instance
        mock_revoke_persistent.return_value = True

        handler = make_mock_handler({"token": "specific_token_to_revoke"})

        result = auth_handler.handle("/api/auth/revoke", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        mock_blacklist_instance.revoke_token.assert_called_with("specific_token_to_revoke")


# ===========================================================================
# Test Method Not Allowed
# ===========================================================================


class TestAuthHandlerMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_register_wrong_method(self, auth_handler):
        handler = make_mock_handler(method="GET")

        result = auth_handler.handle("/api/auth/register", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 405

    def test_login_wrong_method(self, auth_handler):
        handler = make_mock_handler(method="GET")

        result = auth_handler.handle("/api/auth/login", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Test Service Unavailable
# ===========================================================================


class TestAuthHandlerServiceUnavailable:
    """Tests for service unavailable scenarios."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_no_user_store(self):
        # Create handler without user_store
        auth_handler = AuthHandler({})

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "securepass123",
            }
        )

        result = auth_handler.handle("/api/auth/register", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 503
