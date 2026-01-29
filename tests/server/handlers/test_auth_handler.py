"""
Tests for the AuthHandler module.

Comprehensive tests covering:
1. Login with valid/invalid credentials
2. Token generation and validation
3. Token refresh flow
4. Session management
5. Logout handling
6. MFA verification
7. Account lockout after failed attempts
8. Password reset flow
9. Error responses for various failure modes
10. Handler routing for all auth endpoints
11. can_handle method for static and dynamic routes
12. ROUTES attribute
13. Rate limiting decorators
14. Response formatting
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ===========================================================================
# Test Fixtures and Mocks
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
    token_version: int = 0

    def verify_password(self, password: str) -> bool:
        """Mock password verification."""
        return password == "correct_password"

    def verify_api_key(self, key: str) -> bool:
        """Mock API key verification."""
        if self.api_key_hash is not None and key.startswith("ara_"):
            return True
        return self.api_key is not None and self.api_key == key

    def generate_api_key(self, expires_days: int = 365) -> str:
        """Mock API key generation."""
        self.api_key_prefix = "ara_test"
        self.api_key_hash = "hashed_key"
        self.api_key_created_at = datetime.now(timezone.utc)
        self.api_key_expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
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
    authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"
    error_reason: str | None = None
    client_ip: str = "127.0.0.1"
    permissions: set = None
    roles: set = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {
                "*",
                "admin",
                "authentication.read",
                "authentication.write",
                "authentication.revoke",
                "authentication.create",
                "authentication.update",
                "session.list_active",
                "session.revoke",
                "api_key.create",
                "api_key.revoke",
            }
        if self.roles is None:
            self.roles = {"admin", "owner"}
        self.authenticated = self.is_authenticated


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


@dataclass
class MockSession:
    """Mock session object."""

    session_id: str = "session-123"
    user_id: str = "user-123"
    device_info: str = "Chrome/Windows"
    ip_address: str = "127.0.0.1"
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "device_info": self.device_info,
            "ip_address": self.ip_address,
            "last_activity": self.last_activity.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.users_by_email: dict[str, str] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self.token_versions: dict[str, int] = {}
        self.failed_attempts: dict[str, int] = {}
        self.lockout_until: dict[str, datetime | None] = {}

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
        user = self.users.get(owner_id)
        if user:
            user.org_id = org.id
        return org

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def increment_token_version(self, user_id: str) -> int:
        self.token_versions[user_id] = self.token_versions.get(user_id, 0) + 1
        return self.token_versions[user_id]

    def is_account_locked(self, email: str) -> tuple[bool, datetime | None, int]:
        """Check if account is locked."""
        lockout = self.lockout_until.get(email)
        if lockout and lockout > datetime.now(timezone.utc):
            return True, lockout, self.failed_attempts.get(email, 0)
        return False, None, self.failed_attempts.get(email, 0)

    def record_failed_login(self, email: str) -> tuple[int, datetime | None]:
        """Record a failed login attempt."""
        self.failed_attempts[email] = self.failed_attempts.get(email, 0) + 1
        if self.failed_attempts[email] >= 5:
            self.lockout_until[email] = datetime.now(timezone.utc) + timedelta(minutes=15)
            return self.failed_attempts[email], self.lockout_until[email]
        return self.failed_attempts[email], None

    def reset_failed_login_attempts(self, email: str) -> None:
        """Reset failed login attempts."""
        self.failed_attempts[email] = 0
        self.lockout_until[email] = None


class MockSessionManager:
    """Mock session manager for testing."""

    def __init__(self):
        self.sessions: dict[str, list[MockSession]] = {}

    def list_sessions(self, user_id: str) -> list[MockSession]:
        return self.sessions.get(user_id, [])

    def get_session(self, user_id: str, session_id: str) -> MockSession | None:
        sessions = self.sessions.get(user_id, [])
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None

    def revoke_session(self, user_id: str, session_id: str) -> bool:
        sessions = self.sessions.get(user_id, [])
        for i, session in enumerate(sessions):
            if session.session_id == session_id:
                sessions.pop(i)
                return True
        return False


class MockLockoutTracker:
    """Mock lockout tracker for testing."""

    def __init__(self):
        self.locked_emails: set[str] = set()
        self.locked_ips: set[str] = set()
        self.failures: dict[str, int] = {}

    def is_locked(self, email: str = None, ip: str = None) -> bool:
        if email and email in self.locked_emails:
            return True
        if ip and ip in self.locked_ips:
            return True
        return False

    def get_remaining_time(self, email: str = None, ip: str = None) -> int:
        return 300  # 5 minutes

    def record_failure(self, email: str = None, ip: str = None) -> tuple[int, int | None]:
        key = email or ip
        self.failures[key] = self.failures.get(key, 0) + 1
        if self.failures[key] >= 5:
            if email:
                self.locked_emails.add(email)
            return self.failures[key], 300
        return self.failures[key], None

    def reset(self, email: str = None, ip: str = None) -> None:
        key = email or ip
        if key in self.failures:
            del self.failures[key]
        if email in self.locked_emails:
            self.locked_emails.remove(email)
        if ip in self.locked_ips:
            self.locked_ips.remove(ip)


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


def maybe_await(result):
    """Await result if it's a coroutine."""
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiter state before each test."""
    try:
        from aragora.server.handlers.utils.rate_limit import _limiters

        for limiter in _limiters.values():
            limiter.clear()
        yield
        for limiter in _limiters.values():
            limiter.clear()
    except ImportError:
        yield


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    return MockUserStore()


@pytest.fixture
def auth_handler(mock_user_store):
    """Create AuthHandler with mock context."""
    from aragora.server.handlers.auth.handler import AuthHandler

    ctx = {"user_store": mock_user_store}
    return AuthHandler(ctx)


@pytest.fixture
def mock_lockout_tracker():
    """Create a mock lockout tracker."""
    return MockLockoutTracker()


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    return MockSessionManager()


# ===========================================================================
# Test AuthHandler Import and Module
# ===========================================================================


class TestAuthHandlerImport:
    """Tests for importing AuthHandler."""

    def test_can_import_handler(self):
        """AuthHandler can be imported."""
        from aragora.server.handlers.auth.handler import AuthHandler

        assert AuthHandler is not None

    def test_handler_in_all(self):
        """AuthHandler is in __all__."""
        from aragora.server.handlers.auth import handler

        assert "AuthHandler" in handler.__all__

    def test_handler_is_base_handler_subclass(self):
        """AuthHandler is a BaseHandler subclass."""
        from aragora.server.handlers.auth.handler import AuthHandler
        from aragora.server.handlers.base import BaseHandler

        assert issubclass(AuthHandler, BaseHandler)


# ===========================================================================
# Test AuthHandler Routes
# ===========================================================================


class TestAuthHandlerRoutes:
    """Tests for AuthHandler ROUTES attribute."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_routes_is_list(self, handler):
        """ROUTES is a list."""
        assert isinstance(handler.ROUTES, list)

    def test_routes_not_empty(self, handler):
        """ROUTES is not empty."""
        assert len(handler.ROUTES) > 0

    def test_register_route_in_routes(self, handler):
        """Register route is in ROUTES."""
        assert "/api/auth/register" in handler.ROUTES

    def test_login_route_in_routes(self, handler):
        """Login route is in ROUTES."""
        assert "/api/auth/login" in handler.ROUTES

    def test_logout_route_in_routes(self, handler):
        """Logout route is in ROUTES."""
        assert "/api/auth/logout" in handler.ROUTES

    def test_logout_all_route_in_routes(self, handler):
        """Logout-all route is in ROUTES."""
        assert "/api/auth/logout-all" in handler.ROUTES

    def test_refresh_route_in_routes(self, handler):
        """Refresh route is in ROUTES."""
        assert "/api/auth/refresh" in handler.ROUTES

    def test_revoke_route_in_routes(self, handler):
        """Revoke route is in ROUTES."""
        assert "/api/auth/revoke" in handler.ROUTES

    def test_me_route_in_routes(self, handler):
        """Me route is in ROUTES."""
        assert "/api/auth/me" in handler.ROUTES

    def test_password_route_in_routes(self, handler):
        """Password route is in ROUTES."""
        assert "/api/auth/password" in handler.ROUTES

    def test_api_key_route_in_routes(self, handler):
        """API key route is in ROUTES."""
        assert "/api/auth/api-key" in handler.ROUTES

    def test_mfa_routes_in_routes(self, handler):
        """MFA routes are in ROUTES."""
        assert "/api/auth/mfa/setup" in handler.ROUTES
        assert "/api/auth/mfa/enable" in handler.ROUTES
        assert "/api/auth/mfa/disable" in handler.ROUTES
        assert "/api/auth/mfa/verify" in handler.ROUTES
        assert "/api/auth/mfa/backup-codes" in handler.ROUTES

    def test_sessions_route_in_routes(self, handler):
        """Sessions route is in ROUTES."""
        assert "/api/auth/sessions" in handler.ROUTES

    def test_sessions_wildcard_in_routes(self, handler):
        """Sessions wildcard route is in ROUTES."""
        assert "/api/auth/sessions/*" in handler.ROUTES

    def test_password_reset_routes_in_routes(self, handler):
        """Password reset routes are in ROUTES."""
        assert "/api/auth/password/forgot" in handler.ROUTES
        assert "/api/auth/password/reset" in handler.ROUTES
        assert "/api/auth/forgot-password" in handler.ROUTES
        assert "/api/auth/reset-password" in handler.ROUTES


# ===========================================================================
# Test can_handle Method
# ===========================================================================


class TestAuthHandlerCanHandle:
    """Tests for can_handle method."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_can_handle_register(self, handler):
        """Handler can handle register endpoint."""
        assert handler.can_handle("/api/auth/register") is True

    def test_can_handle_login(self, handler):
        """Handler can handle login endpoint."""
        assert handler.can_handle("/api/auth/login") is True

    def test_can_handle_logout(self, handler):
        """Handler can handle logout endpoint."""
        assert handler.can_handle("/api/auth/logout") is True

    def test_can_handle_me(self, handler):
        """Handler can handle me endpoint."""
        assert handler.can_handle("/api/auth/me") is True

    def test_can_handle_mfa_setup(self, handler):
        """Handler can handle MFA setup endpoint."""
        assert handler.can_handle("/api/auth/mfa/setup") is True

    def test_can_handle_sessions(self, handler):
        """Handler can handle sessions endpoint."""
        assert handler.can_handle("/api/auth/sessions") is True

    def test_can_handle_session_id(self, handler):
        """Handler can handle session ID endpoint (wildcard)."""
        assert handler.can_handle("/api/auth/sessions/abc123") is True

    def test_can_handle_session_uuid(self, handler):
        """Handler can handle session UUID endpoint."""
        assert handler.can_handle("/api/auth/sessions/550e8400-e29b-41d4-a716-446655440000") is True

    def test_can_handle_api_keys_wildcard(self, handler):
        """Handler can handle API keys wildcard endpoint."""
        assert handler.can_handle("/api/auth/api-keys/ara_prefix123") is True

    def test_cannot_handle_unrelated_path(self, handler):
        """Handler does not handle unrelated paths."""
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_auth_path(self, handler):
        """Handler does not handle partial auth paths."""
        assert handler.can_handle("/api/v1/auth") is False

    def test_cannot_handle_auth_unknown(self, handler):
        """Handler does not handle unknown auth endpoints."""
        assert handler.can_handle("/api/auth/unknown") is False


# ===========================================================================
# Test Login with Valid/Invalid Credentials
# ===========================================================================


class TestLoginValidInvalidCredentials:
    """Tests for login with valid and invalid credentials."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_with_valid_credentials(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Login succeeds with valid credentials."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

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

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_with_wrong_password(self, mock_lockout, auth_handler, mock_user_store):
        """Login fails with wrong password."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

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

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 401
        data = json.loads(result.body.decode())
        assert "error" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_with_nonexistent_user(self, mock_lockout, auth_handler, mock_user_store):
        """Login fails for nonexistent user without revealing user existence."""
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (1, None)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "nonexistent@example.com",
                "password": "anypassword",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        # Should return 401, not 404, to prevent email enumeration
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_login_missing_email(self, auth_handler):
        """Login fails when email is missing."""
        handler = make_mock_handler({"password": "somepassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_login_missing_password(self, auth_handler):
        """Login fails when password is missing."""
        handler = make_mock_handler({"email": "test@example.com"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_disabled_account(self, mock_lockout, auth_handler, mock_user_store):
        """Login fails for disabled account."""
        user = MockUser(email="test@example.com", is_active=False)
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "correct_password",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Token Generation and Validation
# ===========================================================================


class TestTokenGeneration:
    """Tests for token generation."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_generates_token_pair(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Login returns access and refresh tokens."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_tokens.return_value = MockTokenPair(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_in=3600,
        )
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "correct_password",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data["tokens"]["access_token"] == "test_access_token"
        assert data["tokens"]["refresh_token"] == "test_refresh_token"
        assert data["tokens"]["expires_in"] == 3600

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_generates_token_pair(self, mock_hash, mock_tokens, auth_handler):
        """Registration returns access and refresh tokens."""
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {
                "email": "new@example.com",
                "password": "securepass123",
                "name": "New User",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 201
        data = json.loads(result.body.decode())
        assert "tokens" in data


# ===========================================================================
# Test Token Refresh Flow
# ===========================================================================


class TestTokenRefreshFlow:
    """Tests for token refresh flow."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_refresh_with_valid_token(
        self, mock_revoke, mock_blacklist, mock_tokens, mock_validate, auth_handler, mock_user_store
    ):
        """Token refresh succeeds with valid refresh token."""
        user = MockUser()
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")
        mock_tokens.return_value = MockTokenPair(
            access_token="new_access_token",
            refresh_token="new_refresh_token",
        )
        mock_blacklist_instance = MagicMock()
        mock_blacklist.return_value = mock_blacklist_instance

        handler = make_mock_handler({"refresh_token": "valid_refresh_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data
        assert data["tokens"]["access_token"] == "new_access_token"

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_with_invalid_token(self, mock_validate, auth_handler):
        """Token refresh fails with invalid token."""
        mock_validate.return_value = None

        handler = make_mock_handler({"refresh_token": "invalid_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_refresh_with_missing_token(self, auth_handler):
        """Token refresh fails when token is missing."""
        handler = make_mock_handler({"refresh_token": ""})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_revokes_old_token(self, mock_validate, auth_handler, mock_user_store):
        """Token refresh revokes the old refresh token."""
        user = MockUser()
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")

        with (
            patch("aragora.billing.jwt_auth.create_token_pair") as mock_tokens,
            patch("aragora.billing.jwt_auth.get_token_blacklist") as mock_blacklist,
            patch("aragora.billing.jwt_auth.revoke_token_persistent") as mock_revoke,
        ):
            mock_tokens.return_value = MockTokenPair()
            mock_blacklist_instance = MagicMock()
            mock_blacklist.return_value = mock_blacklist_instance

            handler = make_mock_handler({"refresh_token": "old_refresh_token"})
            result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

            assert result.status_code == 200
            mock_revoke.assert_called_with("old_refresh_token")

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_for_disabled_user(self, mock_validate, auth_handler, mock_user_store):
        """Token refresh fails for disabled user."""
        user = MockUser(is_active=False)
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")

        handler = make_mock_handler({"refresh_token": "valid_refresh_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 401


# ===========================================================================
# Test Session Management
# ===========================================================================


class TestSessionManagement:
    """Tests for session management."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_list_sessions_success(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """List sessions returns all user sessions."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="current-session-id")

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [
            MockSession(session_id="session-1"),
            MockSession(session_id="session-2"),
        ]
        mock_get_manager.return_value = mock_manager

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/sessions", {}, handler, "GET"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "sessions" in data
        assert data["total"] == 2

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_list_sessions_unauthenticated(self, mock_auth, auth_handler):
        """List sessions fails when not authenticated."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/sessions", {}, handler, "GET"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_revoke_session_success(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """Revoke session succeeds for valid session."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="different-session-id")

        mock_manager = MagicMock()
        mock_manager.get_session.return_value = MockSession(session_id="session-to-revoke")
        mock_get_manager.return_value = mock_manager

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(
            auth_handler.handle("/api/auth/sessions/session-to-revoke", {}, handler, "DELETE")
        )

        assert result.status_code == 200
        mock_manager.revoke_session.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_revoke_current_session_fails(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """Cannot revoke current session."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="current-session-id")

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(
            auth_handler.handle("/api/auth/sessions/current-session-id", {}, handler, "DELETE")
        )

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_revoke_nonexistent_session(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """Revoke fails for nonexistent session."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="different-session-id")

        mock_manager = MagicMock()
        mock_manager.get_session.return_value = None
        mock_get_manager.return_value = mock_manager

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(
            auth_handler.handle("/api/auth/sessions/nonexistent", {}, handler, "DELETE")
        )

        assert result.status_code == 404


# ===========================================================================
# Test Logout Handling
# ===========================================================================


class TestLogoutHandling:
    """Tests for logout handling."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_success(
        self, mock_revoke, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Logout succeeds and revokes token."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_revoke.return_value = True
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist.return_value = mock_blacklist_instance

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "logged out" in data.get("message", "").lower()

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_logout_unauthenticated(self, mock_auth, auth_handler):
        """Logout fails when not authenticated."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_success(
        self,
        mock_revoke,
        mock_blacklist,
        mock_extract_token,
        mock_auth,
        auth_handler,
        mock_user_store,
    ):
        """Logout all sessions succeeds."""
        mock_user_store.users["user-123"] = MockUser()
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout-all", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data.get("sessions_invalidated") is True

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_increments_token_version(
        self,
        mock_revoke,
        mock_blacklist,
        mock_extract_token,
        mock_auth,
        auth_handler,
        mock_user_store,
    ):
        """Logout all increments token version."""
        mock_user_store.users["user-123"] = MockUser()
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout-all", {}, handler, "POST"))

        assert result.status_code == 200
        # Token version should be incremented
        assert mock_user_store.token_versions.get("user-123", 0) == 1


# ===========================================================================
# Test MFA Verification
# ===========================================================================


class TestMFAVerification:
    """Tests for MFA verification."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_success(self, mock_auth, auth_handler, mock_user_store):
        """MFA setup generates secret and provisioning URI."""
        pytest.importorskip("pyotp")

        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "secret" in data
        assert "provisioning_uri" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_already_enabled(self, mock_auth, auth_handler, mock_user_store):
        """MFA setup fails when already enabled."""
        pytest.importorskip("pyotp")

        user = MockUser(mfa_enabled=True)
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_success(self, mock_auth, auth_handler, mock_user_store):
        """MFA enable succeeds with valid TOTP code."""
        pyotp = pytest.importorskip("pyotp")

        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_secret=secret)
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        totp = pyotp.TOTP(secret)
        code = totp.now()

        handler = make_mock_handler({"code": code})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/enable", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_invalid_code(self, mock_auth, auth_handler, mock_user_store):
        """MFA enable fails with invalid code."""
        pytest.importorskip("pyotp")

        user = MockUser(mfa_secret="JBSWY3DPEHPK3PXP")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"code": "000000"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/enable", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_mfa_verify_success(
        self, mock_blacklist, mock_tokens, mock_pending, auth_handler, mock_user_store
    ):
        """MFA verify succeeds with valid TOTP code."""
        pyotp = pytest.importorskip("pyotp")

        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        mock_user_store.users["user-123"] = user

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

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_mfa_verify_invalid_pending_token(self, mock_pending, auth_handler):
        """MFA verify fails with invalid pending token."""
        pytest.importorskip("pyotp")

        mock_pending.return_value = None

        handler = make_mock_handler(
            {
                "code": "123456",
                "pending_token": "invalid_token",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_disable_with_password(self, mock_auth, auth_handler, mock_user_store):
        """MFA disable succeeds with password verification."""
        pytest.importorskip("pyotp")

        user = MockUser(mfa_enabled=True, mfa_secret="JBSWY3DPEHPK3PXP")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/disable", {}, handler, "POST"))

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].mfa_enabled is False


# ===========================================================================
# Test Account Lockout After Failed Attempts
# ===========================================================================


class TestAccountLockout:
    """Tests for account lockout after failed attempts."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_account_locked_after_failures(self, mock_lockout, auth_handler, mock_user_store):
        """Account is locked after too many failed attempts."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        # Simulate lockout after failure
        mock_lockout_instance.record_failure.return_value = (5, 300)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "wrong_password",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 429
        data = json.loads(result.body.decode())
        assert "locked" in data.get("error", "").lower()

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_blocked_when_locked(self, mock_lockout, auth_handler, mock_user_store):
        """Login is blocked when account is locked."""
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

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 429

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_lockout_reset_on_success(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Lockout counter is reset on successful login."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

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

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 200
        mock_lockout_instance.reset.assert_called()


# ===========================================================================
# Test Password Reset Flow
# ===========================================================================


class TestPasswordResetFlow:
    """Tests for password reset flow."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_forgot_password_returns_501(self, auth_handler):
        """Forgot password returns 501 (not implemented)."""
        handler = make_mock_handler({"email": "test@example.com"})

        result = maybe_await(auth_handler.handle("/api/auth/forgot-password", {}, handler, "POST"))

        assert result.status_code == 501

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_reset_password_returns_501(self, auth_handler):
        """Reset password returns 501 (not implemented)."""
        handler = make_mock_handler(
            {
                "token": "reset_token",
                "new_password": "new_password123",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/reset-password", {}, handler, "POST"))

        assert result.status_code == 501

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.models.hash_password")
    def test_change_password_success(self, mock_hash, mock_auth, auth_handler, mock_user_store):
        """Password change succeeds with correct current password."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()
        mock_hash.return_value = ("new_hash", "new_salt")

        handler = make_mock_handler(
            {
                "current_password": "correct_password",
                "new_password": "new_secure_password",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "success" in data.get("message", "").lower()

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_wrong_current(self, mock_auth, auth_handler, mock_user_store):
        """Password change fails with wrong current password."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(
            {
                "current_password": "wrong_password",
                "new_password": "new_secure_password",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 401


# ===========================================================================
# Test Error Responses
# ===========================================================================


class TestErrorResponses:
    """Tests for various error responses."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_invalid_json_body(self, auth_handler):
        """Invalid JSON body returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "5"}
        handler.rfile = BytesIO(b"invalid")
        handler.client_address = ("127.0.0.1", 12345)

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_no_user_store(self):
        """Register returns 503 when user store unavailable."""
        from aragora.server.handlers.auth.handler import AuthHandler

        auth_handler = AuthHandler({})

        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "securepass123",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 503

    def test_method_not_allowed_get_on_post_endpoint(self, auth_handler):
        """GET on POST-only endpoint returns 405."""
        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "GET"))

        assert result.status_code == 405

    def test_method_not_allowed_patch(self, auth_handler):
        """PATCH on auth endpoints returns 405."""
        handler = make_mock_handler(method="PATCH")
        handler.command = "PATCH"

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "PATCH"))

        assert result.status_code == 405

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_duplicate_email(self, mock_hash, mock_tokens, auth_handler, mock_user_store):
        """Register with existing email returns 409."""
        existing_user = MockUser(email="existing@example.com")
        mock_user_store.users["user-1"] = existing_user
        mock_user_store.users_by_email["existing@example.com"] = "user-1"

        mock_hash.return_value = ("hashed", "salt")

        handler = make_mock_handler(
            {
                "email": "existing@example.com",
                "password": "securepass123",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 409


# ===========================================================================
# Test API Key Operations
# ===========================================================================


class TestAPIKeyOperations:
    """Tests for API key operations."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_generate_api_key_success(self, mock_auth, auth_handler, mock_user_store):
        """Generate API key succeeds."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/api-key", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "api_key" in data
        assert "prefix" in data

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_revoke_api_key_success(self, mock_auth, auth_handler, mock_user_store):
        """Revoke API key succeeds."""
        user = MockUser(api_key_hash="existing_hash", api_key_prefix="ara_")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({}, method="DELETE")

        result = maybe_await(auth_handler.handle("/api/auth/api-key", {}, handler, "DELETE"))

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].api_key_hash is None

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_list_api_keys(self, mock_auth, auth_handler, mock_user_store):
        """List API keys returns user keys."""
        user = MockUser(
            api_key_prefix="ara_test",
            api_key_created_at=datetime.now(timezone.utc),
            api_key_expires_at=datetime.now(timezone.utc) + timedelta(days=365),
        )
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/api-keys", {}, handler, "GET"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "keys" in data
        assert data["count"] == 1


# ===========================================================================
# Test Routing
# ===========================================================================


class TestAuthHandlerRouting:
    """Tests for handle method routing."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with mock store."""
        from aragora.server.handlers.auth.handler import AuthHandler

        mock_store = MagicMock()
        return AuthHandler(server_context={"user_store": mock_store})

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=b"{}")
        mock.headers = {"Content-Length": "2"}
        mock.command = "POST"
        return mock

    @pytest.mark.asyncio
    async def test_register_routes_to_handler(self, handler, mock_http):
        """Register POST routes to _handle_register."""
        with patch.object(handler, "_handle_register") as mock_method:
            mock_method.return_value = MagicMock()
            await handler.handle("/api/auth/register", {}, mock_http, "POST")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_routes_to_handler(self, handler, mock_http):
        """Login POST routes to _handle_login."""
        with patch.object(handler, "_handle_login") as mock_method:
            mock_method.return_value = MagicMock()
            await handler.handle("/api/auth/login", {}, mock_http, "POST")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_me_get_routes_to_handler(self, handler, mock_http):
        """Me GET routes to _handle_get_me."""
        mock_http.command = "GET"
        with patch.object(handler, "_handle_get_me") as mock_method:
            mock_method.return_value = MagicMock()
            await handler.handle("/api/auth/me", {}, mock_http, "GET")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_me_put_routes_to_handler(self, handler, mock_http):
        """Me PUT routes to _handle_update_me."""
        mock_http.command = "PUT"
        with patch.object(handler, "_handle_update_me") as mock_method:
            mock_method.return_value = MagicMock()
            await handler.handle("/api/auth/me", {}, mock_http, "PUT")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_delete_routes_with_id(self, handler, mock_http):
        """Session DELETE routes with session ID."""
        mock_http.command = "DELETE"
        with patch.object(handler, "_handle_revoke_session") as mock_method:
            mock_method.return_value = MagicMock()
            await handler.handle("/api/auth/sessions/abc123", {}, mock_http, "DELETE")
            mock_method.assert_called_once_with(mock_http, "abc123")


# ===========================================================================
# Test User Info Endpoints
# ===========================================================================


class TestUserInfoEndpoints:
    """Tests for user info endpoints."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_success(self, mock_auth, auth_handler, mock_user_store):
        """Get me returns user info."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "GET"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "user" in data

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_with_organization(self, mock_auth, auth_handler, mock_user_store):
        """Get me returns organization when present."""
        org = MockOrganization()
        user = MockUser(org_id="org-123")
        mock_user_store.users["user-123"] = user
        mock_user_store.orgs["org-123"] = org
        mock_auth.return_value = MockAuthContext(org_id="org-123")

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "GET"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "organization" in data
        assert data["organization"] is not None

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_update_me_success(self, mock_auth, auth_handler, mock_user_store):
        """Update me succeeds."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"name": "Updated Name"}, method="PUT")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "PUT"))

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].name == "Updated Name"


# ===========================================================================
# Test Registration
# ===========================================================================


class TestRegistration:
    """Tests for user registration."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_success(self, mock_hash, mock_tokens, auth_handler):
        """Registration succeeds with valid data."""
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {
                "email": "new@example.com",
                "password": "securepass123",
                "name": "New User",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 201
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_invalid_email(self, auth_handler):
        """Registration fails with invalid email."""
        handler = make_mock_handler(
            {
                "email": "invalid",
                "password": "securepass123",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    def test_register_weak_password(self, auth_handler):
        """Registration fails with weak password."""
        handler = make_mock_handler(
            {
                "email": "test@example.com",
                "password": "short",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400


# ===========================================================================
# Test Token Revocation
# ===========================================================================


class TestTokenRevocation:
    """Tests for token revocation."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_current_token(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Revoke current token succeeds."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist_instance.size.return_value = 1
        mock_blacklist.return_value = mock_blacklist_instance
        mock_revoke_persistent.return_value = True

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/revoke", {}, handler, "POST"))

        assert result.status_code == 200

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_specific_token(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Revoke specific token succeeds."""
        mock_auth.return_value = MockAuthContext()
        mock_blacklist_instance = MagicMock()
        mock_blacklist_instance.revoke_token.return_value = True
        mock_blacklist_instance.size.return_value = 1
        mock_blacklist.return_value = mock_blacklist_instance
        mock_revoke_persistent.return_value = True

        handler = make_mock_handler({"token": "specific_token"})

        result = maybe_await(auth_handler.handle("/api/auth/revoke", {}, handler, "POST"))

        assert result.status_code == 200
        mock_blacklist_instance.revoke_token.assert_called_with("specific_token")


# ===========================================================================
# Test Module Exports
# ===========================================================================


class TestAuthHandlerModuleExports:
    """Tests for module exports."""

    def test_all_exports_auth_handler(self):
        """__all__ exports AuthHandler."""
        from aragora.server.handlers.auth import handler

        assert "AuthHandler" in handler.__all__


# ===========================================================================
# Test MFA Backup Codes
# ===========================================================================


class TestMFABackupCodes:
    """Tests for MFA backup codes."""

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_mfa_verify_with_backup_code(
        self, mock_blacklist, mock_tokens, mock_pending, auth_handler, mock_user_store
    ):
        """MFA verify succeeds with valid backup code."""
        pytest.importorskip("pyotp")
        import hashlib
        import json as json_module

        backup_code = "testcode"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        user = MockUser(
            mfa_enabled=True,
            mfa_secret="JBSWY3DPEHPK3PXP",
            mfa_backup_codes=json_module.dumps([backup_hash]),
        )
        mock_user_store.users["user-123"] = user

        mock_pending.return_value = MagicMock(sub="user-123")
        mock_tokens.return_value = MockTokenPair()
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler(
            {
                "code": backup_code,
                "pending_token": "pending_token_123",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kwargs: lambda fn: fn)
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_regenerate_backup_codes(self, mock_auth, auth_handler, mock_user_store):
        """Regenerate backup codes succeeds."""
        pyotp = pytest.importorskip("pyotp")

        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        totp = pyotp.TOTP(secret)
        code = totp.now()

        handler = make_mock_handler({"code": code})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/backup-codes", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10
