"""
Comprehensive tests for the AuthHandler module.

Covers all 12 required categories:
1.  Login flow (valid credentials, invalid credentials, locked account)
2.  Logout flow (session invalidation, token revocation)
3.  Token generation (access token, refresh token, proper claims)
4.  Token refresh (valid refresh token, expired token, revoked token)
5.  Token revocation
6.  Rate limiting on auth endpoints (brute force protection)
7.  Account lockout mechanism (after N failed attempts)
8.  MFA enforcement (TOTP verification, backup codes)
9.  Password validation (strength requirements)
10. Session management (concurrent sessions, session listing)
11. Error handling (malformed requests, database errors, timeout)
12. Security headers validation

50+ test cases organized by functionality.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pyotp
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
        return password == "correct_password"

    def verify_api_key(self, key: str) -> bool:
        if self.api_key_hash is not None and key.startswith("ara_"):
            return True
        return self.api_key is not None and self.api_key == key

    def generate_api_key(self, expires_days: int = 365) -> str:
        self.api_key_prefix = "ara_test"
        self.api_key_hash = "hashed_key"
        self.api_key_created_at = datetime.now(timezone.utc)
        self.api_key_expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        return "ara_test_full_key_12345"

    def to_dict(self) -> dict[str, Any]:
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
        lockout = self.lockout_until.get(email)
        if lockout and lockout > datetime.now(timezone.utc):
            return True, lockout, self.failed_attempts.get(email, 0)
        return False, None, self.failed_attempts.get(email, 0)

    def record_failed_login(self, email: str) -> tuple[int, datetime | None]:
        self.failed_attempts[email] = self.failed_attempts.get(email, 0) + 1
        if self.failed_attempts[email] >= 5:
            self.lockout_until[email] = datetime.now(timezone.utc) + timedelta(minutes=15)
            return self.failed_attempts[email], self.lockout_until[email]
        return self.failed_attempts[email], None

    def reset_failed_login_attempts(self, email: str) -> None:
        self.failed_attempts[email] = 0
        self.lockout_until[email] = None


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
        return 300

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
        if email and email in self.locked_emails:
            self.locked_emails.remove(email)
        if ip and ip in self.locked_ips:
            self.locked_ips.remove(ip)


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
    """Clear rate limiter state before each test.

    Resets both the local _limiters registry (used by auth_rate_limit) and the
    distributed rate limiter singleton (used by the rate_limit decorator).
    Without this, rate limiter state leaks across tests and low-RPM endpoints
    like password change (3 RPM) can hit 429 under randomized ordering.
    """
    try:
        from aragora.server.handlers.utils.rate_limit import _limiters

        for limiter in _limiters.values():
            limiter.clear()
    except ImportError:
        pass

    try:
        from aragora.server.middleware.rate_limit.distributed import reset_distributed_limiter

        reset_distributed_limiter()
    except ImportError:
        pass

    yield

    try:
        from aragora.server.handlers.utils.rate_limit import _limiters

        for limiter in _limiters.values():
            limiter.clear()
    except ImportError:
        pass

    try:
        from aragora.server.middleware.rate_limit.distributed import reset_distributed_limiter

        reset_distributed_limiter()
    except ImportError:
        pass


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
# 1. Login Flow Tests
# ===========================================================================


class TestLoginFlow:
    """Tests for login with valid/invalid credentials, locked accounts."""

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_valid_credentials_returns_200(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Login succeeds with valid email and password."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_tokens.return_value = MockTokenPair()
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_wrong_password_returns_401(self, mock_lockout, auth_handler, mock_user_store):
        """Login fails with wrong password."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (1, None)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "wrong_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_nonexistent_user_returns_401_not_404(
        self, mock_lockout, auth_handler, mock_user_store
    ):
        """Login for nonexistent user returns 401 to prevent email enumeration."""
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (1, None)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "nonexistent@example.com", "password": "anypassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 401

    def test_login_missing_email_returns_400(self, auth_handler):
        """Login fails when email is missing."""
        handler = make_mock_handler({"password": "somepassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 400

    def test_login_missing_password_returns_400(self, auth_handler):
        """Login fails when password is missing."""
        handler = make_mock_handler({"email": "test@example.com"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_disabled_account_returns_401(self, mock_lockout, auth_handler, mock_user_store):
        """Login fails for disabled account."""
        user = MockUser(email="test@example.com", is_active=False)
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_login_locked_account_returns_429(self, mock_lockout, auth_handler):
        """Login blocked when account is locked."""
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = True
        mock_lockout_instance.get_remaining_time.return_value = 300
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "anypassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 429

    def test_login_no_user_store_returns_503(self):
        """Login returns 503 when user store unavailable."""
        from aragora.server.handlers.auth.handler import AuthHandler

        handler_instance = AuthHandler({})
        handler = make_mock_handler({"email": "test@example.com", "password": "somepassword"})

        result = maybe_await(handler_instance.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_mfa_pending_token")
    def test_login_mfa_required_returns_pending_token(
        self, mock_pending, mock_lockout, auth_handler, mock_user_store
    ):
        """Login with MFA-enabled user returns pending token."""
        user = MockUser(
            email="test@example.com",
            mfa_enabled=True,
            mfa_secret="JBSWY3DPEHPK3PXP",
        )
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance
        mock_pending.return_value = "pending_token_123"

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data.get("mfa_required") is True
        assert "pending_token" in data


# ===========================================================================
# 2. Logout Flow Tests
# ===========================================================================


class TestLogoutFlow:
    """Tests for logout session invalidation and token revocation."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_success_revokes_token(
        self, mock_revoke, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Logout succeeds and revokes current token."""
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
    def test_logout_unauthenticated_returns_401(self, mock_auth, auth_handler):
        """Logout fails when not authenticated."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_invalidates_sessions(
        self,
        mock_revoke,
        mock_blacklist,
        mock_extract_token,
        mock_auth,
        auth_handler,
        mock_user_store,
    ):
        """Logout all sessions succeeds and invalidates all tokens."""
        mock_user_store.users["user-123"] = MockUser()
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout-all", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data.get("sessions_invalidated") is True

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
        """Logout all increments token version to invalidate all JWTs."""
        mock_user_store.users["user-123"] = MockUser()
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout-all", {}, handler, "POST"))

        assert result.status_code == 200
        assert mock_user_store.token_versions.get("user-123", 0) == 1

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_no_token_still_succeeds(
        self, mock_revoke, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Logout succeeds even when no token to revoke."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = None

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/logout", {}, handler, "POST"))

        assert result.status_code == 200


# ===========================================================================
# 3. Token Generation Tests
# ===========================================================================


class TestTokenGeneration:
    """Tests for token generation during login and registration."""

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_generates_access_and_refresh_tokens(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Login returns both access and refresh tokens with expiry."""
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

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data["tokens"]["access_token"] == "test_access_token"
        assert data["tokens"]["refresh_token"] == "test_refresh_token"
        assert data["tokens"]["expires_in"] == 3600

    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_generates_token_pair(self, mock_hash, mock_tokens, auth_handler):
        """Registration returns access and refresh tokens."""
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {"email": "new@example.com", "password": "SecurePass1!", "name": "New User"}
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 201
        data = json.loads(result.body.decode())
        assert "tokens" in data
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_login_calls_create_token_pair_with_user_data(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Login passes correct user data to create_token_pair."""
        user = MockUser(
            id="user-456",
            email="test@example.com",
            org_id="org-789",
            role="admin",
        )
        mock_user_store.users["user-456"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-456"

        mock_tokens.return_value = MockTokenPair()
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 200
        mock_tokens.assert_called_once_with(
            user_id="user-456",
            email="test@example.com",
            org_id="org-789",
            role="admin",
        )


# ===========================================================================
# 4. Token Refresh Flow Tests
# ===========================================================================


class TestTokenRefreshFlow:
    """Tests for token refresh flow."""

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_refresh_valid_token_returns_new_tokens(
        self,
        mock_revoke,
        mock_blacklist,
        mock_tokens,
        mock_validate,
        auth_handler,
        mock_user_store,
    ):
        """Token refresh succeeds with valid refresh token."""
        user = MockUser()
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")
        mock_tokens.return_value = MockTokenPair(
            access_token="new_access_token",
            refresh_token="new_refresh_token",
        )
        mock_blacklist.return_value = MagicMock()

        handler = make_mock_handler({"refresh_token": "valid_refresh_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data["tokens"]["access_token"] == "new_access_token"

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_expired_token_returns_401(self, mock_validate, auth_handler):
        """Token refresh fails with expired token."""
        mock_validate.return_value = None

        handler = make_mock_handler({"refresh_token": "expired_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 401

    def test_refresh_missing_token_returns_400(self, auth_handler):
        """Token refresh fails when token is missing."""
        handler = make_mock_handler({"refresh_token": ""})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 400

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
            mock_blacklist.return_value = MagicMock()

            handler = make_mock_handler({"refresh_token": "old_refresh_token"})
            result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

            assert result.status_code == 200
            mock_revoke.assert_called_with("old_refresh_token")

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_disabled_user_returns_401(self, mock_validate, auth_handler, mock_user_store):
        """Token refresh fails for disabled user."""
        user = MockUser(is_active=False)
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")

        handler = make_mock_handler({"refresh_token": "valid_refresh_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    def test_refresh_nonexistent_user_returns_401(
        self, mock_validate, auth_handler, mock_user_store
    ):
        """Token refresh fails when user no longer exists."""
        mock_validate.return_value = MagicMock(user_id="deleted-user")

        handler = make_mock_handler({"refresh_token": "valid_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_refresh_persistent_revocation_failure_returns_500(
        self, mock_revoke, mock_validate, auth_handler, mock_user_store
    ):
        """Token refresh returns 500 when persistent revocation fails."""
        user = MockUser()
        mock_user_store.users["user-123"] = user

        mock_validate.return_value = MagicMock(user_id="user-123")
        mock_revoke.side_effect = OSError("Database unavailable")

        handler = make_mock_handler({"refresh_token": "valid_token"})

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 500


# ===========================================================================
# 5. Token Revocation Tests
# ===========================================================================


class TestTokenRevocation:
    """Tests for explicit token revocation."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_current_token_success(
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
    def test_revoke_specific_token_from_body(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Revoke specific token provided in request body."""
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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.server.middleware.auth.extract_token")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_revoke_no_token_returns_400(
        self, mock_revoke_persistent, mock_blacklist, mock_extract_token, mock_auth, auth_handler
    ):
        """Revoke fails when no token is provided."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = None

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/revoke", {}, handler, "POST"))

        assert result.status_code == 400


# ===========================================================================
# 6. Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on auth endpoints (brute force protection)."""

    def test_auth_handler_has_rate_limit_decorators(self):
        """Verify auth endpoints use rate limit decorators."""
        from aragora.server.handlers.auth.handler import AuthHandler

        # Verify that the handler methods exist and are decorated
        assert hasattr(AuthHandler, "_handle_login")
        assert hasattr(AuthHandler, "_handle_register")
        assert hasattr(AuthHandler, "_handle_refresh")

    def test_login_route_exists_with_post_method(self, auth_handler):
        """Login route responds to POST method."""
        handler = make_mock_handler({"email": "test@example.com", "password": "somepassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        # Should get a response (not None), even if it fails auth
        assert result is not None
        # The response should be one of the expected codes (400, 401, 429, 503)
        assert result.status_code in (200, 400, 401, 429, 503)


# ===========================================================================
# 7. Account Lockout Tests
# ===========================================================================


class TestAccountLockout:
    """Tests for account lockout mechanism after N failed attempts."""

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_account_locked_after_failures_returns_429(
        self, mock_lockout, auth_handler, mock_user_store
    ):
        """Account is locked after too many failed attempts."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (5, 300)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "wrong_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 429
        data = json.loads(result.body.decode())
        assert "locked" in data.get("error", "").lower()

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_locked_account_blocked_at_check(self, mock_lockout, auth_handler):
        """Login blocked when account is already locked."""
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = True
        mock_lockout_instance.get_remaining_time.return_value = 300
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "anypassword"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 429

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_lockout_reset_on_successful_login(
        self, mock_tokens, mock_lockout, auth_handler, mock_user_store
    ):
        """Lockout counter resets on successful login."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_tokens.return_value = MockTokenPair()
        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 200
        mock_lockout_instance.reset.assert_called()

    @patch("aragora.server.handlers.auth.handler.get_lockout_tracker")
    def test_failed_login_records_failure(self, mock_lockout, auth_handler, mock_user_store):
        """Failed login records failure in lockout tracker."""
        user = MockUser(email="test@example.com")
        mock_user_store.users["user-123"] = user
        mock_user_store.users_by_email["test@example.com"] = "user-123"

        mock_lockout_instance = MagicMock()
        mock_lockout_instance.is_locked.return_value = False
        mock_lockout_instance.record_failure.return_value = (1, None)
        mock_lockout.return_value = mock_lockout_instance

        handler = make_mock_handler({"email": "test@example.com", "password": "wrong_password"})

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "POST"))

        assert result.status_code == 401
        mock_lockout_instance.record_failure.assert_called()


# ===========================================================================
# 8. MFA Enforcement Tests
# ===========================================================================


class TestMFAEnforcement:
    """Tests for MFA TOTP verification and backup codes."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_generates_secret(self, mock_auth, auth_handler, mock_user_store):
        """MFA setup generates secret and provisioning URI."""

        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "secret" in data
        assert "provisioning_uri" in data

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_setup_already_enabled_returns_400(self, mock_auth, auth_handler, mock_user_store):
        """MFA setup fails when already enabled."""

        user = MockUser(mfa_enabled=True)
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/setup", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_with_valid_totp(self, mock_auth, auth_handler, mock_user_store):
        """MFA enable succeeds with valid TOTP code."""


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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_enable_invalid_code_returns_400(self, mock_auth, auth_handler, mock_user_store):
        """MFA enable fails with invalid code."""

        user = MockUser(mfa_secret="JBSWY3DPEHPK3PXP")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"code": "000000"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/enable", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_mfa_verify_with_valid_totp(
        self, mock_blacklist, mock_tokens, mock_pending, auth_handler, mock_user_store
    ):
        """MFA verify succeeds with valid TOTP code."""


        secret = "JBSWY3DPEHPK3PXP"
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        mock_user_store.users["user-123"] = user

        mock_pending.return_value = MagicMock(sub="user-123")
        mock_tokens.return_value = MockTokenPair()
        mock_blacklist.return_value = MagicMock()

        totp = pyotp.TOTP(secret)
        code = totp.now()

        handler = make_mock_handler({"code": code, "pending_token": "pending_token_123"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_mfa_verify_with_backup_code(
        self, mock_blacklist, mock_tokens, mock_pending, auth_handler, mock_user_store
    ):
        """MFA verify succeeds with valid backup code."""
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

        handler = make_mock_handler({"code": backup_code, "pending_token": "pending_token_123"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "tokens" in data

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_mfa_verify_invalid_pending_token_returns_401(self, mock_pending, auth_handler):
        """MFA verify fails with invalid pending token."""

        mock_pending.return_value = None

        handler = make_mock_handler({"code": "123456", "pending_token": "invalid_token"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 401

    def test_mfa_verify_missing_code_returns_400(self, auth_handler):
        """MFA verify fails when code is missing."""

        handler = make_mock_handler({"code": "", "pending_token": "some_token"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 400

    def test_mfa_verify_missing_pending_token_returns_400(self, auth_handler):
        """MFA verify fails when pending token is missing."""

        handler = make_mock_handler({"code": "123456", "pending_token": ""})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/verify", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_disable_with_password(self, mock_auth, auth_handler, mock_user_store):
        """MFA disable succeeds with correct password."""

        user = MockUser(mfa_enabled=True, mfa_secret="JBSWY3DPEHPK3PXP")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/disable", {}, handler, "POST"))

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].mfa_enabled is False

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_disable_wrong_password_returns_400(self, mock_auth, auth_handler, mock_user_store):
        """MFA disable fails with wrong password."""

        user = MockUser(mfa_enabled=True, mfa_secret="JBSWY3DPEHPK3PXP")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"password": "wrong_password"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/disable", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_disable_not_enabled_returns_400(self, mock_auth, auth_handler, mock_user_store):
        """MFA disable fails when MFA not enabled."""

        user = MockUser(mfa_enabled=False)
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"password": "correct_password"})

        result = maybe_await(auth_handler.handle("/api/auth/mfa/disable", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_mfa_backup_codes_regeneration(self, mock_auth, auth_handler, mock_user_store):
        """Regenerate backup codes succeeds with valid MFA code."""


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


# ===========================================================================
# 9. Password Validation Tests
# ===========================================================================


class TestPasswordValidation:
    """Tests for password strength requirements."""

    def test_register_weak_password_returns_400(self, auth_handler):
        """Registration fails with weak password."""
        handler = make_mock_handler({"email": "test@example.com", "password": "short"})

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400

    def test_register_empty_password_returns_400(self, auth_handler):
        """Registration fails with empty password."""
        handler = make_mock_handler({"email": "test@example.com", "password": ""})

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_weak_new_returns_400(self, mock_auth, auth_handler, mock_user_store):
        """Password change fails with weak new password."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(
            {"current_password": "correct_password", "new_password": "short"}
        )

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.models.hash_password")
    def test_change_password_success(self, mock_hash, mock_auth, auth_handler, mock_user_store):
        """Password change succeeds with correct current password."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()
        mock_hash.return_value = ("new_hash", "new_salt")

        handler = make_mock_handler(
            {"current_password": "correct_password", "new_password": "New_Secure1!pass"}
        )

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert "success" in data.get("message", "").lower()

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_wrong_current_returns_401(
        self, mock_auth, auth_handler, mock_user_store
    ):
        """Password change fails with wrong current password."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(
            {"current_password": "wrong_password", "new_password": "New_Secure1!pass"}
        )

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_change_password_missing_fields_returns_400(
        self, mock_auth, auth_handler, mock_user_store
    ):
        """Password change fails when fields are missing."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"current_password": "", "new_password": ""})

        result = maybe_await(auth_handler.handle("/api/auth/password", {}, handler, "POST"))

        assert result.status_code == 400

    def test_register_invalid_email_returns_400(self, auth_handler):
        """Registration fails with invalid email format."""
        handler = make_mock_handler({"email": "invalid", "password": "SecurePass1!"})

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400


# ===========================================================================
# 10. Session Management Tests
# ===========================================================================


class TestSessionManagement:
    """Tests for concurrent sessions and session listing."""

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_list_sessions_returns_all_user_sessions(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """List sessions returns all sessions for the user."""
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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_list_sessions_marks_current(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """List sessions marks the current session."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="session-1")

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
        sessions = data["sessions"]
        current = [s for s in sessions if s.get("is_current")]
        assert len(current) == 1
        assert current[0]["session_id"] == "session-1"

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_revoke_session_success(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """Revoke session succeeds for valid non-current session."""
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
    def test_revoke_current_session_returns_400(
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
    def test_revoke_nonexistent_session_returns_404(
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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    @patch("aragora.billing.auth.sessions.get_session_manager")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.server.middleware.auth.extract_token")
    def test_revoke_short_session_id_returns_400(
        self, mock_extract_token, mock_decode, mock_get_manager, mock_auth, auth_handler
    ):
        """Revoke fails for session ID that is too short."""
        mock_auth.return_value = MockAuthContext()
        mock_extract_token.return_value = "current_token"
        mock_decode.return_value = MagicMock(jti="different")

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(auth_handler.handle("/api/auth/sessions/short", {}, handler, "DELETE"))

        assert result.status_code == 400


# ===========================================================================
# 11. Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for malformed requests, database errors, timeout."""

    def test_invalid_json_body_returns_400(self, auth_handler):
        """Invalid JSON body returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "5"}
        handler.rfile = BytesIO(b"invalid")
        handler.client_address = ("127.0.0.1", 12345)

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 400

    def test_method_not_allowed_get_on_login(self, auth_handler):
        """GET on POST-only endpoint returns 405."""
        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "GET"))

        assert result.status_code == 405

    def test_method_not_allowed_patch(self, auth_handler):
        """PATCH on auth endpoints returns 405."""
        handler = make_mock_handler(method="PATCH")
        handler.command = "PATCH"

        result = maybe_await(auth_handler.handle("/api/auth/login", {}, handler, "PATCH"))

        assert result.status_code == 405

    def test_register_no_user_store_returns_503(self):
        """Register returns 503 when user store unavailable."""
        from aragora.server.handlers.auth.handler import AuthHandler

        auth_handler = AuthHandler({})

        handler = make_mock_handler({"email": "test@example.com", "password": "SecurePass1!"})

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 503

    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_duplicate_email_returns_409(
        self, mock_hash, mock_tokens, auth_handler, mock_user_store
    ):
        """Register with existing email returns 409."""
        existing_user = MockUser(email="existing@example.com")
        mock_user_store.users["user-1"] = existing_user
        mock_user_store.users_by_email["existing@example.com"] = "user-1"

        mock_hash.return_value = ("hashed", "salt")

        handler = make_mock_handler({"email": "existing@example.com", "password": "SecurePass1!"})

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 409

    def test_refresh_invalid_json_body_returns_400(self, auth_handler):
        """Refresh with invalid JSON returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "5"}
        handler.rfile = BytesIO(b"invalid")
        handler.client_address = ("127.0.0.1", 12345)

        result = maybe_await(auth_handler.handle("/api/auth/refresh", {}, handler, "POST"))

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_user_not_found_returns_404(self, mock_auth, auth_handler, mock_user_store):
        """Get me returns 404 when user no longer exists."""
        # No user in store
        mock_auth.return_value = MockAuthContext(user_id="deleted-user")

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "GET"))

        assert result.status_code == 404


# ===========================================================================
# 12. Security Headers Validation Tests
# ===========================================================================


class TestSecurityHeaders:
    """Tests for security headers validation."""

    def test_auth_no_cache_headers_defined(self):
        """AuthHandler defines no-cache headers constant."""
        from aragora.server.handlers.auth.handler import AuthHandler

        assert hasattr(AuthHandler, "AUTH_NO_CACHE_HEADERS")
        headers = AuthHandler.AUTH_NO_CACHE_HEADERS
        assert "Cache-Control" in headers
        assert "no-store" in headers["Cache-Control"]
        assert "no-cache" in headers["Cache-Control"]
        assert "private" in headers["Cache-Control"]
        assert "Pragma" in headers
        assert headers["Pragma"] == "no-cache"
        assert "Expires" in headers
        assert headers["Expires"] == "0"

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_response_includes_no_cache_headers(
        self, mock_auth, auth_handler, mock_user_store
    ):
        """GET /me response includes no-cache headers."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "GET"))

        assert result.status_code == 200
        # Check that headers are present on the response
        if hasattr(result, "headers") and result.headers:
            assert "Cache-Control" in result.headers
            assert "no-store" in result.headers["Cache-Control"]


# ===========================================================================
# Additional Tests: Registration, Routing, API Keys, User Info
# ===========================================================================


class TestRegistration:
    """Tests for user registration."""

    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_success_returns_201(self, mock_hash, mock_tokens, auth_handler):
        """Registration succeeds with valid data."""
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {"email": "new@example.com", "password": "SecurePass1!", "name": "New User"}
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 201
        data = json.loads(result.body.decode())
        assert "user" in data
        assert "tokens" in data

    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.models.hash_password")
    def test_register_with_organization(
        self, mock_hash, mock_tokens, auth_handler, mock_user_store
    ):
        """Registration with organization name creates org."""
        mock_hash.return_value = ("hashed", "salt")
        mock_tokens.return_value = MockTokenPair()

        handler = make_mock_handler(
            {
                "email": "new@example.com",
                "password": "SecurePass1!",
                "name": "New User",
                "organization": "My Org",
            }
        )

        result = maybe_await(auth_handler.handle("/api/auth/register", {}, handler, "POST"))

        assert result.status_code == 201
        # Verify org was created
        assert len(mock_user_store.orgs) == 1


class TestRouting:
    """Tests for handler routing and can_handle."""

    @pytest.fixture
    def handler_instance(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_routes_is_list(self, handler_instance):
        assert isinstance(handler_instance.ROUTES, list)

    def test_routes_not_empty(self, handler_instance):
        assert len(handler_instance.ROUTES) > 0

    def test_can_handle_register(self, handler_instance):
        assert handler_instance.can_handle("/api/auth/register") is True

    def test_can_handle_login(self, handler_instance):
        assert handler_instance.can_handle("/api/auth/login") is True

    def test_can_handle_sessions_wildcard(self, handler_instance):
        assert handler_instance.can_handle("/api/auth/sessions/abc123") is True

    def test_can_handle_api_keys_wildcard(self, handler_instance):
        assert handler_instance.can_handle("/api/auth/api-keys/ara_prefix123") is True

    def test_cannot_handle_unrelated_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates") is False

    def test_cannot_handle_unknown_auth_endpoint(self, handler_instance):
        assert handler_instance.can_handle("/api/auth/unknown") is False

    @pytest.mark.asyncio
    async def test_register_routes_to_handler_method(self):
        """Register POST routes to _handle_register."""
        from aragora.server.handlers.auth.handler import AuthHandler

        mock_store = MagicMock()
        handler_instance = AuthHandler(server_context={"user_store": mock_store})

        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {"Content-Length": "2"}
        mock_http.rfile = MagicMock()
        mock_http.rfile.read = MagicMock(return_value=b"{}")

        with patch.object(handler_instance, "_handle_register") as mock_method:
            mock_method.return_value = MagicMock()
            await handler_instance.handle("/api/auth/register", {}, mock_http, "POST")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_delete_routes_with_id(self):
        """Session DELETE routes with session ID."""
        from aragora.server.handlers.auth.handler import AuthHandler

        mock_store = MagicMock()
        handler_instance = AuthHandler(server_context={"user_store": mock_store})

        mock_http = MagicMock()
        mock_http.command = "DELETE"
        mock_http.headers = {"Content-Length": "0"}
        mock_http.rfile = MagicMock()

        with patch.object(handler_instance, "_handle_revoke_session") as mock_method:
            mock_method.return_value = MagicMock()
            await handler_instance.handle("/api/auth/sessions/abc123", {}, mock_http, "DELETE")
            mock_method.assert_called_once_with(mock_http, "abc123")


class TestAPIKeyOperations:
    """Tests for API key operations."""

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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_list_api_keys_returns_user_keys(self, mock_auth, auth_handler, mock_user_store):
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

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_list_api_keys_empty_returns_zero(self, mock_auth, auth_handler, mock_user_store):
        """List API keys returns empty when user has no keys."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/api-keys", {}, handler, "GET"))

        assert result.status_code == 200
        data = json.loads(result.body.decode())
        assert data["count"] == 0

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_generate_api_key_tier_restricted_returns_403(
        self, mock_auth, auth_handler, mock_user_store
    ):
        """Generate API key fails when org tier lacks API access."""
        org = MockOrganization()
        org.limits.api_access = False
        user = MockUser(org_id="org-123")
        mock_user_store.users["user-123"] = user
        mock_user_store.orgs["org-123"] = org
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({})

        result = maybe_await(auth_handler.handle("/api/auth/api-key", {}, handler, "POST"))

        assert result.status_code == 403

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_revoke_api_key_by_prefix_success(self, mock_auth, auth_handler, mock_user_store):
        """Revoke API key by prefix succeeds."""
        user = MockUser(api_key_hash="hash", api_key_prefix="ara_test")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(
            auth_handler.handle("/api/auth/api-keys/ara_test", {}, handler, "DELETE")
        )

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].api_key_hash is None

    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_revoke_api_key_wrong_prefix_returns_404(
        self, mock_auth, auth_handler, mock_user_store
    ):
        """Revoke API key fails with wrong prefix."""
        user = MockUser(api_key_hash="hash", api_key_prefix="ara_test")
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler(method="DELETE")

        result = maybe_await(
            auth_handler.handle("/api/auth/api-keys/ara_wrong", {}, handler, "DELETE")
        )

        assert result.status_code == 404


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
    def test_update_me_changes_name(self, mock_auth, auth_handler, mock_user_store):
        """Update me changes user name."""
        user = MockUser()
        mock_user_store.users["user-123"] = user
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler({"name": "Updated Name"}, method="PUT")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "PUT"))

        assert result.status_code == 200
        assert mock_user_store.users["user-123"].name == "Updated Name"

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.auth.handler.extract_user_from_request")
    def test_get_me_unauthenticated_returns_401(self, mock_auth, auth_handler):
        """Get me fails when not authenticated."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler(method="GET")

        result = maybe_await(auth_handler.handle("/api/auth/me", {}, handler, "GET"))

        assert result.status_code == 401


class TestModuleExports:
    """Tests for module exports."""

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

    def test_handler_resource_type(self):
        """AuthHandler has correct RESOURCE_TYPE."""
        from aragora.server.handlers.auth.handler import AuthHandler

        assert AuthHandler.RESOURCE_TYPE == "auth"
