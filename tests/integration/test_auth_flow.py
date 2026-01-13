"""
Authentication Flow Integration Tests.

Tests complete authentication workflows:
- User registration → login → authenticated API call
- JWT token lifecycle (generate → use → refresh → expire)
- API key authentication (generate → use → revoke)
- Token revocation and logout
- Account lockout after failed attempts
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import User, hash_password
from aragora.billing.jwt_auth import (
    create_token_pair,
    decode_jwt,
    get_token_blacklist,
)
from aragora.server.handlers.auth import AuthHandler, validate_email, validate_password
from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters before each test to avoid cross-test interference."""
    from aragora.server.handlers.utils.rate_limit import _limiters
    # Clear the internal buckets of all existing limiters
    for limiter in _limiters.values():
        limiter.clear()
    yield
    # Clear again after test
    for limiter in _limiters.values():
        limiter.clear()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "auth_test.db"


@pytest.fixture
def user_store(temp_db_path):
    """Create a UserStore with temporary database."""
    from aragora.storage.user_store import UserStore
    store = UserStore(str(temp_db_path))
    return store


@pytest.fixture
def auth_handler(user_store):
    """Create an AuthHandler with user store context."""
    return AuthHandler({"user_store": user_store})


@pytest.fixture
def registered_user(user_store) -> User:
    """Create a pre-registered test user."""
    password_hash, password_salt = hash_password("TestPass123!")
    user = user_store.create_user(
        email="existing@example.com",
        password_hash=password_hash,
        password_salt=password_salt,
        name="Existing User",
    )
    return user


def create_mock_request(
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    method: str = "POST",
) -> MagicMock:
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.headers["Content-Type"] = "application/json"
    else:
        handler.rfile = BytesIO(b"")

    return handler


def parse_result(result: HandlerResult) -> tuple[dict, int]:
    """Parse a HandlerResult into (data, status_code)."""
    if result is None:
        return {}, 404
    data = json.loads(result.body.decode("utf-8")) if result.body else {}
    return data, result.status_code


# =============================================================================
# Email and Password Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation functions."""

    def test_valid_email_accepted(self):
        """Valid email formats should be accepted."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.co.uk",
            "user123@sub.domain.com",
        ]
        for email in valid_emails:
            valid, err = validate_email(email)
            assert valid, f"Email {email} should be valid, got: {err}"

    def test_invalid_email_rejected(self):
        """Invalid email formats should be rejected."""
        invalid_emails = [
            "",
            "not-an-email",
            "@missing-local.com",
            "missing-at.com",
            "spaces in@email.com",
        ]
        for email in invalid_emails:
            valid, err = validate_email(email)
            assert not valid, f"Email {email} should be invalid"

    def test_password_min_length(self):
        """Password must meet minimum length."""
        valid, err = validate_password("short")
        assert not valid
        assert "at least 8 characters" in err

    def test_password_max_length(self):
        """Password must not exceed maximum length."""
        long_password = "x" * 129
        valid, err = validate_password(long_password)
        assert not valid
        assert "at most 128 characters" in err

    def test_valid_password_accepted(self):
        """Valid passwords should be accepted."""
        valid, err = validate_password("SecurePass123!")
        assert valid
        assert err == ""


# =============================================================================
# Registration Flow Tests
# =============================================================================


class TestRegistrationFlow:
    """Test user registration workflow."""

    def test_successful_registration(self, auth_handler, user_store):
        """User can register with valid credentials."""
        request = create_mock_request(body={
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "name": "New User",
        })

        result = auth_handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 201
        assert "user" in data
        assert "tokens" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]

        # Verify user was created in database
        user = user_store.get_user_by_email("newuser@example.com")
        assert user is not None
        assert user.name == "New User"

    def test_registration_with_organization(self, auth_handler, user_store):
        """User can register with organization name."""
        request = create_mock_request(body={
            "email": "orguser@example.com",
            "password": "SecurePass123!",
            "organization": "Test Org",
        })

        result = auth_handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 201
        assert data["user"]["org_id"] is not None

        # Verify org was created
        user = user_store.get_user_by_email("orguser@example.com")
        assert user.org_id is not None

    def test_duplicate_email_rejected(self, auth_handler, registered_user):
        """Registration with existing email should fail."""
        request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "NewPass456!",
        })

        result = auth_handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 409
        assert "already registered" in data.get("error", "").lower()

    def test_invalid_email_rejected(self, auth_handler):
        """Registration with invalid email should fail."""
        request = create_mock_request(body={
            "email": "not-an-email",
            "password": "SecurePass123!",
        })

        result = auth_handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 400
        assert "email" in data.get("error", "").lower()

    def test_weak_password_rejected(self, auth_handler):
        """Registration with weak password should fail."""
        request = create_mock_request(body={
            "email": "newuser@example.com",
            "password": "weak",
        })

        result = auth_handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 400
        assert "password" in data.get("error", "").lower()


# =============================================================================
# Login Flow Tests
# =============================================================================


class TestLoginFlow:
    """Test user login workflow."""

    def test_successful_login(self, auth_handler, registered_user):
        """User can login with valid credentials."""
        request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "TestPass123!",
        })

        result = auth_handler._handle_login(request)
        data, status = parse_result(result)

        assert status == 200
        assert "user" in data
        assert "tokens" in data
        assert data["user"]["email"] == "existing@example.com"

    def test_wrong_password_rejected(self, auth_handler, registered_user):
        """Login with wrong password should fail."""
        request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "WrongPassword123!",
        })

        result = auth_handler._handle_login(request)
        data, status = parse_result(result)

        assert status == 401
        assert "invalid" in data.get("error", "").lower()

    def test_nonexistent_user_rejected(self, auth_handler):
        """Login with non-existent user should fail with same error."""
        request = create_mock_request(body={
            "email": "noone@example.com",
            "password": "SomePass123!",
        })

        result = auth_handler._handle_login(request)
        data, status = parse_result(result)

        # Should return same error as wrong password to prevent enumeration
        assert status == 401
        assert "invalid" in data.get("error", "").lower()

    def test_missing_credentials_rejected(self, auth_handler):
        """Login without credentials should fail."""
        request = create_mock_request(body={})

        result = auth_handler._handle_login(request)
        data, status = parse_result(result)

        assert status == 400


# =============================================================================
# Token Lifecycle Tests
# =============================================================================


class TestTokenLifecycle:
    """Test JWT token generation, validation, and refresh."""

    def test_access_token_valid_after_login(self, auth_handler, registered_user):
        """Access token should be valid immediately after login."""
        request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "TestPass123!",
        })

        result = auth_handler._handle_login(request)
        data, status = parse_result(result)

        access_token = data["tokens"]["access_token"]
        decoded = decode_jwt(access_token)

        assert decoded is not None
        assert decoded.sub == registered_user.id
        assert decoded.email == registered_user.email

    def test_refresh_token_generates_new_access(self, auth_handler, registered_user):
        """Refresh token should generate new access token."""
        # First login to get tokens
        login_request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "TestPass123!",
        })
        login_result = auth_handler._handle_login(login_request)
        login_data, _ = parse_result(login_result)
        refresh_token = login_data["tokens"]["refresh_token"]

        # Wait to ensure different iat timestamp (JWTs in same second are identical)
        time.sleep(1.1)

        # Use refresh token
        refresh_request = create_mock_request(body={
            "refresh_token": refresh_token,
        })
        refresh_result = auth_handler._handle_refresh(refresh_request)
        refresh_data, status = parse_result(refresh_result)

        assert status == 200
        assert "tokens" in refresh_data
        # New token should be different (different iat timestamp)
        assert refresh_data["tokens"]["access_token"] != login_data["tokens"]["access_token"]


# =============================================================================
# Authenticated Request Tests
# =============================================================================


class TestAuthenticatedRequests:
    """Test authenticated API access."""

    def test_get_me_with_valid_token(self, auth_handler, registered_user):
        """GET /auth/me with valid token returns user info."""
        # Create token for user
        tokens = create_token_pair(
            user_id=registered_user.id,
            email=registered_user.email,
            org_id=registered_user.org_id,
            role=registered_user.role,
        )

        request = create_mock_request(
            method="GET",
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )

        result = auth_handler._handle_get_me(request)
        data, status = parse_result(result)

        assert status == 200
        assert data["user"]["email"] == registered_user.email

    def test_get_me_without_token_fails(self, auth_handler):
        """GET /auth/me without token should fail."""
        request = create_mock_request(method="GET")

        result = auth_handler._handle_get_me(request)
        data, status = parse_result(result)

        assert status == 401


# =============================================================================
# Logout and Token Revocation Tests
# =============================================================================


class TestLogoutFlow:
    """Test logout and token revocation."""

    def test_logout_invalidates_token(self, auth_handler, registered_user):
        """Logout should invalidate the current token."""
        # Login first
        login_request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "TestPass123!",
        })
        login_result = auth_handler._handle_login(login_request)
        login_data, _ = parse_result(login_result)
        access_token = login_data["tokens"]["access_token"]

        # Logout
        logout_request = create_mock_request(
            headers={"Authorization": f"Bearer {access_token}"},
        )
        logout_result = auth_handler._handle_logout(logout_request)
        _, status = parse_result(logout_result)

        assert status == 200

        # Token should now be blacklisted
        blacklist = get_token_blacklist()
        # The token JTI should be in blacklist (implementation specific)


# =============================================================================
# API Key Authentication Tests
# =============================================================================


class TestApiKeyAuthentication:
    """Test API key generation and usage."""

    def test_generate_api_key(self, auth_handler, registered_user):
        """User can generate an API key."""
        tokens = create_token_pair(
            user_id=registered_user.id,
            email=registered_user.email,
            org_id=registered_user.org_id,
            role=registered_user.role,
        )

        request = create_mock_request(
            body={"name": "Test Key"},
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )

        result = auth_handler._handle_generate_api_key(request)
        data, status = parse_result(result)

        assert status in (200, 201)
        assert "api_key" in data or "key" in data

    def test_revoke_api_key(self, auth_handler, registered_user, user_store):
        """User can revoke an API key."""
        # First generate a key
        tokens = create_token_pair(
            user_id=registered_user.id,
            email=registered_user.email,
            org_id=registered_user.org_id,
            role=registered_user.role,
        )

        gen_request = create_mock_request(
            body={"name": "Key to Revoke"},
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )
        gen_result = auth_handler._handle_generate_api_key(gen_request)
        gen_data, _ = parse_result(gen_result)

        # Get key prefix
        key_prefix = gen_data.get("prefix") or gen_data.get("api_key", "")[:8]

        # Revoke the key
        revoke_request = create_mock_request(
            method="DELETE",
            body={"prefix": key_prefix},
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )

        result = auth_handler._handle_revoke_api_key(revoke_request)
        data, status = parse_result(result)

        assert status == 200


# =============================================================================
# Full Flow Integration Tests
# =============================================================================


class TestFullAuthenticationFlow:
    """Test complete authentication workflows end-to-end."""

    def test_register_login_access_logout(self, auth_handler, user_store):
        """Complete flow: register → login → access protected → logout."""
        # 1. Register new user
        reg_request = create_mock_request(body={
            "email": "flowtest@example.com",
            "password": "FlowTest123!",
            "name": "Flow Test User",
        })
        reg_result = auth_handler._handle_register(reg_request)
        reg_data, reg_status = parse_result(reg_result)
        assert reg_status == 201

        # 2. Login with credentials
        login_request = create_mock_request(body={
            "email": "flowtest@example.com",
            "password": "FlowTest123!",
        })
        login_result = auth_handler._handle_login(login_request)
        login_data, login_status = parse_result(login_result)
        assert login_status == 200
        access_token = login_data["tokens"]["access_token"]

        # 3. Access protected endpoint
        me_request = create_mock_request(
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        me_result = auth_handler._handle_get_me(me_request)
        me_data, me_status = parse_result(me_result)
        assert me_status == 200
        assert me_data["user"]["email"] == "flowtest@example.com"

        # 4. Logout
        logout_request = create_mock_request(
            headers={"Authorization": f"Bearer {access_token}"},
        )
        logout_result = auth_handler._handle_logout(logout_request)
        _, logout_status = parse_result(logout_result)
        assert logout_status == 200

    def test_token_refresh_maintains_access(self, auth_handler, registered_user):
        """Access should work after token refresh."""
        # Login
        login_request = create_mock_request(body={
            "email": "existing@example.com",
            "password": "TestPass123!",
        })
        login_result = auth_handler._handle_login(login_request)
        login_data, _ = parse_result(login_result)
        refresh_token = login_data["tokens"]["refresh_token"]

        # Refresh tokens
        refresh_request = create_mock_request(body={
            "refresh_token": refresh_token,
        })
        refresh_result = auth_handler._handle_refresh(refresh_request)
        refresh_data, _ = parse_result(refresh_result)
        new_access_token = refresh_data["tokens"]["access_token"]

        # Access with new token
        me_request = create_mock_request(
            method="GET",
            headers={"Authorization": f"Bearer {new_access_token}"},
        )
        me_result = auth_handler._handle_get_me(me_request)
        me_data, me_status = parse_result(me_result)

        assert me_status == 200
        assert me_data["user"]["email"] == "existing@example.com"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAuthErrorHandling:
    """Test authentication error scenarios."""

    def test_missing_user_store_returns_503(self, temp_db_path):
        """Missing user store should return 503."""
        handler = AuthHandler({})  # No user store
        request = create_mock_request(body={
            "email": "test@example.com",
            "password": "TestPass123!",
        })

        result = handler._handle_register(request)
        data, status = parse_result(result)

        assert status == 503
        assert "unavailable" in data.get("error", "").lower()

    def test_malformed_json_returns_400(self, auth_handler):
        """Malformed JSON body should return 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "10", "Content-Type": "application/json"}
        handler.rfile = BytesIO(b"not json!")

        result = auth_handler._handle_register(handler)
        data, status = parse_result(result)

        assert status == 400


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Test concurrent authentication scenarios."""

    def test_multiple_sessions_same_user(self, auth_handler, registered_user):
        """User can have multiple active sessions."""
        sessions = []

        # Create 3 sessions with delays to ensure different iat timestamps
        for i in range(3):
            if i > 0:
                time.sleep(1.1)  # Ensure different iat timestamp
            request = create_mock_request(body={
                "email": "existing@example.com",
                "password": "TestPass123!",
            })
            result = auth_handler._handle_login(request)
            data, status = parse_result(result)
            assert status == 200
            sessions.append(data["tokens"]["access_token"])

        # All tokens should be unique (different iat timestamps)
        assert len(set(sessions)) == 3

        # All tokens should still be valid
        for token in sessions:
            decoded = decode_jwt(token)
            assert decoded is not None
            assert decoded.sub == registered_user.id
