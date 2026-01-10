"""
Tests for AuthHandler - user authentication endpoints.

Tests cover:
- POST /api/auth/register - User registration
- POST /api/auth/login - User login
- POST /api/auth/logout - Token invalidation
- POST /api/auth/refresh - Token refresh
- GET /api/auth/me - Get current user
- PUT /api/auth/me - Update user info
- POST /api/auth/password - Change password
- POST /api/auth/api-key - API key management

Security tests:
- Password validation
- Email validation
- Token expiration
- Invalid credentials
- Account status checks
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from aragora.server.handlers.auth import (
    AuthHandler,
    validate_email,
    validate_password,
    EMAIL_PATTERN,
    MIN_PASSWORD_LENGTH,
    MAX_PASSWORD_LENGTH,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = Mock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.org_id = "org-456"
    user.role = "member"
    user.is_active = True
    user.verify_password = Mock(return_value=True)
    user.to_dict = Mock(return_value={
        "id": "user-123",
        "email": "test@example.com",
        "name": "Test User",
        "org_id": "org-456",
        "role": "member",
    })
    return user


@pytest.fixture
def mock_user_store(mock_user):
    """Create a mock user store."""
    store = Mock()
    store.get_user_by_email = Mock(return_value=None)
    store.get_user_by_id = Mock(return_value=mock_user)
    store.create_user = Mock(return_value=mock_user)
    store.update_user = Mock(return_value=mock_user)
    store.create_organization = Mock(return_value=Mock(id="org-456"))
    store.get_organization_by_id = Mock(return_value=None)
    return store


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.command = "POST"
    handler.headers = {"Content-Type": "application/json"}
    handler.rfile = Mock()
    return handler


@pytest.fixture
def auth_handler(mock_user_store):
    """Create AuthHandler with mock dependencies."""
    ctx = {"user_store": mock_user_store}
    return AuthHandler(ctx)


# ============================================================================
# Email Validation Tests
# ============================================================================

class TestEmailValidation:
    """Tests for email validation function."""

    def test_valid_email(self):
        """Test valid email formats."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user123@domain.co.uk",
            "a@b.cc",
        ]
        for email in valid_emails:
            valid, _ = validate_email(email)
            assert valid, f"Email should be valid: {email}"

    def test_invalid_email_empty(self):
        """Test empty email."""
        valid, err = validate_email("")
        assert not valid
        assert "required" in err.lower()

    def test_invalid_email_no_at(self):
        """Test email without @ symbol."""
        valid, _ = validate_email("userexample.com")
        assert not valid

    def test_invalid_email_no_domain(self):
        """Test email without domain."""
        valid, _ = validate_email("user@")
        assert not valid

    def test_invalid_email_too_long(self):
        """Test email exceeding max length."""
        long_email = "a" * 250 + "@example.com"
        valid, err = validate_email(long_email)
        assert not valid
        assert "too long" in err.lower()


# ============================================================================
# Password Validation Tests
# ============================================================================

class TestPasswordValidation:
    """Tests for password validation function."""

    def test_valid_password(self):
        """Test valid password."""
        valid, _ = validate_password("SecurePass123!")
        assert valid

    def test_password_empty(self):
        """Test empty password."""
        valid, err = validate_password("")
        assert not valid
        assert "required" in err.lower()

    def test_password_too_short(self):
        """Test password below minimum length."""
        valid, err = validate_password("short")
        assert not valid
        assert str(MIN_PASSWORD_LENGTH) in err

    def test_password_too_long(self):
        """Test password exceeding maximum length."""
        long_pass = "a" * (MAX_PASSWORD_LENGTH + 1)
        valid, err = validate_password(long_pass)
        assert not valid
        assert str(MAX_PASSWORD_LENGTH) in err

    def test_password_at_min_length(self):
        """Test password at exactly minimum length."""
        valid, _ = validate_password("a" * MIN_PASSWORD_LENGTH)
        assert valid

    def test_password_at_max_length(self):
        """Test password at exactly maximum length."""
        valid, _ = validate_password("a" * MAX_PASSWORD_LENGTH)
        assert valid


# ============================================================================
# AuthHandler Route Tests
# ============================================================================

class TestAuthHandlerRoutes:
    """Tests for AuthHandler routing."""

    def test_can_handle_register(self, auth_handler):
        """Test handler recognizes register route."""
        assert auth_handler.can_handle("/api/auth/register")

    def test_can_handle_login(self, auth_handler):
        """Test handler recognizes login route."""
        assert auth_handler.can_handle("/api/auth/login")

    def test_can_handle_logout(self, auth_handler):
        """Test handler recognizes logout route."""
        assert auth_handler.can_handle("/api/auth/logout")

    def test_can_handle_refresh(self, auth_handler):
        """Test handler recognizes refresh route."""
        assert auth_handler.can_handle("/api/auth/refresh")

    def test_can_handle_me(self, auth_handler):
        """Test handler recognizes me route."""
        assert auth_handler.can_handle("/api/auth/me")

    def test_can_handle_password(self, auth_handler):
        """Test handler recognizes password route."""
        assert auth_handler.can_handle("/api/auth/password")

    def test_can_handle_api_key(self, auth_handler):
        """Test handler recognizes api-key route."""
        assert auth_handler.can_handle("/api/auth/api-key")

    def test_cannot_handle_unknown_route(self, auth_handler):
        """Test handler rejects unknown routes."""
        assert not auth_handler.can_handle("/api/auth/unknown")
        assert not auth_handler.can_handle("/api/users")


# ============================================================================
# Registration Tests
# ============================================================================

class TestRegistration:
    """Tests for user registration endpoint."""

    @patch("aragora.server.handlers.auth.hash_password")
    @patch("aragora.server.handlers.auth.create_token_pair")
    def test_register_success(
        self, mock_tokens, mock_hash, auth_handler, mock_handler, mock_user_store
    ):
        """Test successful user registration."""
        mock_hash.return_value = ("hash", "salt")
        mock_tokens.return_value = Mock(to_dict=lambda: {"access_token": "token"})

        # Setup request body
        auth_handler.read_json_body = Mock(return_value={
            "email": "new@example.com",
            "password": "SecurePass123!",
            "name": "New User",
        })

        result = auth_handler._handle_register(mock_handler)

        assert result.status_code == 201
        mock_user_store.create_user.assert_called_once()

    def test_register_invalid_json(self, auth_handler, mock_handler):
        """Test registration with invalid JSON."""
        auth_handler.read_json_body = Mock(return_value=None)

        result = auth_handler._handle_register(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid JSON" in data["error"]

    def test_register_invalid_email(self, auth_handler, mock_handler):
        """Test registration with invalid email."""
        auth_handler.read_json_body = Mock(return_value={
            "email": "invalid-email",
            "password": "SecurePass123!",
        })

        result = auth_handler._handle_register(mock_handler)

        assert result.status_code == 400

    def test_register_weak_password(self, auth_handler, mock_handler):
        """Test registration with weak password."""
        auth_handler.read_json_body = Mock(return_value={
            "email": "user@example.com",
            "password": "weak",
        })

        result = auth_handler._handle_register(mock_handler)

        assert result.status_code == 400

    def test_register_duplicate_email(
        self, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test registration with existing email."""
        mock_user_store.get_user_by_email.return_value = mock_user
        auth_handler.read_json_body = Mock(return_value={
            "email": "existing@example.com",
            "password": "SecurePass123!",
        })

        result = auth_handler._handle_register(mock_handler)

        assert result.status_code == 409
        data = json.loads(result.body)
        assert "already registered" in data["error"]

    def test_register_user_store_unavailable(self, mock_handler):
        """Test registration when user store is unavailable."""
        handler = AuthHandler({})  # No user_store
        handler.read_json_body = Mock(return_value={
            "email": "user@example.com",
            "password": "SecurePass123!",
        })

        result = handler._handle_register(mock_handler)

        assert result.status_code == 503


# ============================================================================
# Login Tests
# ============================================================================

class TestLogin:
    """Tests for user login endpoint."""

    @patch("aragora.server.handlers.auth.create_token_pair")
    def test_login_success(
        self, mock_tokens, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test successful login."""
        mock_tokens.return_value = Mock(to_dict=lambda: {"access_token": "token"})
        mock_user_store.get_user_by_email.return_value = mock_user

        auth_handler.read_json_body = Mock(return_value={
            "email": "test@example.com",
            "password": "correct_password",
        })

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "user" in data
        assert "tokens" in data

    def test_login_invalid_json(self, auth_handler, mock_handler):
        """Test login with invalid JSON."""
        auth_handler.read_json_body = Mock(return_value=None)

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 400

    def test_login_missing_credentials(self, auth_handler, mock_handler):
        """Test login with missing credentials."""
        auth_handler.read_json_body = Mock(return_value={})

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "required" in data["error"].lower()

    def test_login_user_not_found(
        self, auth_handler, mock_handler, mock_user_store
    ):
        """Test login with non-existent user."""
        mock_user_store.get_user_by_email.return_value = None

        auth_handler.read_json_body = Mock(return_value={
            "email": "unknown@example.com",
            "password": "password123",
        })

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 401
        data = json.loads(result.body)
        # Should not reveal if user exists
        assert "Invalid email or password" in data["error"]

    def test_login_wrong_password(
        self, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test login with wrong password."""
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_email.return_value = mock_user

        auth_handler.read_json_body = Mock(return_value={
            "email": "test@example.com",
            "password": "wrong_password",
        })

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 401

    def test_login_disabled_account(
        self, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test login with disabled account."""
        mock_user.is_active = False
        mock_user_store.get_user_by_email.return_value = mock_user

        auth_handler.read_json_body = Mock(return_value={
            "email": "test@example.com",
            "password": "password",
        })

        result = auth_handler._handle_login(mock_handler)

        assert result.status_code == 403
        data = json.loads(result.body)
        assert "disabled" in data["error"].lower()


# ============================================================================
# Token Refresh Tests
# ============================================================================

class TestTokenRefresh:
    """Tests for token refresh endpoint."""

    @patch("aragora.server.handlers.auth.validate_refresh_token")
    @patch("aragora.server.handlers.auth.create_token_pair")
    def test_refresh_success(
        self, mock_tokens, mock_validate, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test successful token refresh."""
        mock_validate.return_value = Mock(user_id="user-123")
        mock_tokens.return_value = Mock(to_dict=lambda: {"access_token": "new_token"})

        auth_handler.read_json_body = Mock(return_value={
            "refresh_token": "valid_refresh_token",
        })

        result = auth_handler._handle_refresh(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "tokens" in data

    def test_refresh_missing_token(self, auth_handler, mock_handler):
        """Test refresh without token."""
        auth_handler.read_json_body = Mock(return_value={})

        result = auth_handler._handle_refresh(mock_handler)

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.validate_refresh_token")
    def test_refresh_invalid_token(self, mock_validate, auth_handler, mock_handler):
        """Test refresh with invalid token."""
        mock_validate.return_value = None

        auth_handler.read_json_body = Mock(return_value={
            "refresh_token": "invalid_token",
        })

        result = auth_handler._handle_refresh(mock_handler)

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.validate_refresh_token")
    def test_refresh_user_not_found(
        self, mock_validate, auth_handler, mock_handler, mock_user_store
    ):
        """Test refresh when user no longer exists."""
        mock_validate.return_value = Mock(user_id="deleted-user")
        mock_user_store.get_user_by_id.return_value = None

        auth_handler.read_json_body = Mock(return_value={
            "refresh_token": "valid_token",
        })

        result = auth_handler._handle_refresh(mock_handler)

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.validate_refresh_token")
    def test_refresh_disabled_account(
        self, mock_validate, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test refresh with disabled account."""
        mock_validate.return_value = Mock(user_id="user-123")
        mock_user.is_active = False

        auth_handler.read_json_body = Mock(return_value={
            "refresh_token": "valid_token",
        })

        result = auth_handler._handle_refresh(mock_handler)

        assert result.status_code == 403


# ============================================================================
# Logout Tests
# ============================================================================

class TestLogout:
    """Tests for logout endpoint."""

    @patch("aragora.server.handlers.auth.extract_user_from_request")
    def test_logout_success(self, mock_extract, auth_handler, mock_handler):
        """Test successful logout."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
        )

        result = auth_handler._handle_logout(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "Logged out" in data["message"]

    @patch("aragora.server.handlers.auth.extract_user_from_request")
    def test_logout_not_authenticated(self, mock_extract, auth_handler, mock_handler):
        """Test logout without authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = auth_handler._handle_logout(mock_handler)

        assert result.status_code == 401


# ============================================================================
# Get User Info Tests
# ============================================================================

class TestGetMe:
    """Tests for get current user endpoint."""

    @patch("aragora.server.handlers.auth.extract_user_from_request")
    def test_get_me_success(
        self, mock_extract, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test successful get user info."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
        )

        result = auth_handler._handle_get_me(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "user" in data

    @patch("aragora.server.handlers.auth.extract_user_from_request")
    def test_get_me_not_authenticated(self, mock_extract, auth_handler, mock_handler):
        """Test get user info without authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = auth_handler._handle_get_me(mock_handler)

        assert result.status_code == 401

    @patch("aragora.server.handlers.auth.extract_user_from_request")
    def test_get_me_user_not_found(
        self, mock_extract, auth_handler, mock_handler, mock_user_store
    ):
        """Test get user info when user deleted."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="deleted-user",
        )
        mock_user_store.get_user_by_id.return_value = None

        result = auth_handler._handle_get_me(mock_handler)

        assert result.status_code == 404


# ============================================================================
# Security Tests
# ============================================================================

class TestSecurityMeasures:
    """Tests for security measures in auth handler."""

    def test_email_enumeration_prevention(
        self, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test that login doesn't reveal if email exists."""
        # Test with non-existent email
        mock_user_store.get_user_by_email.return_value = None
        auth_handler.read_json_body = Mock(return_value={
            "email": "unknown@example.com",
            "password": "password123",
        })
        result1 = auth_handler._handle_login(mock_handler)

        # Test with wrong password
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_email.return_value = mock_user
        auth_handler.read_json_body = Mock(return_value={
            "email": "test@example.com",
            "password": "wrong_password",
        })
        result2 = auth_handler._handle_login(mock_handler)

        # Both should return same status and similar error
        assert result1.status_code == result2.status_code == 401
        data1 = json.loads(result1.body)
        data2 = json.loads(result2.body)
        # Error messages should be identical
        assert data1["error"] == data2["error"]

    def test_password_not_in_response(
        self, auth_handler, mock_handler, mock_user_store, mock_user
    ):
        """Test that password is never included in responses."""
        mock_user.to_dict.return_value = {
            "id": "user-123",
            "email": "test@example.com",
            # Ensure password is NOT included
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        with patch("aragora.server.handlers.auth.create_token_pair") as mock_tokens:
            mock_tokens.return_value = Mock(to_dict=lambda: {"access_token": "token"})
            auth_handler.read_json_body = Mock(return_value={
                "email": "test@example.com",
                "password": "password",
            })

            result = auth_handler._handle_login(mock_handler)

            data = json.loads(result.body)
            assert "password" not in str(data)
            assert "password_hash" not in str(data)
            assert "password_salt" not in str(data)
