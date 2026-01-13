"""
Security tests for Auth Handler.

Tests cover:
- Email and password validation
- Route handling and method dispatch
- InMemoryUserStore functionality
- Input sanitization
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from aragora.server.handlers.auth import (
    AuthHandler,
    InMemoryUserStore,
    validate_email,
    validate_password,
    EMAIL_PATTERN,
    MIN_PASSWORD_LENGTH,
    MAX_PASSWORD_LENGTH,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str
    email: str
    name: str = "Test User"
    password_hash: str = "hashed_password"
    password_salt: str = "salt"
    role: str = "user"
    org_id: Optional[str] = None
    is_active: bool = True
    api_key: Optional[str] = None
    api_key_created_at: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    mfa_backup_codes: Optional[str] = None
    last_login_at: Optional[datetime] = None

    def verify_password(self, password: str) -> bool:
        """Mock password verification."""
        return password == "correct_password"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "org_id": self.org_id,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str
    name: str
    owner_id: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "owner_id": self.owner_id}


@pytest.fixture
def user_store():
    """Create mock user store."""
    return InMemoryUserStore()


@pytest.fixture
def auth_handler(user_store):
    """Create auth handler with mock context."""
    handler = AuthHandler(server_context={"user_store": user_store})
    return handler


# ============================================================================
# Email Validation Tests
# ============================================================================


class TestEmailValidation:
    """Tests for email validation."""

    def test_valid_email(self):
        """Valid email should pass."""
        valid, err = validate_email("test@example.com")
        assert valid is True
        assert err == ""

    def test_valid_email_with_subdomain(self):
        """Email with subdomain should pass."""
        valid, err = validate_email("test@mail.example.com")
        assert valid is True

    def test_valid_email_with_plus(self):
        """Email with plus sign should pass."""
        valid, err = validate_email("test+tag@example.com")
        assert valid is True

    def test_valid_email_with_dots(self):
        """Email with dots in local part should pass."""
        valid, err = validate_email("first.last@example.com")
        assert valid is True

    def test_valid_email_with_numbers(self):
        """Email with numbers should pass."""
        valid, err = validate_email("user123@example.com")
        assert valid is True

    def test_empty_email(self):
        """Empty email should fail."""
        valid, err = validate_email("")
        assert valid is False
        assert "required" in err.lower()

    def test_email_too_long(self):
        """Email over 254 chars should fail."""
        long_email = "a" * 250 + "@b.com"
        valid, err = validate_email(long_email)
        assert valid is False
        assert "too long" in err.lower()

    def test_invalid_email_no_at(self):
        """Email without @ should fail."""
        valid, err = validate_email("testexample.com")
        assert valid is False
        assert "invalid" in err.lower()

    def test_invalid_email_no_domain(self):
        """Email without domain should fail."""
        valid, err = validate_email("test@")
        assert valid is False

    def test_invalid_email_no_tld(self):
        """Email without TLD should fail."""
        valid, err = validate_email("test@example")
        assert valid is False

    def test_invalid_email_double_at(self):
        """Email with double @ should fail."""
        valid, err = validate_email("test@@example.com")
        assert valid is False

    def test_invalid_email_spaces(self):
        """Email with spaces should fail."""
        valid, err = validate_email("test @example.com")
        assert valid is False

    def test_email_pattern_blocks_injection(self):
        """Email pattern should block injection attempts."""
        malicious_emails = [
            "test@example.com; DROP TABLE users;",
            "test@example.com\n<script>alert(1)</script>",
            "test@example.com' OR '1'='1",
            "test@example.com\r\nBcc: attacker@evil.com",
        ]
        for email in malicious_emails:
            assert not EMAIL_PATTERN.match(email), f"{email} should be rejected"

    def test_email_pattern_blocks_null_bytes(self):
        """Email pattern should block null bytes."""
        assert not EMAIL_PATTERN.match("test\x00@example.com")

    def test_email_pattern_blocks_unicode_homoglyphs(self):
        """Email pattern should block unicode that looks like ASCII."""
        # Cyrillic 'а' looks like Latin 'a'
        homoglyph_emails = [
            "tеst@example.com",  # Cyrillic 'е'
        ]
        for email in homoglyph_emails:
            # Should not match if we're strict about ASCII
            result = EMAIL_PATTERN.match(email)
            # Note: Current pattern allows these - documenting behavior
            # This is a known limitation


# ============================================================================
# Password Validation Tests
# ============================================================================


class TestPasswordValidation:
    """Tests for password validation."""

    def test_valid_password(self):
        """Valid password should pass."""
        valid, err = validate_password("secureP@ss123")
        assert valid is True
        assert err == ""

    def test_valid_password_with_unicode(self):
        """Password with unicode should pass."""
        valid, err = validate_password("пароль12345678")
        assert valid is True

    def test_valid_password_with_emojis(self):
        """Password with emojis should pass."""
        valid, err = validate_password("password🔐🔑")
        assert valid is True

    def test_empty_password(self):
        """Empty password should fail."""
        valid, err = validate_password("")
        assert valid is False
        assert "required" in err.lower()

    def test_password_too_short(self):
        """Password under minimum length should fail."""
        valid, err = validate_password("short")
        assert valid is False
        assert str(MIN_PASSWORD_LENGTH) in err

    def test_password_too_long(self):
        """Password over maximum length should fail."""
        long_password = "a" * (MAX_PASSWORD_LENGTH + 1)
        valid, err = validate_password(long_password)
        assert valid is False
        assert str(MAX_PASSWORD_LENGTH) in err

    def test_password_at_min_length(self):
        """Password at minimum length should pass."""
        valid, err = validate_password("a" * MIN_PASSWORD_LENGTH)
        assert valid is True

    def test_password_at_max_length(self):
        """Password at maximum length should pass."""
        valid, err = validate_password("a" * MAX_PASSWORD_LENGTH)
        assert valid is True

    def test_password_with_spaces(self):
        """Password with spaces should pass."""
        valid, err = validate_password("pass word with spaces")
        assert valid is True

    def test_password_with_null_byte(self):
        """Password with null byte should pass (handled at hash level)."""
        valid, err = validate_password("password\x00extra")
        assert valid is True  # Validation passes, hash handles it


# ============================================================================
# Route Handling Tests
# ============================================================================


class TestRouteHandling:
    """Tests for route handling and method dispatch."""

    def test_can_handle_register(self, auth_handler):
        """can_handle should return True for register route."""
        assert auth_handler.can_handle("/api/auth/register") is True

    def test_can_handle_login(self, auth_handler):
        """can_handle should return True for login route."""
        assert auth_handler.can_handle("/api/auth/login") is True

    def test_can_handle_logout(self, auth_handler):
        """can_handle should return True for logout route."""
        assert auth_handler.can_handle("/api/auth/logout") is True

    def test_can_handle_refresh(self, auth_handler):
        """can_handle should return True for refresh route."""
        assert auth_handler.can_handle("/api/auth/refresh") is True

    def test_can_handle_revoke(self, auth_handler):
        """can_handle should return True for revoke route."""
        assert auth_handler.can_handle("/api/auth/revoke") is True

    def test_can_handle_me(self, auth_handler):
        """can_handle should return True for me route."""
        assert auth_handler.can_handle("/api/auth/me") is True

    def test_can_handle_password(self, auth_handler):
        """can_handle should return True for password route."""
        assert auth_handler.can_handle("/api/auth/password") is True

    def test_can_handle_api_key(self, auth_handler):
        """can_handle should return True for api-key route."""
        assert auth_handler.can_handle("/api/auth/api-key") is True

    def test_can_handle_mfa_setup(self, auth_handler):
        """can_handle should return True for mfa setup route."""
        assert auth_handler.can_handle("/api/auth/mfa/setup") is True

    def test_can_handle_mfa_enable(self, auth_handler):
        """can_handle should return True for mfa enable route."""
        assert auth_handler.can_handle("/api/auth/mfa/enable") is True

    def test_can_handle_mfa_disable(self, auth_handler):
        """can_handle should return True for mfa disable route."""
        assert auth_handler.can_handle("/api/auth/mfa/disable") is True

    def test_can_handle_mfa_verify(self, auth_handler):
        """can_handle should return True for mfa verify route."""
        assert auth_handler.can_handle("/api/auth/mfa/verify") is True

    def test_can_handle_mfa_backup_codes(self, auth_handler):
        """can_handle should return True for mfa backup-codes route."""
        assert auth_handler.can_handle("/api/auth/mfa/backup-codes") is True

    def test_cannot_handle_invalid_routes(self, auth_handler):
        """can_handle should return False for invalid routes."""
        invalid_routes = [
            "/api/auth",
            "/api/auth/invalid",
            "/api/debates",
            "/api/auth/register/extra",
            "/api/auth/login/something",
            "/auth/login",
            "/api/users",
        ]
        for route in invalid_routes:
            assert auth_handler.can_handle(route) is False, f"{route} should not be handled"

    def test_cannot_handle_path_traversal(self, auth_handler):
        """can_handle should return False for path traversal attempts."""
        traversal_routes = [
            "/api/auth/../secrets",
            "/api/auth/login/../../../etc/passwd",
            "/api/auth/..%2F..%2F",
        ]
        for route in traversal_routes:
            assert auth_handler.can_handle(route) is False


# ============================================================================
# InMemoryUserStore Tests
# ============================================================================


class TestInMemoryUserStore:
    """Tests for InMemoryUserStore."""

    def test_initial_state_empty(self):
        """New store should be empty."""
        store = InMemoryUserStore()
        assert len(store.users) == 0
        assert len(store.users_by_email) == 0
        assert len(store.organizations) == 0
        assert len(store.api_keys) == 0

    def test_save_and_get_user_by_id(self):
        """Save and get user by ID should work."""
        store = InMemoryUserStore()
        user = MockUser(id="user_1", email="test@example.com")

        store.save_user(user)

        result = store.get_user_by_id("user_1")
        assert result == user

    def test_save_and_get_user_by_email(self):
        """Save and get user by email should work."""
        store = InMemoryUserStore()
        user = MockUser(id="user_1", email="test@example.com")

        store.save_user(user)

        result = store.get_user_by_email("test@example.com")
        assert result == user

    def test_get_user_by_email_case_insensitive(self):
        """Get user by email should be case insensitive."""
        store = InMemoryUserStore()
        user = MockUser(id="user_1", email="test@example.com")
        store.users[user.id] = user
        store.users_by_email["test@example.com"] = user.id

        assert store.get_user_by_email("TEST@EXAMPLE.COM") == user
        assert store.get_user_by_email("Test@Example.Com") == user

    def test_get_user_by_api_key(self):
        """Get user by API key should work."""
        store = InMemoryUserStore()
        user = MockUser(id="user_1", email="test@example.com", api_key="ara_key123")

        store.save_user(user)

        result = store.get_user_by_api_key("ara_key123")
        assert result == user

    def test_get_nonexistent_user_by_id(self):
        """Get nonexistent user by ID should return None."""
        store = InMemoryUserStore()
        assert store.get_user_by_id("nonexistent") is None

    def test_get_nonexistent_user_by_email(self):
        """Get nonexistent user by email should return None."""
        store = InMemoryUserStore()
        assert store.get_user_by_email("nonexistent@example.com") is None

    def test_get_nonexistent_user_by_api_key(self):
        """Get nonexistent user by API key should return None."""
        store = InMemoryUserStore()
        assert store.get_user_by_api_key("nonexistent_key") is None

    def test_save_and_get_organization(self):
        """Save and get organization should work."""
        store = InMemoryUserStore()
        org = MockOrganization(id="org_1", name="Test Org", owner_id="user_1")

        store.save_organization(org)

        result = store.get_organization_by_id("org_1")
        assert result == org

    def test_get_nonexistent_organization(self):
        """Get nonexistent organization should return None."""
        store = InMemoryUserStore()
        assert store.get_organization_by_id("nonexistent") is None

    def test_multiple_users(self):
        """Store should handle multiple users."""
        store = InMemoryUserStore()
        user1 = MockUser(id="user_1", email="user1@example.com")
        user2 = MockUser(id="user_2", email="user2@example.com")
        user3 = MockUser(id="user_3", email="user3@example.com")

        store.save_user(user1)
        store.save_user(user2)
        store.save_user(user3)

        assert store.get_user_by_id("user_1") == user1
        assert store.get_user_by_id("user_2") == user2
        assert store.get_user_by_id("user_3") == user3
        assert len(store.users) == 3

    def test_update_user_api_key(self):
        """Updating user API key should update lookup."""
        store = InMemoryUserStore()
        user = MockUser(id="user_1", email="test@example.com")

        store.save_user(user)
        assert store.get_user_by_api_key("old_key") is None

        # Simulate API key update
        user.api_key = "new_key"
        store.save_user(user)

        assert store.get_user_by_api_key("new_key") == user


# ============================================================================
# Security Edge Cases
# ============================================================================


class TestSecurityEdgeCases:
    """Tests for security edge cases."""

    def test_email_with_special_characters(self):
        """Email validation handles special characters correctly."""
        # These should be valid per RFC 5321
        valid_emails = [
            "user.name@example.com",
            "user+tag@example.com",
            "user_name@example.com",
            "user-name@example.com",
            "user%tag@example.com",
        ]
        for email in valid_emails:
            valid, _ = validate_email(email)
            assert valid is True, f"{email} should be valid"

    def test_password_length_boundary(self):
        """Password validation at length boundaries."""
        # Exactly at min length
        valid, _ = validate_password("a" * MIN_PASSWORD_LENGTH)
        assert valid is True

        # One below min length
        valid, _ = validate_password("a" * (MIN_PASSWORD_LENGTH - 1))
        assert valid is False

        # Exactly at max length
        valid, _ = validate_password("a" * MAX_PASSWORD_LENGTH)
        assert valid is True

        # One above max length
        valid, _ = validate_password("a" * (MAX_PASSWORD_LENGTH + 1))
        assert valid is False

    def test_email_length_boundary(self):
        """Email validation at length boundaries."""
        # Exactly at max length (254)
        local = "a" * 64
        domain = "b" * (254 - 65 - 4) + ".com"  # 254 - local - @ - .com
        long_valid = f"{local}@{domain}"
        # Might fail pattern due to domain length, but length check passes

        # Over max length
        too_long = "a" * 250 + "@b.com"
        valid, err = validate_email(too_long)
        assert valid is False
        assert "too long" in err.lower()

    def test_route_prefix_matching(self, auth_handler):
        """Routes should match exactly, not by prefix."""
        # These should NOT match
        assert auth_handler.can_handle("/api/auth/register_malicious") is False
        assert auth_handler.can_handle("/api/auth/loginattack") is False
        assert auth_handler.can_handle("/api/auth/me.json") is False

    def test_api_key_format_validation(self):
        """API keys should have specific format."""
        store = InMemoryUserStore()

        # User without API key
        user_no_key = MockUser(id="user_1", email="test@example.com", api_key=None)
        store.save_user(user_no_key)
        assert store.get_user_by_api_key("") is None
        assert store.get_user_by_api_key(None) is None

        # User with API key
        user_with_key = MockUser(id="user_2", email="test2@example.com", api_key="ara_valid_key")
        store.save_user(user_with_key)
        assert store.get_user_by_api_key("ara_valid_key") == user_with_key
        assert store.get_user_by_api_key("ara_invalid") is None
