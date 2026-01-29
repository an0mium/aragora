"""
Tests for password reset token storage.

Tests:
- Token generation and validation
- Rate limiting
- Token expiration
- Token consumption
- Backend implementations (in-memory, SQLite)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from aragora.storage.password_reset_store import (
    DEFAULT_TTL_SECONDS,
    InMemoryPasswordResetStore,
    MAX_ATTEMPTS_PER_EMAIL,
    PasswordResetStore,
    ResetTokenData,
    SQLitePasswordResetStore,
)


class TestResetTokenData:
    """Tests for ResetTokenData dataclass."""

    def test_is_expired_false(self):
        """Token not yet expired."""
        data = ResetTokenData(
            email="test@example.com",
            token_hash="abc123",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.fromtimestamp(time.time() + 3600, tz=timezone.utc),
        )
        assert not data.is_expired
        assert data.is_valid

    def test_is_expired_true(self):
        """Token has expired."""
        data = ResetTokenData(
            email="test@example.com",
            token_hash="abc123",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.fromtimestamp(time.time() - 1, tz=timezone.utc),
        )
        assert data.is_expired
        assert not data.is_valid

    def test_used_token_not_valid(self):
        """Used token is not valid."""
        data = ResetTokenData(
            email="test@example.com",
            token_hash="abc123",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.fromtimestamp(time.time() + 3600, tz=timezone.utc),
            used=True,
        )
        assert not data.is_expired
        assert not data.is_valid


class TestInMemoryPasswordResetStore:
    """Tests for in-memory password reset backend."""

    def test_store_and_retrieve_token(self):
        """Store and retrieve a token."""
        backend = InMemoryPasswordResetStore()
        expires_at = time.time() + 3600

        backend.store_token("test@example.com", "hash123", expires_at)
        data = backend.get_token_data("hash123")

        assert data is not None
        assert data.email == "test@example.com"
        assert data.token_hash == "hash123"
        assert not data.used

    def test_mark_used(self):
        """Mark a token as used."""
        backend = InMemoryPasswordResetStore()
        expires_at = time.time() + 3600

        backend.store_token("test@example.com", "hash123", expires_at)
        result = backend.mark_used("hash123")

        assert result is True
        data = backend.get_token_data("hash123")
        assert data is not None
        assert data.used is True

    def test_delete_token(self):
        """Delete a token."""
        backend = InMemoryPasswordResetStore()
        expires_at = time.time() + 3600

        backend.store_token("test@example.com", "hash123", expires_at)
        result = backend.delete_token("hash123")

        assert result is True
        assert backend.get_token_data("hash123") is None

    def test_count_recent_requests(self):
        """Count recent requests for rate limiting."""
        backend = InMemoryPasswordResetStore()
        expires_at = time.time() + 3600

        # Add multiple tokens for same email
        backend.store_token("test@example.com", "hash1", expires_at)
        backend.store_token("test@example.com", "hash2", expires_at)
        backend.store_token("other@example.com", "hash3", expires_at)

        count = backend.count_recent_requests("test@example.com", 3600)
        assert count == 2

        other_count = backend.count_recent_requests("other@example.com", 3600)
        assert other_count == 1

    def test_cleanup_expired(self):
        """Cleanup expired tokens."""
        backend = InMemoryPasswordResetStore()

        # Add one expired and one valid token
        backend.store_token("expired@example.com", "hash1", time.time() - 1)
        backend.store_token("valid@example.com", "hash2", time.time() + 3600)

        removed = backend.cleanup_expired()
        assert removed == 1
        assert backend.get_token_data("hash1") is None
        assert backend.get_token_data("hash2") is not None

    def test_delete_tokens_for_email(self):
        """Delete all tokens for an email."""
        backend = InMemoryPasswordResetStore()
        expires_at = time.time() + 3600

        backend.store_token("test@example.com", "hash1", expires_at)
        backend.store_token("test@example.com", "hash2", expires_at)
        backend.store_token("other@example.com", "hash3", expires_at)

        deleted = backend.delete_tokens_for_email("test@example.com")
        assert deleted == 2
        assert backend.get_token_data("hash1") is None
        assert backend.get_token_data("hash2") is None
        assert backend.get_token_data("hash3") is not None


class TestSQLitePasswordResetStore:
    """Tests for SQLite password reset backend."""

    @pytest.fixture
    def sqlite_backend(self, tmp_path):
        """Create a SQLite backend with a temporary database."""
        return SQLitePasswordResetStore(tmp_path / "test_reset.db")

    def test_store_and_retrieve_token(self, sqlite_backend):
        """Store and retrieve a token."""
        expires_at = time.time() + 3600

        sqlite_backend.store_token("test@example.com", "hash123", expires_at)
        data = sqlite_backend.get_token_data("hash123")

        assert data is not None
        assert data.email == "test@example.com"
        assert data.token_hash == "hash123"
        assert not data.used

    def test_mark_used(self, sqlite_backend):
        """Mark a token as used."""
        expires_at = time.time() + 3600

        sqlite_backend.store_token("test@example.com", "hash123", expires_at)
        result = sqlite_backend.mark_used("hash123")

        assert result is True
        data = sqlite_backend.get_token_data("hash123")
        assert data is not None
        assert data.used is True

    def test_delete_token(self, sqlite_backend):
        """Delete a token."""
        expires_at = time.time() + 3600

        sqlite_backend.store_token("test@example.com", "hash123", expires_at)
        result = sqlite_backend.delete_token("hash123")

        assert result is True
        assert sqlite_backend.get_token_data("hash123") is None

    def test_cleanup_expired(self, sqlite_backend):
        """Cleanup expired tokens."""
        # Add one expired and one valid token
        sqlite_backend.store_token("expired@example.com", "hash1", time.time() - 1)
        sqlite_backend.store_token("valid@example.com", "hash2", time.time() + 3600)

        removed = sqlite_backend.cleanup_expired()
        assert removed == 1
        assert sqlite_backend.get_token_data("hash1") is None
        assert sqlite_backend.get_token_data("hash2") is not None


class TestPasswordResetStore:
    """Tests for high-level PasswordResetStore."""

    @pytest.fixture
    def store(self):
        """Create a store with in-memory backend."""
        backend = InMemoryPasswordResetStore()
        return PasswordResetStore(backend, ttl_seconds=3600, max_attempts=3)

    def test_create_token(self, store):
        """Create a password reset token."""
        token, error = store.create_token("test@example.com")

        assert error is None
        assert token is not None
        assert len(token) > 32  # URL-safe base64 encoded

    def test_validate_token(self, store):
        """Validate a password reset token."""
        token, _ = store.create_token("test@example.com")

        email, error = store.validate_token(token)

        assert error is None
        assert email == "test@example.com"

    def test_validate_invalid_token(self, store):
        """Validate an invalid token."""
        email, error = store.validate_token("invalid_token_12345")

        assert email is None
        assert error == "Invalid or expired reset token"

    def test_consume_token(self, store):
        """Consume a token after use."""
        token, _ = store.create_token("test@example.com")

        # First validation should work
        email, _ = store.validate_token(token)
        assert email == "test@example.com"

        # Consume the token
        result = store.consume_token(token)
        assert result is True

        # Second validation should fail
        email, error = store.validate_token(token)
        assert email is None
        assert error == "Invalid or expired reset token"

    def test_rate_limiting(self, store):
        """Rate limiting prevents too many requests."""
        email = "ratelimit@example.com"

        # Should succeed up to max_attempts
        for i in range(3):
            token, error = store.create_token(email)
            assert token is not None, f"Request {i + 1} should succeed"
            assert error is None

        # Should be rate limited
        token, error = store.create_token(email)
        assert token is None
        assert "Too many" in error

    def test_invalidate_all_tokens_for_email(self, store):
        """Invalidate all tokens for an email after password reset."""
        # Create multiple tokens
        token1, _ = store.create_token("test@example.com")
        token2, _ = store.create_token("test@example.com")

        # Both should be valid
        email1, _ = store.validate_token(token1)
        email2, _ = store.validate_token(token2)
        assert email1 == "test@example.com"
        assert email2 == "test@example.com"

        # Invalidate all
        count = store.invalidate_tokens_for_email("test@example.com")
        assert count == 2

        # Neither should be valid
        email1, error1 = store.validate_token(token1)
        email2, error2 = store.validate_token(token2)
        assert email1 is None
        assert email2 is None

    def test_email_case_insensitive(self, store):
        """Email addresses are case-insensitive."""
        token, _ = store.create_token("Test@Example.COM")

        email, error = store.validate_token(token)

        assert error is None
        assert email == "test@example.com"  # Normalized to lowercase


class TestPasswordResetStoreExpiration:
    """Tests for token expiration."""

    def test_expired_token_validation(self):
        """Expired tokens are rejected."""
        backend = InMemoryPasswordResetStore()
        store = PasswordResetStore(backend, ttl_seconds=1, max_attempts=10)

        token, _ = store.create_token("test@example.com")

        # Wait for expiration
        time.sleep(1.5)

        email, error = store.validate_token(token)
        assert email is None
        assert error == "Reset token has expired"
