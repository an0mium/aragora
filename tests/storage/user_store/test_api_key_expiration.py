"""
Tests for API key expiration checking in UserStore.

Tests the get_user_by_api_key method's expiration validation:
- Expired API key returns None
- Non-expired API key returns user
- API key with no expiration returns user
- Invalid expiration date format still returns user (graceful degradation)
"""

import hashlib
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import User, hash_password
from aragora.storage.user_store.sqlite_store import UserStore


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    db_path.unlink(missing_ok=True)
    # Also clean up WAL and SHM files
    Path(str(db_path) + "-wal").unlink(missing_ok=True)
    Path(str(db_path) + "-shm").unlink(missing_ok=True)


@pytest.fixture
def user_store(temp_db_path):
    """Create a UserStore instance with temporary database."""
    store = UserStore(temp_db_path)
    yield store
    store.close()


@pytest.fixture
def user_with_api_key(user_store):
    """Create a user with an API key for testing."""
    password_hash, password_salt = hash_password("testpass123")
    user = user_store.create_user(
        email="apitest@example.com",
        password_hash=password_hash,
        password_salt=password_salt,
        name="API Test User",
    )
    # Generate and set API key
    api_key = f"ara_test_{user.id[:8]}_secretkey123"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    api_key_prefix = api_key[:12]

    user_store.update_user(
        user.id,
        api_key=api_key,
        api_key_hash=api_key_hash,
        api_key_prefix=api_key_prefix,
        api_key_created_at=datetime.now(timezone.utc).isoformat(),
    )

    return user, api_key


class TestApiKeyExpiration:
    """Tests for API key expiration checking."""

    def test_expired_api_key_returns_none(self, user_store, user_with_api_key):
        """Test that an expired API key returns None."""
        user, api_key = user_with_api_key

        # Set expiration to the past
        expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        user_store.update_user(user.id, api_key_expires_at=expired_time)

        # Attempt to get user by expired API key
        result = user_store.get_user_by_api_key(api_key)

        assert result is None

    def test_non_expired_api_key_returns_user(self, user_store, user_with_api_key):
        """Test that a non-expired API key returns the user."""
        user, api_key = user_with_api_key

        # Set expiration to the future
        future_time = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        user_store.update_user(user.id, api_key_expires_at=future_time)

        # Get user by valid API key
        result = user_store.get_user_by_api_key(api_key)

        assert result is not None
        assert result.id == user.id
        assert result.email == user.email

    def test_api_key_with_no_expiration_returns_user(self, user_store, user_with_api_key):
        """Test that an API key with no expiration date returns the user."""
        user, api_key = user_with_api_key

        # Ensure no expiration is set (should be None by default)
        user_store.update_user(user.id, api_key_expires_at=None)

        # Get user by API key with no expiration
        result = user_store.get_user_by_api_key(api_key)

        assert result is not None
        assert result.id == user.id

    def test_invalid_expiration_date_format_behavior(self, user_store, user_with_api_key):
        """Test behavior when an invalid expiration date format is stored.

        Note: When the repository layer encounters an invalid datetime string, it
        falls back to datetime.now(). This means by the time the expiration check
        runs, the key will appear expired (since any time has passed since parsing).
        This is a fail-secure behavior - invalid dates result in expired keys rather
        than allowing potentially compromised keys to work indefinitely.
        """
        user, api_key = user_with_api_key

        # Set an invalid expiration date format
        # We need to bypass normal validation by updating directly
        with user_store._transaction() as cursor:
            cursor.execute(
                "UPDATE users SET api_key_expires_at = ? WHERE id = ?",
                ("invalid-date-format", user.id),
            )

        # Get user by API key - will fail because invalid date falls back to now()
        # which immediately expires on the subsequent comparison
        result = user_store.get_user_by_api_key(api_key)

        # The key is effectively treated as expired (fail-secure behavior)
        assert result is None

    def test_expired_api_key_logs_debug(self, user_store, user_with_api_key):
        """Test that using an expired API key logs a debug message.

        Note: The repository layer logs at DEBUG level for expired API keys.
        """
        user, api_key = user_with_api_key

        # Set expiration to the past
        expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        user_store.update_user(user.id, api_key_expires_at=expired_time)

        # Patch the repository logger to verify debug message is logged
        with patch("aragora.storage.repositories.users.logger") as mock_logger:
            result = user_store.get_user_by_api_key(api_key)

            assert result is None
            mock_logger.debug.assert_called_once()
            assert "expired" in mock_logger.debug.call_args[0][0].lower()

    def test_api_key_expires_at_boundary(self, user_store, user_with_api_key):
        """Test API key expiration at exact boundary (just expired vs just valid)."""
        user, api_key = user_with_api_key

        # Set expiration to exactly 1 second in the past
        just_expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        user_store.update_user(user.id, api_key_expires_at=just_expired)

        result = user_store.get_user_by_api_key(api_key)
        assert result is None

        # Now set to 1 minute in the future
        still_valid = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        user_store.update_user(user.id, api_key_expires_at=still_valid)

        result = user_store.get_user_by_api_key(api_key)
        assert result is not None
        assert result.id == user.id

    def test_api_key_expiration_with_z_suffix(self, user_store, user_with_api_key):
        """Test that expiration dates with Z suffix are handled correctly."""
        user, api_key = user_with_api_key

        # Set expiration with Z suffix (UTC indicator)
        future_time = datetime.now(timezone.utc) + timedelta(days=30)
        z_format_time = future_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        with user_store._transaction() as cursor:
            cursor.execute(
                "UPDATE users SET api_key_expires_at = ? WHERE id = ?", (z_format_time, user.id)
            )

        # Should still work with Z suffix
        result = user_store.get_user_by_api_key(api_key)

        assert result is not None
        assert result.id == user.id

    def test_invalid_api_key_returns_none(self, user_store, user_with_api_key):
        """Test that an invalid API key returns None (baseline behavior)."""
        result = user_store.get_user_by_api_key("invalid_api_key_that_does_not_exist")
        assert result is None
