"""
Tests for UserRepository.

These tests verify the UserRepository works correctly when used directly,
independent of the UserStore facade.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

from aragora.storage.repositories import UserRepository


@pytest.fixture
def temp_db():
    """Create a temporary database with the users table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # Create users table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            name TEXT DEFAULT '',
            org_id TEXT,
            role TEXT DEFAULT 'member',
            is_active INTEGER DEFAULT 1,
            email_verified INTEGER DEFAULT 0,
            api_key TEXT UNIQUE,
            api_key_hash TEXT UNIQUE,
            api_key_prefix TEXT,
            api_key_created_at TEXT,
            api_key_expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_login_at TEXT,
            mfa_secret TEXT,
            mfa_enabled INTEGER DEFAULT 0,
            mfa_backup_codes TEXT,
            token_version INTEGER DEFAULT 1,
            preferences TEXT
        )
    """
    )
    conn.commit()

    @contextmanager
    def transaction():
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    yield transaction, conn

    conn.close()
    Path(db_path).unlink(missing_ok=True)


class TestUserRepositoryCreate:
    """Tests for user creation."""

    def test_create_user(self, temp_db):
        """Test creating a basic user."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(
            email="test@example.com",
            password_hash="hash123",
            password_salt="salt123",
            name="Test User",
        )

        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.id is not None
        assert user.is_active is True

    def test_create_user_duplicate_email(self, temp_db):
        """Test creating user with duplicate email raises ValueError."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        repo.create(email="test@example.com", password_hash="hash", password_salt="salt")

        with pytest.raises(ValueError, match="Email already exists"):
            repo.create(email="test@example.com", password_hash="hash2", password_salt="salt2")


class TestUserRepositoryGet:
    """Tests for user retrieval."""

    def test_get_by_id(self, temp_db):
        """Test getting user by ID."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        created = repo.create(
            email="test@example.com",
            password_hash="hash",
            password_salt="salt",
        )

        found = repo.get_by_id(created.id)
        assert found is not None
        assert found.email == "test@example.com"

    def test_get_by_id_not_found(self, temp_db):
        """Test getting non-existent user returns None."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        found = repo.get_by_id("nonexistent")
        assert found is None

    def test_get_by_email(self, temp_db):
        """Test getting user by email."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        repo.create(email="test@example.com", password_hash="hash", password_salt="salt")

        found = repo.get_by_email("test@example.com")
        assert found is not None
        assert found.email == "test@example.com"

    def test_get_by_email_lowercases_query(self, temp_db):
        """Test email lookup lowercases the query parameter."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        # Create with lowercase email
        repo.create(email="test@example.com", password_hash="hash", password_salt="salt")

        # Query with uppercase - should find due to lowercase in get_by_email
        found = repo.get_by_email("TEST@EXAMPLE.COM")
        assert found is not None
        assert found.email == "test@example.com"

    def test_get_batch(self, temp_db):
        """Test batch user retrieval."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user1 = repo.create(email="user1@example.com", password_hash="h", password_salt="s")
        user2 = repo.create(email="user2@example.com", password_hash="h", password_salt="s")
        repo.create(email="user3@example.com", password_hash="h", password_salt="s")

        result = repo.get_batch([user1.id, user2.id, "nonexistent"])

        assert len(result) == 2
        assert user1.id in result
        assert user2.id in result
        assert "nonexistent" not in result


class TestUserRepositoryUpdate:
    """Tests for user updates."""

    def test_update_name(self, temp_db):
        """Test updating user name."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        result = repo.update(user.id, name="New Name")
        assert result is True

        updated = repo.get_by_id(user.id)
        assert updated.name == "New Name"

    def test_update_multiple_fields(self, temp_db):
        """Test updating multiple fields at once."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        repo.update(user.id, name="Updated", is_active=False, email_verified=True)

        updated = repo.get_by_id(user.id)
        assert updated.name == "Updated"
        assert updated.is_active is False
        assert updated.email_verified is True

    def test_update_not_found(self, temp_db):
        """Test updating non-existent user returns False."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        result = repo.update("nonexistent", name="Test")
        assert result is False


class TestUserRepositoryDelete:
    """Tests for user deletion."""

    def test_delete_user(self, temp_db):
        """Test deleting a user."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        result = repo.delete(user.id)
        assert result is True

        found = repo.get_by_id(user.id)
        assert found is None

    def test_delete_not_found(self, temp_db):
        """Test deleting non-existent user returns False."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        result = repo.delete("nonexistent")
        assert result is False


class TestUserRepositoryPreferences:
    """Tests for user preferences."""

    def test_set_and_get_preferences(self, temp_db):
        """Test setting and getting user preferences."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        prefs = {"theme": "dark", "notifications": True}
        repo.set_preferences(user.id, prefs)

        result = repo.get_preferences(user.id)
        assert result == prefs

    def test_get_preferences_empty(self, temp_db):
        """Test getting preferences when none set returns empty dict."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        result = repo.get_preferences(user.id)
        assert result == {}


class TestUserRepositoryTokenVersion:
    """Tests for token version management."""

    def test_increment_token_version(self, temp_db):
        """Test incrementing token version."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        user = repo.create(email="test@example.com", password_hash="h", password_salt="s")

        new_version = repo.increment_token_version(user.id)
        assert new_version == 2

        # Increment again
        new_version = repo.increment_token_version(user.id)
        assert new_version == 3

    def test_increment_token_version_not_found(self, temp_db):
        """Test incrementing token version for non-existent user."""
        transaction_fn, _ = temp_db
        repo = UserRepository(transaction_fn)

        result = repo.increment_token_version("nonexistent")
        assert result == 0
