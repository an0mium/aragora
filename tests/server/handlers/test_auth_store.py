"""
Tests for the auth store module.
Tests cover:
- InMemoryUserStore initialization
- User save and retrieval
- Organization save and retrieval
- API key lookup
"""

import pytest
from unittest.mock import MagicMock


class TestInMemoryUserStoreImport:
    """Tests for importing InMemoryUserStore."""

    def test_can_import_store(self):
        """InMemoryUserStore can be imported."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        assert InMemoryUserStore is not None

    def test_store_in_all(self):
        """InMemoryUserStore is in __all__."""
        from aragora.server.handlers.auth import store

        assert "InMemoryUserStore" in store.__all__


class TestInMemoryUserStoreInit:
    """Tests for InMemoryUserStore initialization."""

    def test_init_creates_empty_users(self):
        """Initialization creates empty users dict."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        assert store.users == {}

    def test_init_creates_empty_users_by_email(self):
        """Initialization creates empty users_by_email dict."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        assert store.users_by_email == {}

    def test_init_creates_empty_organizations(self):
        """Initialization creates empty organizations dict."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        assert store.organizations == {}


class TestInMemoryUserStoreSaveUser:
    """Tests for save_user method."""

    def test_save_user_stores_by_id(self):
        """save_user stores user by ID."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"

        store.save_user(user)

        assert store.users["user_123"] is user

    def test_save_user_indexes_by_email(self):
        """save_user indexes user by lowercase email."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "Test@Example.COM"

        store.save_user(user)

        assert store.users_by_email["test@example.com"] == "user_123"

    def test_save_user_overwrites_existing(self):
        """save_user overwrites existing user with same ID."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        user1 = MagicMock()
        user1.id = "user_123"
        user1.email = "old@example.com"

        user2 = MagicMock()
        user2.id = "user_123"
        user2.email = "new@example.com"

        store.save_user(user1)
        store.save_user(user2)

        assert store.users["user_123"] is user2


class TestInMemoryUserStoreGetUserById:
    """Tests for get_user_by_id method."""

    def test_get_existing_user_returns_user(self):
        """get_user_by_id returns existing user."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"
        store.save_user(user)

        result = store.get_user_by_id("user_123")

        assert result is user

    def test_get_nonexistent_user_returns_none(self):
        """get_user_by_id returns None for nonexistent user."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        result = store.get_user_by_id("nonexistent")

        assert result is None


class TestInMemoryUserStoreGetUserByEmail:
    """Tests for get_user_by_email method."""

    def test_get_existing_user_by_email(self):
        """get_user_by_email returns existing user."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"
        store.save_user(user)

        result = store.get_user_by_email("test@example.com")

        assert result is user

    def test_get_user_by_email_case_insensitive(self):
        """get_user_by_email is case insensitive."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"
        store.save_user(user)

        result = store.get_user_by_email("TEST@EXAMPLE.COM")

        assert result is user

    def test_get_nonexistent_email_returns_none(self):
        """get_user_by_email returns None for nonexistent email."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        result = store.get_user_by_email("nonexistent@example.com")

        assert result is None


class TestInMemoryUserStoreGetUserByApiKey:
    """Tests for get_user_by_api_key method."""

    def test_get_user_by_valid_api_key(self):
        """get_user_by_api_key returns user with matching key."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"
        user.verify_api_key = MagicMock(return_value=True)
        store.save_user(user)

        result = store.get_user_by_api_key("valid_key")

        assert result is user
        user.verify_api_key.assert_called_with("valid_key")

    def test_get_user_by_invalid_api_key_returns_none(self):
        """get_user_by_api_key returns None for invalid key."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock()
        user.id = "user_123"
        user.email = "test@example.com"
        user.verify_api_key = MagicMock(return_value=False)
        store.save_user(user)

        result = store.get_user_by_api_key("invalid_key")

        assert result is None

    def test_get_user_without_verify_method(self):
        """get_user_by_api_key skips users without verify_api_key."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        user = MagicMock(spec=[])  # No verify_api_key method
        user.id = "user_123"
        user.email = "test@example.com"
        store.save_user(user)

        result = store.get_user_by_api_key("some_key")

        assert result is None


class TestInMemoryUserStoreSaveOrganization:
    """Tests for save_organization method."""

    def test_save_organization_stores_by_id(self):
        """save_organization stores org by ID."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        org = MagicMock()
        org.id = "org_123"

        store.save_organization(org)

        assert store.organizations["org_123"] is org


class TestInMemoryUserStoreGetOrganizationById:
    """Tests for get_organization_by_id method."""

    def test_get_existing_org_returns_org(self):
        """get_organization_by_id returns existing org."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()
        org = MagicMock()
        org.id = "org_123"
        store.save_organization(org)

        result = store.get_organization_by_id("org_123")

        assert result is org

    def test_get_nonexistent_org_returns_none(self):
        """get_organization_by_id returns None for nonexistent org."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        result = store.get_organization_by_id("nonexistent")

        assert result is None


class TestInMemoryUserStoreMultipleUsers:
    """Tests for multiple user scenarios."""

    def test_multiple_users_stored_correctly(self):
        """Multiple users are stored and retrieved correctly."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        user1 = MagicMock()
        user1.id = "user_1"
        user1.email = "user1@example.com"

        user2 = MagicMock()
        user2.id = "user_2"
        user2.email = "user2@example.com"

        store.save_user(user1)
        store.save_user(user2)

        assert store.get_user_by_id("user_1") is user1
        assert store.get_user_by_id("user_2") is user2
        assert store.get_user_by_email("user1@example.com") is user1
        assert store.get_user_by_email("user2@example.com") is user2

    def test_api_key_search_finds_correct_user(self):
        """API key search finds the correct user among multiple."""
        from aragora.server.handlers.auth.store import InMemoryUserStore

        store = InMemoryUserStore()

        user1 = MagicMock()
        user1.id = "user_1"
        user1.email = "user1@example.com"
        user1.verify_api_key = MagicMock(return_value=False)

        user2 = MagicMock()
        user2.id = "user_2"
        user2.email = "user2@example.com"
        user2.verify_api_key = MagicMock(return_value=True)

        store.save_user(user1)
        store.save_user(user2)

        result = store.get_user_by_api_key("user2_key")

        assert result is user2
