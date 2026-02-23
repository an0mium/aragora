"""Tests for InMemoryUserStore (aragora/server/handlers/auth/store.py).

Covers all public methods, edge cases, and internal data structure invariants:
- save_user / get_user_by_id / get_user_by_email / get_user_by_api_key
- save_organization / get_organization_by_id
- Email case-insensitivity
- API key hash-based verification
- Missing/None lookups
- Overwrite semantics
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.auth.store import InMemoryUserStore


# ---------------------------------------------------------------------------
# Helpers: lightweight User / Organization fakes
# ---------------------------------------------------------------------------


def _make_user(
    user_id: str = "u-1",
    email: str = "alice@example.com",
    *,
    verify_api_key_fn=None,
) -> MagicMock:
    """Create a minimal mock User with required attributes."""
    user = MagicMock()
    user.id = user_id
    user.email = email
    if verify_api_key_fn is not None:
        user.verify_api_key = verify_api_key_fn
    else:
        # Default: no verify_api_key attribute
        del user.verify_api_key
    return user


def _make_org(org_id: str = "org-1") -> MagicMock:
    """Create a minimal mock Organization."""
    org = MagicMock()
    org.id = org_id
    return org


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for InMemoryUserStore.__init__."""

    def test_initial_users_empty(self):
        store = InMemoryUserStore()
        assert store.users == {}

    def test_initial_users_by_email_empty(self):
        store = InMemoryUserStore()
        assert store.users_by_email == {}

    def test_initial_organizations_empty(self):
        store = InMemoryUserStore()
        assert store.organizations == {}


# ---------------------------------------------------------------------------
# save_user / get_user_by_id
# ---------------------------------------------------------------------------


class TestSaveAndGetUserById:
    """Tests for save_user and get_user_by_id."""

    def test_save_and_retrieve_user(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "alice@example.com")
        store.save_user(user)
        assert store.get_user_by_id("u-1") is user

    def test_get_user_by_id_missing(self):
        store = InMemoryUserStore()
        assert store.get_user_by_id("nonexistent") is None

    def test_save_user_overwrites_existing(self):
        store = InMemoryUserStore()
        user_v1 = _make_user("u-1", "alice@example.com")
        user_v2 = _make_user("u-1", "alice-new@example.com")
        store.save_user(user_v1)
        store.save_user(user_v2)
        assert store.get_user_by_id("u-1") is user_v2

    def test_save_multiple_users(self):
        store = InMemoryUserStore()
        u1 = _make_user("u-1", "alice@example.com")
        u2 = _make_user("u-2", "bob@example.com")
        store.save_user(u1)
        store.save_user(u2)
        assert store.get_user_by_id("u-1") is u1
        assert store.get_user_by_id("u-2") is u2

    def test_save_user_populates_email_index(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "alice@example.com")
        store.save_user(user)
        assert "alice@example.com" in store.users_by_email
        assert store.users_by_email["alice@example.com"] == "u-1"


# ---------------------------------------------------------------------------
# get_user_by_email
# ---------------------------------------------------------------------------


class TestGetUserByEmail:
    """Tests for get_user_by_email with case-insensitive lookup."""

    def test_exact_match(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "alice@example.com")
        store.save_user(user)
        assert store.get_user_by_email("alice@example.com") is user

    def test_case_insensitive_lookup(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "Alice@Example.COM")
        store.save_user(user)
        assert store.get_user_by_email("alice@example.com") is user

    def test_case_insensitive_save(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "ALICE@EXAMPLE.COM")
        store.save_user(user)
        # Internally stored lower-cased
        assert "alice@example.com" in store.users_by_email

    def test_email_not_found(self):
        store = InMemoryUserStore()
        assert store.get_user_by_email("nobody@example.com") is None

    def test_email_not_found_empty_store(self):
        store = InMemoryUserStore()
        assert store.get_user_by_email("test@test.com") is None

    def test_email_lookup_after_overwrite(self):
        """When a user is saved again with a new email, the new email maps to them."""
        store = InMemoryUserStore()
        user_v1 = _make_user("u-1", "old@example.com")
        store.save_user(user_v1)
        user_v2 = _make_user("u-1", "new@example.com")
        store.save_user(user_v2)
        # New email resolves to the user
        assert store.get_user_by_email("new@example.com") is user_v2

    def test_stale_email_after_overwrite(self):
        """Old email index entry still points to the id but returns the new user object."""
        store = InMemoryUserStore()
        user_v1 = _make_user("u-1", "old@example.com")
        store.save_user(user_v1)
        user_v2 = _make_user("u-1", "new@example.com")
        store.save_user(user_v2)
        # Old email still has the id mapping (not cleaned up), but returns the new user
        result = store.get_user_by_email("old@example.com")
        assert result is user_v2

    def test_email_dangling_reference_returns_none(self):
        """If email index points to an id no longer in users, returns None."""
        store = InMemoryUserStore()
        user = _make_user("u-1", "alice@example.com")
        store.save_user(user)
        # Manually remove from users dict (simulating corruption/cleanup)
        del store.users["u-1"]
        assert store.get_user_by_email("alice@example.com") is None


# ---------------------------------------------------------------------------
# get_user_by_api_key
# ---------------------------------------------------------------------------


class TestGetUserByApiKey:
    """Tests for get_user_by_api_key with hash-based verification."""

    def test_returns_none_for_none_key(self):
        store = InMemoryUserStore()
        assert store.get_user_by_api_key(None) is None

    def test_returns_none_for_empty_string_key(self):
        store = InMemoryUserStore()
        assert store.get_user_by_api_key("") is None

    def test_returns_none_when_no_users(self):
        store = InMemoryUserStore()
        assert store.get_user_by_api_key("some-key") is None

    def test_returns_none_when_no_user_has_verify_method(self):
        store = InMemoryUserStore()
        # User without verify_api_key attribute
        user = _make_user("u-1", "alice@example.com")
        store.save_user(user)
        assert store.get_user_by_api_key("some-key") is None

    def test_returns_matching_user(self):
        store = InMemoryUserStore()
        user = _make_user(
            "u-1",
            "alice@example.com",
            verify_api_key_fn=lambda key: key == "correct-key",
        )
        store.save_user(user)
        assert store.get_user_by_api_key("correct-key") is user

    def test_returns_none_when_key_does_not_match(self):
        store = InMemoryUserStore()
        user = _make_user(
            "u-1",
            "alice@example.com",
            verify_api_key_fn=lambda key: False,
        )
        store.save_user(user)
        assert store.get_user_by_api_key("wrong-key") is None

    def test_returns_first_matching_user_among_multiple(self):
        store = InMemoryUserStore()
        u1 = _make_user(
            "u-1",
            "alice@example.com",
            verify_api_key_fn=lambda key: key == "key-1",
        )
        u2 = _make_user(
            "u-2",
            "bob@example.com",
            verify_api_key_fn=lambda key: key == "key-2",
        )
        store.save_user(u1)
        store.save_user(u2)
        assert store.get_user_by_api_key("key-2") is u2

    def test_skips_users_without_verify_api_key(self):
        """Users without the verify_api_key attribute are silently skipped."""
        store = InMemoryUserStore()
        u_no_method = _make_user("u-1", "alice@example.com")
        u_with_method = _make_user(
            "u-2",
            "bob@example.com",
            verify_api_key_fn=lambda key: key == "bobs-key",
        )
        store.save_user(u_no_method)
        store.save_user(u_with_method)
        assert store.get_user_by_api_key("bobs-key") is u_with_method

    def test_verify_api_key_called_with_provided_key(self):
        store = InMemoryUserStore()
        mock_verify = MagicMock(return_value=True)
        user = _make_user(
            "u-1",
            "alice@example.com",
            verify_api_key_fn=mock_verify,
        )
        store.save_user(user)
        store.get_user_by_api_key("test-api-key-123")
        mock_verify.assert_called_once_with("test-api-key-123")

    def test_stops_iteration_on_first_match(self):
        """Once a matching user is found, no further users are checked."""
        store = InMemoryUserStore()
        mock_verify_1 = MagicMock(return_value=True)
        mock_verify_2 = MagicMock(return_value=True)
        u1 = _make_user("u-1", "a@a.com", verify_api_key_fn=mock_verify_1)
        u2 = _make_user("u-2", "b@b.com", verify_api_key_fn=mock_verify_2)
        store.save_user(u1)
        store.save_user(u2)
        result = store.get_user_by_api_key("any-key")
        # First user matches, second should not be checked
        assert result is u1
        mock_verify_1.assert_called_once()
        mock_verify_2.assert_not_called()


# ---------------------------------------------------------------------------
# save_organization / get_organization_by_id
# ---------------------------------------------------------------------------


class TestOrganizations:
    """Tests for save_organization and get_organization_by_id."""

    def test_save_and_retrieve_organization(self):
        store = InMemoryUserStore()
        org = _make_org("org-1")
        store.save_organization(org)
        assert store.get_organization_by_id("org-1") is org

    def test_get_organization_not_found(self):
        store = InMemoryUserStore()
        assert store.get_organization_by_id("nonexistent") is None

    def test_save_organization_overwrites(self):
        store = InMemoryUserStore()
        org_v1 = _make_org("org-1")
        org_v2 = _make_org("org-1")
        store.save_organization(org_v1)
        store.save_organization(org_v2)
        assert store.get_organization_by_id("org-1") is org_v2

    def test_multiple_organizations(self):
        store = InMemoryUserStore()
        o1 = _make_org("org-1")
        o2 = _make_org("org-2")
        store.save_organization(o1)
        store.save_organization(o2)
        assert store.get_organization_by_id("org-1") is o1
        assert store.get_organization_by_id("org-2") is o2

    def test_organizations_independent_of_users(self):
        store = InMemoryUserStore()
        user = _make_user("u-1", "alice@example.com")
        org = _make_org("org-1")
        store.save_user(user)
        store.save_organization(org)
        assert len(store.users) == 1
        assert len(store.organizations) == 1


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify __all__ exports."""

    def test_all_exports_inmemoryuserstore(self):
        from aragora.server.handlers.auth import store as mod

        assert "InMemoryUserStore" in mod.__all__

    def test_all_exports_length(self):
        from aragora.server.handlers.auth import store as mod

        assert len(mod.__all__) == 1
