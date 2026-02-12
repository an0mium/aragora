"""
Tests for PostgresUserStore - PostgreSQL backend for user and organization persistence.

Tests cover:
- Initialization and schema setup
- User CRUD operations (create, read, update, delete)
- User query methods (by email, API key, batch)
- Organization CRUD operations
- Organization membership management
- Usage tracking and billing
- OAuth provider linking
- Audit logging
- Organization invitations
- Account lockout methods
- Admin listing and stats
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User
from aragora.storage.user_store.postgres_store import PostgresUserStore


# =============================================================================
# Fixtures
# =============================================================================


def _make_async_context_manager(conn):
    """Create an async context manager that yields the given connection mock."""

    @asynccontextmanager
    async def _acquire():
        yield conn

    return _acquire


@pytest.fixture
def mock_conn():
    """Create a mock database connection with default return values."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@pytest.fixture
def mock_pool(mock_conn):
    """Create a mock asyncpg pool that yields the mock connection."""
    pool = MagicMock()
    pool.acquire = _make_async_context_manager(mock_conn)
    return pool


@pytest.fixture
def store(mock_pool):
    """Create a PostgresUserStore with a mocked pool."""
    return PostgresUserStore(mock_pool)


@pytest.fixture
def sample_user_row():
    """Return a dict-like mock row for a user."""
    now = datetime.now(timezone.utc)
    row = {
        "id": "user-123",
        "email": "test@example.com",
        "password_hash": "hashed_pw",
        "password_salt": "salt123",
        "name": "Test User",
        "org_id": "org-456",
        "role": "member",
        "is_active": True,
        "email_verified": False,
        "api_key": None,
        "api_key_hash": None,
        "api_key_prefix": None,
        "api_key_created_at": None,
        "api_key_expires_at": None,
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
        "mfa_secret": None,
        "mfa_enabled": False,
        "mfa_backup_codes": None,
        "token_version": 1,
        "failed_login_attempts": 0,
        "lockout_until": None,
        "last_failed_login_at": None,
        "preferences": "{}",
    }
    return row


@pytest.fixture
def sample_org_row():
    """Return a dict-like mock row for an organization."""
    now = datetime.now(timezone.utc)
    return {
        "id": "org-456",
        "name": "Test Org",
        "slug": "test-org",
        "tier": "free",
        "owner_id": "user-123",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "debates_used_this_month": 5,
        "billing_cycle_start": now,
        "settings": "{}",
        "created_at": now,
        "updated_at": now,
    }


@pytest.fixture
def sample_invitation_row():
    """Return a dict-like mock row for an invitation."""
    now = datetime.now(timezone.utc)
    return {
        "id": "inv-789",
        "org_id": "org-456",
        "email": "invitee@example.com",
        "role": "member",
        "token": "tok_abc123",
        "invited_by": "user-123",
        "status": "pending",
        "created_at": now,
        "expires_at": now + timedelta(days=7),
        "accepted_by": None,
        "accepted_at": None,
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for PostgresUserStore initialization and schema setup."""

    def test_constructor_sets_pool(self, mock_pool):
        """Constructor should store the pool and set _initialized to False."""
        store = PostgresUserStore(mock_pool)
        assert store._pool is mock_pool
        assert store._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_executes_schema(self, store, mock_conn):
        """initialize() should execute the INITIAL_SCHEMA SQL."""
        await store.initialize()
        mock_conn.execute.assert_called_once_with(PostgresUserStore.INITIAL_SCHEMA)
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, store, mock_conn):
        """initialize() should be a no-op on second call."""
        await store.initialize()
        await store.initialize()
        # Only called once because second call short-circuits
        mock_conn.execute.assert_called_once()

    def test_schema_constants(self):
        """Verify class-level schema constants."""
        assert PostgresUserStore.SCHEMA_NAME == "user_store"
        assert PostgresUserStore.SCHEMA_VERSION == 1

    def test_close_is_noop(self, store):
        """close() should be a no-op (pool managed externally)."""
        store.close()  # Should not raise


# =============================================================================
# User CRUD Tests
# =============================================================================


class TestUserCreate:
    """Tests for user creation."""

    @pytest.mark.asyncio
    async def test_create_user_async(self, store, mock_conn):
        """create_user_async should insert a row and return a User."""
        user = await store.create_user_async(
            email="alice@example.com",
            password_hash="hash123",
            password_salt="salt456",
            name="Alice",
            org_id="org-1",
            role="admin",
        )
        assert isinstance(user, User)
        assert user.email == "alice@example.com"
        assert user.password_hash == "hash123"
        assert user.password_salt == "salt456"
        assert user.name == "Alice"
        assert user.org_id == "org-1"
        assert user.role == "admin"
        assert user.is_active is True
        assert user.email_verified is False
        assert user.id  # Should be a UUID string
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_defaults(self, store, mock_conn):
        """create_user_async should use sensible defaults."""
        user = await store.create_user_async(
            email="bob@example.com",
            password_hash="h",
            password_salt="s",
        )
        assert user.name == ""
        assert user.org_id is None
        assert user.role == "member"


class TestUserRead:
    """Tests for user retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_user_by_id_found(self, store, mock_conn, sample_user_row):
        """get_user_by_id_async should return a User when the row exists."""
        mock_conn.fetchrow.return_value = sample_user_row
        user = await store.get_user_by_id_async("user-123")
        assert user is not None
        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, store, mock_conn):
        """get_user_by_id_async should return None when no row exists."""
        mock_conn.fetchrow.return_value = None
        user = await store.get_user_by_id_async("nonexistent")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_email_found(self, store, mock_conn, sample_user_row):
        """get_user_by_email_async should return a User by email."""
        mock_conn.fetchrow.return_value = sample_user_row
        user = await store.get_user_by_email_async("test@example.com")
        assert user is not None
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, store, mock_conn):
        """get_user_by_email_async should return None for unknown email."""
        mock_conn.fetchrow.return_value = None
        user = await store.get_user_by_email_async("unknown@example.com")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_api_key(self, store, mock_conn, sample_user_row):
        """get_user_by_api_key_async should hash the key and query."""
        mock_conn.fetchrow.return_value = sample_user_row
        user = await store.get_user_by_api_key_async("my-secret-key")
        assert user is not None
        # Verify the hash was computed correctly
        expected_hash = hashlib.sha256(b"my-secret-key").hexdigest()
        call_args = mock_conn.fetchrow.call_args
        assert call_args[0][1] == expected_hash
        assert call_args[0][2] == "my-secret-key"

    @pytest.mark.asyncio
    async def test_get_users_batch_empty(self, store, mock_conn):
        """get_users_batch_async should return empty dict for empty input."""
        result = await store.get_users_batch_async([])
        assert result == {}
        mock_conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_users_batch_multiple(self, store, mock_conn, sample_user_row):
        """get_users_batch_async should return a dict keyed by user ID."""
        row2 = dict(sample_user_row)
        row2["id"] = "user-999"
        row2["email"] = "other@example.com"
        mock_conn.fetch.return_value = [sample_user_row, row2]

        result = await store.get_users_batch_async(["user-123", "user-999"])
        assert len(result) == 2
        assert "user-123" in result
        assert "user-999" in result


class TestUserUpdate:
    """Tests for user update and delete operations."""

    @pytest.mark.asyncio
    async def test_update_user_success(self, store, mock_conn):
        """update_user_async should build SET clause and return True on success."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.update_user_async("user-123", name="New Name", role="admin")
        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Should contain name and role and updated_at parameters
        assert "New Name" in call_args[0]
        assert "admin" in call_args[0]

    @pytest.mark.asyncio
    async def test_update_user_no_fields(self, store, mock_conn):
        """update_user_async with no fields should return False."""
        result = await store.update_user_async("user-123")
        assert result is False
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, store, mock_conn):
        """update_user_async should return False when no rows updated."""
        mock_conn.execute.return_value = "UPDATE 0"
        result = await store.update_user_async("nonexistent", name="X")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_user_success(self, store, mock_conn):
        """delete_user_async should return True when a row is deleted."""
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.delete_user_async("user-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, store, mock_conn):
        """delete_user_async should return False when no row found."""
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.delete_user_async("nonexistent")
        assert result is False


class TestUserPreferences:
    """Tests for user preferences operations."""

    @pytest.mark.asyncio
    async def test_get_user_preferences_found(self, store, mock_conn):
        """get_user_preferences_async should return parsed JSON preferences."""
        mock_conn.fetchrow.return_value = {"preferences": {"theme": "dark", "lang": "en"}}
        prefs = await store.get_user_preferences_async("user-123")
        assert prefs == {"theme": "dark", "lang": "en"}

    @pytest.mark.asyncio
    async def test_get_user_preferences_as_string(self, store, mock_conn):
        """get_user_preferences_async should parse string-typed JSON."""
        mock_conn.fetchrow.return_value = {"preferences": '{"theme": "light"}'}
        prefs = await store.get_user_preferences_async("user-123")
        assert prefs == {"theme": "light"}

    @pytest.mark.asyncio
    async def test_get_user_preferences_not_found(self, store, mock_conn):
        """get_user_preferences_async should return None for missing user."""
        mock_conn.fetchrow.return_value = None
        prefs = await store.get_user_preferences_async("nonexistent")
        assert prefs is None

    @pytest.mark.asyncio
    async def test_set_user_preferences(self, store, mock_conn):
        """set_user_preferences_async should update and return True on success."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.set_user_preferences_async("user-123", {"theme": "dark"})
        assert result is True
        call_args = mock_conn.execute.call_args
        assert json.dumps({"theme": "dark"}) in call_args[0]


class TestTokenVersion:
    """Tests for token version increment."""

    @pytest.mark.asyncio
    async def test_increment_token_version(self, store, mock_conn):
        """increment_token_version_async should return the new version."""
        mock_conn.fetchrow.return_value = {"token_version": 3}
        version = await store.increment_token_version_async("user-123")
        assert version == 3

    @pytest.mark.asyncio
    async def test_increment_token_version_missing_user(self, store, mock_conn):
        """increment_token_version_async should return 1 when user not found."""
        mock_conn.fetchrow.return_value = None
        version = await store.increment_token_version_async("nonexistent")
        assert version == 1


# =============================================================================
# Organization Tests
# =============================================================================


class TestOrganizationCRUD:
    """Tests for organization create, read, and update operations."""

    @pytest.mark.asyncio
    async def test_create_organization(self, store, mock_conn):
        """create_organization_async should insert org and update owner."""
        org = await store.create_organization_async(
            name="My Org",
            owner_id="user-123",
            tier=SubscriptionTier.PROFESSIONAL,
        )
        assert isinstance(org, Organization)
        assert org.name == "My Org"
        assert org.owner_id == "user-123"
        assert org.tier == SubscriptionTier.PROFESSIONAL
        assert org.slug.startswith("my-org-")
        # Should have made two execute calls: INSERT org + UPDATE owner
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_create_organization_custom_slug(self, store, mock_conn):
        """create_organization_async should use provided slug."""
        org = await store.create_organization_async(
            name="My Org", owner_id="u1", slug="custom-slug"
        )
        assert org.slug == "custom-slug"

    @pytest.mark.asyncio
    async def test_get_organization_by_id_found(self, store, mock_conn, sample_org_row):
        """get_organization_by_id_async should return an Organization."""
        mock_conn.fetchrow.return_value = sample_org_row
        org = await store.get_organization_by_id_async("org-456")
        assert org is not None
        assert org.id == "org-456"
        assert org.name == "Test Org"
        assert org.tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_get_organization_by_id_not_found(self, store, mock_conn):
        """get_organization_by_id_async should return None for missing org."""
        mock_conn.fetchrow.return_value = None
        org = await store.get_organization_by_id_async("nonexistent")
        assert org is None

    @pytest.mark.asyncio
    async def test_get_organization_by_slug(self, store, mock_conn, sample_org_row):
        """get_organization_by_slug_async should find by slug."""
        mock_conn.fetchrow.return_value = sample_org_row
        org = await store.get_organization_by_slug_async("test-org")
        assert org is not None
        assert org.slug == "test-org"

    @pytest.mark.asyncio
    async def test_get_organization_by_stripe_customer(self, store, mock_conn, sample_org_row):
        """get_organization_by_stripe_customer_async should find by Stripe ID."""
        sample_org_row["stripe_customer_id"] = "cus_abc"
        mock_conn.fetchrow.return_value = sample_org_row
        org = await store.get_organization_by_stripe_customer_async("cus_abc")
        assert org is not None

    @pytest.mark.asyncio
    async def test_get_organization_by_subscription(self, store, mock_conn, sample_org_row):
        """get_organization_by_subscription_async should find by subscription ID."""
        sample_org_row["stripe_subscription_id"] = "sub_xyz"
        mock_conn.fetchrow.return_value = sample_org_row
        org = await store.get_organization_by_subscription_async("sub_xyz")
        assert org is not None

    @pytest.mark.asyncio
    async def test_update_organization_success(self, store, mock_conn):
        """update_organization_async should update fields and return True."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.update_organization_async("org-456", name="Renamed Org")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_organization_no_fields(self, store, mock_conn):
        """update_organization_async with no fields should return False."""
        result = await store.update_organization_async("org-456")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_organization_settings_json_serialized(self, store, mock_conn):
        """update_organization_async should JSON-serialize dict settings."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.update_organization_async("org-456", settings={"theme": "dark"})
        assert result is True
        call_args = mock_conn.execute.call_args
        # The settings dict should have been serialized to JSON string
        assert json.dumps({"theme": "dark"}) in call_args[0]

    @pytest.mark.asyncio
    async def test_reset_org_usage(self, store, mock_conn):
        """reset_org_usage_async should reset debates_used_this_month."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.reset_org_usage_async("org-456")
        assert result is True


# =============================================================================
# Organization Membership Tests
# =============================================================================


class TestOrgMembership:
    """Tests for adding/removing users to/from organizations."""

    @pytest.mark.asyncio
    async def test_add_user_to_org(self, store, mock_conn):
        """add_user_to_org_async should update user's org_id and role."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.add_user_to_org_async("user-123", "org-456", "admin")
        assert result is True

    @pytest.mark.asyncio
    async def test_remove_user_from_org(self, store, mock_conn):
        """remove_user_from_org_async should set org_id to NULL and role to member."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.remove_user_from_org_async("user-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_org_members(self, store, mock_conn, sample_user_row):
        """get_org_members_async should return list of User objects."""
        mock_conn.fetch.return_value = [sample_user_row]
        members = await store.get_org_members_async("org-456")
        assert len(members) == 1
        assert members[0].id == "user-123"

    @pytest.mark.asyncio
    async def test_get_org_members_eager_org_not_found(self, store, mock_conn):
        """get_org_members_eager_async should return (None, []) for missing org."""
        mock_conn.fetchrow.return_value = None
        org, members = await store.get_org_members_eager_async("nonexistent")
        assert org is None
        assert members == []

    @pytest.mark.asyncio
    async def test_get_org_members_eager_with_members(
        self, store, mock_conn, sample_org_row, sample_user_row
    ):
        """get_org_members_eager_async should return (org, members) when org exists."""
        mock_conn.fetchrow.return_value = sample_org_row
        mock_conn.fetch.return_value = [sample_user_row]
        org, members = await store.get_org_members_eager_async("org-456")
        assert org is not None
        assert org.id == "org-456"
        assert len(members) == 1


# =============================================================================
# Usage Tracking Tests
# =============================================================================


class TestUsageTracking:
    """Tests for usage increment and tracking."""

    @pytest.mark.asyncio
    async def test_increment_usage(self, store, mock_conn):
        """increment_usage_async should return the new debate count."""
        mock_conn.fetchrow.return_value = {"debates_used_this_month": 10}
        count = await store.increment_usage_async("org-456", count=2)
        assert count == 10

    @pytest.mark.asyncio
    async def test_increment_usage_missing_org(self, store, mock_conn):
        """increment_usage_async should return 0 for a missing org."""
        mock_conn.fetchrow.return_value = None
        count = await store.increment_usage_async("nonexistent")
        assert count == 0

    @pytest.mark.asyncio
    async def test_record_usage_event(self, store, mock_conn):
        """record_usage_event_async should insert a usage_events row."""
        await store.record_usage_event_async(
            org_id="org-456",
            event_type="debate",
            count=1,
            metadata={"topic": "rate-limiting"},
        )
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "org-456" in call_args
        assert "debate" in call_args

    @pytest.mark.asyncio
    async def test_reset_monthly_usage(self, store, mock_conn):
        """reset_monthly_usage_async should return the number of orgs reset."""
        mock_conn.execute.return_value = "UPDATE 5"
        count = await store.reset_monthly_usage_async()
        assert count == 5

    @pytest.mark.asyncio
    async def test_get_usage_summary_org_not_found(self, store, mock_conn):
        """get_usage_summary_async should return empty dict for missing org."""
        mock_conn.fetchrow.return_value = None
        summary = await store.get_usage_summary_async("nonexistent")
        assert summary == {}

    @pytest.mark.asyncio
    async def test_get_usage_summary(self, store, mock_conn, sample_org_row):
        """get_usage_summary_async should aggregate events by type."""
        # First fetchrow call returns org data, second one also uses fetchrow for org lookup
        mock_conn.fetchrow.return_value = sample_org_row
        mock_conn.fetch.return_value = [
            {"event_type": "debate", "total": 15},
            {"event_type": "api_call", "total": 42},
        ]
        summary = await store.get_usage_summary_async("org-456")
        assert summary["org_id"] == "org-456"
        assert summary["debates_used_this_month"] == 5
        assert summary["events"]["debate"] == 15
        assert summary["events"]["api_call"] == 42


# =============================================================================
# OAuth Provider Tests
# =============================================================================


class TestOAuthProviders:
    """Tests for OAuth provider linking and lookup."""

    @pytest.mark.asyncio
    async def test_link_oauth_provider_success(self, store, mock_conn):
        """link_oauth_provider_async should return True on success."""
        result = await store.link_oauth_provider_async(
            user_id="user-123",
            provider="google",
            provider_user_id="goog-456",
            email="test@gmail.com",
        )
        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_oauth_provider_failure(self, store, mock_conn):
        """link_oauth_provider_async should return False on database error."""
        mock_conn.execute.side_effect = Exception("unique constraint violation")
        result = await store.link_oauth_provider_async(
            user_id="user-123",
            provider="google",
            provider_user_id="goog-456",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_oauth_provider(self, store, mock_conn):
        """unlink_oauth_provider_async should return True on successful delete."""
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.unlink_oauth_provider_async("user-123", "google")
        assert result is True

    @pytest.mark.asyncio
    async def test_unlink_oauth_provider_not_found(self, store, mock_conn):
        """unlink_oauth_provider_async should return False when no link found."""
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.unlink_oauth_provider_async("user-123", "github")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_by_oauth_found(self, store, mock_conn, sample_user_row):
        """get_user_by_oauth_async should find user through oauth link."""
        # First fetchrow returns the oauth row, second returns the user row
        mock_conn.fetchrow.side_effect = [
            {"user_id": "user-123"},
            sample_user_row,
        ]
        user = await store.get_user_by_oauth_async("google", "goog-456")
        assert user is not None
        assert user.id == "user-123"

    @pytest.mark.asyncio
    async def test_get_user_by_oauth_not_found(self, store, mock_conn):
        """get_user_by_oauth_async should return None when no link exists."""
        mock_conn.fetchrow.return_value = None
        user = await store.get_user_by_oauth_async("google", "unknown")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_oauth_providers_list(self, store, mock_conn):
        """get_user_oauth_providers_async should return list of provider dicts."""
        now = datetime.now(timezone.utc)
        mock_conn.fetch.return_value = [
            {
                "provider": "google",
                "provider_user_id": "goog-1",
                "email": "a@gmail.com",
                "linked_at": now,
            },
            {
                "provider": "github",
                "provider_user_id": "gh-2",
                "email": None,
                "linked_at": None,
            },
        ]
        providers = await store.get_user_oauth_providers_async("user-123")
        assert len(providers) == 2
        assert providers[0]["provider"] == "google"
        assert providers[0]["email"] == "a@gmail.com"
        assert providers[1]["linked_at"] is None


# =============================================================================
# Audit Log Tests
# =============================================================================


class TestAuditLog:
    """Tests for audit logging operations."""

    @pytest.mark.asyncio
    async def test_log_audit_event(self, store, mock_conn):
        """log_audit_event_async should insert and return the event ID."""
        mock_conn.fetchrow.return_value = {"id": 42}
        event_id = await store.log_audit_event_async(
            action="user.create",
            resource_type="user",
            resource_id="user-123",
            user_id="admin-1",
            org_id="org-456",
            new_value={"email": "new@example.com"},
            ip_address="127.0.0.1",
        )
        assert event_id == 42

    @pytest.mark.asyncio
    async def test_log_audit_event_no_return(self, store, mock_conn):
        """log_audit_event_async should return 0 if no row returned."""
        mock_conn.fetchrow.return_value = None
        event_id = await store.log_audit_event_async(
            action="user.delete",
            resource_type="user",
        )
        assert event_id == 0

    @pytest.mark.asyncio
    async def test_get_audit_log_with_filters(self, store, mock_conn):
        """get_audit_log_async should build query with filters."""
        now = datetime.now(timezone.utc)
        mock_conn.fetch.return_value = [
            {
                "id": 1,
                "timestamp": now,
                "user_id": "u1",
                "org_id": "o1",
                "action": "user.create",
                "resource_type": "user",
                "resource_id": "u1",
                "old_value": None,
                "new_value": '{"email": "x@y.com"}',
                "metadata": "{}",
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
            }
        ]
        results = await store.get_audit_log_async(
            org_id="o1", action="user.create", limit=10, offset=0
        )
        assert len(results) == 1
        assert results[0]["action"] == "user.create"
        assert results[0]["new_value"] == {"email": "x@y.com"}

    @pytest.mark.asyncio
    async def test_get_audit_log_count(self, store, mock_conn):
        """get_audit_log_count_async should return the count."""
        mock_conn.fetchrow.return_value = {0: 15}
        mock_conn.fetchrow.return_value = MagicMock(__getitem__=lambda self, k: 15)
        count = await store.get_audit_log_count_async(org_id="org-456")
        assert count == 15


# =============================================================================
# Invitation Tests
# =============================================================================


class TestInvitations:
    """Tests for organization invitation operations."""

    @pytest.mark.asyncio
    async def test_create_invitation_success(self, store, mock_conn):
        """create_invitation_async should return True on success."""
        invitation = OrganizationInvitation(
            id="inv-1",
            org_id="org-456",
            email="new@example.com",
            role="member",
            token="tok_xyz",
            invited_by="user-123",
            status="pending",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        result = await store.create_invitation_async(invitation)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_invitation_failure(self, store, mock_conn):
        """create_invitation_async should return False on database error."""
        mock_conn.execute.side_effect = Exception("duplicate key")
        invitation = OrganizationInvitation(
            id="inv-dup",
            org_id="org-456",
            email="dup@example.com",
            token="tok_dup",
        )
        result = await store.create_invitation_async(invitation)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_invitation_by_token(self, store, mock_conn, sample_invitation_row):
        """get_invitation_by_token_async should return an invitation."""
        mock_conn.fetchrow.return_value = sample_invitation_row
        inv = await store.get_invitation_by_token_async("tok_abc123")
        assert inv is not None
        assert inv.token == "tok_abc123"
        assert inv.status == "pending"

    @pytest.mark.asyncio
    async def test_get_invitation_by_email(self, store, mock_conn, sample_invitation_row):
        """get_invitation_by_email_async should find pending invitation."""
        mock_conn.fetchrow.return_value = sample_invitation_row
        inv = await store.get_invitation_by_email_async("org-456", "invitee@example.com")
        assert inv is not None
        assert inv.email == "invitee@example.com"

    @pytest.mark.asyncio
    async def test_update_invitation_status_with_accepted_at(self, store, mock_conn):
        """update_invitation_status_async should handle accepted_at parameter."""
        mock_conn.execute.return_value = "UPDATE 1"
        now = datetime.now(timezone.utc)
        result = await store.update_invitation_status_async("inv-1", "accepted", accepted_at=now)
        assert result is True
        call_args = mock_conn.execute.call_args[0]
        assert "accepted" in call_args
        assert now in call_args

    @pytest.mark.asyncio
    async def test_update_invitation_status_without_accepted_at(self, store, mock_conn):
        """update_invitation_status_async without accepted_at should use simpler query."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.update_invitation_status_async("inv-1", "revoked")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_invitation(self, store, mock_conn):
        """delete_invitation_async should return True on success."""
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.delete_invitation_async("inv-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_expired_invitations(self, store, mock_conn):
        """cleanup_expired_invitations_async should return count of updated rows."""
        mock_conn.execute.return_value = "UPDATE 3"
        count = await store.cleanup_expired_invitations_async()
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_invitations_for_org(self, store, mock_conn, sample_invitation_row):
        """get_invitations_for_org_async should return list of invitations."""
        mock_conn.fetch.return_value = [sample_invitation_row]
        invitations = await store.get_invitations_for_org_async("org-456")
        assert len(invitations) == 1
        assert invitations[0].org_id == "org-456"

    @pytest.mark.asyncio
    async def test_get_pending_invitations_by_email(self, store, mock_conn, sample_invitation_row):
        """get_pending_invitations_by_email_async should return pending invitations."""
        mock_conn.fetch.return_value = [sample_invitation_row]
        invitations = await store.get_pending_invitations_by_email_async("invitee@example.com")
        assert len(invitations) == 1
        assert invitations[0].status == "pending"


# =============================================================================
# Account Lockout Tests
# =============================================================================


class TestAccountLockout:
    """Tests for account lockout and failed login tracking."""

    @pytest.mark.asyncio
    async def test_is_account_locked_no_user(self, store, mock_conn):
        """is_account_locked_async should return (False, None, 0) for missing user."""
        mock_conn.fetchrow.return_value = None
        locked, until, attempts = await store.is_account_locked_async("unknown@x.com")
        assert locked is False
        assert until is None
        assert attempts == 0

    @pytest.mark.asyncio
    async def test_is_account_locked_not_locked(self, store, mock_conn):
        """is_account_locked_async should detect non-locked account."""
        mock_conn.fetchrow.return_value = {
            "failed_login_attempts": 2,
            "lockout_until": None,
        }
        locked, until, attempts = await store.is_account_locked_async("user@x.com")
        assert locked is False
        assert until is None
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_is_account_locked_active_lockout(self, store, mock_conn):
        """is_account_locked_async should detect active lockout."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_conn.fetchrow.return_value = {
            "failed_login_attempts": 5,
            "lockout_until": future,
        }
        locked, until, attempts = await store.is_account_locked_async("user@x.com")
        assert locked is True
        assert until == future
        assert attempts == 5

    @pytest.mark.asyncio
    async def test_is_account_locked_expired_lockout(self, store, mock_conn):
        """is_account_locked_async should detect expired lockout as not locked."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_conn.fetchrow.return_value = {
            "failed_login_attempts": 5,
            "lockout_until": past,
        }
        locked, until, attempts = await store.is_account_locked_async("user@x.com")
        assert locked is False
        assert until is None
        assert attempts == 5

    @pytest.mark.asyncio
    async def test_record_failed_login_no_lockout(self, store, mock_conn):
        """record_failed_login_async below threshold should not set lockout."""
        mock_conn.fetchrow.return_value = {"failed_login_attempts": 1}
        attempts, lockout_until = await store.record_failed_login_async("user@x.com")
        assert attempts == 1
        assert lockout_until is None

    @pytest.mark.asyncio
    async def test_record_failed_login_triggers_lockout(self, store, mock_conn):
        """record_failed_login_async at threshold should set lockout."""
        mock_conn.fetchrow.return_value = {"failed_login_attempts": 3}
        attempts, lockout_until = await store.record_failed_login_async("user@x.com")
        assert attempts == 3
        assert lockout_until is not None
        # Should have made an additional execute call to set lockout_until
        assert mock_conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_record_failed_login_missing_user(self, store, mock_conn):
        """record_failed_login_async for unknown email should return (0, None)."""
        mock_conn.fetchrow.return_value = None
        attempts, lockout_until = await store.record_failed_login_async("unknown@x.com")
        assert attempts == 0
        assert lockout_until is None

    @pytest.mark.asyncio
    async def test_reset_failed_login_attempts(self, store, mock_conn):
        """reset_failed_login_attempts_async should clear lockout data."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.reset_failed_login_attempts_async("user@x.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_lockout_info_no_user(self, store, mock_conn):
        """get_lockout_info_async should return {exists: False} for missing user."""
        mock_conn.fetchrow.return_value = None
        info = await store.get_lockout_info_async("unknown@x.com")
        assert info == {"exists": False}

    @pytest.mark.asyncio
    async def test_get_lockout_info_with_data(self, store, mock_conn):
        """get_lockout_info_async should return detailed lockout state."""
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        last_failed = datetime.now(timezone.utc) - timedelta(minutes=1)
        mock_conn.fetchrow.return_value = {
            "failed_login_attempts": 6,
            "lockout_until": future,
            "last_failed_login_at": last_failed,
        }
        info = await store.get_lockout_info_async("user@x.com")
        assert info["exists"] is True
        assert info["failed_attempts"] == 6
        assert info["is_locked"] is True
        assert info["lockout_until"] is not None
        assert info["last_failed_at"] is not None


# =============================================================================
# Admin Methods Tests
# =============================================================================


class TestAdminMethods:
    """Tests for admin listing and statistics methods."""

    @pytest.mark.asyncio
    async def test_list_all_organizations_no_filter(self, store, mock_conn, sample_org_row):
        """list_all_organizations_async without filter should list all."""
        count_row = MagicMock(__getitem__=lambda self, k: 1)
        mock_conn.fetchrow.return_value = count_row
        mock_conn.fetch.return_value = [sample_org_row]

        orgs, total = await store.list_all_organizations_async(limit=10, offset=0)
        assert total == 1
        assert len(orgs) == 1
        assert orgs[0].name == "Test Org"

    @pytest.mark.asyncio
    async def test_list_all_organizations_with_tier_filter(self, store, mock_conn, sample_org_row):
        """list_all_organizations_async with tier_filter should filter by tier."""
        count_row = MagicMock(__getitem__=lambda self, k: 1)
        mock_conn.fetchrow.return_value = count_row
        mock_conn.fetch.return_value = [sample_org_row]

        orgs, total = await store.list_all_organizations_async(
            limit=10, offset=0, tier_filter="free"
        )
        assert total == 1
        # Verify tier_filter was passed to the query
        fetchrow_call = mock_conn.fetchrow.call_args
        assert "free" in fetchrow_call[0]

    @pytest.mark.asyncio
    async def test_list_all_users(self, store, mock_conn, sample_user_row):
        """list_all_users_async should return paginated user list."""
        count_row = MagicMock(__getitem__=lambda self, k: 1)
        mock_conn.fetchrow.return_value = count_row
        mock_conn.fetch.return_value = [sample_user_row]

        users, total = await store.list_all_users_async(limit=10, offset=0)
        assert total == 1
        assert len(users) == 1
        assert users[0].email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_admin_stats(self, store, mock_conn):
        """get_admin_stats_async should return aggregated stats."""
        # Mock a sequence of fetchrow/fetch calls
        mock_conn.fetchrow.side_effect = [
            MagicMock(__getitem__=lambda s, k: 100),  # total_users
            MagicMock(__getitem__=lambda s, k: 80),  # active_users
            MagicMock(__getitem__=lambda s, k: 10),  # total_organizations
            MagicMock(__getitem__=lambda s, k: 500 if k == "total" else 500),  # total_debates
            MagicMock(__getitem__=lambda s, k: 25),  # users_active_24h
            MagicMock(__getitem__=lambda s, k: 5),  # new_users_7d
            MagicMock(__getitem__=lambda s, k: 2),  # new_orgs_7d
        ]
        mock_conn.fetch.return_value = [
            {"tier": "free", "count": 7},
            {"tier": "professional", "count": 3},
        ]

        stats = await store.get_admin_stats_async()
        assert stats["total_users"] == 100
        assert stats["active_users"] == 80
        assert stats["total_organizations"] == 10
        assert stats["tier_distribution"] == {"free": 7, "professional": 3}


# =============================================================================
# Row Conversion Tests
# =============================================================================


class TestRowConversion:
    """Tests for _row_to_user, _row_to_org, _row_to_invitation helpers."""

    def test_row_to_user(self, store, sample_user_row):
        """_row_to_user should convert a row dict to a User object."""
        user = store._row_to_user(sample_user_row)
        assert isinstance(user, User)
        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.mfa_enabled is False
        assert user.token_version == 1

    def test_row_to_org(self, store, sample_org_row):
        """_row_to_org should convert a row dict to an Organization object."""
        org = store._row_to_org(sample_org_row)
        assert isinstance(org, Organization)
        assert org.id == "org-456"
        assert org.tier == SubscriptionTier.FREE
        assert org.debates_used_this_month == 5

    def test_row_to_org_settings_as_string(self, store, sample_org_row):
        """_row_to_org should parse JSON string settings."""
        sample_org_row["settings"] = '{"feature_flags": ["beta"]}'
        org = store._row_to_org(sample_org_row)
        assert org.settings == {"feature_flags": ["beta"]}

    def test_row_to_invitation(self, store, sample_invitation_row):
        """_row_to_invitation should convert a row dict to an OrganizationInvitation."""
        inv = store._row_to_invitation(sample_invitation_row)
        assert isinstance(inv, OrganizationInvitation)
        assert inv.id == "inv-789"
        assert inv.status == "pending"
        assert inv.email == "invitee@example.com"

    def test_lockout_constants(self):
        """Verify lockout threshold and duration constants."""
        assert PostgresUserStore.LOCKOUT_THRESHOLD_1 == 3
        assert PostgresUserStore.LOCKOUT_THRESHOLD_2 == 6
        assert PostgresUserStore.LOCKOUT_THRESHOLD_3 == 10
        assert PostgresUserStore.LOCKOUT_DURATION_1 == timedelta(minutes=5)
        assert PostgresUserStore.LOCKOUT_DURATION_2 == timedelta(minutes=30)
        assert PostgresUserStore.LOCKOUT_DURATION_3 == timedelta(hours=24)


# =============================================================================
# Batch Update Tests
# =============================================================================


class TestBatchUpdate:
    """Tests for batch user update operations."""

    @pytest.mark.asyncio
    async def test_update_users_batch(self, store, mock_conn):
        """update_users_batch_async should update multiple users."""
        mock_conn.execute.return_value = "UPDATE 1"
        updates = [
            {"user_id": "u1", "name": "Alice Updated"},
            {"id": "u2", "role": "admin"},
        ]
        count = await store.update_users_batch_async(updates)
        assert count == 2

    @pytest.mark.asyncio
    async def test_update_users_batch_skip_empty(self, store, mock_conn):
        """update_users_batch_async should skip entries without fields."""
        mock_conn.execute.return_value = "UPDATE 1"
        updates = [
            {"user_id": "u1"},  # No fields to update after removing user_id
        ]
        count = await store.update_users_batch_async(updates)
        assert count == 0
