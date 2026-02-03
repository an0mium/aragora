"""Tests for PostgresUserStore - PostgreSQL backend for user/org persistence."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User
from aragora.storage.user_store.postgres_store import PostgresUserStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pool():
    """Create a mock asyncpg Pool with acquire context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def _make_user_row(**overrides):
    """Create a mock database row for a user."""
    now = datetime.now(timezone.utc)
    defaults = {
        "id": "user-1",
        "email": "test@example.com",
        "password_hash": "hashed_pw",
        "password_salt": "salt123",
        "name": "Test User",
        "org_id": "org-1",
        "role": "member",
        "is_active": True,
        "email_verified": True,
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
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    row.get = lambda key, default=None: defaults.get(key, default)
    return row


def _make_org_row(**overrides):
    """Create a mock database row for an organization."""
    now = datetime.now(timezone.utc)
    defaults = {
        "id": "org-1",
        "name": "Test Org",
        "slug": "test-org",
        "tier": "free",
        "owner_id": "user-1",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "debates_used_this_month": 5,
        "billing_cycle_start": now,
        "settings": "{}",
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    row.get = lambda key, default=None: defaults.get(key, default)
    return row


def _make_invitation_row(**overrides):
    """Create a mock database row for an invitation."""
    now = datetime.now(timezone.utc)
    defaults = {
        "id": "inv-1",
        "org_id": "org-1",
        "email": "invite@example.com",
        "role": "member",
        "token": "tok_abc123",
        "invited_by": "user-1",
        "status": "pending",
        "created_at": now,
        "expires_at": now + timedelta(days=7),
        "accepted_by": None,
        "accepted_at": None,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    row.get = lambda key, default=None: defaults.get(key, default)
    return row


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestPostgresUserStoreInit:
    """Test PostgresUserStore initialization."""

    def test_init(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        assert store._pool is pool
        assert store._initialized is False

    async def test_initialize_schema(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        await store.initialize()
        conn.execute.assert_called_once_with(store.INITIAL_SCHEMA)
        assert store._initialized is True

    async def test_initialize_idempotent(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        await store.initialize()
        await store.initialize()
        # Should only call execute once
        conn.execute.assert_called_once()

    def test_close_noop(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        store.close()  # Should not raise

    def test_schema_constants(self):
        assert PostgresUserStore.SCHEMA_NAME == "user_store"
        assert PostgresUserStore.SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# _row_to_user tests
# ---------------------------------------------------------------------------


class TestRowToUser:
    """Test row-to-User conversion."""

    def test_basic_conversion(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_user_row()
        user = store._row_to_user(row)
        assert isinstance(user, User)
        assert user.id == "user-1"
        assert user.email == "test@example.com"
        assert user.role == "member"
        assert user.is_active is True

    def test_null_name_becomes_empty(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_user_row(name=None)
        user = store._row_to_user(row)
        assert user.name == ""

    def test_null_role_defaults_to_member(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_user_row(role=None)
        user = store._row_to_user(row)
        assert user.role == "member"

    def test_null_token_version_defaults_to_1(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_user_row(token_version=None)
        user = store._row_to_user(row)
        assert user.token_version == 1

    def test_boolean_fields_coerced(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_user_row(is_active=1, email_verified=0, mfa_enabled=0)
        user = store._row_to_user(row)
        assert user.is_active is True
        assert user.email_verified is False
        assert user.mfa_enabled is False


# ---------------------------------------------------------------------------
# _row_to_org tests
# ---------------------------------------------------------------------------


class TestRowToOrg:
    """Test row-to-Organization conversion."""

    def test_basic_conversion(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row()
        org = store._row_to_org(row)
        assert isinstance(org, Organization)
        assert org.id == "org-1"
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        assert org.tier == SubscriptionTier.FREE

    def test_null_tier_defaults_to_free(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(tier=None)
        org = store._row_to_org(row)
        assert org.tier == SubscriptionTier.FREE

    def test_null_debates_defaults_to_zero(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(debates_used_this_month=None)
        org = store._row_to_org(row)
        assert org.debates_used_this_month == 0

    def test_json_string_settings_parsed(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(settings='{"key": "value"}')
        org = store._row_to_org(row)
        assert org.settings == {"key": "value"}

    def test_dict_settings_used_directly(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(settings={"key": "value"})
        org = store._row_to_org(row)
        assert org.settings == {"key": "value"}

    def test_null_settings_defaults_to_empty_dict(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(settings=None)
        org = store._row_to_org(row)
        assert org.settings == {}

    def test_enterprise_tier(self):
        pool, _ = _make_pool()
        store = PostgresUserStore(pool)
        row = _make_org_row(tier="enterprise")
        org = store._row_to_org(row)
        assert org.tier == SubscriptionTier.ENTERPRISE


# ---------------------------------------------------------------------------
# User CRUD tests
# ---------------------------------------------------------------------------


class TestUserCrud:
    """Test user CRUD operations."""

    async def test_create_user_async(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        user = await store.create_user_async(
            email="new@example.com",
            password_hash="hashed",
            password_salt="salt",
            name="New User",
            org_id="org-1",
            role="admin",
        )

        assert isinstance(user, User)
        assert user.email == "new@example.com"
        assert user.name == "New User"
        assert user.org_id == "org-1"
        assert user.role == "admin"
        assert user.is_active is True
        assert user.email_verified is False
        assert user.id  # UUID generated
        conn.execute.assert_called_once()

    async def test_get_user_by_id_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_user_row()

        user = await store.get_user_by_id_async("user-1")
        assert user is not None
        assert user.id == "user-1"
        conn.fetchrow.assert_called_once()

    async def test_get_user_by_id_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        user = await store.get_user_by_id_async("nonexistent")
        assert user is None

    async def test_get_user_by_email_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_user_row(email="found@example.com")

        user = await store.get_user_by_email_async("found@example.com")
        assert user is not None
        assert user.email == "found@example.com"

    async def test_get_user_by_email_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        user = await store.get_user_by_email_async("nope@example.com")
        assert user is None

    async def test_get_user_by_api_key_hashes_key(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_user_row()

        api_key = "ara_test_key_12345"
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()

        user = await store.get_user_by_api_key_async(api_key)
        assert user is not None
        # Verify the hash was passed as first param
        call_args = conn.fetchrow.call_args
        assert call_args[0][1] == expected_hash
        assert call_args[0][2] == api_key

    async def test_get_users_batch_empty(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        result = await store.get_users_batch_async([])
        assert result == {}
        conn.fetch.assert_not_called()

    async def test_get_users_batch_returns_dict(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetch.return_value = [
            _make_user_row(id="user-1"),
            _make_user_row(id="user-2"),
        ]

        result = await store.get_users_batch_async(["user-1", "user-2"])
        assert "user-1" in result
        assert "user-2" in result

    async def test_update_user_no_fields(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        result = await store.update_user_async("user-1")
        assert result is False
        conn.execute.assert_not_called()

    async def test_update_user_with_fields(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.update_user_async("user-1", name="Updated", role="admin")
        assert result is True
        conn.execute.assert_called_once()
        # Verify the SQL contains SET clauses
        sql = conn.execute.call_args[0][0]
        assert "SET" in sql
        assert "name" in sql
        assert "role" in sql

    async def test_update_user_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 0"

        result = await store.update_user_async("nonexistent", name="Updated")
        assert result is False

    async def test_delete_user_success(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "DELETE 1"

        result = await store.delete_user_async("user-1")
        assert result is True

    async def test_delete_user_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "DELETE 0"

        result = await store.delete_user_async("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# User preferences tests
# ---------------------------------------------------------------------------


class TestUserPreferences:
    """Test user preferences operations."""

    async def test_get_preferences_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {"theme": "dark"}
        conn.fetchrow.return_value = row

        prefs = await store.get_user_preferences_async("user-1")
        assert prefs == {"theme": "dark"}

    async def test_get_preferences_json_string(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: '{"theme": "dark"}'
        conn.fetchrow.return_value = row

        prefs = await store.get_user_preferences_async("user-1")
        assert prefs == {"theme": "dark"}

    async def test_get_preferences_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        prefs = await store.get_user_preferences_async("nonexistent")
        assert prefs is None

    async def test_set_preferences_success(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.set_user_preferences_async("user-1", {"theme": "dark"})
        assert result is True
        call_args = conn.execute.call_args[0]
        assert json.loads(call_args[1]) == {"theme": "dark"}

    async def test_set_preferences_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 0"

        result = await store.set_user_preferences_async("nonexistent", {})
        assert result is False


# ---------------------------------------------------------------------------
# Token version tests
# ---------------------------------------------------------------------------


class TestTokenVersion:
    """Test token version management."""

    async def test_increment_returns_new_version(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 5
        conn.fetchrow.return_value = row

        version = await store.increment_token_version_async("user-1")
        assert version == 5

    async def test_increment_not_found_returns_1(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        version = await store.increment_token_version_async("nonexistent")
        assert version == 1


# ---------------------------------------------------------------------------
# Organization CRUD tests
# ---------------------------------------------------------------------------


class TestOrganizationCrud:
    """Test organization CRUD operations."""

    async def test_create_organization(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        org = await store.create_organization_async(
            name="New Org",
            owner_id="user-1",
            tier=SubscriptionTier.PROFESSIONAL,
        )

        assert isinstance(org, Organization)
        assert org.name == "New Org"
        assert org.owner_id == "user-1"
        assert org.tier == SubscriptionTier.PROFESSIONAL
        assert org.id  # UUID generated
        # Should call execute twice: INSERT org + UPDATE user
        assert conn.execute.call_count == 2

    async def test_create_organization_auto_slug(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        org = await store.create_organization_async(name="My Test Org", owner_id="user-1")

        assert org.slug.startswith("my-test-org-")

    async def test_create_organization_custom_slug(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        org = await store.create_organization_async(
            name="Org", owner_id="user-1", slug="custom-slug"
        )

        assert org.slug == "custom-slug"

    async def test_get_organization_by_id_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_org_row()

        org = await store.get_organization_by_id_async("org-1")
        assert org is not None
        assert org.id == "org-1"

    async def test_get_organization_by_id_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        org = await store.get_organization_by_id_async("nonexistent")
        assert org is None

    async def test_get_organization_by_slug(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_org_row(slug="my-org")

        org = await store.get_organization_by_slug_async("my-org")
        assert org is not None
        assert org.slug == "my-org"

    async def test_get_organization_by_stripe_customer(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_org_row(stripe_customer_id="cus_123")

        org = await store.get_organization_by_stripe_customer_async("cus_123")
        assert org is not None

    async def test_get_organization_by_subscription(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_org_row(stripe_subscription_id="sub_123")

        org = await store.get_organization_by_subscription_async("sub_123")
        assert org is not None

    async def test_update_organization_no_fields(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        result = await store.update_organization_async("org-1")
        assert result is False

    async def test_update_organization_with_fields(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.update_organization_async("org-1", name="Updated")
        assert result is True

    async def test_update_organization_settings_serialized(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.update_organization_async("org-1", settings={"feature": True})
        assert result is True
        # Verify settings was JSON-serialized
        call_args = conn.execute.call_args[0]
        # The settings value should be a JSON string
        assert json.loads(call_args[1]) == {"feature": True}

    async def test_reset_org_usage(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.reset_org_usage_async("org-1")
        assert result is True


# ---------------------------------------------------------------------------
# Organization membership tests
# ---------------------------------------------------------------------------


class TestOrgMembership:
    """Test organization membership operations."""

    async def test_add_user_to_org(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.add_user_to_org_async("user-1", "org-1", "admin")
        assert result is True

    async def test_remove_user_from_org(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.remove_user_from_org_async("user-1")
        assert result is True
        # Verify role reset to member and org_id set to NULL
        sql = conn.execute.call_args[0][0]
        assert "NULL" in sql
        assert "member" in sql

    async def test_get_org_members(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetch.return_value = [
            _make_user_row(id="user-1"),
            _make_user_row(id="user-2"),
        ]

        members = await store.get_org_members_async("org-1")
        assert len(members) == 2
        assert all(isinstance(m, User) for m in members)

    async def test_get_org_members_eager_org_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_org_row()
        conn.fetch.return_value = [_make_user_row()]

        org, members = await store.get_org_members_eager_async("org-1")
        assert org is not None
        assert len(members) == 1

    async def test_get_org_members_eager_org_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        org, members = await store.get_org_members_eager_async("nonexistent")
        assert org is None
        assert members == []


# ---------------------------------------------------------------------------
# Usage tracking tests
# ---------------------------------------------------------------------------


class TestUsageTracking:
    """Test usage tracking operations."""

    async def test_increment_usage(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 10
        conn.fetchrow.return_value = row

        count = await store.increment_usage_async("org-1", count=3)
        assert count == 10

    async def test_increment_usage_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        count = await store.increment_usage_async("nonexistent")
        assert count == 0

    async def test_record_usage_event(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        await store.record_usage_event_async("org-1", "debate", count=1, metadata={"topic": "AI"})
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        assert call_args[1] == "org-1"
        assert call_args[2] == "debate"
        assert call_args[3] == 1

    async def test_record_usage_event_no_metadata(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        await store.record_usage_event_async("org-1", "debate")
        call_args = conn.execute.call_args[0]
        # Metadata should be serialized as "{}"
        assert json.loads(call_args[4]) == {}

    async def test_reset_monthly_usage(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 5"

        count = await store.reset_monthly_usage_async()
        assert count == 5

    async def test_get_usage_summary_org_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        summary = await store.get_usage_summary_async("nonexistent")
        assert summary == {}

    async def test_get_usage_summary_with_data(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        # First fetchrow is for get_organization_by_id
        conn.fetchrow.return_value = _make_org_row(debates_used_this_month=42)
        # fetch is for usage events
        event_row1 = MagicMock()
        event_row1.__getitem__ = lambda self, key: {"event_type": "debate", "total": 42}[key]
        event_row2 = MagicMock()
        event_row2.__getitem__ = lambda self, key: {"event_type": "api_call", "total": 100}[key]
        conn.fetch.return_value = [event_row1, event_row2]

        summary = await store.get_usage_summary_async("org-1")
        assert summary["org_id"] == "org-1"
        assert summary["debates_used_this_month"] == 42
        assert summary["events"]["debate"] == 42
        assert summary["events"]["api_call"] == 100


# ---------------------------------------------------------------------------
# OAuth provider tests
# ---------------------------------------------------------------------------


class TestOAuthProvider:
    """Test OAuth provider linking operations."""

    async def test_link_oauth_provider_success(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        result = await store.link_oauth_provider_async(
            "user-1", "google", "google-id-123", "user@gmail.com"
        )
        assert result is True
        conn.execute.assert_called_once()

    async def test_link_oauth_provider_failure(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.side_effect = Exception("DB error")

        result = await store.link_oauth_provider_async("user-1", "google", "google-id-123")
        assert result is False

    async def test_unlink_oauth_provider_success(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "DELETE 1"

        result = await store.unlink_oauth_provider_async("user-1", "google")
        assert result is True

    async def test_unlink_oauth_provider_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "DELETE 0"

        result = await store.unlink_oauth_provider_async("user-1", "google")
        assert result is False

    async def test_get_user_by_oauth_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        # First fetchrow returns oauth row with user_id
        oauth_row = MagicMock()
        oauth_row.__getitem__ = lambda self, key: "user-1"
        # Second fetchrow returns user row
        conn.fetchrow.side_effect = [oauth_row, _make_user_row()]

        user = await store.get_user_by_oauth_async("google", "google-id-123")
        assert user is not None
        assert user.id == "user-1"

    async def test_get_user_by_oauth_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        user = await store.get_user_by_oauth_async("google", "nonexistent")
        assert user is None

    async def test_get_user_oauth_providers(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        now = datetime.now(timezone.utc)
        provider_row = MagicMock()
        provider_row.__getitem__ = lambda self, key: {
            "provider": "google",
            "provider_user_id": "gid-123",
            "email": "user@gmail.com",
            "linked_at": now,
        }[key]
        conn.fetch.return_value = [provider_row]

        providers = await store.get_user_oauth_providers_async("user-1")
        assert len(providers) == 1
        assert providers[0]["provider"] == "google"
        assert providers[0]["email"] == "user@gmail.com"


# ---------------------------------------------------------------------------
# Audit logging tests
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """Test audit logging operations."""

    async def test_log_audit_event(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 42
        conn.fetchrow.return_value = row

        event_id = await store.log_audit_event_async(
            action="user.login",
            resource_type="user",
            resource_id="user-1",
            user_id="user-1",
            ip_address="127.0.0.1",
        )
        assert event_id == 42

    async def test_log_audit_event_with_values(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 1
        conn.fetchrow.return_value = row

        await store.log_audit_event_async(
            action="user.update",
            resource_type="user",
            old_value={"name": "Old"},
            new_value={"name": "New"},
            metadata={"reason": "test"},
        )
        call_args = conn.fetchrow.call_args[0]
        # old_value and new_value should be JSON
        assert json.loads(call_args[7]) == {"name": "Old"}
        assert json.loads(call_args[8]) == {"name": "New"}

    async def test_log_audit_event_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        event_id = await store.log_audit_event_async(action="test", resource_type="test")
        assert event_id == 0

    async def test_get_audit_log_basic(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "id": 1,
            "timestamp": now,
            "user_id": "user-1",
            "org_id": "org-1",
            "action": "user.login",
            "resource_type": "user",
            "resource_id": "user-1",
            "old_value": None,
            "new_value": None,
            "metadata": "{}",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
        }[key]
        conn.fetch.return_value = [row]

        entries = await store.get_audit_log_async(org_id="org-1")
        assert len(entries) == 1
        assert entries[0]["action"] == "user.login"

    async def test_get_audit_log_with_filters(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetch.return_value = []

        await store.get_audit_log_async(
            org_id="org-1",
            user_id="user-1",
            action="user.login",
            resource_type="user",
            since=datetime.now(timezone.utc),
            until=datetime.now(timezone.utc),
            limit=10,
            offset=5,
        )
        # Verify the query was constructed with all filters
        sql = conn.fetch.call_args[0][0]
        assert "org_id" in sql
        assert "user_id" in sql
        assert "action" in sql
        assert "resource_type" in sql
        assert "timestamp >=" in sql
        assert "timestamp <=" in sql

    async def test_get_audit_log_count(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, idx: 42
        conn.fetchrow.return_value = row

        count = await store.get_audit_log_count_async(org_id="org-1")
        assert count == 42


# ---------------------------------------------------------------------------
# Invitation tests
# ---------------------------------------------------------------------------


class TestInvitations:
    """Test organization invitation operations."""

    async def test_create_invitation_success(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        invitation = OrganizationInvitation(
            id="inv-1",
            org_id="org-1",
            email="invite@example.com",
            token="tok_abc",
            invited_by="user-1",
        )

        result = await store.create_invitation_async(invitation)
        assert result is True
        conn.execute.assert_called_once()

    async def test_create_invitation_failure(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.side_effect = Exception("Duplicate token")

        invitation = OrganizationInvitation(
            id="inv-1",
            org_id="org-1",
            email="invite@example.com",
            token="tok_abc",
        )

        result = await store.create_invitation_async(invitation)
        assert result is False

    async def test_get_invitation_by_id(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_invitation_row()

        inv = await store.get_invitation_by_id_async("inv-1")
        assert inv is not None
        assert inv.id == "inv-1"
        assert isinstance(inv, OrganizationInvitation)

    async def test_get_invitation_by_token(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_invitation_row(token="my-token")

        inv = await store.get_invitation_by_token_async("my-token")
        assert inv is not None
        assert inv.token == "my-token"

    async def test_get_invitation_by_email(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = _make_invitation_row()

        inv = await store.get_invitation_by_email_async("org-1", "invite@example.com")
        assert inv is not None

    async def test_get_invitations_for_org(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetch.return_value = [_make_invitation_row(), _make_invitation_row(id="inv-2")]

        invitations = await store.get_invitations_for_org_async("org-1")
        assert len(invitations) == 2

    async def test_update_invitation_status(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.update_invitation_status_async("inv-1", "accepted")
        assert result is True

    async def test_update_invitation_status_with_accepted_at(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"
        now = datetime.now(timezone.utc)

        result = await store.update_invitation_status_async("inv-1", "accepted", accepted_at=now)
        assert result is True
        sql = conn.execute.call_args[0][0]
        assert "accepted_at" in sql

    async def test_delete_invitation(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "DELETE 1"

        result = await store.delete_invitation_async("inv-1")
        assert result is True

    async def test_cleanup_expired_invitations(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 3"

        count = await store.cleanup_expired_invitations_async()
        assert count == 3


# ---------------------------------------------------------------------------
# Account lockout tests
# ---------------------------------------------------------------------------


class TestAccountLockout:
    """Test account lockout operations."""

    async def test_not_locked_when_user_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        locked, until, attempts = await store.is_account_locked_async("nope@example.com")
        assert locked is False
        assert until is None
        assert attempts == 0

    async def test_not_locked_when_no_lockout(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "failed_login_attempts": 2,
            "lockout_until": None,
        }[key]
        conn.fetchrow.return_value = row

        locked, until, attempts = await store.is_account_locked_async("test@example.com")
        assert locked is False
        assert until is None
        assert attempts == 2

    async def test_locked_when_lockout_in_future(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "failed_login_attempts": 5,
            "lockout_until": future,
        }[key]
        conn.fetchrow.return_value = row

        locked, until, attempts = await store.is_account_locked_async("test@example.com")
        assert locked is True
        assert until == future
        assert attempts == 5

    async def test_not_locked_when_lockout_expired(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "failed_login_attempts": 5,
            "lockout_until": past,
        }[key]
        conn.fetchrow.return_value = row

        locked, until, attempts = await store.is_account_locked_async("test@example.com")
        assert locked is False
        assert until is None
        assert attempts == 5

    async def test_record_failed_login_no_lockout(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 2  # 2 attempts, below threshold
        conn.fetchrow.return_value = row

        attempts, lockout_until = await store.record_failed_login_async("test@example.com")
        assert attempts == 2
        assert lockout_until is None
        # Should only call fetchrow, not execute (no lockout set)
        assert conn.execute.call_count == 0

    async def test_record_failed_login_threshold_1(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 3  # LOCKOUT_THRESHOLD_1
        conn.fetchrow.return_value = row

        attempts, lockout_until = await store.record_failed_login_async("test@example.com")
        assert attempts == 3
        assert lockout_until is not None
        # Should have set lockout in DB
        conn.execute.assert_called_once()

    async def test_record_failed_login_threshold_2(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 6  # LOCKOUT_THRESHOLD_2
        conn.fetchrow.return_value = row

        attempts, lockout_until = await store.record_failed_login_async("test@example.com")
        assert attempts == 6
        assert lockout_until is not None

    async def test_record_failed_login_threshold_3(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        row = MagicMock()
        row.__getitem__ = lambda self, key: 10  # LOCKOUT_THRESHOLD_3
        conn.fetchrow.return_value = row

        attempts, lockout_until = await store.record_failed_login_async("test@example.com")
        assert attempts == 10
        assert lockout_until is not None

    async def test_record_failed_login_user_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        attempts, lockout_until = await store.record_failed_login_async("nope@example.com")
        assert attempts == 0
        assert lockout_until is None

    async def test_reset_failed_login_attempts(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        result = await store.reset_failed_login_attempts_async("test@example.com")
        assert result is True
        sql = conn.execute.call_args[0][0]
        assert "failed_login_attempts = 0" in sql
        assert "lockout_until = NULL" in sql

    async def test_get_lockout_info_user_not_found(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.fetchrow.return_value = None

        info = await store.get_lockout_info_async("nope@example.com")
        assert info == {"exists": False}

    async def test_get_lockout_info_locked(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "failed_login_attempts": 5,
            "lockout_until": future,
            "last_failed_login_at": datetime.now(timezone.utc),
        }[key]
        conn.fetchrow.return_value = row

        info = await store.get_lockout_info_async("test@example.com")
        assert info["exists"] is True
        assert info["is_locked"] is True
        assert info["failed_attempts"] == 5

    def test_lockout_constants(self):
        """Verify lockout policy constants are reasonable."""
        assert PostgresUserStore.LOCKOUT_THRESHOLD_1 == 3
        assert PostgresUserStore.LOCKOUT_THRESHOLD_2 == 6
        assert PostgresUserStore.LOCKOUT_THRESHOLD_3 == 10
        assert PostgresUserStore.LOCKOUT_DURATION_1 == timedelta(minutes=5)
        assert PostgresUserStore.LOCKOUT_DURATION_2 == timedelta(minutes=30)
        assert PostgresUserStore.LOCKOUT_DURATION_3 == timedelta(hours=24)


# ---------------------------------------------------------------------------
# Admin method tests
# ---------------------------------------------------------------------------


class TestAdminMethods:
    """Test admin operations."""

    async def test_list_all_organizations_no_filter(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        total_row = MagicMock()
        total_row.__getitem__ = lambda self, idx: 2
        conn.fetchrow.return_value = total_row
        conn.fetch.return_value = [_make_org_row(id="org-1"), _make_org_row(id="org-2")]

        orgs, total = await store.list_all_organizations_async(limit=50, offset=0)
        assert total == 2
        assert len(orgs) == 2

    async def test_list_all_organizations_with_tier_filter(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        total_row = MagicMock()
        total_row.__getitem__ = lambda self, idx: 1
        conn.fetchrow.return_value = total_row
        conn.fetch.return_value = [_make_org_row(tier="enterprise")]

        orgs, total = await store.list_all_organizations_async(tier_filter="enterprise")
        assert total == 1
        assert len(orgs) == 1

    async def test_list_all_users_no_filters(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        total_row = MagicMock()
        total_row.__getitem__ = lambda self, idx: 3
        conn.fetchrow.return_value = total_row
        conn.fetch.return_value = [_make_user_row()]

        users, total = await store.list_all_users_async()
        assert total == 3
        assert len(users) == 1

    async def test_list_all_users_with_filters(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        total_row = MagicMock()
        total_row.__getitem__ = lambda self, idx: 1
        conn.fetchrow.return_value = total_row
        conn.fetch.return_value = [_make_user_row()]

        users, total = await store.list_all_users_async(
            org_id_filter="org-1",
            role_filter="admin",
            active_only=True,
        )
        # Verify filters were applied
        count_sql = conn.fetchrow.call_args[0][0]
        assert "org_id" in count_sql
        assert "role" in count_sql
        assert "is_active" in count_sql

    async def test_get_admin_stats(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        # Mock multiple fetchrow calls
        counts = [10, 8, 3, 100, 5, 2, 1]
        idx = [0]

        def make_count_row(count):
            row = MagicMock()
            row.__getitem__ = lambda self, key: count
            row.get = lambda key, default=None: count
            return row

        def fetchrow_side_effect(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            if i < len(counts):
                return make_count_row(counts[i])
            return make_count_row(0)

        conn.fetchrow.side_effect = fetchrow_side_effect

        # Mock the tier distribution fetch
        tier_row = MagicMock()
        tier_row.__getitem__ = lambda self, key: {"tier": "free", "count": 2}[key]
        conn.fetch.return_value = [tier_row]

        stats = await store.get_admin_stats_async()
        assert isinstance(stats, dict)
        assert "total_users" in stats
        assert "active_users" in stats
        assert "total_organizations" in stats
        assert "tier_distribution" in stats


# ---------------------------------------------------------------------------
# Batch update tests
# ---------------------------------------------------------------------------


class TestBatchUpdates:
    """Test batch update operations."""

    async def test_update_users_batch(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        updates = [
            {"user_id": "user-1", "name": "Updated 1"},
            {"user_id": "user-2", "name": "Updated 2"},
        ]

        count = await store.update_users_batch_async(updates)
        assert count == 2

    async def test_update_users_batch_with_id_key(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)
        conn.execute.return_value = "UPDATE 1"

        updates = [{"id": "user-1", "name": "Updated"}]

        count = await store.update_users_batch_async(updates)
        assert count == 1

    async def test_update_users_batch_skips_empty(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        updates = [{"user_id": "user-1"}]  # No actual fields to update

        count = await store.update_users_batch_async(updates)
        assert count == 0

    async def test_update_users_batch_skips_no_id(self):
        pool, conn = _make_pool()
        store = PostgresUserStore(pool)

        updates = [{"name": "No ID"}]  # No user_id or id

        count = await store.update_users_batch_async(updates)
        assert count == 0
