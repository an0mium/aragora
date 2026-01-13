"""
Tests for the SQLite-backed OrganizationStore.

Covers:
- Organization CRUD operations
- Slug generation and uniqueness
- Invitation CRUD operations
- Token-based lookups
- Expiration and cleanup
- Member management
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User
from aragora.exceptions import ConfigurationError
from aragora.storage.organization_store import OrganizationStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_orgs.db"


@pytest.fixture
def db_with_schema(temp_db):
    """Create database with required schema."""
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            owner_id TEXT,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            debates_used_this_month INTEGER DEFAULT 0,
            billing_cycle_start TEXT,
            settings TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_invitations (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            token TEXT UNIQUE NOT NULL,
            invited_by TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            expires_at TEXT,
            accepted_by TEXT,
            accepted_at TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT,
            org_id TEXT,
            role TEXT DEFAULT 'member'
        )
    """
    )
    conn.commit()
    conn.close()
    yield temp_db


@pytest.fixture
def store(db_with_schema):
    """Create an OrganizationStore instance."""
    return OrganizationStore(db_with_schema)


@pytest.fixture
def mock_update_user():
    """Mock update_user callback."""
    return Mock(return_value=True)


@pytest.fixture
def mock_row_to_user():
    """Mock row_to_user callback."""

    def converter(row):
        return User(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            org_id=row["org_id"],
            role=row["role"],
        )

    return converter


@pytest.fixture
def store_with_callbacks(db_with_schema, mock_update_user, mock_row_to_user):
    """Create an OrganizationStore with callbacks for user operations."""
    return OrganizationStore(
        db_with_schema,
        update_user=mock_update_user,
        row_to_user=mock_row_to_user,
    )


# =============================================================================
# Organization CRUD Tests
# =============================================================================


class TestOrganizationCRUD:
    """Test basic organization CRUD operations."""

    def test_create_organization(self, store):
        """Test creating an organization."""
        org = store.create_organization(
            name="Test Org",
            owner_id="user_123",
            tier=SubscriptionTier.FREE,
        )

        assert org is not None
        assert org.name == "Test Org"
        assert org.owner_id == "user_123"
        assert org.tier == SubscriptionTier.FREE
        assert org.id is not None
        assert org.slug == "test-org"

    def test_create_organization_custom_slug(self, store):
        """Test creating organization with custom slug."""
        org = store.create_organization(
            name="Test Org",
            owner_id="user_123",
            slug="custom-slug",
        )

        assert org.slug == "custom-slug"

    def test_create_organization_slug_uniqueness(self, store):
        """Test that duplicate slugs are made unique."""
        org1 = store.create_organization(
            name="Test Org",
            owner_id="user_1",
        )
        org2 = store.create_organization(
            name="Test Org",
            owner_id="user_2",
        )

        assert org1.slug == "test-org"
        assert org2.slug != org1.slug
        assert org2.slug.startswith("test-org-")

    def test_get_organization_by_id(self, store):
        """Test retrieving organization by ID."""
        org = store.create_organization(name="Get By ID Org", owner_id="user_1")

        retrieved = store.get_organization_by_id(org.id)

        assert retrieved is not None
        assert retrieved.id == org.id
        assert retrieved.name == "Get By ID Org"

    def test_get_organization_by_id_not_found(self, store):
        """Test retrieving non-existent organization by ID."""
        result = store.get_organization_by_id("nonexistent")
        assert result is None

    def test_get_organization_by_slug(self, store):
        """Test retrieving organization by slug."""
        org = store.create_organization(name="Slug Org", owner_id="user_1")

        retrieved = store.get_organization_by_slug(org.slug)

        assert retrieved is not None
        assert retrieved.slug == org.slug

    def test_get_organization_by_slug_not_found(self, store):
        """Test retrieving non-existent organization by slug."""
        result = store.get_organization_by_slug("nonexistent")
        assert result is None

    def test_get_organization_by_stripe_customer(self, store):
        """Test retrieving organization by Stripe customer ID."""
        org = store.create_organization(name="Stripe Org", owner_id="user_1")
        store.update_organization(org.id, stripe_customer_id="cus_123")

        retrieved = store.get_organization_by_stripe_customer("cus_123")

        assert retrieved is not None
        assert retrieved.id == org.id

    def test_get_organization_by_subscription(self, store):
        """Test retrieving organization by Stripe subscription ID."""
        org = store.create_organization(name="Sub Org", owner_id="user_1")
        store.update_organization(org.id, stripe_subscription_id="sub_123")

        retrieved = store.get_organization_by_subscription("sub_123")

        assert retrieved is not None
        assert retrieved.id == org.id


class TestOrganizationUpdate:
    """Test organization update operations."""

    def test_update_organization_name(self, store):
        """Test updating organization name."""
        org = store.create_organization(name="Original Name", owner_id="user_1")

        result = store.update_organization(org.id, name="Updated Name")

        assert result is True
        updated = store.get_organization_by_id(org.id)
        assert updated.name == "Updated Name"

    def test_update_organization_tier(self, store):
        """Test updating organization tier."""
        org = store.create_organization(name="Tier Org", owner_id="user_1")

        result = store.update_organization(org.id, tier=SubscriptionTier.PROFESSIONAL)

        assert result is True
        updated = store.get_organization_by_id(org.id)
        assert updated.tier == SubscriptionTier.PROFESSIONAL

    def test_update_organization_settings(self, store):
        """Test updating organization settings."""
        org = store.create_organization(name="Settings Org", owner_id="user_1")
        new_settings = {"feature_flags": {"beta": True}, "theme": "dark"}

        result = store.update_organization(org.id, settings=new_settings)

        assert result is True
        updated = store.get_organization_by_id(org.id)
        assert updated.settings == new_settings

    def test_update_organization_multiple_fields(self, store):
        """Test updating multiple organization fields at once."""
        org = store.create_organization(name="Multi Update Org", owner_id="user_1")

        result = store.update_organization(
            org.id,
            name="New Name",
            stripe_customer_id="cus_new",
            debates_used_this_month=10,
        )

        assert result is True
        updated = store.get_organization_by_id(org.id)
        assert updated.name == "New Name"
        assert updated.stripe_customer_id == "cus_new"
        assert updated.debates_used_this_month == 10

    def test_update_organization_no_fields(self, store):
        """Test updating with no fields returns False."""
        org = store.create_organization(name="No Update Org", owner_id="user_1")

        result = store.update_organization(org.id)

        assert result is False

    def test_update_organization_unknown_field(self, store):
        """Test that unknown fields are ignored."""
        org = store.create_organization(name="Unknown Field Org", owner_id="user_1")

        # This should not raise an error, just ignore the field
        result = store.update_organization(org.id, unknown_field="value")

        assert result is False  # No valid fields to update

    def test_reset_org_usage(self, store):
        """Test resetting organization usage."""
        org = store.create_organization(name="Usage Org", owner_id="user_1")
        store.update_organization(org.id, debates_used_this_month=50)

        result = store.reset_org_usage(org.id)

        assert result is True
        updated = store.get_organization_by_id(org.id)
        assert updated.debates_used_this_month == 0


class TestOrganizationMembers:
    """Test member management operations."""

    def test_add_user_to_org(self, store_with_callbacks, mock_update_user):
        """Test adding user to organization."""
        org = store_with_callbacks.create_organization(name="Member Org", owner_id="user_1")

        result = store_with_callbacks.add_user_to_org("user_2", org.id, role="member")

        assert result is True
        mock_update_user.assert_called_with("user_2", org_id=org.id, role="member")

    def test_add_user_to_org_no_callback_raises(self, store):
        """Test that add_user_to_org raises without callback."""
        org = store.create_organization(name="No Callback Org", owner_id="user_1")

        with pytest.raises(ConfigurationError):
            store.add_user_to_org("user_2", org.id)

    def test_remove_user_from_org(self, store_with_callbacks, mock_update_user):
        """Test removing user from organization."""
        result = store_with_callbacks.remove_user_from_org("user_2")

        assert result is True
        mock_update_user.assert_called_with("user_2", org_id=None, role="member")

    def test_remove_user_from_org_no_callback_raises(self, store):
        """Test that remove_user_from_org raises without callback."""
        with pytest.raises(ConfigurationError):
            store.remove_user_from_org("user_2")

    def test_get_org_members_no_callback_raises(self, store):
        """Test that get_org_members raises without callback."""
        org = store.create_organization(name="Members Org", owner_id="user_1")

        with pytest.raises(ConfigurationError):
            store.get_org_members(org.id)


# =============================================================================
# Invitation Tests
# =============================================================================


class TestInvitationCRUD:
    """Test invitation CRUD operations."""

    def test_create_invitation(self, store):
        """Test creating an invitation."""
        org = store.create_organization(name="Invite Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="invite@example.com",
            role="member",
            invited_by="user_1",
        )

        result = store.create_invitation(invitation)

        assert result is True

    def test_get_invitation_by_id(self, store):
        """Test retrieving invitation by ID."""
        org = store.create_organization(name="Invite ID Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="byid@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        retrieved = store.get_invitation_by_id(invitation.id)

        assert retrieved is not None
        assert retrieved.id == invitation.id
        assert retrieved.email == "byid@example.com"

    def test_get_invitation_by_id_not_found(self, store):
        """Test retrieving non-existent invitation by ID."""
        result = store.get_invitation_by_id("nonexistent")
        assert result is None

    def test_get_invitation_by_token(self, store):
        """Test retrieving invitation by token."""
        org = store.create_organization(name="Token Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="token@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        retrieved = store.get_invitation_by_token(invitation.token)

        assert retrieved is not None
        assert retrieved.token == invitation.token

    def test_get_invitation_by_token_not_found(self, store):
        """Test retrieving non-existent invitation by token."""
        result = store.get_invitation_by_token("nonexistent_token")
        assert result is None

    def test_get_invitation_by_email(self, store):
        """Test retrieving invitation by email and org."""
        org = store.create_organization(name="Email Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="email@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        retrieved = store.get_invitation_by_email("email@example.com", org.id)

        assert retrieved is not None
        assert retrieved.email == "email@example.com"
        assert retrieved.org_id == org.id

    def test_get_invitations_for_org(self, store):
        """Test retrieving all invitations for an organization."""
        org = store.create_organization(name="Multi Invite Org", owner_id="user_1")

        for i in range(3):
            invitation = OrganizationInvitation(
                org_id=org.id,
                email=f"user{i}@example.com",
                role="member",
                invited_by="user_1",
            )
            store.create_invitation(invitation)

        invitations = store.get_invitations_for_org(org.id)

        assert len(invitations) == 3

    def test_get_pending_invitations_by_email(self, store):
        """Test retrieving pending invitations by email."""
        org1 = store.create_organization(name="Pending Org 1", owner_id="user_1")
        org2 = store.create_organization(name="Pending Org 2", owner_id="user_2")

        for org in [org1, org2]:
            invitation = OrganizationInvitation(
                org_id=org.id,
                email="pending@example.com",
                role="member",
                invited_by="user_1",
            )
            store.create_invitation(invitation)

        invitations = store.get_pending_invitations_by_email("pending@example.com")

        assert len(invitations) == 2


class TestInvitationStatus:
    """Test invitation status operations."""

    def test_update_invitation_status_accepted(self, store):
        """Test updating invitation status to accepted."""
        org = store.create_organization(name="Accept Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="accept@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        result = store.update_invitation_status(
            invitation.id,
            status="accepted",
            accepted_by="user_2",
            accepted_at=datetime.utcnow(),
        )

        assert result is True
        updated = store.get_invitation_by_id(invitation.id)
        assert updated.status == "accepted"
        assert updated.accepted_by == "user_2"
        assert updated.accepted_at is not None

    def test_update_invitation_status_cancelled(self, store):
        """Test updating invitation status to cancelled."""
        org = store.create_organization(name="Cancel Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="cancel@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        result = store.update_invitation_status(invitation.id, status="cancelled")

        assert result is True
        updated = store.get_invitation_by_id(invitation.id)
        assert updated.status == "cancelled"

    def test_delete_invitation(self, store):
        """Test deleting an invitation."""
        org = store.create_organization(name="Delete Org", owner_id="user_1")
        invitation = OrganizationInvitation(
            org_id=org.id,
            email="delete@example.com",
            role="member",
            invited_by="user_1",
        )
        store.create_invitation(invitation)

        result = store.delete_invitation(invitation.id)

        assert result is True
        assert store.get_invitation_by_id(invitation.id) is None


class TestInvitationCleanup:
    """Test invitation cleanup operations."""

    def test_cleanup_expired_invitations(self, store):
        """Test cleaning up expired invitations."""
        org = store.create_organization(name="Cleanup Org", owner_id="user_1")

        # Create expired invitation
        expired_invitation = OrganizationInvitation(
            org_id=org.id,
            email="expired@example.com",
            role="member",
            invited_by="user_1",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        store.create_invitation(expired_invitation)

        # Create valid invitation
        valid_invitation = OrganizationInvitation(
            org_id=org.id,
            email="valid@example.com",
            role="member",
            invited_by="user_1",
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        store.create_invitation(valid_invitation)

        count = store.cleanup_expired_invitations()

        assert count == 1
        assert store.get_invitation_by_id(expired_invitation.id) is None
        assert store.get_invitation_by_id(valid_invitation.id) is not None


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Test database connection management."""

    def test_close(self, store):
        """Test closing the store."""
        # Should not raise
        store.close()

    def test_external_connection(self, db_with_schema):
        """Test using external connection factory."""

        def get_conn():
            conn = sqlite3.connect(str(db_with_schema), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

        store = OrganizationStore(db_with_schema, get_connection=get_conn)
        org = store.create_organization(name="External Conn Org", owner_id="user_1")

        assert org is not None
        assert store.get_organization_by_id(org.id) is not None
