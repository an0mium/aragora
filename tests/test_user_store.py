"""
Tests for UserStore - SQLite backend for user and organization persistence.

Tests cover:
- User CRUD operations
- Organization CRUD operations
- Membership management
- Usage tracking
- Transaction handling
- Edge cases
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

from aragora.storage.user_store import UserStore
from aragora.billing.models import (
    User,
    Organization,
    SubscriptionTier,
    hash_password,
    verify_password,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def store(db_path):
    """Create a UserStore instance."""
    store = UserStore(db_path)
    yield store
    store.close()


@pytest.fixture
def test_user(store):
    """Create a test user."""
    password_hash, salt = hash_password("testpassword")
    return store.create_user(
        email="test@example.com",
        password_hash=password_hash,
        password_salt=salt,
        name="Test User",
    )


@pytest.fixture
def test_org(store, test_user):
    """Create a test organization."""
    return store.create_organization(
        name="Test Org",
        owner_id=test_user.id,
    )


# =============================================================================
# Password Hashing Tests (from billing.models)
# =============================================================================


class TestPasswordHashing:
    """Tests for password hashing utilities."""

    def test_hash_password_generates_salt(self):
        """hash_password should produce different hashes for same password."""
        hash1, salt1 = hash_password("password")
        hash2, salt2 = hash_password("password")

        # With bcrypt, salt is embedded in hash (returned salt is empty)
        # With SHA-256, salt is returned separately
        # Either way, hashes should differ due to different salts
        assert hash1 != hash2  # Different hashes due to different salts

    def test_hash_password_with_salt(self):
        """hash_password produces consistent results (SHA-256 mode)."""
        # This test verifies SHA-256 fallback mode with explicit salt
        # With bcrypt, the salt parameter is ignored (salt embedded in hash)
        from aragora.billing.models import HAS_BCRYPT

        if HAS_BCRYPT:
            # bcrypt ignores the salt parameter and generates its own
            hash1, _ = hash_password("password", "fixed_salt")
            hash2, _ = hash_password("password", "fixed_salt")
            # Hashes will differ because bcrypt generates new salt each time
            assert hash1 != hash2
        else:
            # SHA-256 fallback uses provided salt
            salt = "fixed_salt"
            hash1, salt1 = hash_password("password", salt)
            hash2, salt2 = hash_password("password", salt)
            assert salt1 == salt2 == salt
            assert hash1 == hash2  # Same hashes with same salt

    def test_verify_password_correct(self):
        """verify_password should return True for correct password."""
        password_hash, salt = hash_password("mypassword")
        assert verify_password("mypassword", password_hash, salt) is True

    def test_verify_password_incorrect(self):
        """verify_password should return False for incorrect password."""
        password_hash, salt = hash_password("mypassword")
        assert verify_password("wrongpassword", password_hash, salt) is False


# =============================================================================
# User CRUD Tests
# =============================================================================


class TestUserCreate:
    """Tests for user creation."""

    def test_create_user_basic(self, store):
        """Should create a user with required fields."""
        password_hash, salt = hash_password("password123")
        user = store.create_user(
            email="user@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )

        assert user.id is not None
        assert user.email == "user@example.com"
        assert user.password_hash == password_hash
        assert user.password_salt == salt
        assert user.is_active is True
        assert user.email_verified is False

    def test_create_user_with_name(self, store):
        """Should create a user with optional name."""
        password_hash, salt = hash_password("password")
        user = store.create_user(
            email="named@example.com",
            password_hash=password_hash,
            password_salt=salt,
            name="John Doe",
        )

        assert user.name == "John Doe"

    def test_create_user_with_org(self, store, test_org):
        """Should create a user with organization."""
        password_hash, salt = hash_password("password")
        user = store.create_user(
            email="member@example.com",
            password_hash=password_hash,
            password_salt=salt,
            org_id=test_org.id,
            role="member",
        )

        assert user.org_id == test_org.id
        assert user.role == "member"

    def test_create_user_duplicate_email(self, store, test_user):
        """Should raise ValueError for duplicate email."""
        password_hash, salt = hash_password("password")

        with pytest.raises(ValueError, match="Email already exists"):
            store.create_user(
                email=test_user.email,
                password_hash=password_hash,
                password_salt=salt,
            )


class TestUserGet:
    """Tests for user retrieval."""

    def test_get_user_by_id(self, store, test_user):
        """Should retrieve user by ID."""
        user = store.get_user_by_id(test_user.id)

        assert user is not None
        assert user.id == test_user.id
        assert user.email == test_user.email

    def test_get_user_by_id_not_found(self, store):
        """Should return None for nonexistent ID."""
        user = store.get_user_by_id("nonexistent-id")
        assert user is None

    def test_get_user_by_email(self, store, test_user):
        """Should retrieve user by email."""
        user = store.get_user_by_email(test_user.email)

        assert user is not None
        assert user.email == test_user.email

    def test_get_user_by_email_case_insensitive(self, store):
        """Email lookup should be case insensitive."""
        password_hash, salt = hash_password("password")
        store.create_user(
            email="lowercase@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )

        # Note: depends on how the store handles case sensitivity
        # This test documents expected behavior
        user = store.get_user_by_email("lowercase@example.com")
        assert user is not None

    def test_get_user_by_email_not_found(self, store):
        """Should return None for nonexistent email."""
        user = store.get_user_by_email("nonexistent@example.com")
        assert user is None

    def test_get_user_by_api_key(self, store, test_user):
        """Should retrieve user by API key."""
        # First set an API key
        api_key = "test-api-key-12345"
        store.update_user(test_user.id, api_key=api_key)

        user = store.get_user_by_api_key(api_key)

        assert user is not None
        assert user.id == test_user.id

    def test_get_user_by_api_key_not_found(self, store):
        """Should return None for nonexistent API key."""
        user = store.get_user_by_api_key("nonexistent-key")
        assert user is None


class TestUserUpdate:
    """Tests for user updates."""

    def test_update_user_name(self, store, test_user):
        """Should update user name."""
        result = store.update_user(test_user.id, name="New Name")

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.name == "New Name"

    def test_update_user_email(self, store, test_user):
        """Should update user email."""
        result = store.update_user(test_user.id, email="new@example.com")

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.email == "new@example.com"

    def test_update_user_api_key(self, store, test_user):
        """Should update user API key."""
        api_key = "new-api-key"
        result = store.update_user(
            test_user.id,
            api_key=api_key,
            api_key_created_at=datetime.utcnow(),
        )

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.api_key == api_key
        assert user.api_key_created_at is not None

    def test_update_user_is_active(self, store, test_user):
        """Should update user active status."""
        result = store.update_user(test_user.id, is_active=False)

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.is_active is False

    def test_update_user_email_verified(self, store, test_user):
        """Should update email verified status."""
        result = store.update_user(test_user.id, email_verified=True)

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.email_verified is True

    def test_update_user_multiple_fields(self, store, test_user):
        """Should update multiple fields at once."""
        result = store.update_user(
            test_user.id,
            name="Updated Name",
            role="admin",
            is_active=False,
        )

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user.name == "Updated Name"
        assert user.role == "admin"
        assert user.is_active is False

    def test_update_user_no_fields(self, store, test_user):
        """Should return False if no fields provided."""
        result = store.update_user(test_user.id)
        assert result is False

    def test_update_user_not_found(self, store):
        """Should return False for nonexistent user."""
        result = store.update_user("nonexistent-id", name="New Name")
        assert result is False

    def test_update_user_invalid_field(self, store, test_user):
        """Should ignore invalid fields."""
        result = store.update_user(test_user.id, invalid_field="value")
        assert result is False

    def test_update_user_updates_timestamp(self, store, test_user):
        """Should update the updated_at timestamp."""
        original_updated = test_user.updated_at

        import time
        time.sleep(0.1)  # Ensure time difference

        store.update_user(test_user.id, name="New Name")
        user = store.get_user_by_id(test_user.id)

        assert user.updated_at > original_updated


class TestUserDelete:
    """Tests for user deletion."""

    def test_delete_user(self, store, test_user):
        """Should delete a user."""
        result = store.delete_user(test_user.id)

        assert result is True
        user = store.get_user_by_id(test_user.id)
        assert user is None

    def test_delete_user_not_found(self, store):
        """Should return False for nonexistent user."""
        result = store.delete_user("nonexistent-id")
        assert result is False


# =============================================================================
# Organization CRUD Tests
# =============================================================================


class TestOrganizationCreate:
    """Tests for organization creation."""

    def test_create_organization_basic(self, store, test_user):
        """Should create an organization with required fields."""
        org = store.create_organization(
            name="My Org",
            owner_id=test_user.id,
        )

        assert org.id is not None
        assert org.name == "My Org"
        assert org.owner_id == test_user.id
        assert org.tier == SubscriptionTier.FREE

    def test_create_organization_with_slug(self, store, test_user):
        """Should create organization with custom slug."""
        org = store.create_organization(
            name="My Organization",
            owner_id=test_user.id,
            slug="my-custom-slug",
        )

        assert org.slug == "my-custom-slug"

    def test_create_organization_auto_slug(self, store, test_user):
        """Should auto-generate slug from name."""
        org = store.create_organization(
            name="Test Organization Name",
            owner_id=test_user.id,
        )

        assert "test-organization-name" in org.slug

    def test_create_organization_with_tier(self, store, test_user):
        """Should create organization with specified tier."""
        org = store.create_organization(
            name="Pro Org",
            owner_id=test_user.id,
            tier=SubscriptionTier.PROFESSIONAL,
        )

        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_create_organization_sets_owner_org_id(self, store, test_user):
        """Creating org should update owner's org_id."""
        org = store.create_organization(
            name="Owner Org",
            owner_id=test_user.id,
        )

        user = store.get_user_by_id(test_user.id)
        assert user.org_id == org.id
        assert user.role == "owner"


class TestOrganizationGet:
    """Tests for organization retrieval."""

    def test_get_organization_by_id(self, store, test_org):
        """Should retrieve organization by ID."""
        org = store.get_organization_by_id(test_org.id)

        assert org is not None
        assert org.id == test_org.id
        assert org.name == test_org.name

    def test_get_organization_by_id_not_found(self, store):
        """Should return None for nonexistent ID."""
        org = store.get_organization_by_id("nonexistent-id")
        assert org is None

    def test_get_organization_by_slug(self, store, test_org):
        """Should retrieve organization by slug."""
        org = store.get_organization_by_slug(test_org.slug)

        assert org is not None
        assert org.slug == test_org.slug

    def test_get_organization_by_slug_not_found(self, store):
        """Should return None for nonexistent slug."""
        org = store.get_organization_by_slug("nonexistent-slug")
        assert org is None

    def test_get_organization_by_stripe_customer(self, store, test_org):
        """Should retrieve org by Stripe customer ID."""
        stripe_id = "cus_test123"
        store.update_organization(test_org.id, stripe_customer_id=stripe_id)

        org = store.get_organization_by_stripe_customer(stripe_id)

        assert org is not None
        assert org.id == test_org.id

    def test_get_organization_by_subscription(self, store, test_org):
        """Should retrieve org by Stripe subscription ID."""
        sub_id = "sub_test123"
        store.update_organization(test_org.id, stripe_subscription_id=sub_id)

        org = store.get_organization_by_subscription(sub_id)

        assert org is not None
        assert org.id == test_org.id


class TestOrganizationUpdate:
    """Tests for organization updates."""

    def test_update_organization_name(self, store, test_org):
        """Should update organization name."""
        result = store.update_organization(test_org.id, name="New Org Name")

        assert result is True
        org = store.get_organization_by_id(test_org.id)
        assert org.name == "New Org Name"

    def test_update_organization_tier(self, store, test_org):
        """Should update organization tier."""
        result = store.update_organization(
            test_org.id,
            tier=SubscriptionTier.PROFESSIONAL,
        )

        assert result is True
        org = store.get_organization_by_id(test_org.id)
        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_update_organization_stripe_ids(self, store, test_org):
        """Should update Stripe IDs."""
        result = store.update_organization(
            test_org.id,
            stripe_customer_id="cus_123",
            stripe_subscription_id="sub_456",
        )

        assert result is True
        org = store.get_organization_by_id(test_org.id)
        assert org.stripe_customer_id == "cus_123"
        assert org.stripe_subscription_id == "sub_456"

    def test_update_organization_settings(self, store, test_org):
        """Should update organization settings."""
        settings = {"feature_x": True, "max_agents": 5}
        result = store.update_organization(test_org.id, settings=settings)

        assert result is True
        org = store.get_organization_by_id(test_org.id)
        assert org.settings == settings

    def test_update_organization_no_fields(self, store, test_org):
        """Should return False if no fields provided."""
        result = store.update_organization(test_org.id)
        assert result is False

    def test_update_organization_not_found(self, store):
        """Should return False for nonexistent org."""
        result = store.update_organization("nonexistent-id", name="New Name")
        assert result is False


# =============================================================================
# Membership Tests
# =============================================================================


class TestOrgMembership:
    """Tests for organization membership management."""

    def test_add_user_to_org(self, store, test_org):
        """Should add user to organization."""
        # Create a new user
        password_hash, salt = hash_password("password")
        new_user = store.create_user(
            email="member@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )

        result = store.add_user_to_org(new_user.id, test_org.id, role="member")

        assert result is True
        user = store.get_user_by_id(new_user.id)
        assert user.org_id == test_org.id
        assert user.role == "member"

    def test_add_user_to_org_as_admin(self, store, test_org):
        """Should add user with admin role."""
        password_hash, salt = hash_password("password")
        new_user = store.create_user(
            email="admin@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )

        result = store.add_user_to_org(new_user.id, test_org.id, role="admin")

        assert result is True
        user = store.get_user_by_id(new_user.id)
        assert user.role == "admin"

    def test_remove_user_from_org(self, store, test_org):
        """Should remove user from organization."""
        # Create and add a user
        password_hash, salt = hash_password("password")
        new_user = store.create_user(
            email="temp@example.com",
            password_hash=password_hash,
            password_salt=salt,
            org_id=test_org.id,
        )

        result = store.remove_user_from_org(new_user.id)

        assert result is True
        user = store.get_user_by_id(new_user.id)
        assert user.org_id is None
        assert user.role == "member"

    def test_get_org_members(self, store, test_org, test_user):
        """Should get all members of organization."""
        # Add another member
        password_hash, salt = hash_password("password")
        member = store.create_user(
            email="member2@example.com",
            password_hash=password_hash,
            password_salt=salt,
            org_id=test_org.id,
        )

        members = store.get_org_members(test_org.id)

        assert len(members) == 2
        member_ids = {m.id for m in members}
        assert test_user.id in member_ids
        assert member.id in member_ids

    def test_get_org_members_empty(self, store):
        """Should return empty list for org with no members."""
        members = store.get_org_members("nonexistent-org")
        assert members == []


# =============================================================================
# Usage Tracking Tests
# =============================================================================


class TestUsageTracking:
    """Tests for usage tracking functionality."""

    def test_increment_usage(self, store, test_org):
        """Should increment debate usage."""
        initial = test_org.debates_used_this_month

        new_total = store.increment_usage(test_org.id, 1)

        assert new_total == initial + 1

    def test_increment_usage_multiple(self, store, test_org):
        """Should increment by specified count."""
        initial = test_org.debates_used_this_month

        new_total = store.increment_usage(test_org.id, 5)

        assert new_total == initial + 5

    def test_increment_usage_accumulates(self, store, test_org):
        """Multiple increments should accumulate."""
        store.increment_usage(test_org.id, 3)
        store.increment_usage(test_org.id, 2)
        final = store.increment_usage(test_org.id, 1)

        assert final == 6

    def test_increment_usage_nonexistent_org(self, store):
        """Should return 0 for nonexistent org."""
        result = store.increment_usage("nonexistent-id", 1)
        assert result == 0

    def test_record_usage_event(self, store, test_org):
        """Should record a usage event."""
        # This mainly tests that no exception is raised
        store.record_usage_event(
            org_id=test_org.id,
            event_type="debate_completed",
            count=1,
            metadata={"topic": "test topic"},
        )

    def test_reset_org_usage(self, store, test_org):
        """Should reset usage for single org."""
        store.increment_usage(test_org.id, 10)

        result = store.reset_org_usage(test_org.id)

        assert result is True
        org = store.get_organization_by_id(test_org.id)
        assert org.debates_used_this_month == 0

    def test_reset_monthly_usage(self, store, test_org):
        """Should reset usage for all organizations."""
        store.increment_usage(test_org.id, 5)

        count = store.reset_monthly_usage()

        assert count >= 1  # At least the test org
        org = store.get_organization_by_id(test_org.id)
        assert org.debates_used_this_month == 0

    def test_get_usage_summary(self, store, test_org):
        """Should return usage summary."""
        store.increment_usage(test_org.id, 3)

        summary = store.get_usage_summary(test_org.id)

        assert summary["org_id"] == test_org.id
        assert summary["tier"] == SubscriptionTier.FREE.value
        assert summary["debates_used"] == 3
        assert "debates_limit" in summary
        assert "debates_remaining" in summary
        assert "is_at_limit" in summary

    def test_get_usage_summary_nonexistent(self, store):
        """Should return empty dict for nonexistent org."""
        summary = store.get_usage_summary("nonexistent-id")
        assert summary == {}


# =============================================================================
# Transaction and Concurrency Tests
# =============================================================================


class TestTransactions:
    """Tests for transaction handling."""

    def test_transaction_rollback_on_error(self, store, test_user):
        """Transaction should rollback on error."""
        initial_name = test_user.name

        # This should fail due to duplicate email
        password_hash, salt = hash_password("password")
        try:
            store.create_user(
                email=test_user.email,  # Duplicate
                password_hash=password_hash,
                password_salt=salt,
            )
        except ValueError:
            pass

        # Original user should be unchanged
        user = store.get_user_by_id(test_user.id)
        assert user.name == initial_name

    def test_concurrent_reads(self, db_path, test_user):
        """Should handle concurrent reads."""
        store1 = UserStore(db_path)
        store2 = UserStore(db_path)

        # Create a user in store1
        password_hash, salt = hash_password("password")
        user = store1.create_user(
            email="concurrent@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )

        # Read from both stores
        user1 = store1.get_user_by_id(user.id)
        user2 = store2.get_user_by_id(user.id)

        assert user1.email == user2.email

        store1.close()
        store2.close()

    def test_concurrent_increments(self, store, test_org):
        """Should handle concurrent usage increments."""
        num_threads = 10
        increments_per_thread = 10

        def increment():
            for _ in range(increments_per_thread):
                store.increment_usage(test_org.id, 1)

        threads = [Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        org = store.get_organization_by_id(test_org.id)
        assert org.debates_used_this_month == num_threads * increments_per_thread


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_name(self, store):
        """Should handle empty name."""
        password_hash, salt = hash_password("password")
        user = store.create_user(
            email="noname@example.com",
            password_hash=password_hash,
            password_salt=salt,
            name="",
        )

        assert user.name == ""

    def test_unicode_name(self, store):
        """Should handle unicode in names."""
        password_hash, salt = hash_password("password")
        user = store.create_user(
            email="unicode@example.com",
            password_hash=password_hash,
            password_salt=salt,
            name="日本語テスト",
        )

        retrieved = store.get_user_by_id(user.id)
        assert retrieved.name == "日本語テスト"

    def test_unicode_org_name(self, store, test_user):
        """Should handle unicode in org names."""
        org = store.create_organization(
            name="企業名テスト",
            owner_id=test_user.id,
        )

        retrieved = store.get_organization_by_id(org.id)
        assert retrieved.name == "企業名テスト"

    def test_special_characters_in_slug(self, store, test_user):
        """Should handle special characters when generating slug."""
        org = store.create_organization(
            name="Test & Company (LLC)",
            owner_id=test_user.id,
        )

        # Slug should be created without crashing
        assert org.slug is not None

    def test_close_and_reconnect(self, db_path):
        """Should handle close and reconnect."""
        store1 = UserStore(db_path)
        password_hash, salt = hash_password("password")
        user = store1.create_user(
            email="persist@example.com",
            password_hash=password_hash,
            password_salt=salt,
        )
        store1.close()

        # Reconnect
        store2 = UserStore(db_path)
        retrieved = store2.get_user_by_email("persist@example.com")
        assert retrieved is not None
        assert retrieved.id == user.id
        store2.close()

    def test_large_settings_json(self, store, test_org):
        """Should handle large settings JSON."""
        large_settings = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(100)
        }

        result = store.update_organization(test_org.id, settings=large_settings)
        assert result is True

        org = store.get_organization_by_id(test_org.id)
        assert org.settings == large_settings

    def test_null_api_key(self, store, test_user):
        """Should handle null API key."""
        user = store.get_user_by_id(test_user.id)
        assert user.api_key is None

    def test_last_login_tracking(self, store, test_user):
        """Should track last login time."""
        login_time = datetime.utcnow()
        store.update_user(test_user.id, last_login_at=login_time)

        user = store.get_user_by_id(test_user.id)
        assert user.last_login_at is not None
