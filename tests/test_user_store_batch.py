"""
Tests for UserStore batch operations (Phase 5 optimization).

Tests:
- get_users_batch: Fetch multiple users in single query
- update_users_batch: Update multiple users with executemany
- get_org_members_eager: Get org and members in single operation
- get_orgs_with_members_batch: Batch fetch multiple orgs with members
"""

import pytest
import hashlib
import secrets
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from aragora.storage.user_store import UserStore


def _hash_password(password: str) -> tuple[str, str]:
    """Create password hash and salt for testing."""
    salt = secrets.token_hex(16)
    hash_input = f"{password}{salt}".encode()
    password_hash = hashlib.sha256(hash_input).hexdigest()
    return password_hash, salt


class TestGetUsersBatch:
    """Tests for get_users_batch method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary UserStore."""
        db_path = tmp_path / "test_users.db"
        store = UserStore(str(db_path))
        return store

    def test_empty_list_returns_empty_dict(self, store):
        """Test that empty list returns empty dict."""
        result = store.get_users_batch([])
        assert result == {}

    def test_returns_found_users(self, store):
        """Test that found users are returned in dict."""
        # Create test users
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="User One",
        )
        user2 = store.create_user(
            email="user2@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="User Two",
        )

        # Fetch batch
        result = store.get_users_batch([user1.id, user2.id])

        assert len(result) == 2
        assert user1.id in result
        assert user2.id in result
        assert result[user1.id].email == "user1@test.com"
        assert result[user2.id].email == "user2@test.com"

    def test_missing_users_not_in_result(self, store):
        """Test that non-existent user IDs are not in result."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        result = store.get_users_batch([user1.id, "nonexistent-id"])

        assert len(result) == 1
        assert user1.id in result
        assert "nonexistent-id" not in result

    def test_handles_duplicate_ids(self, store):
        """Test that duplicate IDs don't cause issues."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        # Pass same ID multiple times
        result = store.get_users_batch([user1.id, user1.id, user1.id])

        assert len(result) == 1
        assert user1.id in result


class TestUpdateUsersBatch:
    """Tests for update_users_batch method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary UserStore."""
        db_path = tmp_path / "test_users.db"
        store = UserStore(str(db_path))
        return store

    def test_empty_list_returns_zero(self, store):
        """Test that empty list returns 0."""
        result = store.update_users_batch([])
        assert result == 0

    def test_updates_multiple_users(self, store):
        """Test that multiple users are updated."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="Original Name 1",
        )
        user2 = store.create_user(
            email="user2@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="Original Name 2",
        )

        result = store.update_users_batch([
            {"user_id": user1.id, "name": "Updated Name 1"},
            {"user_id": user2.id, "name": "Updated Name 2"},
        ])

        assert result >= 1  # At least some updated

        # Verify updates
        updated1 = store.get_user_by_id(user1.id)
        updated2 = store.get_user_by_id(user2.id)
        assert updated1.name == "Updated Name 1"
        assert updated2.name == "Updated Name 2"

    def test_updates_different_fields(self, store):
        """Test updating different fields for different users."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            role="member",
        )
        user2 = store.create_user(
            email="user2@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="Original",
        )

        store.update_users_batch([
            {"user_id": user1.id, "role": "admin"},
            {"user_id": user2.id, "name": "New Name"},
        ])

        updated1 = store.get_user_by_id(user1.id)
        updated2 = store.get_user_by_id(user2.id)
        assert updated1.role == "admin"
        assert updated2.name == "New Name"

    def test_skips_updates_without_user_id(self, store):
        """Test that updates without user_id are skipped."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            name="Original",
        )

        result = store.update_users_batch([
            {"name": "Should be skipped"},  # No user_id
            {"user_id": user1.id, "name": "Should update"},
        ])

        updated1 = store.get_user_by_id(user1.id)
        assert updated1.name == "Should update"

    def test_updates_boolean_fields(self, store):
        """Test updating boolean fields."""
        pw_hash, pw_salt = _hash_password("password123")
        user1 = store.create_user(
            email="user1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        store.update_users_batch([
            {"user_id": user1.id, "is_active": False, "email_verified": True},
        ])

        updated = store.get_user_by_id(user1.id)
        assert updated.is_active is False
        assert updated.email_verified is True


class TestGetOrgMembersEager:
    """Tests for get_org_members_eager method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary UserStore."""
        db_path = tmp_path / "test_users.db"
        store = UserStore(str(db_path))
        return store

    def test_returns_none_for_nonexistent_org(self, store):
        """Test that nonexistent org returns (None, [])."""
        org, members = store.get_org_members_eager("nonexistent-org")
        assert org is None
        assert members == []

    def test_returns_org_and_members(self, store):
        """Test that org and members are returned together."""
        # Create owner first
        pw_hash, pw_salt = _hash_password("password123")
        owner = store.create_user(
            email="owner@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        # Create org with owner
        org = store.create_organization(
            name="Test Org",
            owner_id=owner.id,
            slug="test-org",
        )

        # Update owner to be in org
        store.update_user(owner.id, org_id=org.id)

        # Create another member
        user2 = store.create_user(
            email="user2@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            org_id=org.id,
        )

        # Fetch eager
        fetched_org, members = store.get_org_members_eager(org.id)

        assert fetched_org is not None
        assert fetched_org.id == org.id
        assert fetched_org.name == "Test Org"
        assert len(members) == 2
        member_emails = {m.email for m in members}
        assert "owner@test.com" in member_emails
        assert "user2@test.com" in member_emails

    def test_returns_org_with_only_owner(self, store):
        """Test that org with only owner returns owner as member."""
        # Create owner
        pw_hash, pw_salt = _hash_password("password123")
        owner = store.create_user(
            email="owner@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        # Create org - owner gets added to org during creation
        org = store.create_organization(
            name="Owner Only Org",
            owner_id=owner.id,
            slug="owner-only-org",
        )

        fetched_org, members = store.get_org_members_eager(org.id)

        assert fetched_org is not None
        assert fetched_org.id == org.id
        # Owner is automatically added to the org
        assert len(members) == 1
        assert members[0].id == owner.id


class TestGetOrgsWithMembersBatch:
    """Tests for get_orgs_with_members_batch method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary UserStore."""
        db_path = tmp_path / "test_users.db"
        store = UserStore(str(db_path))
        return store

    def test_empty_list_returns_empty_dict(self, store):
        """Test that empty list returns empty dict."""
        result = store.get_orgs_with_members_batch([])
        assert result == {}

    def test_returns_multiple_orgs_with_members(self, store):
        """Test batch fetch of multiple orgs with members."""
        pw_hash, pw_salt = _hash_password("password123")

        # Create owners
        owner1 = store.create_user(
            email="owner1@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )
        owner2 = store.create_user(
            email="owner2@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )

        # Create orgs
        org1 = store.create_organization(name="Org 1", owner_id=owner1.id, slug="org-1")
        org2 = store.create_organization(name="Org 2", owner_id=owner2.id, slug="org-2")

        # Update owners to be in their orgs
        store.update_user(owner1.id, org_id=org1.id)
        store.update_user(owner2.id, org_id=org2.id)

        # Create additional member for org2
        user3 = store.create_user(
            email="user3@org2.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
            org_id=org2.id,
        )

        # Batch fetch
        result = store.get_orgs_with_members_batch([org1.id, org2.id])

        assert len(result) == 2
        assert org1.id in result
        assert org2.id in result

        org1_data, org1_members = result[org1.id]
        assert org1_data.name == "Org 1"
        assert len(org1_members) == 1

        org2_data, org2_members = result[org2.id]
        assert org2_data.name == "Org 2"
        assert len(org2_members) == 2

    def test_missing_orgs_not_in_result(self, store):
        """Test that non-existent orgs are not in result."""
        pw_hash, pw_salt = _hash_password("password123")
        owner = store.create_user(
            email="owner@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )
        org1 = store.create_organization(name="Org 1", owner_id=owner.id, slug="org-1")

        result = store.get_orgs_with_members_batch([org1.id, "nonexistent-id"])

        assert len(result) == 1
        assert org1.id in result
        assert "nonexistent-id" not in result

    def test_handles_duplicate_ids(self, store):
        """Test that duplicate IDs don't cause issues."""
        pw_hash, pw_salt = _hash_password("password123")
        owner = store.create_user(
            email="owner@test.com",
            password_hash=pw_hash,
            password_salt=pw_salt,
        )
        org1 = store.create_organization(name="Org 1", owner_id=owner.id, slug="org-1")

        result = store.get_orgs_with_members_batch([org1.id, org1.id, org1.id])

        assert len(result) == 1
        assert org1.id in result


class TestCompositeIndexes:
    """Tests to verify composite indexes are created."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary UserStore."""
        db_path = tmp_path / "test_users.db"
        store = UserStore(str(db_path))
        return store

    def test_org_role_index_exists(self, store):
        """Test that idx_users_org_role index is created."""
        with store._transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_users_org_role'"
            )
            result = cursor.fetchone()
        assert result is not None

    def test_email_active_index_exists(self, store):
        """Test that idx_users_email_active index is created."""
        with store._transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_users_email_active'"
            )
            result = cursor.fetchone()
        assert result is not None

    def test_usage_org_type_index_exists(self, store):
        """Test that idx_usage_org_type index is created."""
        with store._transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_usage_org_type'"
            )
            result = cursor.fetchone()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
