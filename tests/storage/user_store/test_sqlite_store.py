"""
Tests for UserStore SQLite backend.

Comprehensive tests for user persistence including:
- User CRUD operations
- Organization management
- Password/credential handling
- Concurrent access
- Data integrity
"""

import asyncio
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.billing.models import Organization, SubscriptionTier, User, hash_password
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


class TestUserStoreInitialization:
    """Tests for UserStore initialization and schema."""

    def test_creates_database_file(self, temp_db_path):
        """Test that UserStore creates the database file."""
        store = UserStore(temp_db_path)
        assert temp_db_path.exists()
        store.close()

    def test_creates_parent_directory(self, tmp_path):
        """Test that UserStore creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "test.db"
        store = UserStore(nested_path)
        assert nested_path.parent.exists()
        store.close()

    def test_initializes_tables(self, user_store, temp_db_path):
        """Test that UserStore creates required tables."""
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {
            "users",
            "organizations",
            "usage_events",
            "oauth_providers",
            "audit_log",
            "org_invitations",
        }
        assert expected_tables.issubset(tables)

    def test_uses_wal_mode(self, user_store, temp_db_path):
        """Test that UserStore uses WAL journal mode for better concurrency."""
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"


class TestUserCreation:
    """Tests for user creation."""

    def test_create_user_basic(self, user_store):
        """Test creating a basic user."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="test@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Test User",
        )

        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.id is not None
        assert len(user.id) > 0
        assert user.is_active is True
        assert user.role == "member"

    def test_create_user_with_org_and_role(self, user_store):
        """Test creating user with organization and custom role."""
        password_hash, password_salt = hash_password("testpass123")

        # First create an organization
        org = user_store.create_organization(
            name="Test Org",
            owner_id="temp-owner",
        )

        user = user_store.create_user(
            email="admin@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Admin User",
            org_id=org.id,
            role="admin",
        )

        assert user.org_id == org.id
        assert user.role == "admin"

    def test_create_user_duplicate_email_raises(self, user_store):
        """Test that creating user with duplicate email raises ValueError."""
        password_hash, password_salt = hash_password("testpass123")

        user_store.create_user(
            email="duplicate@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        with pytest.raises(ValueError, match="already exists"):
            user_store.create_user(
                email="duplicate@example.com",
                password_hash=password_hash,
                password_salt=password_salt,
            )


class TestUserRetrieval:
    """Tests for user retrieval operations."""

    def test_get_user_by_id(self, user_store):
        """Test retrieving user by ID."""
        password_hash, password_salt = hash_password("testpass123")
        created = user_store.create_user(
            email="byid@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="By ID User",
        )

        retrieved = user_store.get_user_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.email == "byid@example.com"
        assert retrieved.name == "By ID User"

    def test_get_user_by_id_not_found(self, user_store):
        """Test retrieving non-existent user returns None."""
        result = user_store.get_user_by_id("non-existent-id")
        assert result is None

    def test_get_user_by_email(self, user_store):
        """Test retrieving user by email."""
        password_hash, password_salt = hash_password("testpass123")
        created = user_store.create_user(
            email="byemail@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="By Email User",
        )

        retrieved = user_store.get_user_by_email("byemail@example.com")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "By Email User"

    def test_get_user_by_email_not_found(self, user_store):
        """Test retrieving user by non-existent email returns None."""
        result = user_store.get_user_by_email("notfound@example.com")
        assert result is None

    def test_get_users_batch(self, user_store):
        """Test batch retrieval of multiple users."""
        password_hash, password_salt = hash_password("testpass123")
        user1 = user_store.create_user(
            email="batch1@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        user2 = user_store.create_user(
            email="batch2@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        user3 = user_store.create_user(
            email="batch3@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        result = user_store.get_users_batch([user1.id, user2.id, "nonexistent"])

        assert user1.id in result
        assert user2.id in result
        assert "nonexistent" not in result
        assert result[user1.id].email == "batch1@example.com"
        assert result[user2.id].email == "batch2@example.com"


class TestUserUpdate:
    """Tests for user update operations."""

    def test_update_user_name(self, user_store):
        """Test updating user name."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="update@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Original Name",
        )

        result = user_store.update_user(user.id, name="Updated Name")

        assert result is True
        updated = user_store.get_user_by_id(user.id)
        assert updated.name == "Updated Name"

    def test_update_user_multiple_fields(self, user_store):
        """Test updating multiple fields at once."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="multiupdate@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Original",
        )

        result = user_store.update_user(
            user.id,
            name="New Name",
            role="admin",
            is_active=False,
        )

        assert result is True
        updated = user_store.get_user_by_id(user.id)
        assert updated.name == "New Name"
        assert updated.role == "admin"
        assert updated.is_active is False

    def test_update_user_not_found(self, user_store):
        """Test updating non-existent user returns False."""
        result = user_store.update_user("nonexistent", name="Test")
        assert result is False

    def test_update_users_batch(self, user_store):
        """Test batch updating multiple users."""
        password_hash, password_salt = hash_password("testpass123")
        user1 = user_store.create_user(
            email="batchupdate1@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="User 1",
        )
        user2 = user_store.create_user(
            email="batchupdate2@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="User 2",
        )

        updates = [
            {"id": user1.id, "name": "Updated 1"},
            {"id": user2.id, "name": "Updated 2"},
        ]
        count = user_store.update_users_batch(updates)

        assert count == 2
        assert user_store.get_user_by_id(user1.id).name == "Updated 1"
        assert user_store.get_user_by_id(user2.id).name == "Updated 2"


class TestUserDeletion:
    """Tests for user deletion."""

    def test_delete_user(self, user_store):
        """Test deleting a user."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="delete@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        result = user_store.delete_user(user.id)

        assert result is True
        assert user_store.get_user_by_id(user.id) is None

    def test_delete_user_not_found(self, user_store):
        """Test deleting non-existent user returns False."""
        result = user_store.delete_user("nonexistent")
        assert result is False


class TestUserListing:
    """Tests for listing users with pagination."""

    def test_list_all_users(self, user_store):
        """Test listing all users."""
        password_hash, password_salt = hash_password("testpass123")

        # Create several users
        for i in range(5):
            user_store.create_user(
                email=f"list{i}@example.com",
                password_hash=password_hash,
                password_salt=password_salt,
            )

        users, total = user_store.list_all_users()

        assert total == 5
        assert len(users) == 5

    def test_list_users_with_pagination(self, user_store):
        """Test listing users with pagination."""
        password_hash, password_salt = hash_password("testpass123")

        for i in range(10):
            user_store.create_user(
                email=f"paginate{i}@example.com",
                password_hash=password_hash,
                password_salt=password_salt,
            )

        page1, total = user_store.list_all_users(limit=3, offset=0)
        page2, _ = user_store.list_all_users(limit=3, offset=3)
        page3, _ = user_store.list_all_users(limit=3, offset=6)

        assert total == 10
        assert len(page1) == 3
        assert len(page2) == 3
        assert len(page3) == 3

        # Ensure no duplicates across pages
        all_ids = {u.id for u in page1} | {u.id for u in page2} | {u.id for u in page3}
        assert len(all_ids) == 9

    def test_list_users_filter_by_org(self, user_store):
        """Test filtering users by organization."""
        password_hash, password_salt = hash_password("testpass123")

        org = user_store.create_organization(name="Filter Org", owner_id="temp")

        user_store.create_user(
            email="inorg@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            org_id=org.id,
        )
        user_store.create_user(
            email="notinorg@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        users, total = user_store.list_all_users(org_id_filter=org.id)

        assert total == 1
        assert users[0].email == "inorg@example.com"

    def test_list_users_filter_by_role(self, user_store):
        """Test filtering users by role."""
        password_hash, password_salt = hash_password("testpass123")

        user_store.create_user(
            email="admin@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            role="admin",
        )
        user_store.create_user(
            email="member@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            role="member",
        )

        admins, total = user_store.list_all_users(role_filter="admin")

        assert total == 1
        assert admins[0].role == "admin"

    def test_list_users_active_only(self, user_store):
        """Test filtering to show only active users."""
        password_hash, password_salt = hash_password("testpass123")

        active_user = user_store.create_user(
            email="active@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        inactive_user = user_store.create_user(
            email="inactive@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        user_store.update_user(inactive_user.id, is_active=False)

        users, total = user_store.list_all_users(active_only=True)

        assert total == 1
        assert users[0].email == "active@example.com"


class TestCredentialHandling:
    """Tests for password and API key handling."""

    def test_get_user_by_api_key(self, user_store):
        """Test retrieving user by API key."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="apikey@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        # Generate API key for the user
        created_user = user_store.get_user_by_id(user.id)
        api_key = created_user.generate_api_key()

        # Update the user with the new API key hash
        user_store.update_user(
            user.id,
            api_key_hash=created_user.api_key_hash,
            api_key_prefix=created_user.api_key_prefix,
            api_key_created_at=created_user.api_key_created_at.isoformat(),
            api_key_expires_at=created_user.api_key_expires_at.isoformat(),
        )

        retrieved = user_store.get_user_by_api_key(api_key)

        assert retrieved is not None
        assert retrieved.id == user.id

    def test_get_user_by_invalid_api_key(self, user_store):
        """Test that invalid API key returns None."""
        result = user_store.get_user_by_api_key("invalid-api-key")
        assert result is None

    def test_user_preferences(self, user_store):
        """Test getting and setting user preferences."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="prefs@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        prefs = {"theme": "dark", "notifications": True}
        result = user_store.set_user_preferences(user.id, prefs)

        assert result is True

        retrieved_prefs = user_store.get_user_preferences(user.id)
        assert retrieved_prefs == prefs

    def test_increment_token_version(self, user_store):
        """Test incrementing token version for invalidation."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="tokenver@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        initial_version = user_store.get_user_by_id(user.id).token_version

        new_version = user_store.increment_token_version(user.id)

        assert new_version == initial_version + 1
        assert user_store.get_user_by_id(user.id).token_version == new_version


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_concurrent_user_creation(self, user_store):
        """Test concurrent user creation doesn't cause conflicts."""
        password_hash, password_salt = hash_password("testpass123")
        errors = []
        created_count = [0]
        lock = threading.Lock()

        def create_user(index):
            try:
                user_store.create_user(
                    email=f"concurrent{index}@example.com",
                    password_hash=password_hash,
                    password_salt=password_salt,
                )
                with lock:
                    created_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_user, i) for i in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        assert created_count[0] == 20

    def test_concurrent_read_write(self, user_store):
        """Test concurrent reads and writes."""
        password_hash, password_salt = hash_password("testpass123")

        # Create initial users
        users = []
        for i in range(5):
            user = user_store.create_user(
                email=f"rw{i}@example.com",
                password_hash=password_hash,
                password_salt=password_salt,
            )
            users.append(user)

        errors = []
        lock = threading.Lock()

        def read_user(user_id):
            try:
                result = user_store.get_user_by_id(user_id)
                assert result is not None
            except Exception as e:
                with lock:
                    errors.append(f"Read error: {e}")

        def update_user(user_id, index):
            try:
                user_store.update_user(user_id, name=f"Updated {index}")
            except Exception as e:
                with lock:
                    errors.append(f"Write error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                user = users[i % len(users)]
                if i % 3 == 0:
                    futures.append(executor.submit(update_user, user.id, i))
                else:
                    futures.append(executor.submit(read_user, user.id))

            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent R/W: {errors}"


class TestDataIntegrity:
    """Tests for data integrity."""

    def test_save_then_retrieve_returns_same_data(self, user_store):
        """Test that saved data can be retrieved unchanged."""
        password_hash, password_salt = hash_password("testpass123")

        original = user_store.create_user(
            email="integrity@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Integrity Test",
            role="admin",
        )

        retrieved = user_store.get_user_by_id(original.id)

        assert retrieved.id == original.id
        assert retrieved.email == original.email
        assert retrieved.password_hash == original.password_hash
        assert retrieved.password_salt == original.password_salt
        assert retrieved.name == original.name
        assert retrieved.role == original.role
        assert retrieved.is_active == original.is_active

    def test_transaction_rollback_on_error(self, user_store, temp_db_path):
        """Test that transactions are rolled back on error."""
        password_hash, password_salt = hash_password("testpass123")

        user_store.create_user(
            email="rollback@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        initial_count = user_store.list_all_users()[1]

        # Try to create a duplicate which should fail
        try:
            user_store.create_user(
                email="rollback@example.com",
                password_hash=password_hash,
                password_salt=password_salt,
            )
        except ValueError:
            pass

        final_count = user_store.list_all_users()[1]
        assert final_count == initial_count

    def test_close_cleans_up_connections(self, temp_db_path):
        """Test that close() properly cleans up connections."""
        store = UserStore(temp_db_path)

        password_hash, password_salt = hash_password("testpass123")
        store.create_user(
            email="cleanup@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )

        store.close()

        # Should be able to open a new connection after close
        new_conn = sqlite3.connect(str(temp_db_path))
        cursor = new_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        new_conn.close()

        assert count == 1


class TestOrganizationOperations:
    """Tests for organization operations via UserStore."""

    def test_create_organization(self, user_store):
        """Test creating an organization."""
        org = user_store.create_organization(
            name="Test Organization",
            owner_id="owner-123",
            tier=SubscriptionTier.PROFESSIONAL,
        )

        assert org.name == "Test Organization"
        assert org.owner_id == "owner-123"
        assert org.tier == SubscriptionTier.PROFESSIONAL
        assert org.slug is not None

    def test_get_organization_by_id(self, user_store):
        """Test retrieving organization by ID."""
        created = user_store.create_organization(
            name="By ID Org",
            owner_id="owner-456",
        )

        retrieved = user_store.get_organization_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "By ID Org"

    def test_get_organization_by_slug(self, user_store):
        """Test retrieving organization by slug."""
        created = user_store.create_organization(
            name="By Slug Org",
            owner_id="owner-789",
            slug="by-slug-org",
        )

        retrieved = user_store.get_organization_by_slug("by-slug-org")

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_add_user_to_org(self, user_store):
        """Test adding user to organization."""
        password_hash, password_salt = hash_password("testpass123")
        user = user_store.create_user(
            email="addtoorg@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        org = user_store.create_organization(name="Add User Org", owner_id="temp")

        result = user_store.add_user_to_org(user.id, org.id, role="member")

        assert result is True
        updated_user = user_store.get_user_by_id(user.id)
        assert updated_user.org_id == org.id

    def test_get_org_members(self, user_store):
        """Test getting all members of an organization."""
        password_hash, password_salt = hash_password("testpass123")
        org = user_store.create_organization(name="Members Org", owner_id="temp")

        user1 = user_store.create_user(
            email="member1@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            org_id=org.id,
        )
        user2 = user_store.create_user(
            email="member2@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            org_id=org.id,
        )

        members = user_store.get_org_members(org.id)

        assert len(members) == 2
        member_emails = {m.email for m in members}
        assert "member1@example.com" in member_emails
        assert "member2@example.com" in member_emails


class TestAdminStats:
    """Tests for admin statistics."""

    def test_get_admin_stats(self, user_store):
        """Test getting admin dashboard statistics."""
        password_hash, password_salt = hash_password("testpass123")

        # Create some users and organizations
        user_store.create_user(
            email="stat1@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        user_store.create_user(
            email="stat2@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
        )
        user_store.create_organization(name="Stats Org", owner_id="temp")

        stats = user_store.get_admin_stats()

        assert "total_users" in stats
        assert stats["total_users"] >= 2
        assert "total_organizations" in stats
        assert stats["total_organizations"] >= 1
        assert "tier_distribution" in stats
        assert "active_users" in stats
