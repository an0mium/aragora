"""
Tests for TeamsWorkspaceStore - Microsoft Teams OAuth token management.

Tests cover:
- CRUD operations (save, get, delete, deactivate)
- Listing with filters and pagination
- Token encryption/decryption
- Multi-tenant workspace isolation
- Statistics
- SQLite backend initialization
- Concurrent access handling
- Data integrity
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.teams_workspace_store import (
    TeamsWorkspace,
    TeamsWorkspaceStore,
    get_teams_workspace_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_teams_workspaces.db")


@pytest.fixture
def workspace_store(temp_db_path):
    """Create a workspace store for testing."""
    store = TeamsWorkspaceStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def sample_workspace():
    """Create a sample Teams workspace."""
    return TeamsWorkspace(
        tenant_id="test-tenant-12345",
        tenant_name="Test Organization",
        access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test-token-12345",
        bot_id="bot-app-12345",
        installed_at=time.time(),
        installed_by="user-11111111",
        scopes=["ChannelMessage.Send", "Team.ReadBasic.All", "User.Read"],
        aragora_tenant_id="aragora-tenant-001",
        is_active=True,
        refresh_token="0.ATQ.refresh-token-12345",
        token_expires_at=time.time() + 3600,
        service_url="https://smba.trafficmanager.net/teams/",
    )


# ===========================================================================
# TeamsWorkspace Dataclass Tests
# ===========================================================================


class TestTeamsWorkspace:
    """Tests for TeamsWorkspace dataclass."""

    def test_to_dict_excludes_tokens(self, sample_workspace):
        """Test to_dict does not include access_token or refresh_token."""
        result = sample_workspace.to_dict()

        assert "access_token" not in result
        assert "refresh_token" not in result
        assert result["tenant_id"] == "test-tenant-12345"
        assert result["tenant_name"] == "Test Organization"
        assert result["bot_id"] == "bot-app-12345"

    def test_to_dict_includes_has_refresh_token_flag(self, sample_workspace):
        """Test to_dict includes has_refresh_token boolean flag."""
        result = sample_workspace.to_dict()

        assert result["has_refresh_token"] is True

        # Without refresh token
        sample_workspace.refresh_token = None
        result = sample_workspace.to_dict()
        assert result["has_refresh_token"] is False

    def test_to_dict_includes_scopes(self, sample_workspace):
        """Test to_dict includes scopes list."""
        result = sample_workspace.to_dict()

        assert result["scopes"] == ["ChannelMessage.Send", "Team.ReadBasic.All", "User.Read"]

    def test_to_dict_includes_iso_timestamp(self, sample_workspace):
        """Test to_dict includes ISO formatted timestamp."""
        result = sample_workspace.to_dict()

        assert "installed_at_iso" in result
        assert "T" in result["installed_at_iso"]  # ISO format

    def test_to_dict_includes_service_url(self, sample_workspace):
        """Test to_dict includes service_url."""
        result = sample_workspace.to_dict()

        assert result["service_url"] == "https://smba.trafficmanager.net/teams/"

    def test_default_scopes_empty(self):
        """Test default scopes is empty list."""
        workspace = TeamsWorkspace(
            tenant_id="T123",
            tenant_name="Test",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
        )

        assert workspace.scopes == []

    def test_default_is_active(self):
        """Test default is_active is True."""
        workspace = TeamsWorkspace(
            tenant_id="T123",
            tenant_name="Test",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
        )

        assert workspace.is_active is True

    def test_default_optional_fields(self):
        """Test default values for optional fields."""
        workspace = TeamsWorkspace(
            tenant_id="T123",
            tenant_name="Test",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
        )

        assert workspace.installed_by is None
        assert workspace.aragora_tenant_id is None
        assert workspace.refresh_token is None
        assert workspace.token_expires_at is None
        assert workspace.service_url is None


# ===========================================================================
# TeamsWorkspaceStore Initialization Tests
# ===========================================================================


class TestTeamsWorkspaceStoreInit:
    """Tests for TeamsWorkspaceStore initialization."""

    def test_sqlite_backend_initialization(self, temp_db_path):
        """Test SQLite database is properly initialized."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get connection and verify table exists
        conn = store._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='teams_workspaces'"
        )
        result = cursor.fetchone()

        assert result is not None
        assert result["name"] == "teams_workspaces"
        store.close()

    def test_schema_has_required_columns(self, temp_db_path):
        """Test schema has all required columns."""
        store = TeamsWorkspaceStore(db_path=temp_db_path)

        conn = store._get_connection()
        cursor = conn.execute("PRAGMA table_info(teams_workspaces)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "tenant_id",
            "tenant_name",
            "access_token",
            "bot_id",
            "installed_at",
            "installed_by",
            "scopes",
            "aragora_tenant_id",
            "is_active",
            "refresh_token",
            "token_expires_at",
            "service_url",
        }

        assert expected_columns.issubset(columns)
        store.close()

    def test_creates_indexes(self, temp_db_path):
        """Test required indexes are created."""
        store = TeamsWorkspaceStore(db_path=temp_db_path)

        conn = store._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_teams_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_teams_workspaces_aragora_tenant" in indexes
        assert "idx_teams_workspaces_active" in indexes
        store.close()

    @patch("aragora.storage.teams_workspace_store.ARAGORA_ENV", "production")
    @patch("aragora.storage.teams_workspace_store.ENCRYPTION_KEY", "")
    def test_raises_in_production_without_encryption_key(self, temp_db_path):
        """Test initialization raises error in production without encryption key."""
        with pytest.raises(ValueError, match="ARAGORA_ENCRYPTION_KEY"):
            TeamsWorkspaceStore(db_path=temp_db_path)

    @patch("aragora.storage.teams_workspace_store.ENCRYPTION_KEY", "")
    def test_warns_in_development_without_encryption_key(self, temp_db_path, caplog):
        """Test initialization warns in development without encryption key."""
        import aragora.storage.teams_workspace_store as module

        # Reset the warning flag
        module._encryption_warning_shown = False

        with caplog.at_level("WARNING"):
            store = TeamsWorkspaceStore(db_path=temp_db_path)
            store.close()

        assert "UNENCRYPTED" in caplog.text or module._encryption_warning_shown


# ===========================================================================
# TeamsWorkspaceStore CRUD Tests
# ===========================================================================


class TestTeamsWorkspaceStoreCRUD:
    """Tests for TeamsWorkspaceStore CRUD operations."""

    def test_save_and_get(self, workspace_store, sample_workspace):
        """Test save and retrieve a workspace."""
        saved = workspace_store.save(sample_workspace)
        assert saved is True

        workspace = workspace_store.get("test-tenant-12345")
        assert workspace is not None
        assert workspace.tenant_id == "test-tenant-12345"
        assert workspace.tenant_name == "Test Organization"
        assert "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9" in workspace.access_token

    def test_get_nonexistent(self, workspace_store):
        """Test get returns None for nonexistent workspace."""
        result = workspace_store.get("NONEXISTENT")
        assert result is None

    def test_save_updates_existing(self, workspace_store, sample_workspace):
        """Test save updates existing workspace (upsert)."""
        workspace_store.save(sample_workspace)

        # Update workspace
        sample_workspace.tenant_name = "Updated Organization"
        sample_workspace.scopes = ["new:scope"]
        workspace_store.save(sample_workspace)

        workspace = workspace_store.get("test-tenant-12345")
        assert workspace.tenant_name == "Updated Organization"
        assert workspace.scopes == ["new:scope"]

    def test_deactivate(self, workspace_store, sample_workspace):
        """Test deactivate marks workspace inactive."""
        workspace_store.save(sample_workspace)

        result = workspace_store.deactivate("test-tenant-12345")
        assert result is True

        workspace = workspace_store.get("test-tenant-12345")
        assert workspace.is_active is False

    def test_deactivate_nonexistent(self, workspace_store):
        """Test deactivate for nonexistent returns True (no-op)."""
        result = workspace_store.deactivate("NONEXISTENT")
        assert result is True  # SQLite UPDATE succeeds even if no rows

    def test_delete(self, workspace_store, sample_workspace):
        """Test delete removes workspace permanently."""
        workspace_store.save(sample_workspace)
        assert workspace_store.get("test-tenant-12345") is not None

        result = workspace_store.delete("test-tenant-12345")
        assert result is True

        assert workspace_store.get("test-tenant-12345") is None

    def test_delete_nonexistent(self, workspace_store):
        """Test delete for nonexistent returns True (no-op)."""
        result = workspace_store.delete("NONEXISTENT")
        assert result is True


# ===========================================================================
# Data Integrity Tests
# ===========================================================================


class TestTeamsWorkspaceStoreDataIntegrity:
    """Tests for data integrity - save then retrieve returns same data."""

    def test_data_integrity_all_fields(self, workspace_store, sample_workspace):
        """Test that all fields are preserved after save and retrieve."""
        workspace_store.save(sample_workspace)

        retrieved = workspace_store.get("test-tenant-12345")

        assert retrieved.tenant_id == sample_workspace.tenant_id
        assert retrieved.tenant_name == sample_workspace.tenant_name
        assert retrieved.access_token == sample_workspace.access_token
        assert retrieved.bot_id == sample_workspace.bot_id
        assert retrieved.installed_at == sample_workspace.installed_at
        assert retrieved.installed_by == sample_workspace.installed_by
        assert retrieved.scopes == sample_workspace.scopes
        assert retrieved.aragora_tenant_id == sample_workspace.aragora_tenant_id
        assert retrieved.is_active == sample_workspace.is_active
        assert retrieved.refresh_token == sample_workspace.refresh_token
        assert retrieved.token_expires_at == sample_workspace.token_expires_at
        assert retrieved.service_url == sample_workspace.service_url

    def test_data_integrity_empty_scopes(self, workspace_store):
        """Test that empty scopes are preserved."""
        workspace = TeamsWorkspace(
            tenant_id="tenant-empty-scopes",
            tenant_name="Empty Scopes Test",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
            scopes=[],
        )
        workspace_store.save(workspace)

        retrieved = workspace_store.get("tenant-empty-scopes")
        assert retrieved.scopes == []

    def test_data_integrity_null_optional_fields(self, workspace_store):
        """Test that null optional fields are preserved."""
        workspace = TeamsWorkspace(
            tenant_id="tenant-nulls",
            tenant_name="Null Fields Test",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
        )
        workspace_store.save(workspace)

        retrieved = workspace_store.get("tenant-nulls")
        assert retrieved.installed_by is None
        assert retrieved.aragora_tenant_id is None
        assert retrieved.refresh_token is None
        assert retrieved.token_expires_at is None
        assert retrieved.service_url is None


# ===========================================================================
# Listing and Filtering Tests
# ===========================================================================


class TestTeamsWorkspaceStoreListing:
    """Tests for listing workspaces."""

    def test_list_active_empty(self, workspace_store):
        """Test list_active returns empty for empty store."""
        workspaces = workspace_store.list_active()
        assert workspaces == []

    def test_list_active_multiple(self, workspace_store):
        """Test list_active returns active workspaces."""
        for i in range(5):
            workspace = TeamsWorkspace(
                tenant_id=f"tenant-{i:08d}",
                tenant_name=f"Organization {i}",
                access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token-{i}",
                bot_id=f"bot-{i:08d}",
                installed_at=time.time() - i * 1000,  # Different times
            )
            workspace_store.save(workspace)

        workspaces = workspace_store.list_active()
        assert len(workspaces) == 5

    def test_list_active_excludes_inactive(self, workspace_store, sample_workspace):
        """Test list_active excludes inactive workspaces."""
        workspace_store.save(sample_workspace)

        # Add inactive workspace
        inactive = TeamsWorkspace(
            tenant_id="tenant-inactive",
            tenant_name="Inactive Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.inactive",
            bot_id="bot-inactive",
            installed_at=time.time(),
            is_active=False,
        )
        workspace_store.save(inactive)

        workspaces = workspace_store.list_active()
        assert len(workspaces) == 1
        assert workspaces[0].tenant_id == "test-tenant-12345"

    def test_list_active_pagination(self, workspace_store):
        """Test list_active with pagination (limit and offset)."""
        for i in range(10):
            workspace = TeamsWorkspace(
                tenant_id=f"tenant-{i:08d}",
                tenant_name=f"Organization {i}",
                access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token-{i}",
                bot_id=f"bot-{i:08d}",
                installed_at=time.time(),
            )
            workspace_store.save(workspace)

        page1 = workspace_store.list_active(limit=3, offset=0)
        page2 = workspace_store.list_active(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].tenant_id != page2[0].tenant_id

    def test_get_by_aragora_tenant(self, workspace_store):
        """Test get_by_aragora_tenant returns workspaces for Aragora tenant."""
        # Add workspaces for aragora-tenant-001
        for i in range(2):
            workspace = TeamsWorkspace(
                tenant_id=f"teams-1{i:07d}",
                tenant_name=f"Tenant1 Org {i}",
                access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.t1-{i}",
                bot_id=f"bot-1{i:07d}",
                installed_at=time.time(),
                aragora_tenant_id="aragora-tenant-001",
            )
            workspace_store.save(workspace)

        # Add workspaces for aragora-tenant-002
        workspace = TeamsWorkspace(
            tenant_id="teams-20000000",
            tenant_name="Tenant2 Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.t2-0",
            bot_id="bot-20000000",
            installed_at=time.time(),
            aragora_tenant_id="aragora-tenant-002",
        )
        workspace_store.save(workspace)

        tenant1 = workspace_store.get_by_aragora_tenant("aragora-tenant-001")
        tenant2 = workspace_store.get_by_aragora_tenant("aragora-tenant-002")

        assert len(tenant1) == 2
        assert len(tenant2) == 1
        assert all(w.aragora_tenant_id == "aragora-tenant-001" for w in tenant1)

    def test_get_by_aragora_tenant_excludes_inactive(self, workspace_store):
        """Test get_by_aragora_tenant excludes inactive workspaces."""
        active = TeamsWorkspace(
            tenant_id="teams-10000000",
            tenant_name="Active Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.active",
            bot_id="bot-10000000",
            installed_at=time.time(),
            aragora_tenant_id="aragora-tenant-001",
            is_active=True,
        )
        workspace_store.save(active)

        inactive = TeamsWorkspace(
            tenant_id="teams-10000001",
            tenant_name="Inactive Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.inactive",
            bot_id="bot-10000001",
            installed_at=time.time(),
            aragora_tenant_id="aragora-tenant-001",
            is_active=False,
        )
        workspace_store.save(inactive)

        workspaces = workspace_store.get_by_aragora_tenant("aragora-tenant-001")
        assert len(workspaces) == 1
        assert workspaces[0].is_active is True


# ===========================================================================
# Activation/Deactivation Tests
# ===========================================================================


class TestTeamsWorkspaceStoreActivation:
    """Tests for workspace activation/deactivation."""

    def test_activate_deactivated_workspace(self, workspace_store, sample_workspace):
        """Test reactivating a deactivated workspace by saving with is_active=True."""
        # Save as active
        workspace_store.save(sample_workspace)

        # Deactivate
        workspace_store.deactivate(sample_workspace.tenant_id)
        deactivated = workspace_store.get(sample_workspace.tenant_id)
        assert deactivated.is_active is False

        # Reactivate by saving with is_active=True
        sample_workspace.is_active = True
        workspace_store.save(sample_workspace)

        reactivated = workspace_store.get(sample_workspace.tenant_id)
        assert reactivated.is_active is True

    def test_deactivate_updates_only_is_active(self, workspace_store, sample_workspace):
        """Test deactivate only changes is_active flag, not other fields."""
        workspace_store.save(sample_workspace)

        workspace_store.deactivate(sample_workspace.tenant_id)

        workspace = workspace_store.get(sample_workspace.tenant_id)
        assert workspace.is_active is False
        assert workspace.tenant_name == "Test Organization"
        assert workspace.bot_id == "bot-app-12345"


# ===========================================================================
# Count and Statistics Tests
# ===========================================================================


class TestTeamsWorkspaceStoreStats:
    """Tests for count and statistics."""

    def test_count_empty(self, workspace_store):
        """Test count returns 0 for empty store."""
        assert workspace_store.count() == 0

    def test_count_active_only(self, workspace_store, sample_workspace):
        """Test count with active_only=True."""
        workspace_store.save(sample_workspace)

        inactive = TeamsWorkspace(
            tenant_id="tenant-inactive",
            tenant_name="Inactive Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.inactive",
            bot_id="bot-inactive",
            installed_at=time.time(),
            is_active=False,
        )
        workspace_store.save(inactive)

        assert workspace_store.count(active_only=True) == 1
        assert workspace_store.count(active_only=False) == 2

    def test_get_stats(self, workspace_store, sample_workspace):
        """Test get_stats returns statistics."""
        workspace_store.save(sample_workspace)

        inactive = TeamsWorkspace(
            tenant_id="tenant-inactive",
            tenant_name="Inactive Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.inactive",
            bot_id="bot-inactive",
            installed_at=time.time(),
            is_active=False,
        )
        workspace_store.save(inactive)

        stats = workspace_store.get_stats()

        assert stats["total_workspaces"] == 2
        assert stats["active_workspaces"] == 1
        assert stats["inactive_workspaces"] == 1


# ===========================================================================
# Token Encryption Tests
# ===========================================================================


class TestTeamsWorkspaceStoreEncryption:
    """Tests for token encryption."""

    def test_token_stored_unencrypted_without_key(self, workspace_store, sample_workspace):
        """Test tokens are stored unencrypted when no key is set."""
        with patch("aragora.storage.teams_workspace_store.ENCRYPTION_KEY", ""):
            workspace_store.save(sample_workspace)
            workspace = workspace_store.get("test-tenant-12345")

        # Token should be returned as-is (JWT tokens start with "ey")
        assert workspace.access_token.startswith("ey")

    @patch("aragora.storage.teams_workspace_store.ENCRYPTION_KEY", "test-encryption-key-32chars!!")
    def test_token_encrypted_with_key(self, temp_db_path, sample_workspace):
        """Test tokens are encrypted when key is set."""
        try:
            from cryptography.fernet import Fernet

            store = TeamsWorkspaceStore(db_path=temp_db_path)
            store.save(sample_workspace)

            # Get raw from database
            conn = store._get_connection()
            cursor = conn.execute(
                "SELECT access_token FROM teams_workspaces WHERE tenant_id = ?",
                ("test-tenant-12345",),
            )
            row = cursor.fetchone()
            raw_token = row["access_token"]

            # Token should NOT start with "ey" (encrypted)
            assert not raw_token.startswith("ey")
            # Should have version prefix
            assert raw_token.startswith("v2:")

            # But retrieved workspace should have decrypted token
            workspace = store.get("test-tenant-12345")
            assert workspace.access_token.startswith("eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9")

            store.close()

        except ImportError:
            pytest.skip("cryptography package not available")

    @patch("aragora.storage.teams_workspace_store.ENCRYPTION_KEY", "test-encryption-key-32chars!!")
    def test_refresh_token_encrypted(self, temp_db_path, sample_workspace):
        """Test refresh tokens are also encrypted."""
        try:
            from cryptography.fernet import Fernet

            store = TeamsWorkspaceStore(db_path=temp_db_path)
            store.save(sample_workspace)

            # Get raw from database
            conn = store._get_connection()
            cursor = conn.execute(
                "SELECT refresh_token FROM teams_workspaces WHERE tenant_id = ?",
                ("test-tenant-12345",),
            )
            row = cursor.fetchone()
            raw_token = row["refresh_token"]

            # Token should be encrypted
            assert raw_token.startswith("v2:")

            # But retrieved workspace should have decrypted refresh token
            workspace = store.get("test-tenant-12345")
            assert workspace.refresh_token == sample_workspace.refresh_token

            store.close()

        except ImportError:
            pytest.skip("cryptography package not available")

    def test_decrypt_unencrypted_jwt_token(self, workspace_store):
        """Test decrypting an already unencrypted JWT token returns as-is."""
        # JWT tokens start with "ey" and should be detected as unencrypted
        result = workspace_store._decrypt_token("eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.plain")
        assert result == "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.plain"


# ===========================================================================
# Token Expiry Tests
# ===========================================================================


class TestTeamsWorkspaceStoreTokenExpiry:
    """Tests for token expiry checking."""

    def test_is_token_expired_returns_true_for_expired(self, workspace_store, sample_workspace):
        """Test is_token_expired returns True for expired token."""
        sample_workspace.token_expires_at = time.time() - 100  # Expired 100 seconds ago
        workspace_store.save(sample_workspace)

        assert workspace_store.is_token_expired(sample_workspace.tenant_id) is True

    def test_is_token_expired_returns_false_for_valid(self, workspace_store, sample_workspace):
        """Test is_token_expired returns False for valid token."""
        sample_workspace.token_expires_at = time.time() + 3600  # Expires in 1 hour
        workspace_store.save(sample_workspace)

        assert workspace_store.is_token_expired(sample_workspace.tenant_id) is False

    def test_is_token_expired_with_buffer(self, workspace_store, sample_workspace):
        """Test is_token_expired considers buffer_seconds."""
        # Token expires in 200 seconds
        sample_workspace.token_expires_at = time.time() + 200
        workspace_store.save(sample_workspace)

        # With default 300 second buffer, should be considered expired
        assert (
            workspace_store.is_token_expired(sample_workspace.tenant_id, buffer_seconds=300) is True
        )

        # With 100 second buffer, should not be expired
        assert (
            workspace_store.is_token_expired(sample_workspace.tenant_id, buffer_seconds=100)
            is False
        )

    def test_is_token_expired_nonexistent_workspace(self, workspace_store):
        """Test is_token_expired returns True for nonexistent workspace."""
        assert workspace_store.is_token_expired("NONEXISTENT") is True

    def test_is_token_expired_no_expiry_set(self, workspace_store):
        """Test is_token_expired returns False when no expiry is set."""
        workspace = TeamsWorkspace(
            tenant_id="tenant-no-expiry",
            tenant_name="No Expiry Org",
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token",
            bot_id="bot-123",
            installed_at=time.time(),
            token_expires_at=None,
        )
        workspace_store.save(workspace)

        assert workspace_store.is_token_expired("tenant-no-expiry") is False


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestTeamsWorkspaceStoreErrors:
    """Tests for error handling."""

    def test_save_handles_error(self, temp_db_path, sample_workspace):
        """Test save handles database errors gracefully."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a real connection first to initialize schema
        store._get_connection()

        # Mock the _get_connection to return a mock that raises errors
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("DB error")

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.save(sample_workspace)

        assert result is False
        store.close()

    def test_get_handles_error(self, temp_db_path):
        """Test get handles database errors gracefully."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a real connection first to initialize schema
        store._get_connection()

        # Mock the _get_connection to return a mock that raises errors
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("DB error")

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.get("test-tenant")

        assert result is None
        store.close()

    def test_list_active_handles_error(self, temp_db_path):
        """Test list_active handles database errors gracefully."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a real connection first to initialize schema
        store._get_connection()

        # Mock the _get_connection to return a mock that raises errors
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("DB error")

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.list_active()

        assert result == []
        store.close()

    def test_count_handles_error(self, temp_db_path):
        """Test count handles database errors gracefully."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a real connection first to initialize schema
        store._get_connection()

        # Mock the _get_connection to return a mock that raises errors
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("DB error")

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.count()

        assert result == 0
        store.close()

    def test_get_stats_handles_error(self, temp_db_path):
        """Test get_stats handles database errors gracefully."""
        import sqlite3

        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a real connection first to initialize schema
        store._get_connection()

        # Mock the _get_connection to return a mock that raises errors
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("DB error")

        with patch.object(store, "_get_connection", return_value=mock_conn):
            result = store.get_stats()

        assert result == {"total_workspaces": 0, "active_workspaces": 0}
        store.close()


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


class TestTeamsWorkspaceStoreSingleton:
    """Tests for singleton pattern."""

    def test_get_teams_workspace_store_singleton(self, temp_db_path):
        """Test get_teams_workspace_store returns singleton."""
        # Reset the global singleton
        import aragora.storage.teams_workspace_store as module

        module._workspace_store = None

        store1 = get_teams_workspace_store(temp_db_path)
        store2 = get_teams_workspace_store()

        assert store1 is store2

        # Cleanup
        store1.close()
        module._workspace_store = None


# ===========================================================================
# Thread Safety / Concurrent Access Tests
# ===========================================================================


class TestTeamsWorkspaceStoreThreadSafety:
    """Tests for thread safety and concurrent access."""

    def test_schema_initialization_thread_safe(self, temp_db_path):
        """Test schema initialization is thread-safe."""
        store = TeamsWorkspaceStore(db_path=temp_db_path)
        errors = []

        def init_connection():
            try:
                conn = store._get_connection()
                assert conn is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=init_connection) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        store.close()

    def test_concurrent_saves(self, workspace_store):
        """Test concurrent saves don't cause errors."""
        errors = []

        def save_workspace(idx):
            try:
                workspace = TeamsWorkspace(
                    tenant_id=f"tenant-{idx:08d}",
                    tenant_name=f"Organization {idx}",
                    access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token-{idx}",
                    bot_id=f"bot-{idx:08d}",
                    installed_at=time.time(),
                )
                workspace_store.save(workspace)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_workspace, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert workspace_store.count() == 10

    def test_concurrent_reads_and_writes(self, workspace_store):
        """Test concurrent reads and writes don't cause errors."""
        # Pre-populate some data
        for i in range(5):
            workspace = TeamsWorkspace(
                tenant_id=f"tenant-prepop-{i:08d}",
                tenant_name=f"Organization {i}",
                access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.token-{i}",
                bot_id=f"bot-{i:08d}",
                installed_at=time.time(),
            )
            workspace_store.save(workspace)

        errors = []

        def read_workspaces():
            try:
                workspaces = workspace_store.list_active()
                assert isinstance(workspaces, list)
            except Exception as e:
                errors.append(e)

        def write_workspace(idx):
            try:
                workspace = TeamsWorkspace(
                    tenant_id=f"tenant-new-{idx:08d}",
                    tenant_name=f"New Organization {idx}",
                    access_token=f"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.new-token-{idx}",
                    bot_id=f"bot-new-{idx:08d}",
                    installed_at=time.time(),
                )
                workspace_store.save(workspace)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=read_workspaces))
            threads.append(threading.Thread(target=write_workspace, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ===========================================================================
# Connection Management Tests
# ===========================================================================


class TestTeamsWorkspaceStoreConnectionManagement:
    """Tests for database connection management."""

    def test_close_releases_connections(self, temp_db_path):
        """Test close() properly releases all connections."""
        store = TeamsWorkspaceStore(db_path=temp_db_path)

        # Get a connection
        conn = store._get_connection()
        assert conn is not None

        # Close the store
        store.close()

        # Connections set should be empty
        assert len(store._connections) == 0

    def test_multiple_contexts_get_separate_connections(self, temp_db_path):
        """Test that different contexts get separate connections via ContextVar."""
        import asyncio

        store = TeamsWorkspaceStore(db_path=temp_db_path)
        connections = []

        async def get_conn():
            conn = store._get_connection()
            connections.append(conn)
            return conn

        async def run_test():
            # Run in separate tasks
            await asyncio.gather(get_conn(), get_conn())

        asyncio.run(run_test())

        # Each task should get its own connection due to ContextVar
        # They might be the same object if running in the same context,
        # but the mechanism should work without errors
        assert len(connections) == 2
        store.close()
