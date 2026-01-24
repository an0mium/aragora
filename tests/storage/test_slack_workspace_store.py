"""
Tests for SlackWorkspaceStore - Slack OAuth token management.

Tests cover:
- CRUD operations (save, get, delete, deactivate)
- Listing with filters and pagination
- Token encryption/decryption
- Multi-tenant workspace isolation
- Statistics
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.slack_workspace_store import (
    SlackWorkspace,
    SlackWorkspaceStore,
    get_slack_workspace_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_slack_workspaces.db")


@pytest.fixture
def workspace_store(temp_db_path):
    """Create a workspace store for testing."""
    return SlackWorkspaceStore(db_path=temp_db_path)


@pytest.fixture
def sample_workspace():
    """Create a sample workspace."""
    return SlackWorkspace(
        workspace_id="T12345678",
        workspace_name="Test Workspace",
        access_token="xoxb-test-token-12345",
        bot_user_id="U87654321",
        installed_at=time.time(),
        installed_by="U11111111",
        scopes=["channels:history", "chat:write", "commands"],
        tenant_id="tenant-001",
        is_active=True,
    )


# ===========================================================================
# SlackWorkspace Dataclass Tests
# ===========================================================================


class TestSlackWorkspace:
    """Tests for SlackWorkspace dataclass."""

    def test_to_dict_excludes_token(self, sample_workspace):
        """Test to_dict does not include access_token."""
        result = sample_workspace.to_dict()

        assert "access_token" not in result
        assert result["workspace_id"] == "T12345678"
        assert result["workspace_name"] == "Test Workspace"
        assert result["bot_user_id"] == "U87654321"

    def test_to_dict_includes_scopes(self, sample_workspace):
        """Test to_dict includes scopes list."""
        result = sample_workspace.to_dict()

        assert result["scopes"] == ["channels:history", "chat:write", "commands"]

    def test_to_dict_includes_iso_timestamp(self, sample_workspace):
        """Test to_dict includes ISO formatted timestamp."""
        result = sample_workspace.to_dict()

        assert "installed_at_iso" in result
        assert "T" in result["installed_at_iso"]  # ISO format

    def test_default_scopes_empty(self):
        """Test default scopes is empty list."""
        workspace = SlackWorkspace(
            workspace_id="T123",
            workspace_name="Test",
            access_token="xoxb-token",
            bot_user_id="U123",
            installed_at=time.time(),
        )

        assert workspace.scopes == []

    def test_default_is_active(self):
        """Test default is_active is True."""
        workspace = SlackWorkspace(
            workspace_id="T123",
            workspace_name="Test",
            access_token="xoxb-token",
            bot_user_id="U123",
            installed_at=time.time(),
        )

        assert workspace.is_active is True


# ===========================================================================
# SlackWorkspaceStore CRUD Tests
# ===========================================================================


class TestSlackWorkspaceStoreCRUD:
    """Tests for SlackWorkspaceStore CRUD operations."""

    def test_save_and_get(self, workspace_store, sample_workspace):
        """Test save and retrieve a workspace."""
        saved = workspace_store.save(sample_workspace)
        assert saved is True

        workspace = workspace_store.get("T12345678")
        assert workspace is not None
        assert workspace.workspace_id == "T12345678"
        assert workspace.workspace_name == "Test Workspace"
        assert workspace.access_token == "xoxb-test-token-12345"

    def test_get_nonexistent(self, workspace_store):
        """Test get returns None for nonexistent workspace."""
        result = workspace_store.get("NONEXISTENT")
        assert result is None

    def test_save_updates_existing(self, workspace_store, sample_workspace):
        """Test save updates existing workspace (upsert)."""
        workspace_store.save(sample_workspace)

        # Update workspace
        sample_workspace.workspace_name = "Updated Workspace"
        sample_workspace.scopes = ["new:scope"]
        workspace_store.save(sample_workspace)

        workspace = workspace_store.get("T12345678")
        assert workspace.workspace_name == "Updated Workspace"
        assert workspace.scopes == ["new:scope"]

    def test_deactivate(self, workspace_store, sample_workspace):
        """Test deactivate marks workspace inactive."""
        workspace_store.save(sample_workspace)

        result = workspace_store.deactivate("T12345678")
        assert result is True

        workspace = workspace_store.get("T12345678")
        assert workspace.is_active is False

    def test_deactivate_nonexistent(self, workspace_store):
        """Test deactivate for nonexistent returns True (no-op)."""
        result = workspace_store.deactivate("NONEXISTENT")
        assert result is True  # SQLite UPDATE succeeds even if no rows

    def test_delete(self, workspace_store, sample_workspace):
        """Test delete removes workspace permanently."""
        workspace_store.save(sample_workspace)
        assert workspace_store.get("T12345678") is not None

        result = workspace_store.delete("T12345678")
        assert result is True

        assert workspace_store.get("T12345678") is None

    def test_delete_nonexistent(self, workspace_store):
        """Test delete for nonexistent returns True (no-op)."""
        result = workspace_store.delete("NONEXISTENT")
        assert result is True


# ===========================================================================
# Listing and Filtering Tests
# ===========================================================================


class TestSlackWorkspaceStoreListing:
    """Tests for listing workspaces."""

    def test_list_active_empty(self, workspace_store):
        """Test list_active returns empty for empty store."""
        workspaces = workspace_store.list_active()
        assert workspaces == []

    def test_list_active_multiple(self, workspace_store):
        """Test list_active returns active workspaces."""
        for i in range(5):
            workspace = SlackWorkspace(
                workspace_id=f"T{i:08d}",
                workspace_name=f"Workspace {i}",
                access_token=f"xoxb-token-{i}",
                bot_user_id=f"U{i:08d}",
                installed_at=time.time() - i * 1000,  # Different times
            )
            workspace_store.save(workspace)

        workspaces = workspace_store.list_active()
        assert len(workspaces) == 5

    def test_list_active_excludes_inactive(self, workspace_store, sample_workspace):
        """Test list_active excludes inactive workspaces."""
        workspace_store.save(sample_workspace)

        # Add inactive workspace
        inactive = SlackWorkspace(
            workspace_id="T99999999",
            workspace_name="Inactive",
            access_token="xoxb-inactive",
            bot_user_id="U99999999",
            installed_at=time.time(),
            is_active=False,
        )
        workspace_store.save(inactive)

        workspaces = workspace_store.list_active()
        assert len(workspaces) == 1
        assert workspaces[0].workspace_id == "T12345678"

    def test_list_active_pagination(self, workspace_store):
        """Test list_active with pagination."""
        for i in range(10):
            workspace = SlackWorkspace(
                workspace_id=f"T{i:08d}",
                workspace_name=f"Workspace {i}",
                access_token=f"xoxb-token-{i}",
                bot_user_id=f"U{i:08d}",
                installed_at=time.time(),
            )
            workspace_store.save(workspace)

        page1 = workspace_store.list_active(limit=3, offset=0)
        page2 = workspace_store.list_active(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].workspace_id != page2[0].workspace_id

    def test_get_by_tenant(self, workspace_store):
        """Test get_by_tenant returns workspaces for tenant."""
        # Add workspaces for tenant-001
        for i in range(2):
            workspace = SlackWorkspace(
                workspace_id=f"T1{i:07d}",
                workspace_name=f"Tenant1 Workspace {i}",
                access_token=f"xoxb-t1-{i}",
                bot_user_id=f"U1{i:07d}",
                installed_at=time.time(),
                tenant_id="tenant-001",
            )
            workspace_store.save(workspace)

        # Add workspaces for tenant-002
        workspace = SlackWorkspace(
            workspace_id="T20000000",
            workspace_name="Tenant2 Workspace",
            access_token="xoxb-t2-0",
            bot_user_id="U20000000",
            installed_at=time.time(),
            tenant_id="tenant-002",
        )
        workspace_store.save(workspace)

        tenant1 = workspace_store.get_by_tenant("tenant-001")
        tenant2 = workspace_store.get_by_tenant("tenant-002")

        assert len(tenant1) == 2
        assert len(tenant2) == 1
        assert all(w.tenant_id == "tenant-001" for w in tenant1)

    def test_get_by_tenant_excludes_inactive(self, workspace_store):
        """Test get_by_tenant excludes inactive workspaces."""
        active = SlackWorkspace(
            workspace_id="T10000000",
            workspace_name="Active",
            access_token="xoxb-active",
            bot_user_id="U10000000",
            installed_at=time.time(),
            tenant_id="tenant-001",
            is_active=True,
        )
        workspace_store.save(active)

        inactive = SlackWorkspace(
            workspace_id="T10000001",
            workspace_name="Inactive",
            access_token="xoxb-inactive",
            bot_user_id="U10000001",
            installed_at=time.time(),
            tenant_id="tenant-001",
            is_active=False,
        )
        workspace_store.save(inactive)

        workspaces = workspace_store.get_by_tenant("tenant-001")
        assert len(workspaces) == 1
        assert workspaces[0].is_active is True


# ===========================================================================
# Count and Statistics Tests
# ===========================================================================


class TestSlackWorkspaceStoreStats:
    """Tests for count and statistics."""

    def test_count_empty(self, workspace_store):
        """Test count returns 0 for empty store."""
        assert workspace_store.count() == 0

    def test_count_active_only(self, workspace_store, sample_workspace):
        """Test count with active_only=True."""
        workspace_store.save(sample_workspace)

        inactive = SlackWorkspace(
            workspace_id="T99999999",
            workspace_name="Inactive",
            access_token="xoxb-inactive",
            bot_user_id="U99999999",
            installed_at=time.time(),
            is_active=False,
        )
        workspace_store.save(inactive)

        assert workspace_store.count(active_only=True) == 1
        assert workspace_store.count(active_only=False) == 2

    def test_get_stats(self, workspace_store, sample_workspace):
        """Test get_stats returns statistics."""
        workspace_store.save(sample_workspace)

        inactive = SlackWorkspace(
            workspace_id="T99999999",
            workspace_name="Inactive",
            access_token="xoxb-inactive",
            bot_user_id="U99999999",
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


class TestSlackWorkspaceStoreEncryption:
    """Tests for token encryption."""

    def test_token_stored_unencrypted_without_key(self, workspace_store, sample_workspace):
        """Test tokens are stored unencrypted when no key is set."""
        with patch("aragora.storage.slack_workspace_store.ENCRYPTION_KEY", ""):
            workspace_store.save(sample_workspace)
            workspace = workspace_store.get("T12345678")

        # Token should be returned as-is
        assert workspace.access_token.startswith("xoxb-")

    @patch("aragora.storage.slack_workspace_store.ENCRYPTION_KEY", "test-encryption-key-32chars!!")
    def test_token_encrypted_with_key(self, temp_db_path, sample_workspace):
        """Test tokens are encrypted when key is set."""
        try:
            from cryptography.fernet import Fernet

            store = SlackWorkspaceStore(db_path=temp_db_path)
            store.save(sample_workspace)

            # Get raw from database
            conn = store._get_connection()
            cursor = conn.execute(
                "SELECT access_token FROM slack_workspaces WHERE workspace_id = ?",
                ("T12345678",),
            )
            row = cursor.fetchone()
            raw_token = row["access_token"]

            # Token should NOT start with xoxb- (encrypted)
            assert not raw_token.startswith("xoxb-")

            # But retrieved workspace should have decrypted token
            workspace = store.get("T12345678")
            assert workspace.access_token == "xoxb-test-token-12345"

        except ImportError:
            pytest.skip("cryptography package not available")

    def test_decrypt_unencrypted_token(self, workspace_store):
        """Test decrypting an already unencrypted token returns as-is."""
        # Simulate token that's already unencrypted
        result = workspace_store._decrypt_token("xoxb-already-plain")
        assert result == "xoxb-already-plain"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestSlackWorkspaceStoreErrors:
    """Tests for error handling."""

    def test_save_handles_error(self, workspace_store, sample_workspace):
        """Test save handles database errors gracefully."""
        with patch.object(workspace_store, "_get_connection", side_effect=Exception("DB error")):
            result = workspace_store.save(sample_workspace)

        assert result is False

    def test_get_handles_error(self, workspace_store):
        """Test get handles database errors gracefully."""
        with patch.object(workspace_store, "_get_connection", side_effect=Exception("DB error")):
            result = workspace_store.get("T12345678")

        assert result is None

    def test_list_active_handles_error(self, workspace_store):
        """Test list_active handles database errors gracefully."""
        with patch.object(workspace_store, "_get_connection", side_effect=Exception("DB error")):
            result = workspace_store.list_active()

        assert result == []


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


class TestSlackWorkspaceStoreSingleton:
    """Tests for singleton pattern."""

    def test_get_slack_workspace_store_singleton(self, temp_db_path):
        """Test get_slack_workspace_store returns singleton."""
        # Reset the global singleton
        import aragora.storage.slack_workspace_store as module

        module._workspace_store = None

        store1 = get_slack_workspace_store(temp_db_path)
        store2 = get_slack_workspace_store()

        assert store1 is store2

        # Cleanup
        module._workspace_store = None


# ===========================================================================
# Thread Safety Tests
# ===========================================================================


class TestSlackWorkspaceStoreThreadSafety:
    """Tests for thread safety."""

    def test_schema_initialization_thread_safe(self, temp_db_path):
        """Test schema initialization is thread-safe."""
        import threading

        store = SlackWorkspaceStore(db_path=temp_db_path)
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

    def test_concurrent_saves(self, workspace_store):
        """Test concurrent saves don't cause errors."""
        import threading

        errors = []

        def save_workspace(idx):
            try:
                workspace = SlackWorkspace(
                    workspace_id=f"T{idx:08d}",
                    workspace_name=f"Workspace {idx}",
                    access_token=f"xoxb-token-{idx}",
                    bot_user_id=f"U{idx:08d}",
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
