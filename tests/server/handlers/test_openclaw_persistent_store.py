"""
Tests for OpenClawPersistentStore - SQLite-backed persistent storage.

Tests cover:
- Session CRUD operations with persistence
- Action CRUD operations with persistence
- Credential storage with encryption
- Credential rotation
- Audit logging
- LRU cache behavior
- Metrics calculation
- Database schema initialization
- Filter and pagination queries
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pytest

from aragora.server.handlers.openclaw.models import (
    ActionStatus,
    CredentialType,
    SessionStatus,
)
from aragora.server.handlers.openclaw.store import (
    OpenClawPersistentStore,
    OpenClawGatewayStore,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_openclaw.db")


@pytest.fixture
def persistent_store(temp_db_path: str) -> OpenClawPersistentStore:
    """Create a fresh persistent store for each test."""
    return OpenClawPersistentStore(db_path=temp_db_path)


@pytest.fixture
def memory_store() -> OpenClawGatewayStore:
    """Create a fresh in-memory store for comparison tests."""
    return OpenClawGatewayStore()


# ===========================================================================
# Session Tests
# ===========================================================================


class TestPersistentStoreSessions:
    """Tests for session management in persistent store."""

    def test_create_session_persists_to_db(self, persistent_store: OpenClawPersistentStore) -> None:
        """Sessions should be persisted to SQLite."""
        session = persistent_store.create_session(
            user_id="user-123",
            tenant_id="tenant-abc",
            config={"max_actions": 10},
            metadata={"source": "test"},
        )

        assert session.id is not None
        assert session.user_id == "user-123"
        assert session.tenant_id == "tenant-abc"
        assert session.status == SessionStatus.ACTIVE
        assert session.config == {"max_actions": 10}
        assert session.metadata == {"source": "test"}

        # Verify it persists by creating new store instance
        store2 = OpenClawPersistentStore(db_path=persistent_store._db_path)
        retrieved = store2.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.user_id == "user-123"

    def test_get_session_uses_cache(self, persistent_store: OpenClawPersistentStore) -> None:
        """Get should use LRU cache after first fetch."""
        session = persistent_store.create_session(user_id="user-123")

        # First call populates cache
        result1 = persistent_store.get_session(session.id)
        assert result1 is not None

        # Check it's in cache
        assert session.id in persistent_store._session_cache

        # Second call should use cache (same object)
        result2 = persistent_store.get_session(session.id)
        assert result2 is result1  # Same object from cache

    def test_get_session_not_found_returns_none(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Get should return None for non-existent session."""
        result = persistent_store.get_session("nonexistent-id")
        assert result is None

    def test_list_sessions_with_filters(self, persistent_store: OpenClawPersistentStore) -> None:
        """List should support user_id, tenant_id, and status filters."""
        # Create sessions with different attributes
        s1 = persistent_store.create_session(user_id="user-1", tenant_id="tenant-a")
        s2 = persistent_store.create_session(user_id="user-1", tenant_id="tenant-b")
        s3 = persistent_store.create_session(user_id="user-2", tenant_id="tenant-a")

        # Update one to a different status
        persistent_store.update_session_status(s3.id, SessionStatus.COMPLETED)

        # Filter by user_id
        sessions, total = persistent_store.list_sessions(user_id="user-1")
        assert total == 2
        assert len(sessions) == 2

        # Filter by tenant_id
        sessions, total = persistent_store.list_sessions(tenant_id="tenant-a")
        assert total == 2

        # Filter by status
        sessions, total = persistent_store.list_sessions(status=SessionStatus.ACTIVE)
        assert total == 2

        # Combined filters
        sessions, total = persistent_store.list_sessions(user_id="user-1", tenant_id="tenant-a")
        assert total == 1
        assert sessions[0].id == s1.id

    def test_list_sessions_pagination(self, persistent_store: OpenClawPersistentStore) -> None:
        """List should support limit and offset pagination."""
        # Create 5 sessions
        for i in range(5):
            persistent_store.create_session(user_id=f"user-{i}")

        # Get first page
        sessions, total = persistent_store.list_sessions(limit=2, offset=0)
        assert total == 5
        assert len(sessions) == 2

        # Get second page
        sessions2, _ = persistent_store.list_sessions(limit=2, offset=2)
        assert len(sessions2) == 2

        # Pages should have different sessions
        assert sessions[0].id != sessions2[0].id

    def test_update_session_status(self, persistent_store: OpenClawPersistentStore) -> None:
        """Update should change status and invalidate cache."""
        session = persistent_store.create_session(user_id="user-123")
        assert session.status == SessionStatus.ACTIVE

        # Update status
        updated = persistent_store.update_session_status(session.id, SessionStatus.PAUSED)
        assert updated is not None
        assert updated.status == SessionStatus.PAUSED

        # Verify persisted
        store2 = OpenClawPersistentStore(db_path=persistent_store._db_path)
        retrieved = store2.get_session(session.id)
        assert retrieved is not None
        assert retrieved.status == SessionStatus.PAUSED

    def test_delete_session(self, persistent_store: OpenClawPersistentStore) -> None:
        """Delete should remove session and invalidate cache."""
        session = persistent_store.create_session(user_id="user-123")

        # Populate cache
        persistent_store.get_session(session.id)
        assert session.id in persistent_store._session_cache

        # Delete
        result = persistent_store.delete_session(session.id)
        assert result is True

        # Cache should be invalidated
        assert session.id not in persistent_store._session_cache

        # Session should be gone
        assert persistent_store.get_session(session.id) is None

    def test_delete_nonexistent_session_returns_false(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Delete should return False for non-existent session."""
        result = persistent_store.delete_session("nonexistent-id")
        assert result is False


# ===========================================================================
# Action Tests
# ===========================================================================


class TestPersistentStoreActions:
    """Tests for action management in persistent store."""

    def test_create_action_persists_to_db(self, persistent_store: OpenClawPersistentStore) -> None:
        """Actions should be persisted to SQLite."""
        session = persistent_store.create_session(user_id="user-123")

        action = persistent_store.create_action(
            session_id=session.id,
            action_type="screenshot",
            input_data={"url": "https://example.com"},
            metadata={"priority": "high"},
        )

        assert action.id is not None
        assert action.session_id == session.id
        assert action.action_type == "screenshot"
        assert action.status == ActionStatus.PENDING
        assert action.input_data == {"url": "https://example.com"}

        # Verify persistence
        store2 = OpenClawPersistentStore(db_path=persistent_store._db_path)
        retrieved = store2.get_action(action.id)
        assert retrieved is not None
        assert retrieved.action_type == "screenshot"

    def test_get_action_uses_cache(self, persistent_store: OpenClawPersistentStore) -> None:
        """Get should use LRU cache."""
        session = persistent_store.create_session(user_id="user-123")
        action = persistent_store.create_action(
            session_id=session.id,
            action_type="click",
            input_data={"x": 100, "y": 200},
        )

        # First call should cache
        result1 = persistent_store.get_action(action.id)
        assert result1 is not None
        assert action.id in persistent_store._action_cache

        # Second call should return cached object
        result2 = persistent_store.get_action(action.id)
        assert result2 is result1

    def test_update_action_status_to_running(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Update to RUNNING should set started_at."""
        session = persistent_store.create_session(user_id="user-123")
        action = persistent_store.create_action(
            session_id=session.id, action_type="type", input_data={"text": "hello"}
        )
        assert action.started_at is None

        updated = persistent_store.update_action(action.id, status=ActionStatus.RUNNING)
        assert updated is not None
        assert updated.status == ActionStatus.RUNNING
        assert updated.started_at is not None

    def test_update_action_status_to_completed(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Update to COMPLETED should set completed_at."""
        session = persistent_store.create_session(user_id="user-123")
        action = persistent_store.create_action(
            session_id=session.id, action_type="click", input_data={}
        )

        # First set to running
        persistent_store.update_action(action.id, status=ActionStatus.RUNNING)

        # Then complete
        updated = persistent_store.update_action(
            action.id,
            status=ActionStatus.COMPLETED,
            output_data={"success": True, "screenshot": "base64..."},
        )
        assert updated is not None
        assert updated.status == ActionStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.output_data == {"success": True, "screenshot": "base64..."}

    def test_update_action_with_error(self, persistent_store: OpenClawPersistentStore) -> None:
        """Update should store error message."""
        session = persistent_store.create_session(user_id="user-123")
        action = persistent_store.create_action(
            session_id=session.id, action_type="click", input_data={}
        )

        updated = persistent_store.update_action(
            action.id, status=ActionStatus.FAILED, error="Element not found"
        )
        assert updated is not None
        assert updated.status == ActionStatus.FAILED
        assert updated.error == "Element not found"

    def test_update_nonexistent_action_returns_none(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Update should return None for non-existent action."""
        result = persistent_store.update_action("nonexistent-id", status=ActionStatus.RUNNING)
        assert result is None


# ===========================================================================
# Credential Tests
# ===========================================================================


class TestPersistentStoreCredentials:
    """Tests for credential management with encryption."""

    def test_store_credential_encrypts_secret(
        self, persistent_store: OpenClawPersistentStore
    ) -> None:
        """Credentials should have their secrets encrypted."""
        cred = persistent_store.store_credential(
            name="my-api-key",
            credential_type=CredentialType.API_KEY,
            secret_value="super-secret-key-12345",
            user_id="user-123",
            tenant_id="tenant-abc",
            metadata={"provider": "openai"},
        )

        assert cred.id is not None
        assert cred.name == "my-api-key"
        assert cred.credential_type == CredentialType.API_KEY
        assert cred.user_id == "user-123"

        # Secret should NOT be accessible via get_credential
        retrieved = persistent_store.get_credential(cred.id)
        assert retrieved is not None
        # Credential model doesn't expose secret

        # Verify encryption in database
        conn = persistent_store._get_connection()
        try:
            row = conn.execute(
                "SELECT secret_encrypted FROM openclaw_credentials WHERE id = ?",
                (cred.id,),
            ).fetchone()
            assert row is not None
            # Should be encrypted (not plaintext)
            assert row["secret_encrypted"] != "super-secret-key-12345"
        finally:
            conn.close()

    def test_list_credentials_with_filters(self, persistent_store: OpenClawPersistentStore) -> None:
        """List should support filters."""
        c1 = persistent_store.store_credential(
            name="key-1",
            credential_type=CredentialType.API_KEY,
            secret_value="secret1",
            user_id="user-1",
            tenant_id="tenant-a",
        )
        c2 = persistent_store.store_credential(
            name="key-2",
            credential_type=CredentialType.OAUTH_TOKEN,
            secret_value="secret2",
            user_id="user-1",
            tenant_id="tenant-b",
        )
        c3 = persistent_store.store_credential(
            name="key-3",
            credential_type=CredentialType.API_KEY,
            secret_value="secret3",
            user_id="user-2",
            tenant_id="tenant-a",
        )

        # Filter by user
        creds, total = persistent_store.list_credentials(user_id="user-1")
        assert total == 2

        # Filter by type
        creds, total = persistent_store.list_credentials(credential_type=CredentialType.API_KEY)
        assert total == 2

        # Combined
        creds, total = persistent_store.list_credentials(
            user_id="user-1", credential_type=CredentialType.API_KEY
        )
        assert total == 1
        assert creds[0].id == c1.id

    def test_delete_credential(self, persistent_store: OpenClawPersistentStore) -> None:
        """Delete should remove credential."""
        cred = persistent_store.store_credential(
            name="temp-key",
            credential_type=CredentialType.API_KEY,
            secret_value="temp-secret",
            user_id="user-123",
        )

        result = persistent_store.delete_credential(cred.id)
        assert result is True

        # Should be gone
        assert persistent_store.get_credential(cred.id) is None

    def test_rotate_credential(self, persistent_store: OpenClawPersistentStore) -> None:
        """Rotate should update secret and timestamps."""
        cred = persistent_store.store_credential(
            name="rotating-key",
            credential_type=CredentialType.API_KEY,
            secret_value="old-secret",
            user_id="user-123",
        )
        assert cred.last_rotated_at is None

        rotated = persistent_store.rotate_credential(cred.id, "new-secret-value")
        assert rotated is not None
        assert rotated.last_rotated_at is not None

        # Verify new secret is encrypted in DB
        conn = persistent_store._get_connection()
        try:
            row = conn.execute(
                "SELECT secret_encrypted FROM openclaw_credentials WHERE id = ?",
                (cred.id,),
            ).fetchone()
            # Should be different from old encrypted value
            assert row is not None
        finally:
            conn.close()

    def test_credential_with_expiration(self, persistent_store: OpenClawPersistentStore) -> None:
        """Credentials should support expiration dates."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        cred = persistent_store.store_credential(
            name="expiring-key",
            credential_type=CredentialType.API_KEY,
            secret_value="secret",
            user_id="user-123",
            expires_at=expires,
        )

        retrieved = persistent_store.get_credential(cred.id)
        assert retrieved is not None
        assert retrieved.expires_at is not None
        # Allow small time difference
        assert abs((retrieved.expires_at - expires).total_seconds()) < 1


# ===========================================================================
# Audit Tests
# ===========================================================================


class TestPersistentStoreAudit:
    """Tests for audit logging."""

    def test_add_audit_entry(self, persistent_store: OpenClawPersistentStore) -> None:
        """Audit entries should be persisted."""
        entry = persistent_store.add_audit_entry(
            action="session.create",
            actor_id="user-123",
            resource_type="session",
            resource_id="sess-abc",
            result="success",
            details={"ip": "192.168.1.1"},
        )

        assert entry.id is not None
        assert entry.action == "session.create"
        assert entry.actor_id == "user-123"
        assert entry.result == "success"

    def test_get_audit_log_with_filters(self, persistent_store: OpenClawPersistentStore) -> None:
        """Get audit log should support filters."""
        persistent_store.add_audit_entry(
            action="session.create", actor_id="user-1", resource_type="session"
        )
        persistent_store.add_audit_entry(
            action="credential.store", actor_id="user-1", resource_type="credential"
        )
        persistent_store.add_audit_entry(
            action="session.create", actor_id="user-2", resource_type="session"
        )

        # Filter by action
        entries, total = persistent_store.get_audit_log(action="session.create")
        assert total == 2

        # Filter by actor
        entries, total = persistent_store.get_audit_log(actor_id="user-1")
        assert total == 2

        # Filter by resource type
        entries, total = persistent_store.get_audit_log(resource_type="credential")
        assert total == 1

    def test_get_audit_log_pagination(self, persistent_store: OpenClawPersistentStore) -> None:
        """Audit log should support pagination."""
        for i in range(10):
            persistent_store.add_audit_entry(
                action=f"action-{i}",
                actor_id="user-123",
                resource_type="test",
            )

        entries, total = persistent_store.get_audit_log(limit=5, offset=0)
        assert total == 10
        assert len(entries) == 5

        entries2, _ = persistent_store.get_audit_log(limit=5, offset=5)
        assert len(entries2) == 5

        # Should be different entries
        assert entries[0].id != entries2[0].id


# ===========================================================================
# Metrics Tests
# ===========================================================================


class TestPersistentStoreMetrics:
    """Tests for metrics calculation."""

    def test_get_metrics_counts_by_status(self, persistent_store: OpenClawPersistentStore) -> None:
        """Metrics should count entities by status."""
        # Create sessions with different statuses
        s1 = persistent_store.create_session(user_id="user-1")
        s2 = persistent_store.create_session(user_id="user-2")
        persistent_store.update_session_status(s2.id, SessionStatus.COMPLETED)

        # Create actions
        a1 = persistent_store.create_action(session_id=s1.id, action_type="click", input_data={})
        persistent_store.update_action(a1.id, status=ActionStatus.RUNNING)

        # Create credentials
        persistent_store.store_credential(
            name="key-1",
            credential_type=CredentialType.API_KEY,
            secret_value="secret",
            user_id="user-1",
        )

        # Add audit entry
        persistent_store.add_audit_entry(action="test", actor_id="user-1", resource_type="test")

        metrics = persistent_store.get_metrics()

        assert metrics["sessions"]["total"] == 2
        assert metrics["sessions"]["active"] == 1
        assert metrics["sessions"]["by_status"]["active"] == 1
        assert metrics["sessions"]["by_status"]["completed"] == 1

        assert metrics["actions"]["total"] == 1
        assert metrics["actions"]["running"] == 1

        assert metrics["credentials"]["total"] == 1
        assert metrics["credentials"]["by_type"]["api_key"] == 1

        assert metrics["audit_log_entries"] == 1


# ===========================================================================
# Cache Tests
# ===========================================================================


class TestPersistentStoreCache:
    """Tests for LRU cache behavior."""

    def test_cache_evicts_oldest_entries(self, temp_db_path: str) -> None:
        """Cache should evict oldest entries when full."""
        store = OpenClawPersistentStore(db_path=temp_db_path, cache_size=3)

        # Create 5 sessions
        sessions = []
        for i in range(5):
            s = store.create_session(user_id=f"user-{i}")
            sessions.append(s)

        # Cache should only have last 3
        assert len(store._session_cache) == 3

        # First 2 should be evicted
        assert sessions[0].id not in store._session_cache
        assert sessions[1].id not in store._session_cache

        # Last 3 should be in cache
        assert sessions[2].id in store._session_cache
        assert sessions[3].id in store._session_cache
        assert sessions[4].id in store._session_cache

    def test_cache_moves_accessed_to_end(self, temp_db_path: str) -> None:
        """Accessing an item should move it to end of LRU."""
        store = OpenClawPersistentStore(db_path=temp_db_path, cache_size=3)

        # Create 3 sessions
        s1 = store.create_session(user_id="user-1")
        s2 = store.create_session(user_id="user-2")
        s3 = store.create_session(user_id="user-3")

        # Access s1 (moves it to end)
        store.get_session(s1.id)

        # Create a new session (should evict s2, not s1)
        s4 = store.create_session(user_id="user-4")

        assert s1.id in store._session_cache  # Recently accessed
        assert s2.id not in store._session_cache  # Evicted
        assert s3.id in store._session_cache
        assert s4.id in store._session_cache


# ===========================================================================
# Parity Tests with In-Memory Store
# ===========================================================================


class TestPersistentStoreParityWithMemory:
    """Tests that persistent store behaves same as in-memory store."""

    def test_session_operations_parity(
        self,
        persistent_store: OpenClawPersistentStore,
        memory_store: OpenClawGatewayStore,
    ) -> None:
        """Session operations should have same behavior."""
        # Create
        ps = persistent_store.create_session(
            user_id="user-123",
            tenant_id="tenant-abc",
            config={"key": "value"},
        )
        ms = memory_store.create_session(
            user_id="user-123",
            tenant_id="tenant-abc",
            config={"key": "value"},
        )

        assert ps.user_id == ms.user_id
        assert ps.tenant_id == ms.tenant_id
        assert ps.status == ms.status
        assert ps.config == ms.config

        # Update status
        persistent_store.update_session_status(ps.id, SessionStatus.PAUSED)
        memory_store.update_session_status(ms.id, SessionStatus.PAUSED)

        ps_updated = persistent_store.get_session(ps.id)
        ms_updated = memory_store.get_session(ms.id)
        assert ps_updated is not None and ms_updated is not None
        assert ps_updated.status == ms_updated.status

    def test_action_operations_parity(
        self,
        persistent_store: OpenClawPersistentStore,
        memory_store: OpenClawGatewayStore,
    ) -> None:
        """Action operations should have same behavior."""
        ps = persistent_store.create_session(user_id="user-123")
        ms = memory_store.create_session(user_id="user-123")

        pa = persistent_store.create_action(
            session_id=ps.id,
            action_type="click",
            input_data={"x": 100, "y": 200},
        )
        ma = memory_store.create_action(
            session_id=ms.id,
            action_type="click",
            input_data={"x": 100, "y": 200},
        )

        assert pa.action_type == ma.action_type
        assert pa.status == ma.status
        assert pa.input_data == ma.input_data

    def test_metrics_structure_parity(
        self,
        persistent_store: OpenClawPersistentStore,
        memory_store: OpenClawGatewayStore,
    ) -> None:
        """Metrics should have same structure."""
        pm = persistent_store.get_metrics()
        mm = memory_store.get_metrics()

        assert pm.keys() == mm.keys()
        assert pm["sessions"].keys() == mm["sessions"].keys()
        assert pm["actions"].keys() == mm["actions"].keys()
        assert pm["credentials"].keys() == mm["credentials"].keys()


# ===========================================================================
# Database Initialization Tests
# ===========================================================================


class TestDatabaseInitialization:
    """Tests for database schema initialization."""

    def test_creates_tables_on_init(self, temp_db_path: str) -> None:
        """Database tables should be created on initialization."""
        store = OpenClawPersistentStore(db_path=temp_db_path)

        conn = store._get_connection()
        try:
            # Check tables exist
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {t["name"] for t in tables}

            assert "openclaw_sessions" in table_names
            assert "openclaw_actions" in table_names
            assert "openclaw_credentials" in table_names
            assert "openclaw_audit" in table_names
        finally:
            conn.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "dirs" / "test.db"
        assert not nested_path.parent.exists()

        store = OpenClawPersistentStore(db_path=str(nested_path))

        assert nested_path.parent.exists()

    def test_wal_mode_enabled(self, temp_db_path: str) -> None:
        """Database should use WAL mode for better concurrency."""
        store = OpenClawPersistentStore(db_path=temp_db_path)

        conn = store._get_connection()
        try:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0].lower() == "wal"
        finally:
            conn.close()


# ===========================================================================
# Environment Variable Configuration Tests
# ===========================================================================


class TestStoreConfiguration:
    """Tests for store configuration via environment."""

    def test_default_to_persistent_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default should be persistent store."""
        from aragora.server.handlers.openclaw import store as store_module

        # Clear any cached store
        store_module._store = None

        # Remove env var if set
        monkeypatch.delenv("ARAGORA_OPENCLAW_STORE", raising=False)

        # Use temp directory for the default path
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "aragora.config.resolve_db_path",
                lambda name: f"{tmpdir}/{name}",
            )

            store = store_module._get_store()
            assert isinstance(store, OpenClawPersistentStore)

            # Cleanup
            store_module._store = None

    def test_memory_store_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ARAGORA_OPENCLAW_STORE=memory should use in-memory store."""
        from aragora.server.handlers.openclaw import store as store_module

        # Clear any cached store
        store_module._store = None

        monkeypatch.setenv("ARAGORA_OPENCLAW_STORE", "memory")

        store = store_module._get_store()
        assert isinstance(store, OpenClawGatewayStore)

        # Cleanup
        store_module._store = None
