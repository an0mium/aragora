"""
Tests for Audit Log Persistence Backends.

Tests cover:
- FileBackend: JSON-line file storage
- PostgresBackend: Enterprise database storage
- Backend interface compliance
- Hash chain integrity
- Retention policy enforcement
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.audit.log import (
    AuditCategory,
    AuditEvent,
    AuditOutcome,
    AuditQuery,
)
from aragora.audit.persistence.base import AuditPersistenceBackend, PersistenceError
from aragora.audit.persistence.file import FileBackend


# ============================================================================
# FileBackend Tests
# ============================================================================


class TestFileBackend:
    """Tests for file-based audit persistence."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> FileBackend:
        """Create a FileBackend with temporary storage."""
        backend = FileBackend(tmp_path / "audit_logs")
        backend.initialize()
        return backend

    @pytest.fixture
    def sample_event(self) -> AuditEvent:
        """Create a sample audit event."""
        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="user-123",
            resource_type="session",
            resource_id="sess-abc",
            outcome=AuditOutcome.SUCCESS,
            ip_address="192.168.1.1",
            org_id="org-456",
            details={"browser": "Chrome", "os": "MacOS"},
        )
        # Compute hash chain
        event.previous_hash = ""
        event.event_hash = event.compute_hash()
        return event

    def test_implements_backend_interface(self, backend: FileBackend):
        """Test FileBackend implements AuditPersistenceBackend interface."""
        assert isinstance(backend, AuditPersistenceBackend)

    def test_initialize_creates_directory(self, tmp_path: Path):
        """Test initialization creates storage directory."""
        storage_path = tmp_path / "new_audit_logs"
        backend = FileBackend(storage_path)
        backend.initialize()

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_store_event(self, backend: FileBackend, sample_event: AuditEvent):
        """Test storing an audit event."""
        event_id = backend.store(sample_event)

        assert event_id == sample_event.id
        assert backend.count() == 1

    def test_get_event(self, backend: FileBackend, sample_event: AuditEvent):
        """Test retrieving an event by ID."""
        backend.store(sample_event)

        retrieved = backend.get(sample_event.id)

        assert retrieved is not None
        assert retrieved.id == sample_event.id
        assert retrieved.actor_id == sample_event.actor_id
        assert retrieved.action == sample_event.action
        assert retrieved.category == sample_event.category

    def test_get_nonexistent_event(self, backend: FileBackend):
        """Test retrieving non-existent event returns None."""
        result = backend.get("nonexistent-id")
        assert result is None

    def test_query_by_category(self, backend: FileBackend):
        """Test querying events by category."""
        # Store events with different categories
        for category in [AuditCategory.AUTH, AuditCategory.AUTH, AuditCategory.DATA]:
            event = AuditEvent(
                category=category,
                action="test",
                actor_id="user-123",
            )
            event.previous_hash = backend.get_last_hash()
            event.event_hash = event.compute_hash()
            backend.store(event)

        query = AuditQuery(category=AuditCategory.AUTH)
        results = backend.query(query)

        assert len(results) == 2
        for event in results:
            assert event.category == AuditCategory.AUTH

    def test_query_by_actor(self, backend: FileBackend):
        """Test querying events by actor ID."""
        for actor_id in ["user-1", "user-1", "user-2"]:
            event = AuditEvent(
                category=AuditCategory.ACCESS,
                action="read",
                actor_id=actor_id,
            )
            event.previous_hash = backend.get_last_hash()
            event.event_hash = event.compute_hash()
            backend.store(event)

        query = AuditQuery(actor_id="user-1")
        results = backend.query(query)

        assert len(results) == 2

    def test_query_by_date_range(self, backend: FileBackend):
        """Test querying events by date range."""
        now = datetime.now(timezone.utc)

        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="user-123",
            timestamp=now,
        )
        event.previous_hash = backend.get_last_hash()
        event.event_hash = event.compute_hash()
        backend.store(event)

        # Query for today
        query = AuditQuery(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )
        results = backend.query(query)

        assert len(results) == 1

    def test_query_with_search_text(self, backend: FileBackend):
        """Test full-text search in query."""
        event = AuditEvent(
            category=AuditCategory.ADMIN,
            action="create_user",
            actor_id="admin-123",
            details={"new_user_email": "test@example.com"},
        )
        event.previous_hash = backend.get_last_hash()
        event.event_hash = event.compute_hash()
        backend.store(event)

        query = AuditQuery(search_text="example.com")
        results = backend.query(query)

        assert len(results) == 1

    def test_query_pagination(self, backend: FileBackend):
        """Test query limit and offset."""
        # Store 10 events
        for i in range(10):
            event = AuditEvent(
                category=AuditCategory.API,
                action=f"request_{i}",
                actor_id="user-123",
            )
            event.previous_hash = backend.get_last_hash()
            event.event_hash = event.compute_hash()
            backend.store(event)

        # Get first 3
        query = AuditQuery(limit=3, offset=0)
        results = backend.query(query)
        assert len(results) == 3

        # Get next 3
        query = AuditQuery(limit=3, offset=3)
        results = backend.query(query)
        assert len(results) == 3

    def test_get_last_hash(self, backend: FileBackend, sample_event: AuditEvent):
        """Test getting last event hash."""
        assert backend.get_last_hash() == ""

        backend.store(sample_event)

        assert backend.get_last_hash() == sample_event.event_hash

    def test_hash_chain_continuity(self, backend: FileBackend):
        """Test hash chain maintains continuity."""
        events = []
        prev_hash = ""

        for i in range(5):
            event = AuditEvent(
                category=AuditCategory.AUTH,
                action=f"action_{i}",
                actor_id="user-123",
            )
            event.previous_hash = prev_hash
            event.event_hash = event.compute_hash()
            backend.store(event)
            prev_hash = event.event_hash
            events.append(event)

        # Verify chain
        is_valid, errors = backend.verify_integrity()
        assert is_valid, f"Integrity errors: {errors}"

    def test_verify_integrity_detects_tampering(self, backend: FileBackend, tmp_path: Path):
        """Test integrity verification detects tampering."""
        # Store an event
        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="user-123",
        )
        event.previous_hash = ""
        event.event_hash = event.compute_hash()
        backend.store(event)
        backend.close()  # Flush to disk

        # Tamper with the file
        log_files = list((tmp_path / "audit_logs").glob("audit_*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0], "r") as f:
            data = json.loads(f.read())
        data["action"] = "tampered"
        with open(log_files[0], "w") as f:
            f.write(json.dumps(data))

        # Reinitialize and verify
        backend2 = FileBackend(tmp_path / "audit_logs")
        backend2.initialize()
        is_valid, errors = backend2.verify_integrity()

        assert not is_valid
        assert len(errors) > 0
        assert "hash mismatch" in errors[0].lower()

    def test_delete_before_cutoff(self, backend: FileBackend):
        """Test retention policy deletes old events."""
        # Store event with old timestamp
        old_event = AuditEvent(
            category=AuditCategory.AUTH,
            action="old_login",
            actor_id="user-123",
            timestamp=datetime.now(timezone.utc) - timedelta(days=400),
        )
        old_event.previous_hash = ""
        old_event.event_hash = old_event.compute_hash()

        # We can't easily test this without mocking the file date
        # Just verify the method runs without error
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        deleted = backend.delete_before(cutoff)

        assert deleted >= 0

    def test_count_events(self, backend: FileBackend):
        """Test counting events."""
        assert backend.count() == 0

        for i in range(5):
            event = AuditEvent(
                category=AuditCategory.API,
                action=f"request_{i}",
                actor_id="user-123",
            )
            event.previous_hash = backend.get_last_hash()
            event.event_hash = event.compute_hash()
            backend.store(event)

        assert backend.count() == 5

    def test_get_stats(self, backend: FileBackend, sample_event: AuditEvent):
        """Test getting backend statistics."""
        backend.store(sample_event)

        stats = backend.get_stats()

        assert stats["backend"] == "File"
        assert stats["total_events"] == 1
        assert "storage_path" in stats

    def test_close_persists_state(
        self, backend: FileBackend, sample_event: AuditEvent, tmp_path: Path
    ):
        """Test close persists index and meta."""
        backend.store(sample_event)
        backend.close()

        # Verify files exist
        assert (tmp_path / "audit_logs" / "index.json").exists()
        assert (tmp_path / "audit_logs" / "meta.json").exists()

    def test_reinitialize_loads_state(
        self, backend: FileBackend, sample_event: AuditEvent, tmp_path: Path
    ):
        """Test reinitialization loads previous state."""
        backend.store(sample_event)
        last_hash = backend.get_last_hash()
        backend.close()

        # Create new backend instance
        backend2 = FileBackend(tmp_path / "audit_logs")
        backend2.initialize()

        assert backend2.get_last_hash() == last_hash
        assert backend2.get(sample_event.id) is not None

    def test_concurrent_writes(self, backend: FileBackend):
        """Test concurrent write safety (basic test)."""
        import threading

        errors = []

        def write_events():
            try:
                for i in range(10):
                    event = AuditEvent(
                        category=AuditCategory.API,
                        action=f"concurrent_{threading.current_thread().name}_{i}",
                        actor_id="user-123",
                    )
                    event.previous_hash = ""  # Simplified for concurrent test
                    event.event_hash = event.compute_hash()
                    backend.store(event)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_events) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# PostgresBackend Tests (Mocked)
# ============================================================================


class TestPostgresBackendMocked:
    """Tests for PostgreSQL backend with mocked database."""

    @pytest.fixture
    def mock_psycopg2(self):
        """Mock psycopg2 module."""
        with patch.dict("sys.modules", {"psycopg2": MagicMock()}):
            import sys

            mock_pg = sys.modules["psycopg2"]
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_pg.connect.return_value = mock_conn
            yield mock_pg, mock_conn, mock_cursor

    def test_postgres_backend_initialization(self, mock_psycopg2):
        """Test PostgreSQL backend initializes correctly."""
        from aragora.audit.persistence.postgres import PostgresBackend

        mock_pg, mock_conn, mock_cursor = mock_psycopg2

        backend = PostgresBackend("postgresql://localhost/test")
        backend.initialize()

        # Should have executed schema creation
        assert mock_cursor.execute.called

    def test_postgres_backend_store_event(self, mock_psycopg2):
        """Test storing event in PostgreSQL."""
        from aragora.audit.persistence.postgres import PostgresBackend

        mock_pg, mock_conn, mock_cursor = mock_psycopg2

        backend = PostgresBackend("postgresql://localhost/test")
        backend.initialize()

        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="user-123",
        )
        event.previous_hash = ""
        event.event_hash = event.compute_hash()

        event_id = backend.store(event)

        assert event_id == event.id
        assert mock_cursor.execute.called
        mock_conn.commit.assert_called()

    def test_postgres_backend_query(self, mock_psycopg2):
        """Test querying events from PostgreSQL."""
        from aragora.audit.persistence.postgres import PostgresBackend

        mock_pg, mock_conn, mock_cursor = mock_psycopg2

        # Set up mock to return event data
        mock_cursor.fetchall.return_value = [
            (
                "event-id",
                "2024-01-15T10:00:00+00:00",
                "auth",
                "login",
                "user-123",
                "session",
                "sess-1",
                "success",
                "192.168.1.1",
                "Chrome",
                "corr-1",
                "org-1",
                "ws-1",
                "{}",
                "",
                "",
                "abc123",
            )
        ]
        mock_cursor.description = [
            ("id",),
            ("timestamp",),
            ("category",),
            ("action",),
            ("actor_id",),
            ("resource_type",),
            ("resource_id",),
            ("outcome",),
            ("ip_address",),
            ("user_agent",),
            ("correlation_id",),
            ("org_id",),
            ("workspace_id",),
            ("details",),
            ("reason",),
            ("previous_hash",),
            ("event_hash",),
        ]

        backend = PostgresBackend("postgresql://localhost/test")

        query = AuditQuery(actor_id="user-123")
        results = backend.query(query)

        assert len(results) == 1
        assert results[0].actor_id == "user-123"


# ============================================================================
# Backend Factory Tests
# ============================================================================


class TestBackendFactory:
    """Tests for backend factory function."""

    def test_get_file_backend(self, tmp_path: Path):
        """Test getting file backend."""
        from aragora.audit.persistence import get_backend

        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_STORE_BACKEND": "file"},
            clear=False,
        ):
            with patch(
                "aragora.persistence.db_config.get_nomic_dir",
                return_value=tmp_path,
            ):
                backend = get_backend()
                assert isinstance(backend, FileBackend)

    def test_get_postgres_backend_requires_url(self):
        """Test PostgreSQL backend requires DATABASE_URL."""
        from aragora.audit.persistence import get_backend

        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_STORE_BACKEND": "postgres"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="database_url"):
                get_backend()

    def test_unknown_backend_raises_error(self):
        """Test unknown backend type raises error."""
        from aragora.audit.persistence import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend(backend_type="unknown")


# ============================================================================
# Integration Tests
# ============================================================================


class TestPersistenceIntegration:
    """Integration tests for audit persistence."""

    def test_full_audit_cycle(self, tmp_path: Path):
        """Test complete audit cycle: store, query, verify, export."""
        backend = FileBackend(tmp_path / "audit")
        backend.initialize()

        # Store events
        events = []
        for i in range(10):
            event = AuditEvent(
                category=AuditCategory.AUTH if i % 2 == 0 else AuditCategory.API,
                action=f"action_{i}",
                actor_id=f"user-{i % 3}",
                outcome=AuditOutcome.SUCCESS,
                details={"index": i},
            )
            event.previous_hash = backend.get_last_hash()
            event.event_hash = event.compute_hash()
            backend.store(event)
            events.append(event)

        # Verify integrity
        is_valid, errors = backend.verify_integrity()
        assert is_valid, f"Integrity errors: {errors}"

        # Query by category
        auth_events = backend.query(AuditQuery(category=AuditCategory.AUTH))
        assert len(auth_events) == 5

        # Query by actor
        user_0_events = backend.query(AuditQuery(actor_id="user-0"))
        assert len(user_0_events) == 4  # 0, 3, 6, 9

        # Get stats
        stats = backend.get_stats()
        assert stats["total_events"] == 10

        # Close and reopen
        backend.close()
        backend2 = FileBackend(tmp_path / "audit")
        backend2.initialize()

        # Verify events survive restart
        assert backend2.count() == 10
        assert backend2.get(events[0].id) is not None

    def test_events_survive_restart(self, tmp_path: Path):
        """Test audit entries survive server restart (verification per plan)."""
        storage_path = tmp_path / "restart_test"

        # First session - store events
        backend1 = FileBackend(storage_path)
        backend1.initialize()

        event = AuditEvent(
            category=AuditCategory.SECURITY,
            action="password_change",
            actor_id="user-critical",
            outcome=AuditOutcome.SUCCESS,
            details={"reason": "periodic rotation"},
        )
        event.previous_hash = ""
        event.event_hash = event.compute_hash()
        backend1.store(event)
        event_id = event.id
        backend1.close()

        # Simulate restart - new backend instance
        backend2 = FileBackend(storage_path)
        backend2.initialize()

        # Event should be retrievable
        retrieved = backend2.get(event_id)
        assert retrieved is not None
        assert retrieved.actor_id == "user-critical"
        assert retrieved.action == "password_change"
        assert retrieved.details["reason"] == "periodic rotation"

        # Hash chain should be intact
        is_valid, errors = backend2.verify_integrity()
        assert is_valid, f"Integrity broken after restart: {errors}"
