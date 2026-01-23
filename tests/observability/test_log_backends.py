"""
Tests for audit log storage backends.

Tests LocalFileBackend, PostgreSQLAuditBackend, and the factory function.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.observability.log_types import AuditBackend, AuditEntry, DailyAnchor

# Check if psycopg2 is available
try:
    import psycopg2

    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

needs_psycopg2 = pytest.mark.skipif(not POSTGRESQL_AVAILABLE, reason="psycopg2 not installed")


class TestAuditBackendEnum:
    """Tests for AuditBackend enum."""

    def test_all_backends_exist(self):
        """All expected backends are defined."""
        assert AuditBackend.LOCAL.value == "local"
        assert AuditBackend.POSTGRESQL.value == "postgresql"
        assert AuditBackend.S3_OBJECT_LOCK.value == "s3_object_lock"
        assert AuditBackend.QLDB.value == "qldb"


class TestLocalFileBackend:
    """Tests for LocalFileBackend."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_entry(self) -> AuditEntry:
        """Create a sample audit entry."""
        return AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            details={"key": "value"},
            previous_hash="0" * 64,
            entry_hash="a" * 64,
            workspace_id="ws-1",
        )

    @pytest.mark.asyncio
    async def test_append_and_get_entry(self, temp_dir, sample_entry):
        """Can append and retrieve entries by ID."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        await backend.append(sample_entry)

        retrieved = await backend.get_entry("entry-1")

        assert retrieved is not None
        assert retrieved.id == "entry-1"
        assert retrieved.event_type == "test_event"
        assert retrieved.actor == "user@example.com"
        assert retrieved.details == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_by_sequence(self, temp_dir, sample_entry):
        """Can retrieve entries by sequence number."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        await backend.append(sample_entry)

        retrieved = await backend.get_by_sequence(1)

        assert retrieved is not None
        assert retrieved.sequence_number == 1

    @pytest.mark.asyncio
    async def test_get_last_entry(self, temp_dir):
        """Can retrieve the most recent entry."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        # Append multiple entries
        for i in range(1, 4):
            entry = AuditEntry(
                id=f"entry-{i}",
                sequence_number=i,
                timestamp=datetime(2024, 1, 1, 12, 0, i, tzinfo=timezone.utc),
                event_type="test_event",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
                previous_hash="0" * 64,
                entry_hash=f"{i}" * 64,
            )
            await backend.append(entry)

        last = await backend.get_last_entry()

        assert last is not None
        assert last.id == "entry-3"
        assert last.sequence_number == 3

    @pytest.mark.asyncio
    async def test_get_entries_range(self, temp_dir):
        """Can retrieve entries in a sequence range."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        # Append 5 entries
        for i in range(1, 6):
            entry = AuditEntry(
                id=f"entry-{i}",
                sequence_number=i,
                timestamp=datetime(2024, 1, 1, 12, 0, i, tzinfo=timezone.utc),
                event_type="test_event",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
                previous_hash="0" * 64,
                entry_hash=f"{i}" * 64,
            )
            await backend.append(entry)

        entries = await backend.get_entries_range(2, 4)

        assert len(entries) == 3
        assert entries[0].sequence_number == 2
        assert entries[2].sequence_number == 4

    @pytest.mark.asyncio
    async def test_query_with_filters(self, temp_dir):
        """Can query entries with filters."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        # Append entries with different event types
        for i, event_type in enumerate(["create", "update", "delete", "create"], start=1):
            entry = AuditEntry(
                id=f"entry-{i}",
                sequence_number=i,
                timestamp=datetime(2024, 1, 1, 12, 0, i, tzinfo=timezone.utc),
                event_type=event_type,
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id=f"doc-{i}",
                action=event_type,
                previous_hash="0" * 64,
                entry_hash=f"{i}" * 64,
            )
            await backend.append(entry)

        # Query only "create" events
        entries = await backend.query(event_types=["create"])

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_query_with_time_range(self, temp_dir):
        """Can query entries within a time range."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        # Append entries on different days
        for i in range(1, 4):
            entry = AuditEntry(
                id=f"entry-{i}",
                sequence_number=i,
                timestamp=datetime(2024, 1, i, 12, 0, 0, tzinfo=timezone.utc),
                event_type="test_event",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
                previous_hash="0" * 64,
                entry_hash=f"{i}" * 64,
            )
            await backend.append(entry)

        # Query entries from Jan 1-2 only
        entries = await backend.query(
            start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, 23, 59, 59, tzinfo=timezone.utc),
        )

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_count(self, temp_dir):
        """Can count entries matching filters."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        for i in range(1, 6):
            entry = AuditEntry(
                id=f"entry-{i}",
                sequence_number=i,
                timestamp=datetime(2024, 1, 1, 12, 0, i, tzinfo=timezone.utc),
                event_type="test_event",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
                previous_hash="0" * 64,
                entry_hash=f"{i}" * 64,
            )
            await backend.append(entry)

        count = await backend.count()

        assert count == 5

    @pytest.mark.asyncio
    async def test_save_and_get_anchor(self, temp_dir):
        """Can save and retrieve daily anchors."""
        from aragora.observability.log_backends import LocalFileBackend

        backend = LocalFileBackend(temp_dir)

        anchor = DailyAnchor(
            date="2024-01-01",
            first_sequence=1,
            last_sequence=100,
            entry_count=100,
            merkle_root="a" * 64,
            chain_hash="b" * 64,
            created_at=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        await backend.save_anchor(anchor)

        retrieved = await backend.get_anchor("2024-01-01")

        assert retrieved is not None
        assert retrieved.date == "2024-01-01"
        assert retrieved.entry_count == 100
        assert retrieved.merkle_root == "a" * 64


@needs_psycopg2
class TestPostgreSQLAuditBackend:
    """Tests for PostgreSQLAuditBackend with mocked connections."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()

        # Set up cursor as context manager
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)

        conn.cursor = MagicMock(return_value=cursor)
        pool.getconn = MagicMock(return_value=conn)
        pool.putconn = MagicMock()

        return pool, conn, cursor

    @pytest.fixture
    def sample_entry(self) -> AuditEntry:
        """Create a sample audit entry."""
        return AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            details={"key": "value"},
            previous_hash="0" * 64,
            entry_hash="a" * 64,
            workspace_id="ws-1",
        )

    def test_init_requires_database_url(self):
        """Initialization fails without database URL."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL"):
                PostgreSQLAuditBackend()

    @pytest.mark.asyncio
    async def test_append_executes_insert(self, mock_pool, sample_entry):
        """Append executes INSERT statement."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool  # Bypass _init_db

            await backend.append(sample_entry)

            # Verify INSERT was called
            cursor.execute.assert_called()
            call_args = cursor.execute.call_args
            assert "INSERT INTO" in call_args[0][0]
            assert sample_entry.id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_entry_returns_entry(self, mock_pool, sample_entry):
        """get_entry returns entry when found."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool

        # Mock cursor to return a row
        cursor.fetchone.return_value = (
            sample_entry.id,
            sample_entry.timestamp,
            sample_entry.sequence_number,
            sample_entry.previous_hash,
            sample_entry.entry_hash,
            sample_entry.event_type,
            sample_entry.actor,
            sample_entry.actor_type,
            sample_entry.resource_type,
            sample_entry.resource_id,
            sample_entry.action,
            sample_entry.details,
            sample_entry.correlation_id,
            sample_entry.workspace_id,
            sample_entry.ip_address,
            sample_entry.user_agent,
            sample_entry.signature,
        )

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            entry = await backend.get_entry("entry-1")

            assert entry is not None
            assert entry.id == "entry-1"
            assert entry.event_type == "test_event"

    @pytest.mark.asyncio
    async def test_get_entry_returns_none_when_not_found(self, mock_pool):
        """get_entry returns None when entry not found."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = None

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            entry = await backend.get_entry("nonexistent")

            assert entry is None

    @pytest.mark.asyncio
    async def test_get_last_entry_orders_by_sequence(self, mock_pool, sample_entry):
        """get_last_entry orders by sequence DESC."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = (
            sample_entry.id,
            sample_entry.timestamp,
            sample_entry.sequence_number,
            sample_entry.previous_hash,
            sample_entry.entry_hash,
            sample_entry.event_type,
            sample_entry.actor,
            sample_entry.actor_type,
            sample_entry.resource_type,
            sample_entry.resource_id,
            sample_entry.action,
            sample_entry.details,
            sample_entry.correlation_id,
            sample_entry.workspace_id,
            sample_entry.ip_address,
            sample_entry.user_agent,
            sample_entry.signature,
        )

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            await backend.get_last_entry()

            # Verify ORDER BY DESC
            call_args = cursor.execute.call_args[0][0]
            assert "ORDER BY sequence_number DESC" in call_args

    @pytest.mark.asyncio
    async def test_query_with_filters(self, mock_pool):
        """query applies filters correctly."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool
        cursor.fetchall.return_value = []

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            await backend.query(
                event_types=["test_event"],
                actors=["user@example.com"],
                workspace_id="ws-1",
                limit=50,
            )

            # Verify filters in query
            call_args = cursor.execute.call_args[0][0]
            assert "event_type = ANY" in call_args
            assert "actor = ANY" in call_args
            assert "workspace_id = " in call_args

    @pytest.mark.asyncio
    async def test_count_returns_count(self, mock_pool):
        """count returns entry count."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = (42,)

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            count = await backend.count()

            assert count == 42

    @pytest.mark.asyncio
    async def test_save_anchor_uses_upsert(self, mock_pool):
        """save_anchor uses ON CONFLICT DO UPDATE."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool

        anchor = DailyAnchor(
            date="2024-01-01",
            first_sequence=1,
            last_sequence=100,
            entry_count=100,
            merkle_root="a" * 64,
            chain_hash="b" * 64,
            created_at=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            await backend.save_anchor(anchor)

            call_args = cursor.execute.call_args[0][0]
            assert "ON CONFLICT" in call_args
            assert "DO UPDATE" in call_args

    @pytest.mark.asyncio
    async def test_get_anchor_returns_anchor(self, mock_pool):
        """get_anchor returns anchor when found."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = (
            "2024-01-01",
            1,
            100,
            100,
            "a" * 64,
            "b" * 64,
            datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            anchor = await backend.get_anchor("2024-01-01")

            assert anchor is not None
            assert anchor.date == "2024-01-01"
            assert anchor.entry_count == 100

    def test_close_closes_pool(self, mock_pool):
        """close() closes the connection pool."""
        from aragora.observability.log_backends import PostgreSQLAuditBackend

        pool, conn, cursor = mock_pool

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=pool):
            backend = PostgreSQLAuditBackend(database_url="postgresql://test")
            backend._pool = pool

            backend.close()

            pool.closeall.assert_called_once()
            assert backend._pool is None


class TestCreateAuditBackend:
    """Tests for the create_audit_backend factory function."""

    def test_create_local_backend(self):
        """Can create local file backend."""
        from aragora.observability.log_backends import (
            LocalFileBackend,
            create_audit_backend,
        )

        with TemporaryDirectory() as tmpdir:
            backend = create_audit_backend("local", log_dir=tmpdir)

            assert isinstance(backend, LocalFileBackend)

    def test_create_local_backend_from_env(self):
        """Local backend uses ARAGORA_AUDIT_LOG_DIR env var."""
        from aragora.observability.log_backends import (
            LocalFileBackend,
            create_audit_backend,
        )

        with TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ARAGORA_AUDIT_LOG_DIR": tmpdir}):
                backend = create_audit_backend("local")

                assert isinstance(backend, LocalFileBackend)
                assert str(backend.log_dir) == tmpdir

    @needs_psycopg2
    def test_create_postgresql_backend(self):
        """Can create PostgreSQL backend."""
        from aragora.observability.log_backends import (
            PostgreSQLAuditBackend,
            create_audit_backend,
        )

        mock_pool = MagicMock()
        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
            backend = create_audit_backend(
                "postgresql",
                database_url="postgresql://user:pass@localhost/testdb",
            )

            assert isinstance(backend, PostgreSQLAuditBackend)

    def test_create_s3_backend_requires_bucket(self):
        """S3 backend requires bucket parameter."""
        from aragora.observability.log_backends import create_audit_backend

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="bucket"):
                create_audit_backend("s3_object_lock")

    def test_create_s3_backend_from_env(self):
        """S3 backend uses ARAGORA_AUDIT_S3_BUCKET env var."""
        from aragora.observability.log_backends import (
            S3ObjectLockBackend,
            create_audit_backend,
        )

        with patch.dict("os.environ", {"ARAGORA_AUDIT_S3_BUCKET": "test-bucket"}):
            backend = create_audit_backend("s3_object_lock")

            assert isinstance(backend, S3ObjectLockBackend)
            assert backend.bucket == "test-bucket"

    def test_create_unknown_backend_raises(self):
        """Unknown backend type raises ValueError."""
        from aragora.observability.log_backends import create_audit_backend

        with pytest.raises(ValueError, match="Unknown backend type"):
            create_audit_backend("unknown_backend")

    def test_backend_type_case_insensitive(self):
        """Backend type is case insensitive."""
        from aragora.observability.log_backends import (
            LocalFileBackend,
            create_audit_backend,
        )

        with TemporaryDirectory() as tmpdir:
            backend = create_audit_backend("LOCAL", log_dir=tmpdir)
            assert isinstance(backend, LocalFileBackend)

            backend2 = create_audit_backend("Local", log_dir=tmpdir)
            assert isinstance(backend2, LocalFileBackend)
