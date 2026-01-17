"""
Tests for ImmutableAuditLog - tamper-evident audit logging system.

Tests cover:
- Hash chain integrity
- Entry creation and retrieval
- Tamper detection
- Daily anchors
- Query and export functionality
- Multiple backend support
"""

import json
import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from aragora.observability.immutable_log import (
    ImmutableAuditLog,
    AuditEntry,
    AuditBackend,
    DailyAnchor,
    VerificationResult,
    LocalFileBackend,
    get_audit_log,
    init_audit_log,
    audit_finding_created,
    audit_finding_updated,
    audit_document_uploaded,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def local_backend(temp_log_dir):
    """Create a local file backend for testing."""
    return LocalFileBackend(temp_log_dir)


@pytest.fixture
def audit_log(local_backend):
    """Create an ImmutableAuditLog with local backend."""
    return ImmutableAuditLog(backend=local_backend)


@pytest.fixture
def signed_audit_log(local_backend):
    """Create an ImmutableAuditLog with signing key."""
    signing_key = b"test-signing-key-for-hmac-256"
    return ImmutableAuditLog(backend=local_backend, signing_key=signing_key)


# ============================================================================
# AuditEntry Tests
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            id="entry-123",
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            previous_hash="0" * 64,
            entry_hash="abc123",
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
        )

        assert entry.id == "entry-123"
        assert entry.sequence_number == 1
        assert entry.event_type == "test_event"
        assert entry.actor_type == "user"

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        now = datetime.now(timezone.utc)
        entry = AuditEntry(
            id="entry-123",
            timestamp=now,
            sequence_number=1,
            previous_hash="0" * 64,
            entry_hash="abc123",
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
            details={"key": "value"},
        )

        data = entry.to_dict()

        assert data["id"] == "entry-123"
        assert data["timestamp"] == now.isoformat()
        assert data["details"] == {"key": "value"}
        assert data["actor_type"] == "user"

    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "entry-123",
            "timestamp": now.isoformat(),
            "sequence_number": 1,
            "previous_hash": "0" * 64,
            "entry_hash": "abc123",
            "event_type": "test_event",
            "actor": "user@example.com",
            "actor_type": "user",
            "resource_type": "document",
            "resource_id": "doc-456",
            "action": "create",
        }

        entry = AuditEntry.from_dict(data)

        assert entry.id == "entry-123"
        assert entry.sequence_number == 1
        assert entry.actor_type == "user"

    def test_entry_compute_hash(self):
        """Test computing entry hash is deterministic."""
        entry = AuditEntry(
            id="entry-123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            sequence_number=1,
            previous_hash="0" * 64,
            entry_hash="",
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
        )

        hash1 = entry.compute_hash()
        hash2 = entry.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex


# ============================================================================
# LocalFileBackend Tests
# ============================================================================


class TestLocalFileBackend:
    """Tests for local file storage backend."""

    @pytest.mark.asyncio
    async def test_append_and_get(self, local_backend):
        """Test appending and retrieving entry."""
        entry = AuditEntry(
            id="entry-123",
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            previous_hash="0" * 64,
            entry_hash="abc123",
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
        )

        await local_backend.append(entry)
        retrieved = await local_backend.get_entry("entry-123")

        assert retrieved is not None
        assert retrieved.id == "entry-123"
        assert retrieved.event_type == "test_event"

    @pytest.mark.asyncio
    async def test_get_by_sequence(self, local_backend):
        """Test getting entry by sequence number."""
        entry = AuditEntry(
            id="entry-123",
            timestamp=datetime.now(timezone.utc),
            sequence_number=42,
            previous_hash="0" * 64,
            entry_hash="abc123",
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
        )

        await local_backend.append(entry)
        retrieved = await local_backend.get_by_sequence(42)

        assert retrieved is not None
        assert retrieved.sequence_number == 42

    @pytest.mark.asyncio
    async def test_get_last_entry(self, local_backend):
        """Test getting the most recent entry."""
        for i in range(3):
            entry = AuditEntry(
                id=f"entry-{i}",
                timestamp=datetime.now(timezone.utc),
                sequence_number=i + 1,
                previous_hash="0" * 64,
                entry_hash=f"hash{i}",
                event_type="test_event",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id="doc-456",
                action="create",
            )
            await local_backend.append(entry)

        last = await local_backend.get_last_entry()

        assert last is not None
        assert last.sequence_number == 3
        assert last.id == "entry-2"

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, local_backend):
        """Test querying entries by event type."""
        for i, event_type in enumerate(["create", "update", "create"]):
            entry = AuditEntry(
                id=f"entry-{i}",
                timestamp=datetime.now(timezone.utc),
                sequence_number=i + 1,
                previous_hash="0" * 64,
                entry_hash=f"hash{i}",
                event_type=event_type,
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id="doc-456",
                action="test",
            )
            await local_backend.append(entry)

        results = await local_backend.query(event_types=["create"])

        assert len(results) == 2
        assert all(e.event_type == "create" for e in results)

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, local_backend):
        """Test querying entries by time range."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        entry = AuditEntry(
            id="entry-1",
            timestamp=now,
            sequence_number=1,
            previous_hash="0" * 64,
            entry_hash="hash1",
            event_type="test",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-456",
            action="create",
        )
        await local_backend.append(entry)

        results = await local_backend.query(start_time=yesterday, end_time=tomorrow)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_count(self, local_backend):
        """Test counting entries."""
        for i in range(5):
            entry = AuditEntry(
                id=f"entry-{i}",
                timestamp=datetime.now(timezone.utc),
                sequence_number=i + 1,
                previous_hash="0" * 64,
                entry_hash=f"hash{i}",
                event_type="test",
                actor="user@example.com",
                actor_type="user",
                resource_type="document",
                resource_id="doc-456",
                action="create",
            )
            await local_backend.append(entry)

        count = await local_backend.count()

        assert count == 5


# ============================================================================
# ImmutableAuditLog Tests
# ============================================================================


class TestImmutableAuditLog:
    """Tests for main ImmutableAuditLog class."""

    @pytest.mark.asyncio
    async def test_append_first_entry(self, audit_log):
        """Test appending first entry with genesis hash."""
        entry = await audit_log.append(
            event_type="test_event",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-123",
            action="create",
        )

        assert entry.sequence_number == 1
        assert entry.previous_hash == "0" * 64  # Genesis hash
        assert len(entry.entry_hash) == 64

    @pytest.mark.asyncio
    async def test_append_chain_integrity(self, audit_log):
        """Test that hash chain is maintained."""
        entry1 = await audit_log.append(
            event_type="test_1",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-123",
            action="create",
        )

        entry2 = await audit_log.append(
            event_type="test_2",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-456",
            action="update",
        )

        assert entry2.previous_hash == entry1.entry_hash
        assert entry2.sequence_number == 2

    @pytest.mark.asyncio
    async def test_append_with_details(self, audit_log):
        """Test appending entry with additional details."""
        entry = await audit_log.append(
            event_type="finding_created",
            actor="user@example.com",
            resource_type="finding",
            resource_id="f-123",
            action="create",
            details={"severity": "high", "category": "security"},
            workspace_id="ws-456",
            correlation_id="corr-789",
        )

        assert entry.details["severity"] == "high"
        assert entry.workspace_id == "ws-456"
        assert entry.correlation_id == "corr-789"

    @pytest.mark.asyncio
    async def test_verify_integrity_empty_log(self, audit_log):
        """Test verifying integrity of empty log."""
        result = await audit_log.verify_integrity()

        assert result.is_valid is True
        assert result.entries_checked == 0
        assert "No entries" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_verify_integrity_valid_chain(self, audit_log):
        """Test verifying integrity of valid chain."""
        for i in range(5):
            await audit_log.append(
                event_type=f"event_{i}",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
            )

        result = await audit_log.verify_integrity()

        assert result.is_valid is True
        assert result.entries_checked == 5
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_verify_integrity_detects_tampering(self, local_backend):
        """Test that verification detects tampered entries."""
        audit_log = ImmutableAuditLog(backend=local_backend)

        # Create valid chain
        await audit_log.append(
            event_type="event_1",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-1",
            action="create",
        )

        await audit_log.append(
            event_type="event_2",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-2",
            action="create",
        )

        # Tamper with the log file directly
        log_file = Path(local_backend.log_dir) / "audit.jsonl"
        lines = log_file.read_text().strip().split("\n")
        data = json.loads(lines[0])
        data["event_type"] = "TAMPERED"
        lines[0] = json.dumps(data)
        log_file.write_text("\n".join(lines) + "\n")

        # Reload backend to pick up tampered data
        local_backend._rebuild_index()

        result = await audit_log.verify_integrity()

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Hash mismatch" in result.errors[0] or "Chain broken" in result.errors[0]

    @pytest.mark.asyncio
    async def test_signed_entries(self, signed_audit_log):
        """Test that entries are signed when signing key is provided."""
        entry = await signed_audit_log.append(
            event_type="test_event",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-123",
            action="create",
        )

        assert entry.signature is not None
        assert len(entry.signature) == 64  # HMAC-SHA256 hex

    @pytest.mark.asyncio
    async def test_query_entries(self, audit_log):
        """Test querying entries with filters."""
        for i in range(10):
            await audit_log.append(
                event_type="create" if i % 2 == 0 else "update",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="test",
            )

        results = await audit_log.query(event_types=["create"], limit=5)

        assert len(results) == 5
        assert all(e.event_type == "create" for e in results)

    @pytest.mark.asyncio
    async def test_export_json(self, audit_log):
        """Test exporting entries as JSON."""
        await audit_log.append(
            event_type="test_event",
            actor="user@example.com",
            resource_type="document",
            resource_id="doc-123",
            action="create",
        )

        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        data = await audit_log.export_range(yesterday, tomorrow, format="json")

        parsed = json.loads(data)
        assert "entries" in parsed
        assert "entry_count" in parsed
        assert parsed["entry_count"] == 1

    @pytest.mark.asyncio
    async def test_export_jsonl(self, audit_log):
        """Test exporting entries as JSON lines."""
        for i in range(3):
            await audit_log.append(
                event_type=f"event_{i}",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
            )

        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        data = await audit_log.export_range(yesterday, tomorrow, format="jsonl")

        lines = data.strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_get_statistics(self, audit_log):
        """Test getting audit log statistics."""
        for i in range(5):
            await audit_log.append(
                event_type=f"event_{i}",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
            )

        stats = await audit_log.get_statistics()

        assert stats["total_entries"] == 5
        assert stats["last_sequence"] == 5
        assert stats["last_entry_hash"] is not None


# ============================================================================
# Daily Anchor Tests
# ============================================================================


class TestDailyAnchors:
    """Tests for daily anchor functionality."""

    @pytest.mark.asyncio
    async def test_create_daily_anchor(self, audit_log):
        """Test creating a daily anchor."""
        # Create entries
        for i in range(5):
            await audit_log.append(
                event_type=f"event_{i}",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
            )

        # Create anchor for today
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        anchor = await audit_log.create_daily_anchor(today)

        assert anchor is not None
        assert anchor.date == today
        assert anchor.entry_count == 5
        assert len(anchor.merkle_root) == 64

    @pytest.mark.asyncio
    async def test_verify_against_anchor(self, audit_log):
        """Test verifying entries against anchor."""
        # Create entries
        for i in range(5):
            await audit_log.append(
                event_type=f"event_{i}",
                actor="user@example.com",
                resource_type="document",
                resource_id=f"doc-{i}",
                action="create",
            )

        # Create and verify anchor
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await audit_log.create_daily_anchor(today)

        result = await audit_log.verify_anchor(today)

        assert result.is_valid is True
        assert result.entries_checked == 5

    @pytest.mark.asyncio
    async def test_missing_anchor(self, audit_log):
        """Test verifying with missing anchor."""
        result = await audit_log.verify_anchor("2020-01-01")

        assert result.is_valid is False
        assert "No anchor found" in result.errors[0]


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience audit logging functions."""

    @pytest.mark.asyncio
    async def test_audit_finding_created(self, temp_log_dir):
        """Test logging finding creation."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_LOG_DIR": temp_log_dir},
        ):
            # Reset global instance
            import aragora.observability.immutable_log as log_module

            log_module._audit_log = None

            entry = await audit_finding_created(
                finding_id="f-123",
                actor="user@example.com",
                workspace_id="ws-456",
                severity="high",
                category="security",
            )

            assert entry.event_type == "finding_created"
            assert entry.resource_type == "finding"
            assert entry.resource_id == "f-123"
            assert entry.details["severity"] == "high"

    @pytest.mark.asyncio
    async def test_audit_finding_updated(self, temp_log_dir):
        """Test logging finding update."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_LOG_DIR": temp_log_dir},
        ):
            import aragora.observability.immutable_log as log_module

            log_module._audit_log = None

            entry = await audit_finding_updated(
                finding_id="f-123",
                actor="user@example.com",
                changes={"status": "resolved"},
            )

            assert entry.event_type == "finding_updated"
            assert entry.action == "update"
            assert entry.details["changes"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_audit_document_uploaded(self, temp_log_dir):
        """Test logging document upload."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_LOG_DIR": temp_log_dir},
        ):
            import aragora.observability.immutable_log as log_module

            log_module._audit_log = None

            entry = await audit_document_uploaded(
                document_id="doc-123",
                actor="user@example.com",
                workspace_id="ws-456",
                filename="contract.pdf",
            )

            assert entry.event_type == "document_uploaded"
            assert entry.resource_type == "document"
            assert entry.details["filename"] == "contract.pdf"


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for audit log initialization."""

    def test_init_local_backend(self, temp_log_dir):
        """Test initializing with local backend."""
        log = init_audit_log(
            backend=AuditBackend.LOCAL,
            log_dir=temp_log_dir,
        )

        assert log is not None
        assert isinstance(log.backend, LocalFileBackend)

    def test_get_audit_log_creates_instance(self, temp_log_dir):
        """Test get_audit_log creates instance on first call."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_AUDIT_LOG_DIR": temp_log_dir},
        ):
            import aragora.observability.immutable_log as log_module

            log_module._audit_log = None

            log1 = get_audit_log()
            log2 = get_audit_log()

            assert log1 is log2  # Same instance
