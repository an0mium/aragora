"""
Tests for immutable audit logging system.

Tests hash chain integrity, append-only operations, and verification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.observability.log_types import (
    AuditBackend,
    AuditEntry,
    DailyAnchor,
    VerificationResult,
)


class TestAuditBackend:
    """Tests for AuditBackend enum."""

    def test_all_backends_exist(self):
        """All expected backends are defined."""
        assert AuditBackend.LOCAL.value == "local"
        assert AuditBackend.S3_OBJECT_LOCK.value == "s3_object_lock"
        assert AuditBackend.QLDB.value == "qldb"


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_entry_creation(self):
        """Audit entries are created correctly."""
        entry = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            details={"key": "value"},
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        assert entry.id == "entry-1"
        assert entry.sequence_number == 1
        assert entry.event_type == "test_event"
        assert entry.actor == "user@example.com"
        assert entry.actor_type == "user"

    def test_entry_to_dict(self):
        """Entries serialize to dictionary."""
        entry = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        d = entry.to_dict()

        assert d["id"] == "entry-1"
        assert d["event_type"] == "test_event"
        assert d["actor_type"] == "user"
        assert "timestamp" in d

    def test_entry_compute_hash(self):
        """Entries can compute their own hash."""
        entry = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",  # Will be computed
        )

        computed_hash = entry.compute_hash()

        # Hash should be a 64-character hex string (SHA-256)
        assert len(computed_hash) == 64
        assert all(c in "0123456789abcdef" for c in computed_hash)

    def test_entry_hash_deterministic(self):
        """Same entry data produces same hash."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        entry1 = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=timestamp,
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",
        )

        entry2 = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=timestamp,
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",
        )

        assert entry1.compute_hash() == entry2.compute_hash()

    def test_entry_hash_changes_with_data(self):
        """Different data produces different hash."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        entry1 = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=timestamp,
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",
        )

        entry2 = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=timestamp,
            event_type="different_event",  # Different
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",
        )

        assert entry1.compute_hash() != entry2.compute_hash()

    def test_entry_verify_hash(self):
        """Entry hash can be verified by recomputing."""
        entry = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="",
        )
        entry.entry_hash = entry.compute_hash()

        # Verify by recomputing
        assert entry.compute_hash() == entry.entry_hash

    def test_entry_tampered_hash_detectable(self):
        """Tampered entry can be detected by hash mismatch."""
        entry = AuditEntry(
            id="entry-1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="invalid_hash" + "x" * 52,  # Wrong hash
        )

        # Recomputing hash should not match stored hash
        assert entry.compute_hash() != entry.entry_hash

    def test_entry_from_dict(self):
        """Entry can be created from dictionary."""
        data = {
            "id": "entry-1",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "sequence_number": 1,
            "previous_hash": "0" * 64,
            "entry_hash": "a" * 64,
            "event_type": "test_event",
            "actor": "user@example.com",
            "actor_type": "user",
            "resource_type": "document",
            "resource_id": "doc-123",
            "action": "create",
            "details": {"key": "value"},
        }

        entry = AuditEntry.from_dict(data)

        assert entry.id == "entry-1"
        assert entry.actor_type == "user"
        assert entry.details == {"key": "value"}


class TestDailyAnchor:
    """Tests for DailyAnchor dataclass."""

    def test_anchor_creation(self):
        """Daily anchors are created correctly."""
        anchor = DailyAnchor(
            date="2024-01-01",
            merkle_root="abc123" * 10 + "abcd",
            entry_count=100,
            first_sequence=1,
            last_sequence=100,
            chain_hash="def456" * 10 + "defg",
            created_at=datetime.now(timezone.utc),
        )

        assert anchor.date == "2024-01-01"
        assert anchor.entry_count == 100
        assert anchor.first_sequence == 1
        assert anchor.last_sequence == 100
        assert anchor.chain_hash is not None

    def test_anchor_to_dict(self):
        """Anchors serialize to dictionary."""
        now = datetime.now(timezone.utc)
        anchor = DailyAnchor(
            date="2024-01-01",
            merkle_root="a" * 64,
            entry_count=50,
            first_sequence=1,
            last_sequence=50,
            chain_hash="b" * 64,
            created_at=now,
        )

        d = anchor.to_dict()

        assert d["date"] == "2024-01-01"
        assert d["merkle_root"] == "a" * 64
        assert d["entry_count"] == 50
        assert d["chain_hash"] == "b" * 64
        assert "created_at" in d


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_valid(self):
        """Valid verification result."""
        result = VerificationResult(
            is_valid=True,
            entries_checked=1000,
            errors=[],
            warnings=[],
        )

        assert result.is_valid is True
        assert result.entries_checked == 1000
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_result_invalid(self):
        """Invalid verification result with errors."""
        result = VerificationResult(
            is_valid=False,
            entries_checked=1000,
            errors=[
                "Hash mismatch at sequence 500",
                "Chain broken at sequence 501",
            ],
            warnings=["Timestamp gap detected at sequence 300"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_result_to_dict(self):
        """Results serialize to dictionary."""
        result = VerificationResult(
            is_valid=False,
            entries_checked=100,
            errors=["Test error"],
            warnings=["Test warning"],
            first_error_sequence=50,
            verification_time_ms=123.45,
        )

        d = result.to_dict()

        assert d["is_valid"] is False
        assert d["entries_checked"] == 100
        assert "Test error" in d["errors"]
        assert "Test warning" in d["warnings"]
        assert d["first_error_sequence"] == 50
        assert d["verification_time_ms"] == 123.45


class TestImmutableAuditLog:
    """Tests for ImmutableAuditLog class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = MagicMock()
        backend.get_last_entry = AsyncMock(return_value=None)
        backend.append = AsyncMock()
        backend.query = AsyncMock(return_value=[])
        backend.get_entries_range = AsyncMock(return_value=[])
        return backend

    @pytest.mark.asyncio
    async def test_log_initialization(self, mock_backend):
        """Log initializes correctly."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=mock_backend)

        assert log.backend is mock_backend
        assert log._initialized is False

    @pytest.mark.asyncio
    async def test_log_ensure_initialized(self, mock_backend):
        """Log initializes on first operation."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=mock_backend)

        await log._ensure_initialized()

        assert log._initialized is True
        mock_backend.get_last_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_append_first_entry(self, mock_backend):
        """First entry uses genesis hash."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=mock_backend)

        # Mock append to capture the entry
        appended_entry = None

        async def capture_append(entry):
            nonlocal appended_entry
            appended_entry = entry

        mock_backend.append = capture_append

        entry = await log.append(
            event_type="test_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-123",
            action="create",
        )

        assert entry is not None
        assert entry.sequence_number == 1
        assert entry.previous_hash == ImmutableAuditLog.GENESIS_HASH

    @pytest.mark.asyncio
    async def test_log_append_chain(self, mock_backend):
        """Subsequent entries chain to previous."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        # Create a previous entry
        prev_entry = AuditEntry(
            id="prev-1",
            sequence_number=5,
            timestamp=datetime.now(timezone.utc),
            event_type="prev_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-1",
            action="create",
            previous_hash="x" * 64,
            entry_hash="y" * 64,
        )

        mock_backend.get_last_entry = AsyncMock(return_value=prev_entry)

        log = ImmutableAuditLog(backend=mock_backend)

        appended_entry = None

        async def capture_append(entry):
            nonlocal appended_entry
            appended_entry = entry

        mock_backend.append = capture_append

        entry = await log.append(
            event_type="new_event",
            actor="user@example.com",
            actor_type="user",
            resource_type="document",
            resource_id="doc-2",
            action="update",
        )

        assert entry.sequence_number == 6  # prev + 1
        assert entry.previous_hash == prev_entry.entry_hash

    @pytest.mark.asyncio
    async def test_log_query(self, mock_backend):
        """Log can query entries."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        mock_entries = [
            AuditEntry(
                id="e1",
                sequence_number=1,
                timestamp=datetime.now(timezone.utc),
                event_type="test",
                actor="user",
                actor_type="user",
                resource_type="doc",
                resource_id="d1",
                action="create",
                previous_hash="0" * 64,
                entry_hash="a" * 64,
            ),
        ]
        mock_backend.query = AsyncMock(return_value=mock_entries)

        log = ImmutableAuditLog(backend=mock_backend)

        entries = await log.query(
            start_time=datetime.now(timezone.utc) - timedelta(days=7),
            event_types=["test"],
        )

        assert len(entries) == 1
        mock_backend.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_verify_integrity_empty(self, mock_backend):
        """Empty log verifies as valid."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        mock_backend.get_entries_range = AsyncMock(return_value=[])

        log = ImmutableAuditLog(backend=mock_backend)

        result = await log.verify_integrity()

        assert result.is_valid is True
        assert result.entries_checked == 0

    @pytest.mark.asyncio
    async def test_log_verify_integrity_valid_chain(self, mock_backend):
        """Valid chain passes verification."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        # Create a valid chain
        entry1 = AuditEntry(
            id="e1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="d1",
            action="create",
            previous_hash=ImmutableAuditLog.GENESIS_HASH,
            entry_hash="",
        )
        entry1.entry_hash = entry1.compute_hash()

        entry2 = AuditEntry(
            id="e2",
            sequence_number=2,
            timestamp=datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="d2",
            action="create",
            previous_hash=entry1.entry_hash,  # Chain to previous
            entry_hash="",
        )
        entry2.entry_hash = entry2.compute_hash()

        # Set up mock to return the last entry for _ensure_initialized
        mock_backend.get_last_entry = AsyncMock(return_value=entry2)
        mock_backend.get_entries_range = AsyncMock(return_value=[entry1, entry2])

        log = ImmutableAuditLog(backend=mock_backend)

        result = await log.verify_integrity()

        assert result.is_valid is True
        assert result.entries_checked == 2

    @pytest.mark.asyncio
    async def test_log_verify_integrity_broken_chain(self, mock_backend):
        """Broken chain fails verification."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        entry1 = AuditEntry(
            id="e1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="d1",
            action="create",
            previous_hash=ImmutableAuditLog.GENESIS_HASH,
            entry_hash="",
        )
        entry1.entry_hash = entry1.compute_hash()

        entry2 = AuditEntry(
            id="e2",
            sequence_number=2,
            timestamp=datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="d2",
            action="create",
            previous_hash="wrong_hash" + "x" * 54,  # WRONG - breaks chain
            entry_hash="",
        )
        entry2.entry_hash = entry2.compute_hash()

        # Set up mock to return the last entry for _ensure_initialized
        mock_backend.get_last_entry = AsyncMock(return_value=entry2)
        mock_backend.get_entries_range = AsyncMock(return_value=[entry1, entry2])

        log = ImmutableAuditLog(backend=mock_backend)

        result = await log.verify_integrity()

        assert result.is_valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_log_verify_integrity_tampered_entry(self, mock_backend):
        """Tampered entry fails verification."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        entry1 = AuditEntry(
            id="e1",
            sequence_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="d1",
            action="create",
            previous_hash=ImmutableAuditLog.GENESIS_HASH,
            entry_hash="tampered_hash" + "x" * 51,  # Wrong hash
        )

        # Set up mock to return the last entry for _ensure_initialized
        mock_backend.get_last_entry = AsyncMock(return_value=entry1)
        mock_backend.get_entries_range = AsyncMock(return_value=[entry1])

        log = ImmutableAuditLog(backend=mock_backend)

        result = await log.verify_integrity()

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_compute_merkle_root_empty(self):
        """Merkle root of empty list is genesis hash."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=MagicMock())
        root = log._compute_merkle_root([])

        assert root == ImmutableAuditLog.GENESIS_HASH

    def test_compute_merkle_root_single(self):
        """Merkle root of single hash is that hash (after padding)."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=MagicMock())
        hash1 = "a" * 64
        root = log._compute_merkle_root([hash1])

        # Single element gets padded and combined with itself
        assert len(root) == 64

    def test_compute_merkle_root_deterministic(self):
        """Same hashes produce same merkle root."""
        from aragora.observability.immutable_log import ImmutableAuditLog

        log = ImmutableAuditLog(backend=MagicMock())
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]

        root1 = log._compute_merkle_root(hashes.copy())
        root2 = log._compute_merkle_root(hashes.copy())

        assert root1 == root2


class TestAuditHelperFunctions:
    """Tests for audit helper functions."""

    @pytest.mark.asyncio
    async def test_audit_finding_created(self):
        """audit_finding_created helper works."""
        from aragora.observability.immutable_log import (
            audit_finding_created,
            get_audit_log,
        )

        # Mock the global log
        mock_log = MagicMock()
        mock_entry = AuditEntry(
            id="test",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type="finding_created",
            actor="system",
            actor_type="system",
            resource_type="finding",
            resource_id="f-123",
            action="create",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )
        mock_log.append = AsyncMock(return_value=mock_entry)

        with patch("aragora.observability.immutable_log._audit_log", mock_log):
            entry = await audit_finding_created(
                finding_id="f-123",
                actor="user@example.com",
                workspace_id="ws-1",
                severity="high",
                category="security",
            )

            # Should have called append with finding_created event
            mock_log.append.assert_called_once()
            call_kwargs = mock_log.append.call_args.kwargs
            assert call_kwargs["event_type"] == "finding_created"
            assert call_kwargs["resource_id"] == "f-123"
