"""
Tests for immutable audit log types.

Tests cover:
- AuditBackend enum values
- AuditEntry creation, to_dict, from_dict round-trip, compute_hash
- DailyAnchor creation and to_dict
- VerificationResult creation, defaults, and to_dict
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.observability.log_types import (
    AuditBackend,
    AuditEntry,
    DailyAnchor,
    VerificationResult,
)


# --- Fixtures ---


@pytest.fixture
def sample_entry() -> AuditEntry:
    return AuditEntry(
        id="entry-001",
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        sequence_number=1,
        previous_hash="0" * 64,
        entry_hash="abc123",
        event_type="finding.created",
        actor="user-42",
        actor_type="user",
        resource_type="finding",
        resource_id="finding-99",
        action="create",
        details={"severity": "high"},
        correlation_id="corr-001",
        workspace_id="ws-10",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
        signature="sig-xyz",
    )


# --- AuditBackend enum tests ---


class TestAuditBackend:
    def test_values(self):
        assert AuditBackend.LOCAL == "local"
        assert AuditBackend.POSTGRESQL == "postgresql"
        assert AuditBackend.S3_OBJECT_LOCK == "s3_object_lock"
        assert AuditBackend.QLDB == "qldb"

    def test_member_count(self):
        assert len(AuditBackend) == 4

    def test_is_str_enum(self):
        assert isinstance(AuditBackend.LOCAL, str)


# --- AuditEntry tests ---


class TestAuditEntry:
    def test_defaults(self):
        entry = AuditEntry(
            id="e1",
            timestamp=datetime(2025, 1, 1),
            sequence_number=0,
            previous_hash="",
            entry_hash="",
            event_type="test",
            actor="sys",
            actor_type="system",
            resource_type="doc",
            resource_id="d1",
            action="access",
        )
        assert entry.details == {}
        assert entry.correlation_id is None
        assert entry.workspace_id is None
        assert entry.ip_address is None
        assert entry.user_agent is None
        assert entry.signature is None

    def test_to_dict(self, sample_entry: AuditEntry):
        d = sample_entry.to_dict()
        assert d["id"] == "entry-001"
        assert d["timestamp"] == "2025-01-15T12:00:00"
        assert d["sequence_number"] == 1
        assert d["event_type"] == "finding.created"
        assert d["actor"] == "user-42"
        assert d["actor_type"] == "user"
        assert d["details"] == {"severity": "high"}
        assert d["correlation_id"] == "corr-001"
        assert d["workspace_id"] == "ws-10"
        assert d["ip_address"] == "192.168.1.1"
        assert d["user_agent"] == "TestAgent/1.0"
        assert d["signature"] == "sig-xyz"

    def test_from_dict_round_trip(self, sample_entry: AuditEntry):
        d = sample_entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.id == sample_entry.id
        assert restored.timestamp == sample_entry.timestamp
        assert restored.sequence_number == sample_entry.sequence_number
        assert restored.event_type == sample_entry.event_type
        assert restored.details == sample_entry.details
        assert restored.correlation_id == sample_entry.correlation_id
        assert restored.signature == sample_entry.signature

    def test_from_dict_optional_defaults(self):
        """from_dict fills optional fields with defaults when missing."""
        minimal = {
            "id": "e2",
            "timestamp": "2025-06-01T00:00:00",
            "sequence_number": 5,
            "previous_hash": "prev",
            "entry_hash": "hash",
            "event_type": "access",
            "actor": "bot",
            "resource_type": "doc",
            "resource_id": "d5",
            "action": "read",
        }
        entry = AuditEntry.from_dict(minimal)
        assert entry.actor_type == "user"  # default
        assert entry.details == {}
        assert entry.correlation_id is None
        assert entry.workspace_id is None

    def test_compute_hash_deterministic(self, sample_entry: AuditEntry):
        h1 = sample_entry.compute_hash()
        h2 = sample_entry.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_hash_changes_on_mutation(self, sample_entry: AuditEntry):
        h1 = sample_entry.compute_hash()
        sample_entry.action = "delete"
        h2 = sample_entry.compute_hash()
        assert h1 != h2


# --- DailyAnchor tests ---


class TestDailyAnchor:
    def test_to_dict(self):
        anchor = DailyAnchor(
            date="2025-01-15",
            first_sequence=1,
            last_sequence=100,
            entry_count=100,
            merkle_root="mroot",
            chain_hash="chash",
            created_at=datetime(2025, 1, 16, 0, 0, 0),
        )
        d = anchor.to_dict()
        assert d["date"] == "2025-01-15"
        assert d["first_sequence"] == 1
        assert d["last_sequence"] == 100
        assert d["entry_count"] == 100
        assert d["merkle_root"] == "mroot"
        assert d["chain_hash"] == "chash"
        assert d["created_at"] == "2025-01-16T00:00:00"


# --- VerificationResult tests ---


class TestVerificationResult:
    def test_valid_result(self):
        vr = VerificationResult(
            is_valid=True,
            entries_checked=50,
            errors=[],
            warnings=[],
        )
        assert vr.is_valid is True
        assert vr.first_error_sequence is None
        assert vr.verification_time_ms == 0.0

    def test_invalid_result_with_errors(self):
        vr = VerificationResult(
            is_valid=False,
            entries_checked=30,
            errors=["hash mismatch at seq 12"],
            warnings=["gap detected"],
            first_error_sequence=12,
            verification_time_ms=42.5,
        )
        assert vr.is_valid is False
        assert len(vr.errors) == 1
        assert vr.first_error_sequence == 12

    def test_to_dict(self):
        vr = VerificationResult(
            is_valid=True,
            entries_checked=10,
            errors=[],
            warnings=["minor"],
            verification_time_ms=1.5,
        )
        d = vr.to_dict()
        assert d["is_valid"] is True
        assert d["entries_checked"] == 10
        assert d["errors"] == []
        assert d["warnings"] == ["minor"]
        assert d["first_error_sequence"] is None
        assert d["verification_time_ms"] == 1.5
