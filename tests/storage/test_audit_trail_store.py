"""
Tests for AuditTrailStore - Database persistence for audit trails and decision receipts.

Tests cover:
- AuditTrailStore initialization with SQLite backend
- Audit trail CRUD operations (save, get, list, count)
- Decision receipt CRUD operations (save, get, list, count)
- Search by gauntlet_id and other filters
- Pagination support
- Retention policy enforcement (cleanup_expired)
- Trail-to-receipt linking
- Data format validation
- Concurrent writes
- Dataclass serialization
- Factory functions (get_audit_trail_store, reset_audit_trail_store)
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.audit_trail_store import (
    AuditTrailStore,
    StoredReceipt,
    StoredTrail,
    get_audit_trail_store,
    reset_audit_trail_store,
    DEFAULT_RETENTION_DAYS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_audit_trails.db"


@pytest.fixture
def audit_trail_store(temp_db_path):
    """Create an audit trail store for testing."""
    store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
    yield store
    store.close()


@pytest.fixture
def sample_trail_dict():
    """Create a sample audit trail dictionary."""
    return {
        "trail_id": "trail-001",
        "gauntlet_id": "gauntlet-001",
        "created_at": time.time(),
        "verdict": "APPROVED",
        "confidence": 0.85,
        "total_findings": 5,
        "duration_seconds": 12.5,
        "receipt_id": None,
        "events": [
            {"type": "start", "timestamp": time.time() - 10},
            {"type": "finding", "timestamp": time.time() - 5},
            {"type": "end", "timestamp": time.time()},
        ],
    }


@pytest.fixture
def sample_receipt_dict():
    """Create a sample decision receipt dictionary."""
    return {
        "receipt_id": "receipt-001",
        "gauntlet_id": "gauntlet-001",
        "timestamp": time.time(),
        "verdict": "APPROVED",
        "confidence": 0.85,
        "risk_level": "MEDIUM",
        "checksum": "sha256:abc123def456",
        "audit_trail_id": "trail-001",
        "statement": "Test statement for receipt",
        "findings": [],
    }


@pytest.fixture
def populated_store(audit_trail_store, sample_trail_dict, sample_receipt_dict):
    """Audit trail store with sample data."""
    # Add trails
    audit_trail_store.save_trail(sample_trail_dict)

    trail2 = sample_trail_dict.copy()
    trail2["trail_id"] = "trail-002"
    trail2["gauntlet_id"] = "gauntlet-002"
    trail2["verdict"] = "REJECTED"
    trail2["confidence"] = 0.92
    audit_trail_store.save_trail(trail2)

    trail3 = sample_trail_dict.copy()
    trail3["trail_id"] = "trail-003"
    trail3["gauntlet_id"] = "gauntlet-003"
    trail3["verdict"] = "APPROVED"
    trail3["confidence"] = 0.78
    audit_trail_store.save_trail(trail3)

    # Add receipts
    audit_trail_store.save_receipt(sample_receipt_dict)

    receipt2 = sample_receipt_dict.copy()
    receipt2["receipt_id"] = "receipt-002"
    receipt2["gauntlet_id"] = "gauntlet-002"
    receipt2["verdict"] = "REJECTED"
    receipt2["risk_level"] = "HIGH"
    audit_trail_store.save_receipt(receipt2)

    return audit_trail_store


# =============================================================================
# StoredTrail Dataclass Tests
# =============================================================================


class TestStoredTrail:
    """Tests for StoredTrail dataclass."""

    def test_to_dict_basic_fields(self):
        """Test to_dict includes all basic fields."""
        trail = StoredTrail(
            trail_id="trail-001",
            gauntlet_id="gauntlet-001",
            created_at=1700000000.0,
            verdict="APPROVED",
            confidence=0.85,
            total_findings=5,
            duration_seconds=12.5,
            receipt_id="receipt-001",
            data={"events": []},
        )

        result = trail.to_dict()

        assert result["trail_id"] == "trail-001"
        assert result["gauntlet_id"] == "gauntlet-001"
        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.85
        assert result["total_findings"] == 5
        assert result["receipt_id"] == "receipt-001"

    def test_to_dict_includes_data_fields(self):
        """Test to_dict merges data fields into result."""
        trail = StoredTrail(
            trail_id="trail-001",
            gauntlet_id="gauntlet-001",
            created_at=1700000000.0,
            verdict="APPROVED",
            confidence=0.85,
            total_findings=5,
            duration_seconds=12.5,
            receipt_id=None,
            data={"events": [{"type": "start"}], "custom_field": "value"},
        )

        result = trail.to_dict()

        assert result["events"] == [{"type": "start"}]
        assert result["custom_field"] == "value"


# =============================================================================
# StoredReceipt Dataclass Tests
# =============================================================================


class TestStoredReceipt:
    """Tests for StoredReceipt dataclass."""

    def test_to_dict_basic_fields(self):
        """Test to_dict includes all basic fields."""
        receipt = StoredReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-001",
            created_at=1700000000.0,
            verdict="APPROVED",
            confidence=0.85,
            risk_level="MEDIUM",
            checksum="sha256:abc123",
            audit_trail_id="trail-001",
            data={"statement": "Test"},
        )

        result = receipt.to_dict()

        assert result["receipt_id"] == "receipt-001"
        assert result["gauntlet_id"] == "gauntlet-001"
        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.85
        assert result["risk_level"] == "MEDIUM"
        assert result["checksum"] == "sha256:abc123"
        assert result["audit_trail_id"] == "trail-001"

    def test_to_dict_includes_data_fields(self):
        """Test to_dict merges data fields into result."""
        receipt = StoredReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-001",
            created_at=1700000000.0,
            verdict="APPROVED",
            confidence=0.85,
            risk_level="LOW",
            checksum="sha256:xyz",
            audit_trail_id=None,
            data={"statement": "Test statement", "findings": [{"id": 1}]},
        )

        result = receipt.to_dict()

        assert result["statement"] == "Test statement"
        assert result["findings"] == [{"id": 1}]


# =============================================================================
# AuditTrailStore Initialization Tests
# =============================================================================


class TestAuditTrailStoreInit:
    """Tests for AuditTrailStore initialization."""

    def test_init_with_sqlite_backend(self, temp_db_path):
        """Should initialize with SQLite backend."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")

        assert store.backend_type == "sqlite"
        assert store._backend is not None

        store.close()

    def test_init_creates_tables(self, temp_db_path):
        """Should create required tables on init."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")

        # Try to query tables to verify they exist
        trails = store.list_trails()
        assert trails == []

        receipts = store.list_receipts()
        assert receipts == []

        store.close()

    def test_init_creates_parent_directory(self, temp_db_path):
        """Should create parent directory if needed."""
        nested_path = temp_db_path.parent / "nested" / "path" / "audit.db"
        store = AuditTrailStore(db_path=nested_path, backend="sqlite")

        assert nested_path.parent.exists()

        store.close()

    def test_init_postgresql_requires_url(self, temp_db_path):
        """PostgreSQL backend requires database_url."""
        with pytest.raises(ValueError, match="PostgreSQL backend requires DATABASE_URL"):
            AuditTrailStore(db_path=temp_db_path, backend="postgresql")

    def test_default_retention_days(self, temp_db_path):
        """Should use default retention days."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
        assert store.retention_days == DEFAULT_RETENTION_DAYS
        store.close()

    def test_custom_retention_days(self, temp_db_path):
        """Should accept custom retention days."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite", retention_days=30)
        assert store.retention_days == 30
        store.close()


# =============================================================================
# Audit Trail CRUD Tests
# =============================================================================


class TestAuditTrailCRUD:
    """Tests for audit trail CRUD operations."""

    def test_save_and_get_trail(self, audit_trail_store, sample_trail_dict):
        """Test save and retrieve an audit trail."""
        audit_trail_store.save_trail(sample_trail_dict)

        trail = audit_trail_store.get_trail("trail-001")
        assert trail is not None
        assert trail["trail_id"] == "trail-001"
        assert trail["gauntlet_id"] == "gauntlet-001"
        assert trail["verdict"] == "APPROVED"

    def test_get_nonexistent_trail(self, audit_trail_store):
        """Test get returns None for nonexistent trail."""
        result = audit_trail_store.get_trail("nonexistent-id")
        assert result is None

    def test_save_updates_existing_trail(self, audit_trail_store, sample_trail_dict):
        """Test save updates existing trail (upsert)."""
        audit_trail_store.save_trail(sample_trail_dict)

        # Update verdict
        sample_trail_dict["verdict"] = "REJECTED"
        sample_trail_dict["confidence"] = 0.95
        audit_trail_store.save_trail(sample_trail_dict)

        trail = audit_trail_store.get_trail("trail-001")
        assert trail["verdict"] == "REJECTED"
        assert trail["confidence"] == 0.95

    def test_get_trail_by_gauntlet(self, audit_trail_store, sample_trail_dict):
        """Test retrieve trail by gauntlet_id."""
        audit_trail_store.save_trail(sample_trail_dict)

        trail = audit_trail_store.get_trail_by_gauntlet("gauntlet-001")
        assert trail is not None
        assert trail["trail_id"] == "trail-001"

    def test_get_trail_by_gauntlet_nonexistent(self, audit_trail_store):
        """Test get_trail_by_gauntlet returns None for nonexistent."""
        result = audit_trail_store.get_trail_by_gauntlet("nonexistent")
        assert result is None

    def test_save_trail_with_iso_timestamp(self, audit_trail_store, sample_trail_dict):
        """Test save handles ISO timestamp format."""
        sample_trail_dict["created_at"] = "2024-01-15T10:30:00+00:00"
        audit_trail_store.save_trail(sample_trail_dict)

        trail = audit_trail_store.get_trail("trail-001")
        assert trail is not None

    def test_data_integrity_save_retrieve(self, audit_trail_store, sample_trail_dict):
        """Test data integrity - save then retrieve returns same data."""
        audit_trail_store.save_trail(sample_trail_dict)

        trail = audit_trail_store.get_trail("trail-001")

        assert trail["trail_id"] == sample_trail_dict["trail_id"]
        assert trail["gauntlet_id"] == sample_trail_dict["gauntlet_id"]
        assert trail["verdict"] == sample_trail_dict["verdict"]
        assert trail["confidence"] == sample_trail_dict["confidence"]
        assert trail["total_findings"] == sample_trail_dict["total_findings"]
        assert len(trail["events"]) == len(sample_trail_dict["events"])


# =============================================================================
# Audit Trail List and Pagination Tests
# =============================================================================


class TestAuditTrailList:
    """Tests for audit trail list and filtering."""

    def test_list_empty(self, audit_trail_store):
        """Test list returns empty for empty store."""
        trails = audit_trail_store.list_trails()
        assert trails == []

    def test_list_multiple(self, populated_store):
        """Test list returns multiple trails."""
        trails = populated_store.list_trails(limit=10)
        assert len(trails) == 3

    def test_list_pagination_limit(self, populated_store):
        """Test list respects limit parameter."""
        trails = populated_store.list_trails(limit=2)
        assert len(trails) == 2

    def test_list_pagination_offset(self, populated_store):
        """Test list respects offset parameter."""
        page1 = populated_store.list_trails(limit=2, offset=0)
        page2 = populated_store.list_trails(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 1
        # Different results
        ids1 = {t["trail_id"] for t in page1}
        ids2 = {t["trail_id"] for t in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_filter_by_verdict(self, populated_store):
        """Test list filters by verdict."""
        approved = populated_store.list_trails(verdict="APPROVED")
        rejected = populated_store.list_trails(verdict="REJECTED")

        assert len(approved) == 2
        assert len(rejected) == 1
        assert all(t["verdict"] == "APPROVED" for t in approved)
        assert all(t["verdict"] == "REJECTED" for t in rejected)

    def test_count_trails(self, populated_store):
        """Test count trails."""
        count = populated_store.count_trails()
        assert count == 3

    def test_count_trails_with_filter(self, populated_store):
        """Test count trails with verdict filter."""
        approved_count = populated_store.count_trails(verdict="APPROVED")
        rejected_count = populated_store.count_trails(verdict="REJECTED")

        assert approved_count == 2
        assert rejected_count == 1


# =============================================================================
# Decision Receipt CRUD Tests
# =============================================================================


class TestDecisionReceiptCRUD:
    """Tests for decision receipt CRUD operations."""

    def test_save_and_get_receipt(self, audit_trail_store, sample_receipt_dict):
        """Test save and retrieve a decision receipt."""
        audit_trail_store.save_receipt(sample_receipt_dict)

        receipt = audit_trail_store.get_receipt("receipt-001")
        assert receipt is not None
        assert receipt["receipt_id"] == "receipt-001"
        assert receipt["gauntlet_id"] == "gauntlet-001"
        assert receipt["verdict"] == "APPROVED"

    def test_get_nonexistent_receipt(self, audit_trail_store):
        """Test get returns None for nonexistent receipt."""
        result = audit_trail_store.get_receipt("nonexistent-id")
        assert result is None

    def test_save_updates_existing_receipt(self, audit_trail_store, sample_receipt_dict):
        """Test save updates existing receipt (upsert)."""
        audit_trail_store.save_receipt(sample_receipt_dict)

        # Update verdict
        sample_receipt_dict["verdict"] = "REJECTED"
        sample_receipt_dict["risk_level"] = "HIGH"
        audit_trail_store.save_receipt(sample_receipt_dict)

        receipt = audit_trail_store.get_receipt("receipt-001")
        assert receipt["verdict"] == "REJECTED"
        assert receipt["risk_level"] == "HIGH"

    def test_get_receipt_by_gauntlet(self, audit_trail_store, sample_receipt_dict):
        """Test retrieve receipt by gauntlet_id."""
        audit_trail_store.save_receipt(sample_receipt_dict)

        receipt = audit_trail_store.get_receipt_by_gauntlet("gauntlet-001")
        assert receipt is not None
        assert receipt["receipt_id"] == "receipt-001"

    def test_get_receipt_by_gauntlet_nonexistent(self, audit_trail_store):
        """Test get_receipt_by_gauntlet returns None for nonexistent."""
        result = audit_trail_store.get_receipt_by_gauntlet("nonexistent")
        assert result is None

    def test_save_receipt_with_iso_timestamp(self, audit_trail_store, sample_receipt_dict):
        """Test save handles ISO timestamp format."""
        sample_receipt_dict["timestamp"] = "2024-01-15T10:30:00+00:00"
        audit_trail_store.save_receipt(sample_receipt_dict)

        receipt = audit_trail_store.get_receipt("receipt-001")
        assert receipt is not None

    def test_data_integrity_save_retrieve(self, audit_trail_store, sample_receipt_dict):
        """Test data integrity - save then retrieve returns same data."""
        audit_trail_store.save_receipt(sample_receipt_dict)

        receipt = audit_trail_store.get_receipt("receipt-001")

        assert receipt["receipt_id"] == sample_receipt_dict["receipt_id"]
        assert receipt["gauntlet_id"] == sample_receipt_dict["gauntlet_id"]
        assert receipt["verdict"] == sample_receipt_dict["verdict"]
        assert receipt["confidence"] == sample_receipt_dict["confidence"]
        assert receipt["risk_level"] == sample_receipt_dict["risk_level"]
        assert receipt["checksum"] == sample_receipt_dict["checksum"]


# =============================================================================
# Decision Receipt List and Pagination Tests
# =============================================================================


class TestDecisionReceiptList:
    """Tests for decision receipt list and filtering."""

    def test_list_empty(self, audit_trail_store):
        """Test list returns empty for empty store."""
        receipts = audit_trail_store.list_receipts()
        assert receipts == []

    def test_list_multiple(self, populated_store):
        """Test list returns multiple receipts."""
        receipts = populated_store.list_receipts(limit=10)
        assert len(receipts) == 2

    def test_list_pagination_limit(self, populated_store):
        """Test list respects limit parameter."""
        receipts = populated_store.list_receipts(limit=1)
        assert len(receipts) == 1

    def test_list_pagination_offset(self, populated_store):
        """Test list respects offset parameter."""
        page1 = populated_store.list_receipts(limit=1, offset=0)
        page2 = populated_store.list_receipts(limit=1, offset=1)

        assert len(page1) == 1
        assert len(page2) == 1
        assert page1[0]["receipt_id"] != page2[0]["receipt_id"]

    def test_list_filter_by_verdict(self, populated_store):
        """Test list filters by verdict."""
        approved = populated_store.list_receipts(verdict="APPROVED")
        rejected = populated_store.list_receipts(verdict="REJECTED")

        assert len(approved) == 1
        assert len(rejected) == 1
        assert approved[0]["verdict"] == "APPROVED"
        assert rejected[0]["verdict"] == "REJECTED"

    def test_list_filter_by_risk_level(self, populated_store):
        """Test list filters by risk level."""
        medium_risk = populated_store.list_receipts(risk_level="MEDIUM")
        high_risk = populated_store.list_receipts(risk_level="HIGH")

        assert len(medium_risk) == 1
        assert len(high_risk) == 1
        assert medium_risk[0]["risk_level"] == "MEDIUM"
        assert high_risk[0]["risk_level"] == "HIGH"

    def test_list_filter_combined(self, populated_store):
        """Test list with multiple filters."""
        receipts = populated_store.list_receipts(verdict="REJECTED", risk_level="HIGH")

        assert len(receipts) == 1
        assert receipts[0]["verdict"] == "REJECTED"
        assert receipts[0]["risk_level"] == "HIGH"

    def test_count_receipts(self, populated_store):
        """Test count receipts."""
        count = populated_store.count_receipts()
        assert count == 2

    def test_count_receipts_with_filter(self, populated_store):
        """Test count receipts with filters."""
        approved_count = populated_store.count_receipts(verdict="APPROVED")
        high_risk_count = populated_store.count_receipts(risk_level="HIGH")

        assert approved_count == 1
        assert high_risk_count == 1


# =============================================================================
# Trail-Receipt Linking Tests
# =============================================================================


class TestTrailReceiptLinking:
    """Tests for linking trails and receipts."""

    def test_link_trail_to_receipt(self, audit_trail_store, sample_trail_dict):
        """Test link_trail_to_receipt updates trail with receipt reference."""
        audit_trail_store.save_trail(sample_trail_dict)

        audit_trail_store.link_trail_to_receipt("trail-001", "receipt-xyz")

        # Verify via list (which returns summary data)
        trails = audit_trail_store.list_trails()
        trail = next(t for t in trails if t["trail_id"] == "trail-001")
        assert trail["receipt_id"] == "receipt-xyz"

    def test_link_receipt_to_trail(self, audit_trail_store, sample_receipt_dict):
        """Test link_receipt_to_trail updates receipt with trail reference."""
        sample_receipt_dict["audit_trail_id"] = None
        audit_trail_store.save_receipt(sample_receipt_dict)

        audit_trail_store.link_receipt_to_trail("receipt-001", "trail-xyz")

        # Verify via list (which returns summary data)
        receipts = audit_trail_store.list_receipts()
        receipt = next(r for r in receipts if r["receipt_id"] == "receipt-001")
        assert receipt["audit_trail_id"] == "trail-xyz"


# =============================================================================
# Retention Policy / Cleanup Tests
# =============================================================================


class TestRetentionPolicy:
    """Tests for retention policy enforcement."""

    def test_cleanup_expired_removes_old_trails(self, audit_trail_store, sample_trail_dict):
        """Test cleanup_expired removes old audit trails."""
        # Save a recent trail
        audit_trail_store.save_trail(sample_trail_dict)

        # Manually insert an old trail via direct database access
        old_created_at = time.time() - (400 * 86400)  # 400 days ago
        old_trail = sample_trail_dict.copy()
        old_trail["trail_id"] = "old-trail"
        old_trail["gauntlet_id"] = "old-gauntlet"
        old_trail["created_at"] = old_created_at

        # Need to directly manipulate database for old timestamps
        import sqlite3

        conn = sqlite3.connect(str(audit_trail_store.db_path))
        conn.execute(
            """
            INSERT INTO audit_trails
            (trail_id, gauntlet_id, created_at, verdict, confidence,
             total_findings, duration_seconds, receipt_id, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-trail",
                "old-gauntlet",
                old_created_at,
                "APPROVED",
                0.85,
                0,
                0.0,
                None,
                json.dumps(old_trail),
            ),
        )
        conn.commit()
        conn.close()

        # Verify both exist
        assert audit_trail_store.count_trails() == 2

        # Cleanup
        removed = audit_trail_store.cleanup_expired()

        assert removed >= 1
        assert audit_trail_store.count_trails() == 1
        assert audit_trail_store.get_trail("trail-001") is not None
        assert audit_trail_store.get_trail("old-trail") is None

    def test_cleanup_expired_removes_old_receipts(self, audit_trail_store, sample_receipt_dict):
        """Test cleanup_expired removes old receipts."""
        # Save a recent receipt
        audit_trail_store.save_receipt(sample_receipt_dict)

        # Manually insert an old receipt
        old_created_at = time.time() - (400 * 86400)  # 400 days ago

        import sqlite3

        conn = sqlite3.connect(str(audit_trail_store.db_path))
        conn.execute(
            """
            INSERT INTO decision_receipts
            (receipt_id, gauntlet_id, created_at, verdict, confidence,
             risk_level, checksum, audit_trail_id, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-receipt",
                "old-gauntlet",
                old_created_at,
                "APPROVED",
                0.85,
                "LOW",
                "sha256:old",
                None,
                json.dumps({}),
            ),
        )
        conn.commit()
        conn.close()

        # Verify both exist
        assert audit_trail_store.count_receipts() == 2

        # Cleanup
        removed = audit_trail_store.cleanup_expired()

        assert removed >= 1
        assert audit_trail_store.count_receipts() == 1
        assert audit_trail_store.get_receipt("receipt-001") is not None
        assert audit_trail_store.get_receipt("old-receipt") is None

    def test_cleanup_returns_zero_when_nothing_expired(self, audit_trail_store, sample_trail_dict):
        """Should return 0 when no entries are expired."""
        audit_trail_store.save_trail(sample_trail_dict)

        removed = audit_trail_store.cleanup_expired()
        assert removed == 0


# =============================================================================
# Concurrent Write Tests
# =============================================================================


class TestConcurrentWrites:
    """Tests for concurrent write operations."""

    def test_concurrent_trail_saves(self, audit_trail_store):
        """Test concurrent saves don't cause data corruption."""
        num_threads = 5
        trails_per_thread = 10
        errors = []

        def save_trails(thread_id):
            try:
                for i in range(trails_per_thread):
                    trail_dict = {
                        "trail_id": f"trail-{thread_id}-{i}",
                        "gauntlet_id": f"gauntlet-{thread_id}-{i}",
                        "created_at": time.time(),
                        "verdict": "APPROVED",
                        "confidence": 0.85,
                        "total_findings": i,
                        "duration_seconds": float(i),
                        "events": [],
                    }
                    audit_trail_store.save_trail(trail_dict)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=save_trails, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors occurred: {errors}"

        # Verify all trails were saved
        count = audit_trail_store.count_trails()
        assert count == num_threads * trails_per_thread

    def test_concurrent_receipt_saves(self, audit_trail_store):
        """Test concurrent receipt saves don't cause data corruption."""
        num_threads = 5
        receipts_per_thread = 10
        errors = []

        def save_receipts(thread_id):
            try:
                for i in range(receipts_per_thread):
                    receipt_dict = {
                        "receipt_id": f"receipt-{thread_id}-{i}",
                        "gauntlet_id": f"gauntlet-{thread_id}-{i}",
                        "timestamp": time.time(),
                        "verdict": "APPROVED",
                        "confidence": 0.85,
                        "risk_level": "MEDIUM",
                        "checksum": f"sha256:{thread_id}{i}",
                    }
                    audit_trail_store.save_receipt(receipt_dict)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=save_receipts, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors occurred: {errors}"

        # Verify all receipts were saved
        count = audit_trail_store.count_receipts()
        assert count == num_threads * receipts_per_thread


# =============================================================================
# Data Format Validation Tests
# =============================================================================


class TestDataFormatValidation:
    """Tests for data format handling."""

    def test_save_trail_with_missing_optional_fields(self, audit_trail_store):
        """Test save trail handles missing optional fields."""
        minimal_trail = {
            "trail_id": "minimal-trail",
            "gauntlet_id": "minimal-gauntlet",
            "verdict": "APPROVED",
        }

        audit_trail_store.save_trail(minimal_trail)

        trail = audit_trail_store.get_trail("minimal-trail")
        assert trail is not None
        assert trail["trail_id"] == "minimal-trail"

    def test_save_receipt_with_missing_optional_fields(self, audit_trail_store):
        """Test save receipt handles missing optional fields."""
        minimal_receipt = {
            "receipt_id": "minimal-receipt",
            "gauntlet_id": "minimal-gauntlet",
            "verdict": "APPROVED",
        }

        audit_trail_store.save_receipt(minimal_receipt)

        receipt = audit_trail_store.get_receipt("minimal-receipt")
        assert receipt is not None
        assert receipt["receipt_id"] == "minimal-receipt"

    def test_save_trail_with_complex_data(self, audit_trail_store):
        """Test save trail preserves complex nested data."""
        complex_trail = {
            "trail_id": "complex-trail",
            "gauntlet_id": "complex-gauntlet",
            "created_at": time.time(),
            "verdict": "APPROVED",
            "confidence": 0.85,
            "total_findings": 3,
            "duration_seconds": 10.5,
            "events": [
                {"type": "start", "timestamp": time.time(), "data": {"key": "value"}},
                {"type": "finding", "finding_id": "f1", "severity": "HIGH"},
                {"type": "end", "timestamp": time.time()},
            ],
            "metadata": {
                "agents": ["claude", "gpt4"],
                "config": {"rounds": 3, "threshold": 0.8},
            },
        }

        audit_trail_store.save_trail(complex_trail)

        trail = audit_trail_store.get_trail("complex-trail")
        assert len(trail["events"]) == 3
        assert trail["events"][1]["finding_id"] == "f1"
        assert trail["metadata"]["agents"] == ["claude", "gpt4"]


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for data persistence across store instances."""

    def test_data_persists_across_instances(self, temp_db_path, sample_trail_dict):
        """Data should persist after store is recreated."""
        # Create and populate first store
        store1 = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
        store1.save_trail(sample_trail_dict)
        store1.close()

        # Create new store instance
        store2 = AuditTrailStore(db_path=temp_db_path, backend="sqlite")

        # Verify data persists
        trail = store2.get_trail("trail-001")
        assert trail is not None
        assert trail["gauntlet_id"] == "gauntlet-001"

        store2.close()


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_reset_audit_trail_store(self, temp_db_path):
        """Should reset the default store instance."""
        reset_audit_trail_store()

        # Mock dependencies to avoid production guards
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_DATA_DIR": str(temp_db_path.parent),
                "ARAGORA_ENVIRONMENT": "development",
                "ARAGORA_AUDIT_TRAIL_STORE_BACKEND": "sqlite",
            },
        ):
            with patch(
                "aragora.storage.connection_factory.resolve_database_config"
            ) as mock_resolve:
                from aragora.storage.connection_factory import StorageBackendType

                mock_config = MagicMock()
                mock_config.backend_type = StorageBackendType.SQLITE
                mock_config.dsn = None
                mock_resolve.return_value = mock_config

                with patch("aragora.storage.production_guards.require_distributed_store"):
                    store1 = get_audit_trail_store(db_path=temp_db_path)
                    reset_audit_trail_store()
                    store2 = get_audit_trail_store(db_path=temp_db_path)

                    # Should be different instances
                    assert store1 is not store2

        reset_audit_trail_store()


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    """Tests for close method."""

    def test_close_backend(self, temp_db_path):
        """Should close backend connection."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
        store.save_trail(
            {
                "trail_id": "test",
                "gauntlet_id": "test-gauntlet",
                "verdict": "APPROVED",
            }
        )

        store.close()
        assert store._backend is None

    def test_close_is_idempotent(self, temp_db_path):
        """Should handle multiple close calls."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
        store.close()
        store.close()  # Should not raise


# =============================================================================
# None Backend Guard Tests
# =============================================================================


class TestNoneBackendGuards:
    """Tests for operations when backend is None."""

    def test_operations_with_none_backend(self, temp_db_path):
        """Operations should handle None backend gracefully."""
        store = AuditTrailStore(db_path=temp_db_path, backend="sqlite")
        store.close()  # This sets _backend to None

        # All these should not raise, but return empty/None/0
        assert store.get_trail("test") is None
        assert store.get_trail_by_gauntlet("test") is None
        assert store.list_trails() == []
        assert store.count_trails() == 0

        assert store.get_receipt("test") is None
        assert store.get_receipt_by_gauntlet("test") is None
        assert store.list_receipts() == []
        assert store.count_receipts() == 0

        assert store.cleanup_expired() == 0

        # These should not raise
        store.save_trail({"trail_id": "test", "gauntlet_id": "g", "verdict": "A"})
        store.save_receipt({"receipt_id": "test", "gauntlet_id": "g", "verdict": "A"})
        store.link_trail_to_receipt("t", "r")
        store.link_receipt_to_trail("r", "t")


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        import aragora.storage.audit_trail_store as module

        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.audit_trail_store import (
            AuditTrailStore,
            StoredReceipt,
            StoredTrail,
            get_audit_trail_store,
            reset_audit_trail_store,
        )

        assert AuditTrailStore is not None
        assert StoredTrail is not None
        assert StoredReceipt is not None
        assert callable(get_audit_trail_store)
        assert callable(reset_audit_trail_store)
