"""
Tests for ReceiptStore - Decision receipt storage with signature support.

Tests cover:
- CRUD operations (save, get, list, count)
- Signature operations (update_signature, verify_signature, verify_batch)
- Integrity verification
- Retention/cleanup
- PostgreSQL and SQLite backends
- Filtering and pagination
"""

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.receipt_store import (
    DEFAULT_RETENTION_DAYS,
    ReceiptStore,
    SignatureVerificationResult,
    StoredReceipt,
    get_receipt_store,
    set_receipt_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_receipts.db"


@pytest.fixture
def receipt_store(temp_db_path):
    """Create a receipt store for testing."""
    store = ReceiptStore(db_path=temp_db_path, backend="sqlite")
    yield store


@pytest.fixture
def sample_receipt_dict():
    """Create a sample receipt dictionary."""
    return {
        "receipt_id": "receipt-001",
        "gauntlet_id": "gauntlet-001",
        "debate_id": "debate-001",
        "timestamp": time.time(),
        "verdict": "APPROVED",
        "confidence": 0.85,
        "risk_level": "MEDIUM",
        "risk_score": 0.35,
        "checksum": "sha256:abc123def456",
        "audit_trail_id": "audit-001",
        "statement": "Test statement for receipt",
        "findings": [],
    }


@pytest.fixture
def signed_receipt_dict():
    """Create a sample signed receipt."""
    return {
        "signature": "base64encodedSignature==",
        "signature_metadata": {
            "algorithm": "HMAC-SHA256",
            "key_id": "key-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ===========================================================================
# StoredReceipt Dataclass Tests
# ===========================================================================


class TestStoredReceipt:
    """Tests for StoredReceipt dataclass."""

    def test_to_dict_basic_fields(self):
        """Test to_dict includes all basic fields."""
        receipt = StoredReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-001",
            debate_id="debate-001",
            created_at=1700000000.0,
            expires_at=1800000000.0,
            verdict="APPROVED",
            confidence=0.85,
            risk_level="MEDIUM",
            risk_score=0.35,
            checksum="sha256:abc123",
        )

        result = receipt.to_dict()

        assert result["receipt_id"] == "receipt-001"
        assert result["gauntlet_id"] == "gauntlet-001"
        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.85
        assert result["is_signed"] is False
        assert "signature_metadata" not in result

    def test_to_dict_with_signature(self):
        """Test to_dict includes signature metadata when signed."""
        receipt = StoredReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-001",
            debate_id=None,
            created_at=1700000000.0,
            expires_at=None,
            verdict="REJECTED",
            confidence=0.95,
            risk_level="HIGH",
            risk_score=0.75,
            checksum="sha256:xyz789",
            signature="base64sig==",
            signature_algorithm="HMAC-SHA256",
            signature_key_id="key-001",
            signed_at=1700001000.0,
        )

        result = receipt.to_dict()

        assert result["is_signed"] is True
        assert result["signature_metadata"]["algorithm"] == "HMAC-SHA256"
        assert result["signature_metadata"]["key_id"] == "key-001"
        assert result["signature_metadata"]["signed_at"] == 1700001000.0

    def test_to_full_dict_includes_data(self):
        """Test to_full_dict includes data payload."""
        receipt = StoredReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-001",
            debate_id=None,
            created_at=1700000000.0,
            expires_at=None,
            verdict="APPROVED",
            confidence=0.85,
            risk_level="LOW",
            risk_score=0.15,
            checksum="sha256:abc",
            data={"statement": "Test", "extra_field": "value"},
        )

        result = receipt.to_full_dict()

        assert result["statement"] == "Test"
        assert result["extra_field"] == "value"
        assert result["receipt_id"] == "receipt-001"


class TestSignatureVerificationResult:
    """Tests for SignatureVerificationResult dataclass."""

    def test_to_dict_valid(self):
        """Test to_dict for valid signature."""
        result = SignatureVerificationResult(
            receipt_id="receipt-001",
            is_valid=True,
            algorithm="HMAC-SHA256",
            key_id="key-001",
            signed_at=1700000000.0,
        )

        dict_result = result.to_dict()

        assert dict_result["receipt_id"] == "receipt-001"
        assert dict_result["signature_valid"] is True
        assert dict_result["algorithm"] == "HMAC-SHA256"
        assert dict_result["error"] is None

    def test_to_dict_with_error(self):
        """Test to_dict for invalid signature with error."""
        result = SignatureVerificationResult(
            receipt_id="receipt-001",
            is_valid=False,
            error="Receipt not signed",
        )

        dict_result = result.to_dict()

        assert dict_result["signature_valid"] is False
        assert dict_result["error"] == "Receipt not signed"


# ===========================================================================
# ReceiptStore CRUD Tests
# ===========================================================================


class TestReceiptStoreCRUD:
    """Tests for ReceiptStore CRUD operations."""

    def test_save_and_get(self, receipt_store, sample_receipt_dict):
        """Test save and retrieve a receipt."""
        receipt_id = receipt_store.save(sample_receipt_dict)

        assert receipt_id == "receipt-001"

        receipt = receipt_store.get(receipt_id)
        assert receipt is not None
        assert receipt.receipt_id == "receipt-001"
        assert receipt.gauntlet_id == "gauntlet-001"
        assert receipt.verdict == "APPROVED"
        assert receipt.confidence == 0.85

    def test_save_with_signature(self, receipt_store, sample_receipt_dict, signed_receipt_dict):
        """Test save receipt with signature data."""
        receipt_store.save(sample_receipt_dict, signed_receipt=signed_receipt_dict)

        receipt = receipt_store.get("receipt-001")
        assert receipt is not None
        assert receipt.signature == "base64encodedSignature=="
        assert receipt.signature_algorithm == "HMAC-SHA256"
        assert receipt.signature_key_id == "key-001"

    def test_get_nonexistent(self, receipt_store):
        """Test get returns None for nonexistent receipt."""
        result = receipt_store.get("nonexistent-id")
        assert result is None

    def test_get_by_gauntlet(self, receipt_store, sample_receipt_dict):
        """Test retrieve by gauntlet_id."""
        receipt_store.save(sample_receipt_dict)

        receipt = receipt_store.get_by_gauntlet("gauntlet-001")
        assert receipt is not None
        assert receipt.receipt_id == "receipt-001"

    def test_get_by_gauntlet_nonexistent(self, receipt_store):
        """Test get_by_gauntlet returns None for nonexistent."""
        result = receipt_store.get_by_gauntlet("nonexistent")
        assert result is None

    def test_save_updates_existing(self, receipt_store, sample_receipt_dict):
        """Test save updates existing receipt (upsert)."""
        receipt_store.save(sample_receipt_dict)

        # Update verdict
        sample_receipt_dict["verdict"] = "REJECTED"
        sample_receipt_dict["confidence"] = 0.95
        receipt_store.save(sample_receipt_dict)

        receipt = receipt_store.get("receipt-001")
        assert receipt.verdict == "REJECTED"
        assert receipt.confidence == 0.95

    def test_save_with_iso_timestamp(self, receipt_store, sample_receipt_dict):
        """Test save handles ISO timestamp format."""
        sample_receipt_dict["timestamp"] = "2024-01-15T10:30:00+00:00"
        receipt_store.save(sample_receipt_dict)

        receipt = receipt_store.get("receipt-001")
        assert receipt is not None
        assert receipt.created_at > 0


class TestReceiptStoreList:
    """Tests for ReceiptStore list and filtering."""

    def test_list_empty(self, receipt_store):
        """Test list returns empty for empty store."""
        receipts = receipt_store.list()
        assert receipts == []

    def test_list_multiple(self, receipt_store, sample_receipt_dict):
        """Test list returns multiple receipts."""
        for i in range(5):
            sample_receipt_dict["receipt_id"] = f"receipt-{i:03d}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i:03d}"
            receipt_store.save(sample_receipt_dict)

        receipts = receipt_store.list(limit=10)
        assert len(receipts) == 5

    def test_list_pagination(self, receipt_store, sample_receipt_dict):
        """Test list pagination with limit and offset."""
        for i in range(10):
            sample_receipt_dict["receipt_id"] = f"receipt-{i:03d}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i:03d}"
            receipt_store.save(sample_receipt_dict)

        page1 = receipt_store.list(limit=3, offset=0)
        page2 = receipt_store.list(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].receipt_id != page2[0].receipt_id

    def test_list_filter_by_verdict(self, receipt_store, sample_receipt_dict):
        """Test list filters by verdict."""
        # Add APPROVED receipts
        for i in range(3):
            sample_receipt_dict["receipt_id"] = f"approved-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-a-{i}"
            sample_receipt_dict["verdict"] = "APPROVED"
            receipt_store.save(sample_receipt_dict)

        # Add REJECTED receipts
        for i in range(2):
            sample_receipt_dict["receipt_id"] = f"rejected-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-r-{i}"
            sample_receipt_dict["verdict"] = "REJECTED"
            receipt_store.save(sample_receipt_dict)

        approved = receipt_store.list(verdict="APPROVED")
        rejected = receipt_store.list(verdict="REJECTED")

        assert len(approved) == 3
        assert len(rejected) == 2
        assert all(r.verdict == "APPROVED" for r in approved)
        assert all(r.verdict == "REJECTED" for r in rejected)

    def test_list_filter_by_risk_level(self, receipt_store, sample_receipt_dict):
        """Test list filters by risk level."""
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        for i, level in enumerate(risk_levels):
            sample_receipt_dict["receipt_id"] = f"receipt-{level}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{level}"
            sample_receipt_dict["risk_level"] = level
            receipt_store.save(sample_receipt_dict)

        high_risk = receipt_store.list(risk_level="HIGH")
        assert len(high_risk) == 1
        assert high_risk[0].risk_level == "HIGH"

    def test_list_filter_signed_only(self, receipt_store, sample_receipt_dict, signed_receipt_dict):
        """Test list filters signed receipts only."""
        # Add unsigned receipt
        sample_receipt_dict["receipt_id"] = "unsigned"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-unsigned"
        receipt_store.save(sample_receipt_dict)

        # Add signed receipt
        sample_receipt_dict["receipt_id"] = "signed"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-signed"
        receipt_store.save(sample_receipt_dict, signed_receipt=signed_receipt_dict)

        all_receipts = receipt_store.list()
        signed_only = receipt_store.list(signed_only=True)

        assert len(all_receipts) == 2
        assert len(signed_only) == 1
        assert signed_only[0].receipt_id == "signed"

    def test_list_date_range(self, receipt_store, sample_receipt_dict):
        """Test list filters by date range."""
        now = time.time()

        # Old receipt
        sample_receipt_dict["receipt_id"] = "old"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-old"
        sample_receipt_dict["timestamp"] = now - 86400 * 10  # 10 days ago
        receipt_store.save(sample_receipt_dict)

        # Recent receipt
        sample_receipt_dict["receipt_id"] = "recent"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-recent"
        sample_receipt_dict["timestamp"] = now
        receipt_store.save(sample_receipt_dict)

        recent = receipt_store.list(date_from=now - 86400)  # Last day
        assert len(recent) == 1
        assert recent[0].receipt_id == "recent"

    def test_list_sorting(self, receipt_store, sample_receipt_dict):
        """Test list sorting by different fields."""
        confidences = [0.5, 0.9, 0.7]
        for i, conf in enumerate(confidences):
            sample_receipt_dict["receipt_id"] = f"receipt-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i}"
            sample_receipt_dict["confidence"] = conf
            receipt_store.save(sample_receipt_dict)

        desc = receipt_store.list(sort_by="confidence", order="desc")
        asc = receipt_store.list(sort_by="confidence", order="asc")

        assert desc[0].confidence == 0.9
        assert asc[0].confidence == 0.5

    def test_count(self, receipt_store, sample_receipt_dict):
        """Test count receipts."""
        assert receipt_store.count() == 0

        for i in range(5):
            sample_receipt_dict["receipt_id"] = f"receipt-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i}"
            receipt_store.save(sample_receipt_dict)

        assert receipt_store.count() == 5

    def test_count_with_filters(self, receipt_store, sample_receipt_dict):
        """Test count with filters."""
        for i in range(3):
            sample_receipt_dict["receipt_id"] = f"approved-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-a-{i}"
            sample_receipt_dict["verdict"] = "APPROVED"
            receipt_store.save(sample_receipt_dict)

        for i in range(2):
            sample_receipt_dict["receipt_id"] = f"rejected-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-r-{i}"
            sample_receipt_dict["verdict"] = "REJECTED"
            receipt_store.save(sample_receipt_dict)

        assert receipt_store.count(verdict="APPROVED") == 3
        assert receipt_store.count(verdict="REJECTED") == 2


# ===========================================================================
# Signature Operations Tests
# ===========================================================================


class TestReceiptStoreSignatures:
    """Tests for signature operations."""

    def test_update_signature(self, receipt_store, sample_receipt_dict):
        """Test update_signature adds signature to receipt."""
        receipt_store.save(sample_receipt_dict)

        updated = receipt_store.update_signature(
            receipt_id="receipt-001",
            signature="newSignature==",
            algorithm="Ed25519",
            key_id="key-002",
        )

        assert updated is True

        receipt = receipt_store.get("receipt-001")
        assert receipt.signature == "newSignature=="
        assert receipt.signature_algorithm == "Ed25519"
        assert receipt.signature_key_id == "key-002"
        assert receipt.signed_at is not None

    def test_update_signature_nonexistent(self, receipt_store):
        """Test update_signature returns False for nonexistent receipt."""
        updated = receipt_store.update_signature(
            receipt_id="nonexistent",
            signature="sig",
            algorithm="HMAC-SHA256",
            key_id="key",
        )
        assert updated is False

    def test_verify_signature_not_found(self, receipt_store):
        """Test verify_signature for nonexistent receipt."""
        result = receipt_store.verify_signature("nonexistent")

        assert result.is_valid is False
        assert "not found" in result.error.lower()

    def test_verify_signature_unsigned(self, receipt_store, sample_receipt_dict):
        """Test verify_signature for unsigned receipt."""
        receipt_store.save(sample_receipt_dict)

        result = receipt_store.verify_signature("receipt-001")

        assert result.is_valid is False
        assert "not signed" in result.error.lower()

    def test_verify_signature_valid(self, receipt_store, sample_receipt_dict, signed_receipt_dict):
        """Test verify_signature for validly signed receipt."""
        receipt_store.save(sample_receipt_dict, signed_receipt=signed_receipt_dict)

        # Mock the signing module
        mock_signer = MagicMock()
        mock_signer.verify.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "aragora.gauntlet.signing": MagicMock(
                    ReceiptSigner=MagicMock(return_value=mock_signer),
                    SignatureMetadata=MagicMock(),
                    SignedReceipt=MagicMock(),
                )
            },
        ):
            result = receipt_store.verify_signature("receipt-001")

        assert result.is_valid is True
        assert result.algorithm == "HMAC-SHA256"

    def test_verify_batch(self, receipt_store, sample_receipt_dict, signed_receipt_dict):
        """Test batch signature verification."""
        # Add unsigned
        sample_receipt_dict["receipt_id"] = "unsigned"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-unsigned"
        receipt_store.save(sample_receipt_dict)

        # Add signed
        sample_receipt_dict["receipt_id"] = "signed"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-signed"
        receipt_store.save(sample_receipt_dict, signed_receipt=signed_receipt_dict)

        results, summary = receipt_store.verify_batch(["unsigned", "signed", "nonexistent"])

        assert len(results) == 3
        assert summary["total"] == 3
        assert summary["not_signed"] >= 1  # At least the unsigned one


# ===========================================================================
# Integrity Verification Tests
# ===========================================================================


class TestReceiptStoreIntegrity:
    """Tests for integrity verification."""

    def test_verify_integrity_not_found(self, receipt_store):
        """Test verify_integrity for nonexistent receipt."""
        result = receipt_store.verify_integrity("nonexistent")

        assert result["integrity_valid"] is False
        assert "not found" in result["error"].lower()

    def test_verify_integrity_valid(self, receipt_store, sample_receipt_dict):
        """Test verify_integrity for valid checksum."""
        sample_receipt_dict["checksum"] = "sha256:valid"
        receipt_store.save(sample_receipt_dict)

        # Mock DecisionReceipt to return same checksum
        mock_receipt = MagicMock()
        mock_receipt._compute_checksum.return_value = "sha256:valid"
        mock_receipt_class = MagicMock()
        mock_receipt_class.from_dict.return_value = mock_receipt

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = receipt_store.verify_integrity("receipt-001")

        assert result["integrity_valid"] is True
        assert result["stored_checksum"] == "sha256:valid"

    def test_verify_integrity_invalid(self, receipt_store, sample_receipt_dict):
        """Test verify_integrity for mismatched checksum."""
        sample_receipt_dict["checksum"] = "sha256:original"
        receipt_store.save(sample_receipt_dict)

        # Mock DecisionReceipt to return different checksum
        mock_receipt = MagicMock()
        mock_receipt._compute_checksum.return_value = "sha256:tampered"
        mock_receipt_class = MagicMock()
        mock_receipt_class.from_dict.return_value = mock_receipt

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = receipt_store.verify_integrity("receipt-001")

        assert result["integrity_valid"] is False
        assert result["stored_checksum"] == "sha256:original"
        assert result["computed_checksum"] == "sha256:tampered"


# ===========================================================================
# Retention and Cleanup Tests
# ===========================================================================


class TestReceiptStoreRetention:
    """Tests for retention and cleanup."""

    def test_cleanup_expired(self, receipt_store, sample_receipt_dict):
        """Test cleanup_expired removes old receipts."""
        now = time.time()

        # Add old receipt
        sample_receipt_dict["receipt_id"] = "old"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-old"
        sample_receipt_dict["timestamp"] = now - 86400 * 100  # 100 days ago
        receipt_store.save(sample_receipt_dict)

        # Add recent receipt
        sample_receipt_dict["receipt_id"] = "recent"
        sample_receipt_dict["gauntlet_id"] = "gauntlet-recent"
        sample_receipt_dict["timestamp"] = now
        receipt_store.save(sample_receipt_dict)

        assert receipt_store.count() == 2

        # Cleanup with 50 day retention
        removed = receipt_store.cleanup_expired(retention_days=50)

        assert removed == 1
        assert receipt_store.count() == 1
        assert receipt_store.get("recent") is not None
        assert receipt_store.get("old") is None

    def test_cleanup_no_expired(self, receipt_store, sample_receipt_dict):
        """Test cleanup when no receipts are expired."""
        receipt_store.save(sample_receipt_dict)

        removed = receipt_store.cleanup_expired(retention_days=365)

        assert removed == 0
        assert receipt_store.count() == 1


# ===========================================================================
# Statistics Tests
# ===========================================================================


class TestReceiptStoreStats:
    """Tests for receipt statistics."""

    def test_get_stats_empty(self, receipt_store):
        """Test get_stats for empty store."""
        stats = receipt_store.get_stats()

        assert stats["total"] == 0
        assert stats["signed"] == 0

    def test_get_stats_with_data(self, receipt_store, sample_receipt_dict, signed_receipt_dict):
        """Test get_stats with receipts."""
        # Add various receipts
        verdicts = ["APPROVED", "REJECTED", "APPROVED"]
        risks = ["LOW", "HIGH", "MEDIUM"]

        for i, (verdict, risk) in enumerate(zip(verdicts, risks)):
            sample_receipt_dict["receipt_id"] = f"receipt-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i}"
            sample_receipt_dict["verdict"] = verdict
            sample_receipt_dict["risk_level"] = risk
            if i == 0:
                receipt_store.save(sample_receipt_dict, signed_receipt=signed_receipt_dict)
            else:
                receipt_store.save(sample_receipt_dict)

        stats = receipt_store.get_stats()

        assert stats["total"] == 3
        assert stats["signed"] == 1
        assert stats["unsigned"] == 2
        assert stats["by_verdict"]["approved"] == 2
        assert stats["by_verdict"]["rejected"] == 1
        assert stats["by_risk_level"]["high"] == 1


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


class TestReceiptStoreSingleton:
    """Tests for singleton pattern."""

    def test_get_receipt_store_singleton(self, temp_db_path):
        """Test get_receipt_store returns singleton."""
        set_receipt_store(None)  # Reset

        with patch.dict("os.environ", {"ARAGORA_DATA_DIR": str(temp_db_path.parent)}):
            store1 = get_receipt_store()
            store2 = get_receipt_store()

        assert store1 is store2

        set_receipt_store(None)  # Cleanup

    def test_set_receipt_store(self, temp_db_path):
        """Test set_receipt_store replaces singleton."""
        custom_store = ReceiptStore(db_path=temp_db_path)

        set_receipt_store(custom_store)
        retrieved = get_receipt_store()

        assert retrieved is custom_store

        set_receipt_store(None)  # Cleanup


# ===========================================================================
# Backend Configuration Tests
# ===========================================================================


class TestReceiptStoreBackends:
    """Tests for backend configuration."""

    def test_sqlite_backend(self, temp_db_path):
        """Test SQLite backend initialization."""
        store = ReceiptStore(db_path=temp_db_path, backend="sqlite")

        assert store.backend_type == "sqlite"
        assert store._backend is not None

    def test_postgresql_requires_url(self):
        """Test PostgreSQL backend requires DATABASE_URL."""
        with pytest.raises(ValueError, match="requires DATABASE_URL"):
            ReceiptStore(backend="postgresql")

    def test_default_retention_days(self, temp_db_path):
        """Test default retention days from environment."""
        store = ReceiptStore(db_path=temp_db_path)
        assert store.retention_days == DEFAULT_RETENTION_DAYS

    def test_custom_retention_days(self, temp_db_path):
        """Test custom retention days."""
        store = ReceiptStore(db_path=temp_db_path, retention_days=365)
        assert store.retention_days == 365


# ===========================================================================
# GDPR Compliance Tests
# ===========================================================================


class TestReceiptStoreGDPR:
    """Tests for GDPR compliance features."""

    def test_get_by_user_no_matches(self, receipt_store, sample_receipt_dict):
        """Test get_by_user returns empty for no matches."""
        receipt_store.save(sample_receipt_dict)

        receipts, total = receipt_store.get_by_user("nonexistent-user")

        assert receipts == []
        assert total == 0

    def test_get_by_user_matches_user_id(self, receipt_store, sample_receipt_dict):
        """Test get_by_user matches user_id field in data."""
        sample_receipt_dict["user_id"] = "user-123"
        receipt_store.save(sample_receipt_dict)

        receipts, total = receipt_store.get_by_user("user-123")

        assert total == 1
        assert len(receipts) == 1
        assert receipts[0].receipt_id == "receipt-001"

    def test_get_by_user_matches_created_by(self, receipt_store, sample_receipt_dict):
        """Test get_by_user matches created_by field in data."""
        sample_receipt_dict["created_by"] = "admin-user-456"
        receipt_store.save(sample_receipt_dict)

        receipts, total = receipt_store.get_by_user("admin-user-456")

        assert total == 1
        assert len(receipts) == 1

    def test_get_by_user_matches_requestor_id(self, receipt_store, sample_receipt_dict):
        """Test get_by_user matches requestor_id field in data."""
        sample_receipt_dict["requestor_id"] = "requestor-789"
        receipt_store.save(sample_receipt_dict)

        receipts, total = receipt_store.get_by_user("requestor-789")

        assert total == 1
        assert len(receipts) == 1

    def test_get_by_user_pagination(self, receipt_store, sample_receipt_dict):
        """Test get_by_user pagination."""
        for i in range(5):
            sample_receipt_dict["receipt_id"] = f"receipt-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-{i}"
            sample_receipt_dict["user_id"] = "paginated-user"
            receipt_store.save(sample_receipt_dict.copy())

        receipts, total = receipt_store.get_by_user("paginated-user", limit=2, offset=0)

        assert total == 5
        assert len(receipts) == 2

        receipts2, _ = receipt_store.get_by_user("paginated-user", limit=2, offset=2)
        assert len(receipts2) == 2

    def test_get_retention_status_empty(self, receipt_store):
        """Test get_retention_status with no receipts."""
        status = receipt_store.get_retention_status()

        assert status["total_receipts"] == 0
        assert "retention_policy" in status
        assert status["retention_policy"]["retention_days"] == DEFAULT_RETENTION_DAYS

    def test_get_retention_status_with_receipts(self, receipt_store, sample_receipt_dict):
        """Test get_retention_status with some receipts."""
        for i in range(3):
            sample_receipt_dict["receipt_id"] = f"retention-{i}"
            sample_receipt_dict["gauntlet_id"] = f"gauntlet-ret-{i}"
            receipt_store.save(sample_receipt_dict.copy())

        status = receipt_store.get_retention_status()

        assert status["total_receipts"] == 3
        assert "age_distribution" in status
        assert "expiring_receipts" in status
        assert "timestamps" in status
        assert status["timestamps"]["newest_receipt"] is not None

    def test_get_retention_status_includes_all_fields(self, receipt_store, sample_receipt_dict):
        """Test get_retention_status returns all required fields."""
        receipt_store.save(sample_receipt_dict)

        status = receipt_store.get_retention_status()

        # Check required fields exist
        assert "retention_policy" in status
        assert "retention_days" in status["retention_policy"]
        assert "retention_years" in status["retention_policy"]
        assert "age_distribution" in status
        assert "expiring_receipts" in status
        assert "already_expired" in status
        assert "timestamps" in status
        assert "generated_at" in status
