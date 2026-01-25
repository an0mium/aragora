"""
Tests for consent management system.

Tests:
- Consent grant/revoke lifecycle
- Consent checking
- History tracking
- Bulk operations (right to be forgotten)
- GDPR data export
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.privacy.consent import (
    ConsentExport,
    ConsentManager,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    ConsentStore,
)


@pytest.fixture
def temp_store_path():
    """Create a temporary path for the consent store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "consent.json"


@pytest.fixture
def store(temp_store_path: Path):
    """Create a temporary consent store."""
    return ConsentStore(temp_store_path)


@pytest.fixture
def manager(store: ConsentStore):
    """Create a consent manager with temp store."""
    return ConsentManager(store)


class TestConsentRecord:
    """Tests for ConsentRecord dataclass."""

    def test_record_creation(self):
        """Test basic record creation."""
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
        )

        assert record.user_id == "user-123"
        assert record.purpose == ConsentPurpose.ANALYTICS
        assert record.granted is True
        assert record.version == "v1.0"

    def test_record_is_valid_when_granted(self):
        """Test is_valid returns True for granted consent."""
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
            status=ConsentStatus.ACTIVE,
        )

        assert record.is_valid is True

    def test_record_not_valid_when_revoked(self):
        """Test is_valid returns False for revoked consent."""
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=False,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
            status=ConsentStatus.REVOKED,
        )

        assert record.is_valid is False

    def test_record_not_valid_when_expired(self):
        """Test is_valid returns False for expired consent."""
        past_time = datetime.now(timezone.utc) - timedelta(days=1)

        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
            status=ConsentStatus.ACTIVE,
            expires_at=past_time,  # Already expired
        )

        assert record.is_valid is False

    def test_record_to_dict(self):
        """Test record serialization."""
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
        )

        data = record.to_dict()

        assert data["id"] == "test-id"
        assert data["user_id"] == "user-123"
        assert data["purpose"] == "analytics"
        assert data["granted"] is True

    def test_record_from_dict(self):
        """Test record deserialization."""
        data = {
            "id": "test-id",
            "user_id": "user-123",
            "purpose": "analytics",
            "granted": True,
            "granted_at": "2024-01-15T10:00:00+00:00",
            "version": "v1.0",
        }

        record = ConsentRecord.from_dict(data)

        assert record.id == "test-id"
        assert record.purpose == ConsentPurpose.ANALYTICS
        assert record.granted is True


class TestConsentStore:
    """Tests for ConsentStore persistence."""

    def test_store_initialization(self, temp_store_path: Path):
        """Test store creates storage directory."""
        store = ConsentStore(temp_store_path)

        assert temp_store_path.parent.exists()

    def test_save_and_get(self, store: ConsentStore):
        """Test saving and retrieving a record."""
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
        )

        store.save(record)
        retrieved = store.get("user-123", ConsentPurpose.ANALYTICS)

        assert retrieved is not None
        assert retrieved.id == "test-id"
        assert retrieved.granted is True

    def test_get_nonexistent(self, store: ConsentStore):
        """Test getting a nonexistent record."""
        result = store.get("nonexistent", ConsentPurpose.ANALYTICS)

        assert result is None

    def test_get_all_for_user(self, store: ConsentStore):
        """Test getting all records for a user."""
        # Save multiple consents
        for purpose in [ConsentPurpose.ANALYTICS, ConsentPurpose.MARKETING]:
            record = ConsentRecord(
                id=f"id-{purpose.value}",
                user_id="user-123",
                purpose=purpose,
                granted=True,
                granted_at=datetime.now(timezone.utc),
                version="v1.0",
            )
            store.save(record)

        records = store.get_all_for_user("user-123")

        assert len(records) == 2

    def test_delete_all_for_user(self, store: ConsentStore):
        """Test deleting all records for a user."""
        # Save some records
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
        )
        store.save(record)

        count = store.delete_all_for_user("user-123")

        assert count == 1
        assert store.get("user-123", ConsentPurpose.ANALYTICS) is None

    def test_get_users_with_consent(self, store: ConsentStore):
        """Test getting users with consent for a purpose."""
        # User 1 has analytics consent
        store.save(
            ConsentRecord(
                id="id-1",
                user_id="user-1",
                purpose=ConsentPurpose.ANALYTICS,
                granted=True,
                granted_at=datetime.now(timezone.utc),
                version="v1.0",
                status=ConsentStatus.ACTIVE,
            )
        )

        # User 2 has revoked analytics consent
        store.save(
            ConsentRecord(
                id="id-2",
                user_id="user-2",
                purpose=ConsentPurpose.ANALYTICS,
                granted=False,
                granted_at=datetime.now(timezone.utc),
                version="v1.0",
                status=ConsentStatus.REVOKED,
            )
        )

        users = store.get_users_with_consent(ConsentPurpose.ANALYTICS)

        assert "user-1" in users
        assert "user-2" not in users

    def test_persistence(self, temp_store_path: Path):
        """Test that data persists across store instances."""
        # Save with first store instance
        store1 = ConsentStore(temp_store_path)
        record = ConsentRecord(
            id="test-id",
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version="v1.0",
        )
        store1.save(record)

        # Load with new store instance
        store2 = ConsentStore(temp_store_path)
        retrieved = store2.get("user-123", ConsentPurpose.ANALYTICS)

        assert retrieved is not None
        assert retrieved.id == "test-id"


class TestConsentManager:
    """Tests for ConsentManager operations."""

    def test_grant_consent(self, manager: ConsentManager):
        """Test granting consent."""
        record = manager.grant_consent(
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            version="v1.0",
        )

        assert record.granted is True
        assert record.purpose == ConsentPurpose.ANALYTICS
        assert record.status == ConsentStatus.ACTIVE

    def test_grant_consent_with_metadata(self, manager: ConsentManager):
        """Test granting consent with metadata."""
        record = manager.grant_consent(
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            version="v1.0",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            metadata={"campaign": "signup"},
        )

        assert record.ip_address == "192.168.1.1"
        assert record.user_agent == "Mozilla/5.0"
        assert record.metadata["campaign"] == "signup"

    def test_revoke_consent(self, manager: ConsentManager):
        """Test revoking consent."""
        # First grant
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        # Then revoke
        record = manager.revoke_consent("user-123", ConsentPurpose.ANALYTICS)

        assert record is not None
        assert record.granted is False
        assert record.status == ConsentStatus.REVOKED
        assert record.revoked_at is not None

    def test_revoke_nonexistent_consent(self, manager: ConsentManager):
        """Test revoking consent that doesn't exist."""
        result = manager.revoke_consent("user-123", ConsentPurpose.ANALYTICS)

        assert result is None

    def test_check_consent_granted(self, manager: ConsentManager):
        """Test checking granted consent."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is True

    def test_check_consent_not_granted(self, manager: ConsentManager):
        """Test checking consent that was never granted."""
        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is False

    def test_check_consent_revoked(self, manager: ConsentManager):
        """Test checking revoked consent."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.revoke_consent("user-123", ConsentPurpose.ANALYTICS)

        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is False

    def test_get_consent_status(self, manager: ConsentManager):
        """Test getting consent status."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        status = manager.get_consent_status("user-123", ConsentPurpose.ANALYTICS)

        assert status is not None
        assert status.granted is True

    def test_get_all_consents(self, manager: ConsentManager):
        """Test getting all consents for a user."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")

        consents = manager.get_all_consents("user-123")

        assert len(consents) == 2


class TestConsentHistory:
    """Tests for consent history tracking."""

    def test_history_tracks_grants(self, manager: ConsentManager):
        """Test that grant operations are tracked in history."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        history = manager.get_consent_history("user-123")

        assert len(history) == 1
        assert history[0].granted is True

    def test_history_tracks_revokes(self, manager: ConsentManager):
        """Test that revoke operations are tracked in history."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.revoke_consent("user-123", ConsentPurpose.ANALYTICS)

        history = manager.get_consent_history("user-123")

        assert len(history) == 2
        assert history[0].granted is True
        assert history[1].granted is False

    def test_history_chronological(self, manager: ConsentManager):
        """Test that history is in chronological order."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")

        history = manager.get_consent_history("user-123")

        assert len(history) == 2
        assert history[0].purpose == ConsentPurpose.ANALYTICS
        assert history[1].purpose == ConsentPurpose.MARKETING


class TestBulkOperations:
    """Tests for bulk consent operations."""

    def test_bulk_revoke_for_user(self, manager: ConsentManager):
        """Test revoking all consents for a user."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")

        count = manager.bulk_revoke_for_user("user-123")

        assert count == 2
        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is False
        assert manager.check_consent("user-123", ConsentPurpose.MARKETING) is False

    def test_delete_user_data(self, manager: ConsentManager):
        """Test deleting all user consent data."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        count = manager.delete_user_data("user-123")

        assert count == 1
        assert len(manager.get_all_consents("user-123")) == 0

    def test_get_users_with_consent(self, manager: ConsentManager):
        """Test getting all users with specific consent."""
        manager.grant_consent("user-1", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-2", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-3", ConsentPurpose.MARKETING, "v1.0")

        users = manager.get_users_with_consent(ConsentPurpose.ANALYTICS)

        assert "user-1" in users
        assert "user-2" in users
        assert "user-3" not in users


class TestGDPRDataExport:
    """Tests for GDPR data export functionality."""

    def test_export_consent_data(self, manager: ConsentManager):
        """Test exporting consent data."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")

        export = manager.export_consent_data("user-123")

        assert export.user_id == "user-123"
        assert len(export.records) == 2
        assert export.exported_at is not None

    def test_export_to_dict(self, manager: ConsentManager):
        """Test export serialization to dict."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        export = manager.export_consent_data("user-123")
        data = export.to_dict()

        assert data["user_id"] == "user-123"
        assert data["record_count"] == 1
        assert "exported_at" in data
        assert "records" in data

    def test_export_to_json(self, manager: ConsentManager):
        """Test export to JSON string."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        export = manager.export_consent_data("user-123")
        json_str = export.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["user_id"] == "user-123"

    def test_export_empty_user(self, manager: ConsentManager):
        """Test export for user with no consents."""
        export = manager.export_consent_data("nonexistent")

        assert export.user_id == "nonexistent"
        assert len(export.records) == 0


class TestConsentVerification:
    """Tests for consent verification."""

    def test_verify_consent_all_present(self, manager: ConsentManager):
        """Test verification when all required consents are present."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")

        all_consented, missing = manager.verify_consent(
            "user-123",
            [ConsentPurpose.ANALYTICS, ConsentPurpose.MARKETING],
        )

        assert all_consented is True
        assert len(missing) == 0

    def test_verify_consent_some_missing(self, manager: ConsentManager):
        """Test verification when some consents are missing."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")

        all_consented, missing = manager.verify_consent(
            "user-123",
            [ConsentPurpose.ANALYTICS, ConsentPurpose.MARKETING],
        )

        assert all_consented is False
        assert ConsentPurpose.MARKETING in missing

    def test_verify_consent_none_present(self, manager: ConsentManager):
        """Test verification when no consents are present."""
        all_consented, missing = manager.verify_consent(
            "user-123",
            [ConsentPurpose.ANALYTICS, ConsentPurpose.MARKETING],
        )

        assert all_consented is False
        assert len(missing) == 2


class TestConsentPurposeEnum:
    """Tests for ConsentPurpose enum."""

    def test_all_purposes_defined(self):
        """Test that expected purposes are defined."""
        assert hasattr(ConsentPurpose, "DEBATE_PROCESSING")
        assert hasattr(ConsentPurpose, "KNOWLEDGE_STORAGE")
        assert hasattr(ConsentPurpose, "ANALYTICS")
        assert hasattr(ConsentPurpose, "MARKETING")
        assert hasattr(ConsentPurpose, "THIRD_PARTY_SHARING")

    def test_purpose_values_are_strings(self):
        """Test that purpose values are lowercase strings."""
        for purpose in ConsentPurpose:
            assert isinstance(purpose.value, str)
            assert purpose.value == purpose.value.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_regrant_consent(self, manager: ConsentManager):
        """Test granting consent that was previously revoked."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.revoke_consent("user-123", ConsentPurpose.ANALYTICS)

        # Regrant
        record = manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v2.0")

        assert record.granted is True
        assert record.version == "v2.0"
        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is True

    def test_consent_with_expiration(self, manager: ConsentManager):
        """Test consent that expires."""
        future_time = datetime.now(timezone.utc) + timedelta(days=30)

        record = manager.grant_consent(
            user_id="user-123",
            purpose=ConsentPurpose.ANALYTICS,
            version="v1.0",
            expires_at=future_time,
        )

        assert record.expires_at == future_time
        # Should still be valid
        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is True

    def test_multiple_users_same_purpose(self, manager: ConsentManager):
        """Test multiple users can have same purpose consent."""
        manager.grant_consent("user-1", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-2", ConsentPurpose.ANALYTICS, "v1.0")

        assert manager.check_consent("user-1", ConsentPurpose.ANALYTICS) is True
        assert manager.check_consent("user-2", ConsentPurpose.ANALYTICS) is True

    def test_same_user_multiple_purposes(self, manager: ConsentManager):
        """Test same user can have multiple purpose consents."""
        manager.grant_consent("user-123", ConsentPurpose.ANALYTICS, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.MARKETING, "v1.0")
        manager.grant_consent("user-123", ConsentPurpose.AI_TRAINING, "v1.0")

        assert manager.check_consent("user-123", ConsentPurpose.ANALYTICS) is True
        assert manager.check_consent("user-123", ConsentPurpose.MARKETING) is True
        assert manager.check_consent("user-123", ConsentPurpose.AI_TRAINING) is True
