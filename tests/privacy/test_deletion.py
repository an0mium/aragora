"""
Tests for GDPR Deletion Infrastructure.

Tests cover:
- DeletionRequest lifecycle
- LegalHold management
- GDPRDeletionScheduler scheduling and execution
- DeletionCascadeManager cascade deletion
- DataErasureVerifier verification and certification
- Integration flows
"""

from __future__ import annotations

import json
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from aragora.privacy.deletion import (
    ConsentDeleter,
    DataErasureVerifier,
    DeletionCascadeManager,
    DeletionCertificate,
    DeletionRequest,
    DeletionStatus,
    DeletionStore,
    EntityType,
    GDPRDeletionScheduler,
    LegalHold,
    LegalHoldManager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "deletion_store.json"


@pytest.fixture
def deletion_store(temp_storage_path):
    """Create a fresh deletion store."""
    return DeletionStore(storage_path=temp_storage_path)


@pytest.fixture
def cascade_manager():
    """Create a cascade manager with mock deleters."""
    manager = DeletionCascadeManager()
    return manager


@pytest.fixture
def mock_deleter():
    """Create a mock entity deleter."""
    deleter = MagicMock()
    deleter.delete_for_user = AsyncMock(return_value=5)
    deleter.verify_deleted = AsyncMock(return_value=True)
    return deleter


@pytest.fixture
def verifier(cascade_manager):
    """Create a data erasure verifier."""
    return DataErasureVerifier(cascade_manager)


@pytest.fixture
def scheduler(deletion_store, cascade_manager, verifier):
    """Create a deletion scheduler."""
    return GDPRDeletionScheduler(
        store=deletion_store,
        cascade_manager=cascade_manager,
        verifier=verifier,
        check_interval_seconds=1,
    )


@pytest.fixture
def legal_hold_manager(deletion_store):
    """Create a legal hold manager."""
    return LegalHoldManager(store=deletion_store)


# ============================================================================
# DeletionRequest Tests
# ============================================================================


class TestDeletionRequest:
    """Tests for DeletionRequest dataclass."""

    def test_create_request(self):
        """Test creating a deletion request."""
        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id="req-001",
            user_id="user-123",
            scheduled_for=now + timedelta(days=30),
            reason="User request",
            created_at=now,
        )

        assert request.request_id == "req-001"
        assert request.user_id == "user-123"
        assert request.status == DeletionStatus.PENDING
        assert request.entities_deleted == {}
        assert request.errors == []

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id="req-001",
            user_id="user-123",
            scheduled_for=now + timedelta(days=30),
            reason="User request",
            created_at=now,
            entities_deleted={"user": 1, "preferences": 5},
        )

        data = request.to_dict()

        assert data["request_id"] == "req-001"
        assert data["user_id"] == "user-123"
        assert data["status"] == "pending"
        assert data["entities_deleted"]["user"] == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "request_id": "req-002",
            "user_id": "user-456",
            "scheduled_for": (now + timedelta(days=30)).isoformat(),
            "reason": "GDPR request",
            "created_at": now.isoformat(),
            "status": "completed",
            "entities_deleted": {"consent_records": 3},
        }

        request = DeletionRequest.from_dict(data)

        assert request.request_id == "req-002"
        assert request.user_id == "user-456"
        assert request.status == DeletionStatus.COMPLETED
        assert request.entities_deleted["consent_records"] == 3


# ============================================================================
# LegalHold Tests
# ============================================================================


class TestLegalHold:
    """Tests for LegalHold dataclass."""

    def test_create_hold(self):
        """Test creating a legal hold."""
        now = datetime.now(timezone.utc)
        hold = LegalHold(
            hold_id="hold-001",
            user_ids=["user-1", "user-2"],
            reason="Litigation hold",
            created_by="legal-team",
            created_at=now,
            case_reference="CASE-2024-001",
        )

        assert hold.hold_id == "hold-001"
        assert len(hold.user_ids) == 2
        assert hold.is_active is True
        assert hold.case_reference == "CASE-2024-001"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        hold = LegalHold(
            hold_id="hold-002",
            user_ids=["user-3"],
            reason="Investigation",
            created_by="admin",
            created_at=now,
            expires_at=now + timedelta(days=90),
        )

        data = hold.to_dict()

        assert data["hold_id"] == "hold-002"
        assert data["user_ids"] == ["user-3"]
        assert data["is_active"] is True
        assert data["expires_at"] is not None


# ============================================================================
# DeletionStore Tests
# ============================================================================


class TestDeletionStore:
    """Tests for DeletionStore persistence."""

    def test_save_and_get_request(self, deletion_store):
        """Test saving and retrieving a request."""
        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id="req-001",
            user_id="user-123",
            scheduled_for=now + timedelta(days=30),
            reason="Test",
            created_at=now,
        )

        deletion_store.save_request(request)
        retrieved = deletion_store.get_request("req-001")

        assert retrieved is not None
        assert retrieved.request_id == "req-001"
        assert retrieved.user_id == "user-123"

    def test_get_requests_for_user(self, deletion_store):
        """Test getting all requests for a user."""
        now = datetime.now(timezone.utc)

        for i in range(3):
            request = DeletionRequest(
                request_id=f"req-{i}",
                user_id="user-123",
                scheduled_for=now + timedelta(days=30),
                reason="Test",
                created_at=now,
            )
            deletion_store.save_request(request)

        # Add one for different user
        other_request = DeletionRequest(
            request_id="req-other",
            user_id="user-456",
            scheduled_for=now + timedelta(days=30),
            reason="Test",
            created_at=now,
        )
        deletion_store.save_request(other_request)

        requests = deletion_store.get_requests_for_user("user-123")

        assert len(requests) == 3

    def test_get_pending_requests(self, deletion_store):
        """Test getting pending requests ready for execution."""
        now = datetime.now(timezone.utc)

        # Past scheduled - should be included
        past_request = DeletionRequest(
            request_id="req-past",
            user_id="user-1",
            scheduled_for=now - timedelta(days=1),
            reason="Past",
            created_at=now - timedelta(days=31),
        )
        deletion_store.save_request(past_request)

        # Future scheduled - should NOT be included
        future_request = DeletionRequest(
            request_id="req-future",
            user_id="user-2",
            scheduled_for=now + timedelta(days=10),
            reason="Future",
            created_at=now,
        )
        deletion_store.save_request(future_request)

        # Already completed - should NOT be included
        completed_request = DeletionRequest(
            request_id="req-completed",
            user_id="user-3",
            scheduled_for=now - timedelta(days=5),
            reason="Done",
            created_at=now - timedelta(days=35),
            status=DeletionStatus.COMPLETED,
        )
        deletion_store.save_request(completed_request)

        pending = deletion_store.get_pending_requests()

        assert len(pending) == 1
        assert pending[0].request_id == "req-past"

    def test_save_and_get_hold(self, deletion_store):
        """Test saving and retrieving a legal hold."""
        now = datetime.now(timezone.utc)
        hold = LegalHold(
            hold_id="hold-001",
            user_ids=["user-1", "user-2"],
            reason="Test hold",
            created_by="admin",
            created_at=now,
        )

        deletion_store.save_hold(hold)
        retrieved = deletion_store.get_hold("hold-001")

        assert retrieved is not None
        assert retrieved.hold_id == "hold-001"
        assert len(retrieved.user_ids) == 2

    def test_get_active_holds_for_user(self, deletion_store):
        """Test getting active holds for a user."""
        now = datetime.now(timezone.utc)

        # Active hold
        active_hold = LegalHold(
            hold_id="hold-active",
            user_ids=["user-123"],
            reason="Active hold",
            created_by="admin",
            created_at=now,
        )
        deletion_store.save_hold(active_hold)

        # Expired hold
        expired_hold = LegalHold(
            hold_id="hold-expired",
            user_ids=["user-123"],
            reason="Expired hold",
            created_by="admin",
            created_at=now - timedelta(days=90),
            expires_at=now - timedelta(days=1),
        )
        deletion_store.save_hold(expired_hold)

        # Inactive hold
        inactive_hold = LegalHold(
            hold_id="hold-inactive",
            user_ids=["user-123"],
            reason="Inactive hold",
            created_by="admin",
            created_at=now,
            is_active=False,
        )
        deletion_store.save_hold(inactive_hold)

        holds = deletion_store.get_active_holds_for_user("user-123")

        assert len(holds) == 1
        assert holds[0].hold_id == "hold-active"

    def test_persistence(self, temp_storage_path):
        """Test that data persists across store instances."""
        now = datetime.now(timezone.utc)

        # Create store and add data
        store1 = DeletionStore(storage_path=temp_storage_path)
        request = DeletionRequest(
            request_id="req-persist",
            user_id="user-persist",
            scheduled_for=now + timedelta(days=30),
            reason="Persistence test",
            created_at=now,
        )
        store1.save_request(request)

        # Create new store instance
        store2 = DeletionStore(storage_path=temp_storage_path)
        retrieved = store2.get_request("req-persist")

        assert retrieved is not None
        assert retrieved.request_id == "req-persist"


# ============================================================================
# DeletionCascadeManager Tests
# ============================================================================


class TestDeletionCascadeManager:
    """Tests for DeletionCascadeManager."""

    def test_register_deleter(self, cascade_manager, mock_deleter):
        """Test registering a deleter."""
        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)

        assert EntityType.CONSENT_RECORDS in cascade_manager._deleters

    @pytest.mark.asyncio
    async def test_execute_cascade_deletion(self, cascade_manager, mock_deleter):
        """Test executing cascade deletion."""
        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)
        cascade_manager.register_deleter(EntityType.PREFERENCES, mock_deleter)

        results = await cascade_manager.execute_cascade_deletion("user-123")

        assert "consent_records" in results
        assert "preferences" in results
        mock_deleter.delete_for_user.assert_called()

    @pytest.mark.asyncio
    async def test_execute_with_exclusions(self, cascade_manager, mock_deleter):
        """Test cascade deletion with excluded types."""
        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)
        cascade_manager.register_deleter(EntityType.PREFERENCES, mock_deleter)
        cascade_manager.register_deleter(EntityType.AUDIT_LOGS, mock_deleter)

        results = await cascade_manager.execute_cascade_deletion(
            "user-123",
            exclude_types=[EntityType.AUDIT_LOGS],
        )

        assert "consent_records" in results
        assert "preferences" in results
        assert "audit_logs" not in results

    @pytest.mark.asyncio
    async def test_verify_cascade_deletion(self, cascade_manager, mock_deleter):
        """Test verifying cascade deletion."""
        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)

        all_verified, failed = await cascade_manager.verify_cascade_deletion("user-123")

        assert all_verified is True
        assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_verify_cascade_deletion_failure(self, cascade_manager):
        """Test verification failure detection."""
        failing_deleter = MagicMock()
        failing_deleter.delete_for_user = AsyncMock(return_value=0)
        failing_deleter.verify_deleted = AsyncMock(return_value=False)  # Data still exists

        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, failing_deleter)

        all_verified, failed = await cascade_manager.verify_cascade_deletion("user-123")

        assert all_verified is False
        assert "consent_records" in failed


# ============================================================================
# DataErasureVerifier Tests
# ============================================================================


class TestDataErasureVerifier:
    """Tests for DataErasureVerifier."""

    @pytest.mark.asyncio
    async def test_verify_and_certify(self, cascade_manager, mock_deleter):
        """Test verification and certificate generation."""
        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)
        verifier = DataErasureVerifier(cascade_manager)

        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id="req-001",
            user_id="user-123",
            scheduled_for=now,
            reason="Test",
            created_at=now,
            completed_at=now,
            entities_deleted={"consent_records": 5},
        )

        certificate = await verifier.verify_and_certify(request)

        assert certificate is not None
        assert certificate.request_id == "req-001"
        assert certificate.user_id == "user-123"
        assert certificate.verification_hash is not None
        assert certificate.signature is not None

    def test_verify_certificate(self, cascade_manager):
        """Test certificate verification."""
        verifier = DataErasureVerifier(cascade_manager)
        now = datetime.now(timezone.utc)

        # Create a valid certificate
        certificate = DeletionCertificate(
            certificate_id="cert-001",
            request_id="req-001",
            user_id="user-123",
            issued_at=now,
            entities_deleted={"user": 1},
            verification_hash="abc123",
            signed_by="test",
            signature="valid",
        )

        # This should fail because signature doesn't match
        is_valid = verifier.verify_certificate(certificate)

        # The signature was manually set, so it won't match
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_verify_and_certify_failure(self, cascade_manager):
        """Test verification failure."""
        failing_deleter = MagicMock()
        failing_deleter.verify_deleted = AsyncMock(return_value=False)

        cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, failing_deleter)
        verifier = DataErasureVerifier(cascade_manager)

        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id="req-001",
            user_id="user-123",
            scheduled_for=now,
            reason="Test",
            created_at=now,
            completed_at=now,
        )

        with pytest.raises(RuntimeError, match="Verification failed"):
            await verifier.verify_and_certify(request)


# ============================================================================
# GDPRDeletionScheduler Tests
# ============================================================================


class TestGDPRDeletionScheduler:
    """Tests for GDPRDeletionScheduler."""

    def test_schedule_deletion(self, scheduler):
        """Test scheduling a deletion."""
        request = scheduler.schedule_deletion(
            user_id="user-123",
            grace_period_days=30,
            reason="User request",
        )

        assert request is not None
        assert request.user_id == "user-123"
        assert request.status == DeletionStatus.PENDING
        assert (request.scheduled_for - request.created_at).days == 30

    def test_schedule_deletion_with_legal_hold(self, scheduler, legal_hold_manager):
        """Test that scheduling fails for users on legal hold."""
        # Create a legal hold
        legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="Investigation",
            created_by="admin",
        )

        with pytest.raises(ValueError, match="legal hold"):
            scheduler.schedule_deletion(user_id="user-123")

    def test_cancel_deletion(self, scheduler):
        """Test cancelling a deletion."""
        request = scheduler.schedule_deletion(user_id="user-123")

        cancelled = scheduler.cancel_deletion(
            request.request_id,
            reason="User changed mind",
        )

        assert cancelled is not None
        assert cancelled.status == DeletionStatus.CANCELLED
        assert cancelled.cancelled_reason == "User changed mind"

    def test_cancel_non_pending_deletion(self, scheduler):
        """Test that cancelling a completed deletion fails."""
        request = scheduler.schedule_deletion(user_id="user-123")

        # Manually mark as completed
        request.status = DeletionStatus.COMPLETED
        scheduler._store.save_request(request)

        with pytest.raises(ValueError, match="Cannot cancel"):
            scheduler.cancel_deletion(request.request_id)

    @pytest.mark.asyncio
    async def test_execute_deletion(self, scheduler, mock_deleter):
        """Test executing a deletion."""
        scheduler._cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)

        request = scheduler.schedule_deletion(
            user_id="user-123",
            grace_period_days=0,  # Immediate
        )

        result = await scheduler.execute_deletion(request.request_id)

        assert result.status == DeletionStatus.COMPLETED
        assert result.completed_at is not None
        assert result.deletion_certificate is not None

    @pytest.mark.asyncio
    async def test_execute_deletion_blocked_by_hold(self, scheduler, legal_hold_manager):
        """Test that execution is blocked by legal hold."""
        request = scheduler.schedule_deletion(user_id="user-123")

        # Create hold after scheduling - this automatically puts request on HELD status
        legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="New investigation",
            created_by="admin",
        )

        # Check request was automatically put on hold by create_hold
        updated = scheduler._store.get_request(request.request_id)
        assert updated.status == DeletionStatus.HELD

        # Trying to execute a HELD request should fail
        with pytest.raises(ValueError, match="Cannot execute deletion in status held"):
            await scheduler.execute_deletion(request.request_id)

    @pytest.mark.asyncio
    async def test_process_pending_deletions(self, scheduler, mock_deleter):
        """Test processing multiple pending deletions."""
        scheduler._cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)

        # Create multiple requests with past scheduled dates
        now = datetime.now(timezone.utc)
        for i in range(3):
            request = DeletionRequest(
                request_id=f"req-{i}",
                user_id=f"user-{i}",
                scheduled_for=now - timedelta(days=1),
                reason="Past due",
                created_at=now - timedelta(days=31),
            )
            scheduler._store.save_request(request)

        processed = await scheduler.process_pending_deletions()

        assert len(processed) == 3
        for result in processed:
            assert result.status in [DeletionStatus.COMPLETED, DeletionStatus.FAILED]


# ============================================================================
# LegalHoldManager Tests
# ============================================================================


class TestLegalHoldManager:
    """Tests for LegalHoldManager."""

    def test_create_hold(self, legal_hold_manager):
        """Test creating a legal hold."""
        hold = legal_hold_manager.create_hold(
            user_ids=["user-1", "user-2"],
            reason="Investigation",
            created_by="legal-team",
            case_reference="CASE-001",
        )

        assert hold is not None
        assert hold.is_active is True
        assert len(hold.user_ids) == 2
        assert hold.case_reference == "CASE-001"

    def test_create_hold_updates_pending_requests(self, scheduler, legal_hold_manager):
        """Test that creating hold updates pending deletion requests."""
        # Schedule a deletion
        request = scheduler.schedule_deletion(user_id="user-123")
        assert request.status == DeletionStatus.PENDING

        # Create hold
        hold = legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="Investigation",
            created_by="admin",
        )

        # Check request was updated
        updated = scheduler._store.get_request(request.request_id)
        assert updated.status == DeletionStatus.HELD
        assert updated.hold_id == hold.hold_id

    def test_release_hold(self, legal_hold_manager):
        """Test releasing a legal hold."""
        hold = legal_hold_manager.create_hold(
            user_ids=["user-1"],
            reason="Investigation",
            created_by="admin",
        )

        released = legal_hold_manager.release_hold(
            hold.hold_id,
            released_by="legal-team",
        )

        assert released is not None
        assert released.is_active is False
        assert released.released_at is not None
        assert released.released_by == "legal-team"

    def test_release_hold_reactivates_requests(self, scheduler, legal_hold_manager):
        """Test that releasing hold reactivates pending requests."""
        # Schedule and then hold
        request = scheduler.schedule_deletion(user_id="user-123")
        hold = legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="Investigation",
            created_by="admin",
        )

        # Verify on hold
        held = scheduler._store.get_request(request.request_id)
        assert held.status == DeletionStatus.HELD

        # Release hold
        legal_hold_manager.release_hold(hold.hold_id, released_by="admin")

        # Verify reactivated
        reactivated = scheduler._store.get_request(request.request_id)
        assert reactivated.status == DeletionStatus.PENDING
        assert reactivated.hold_id is None

    def test_is_user_on_hold(self, legal_hold_manager):
        """Test checking if user is on hold."""
        assert legal_hold_manager.is_user_on_hold("user-123") is False

        legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="Test",
            created_by="admin",
        )

        assert legal_hold_manager.is_user_on_hold("user-123") is True

    def test_get_active_holds(self, legal_hold_manager):
        """Test getting all active holds."""
        legal_hold_manager.create_hold(
            user_ids=["user-1"],
            reason="Hold 1",
            created_by="admin",
        )
        hold2 = legal_hold_manager.create_hold(
            user_ids=["user-2"],
            reason="Hold 2",
            created_by="admin",
        )

        # Release one
        legal_hold_manager.release_hold(hold2.hold_id, released_by="admin")

        active = legal_hold_manager.get_active_holds()

        assert len(active) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestDeletionIntegration:
    """Integration tests for the full deletion workflow."""

    @pytest.mark.asyncio
    async def test_full_deletion_workflow(self, scheduler, mock_deleter):
        """Test complete deletion workflow: schedule -> execute -> verify."""
        scheduler._cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)
        scheduler._cascade_manager.register_deleter(EntityType.PREFERENCES, mock_deleter)

        # 1. Schedule deletion
        request = scheduler.schedule_deletion(
            user_id="user-123",
            grace_period_days=0,
            reason="GDPR Article 17 request",
        )
        assert request.status == DeletionStatus.PENDING

        # 2. Execute deletion
        result = await scheduler.execute_deletion(request.request_id)
        assert result.status == DeletionStatus.COMPLETED
        assert "consent_records" in result.entities_deleted
        assert "preferences" in result.entities_deleted

        # 3. Verify certificate was generated
        assert result.deletion_certificate is not None
        assert result.verification_hash is not None

        # 4. Check certificate is stored
        cert_id = result.deletion_certificate["certificate_id"]
        stored_cert = scheduler._store.get_certificate(cert_id)
        assert stored_cert is not None

    @pytest.mark.asyncio
    async def test_deletion_with_legal_hold_lifecycle(
        self,
        scheduler,
        legal_hold_manager,
        mock_deleter,
    ):
        """Test deletion blocked by hold, then executed after release."""
        scheduler._cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, mock_deleter)

        # 1. Schedule deletion
        request = scheduler.schedule_deletion(user_id="user-123")
        assert request.status == DeletionStatus.PENDING

        # 2. Create legal hold - automatically puts request on HELD status
        hold = legal_hold_manager.create_hold(
            user_ids=["user-123"],
            reason="Litigation",
            created_by="legal",
        )

        # 3. Verify request was automatically put on hold
        held = scheduler._store.get_request(request.request_id)
        assert held.status == DeletionStatus.HELD
        assert held.hold_id == hold.hold_id

        # 4. Verify execution is blocked for HELD requests
        with pytest.raises(ValueError, match="Cannot execute deletion in status held"):
            await scheduler.execute_deletion(request.request_id)

        # 5. Release hold - automatically reactivates request
        legal_hold_manager.release_hold(hold.hold_id, released_by="legal")

        # 6. Verify request was reactivated
        reactivated = scheduler._store.get_request(request.request_id)
        assert reactivated.status == DeletionStatus.PENDING

        # 7. Now deletion should work
        result = await scheduler.execute_deletion(request.request_id)
        assert result.status == DeletionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_deletion_failure_handling(self, scheduler):
        """Test handling of deletion failures."""
        failing_deleter = MagicMock()
        failing_deleter.delete_for_user = AsyncMock(side_effect=RuntimeError("Database error"))
        failing_deleter.verify_deleted = AsyncMock(return_value=True)

        scheduler._cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, failing_deleter)

        request = scheduler.schedule_deletion(user_id="user-123", grace_period_days=0)

        with pytest.raises(RuntimeError, match="Database error"):
            await scheduler.execute_deletion(request.request_id)

        # Check request was marked as failed
        failed = scheduler._store.get_request(request.request_id)
        assert failed.status == DeletionStatus.FAILED
        assert len(failed.errors) > 0


# ============================================================================
# ConsentDeleter Tests
# ============================================================================


class TestConsentDeleter:
    """Tests for ConsentDeleter."""

    @pytest.mark.asyncio
    async def test_delete_for_user(self):
        """Test deleting consent records."""
        mock_manager = MagicMock()
        mock_manager.delete_user_data.return_value = 5

        with patch("aragora.privacy.consent.get_consent_manager", return_value=mock_manager):
            deleter = ConsentDeleter()
            count = await deleter.delete_for_user("user-123")

            assert count == 5
            mock_manager.delete_user_data.assert_called_once_with("user-123")

    @pytest.mark.asyncio
    async def test_verify_deleted(self):
        """Test verifying consent records deleted."""
        mock_manager = MagicMock()
        mock_manager.get_all_consents.return_value = []

        with patch("aragora.privacy.consent.get_consent_manager", return_value=mock_manager):
            deleter = ConsentDeleter()
            is_deleted = await deleter.verify_deleted("user-123")

            assert is_deleted is True


__all__ = [
    "TestDeletionRequest",
    "TestLegalHold",
    "TestDeletionStore",
    "TestDeletionCascadeManager",
    "TestDataErasureVerifier",
    "TestGDPRDeletionScheduler",
    "TestLegalHoldManager",
    "TestDeletionIntegration",
    "TestConsentDeleter",
]
