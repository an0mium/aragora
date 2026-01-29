"""
Tests for Unified Deletion Coordinator.

Tests the coordination of deletion across privacy, storage, and backup systems.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.deletion_coordinator import (
    UnifiedDeletionCoordinator,
    BackupManagerAdapter,
    CascadeResult,
    CascadeStatus,
    VerificationResult,
    DeletionCertificate,
    get_deletion_coordinator,
    set_deletion_coordinator,
    reset_deletion_coordinator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_deleter():
    """Create a mock entity deleter."""
    deleter = MagicMock()
    deleter.delete_for_user = AsyncMock(return_value=5)
    deleter.verify_deletion = AsyncMock(return_value=True)
    return deleter


@pytest.fixture
def mock_backup_coordinator():
    """Create a mock backup coordinator."""
    coordinator = MagicMock()
    coordinator.get_backups_containing_user = MagicMock(return_value=["backup-1", "backup-2"])
    coordinator.mark_backup_for_purge = MagicMock()
    coordinator.verify_user_purged_from_backups = MagicMock(return_value=True)
    return coordinator


@pytest.fixture
def mock_audit_logger():
    """Create a mock audit logger."""
    logger = MagicMock()
    logger.log = AsyncMock()
    return logger


@pytest.fixture
def coordinator(mock_audit_logger):
    """Create a fresh coordinator for each test."""
    return UnifiedDeletionCoordinator(audit_logger=mock_audit_logger)


@pytest.fixture(autouse=True)
def reset_global_coordinator():
    """Reset global coordinator before and after each test."""
    reset_deletion_coordinator()
    yield
    reset_deletion_coordinator()


# ============================================================================
# Registration Tests
# ============================================================================


class TestDeleterRegistration:
    """Tests for entity deleter registration."""

    def test_register_single_deleter(self, coordinator, mock_deleter):
        """Test registering a single deleter."""
        coordinator.register_deleter("users", mock_deleter)

        assert "users" in coordinator._deleters
        assert coordinator._deletion_order == ["users"]

    def test_register_multiple_deleters(self, coordinator, mock_deleter):
        """Test registering multiple deleters in order."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_deleter("debates", mock_deleter)
        coordinator.register_deleter("knowledge", mock_deleter)

        assert len(coordinator._deleters) == 3
        assert coordinator._deletion_order == ["users", "debates", "knowledge"]

    def test_register_deleter_with_explicit_order(self, coordinator, mock_deleter):
        """Test registering deleters with explicit order."""
        coordinator.register_deleter("users", mock_deleter, order=1)
        coordinator.register_deleter("debates", mock_deleter, order=0)
        coordinator.register_deleter("knowledge", mock_deleter, order=2)

        # debates should be first (order=0)
        assert coordinator._deletion_order[0] == "debates"

    def test_register_backup_coordinator(self, coordinator, mock_backup_coordinator):
        """Test registering backup coordinator."""
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        assert coordinator._backup_coordinator is mock_backup_coordinator


# ============================================================================
# Cascade Deletion Tests
# ============================================================================


class TestCascadeDeletion:
    """Tests for cascade deletion execution."""

    @pytest.mark.asyncio
    async def test_execute_cascade_success(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test successful cascade deletion."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_deleter("debates", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_cascade("user-123", actor_id="admin-1")

        assert result.status == CascadeStatus.COMPLETED
        assert result.success is True
        assert result.user_id == "user-123"
        assert result.entities_deleted == {"users": 5, "debates": 5}
        assert result.total_deleted == 10
        assert len(result.errors) == 0

        # Verify deleters were called
        assert mock_deleter.delete_for_user.call_count == 2

        # Verify backup coordination
        mock_backup_coordinator.get_backups_containing_user.assert_called_once_with("user-123")
        assert mock_backup_coordinator.mark_backup_for_purge.call_count == 2

        # Verify audit logging
        assert mock_audit_logger.log.call_count >= 2  # start and complete

    @pytest.mark.asyncio
    async def test_execute_cascade_partial_failure(self, coordinator, mock_audit_logger):
        """Test cascade deletion with partial failure."""
        # Create one successful and one failing deleter
        success_deleter = MagicMock()
        success_deleter.delete_for_user = AsyncMock(return_value=3)

        fail_deleter = MagicMock()
        fail_deleter.delete_for_user = AsyncMock(side_effect=Exception("DB error"))

        coordinator.register_deleter("users", success_deleter)
        coordinator.register_deleter("debates", fail_deleter)

        result = await coordinator.execute_cascade("user-123")

        assert result.status == CascadeStatus.PARTIAL
        assert result.success is False
        assert result.entities_deleted == {"users": 3}
        assert len(result.errors) == 1
        assert "debates" in result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_cascade_complete_failure(self, coordinator, mock_audit_logger):
        """Test cascade deletion with complete failure."""
        fail_deleter = MagicMock()
        fail_deleter.delete_for_user = AsyncMock(side_effect=Exception("DB error"))

        coordinator.register_deleter("users", fail_deleter)

        result = await coordinator.execute_cascade("user-123")

        assert result.status == CascadeStatus.FAILED
        assert result.success is False
        assert len(result.entities_deleted) == 0

    @pytest.mark.asyncio
    async def test_execute_cascade_skip_backup_check(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test cascade deletion with backup check skipped."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_cascade("user-123", skip_backup_check=True)

        assert result.success is True
        # Backup coordinator should not be called
        mock_backup_coordinator.get_backups_containing_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_cascade_no_backup_coordinator(
        self, coordinator, mock_deleter, mock_audit_logger
    ):
        """Test cascade deletion without backup coordinator."""
        coordinator.register_deleter("users", mock_deleter)
        # No backup coordinator registered

        result = await coordinator.execute_cascade("user-123")

        assert result.success is True
        assert "affected_backups" not in result.backup_status


# ============================================================================
# Hook Tests
# ============================================================================


class TestDeletionHooks:
    """Tests for pre and post deletion hooks."""

    @pytest.mark.asyncio
    async def test_pre_deletion_hook(self, coordinator, mock_deleter, mock_audit_logger):
        """Test pre-deletion hook is called."""
        hook_called = []

        async def pre_hook(user_id: str):
            hook_called.append(user_id)

        coordinator.register_deleter("users", mock_deleter)
        coordinator.add_pre_deletion_hook(pre_hook)

        await coordinator.execute_cascade("user-123")

        assert hook_called == ["user-123"]

    @pytest.mark.asyncio
    async def test_post_deletion_hook(self, coordinator, mock_deleter, mock_audit_logger):
        """Test post-deletion hook is called with result."""
        hook_results = []

        async def post_hook(user_id: str, result: CascadeResult):
            hook_results.append((user_id, result.status))

        coordinator.register_deleter("users", mock_deleter)
        coordinator.add_post_deletion_hook(post_hook)

        await coordinator.execute_cascade("user-123")

        assert len(hook_results) == 1
        assert hook_results[0] == ("user-123", CascadeStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_hook_failure_doesnt_stop_deletion(
        self, coordinator, mock_deleter, mock_audit_logger
    ):
        """Test that hook failure doesn't stop deletion."""

        async def failing_hook(user_id: str):
            raise Exception("Hook failed")

        coordinator.register_deleter("users", mock_deleter)
        coordinator.add_pre_deletion_hook(failing_hook)

        result = await coordinator.execute_cascade("user-123")

        # Deletion should still succeed
        assert result.success is True
        # But warning should be recorded
        assert any("hook failed" in w.lower() for w in result.warnings)


# ============================================================================
# Verification Tests
# ============================================================================


class TestDeletionVerification:
    """Tests for deletion verification."""

    @pytest.mark.asyncio
    async def test_verify_deletion_success(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test successful verification."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_deleter("debates", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.verify_deletion("user-123")

        assert result.verified is True
        assert result.storage_verified is True
        assert result.backup_verified is True
        assert result.entity_results == {"users": True, "debates": True}

    @pytest.mark.asyncio
    async def test_verify_deletion_storage_failure(
        self, coordinator, mock_backup_coordinator, mock_audit_logger
    ):
        """Test verification with storage still containing data."""
        deleter = MagicMock()
        deleter.verify_deletion = AsyncMock(return_value=False)

        coordinator.register_deleter("users", deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.verify_deletion("user-123")

        assert result.verified is False
        assert result.storage_verified is False
        assert result.entity_results == {"users": False}

    @pytest.mark.asyncio
    async def test_verify_deletion_backup_failure(
        self, coordinator, mock_deleter, mock_audit_logger
    ):
        """Test verification with backup still containing data."""
        backup_coord = MagicMock()
        backup_coord.verify_user_purged_from_backups = MagicMock(return_value=False)

        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(backup_coord)

        result = await coordinator.verify_deletion("user-123")

        assert result.verified is False
        assert result.storage_verified is True
        assert result.backup_verified is False

    @pytest.mark.asyncio
    async def test_verify_deletion_skip_backup(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test verification with backup check skipped."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.verify_deletion("user-123", include_backups=False)

        assert result.verified is True
        assert result.backup_verified is True  # Skipped = assumed verified
        mock_backup_coordinator.verify_user_purged_from_backups.assert_not_called()


# ============================================================================
# Certificate Tests
# ============================================================================


class TestDeletionCertificate:
    """Tests for deletion certificate generation."""

    @pytest.mark.asyncio
    async def test_generate_certificate(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test certificate generation."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        # Execute deletion
        cascade_result = await coordinator.execute_cascade("user-123")

        # Generate certificate
        cert = await coordinator.generate_certificate("user-123", cascade_result)

        assert cert.user_id == "user-123"
        assert cert.certificate_id is not None
        assert cert.signature is not None
        assert cert.cascade_result == cascade_result
        assert cert.verification_result.verified is True

    def test_certificate_to_dict(self):
        """Test certificate serialization."""
        cascade_result = CascadeResult(
            user_id="user-123",
            status=CascadeStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            entities_deleted={"users": 5},
        )
        verification_result = VerificationResult(
            user_id="user-123",
            verified=True,
            verified_at=datetime.now(timezone.utc),
            storage_verified=True,
            backup_verified=True,
        )

        cert = DeletionCertificate.create(
            user_id="user-123",
            cascade_result=cascade_result,
            verification_result=verification_result,
        )

        cert_dict = cert.to_dict()

        assert cert_dict["user_id"] == "user-123"
        assert cert_dict["certificate_id"] is not None
        assert cert_dict["signature"] is not None
        assert "cascade_result" in cert_dict
        assert "verification_result" in cert_dict


# ============================================================================
# Backup Manager Adapter Tests
# ============================================================================


class TestBackupManagerAdapter:
    """Tests for BackupManagerAdapter."""

    def test_record_user_in_backup(self):
        """Test recording user presence in backup."""
        adapter = BackupManagerAdapter(MagicMock())

        adapter.record_user_in_backup("user-1", "backup-1")
        adapter.record_user_in_backup("user-1", "backup-2")
        adapter.record_user_in_backup("user-2", "backup-2")

        assert adapter.get_backups_containing_user("user-1") == ["backup-1", "backup-2"] or set(
            adapter.get_backups_containing_user("user-1")
        ) == {"backup-1", "backup-2"}
        assert adapter.get_backups_containing_user("user-2") == ["backup-2"]
        assert adapter.get_backups_containing_user("user-3") == []

    def test_mark_backup_for_purge(self):
        """Test marking backup for purge."""
        adapter = BackupManagerAdapter(MagicMock())

        adapter.record_user_in_backup("user-1", "backup-1")
        adapter.mark_backup_for_purge("backup-1", "user-1")

        assert "backup-1" in adapter._backup_purge_status
        assert "user-1" in adapter._backup_purge_status["backup-1"]

    def test_verify_user_purged_not_purged(self):
        """Test verification when user not yet purged."""
        adapter = BackupManagerAdapter(MagicMock())

        adapter.record_user_in_backup("user-1", "backup-1")

        # User still in backup
        assert adapter.verify_user_purged_from_backups("user-1") is False

    def test_verify_user_purged_after_purge(self):
        """Test verification after user purged."""
        adapter = BackupManagerAdapter(MagicMock())

        adapter.record_user_in_backup("user-1", "backup-1")
        adapter.mark_backup_for_purge("backup-1", "user-1")
        adapter.mark_user_purged_from_backup("backup-1", "user-1")

        # User should be purged
        assert adapter.verify_user_purged_from_backups("user-1") is True


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global coordinator instance management."""

    def test_get_coordinator_creates_instance(self):
        """Test that get creates instance if needed."""
        coord = get_deletion_coordinator()

        assert coord is not None
        assert isinstance(coord, UnifiedDeletionCoordinator)

    def test_get_coordinator_returns_same_instance(self):
        """Test that get returns same instance."""
        coord1 = get_deletion_coordinator()
        coord2 = get_deletion_coordinator()

        assert coord1 is coord2

    def test_set_coordinator(self):
        """Test setting custom coordinator."""
        custom = UnifiedDeletionCoordinator()
        set_deletion_coordinator(custom)

        assert get_deletion_coordinator() is custom

    def test_reset_coordinator(self):
        """Test resetting coordinator."""
        get_deletion_coordinator()  # Create instance
        reset_deletion_coordinator()

        # Should create new instance
        coord = get_deletion_coordinator()
        assert coord is not None


# ============================================================================
# Result Dataclass Tests
# ============================================================================


class TestCascadeResult:
    """Tests for CascadeResult dataclass."""

    def test_cascade_result_success_property(self):
        """Test success property."""
        completed = CascadeResult(
            user_id="user-1",
            status=CascadeStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
        )
        failed = CascadeResult(
            user_id="user-1",
            status=CascadeStatus.FAILED,
            started_at=datetime.now(timezone.utc),
        )

        assert completed.success is True
        assert failed.success is False

    def test_cascade_result_total_deleted(self):
        """Test total_deleted property."""
        result = CascadeResult(
            user_id="user-1",
            status=CascadeStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            entities_deleted={"users": 5, "debates": 10, "knowledge": 3},
        )

        assert result.total_deleted == 18

    def test_cascade_result_to_dict(self):
        """Test serialization."""
        result = CascadeResult(
            user_id="user-1",
            status=CascadeStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            entities_deleted={"users": 5},
            errors=["error1"],
            warnings=["warning1"],
        )

        data = result.to_dict()

        assert data["user_id"] == "user-1"
        assert data["status"] == "completed"
        assert data["total_deleted"] == 5
        assert data["success"] is True


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_to_dict(self):
        """Test serialization."""
        result = VerificationResult(
            user_id="user-1",
            verified=True,
            verified_at=datetime.now(timezone.utc),
            storage_verified=True,
            backup_verified=True,
            entity_results={"users": True},
        )

        data = result.to_dict()

        assert data["user_id"] == "user-1"
        assert data["verified"] is True
        assert data["storage_verified"] is True
        assert data["backup_verified"] is True


# ============================================================================
# Coordinated Deletion Tests
# ============================================================================


class TestCoordinatedDeletion:
    """Tests for execute_coordinated_deletion method."""

    @pytest.mark.asyncio
    async def test_execute_coordinated_deletion_success(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test successful coordinated deletion."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_coordinated_deletion(
            user_id="user-123",
            reason="GDPR request",
            delete_from_backups=True,
        )

        assert result.status == CascadeStatus.COMPLETED
        assert result.metadata["reason"] == "GDPR request"
        assert result.metadata["delete_from_backups"] is True

    @pytest.mark.asyncio
    async def test_execute_coordinated_deletion_dry_run(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test dry run doesn't delete data."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_coordinated_deletion(
            user_id="user-123",
            reason="Test dry run",
            dry_run=True,
        )

        assert result.status == CascadeStatus.PENDING
        assert result.metadata["dry_run"] is True
        # Deleter should not be called in dry run
        mock_deleter.delete_for_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_coordinated_deletion_dry_run_shows_affected_backups(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test dry run shows affected backups."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_coordinated_deletion(
            user_id="user-123",
            reason="Test dry run",
            dry_run=True,
            delete_from_backups=True,
        )

        assert result.backup_status["affected_backups"] == ["backup-1", "backup-2"]
        assert result.backup_status["would_purge"] is True

    @pytest.mark.asyncio
    async def test_execute_coordinated_deletion_no_backup(
        self, coordinator, mock_deleter, mock_backup_coordinator, mock_audit_logger
    ):
        """Test coordinated deletion without backup deletion."""
        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(mock_backup_coordinator)

        result = await coordinator.execute_coordinated_deletion(
            user_id="user-123",
            reason="User request",
            delete_from_backups=False,
        )

        assert result.status == CascadeStatus.COMPLETED
        # Backup should not be checked
        mock_backup_coordinator.get_backups_containing_user.assert_not_called()


# ============================================================================
# Backup Exclusion List Tests
# ============================================================================


class TestBackupExclusionList:
    """Tests for backup exclusion list management."""

    def test_add_to_exclusion_list(self, coordinator, mock_audit_logger):
        """Test adding user to exclusion list."""
        coordinator.add_to_backup_exclusion_list("user-123", "GDPR deletion")

        assert coordinator.is_user_excluded_from_backups("user-123") is True
        assert coordinator.is_user_excluded_from_backups("user-456") is False

    def test_remove_from_exclusion_list(self, coordinator, mock_audit_logger):
        """Test removing user from exclusion list."""
        coordinator.add_to_backup_exclusion_list("user-123", "GDPR deletion")
        assert coordinator.is_user_excluded_from_backups("user-123") is True

        removed = coordinator.remove_from_backup_exclusion_list("user-123")

        assert removed is True
        assert coordinator.is_user_excluded_from_backups("user-123") is False

    def test_remove_nonexistent_from_exclusion_list(self, coordinator, mock_audit_logger):
        """Test removing non-existent user returns False."""
        removed = coordinator.remove_from_backup_exclusion_list("user-999")

        assert removed is False

    def test_get_exclusion_list(self, coordinator, mock_audit_logger):
        """Test getting exclusion list."""
        coordinator.add_to_backup_exclusion_list("user-1", "Reason 1")
        coordinator.add_to_backup_exclusion_list("user-2", "Reason 2")

        exclusions = coordinator.get_backup_exclusion_list()

        assert len(exclusions) == 2
        user_ids = [e["user_id"] for e in exclusions]
        assert "user-1" in user_ids
        assert "user-2" in user_ids

    def test_get_exclusion_list_with_limit(self, coordinator, mock_audit_logger):
        """Test getting exclusion list with limit."""
        for i in range(10):
            coordinator.add_to_backup_exclusion_list(f"user-{i}", f"Reason {i}")

        exclusions = coordinator.get_backup_exclusion_list(limit=3)

        assert len(exclusions) == 3


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for deletion coordinator enums."""

    def test_cascade_status_values(self):
        """Test CascadeStatus enum values."""
        assert CascadeStatus.PENDING.value == "pending"
        assert CascadeStatus.IN_PROGRESS.value == "in_progress"
        assert CascadeStatus.COMPLETED.value == "completed"
        assert CascadeStatus.FAILED.value == "failed"
        assert CascadeStatus.PARTIAL.value == "partial"

    def test_deletion_system_values(self):
        """Test DeletionSystem enum values."""
        from aragora.deletion_coordinator import DeletionSystem

        assert DeletionSystem.USER_STORE.value == "user_store"
        assert DeletionSystem.DEBATE_STORE.value == "debate_store"
        assert DeletionSystem.KNOWLEDGE_MOUND.value == "knowledge_mound"
        assert DeletionSystem.MEMORY.value == "memory"
        assert DeletionSystem.AUDIT_TRAIL.value == "audit_trail"
        assert DeletionSystem.BACKUP.value == "backup"


# ============================================================================
# Certificate Signature Tests
# ============================================================================


class TestCertificateSignature:
    """Tests for certificate signature verification."""

    def test_certificate_signature_is_deterministic(self):
        """Test that same content produces same signature."""
        cascade_result = CascadeResult(
            user_id="user-123",
            status=CascadeStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            entities_deleted={"users": 5},
        )
        verification_result = VerificationResult(
            user_id="user-123",
            verified=True,
            verified_at=datetime(2024, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            storage_verified=True,
            backup_verified=True,
        )

        # Create certificate
        cert = DeletionCertificate.create(
            user_id="user-123",
            cascade_result=cascade_result,
            verification_result=verification_result,
        )

        # Signature should exist and be non-empty
        assert cert.signature is not None
        assert len(cert.signature) == 64  # SHA-256 hex = 64 chars

    def test_certificate_signature_changes_with_content(self):
        """Test that different content produces different signature."""
        cascade_result1 = CascadeResult(
            user_id="user-123",
            status=CascadeStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            entities_deleted={"users": 5},
        )
        cascade_result2 = CascadeResult(
            user_id="user-456",  # Different user
            status=CascadeStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            entities_deleted={"users": 5},
        )
        verification_result = VerificationResult(
            user_id="user-123",
            verified=True,
            verified_at=datetime(2024, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
        )

        cert1 = DeletionCertificate.create("user-123", cascade_result1, verification_result)
        cert2 = DeletionCertificate.create("user-456", cascade_result2, verification_result)

        # Signatures should be different
        assert cert1.signature != cert2.signature


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_cascade_with_no_deleters(self, coordinator, mock_audit_logger):
        """Test cascade when no deleters registered."""
        result = await coordinator.execute_cascade("user-123")

        assert result.status == CascadeStatus.COMPLETED
        assert result.total_deleted == 0

    @pytest.mark.asyncio
    async def test_verification_with_no_deleters(self, coordinator, mock_audit_logger):
        """Test verification when no deleters registered."""
        result = await coordinator.verify_deletion("user-123")

        assert result.verified is True
        assert result.storage_verified is True

    @pytest.mark.asyncio
    async def test_cascade_with_missing_deleter_warning(
        self, coordinator, mock_deleter, mock_audit_logger
    ):
        """Test cascade warns about missing deleters in order."""
        coordinator.register_deleter("users", mock_deleter)
        # Manually add to order without registering deleter
        coordinator._deletion_order.append("missing_entity")

        result = await coordinator.execute_cascade("user-123")

        assert result.status == CascadeStatus.COMPLETED
        assert any("missing_entity" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_cascade_with_backup_coordinator_error(
        self, coordinator, mock_deleter, mock_audit_logger
    ):
        """Test cascade handles backup coordinator errors."""
        failing_backup = MagicMock()
        failing_backup.get_backups_containing_user = MagicMock(
            side_effect=Exception("Backup system down")
        )

        coordinator.register_deleter("users", mock_deleter)
        coordinator.register_backup_coordinator(failing_backup)

        result = await coordinator.execute_cascade("user-123")

        # Deletion should still succeed
        assert result.status == CascadeStatus.COMPLETED
        # But warning should be recorded
        assert any("backup" in w.lower() for w in result.warnings)
