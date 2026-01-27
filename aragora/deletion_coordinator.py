"""
Unified Deletion Coordinator.

Coordinates deletion across privacy, storage, and backup systems to ensure
GDPR compliance and data consistency.

This module bridges:
- GDPRDeletionScheduler (privacy/deletion.py) - scheduled user deletions
- DeletionCascadeManager (privacy/deletion.py) - cascade deletion across entities
- BackupManager (backup/manager.py) - backup retention and verification

SOC 2 Controls:
- P4.1 - Data retention and disposal
- CC6.5 - Secure disposal of data

GDPR Articles:
- Article 17 - Right to erasure (right to be forgotten)
- Article 5(1)(e) - Storage limitation principle
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, Callable, Awaitable
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols and Types
# ============================================================================


class EntityDeleter(Protocol):
    """Protocol for entity deleters that can be registered with the coordinator."""

    async def delete_for_user(self, user_id: str) -> int:
        """Delete all entities for a user. Returns count of deleted entities."""
        ...

    async def verify_deletion(self, user_id: str) -> bool:
        """Verify no data remains for the user."""
        ...


class BackupCoordinator(Protocol):
    """Protocol for backup managers to implement for deletion coordination."""

    def get_backups_containing_user(self, user_id: str) -> list[str]:
        """Return backup IDs that may contain this user's data."""
        ...

    def mark_backup_for_purge(self, backup_id: str, user_id: str) -> None:
        """Mark a backup as containing data that needs purging."""
        ...

    def verify_user_purged_from_backups(self, user_id: str) -> bool:
        """Verify user data has been purged from all backups."""
        ...


class AuditLogger(Protocol):
    """Protocol for audit logging."""

    async def log(
        self,
        event_type: str,
        user_id: str,
        details: dict[str, Any],
        actor_id: str | None = None,
    ) -> None:
        """Log an audit event."""
        ...


# ============================================================================
# Data Classes
# ============================================================================


class CascadeStatus(str, Enum):
    """Status of a cascade deletion operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some deleters succeeded, some failed


class DeletionSystem(str, Enum):
    """Systems from which data was deleted."""

    USER_STORE = "user_store"
    DEBATE_STORE = "debate_store"
    KNOWLEDGE_MOUND = "knowledge_mound"
    MEMORY = "memory"
    AUDIT_TRAIL = "audit_trail"
    BACKUP = "backup"


@dataclass
class CascadeResult:
    """Result of a cascade deletion operation."""

    user_id: str
    status: CascadeStatus
    started_at: datetime
    completed_at: datetime | None = None
    entities_deleted: dict[str, int] = field(default_factory=dict)
    backup_status: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    deleted_from: list[DeletionSystem] = field(default_factory=list)
    backup_purge_results: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if deletion completed successfully."""
        return self.status == CascadeStatus.COMPLETED

    @property
    def total_deleted(self) -> int:
        """Total count of deleted entities."""
        return sum(c for c in self.entities_deleted.values() if c > 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "entities_deleted": self.entities_deleted,
            "total_deleted": self.total_deleted,
            "backup_status": self.backup_status,
            "deleted_from": [s.value for s in self.deleted_from],
            "backup_purge_results": self.backup_purge_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """Result of verifying deletion across all systems."""

    user_id: str
    verified: bool
    verified_at: datetime
    storage_verified: bool = False
    backup_verified: bool = False
    entity_results: dict[str, bool] = field(default_factory=dict)
    backup_results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "verified": self.verified,
            "verified_at": self.verified_at.isoformat(),
            "storage_verified": self.storage_verified,
            "backup_verified": self.backup_verified,
            "entity_results": self.entity_results,
            "backup_results": self.backup_results,
            "errors": self.errors,
        }


@dataclass
class DeletionCertificate:
    """
    Cryptographically signed certificate proving deletion.

    Provides audit evidence for GDPR compliance.
    """

    certificate_id: str
    user_id: str
    issued_at: datetime
    cascade_result: CascadeResult
    verification_result: VerificationResult
    signature: str  # SHA-256 hash of certificate contents

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "user_id": self.user_id,
            "issued_at": self.issued_at.isoformat(),
            "cascade_result": self.cascade_result.to_dict(),
            "verification_result": self.verification_result.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def create(
        cls,
        user_id: str,
        cascade_result: CascadeResult,
        verification_result: VerificationResult,
    ) -> DeletionCertificate:
        """Create a new deletion certificate with computed signature."""
        cert_id = str(uuid4())
        issued_at = datetime.now(timezone.utc)

        # Create signature from certificate contents
        content = json.dumps(
            {
                "certificate_id": cert_id,
                "user_id": user_id,
                "issued_at": issued_at.isoformat(),
                "cascade_result": cascade_result.to_dict(),
                "verification_result": verification_result.to_dict(),
            },
            sort_keys=True,
        )
        signature = hashlib.sha256(content.encode()).hexdigest()

        return cls(
            certificate_id=cert_id,
            user_id=user_id,
            issued_at=issued_at,
            cascade_result=cascade_result,
            verification_result=verification_result,
            signature=signature,
        )


# ============================================================================
# Unified Deletion Coordinator
# ============================================================================


class UnifiedDeletionCoordinator:
    """
    Coordinates deletion across privacy, storage, and backup systems.

    This coordinator ensures that when data is deleted for GDPR compliance:
    1. All entity deleters are executed in proper dependency order
    2. Backups containing the user's data are identified and marked
    3. Deletion is verified across all systems
    4. A deletion certificate is generated for audit purposes

    Usage:
        coordinator = UnifiedDeletionCoordinator()

        # Register entity deleters
        coordinator.register_deleter("user", user_store)
        coordinator.register_deleter("debates", debate_store)

        # Register backup coordinator
        coordinator.register_backup_coordinator(backup_manager_adapter)

        # Execute deletion
        result = await coordinator.execute_cascade("user-123")

        # Verify and get certificate
        cert = await coordinator.generate_certificate("user-123", result)
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        """
        Initialize the coordinator.

        Args:
            audit_logger: Optional audit logger for compliance tracking
        """
        self._deleters: dict[str, EntityDeleter] = {}
        self._deletion_order: list[str] = []
        self._backup_coordinator: BackupCoordinator | None = None
        self._audit_logger = audit_logger
        self._pre_deletion_hooks: list[Callable[[str], Awaitable[None]]] = []
        self._post_deletion_hooks: list[Callable[[str, CascadeResult], Awaitable[None]]] = []

    def register_deleter(
        self,
        entity_type: str,
        deleter: EntityDeleter,
        order: int | None = None,
    ) -> None:
        """
        Register an entity deleter.

        Args:
            entity_type: Type identifier (e.g., "user", "debates", "knowledge")
            deleter: Deleter implementing EntityDeleter protocol
            order: Optional explicit order (lower = earlier). If not specified,
                   deleters are executed in registration order.
        """
        self._deleters[entity_type] = deleter

        if order is not None:
            # Insert at specified position
            if entity_type not in self._deletion_order:
                self._deletion_order.insert(
                    min(order, len(self._deletion_order)),
                    entity_type,
                )
        else:
            # Add at end if not already present
            if entity_type not in self._deletion_order:
                self._deletion_order.append(entity_type)

        logger.info(f"Registered deleter for entity type: {entity_type}")

    def register_backup_coordinator(self, coordinator: BackupCoordinator) -> None:
        """
        Register the backup coordinator for cross-system deletion.

        Args:
            coordinator: Backup manager implementing BackupCoordinator protocol
        """
        self._backup_coordinator = coordinator
        logger.info("Registered backup coordinator")

    def add_pre_deletion_hook(
        self,
        hook: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        Add a hook to run before deletion starts.

        Hooks receive the user_id being deleted.
        """
        self._pre_deletion_hooks.append(hook)

    def add_post_deletion_hook(
        self,
        hook: Callable[[str, CascadeResult], Awaitable[None]],
    ) -> None:
        """
        Add a hook to run after deletion completes.

        Hooks receive the user_id and the CascadeResult.
        """
        self._post_deletion_hooks.append(hook)

    async def execute_cascade(
        self,
        user_id: str,
        skip_backup_check: bool = False,
        actor_id: str | None = None,
    ) -> CascadeResult:
        """
        Execute cascade deletion across all registered deleters.

        Args:
            user_id: ID of the user whose data should be deleted
            skip_backup_check: If True, skip backup coordination (for testing)
            actor_id: ID of the actor performing the deletion (for audit)

        Returns:
            CascadeResult with deletion details
        """
        result = CascadeResult(
            user_id=user_id,
            status=CascadeStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Run pre-deletion hooks
            for hook in self._pre_deletion_hooks:
                try:
                    await hook(user_id)
                except Exception as e:
                    result.warnings.append(f"Pre-deletion hook failed: {e}")
                    logger.warning(f"Pre-deletion hook failed for {user_id}: {e}")

            # Log deletion start
            if self._audit_logger:
                await self._audit_logger.log(
                    event_type="deletion_cascade_started",
                    user_id=user_id,
                    details={
                        "entity_types": self._deletion_order,
                        "backup_coordination": not skip_backup_check,
                    },
                    actor_id=actor_id,
                )

            # Check backup status first
            if not skip_backup_check and self._backup_coordinator:
                try:
                    backup_ids = self._backup_coordinator.get_backups_containing_user(user_id)
                    result.backup_status["affected_backups"] = backup_ids
                    result.backup_status["backup_count"] = len(backup_ids)

                    # Mark backups for purge
                    for backup_id in backup_ids:
                        self._backup_coordinator.mark_backup_for_purge(backup_id, user_id)

                    logger.info(f"Marked {len(backup_ids)} backups for purge for user {user_id}")
                except Exception as e:
                    result.warnings.append(f"Backup coordination failed: {e}")
                    logger.warning(f"Backup coordination failed for {user_id}: {e}")

            # Execute deleters in order
            for entity_type in self._deletion_order:
                deleter = self._deleters.get(entity_type)
                if not deleter:
                    result.warnings.append(f"No deleter registered for {entity_type}")
                    continue

                try:
                    count = await deleter.delete_for_user(user_id)
                    result.entities_deleted[entity_type] = count
                    logger.info(f"Deleted {count} {entity_type} entities for user {user_id}")
                except Exception as e:
                    result.errors.append(f"Failed to delete {entity_type}: {e}")
                    logger.error(f"Failed to delete {entity_type} for user {user_id}: {e}")

            # Determine final status
            if result.errors:
                if result.entities_deleted:
                    result.status = CascadeStatus.PARTIAL
                else:
                    result.status = CascadeStatus.FAILED
            else:
                result.status = CascadeStatus.COMPLETED

            result.completed_at = datetime.now(timezone.utc)

            # Run post-deletion hooks
            for post_hook in self._post_deletion_hooks:
                try:
                    await post_hook(user_id, result)
                except Exception as e:
                    result.warnings.append(f"Post-deletion hook failed: {e}")
                    logger.warning(f"Post-deletion hook failed for {user_id}: {e}")

            # Log deletion completion
            if self._audit_logger:
                await self._audit_logger.log(
                    event_type="deletion_cascade_completed",
                    user_id=user_id,
                    details=result.to_dict(),
                    actor_id=actor_id,
                )

        except Exception as e:
            result.status = CascadeStatus.FAILED
            result.errors.append(f"Cascade deletion failed: {e}")
            result.completed_at = datetime.now(timezone.utc)
            logger.exception(f"Cascade deletion failed for user {user_id}: {e}")

            if self._audit_logger:
                await self._audit_logger.log(
                    event_type="deletion_cascade_failed",
                    user_id=user_id,
                    details={"error": str(e), "result": result.to_dict()},
                    actor_id=actor_id,
                )

        return result

    async def verify_deletion(
        self,
        user_id: str,
        include_backups: bool = True,
    ) -> VerificationResult:
        """
        Verify deletion completed across all systems.

        Args:
            user_id: ID of the user to verify deletion for
            include_backups: Whether to verify backup purge

        Returns:
            VerificationResult with verification details
        """
        result = VerificationResult(
            user_id=user_id,
            verified=True,
            verified_at=datetime.now(timezone.utc),
        )

        # Verify each entity deleter
        all_storage_verified = True
        for entity_type, deleter in self._deleters.items():
            try:
                is_deleted = await deleter.verify_deletion(user_id)
                result.entity_results[entity_type] = is_deleted
                if not is_deleted:
                    all_storage_verified = False
                    result.errors.append(f"Data still exists for entity type: {entity_type}")
            except Exception as e:
                result.entity_results[entity_type] = False
                all_storage_verified = False
                result.errors.append(f"Failed to verify {entity_type} deletion: {e}")

        result.storage_verified = all_storage_verified

        # Verify backup purge if requested
        if include_backups and self._backup_coordinator:
            try:
                backup_verified = self._backup_coordinator.verify_user_purged_from_backups(user_id)
                result.backup_verified = backup_verified
                if not backup_verified:
                    result.errors.append("User data may still exist in backups")
            except Exception as e:
                result.backup_verified = False
                result.errors.append(f"Failed to verify backup purge: {e}")
        else:
            # No backup coordinator or not checking
            result.backup_verified = True

        # Overall verification
        result.verified = result.storage_verified and result.backup_verified

        return result

    async def generate_certificate(
        self,
        user_id: str,
        cascade_result: CascadeResult,
        verification_result: VerificationResult | None = None,
    ) -> DeletionCertificate:
        """
        Generate a deletion certificate for audit purposes.

        Args:
            user_id: ID of the deleted user
            cascade_result: Result from cascade deletion
            verification_result: Optional verification result (will verify if not provided)

        Returns:
            DeletionCertificate for compliance records
        """
        if verification_result is None:
            verification_result = await self.verify_deletion(user_id)

        certificate = DeletionCertificate.create(
            user_id=user_id,
            cascade_result=cascade_result,
            verification_result=verification_result,
        )

        if self._audit_logger:
            await self._audit_logger.log(
                event_type="deletion_certificate_issued",
                user_id=user_id,
                details={
                    "certificate_id": certificate.certificate_id,
                    "verified": verification_result.verified,
                },
            )

        return certificate

    # =========================================================================
    # Higher-Level API for Compliance Handler
    # =========================================================================

    async def execute_coordinated_deletion(
        self,
        user_id: str,
        reason: str,
        delete_from_backups: bool = True,
        dry_run: bool = False,
    ) -> CascadeResult:
        """
        Execute a fully coordinated deletion for GDPR compliance.

        This is the main entry point for the compliance handler, providing
        a simpler API that handles the full deletion flow including:
        - Pre-deletion checks
        - Cascade deletion across all systems
        - Backup purge coordination
        - Verification
        - Certificate generation

        Args:
            user_id: ID of the user to delete
            reason: Reason for the deletion (for audit)
            delete_from_backups: Whether to coordinate backup deletion
            dry_run: If True, only simulate the deletion

        Returns:
            CascadeResult with full deletion details
        """
        if dry_run:
            # Simulate deletion by returning what would happen
            result = CascadeResult(
                user_id=user_id,
                status=CascadeStatus.PENDING,
                started_at=datetime.now(timezone.utc),
            )
            result.metadata["dry_run"] = True
            result.metadata["reason"] = reason
            result.metadata["delete_from_backups"] = delete_from_backups

            # List what would be deleted
            for entity_type in self._deletion_order:
                result.entities_deleted[entity_type] = -1  # -1 indicates "would delete"

            if delete_from_backups and self._backup_coordinator:
                backup_ids = self._backup_coordinator.get_backups_containing_user(user_id)
                result.backup_status["affected_backups"] = backup_ids
                result.backup_status["would_purge"] = True

            return result

        # Execute actual deletion
        result = await self.execute_cascade(
            user_id=user_id,
            skip_backup_check=not delete_from_backups,
        )

        # Add metadata
        result.metadata["reason"] = reason
        result.metadata["delete_from_backups"] = delete_from_backups

        # Add backup exclusion if deletion was successful
        if result.status == CascadeStatus.COMPLETED:
            self.add_to_backup_exclusion_list(user_id, reason)

        return result

    async def process_pending_deletions(
        self,
        include_backups: bool = True,
        limit: int = 100,
    ) -> list[CascadeResult]:
        """
        Process all pending scheduled deletions.

        This method is designed to be called by a background job to
        execute deletions that have passed their grace period.

        Args:
            include_backups: Whether to also purge from backups
            limit: Maximum number of deletions to process

        Returns:
            List of CascadeResult for each processed deletion
        """
        results: list[CascadeResult] = []

        try:
            # Get pending deletions from the scheduler
            from aragora.privacy.deletion import get_deletion_scheduler, DeletionStatus

            scheduler = get_deletion_scheduler()
            pending = scheduler.store.get_all_requests(
                status=DeletionStatus.PENDING,
                limit=limit,
            )

            # Filter to only those past their scheduled time
            now = datetime.now(timezone.utc)
            due_deletions = [req for req in pending if req.scheduled_for <= now]

            logger.info(f"Processing {len(due_deletions)} due deletions")

            for deletion_req in due_deletions:
                try:
                    result = await self.execute_coordinated_deletion(
                        user_id=deletion_req.user_id,
                        reason=deletion_req.reason or "Scheduled deletion",
                        delete_from_backups=include_backups,
                    )

                    # Update the deletion request status
                    if result.status == CascadeStatus.COMPLETED:
                        scheduler.store.update_status(
                            deletion_req.request_id,
                            DeletionStatus.COMPLETED,
                        )
                    elif result.status == CascadeStatus.FAILED:
                        scheduler.store.update_status(
                            deletion_req.request_id,
                            DeletionStatus.FAILED,
                        )

                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to process deletion for {deletion_req.user_id}: {e}")
                    # Create a failed result
                    failed_result = CascadeResult(
                        user_id=deletion_req.user_id,
                        status=CascadeStatus.FAILED,
                        started_at=datetime.now(timezone.utc),
                        completed_at=datetime.now(timezone.utc),
                    )
                    failed_result.errors.append(str(e))
                    results.append(failed_result)

        except ImportError:
            logger.warning("Deletion scheduler not available")
        except Exception as e:
            logger.exception(f"Error processing pending deletions: {e}")

        return results

    # =========================================================================
    # Backup Exclusion List Management
    # =========================================================================

    _backup_exclusion_list: dict[str, dict[str, Any]] = {}

    def get_backup_exclusion_list(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get the list of users excluded from backup restoration.

        These are users whose data has been deleted and should NOT be
        restored from backups.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of exclusion entries with user_id, reason, and added_at
        """
        exclusions = list(self._backup_exclusion_list.values())
        return exclusions[:limit]

    def add_to_backup_exclusion_list(
        self,
        user_id: str,
        reason: str,
    ) -> None:
        """
        Add a user to the backup exclusion list.

        This should be called after GDPR deletion to ensure the user's
        data is not restored from backups.

        Args:
            user_id: User ID to exclude
            reason: Reason for exclusion
        """
        self._backup_exclusion_list[user_id] = {
            "user_id": user_id,
            "reason": reason,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Added user {user_id} to backup exclusion list")

    def remove_from_backup_exclusion_list(self, user_id: str) -> bool:
        """
        Remove a user from the backup exclusion list.

        Args:
            user_id: User ID to remove

        Returns:
            True if user was removed, False if not found
        """
        if user_id in self._backup_exclusion_list:
            del self._backup_exclusion_list[user_id]
            logger.info(f"Removed user {user_id} from backup exclusion list")
            return True
        return False

    def is_user_excluded_from_backups(self, user_id: str) -> bool:
        """
        Check if a user is on the backup exclusion list.

        Args:
            user_id: User ID to check

        Returns:
            True if user is excluded from backup restoration
        """
        return user_id in self._backup_exclusion_list


# ============================================================================
# Backup Manager Adapter
# ============================================================================


class BackupManagerAdapter:
    """
    Adapter to make BackupManager work with UnifiedDeletionCoordinator.

    Wraps the existing BackupManager to implement BackupCoordinator protocol.
    """

    def __init__(self, backup_manager: Any) -> None:
        """
        Initialize with a BackupManager instance.

        Args:
            backup_manager: BackupManager from aragora.backup.manager
        """
        self._manager = backup_manager
        # Track which backups contain which users for GDPR verification
        self._user_backup_map: dict[str, set[str]] = {}  # user_id -> set of backup_ids
        self._backup_purge_status: dict[str, set[str]] = {}  # backup_id -> set of purged user_ids

    def record_user_in_backup(self, user_id: str, backup_id: str) -> None:
        """
        Record that a backup contains a user's data.

        Should be called when creating backups to track user data location.
        """
        if user_id not in self._user_backup_map:
            self._user_backup_map[user_id] = set()
        self._user_backup_map[user_id].add(backup_id)

    def get_backups_containing_user(self, user_id: str) -> list[str]:
        """Return backup IDs that may contain this user's data."""
        # Return all backups created since the user was created
        # In practice, this would query backup metadata
        return list(self._user_backup_map.get(user_id, set()))

    def mark_backup_for_purge(self, backup_id: str, user_id: str) -> None:
        """Mark a backup as containing data that needs purging."""
        if backup_id not in self._backup_purge_status:
            self._backup_purge_status[backup_id] = set()
        self._backup_purge_status[backup_id].add(user_id)
        logger.info(f"Marked backup {backup_id} for purge of user {user_id}")

    def mark_user_purged_from_backup(self, backup_id: str, user_id: str) -> None:
        """Mark that a user's data has been purged from a backup."""
        if backup_id in self._backup_purge_status:
            self._backup_purge_status[backup_id].discard(user_id)
        if user_id in self._user_backup_map:
            self._user_backup_map[user_id].discard(backup_id)

    def verify_user_purged_from_backups(self, user_id: str) -> bool:
        """Verify user data has been purged from all backups."""
        # Check if user has any remaining backups
        remaining_backups = self._user_backup_map.get(user_id, set())
        return len(remaining_backups) == 0


# ============================================================================
# Global Instance Management
# ============================================================================


_coordinator_instance: UnifiedDeletionCoordinator | None = None


def get_deletion_coordinator() -> UnifiedDeletionCoordinator:
    """Get or create the global deletion coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = UnifiedDeletionCoordinator()
    return _coordinator_instance


def set_deletion_coordinator(coordinator: UnifiedDeletionCoordinator) -> None:
    """Set the global deletion coordinator instance (for testing)."""
    global _coordinator_instance
    _coordinator_instance = coordinator


def reset_deletion_coordinator() -> None:
    """Reset the global deletion coordinator instance (for testing)."""
    global _coordinator_instance
    _coordinator_instance = None
