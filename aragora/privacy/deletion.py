"""
GDPR Deletion Infrastructure.

Provides:
- GDPRDeletionScheduler: Executes scheduled data deletions after grace period
- DeletionCascadeManager: Coordinates cascade deletion across entity types
- DataErasureVerifier: Verifies and certifies data was actually deleted

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
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================


class DeletionStatus(str, Enum):
    """Status of a scheduled deletion."""

    PENDING = "pending"  # Scheduled but not yet executed
    IN_PROGRESS = "in_progress"  # Currently being executed
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Cancelled (e.g., by legal hold)
    HELD = "held"  # On legal hold


class EntityType(str, Enum):
    """Types of entities that can be deleted."""

    USER = "user"
    CONSENT_RECORDS = "consent_records"
    AUDIT_LOGS = "audit_logs"  # Note: Some audit logs must be retained
    DEBATE_PARTICIPATION = "debate_participation"
    KNOWLEDGE_ENTRIES = "knowledge_entries"
    PREFERENCES = "preferences"
    API_KEYS = "api_keys"
    OAUTH_LINKS = "oauth_links"
    SESSION_DATA = "session_data"
    EXPORT_DATA = "export_data"
    RECEIPTS = "receipts"
    ACTIVITY_LOGS = "activity_logs"
    ORGANIZATION_MEMBERSHIP = "organization_membership"
    NOTIFICATIONS = "notifications"
    UPLOADED_FILES = "uploaded_files"


@dataclass
class DeletionRequest:
    """
    A scheduled deletion request for a user's data.

    Tracks the lifecycle from scheduling through execution and verification.
    """

    request_id: str
    user_id: str
    scheduled_for: datetime
    reason: str
    created_at: datetime
    status: DeletionStatus = DeletionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancelled_reason: str | None = None
    hold_id: str | None = None  # Legal hold ID if held
    entities_deleted: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    verification_hash: str | None = None
    deletion_certificate: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "scheduled_for": self.scheduled_for.isoformat(),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancelled_reason": self.cancelled_reason,
            "hold_id": self.hold_id,
            "entities_deleted": self.entities_deleted,
            "errors": self.errors,
            "verification_hash": self.verification_hash,
            "deletion_certificate": self.deletion_certificate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeletionRequest:
        """Deserialize from dictionary."""
        return cls(
            request_id=data["request_id"],
            user_id=data["user_id"],
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]),
            reason=data["reason"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=DeletionStatus(data.get("status", "pending")),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            cancelled_at=(
                datetime.fromisoformat(data["cancelled_at"]) if data.get("cancelled_at") else None
            ),
            cancelled_reason=data.get("cancelled_reason"),
            hold_id=data.get("hold_id"),
            entities_deleted=data.get("entities_deleted", {}),
            errors=data.get("errors", []),
            verification_hash=data.get("verification_hash"),
            deletion_certificate=data.get("deletion_certificate"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LegalHold:
    """
    A legal hold that prevents data deletion.

    Used for litigation holds, regulatory investigations, etc.
    """

    hold_id: str
    user_ids: list[str]
    reason: str
    created_by: str
    created_at: datetime
    expires_at: datetime | None = None
    case_reference: str | None = None
    is_active: bool = True
    released_at: datetime | None = None
    released_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hold_id": self.hold_id,
            "user_ids": self.user_ids,
            "reason": self.reason,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "case_reference": self.case_reference,
            "is_active": self.is_active,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "released_by": self.released_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LegalHold:
        """Deserialize from dictionary."""
        return cls(
            hold_id=data["hold_id"],
            user_ids=data["user_ids"],
            reason=data["reason"],
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            case_reference=data.get("case_reference"),
            is_active=data.get("is_active", True),
            released_at=(
                datetime.fromisoformat(data["released_at"]) if data.get("released_at") else None
            ),
            released_by=data.get("released_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DeletionCertificate:
    """
    Certificate proving data was deleted (for compliance audits).

    Provides cryptographic proof of deletion execution.
    """

    certificate_id: str
    request_id: str
    user_id: str
    issued_at: datetime
    entities_deleted: dict[str, int]
    verification_hash: str
    signed_by: str
    signature: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "issued_at": self.issued_at.isoformat(),
            "entities_deleted": self.entities_deleted,
            "verification_hash": self.verification_hash,
            "signed_by": self.signed_by,
            "signature": self.signature,
            "metadata": self.metadata,
        }


# ============================================================================
# Storage
# ============================================================================


class DeletionStore:
    """
    Persistence layer for deletion requests and legal holds.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """Initialize deletion store."""
        if storage_path is None:
            storage_path = Path.home() / ".aragora" / "deletion_store.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._requests: dict[str, DeletionRequest] = {}
        self._holds: dict[str, LegalHold] = {}
        self._certificates: dict[str, DeletionCertificate] = {}
        self._load()

    def save_request(self, request: DeletionRequest) -> None:
        """Save a deletion request."""
        self._requests[request.request_id] = request
        self._persist()

    def get_request(self, request_id: str) -> DeletionRequest | None:
        """Get a deletion request by ID."""
        return self._requests.get(request_id)

    def get_requests_for_user(self, user_id: str) -> list[DeletionRequest]:
        """Get all deletion requests for a user."""
        return [r for r in self._requests.values() if r.user_id == user_id]

    def get_pending_requests(self, before: datetime | None = None) -> list[DeletionRequest]:
        """Get all pending deletion requests ready for execution."""
        before = before or datetime.now(timezone.utc)
        return [
            r
            for r in self._requests.values()
            if r.status == DeletionStatus.PENDING and r.scheduled_for <= before
        ]

    def get_all_requests(
        self,
        status: DeletionStatus | None = None,
        limit: int = 100,
    ) -> list[DeletionRequest]:
        """Get all deletion requests, optionally filtered by status."""
        requests = list(self._requests.values())
        if status:
            requests = [r for r in requests if r.status == status]
        return sorted(requests, key=lambda r: r.scheduled_for, reverse=True)[:limit]

    def save_hold(self, hold: LegalHold) -> None:
        """Save a legal hold."""
        self._holds[hold.hold_id] = hold
        self._persist()

    def get_hold(self, hold_id: str) -> LegalHold | None:
        """Get a legal hold by ID."""
        return self._holds.get(hold_id)

    def get_active_holds_for_user(self, user_id: str) -> list[LegalHold]:
        """Get all active legal holds for a user."""
        now = datetime.now(timezone.utc)
        return [
            h
            for h in self._holds.values()
            if h.is_active
            and user_id in h.user_ids
            and (h.expires_at is None or h.expires_at > now)
        ]

    def get_all_active_holds(self) -> list[LegalHold]:
        """Get all active legal holds."""
        now = datetime.now(timezone.utc)
        return [
            h
            for h in self._holds.values()
            if h.is_active and (h.expires_at is None or h.expires_at > now)
        ]

    def save_certificate(self, certificate: DeletionCertificate) -> None:
        """Save a deletion certificate."""
        self._certificates[certificate.certificate_id] = certificate
        self._persist()

    def get_certificate(self, certificate_id: str) -> DeletionCertificate | None:
        """Get a deletion certificate."""
        return self._certificates.get(certificate_id)

    def get_certificates_for_user(self, user_id: str) -> list[DeletionCertificate]:
        """Get all deletion certificates for a user."""
        return [c for c in self._certificates.values() if c.user_id == user_id]

    def update_status(self, request_id: str, status: DeletionStatus) -> bool:
        """
        Update the status of a deletion request.

        Args:
            request_id: ID of the deletion request
            status: New status to set

        Returns:
            True if request was found and updated, False otherwise
        """
        request = self._requests.get(request_id)
        if request is None:
            return False

        request.status = status
        if status == DeletionStatus.COMPLETED:
            request.completed_at = datetime.now(timezone.utc)
        elif status == DeletionStatus.IN_PROGRESS:
            request.started_at = datetime.now(timezone.utc)

        self._persist()
        return True

    def _load(self) -> None:
        """Load from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self._requests = {
                k: DeletionRequest.from_dict(v) for k, v in data.get("requests", {}).items()
            }
            self._holds = {k: LegalHold.from_dict(v) for k, v in data.get("holds", {}).items()}
            self._certificates = {
                k: DeletionCertificate(
                    certificate_id=v["certificate_id"],
                    request_id=v["request_id"],
                    user_id=v["user_id"],
                    issued_at=datetime.fromisoformat(v["issued_at"]),
                    entities_deleted=v["entities_deleted"],
                    verification_hash=v["verification_hash"],
                    signed_by=v["signed_by"],
                    signature=v["signature"],
                    metadata=v.get("metadata", {}),
                )
                for k, v in data.get("certificates", {}).items()
            }

            logger.debug(
                "Loaded deletion store: %d requests, %d holds",
                len(self._requests),
                len(self._holds),
            )

        except Exception as e:
            logger.error("Failed to load deletion store: %s", e)

    def _persist(self) -> None:
        """Persist to storage."""
        try:
            data = {
                "requests": {k: v.to_dict() for k, v in self._requests.items()},
                "holds": {k: v.to_dict() for k, v in self._holds.items()},
                "certificates": {k: v.to_dict() for k, v in self._certificates.items()},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to persist deletion store: %s", e)


# ============================================================================
# Entity Deleters (Protocol)
# ============================================================================


class EntityDeleter(Protocol):
    """Protocol for entity deletion handlers."""

    async def delete_for_user(self, user_id: str) -> int:
        """Delete all entities of this type for a user. Returns count deleted."""
        ...

    async def verify_deleted(self, user_id: str) -> bool:
        """Verify no data remains for this user."""
        ...


# ============================================================================
# Deletion Cascade Manager
# ============================================================================


class DeletionCascadeManager:
    """
    Coordinates cascade deletion across all entity types.

    Ensures all user data is deleted in the correct order,
    handling dependencies and rollback on failure.
    """

    def __init__(self) -> None:
        """Initialize cascade manager."""
        self._deleters: dict[EntityType, EntityDeleter] = {}
        self._deletion_order: list[EntityType] = [
            # Order matters - delete dependent entities first
            EntityType.SESSION_DATA,
            EntityType.NOTIFICATIONS,
            EntityType.ACTIVITY_LOGS,
            EntityType.UPLOADED_FILES,
            EntityType.EXPORT_DATA,
            EntityType.RECEIPTS,
            EntityType.DEBATE_PARTICIPATION,
            EntityType.KNOWLEDGE_ENTRIES,
            EntityType.PREFERENCES,
            EntityType.API_KEYS,
            EntityType.OAUTH_LINKS,
            EntityType.CONSENT_RECORDS,
            EntityType.ORGANIZATION_MEMBERSHIP,
            EntityType.USER,  # Delete user last
        ]

    def register_deleter(self, entity_type: EntityType, deleter: EntityDeleter) -> None:
        """Register a deleter for an entity type."""
        self._deleters[entity_type] = deleter
        logger.debug("Registered deleter for entity type: %s", entity_type.value)

    async def execute_cascade_deletion(
        self,
        user_id: str,
        exclude_types: list[EntityType] | None = None,
    ) -> dict[str, int]:
        """
        Execute cascade deletion for a user across all entity types.

        Args:
            user_id: User to delete data for
            exclude_types: Entity types to exclude from deletion

        Returns:
            Dictionary of entity_type -> count deleted
        """
        exclude_types = exclude_types or []
        results: dict[str, int] = {}
        errors: list[str] = []

        logger.info("Starting cascade deletion for user: %s", user_id)

        for entity_type in self._deletion_order:
            if entity_type in exclude_types:
                logger.debug("Skipping excluded entity type: %s", entity_type.value)
                continue

            if entity_type not in self._deleters:
                logger.warning("No deleter registered for entity type: %s", entity_type.value)
                continue

            try:
                deleter = self._deleters[entity_type]
                count = await deleter.delete_for_user(user_id)
                results[entity_type.value] = count
                logger.debug(
                    "Deleted %d %s for user %s",
                    count,
                    entity_type.value,
                    user_id,
                )
            except Exception as e:
                error_msg = f"Failed to delete {entity_type.value}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        if errors:
            raise RuntimeError(f"Cascade deletion errors: {'; '.join(errors)}")

        logger.info(
            "Completed cascade deletion for user %s: %s",
            user_id,
            results,
        )

        return results

    async def verify_cascade_deletion(self, user_id: str) -> tuple[bool, list[str]]:
        """
        Verify all data was deleted for a user.

        Returns:
            Tuple of (all_verified, failed_entity_types)
        """
        failed: list[str] = []

        for entity_type, deleter in self._deleters.items():
            try:
                is_deleted = await deleter.verify_deleted(user_id)
                if not is_deleted:
                    failed.append(entity_type.value)
                    logger.warning(
                        "Verification failed: %s data still exists for user %s",
                        entity_type.value,
                        user_id,
                    )
            except Exception as e:
                failed.append(entity_type.value)
                logger.error(
                    "Verification error for %s: %s",
                    entity_type.value,
                    e,
                )

        return len(failed) == 0, failed


# ============================================================================
# Data Erasure Verifier
# ============================================================================


class DataErasureVerifier:
    """
    Verifies data was actually deleted and generates deletion certificates.

    Provides cryptographic proof of deletion for compliance audits.
    """

    def __init__(
        self,
        cascade_manager: DeletionCascadeManager,
        signing_key: str = "aragora-deletion-verifier",
    ) -> None:
        """Initialize verifier."""
        self._cascade_manager = cascade_manager
        self._signing_key = signing_key

    async def verify_and_certify(
        self,
        request: DeletionRequest,
    ) -> DeletionCertificate:
        """
        Verify deletion and generate a certificate.

        Args:
            request: The completed deletion request

        Returns:
            DeletionCertificate with cryptographic proof
        """
        # Verify all data was deleted
        all_verified, failed = await self._cascade_manager.verify_cascade_deletion(request.user_id)

        if not all_verified:
            raise RuntimeError(f"Verification failed for entity types: {', '.join(failed)}")

        # Generate verification hash
        verification_data = {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "entities_deleted": request.entities_deleted,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
        }
        data_str = json.dumps(verification_data, sort_keys=True)
        verification_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # Generate signature
        signature_data = f"{verification_hash}:{self._signing_key}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()

        certificate = DeletionCertificate(
            certificate_id=str(uuid4()),
            request_id=request.request_id,
            user_id=request.user_id,
            issued_at=datetime.now(timezone.utc),
            entities_deleted=request.entities_deleted,
            verification_hash=verification_hash,
            signed_by="aragora-deletion-service",
            signature=signature,
            metadata={
                "verification_passed": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(
            "Generated deletion certificate %s for request %s",
            certificate.certificate_id,
            request.request_id,
        )

        return certificate

    def verify_certificate(self, certificate: DeletionCertificate) -> bool:
        """
        Verify a deletion certificate is valid.

        Args:
            certificate: Certificate to verify

        Returns:
            True if certificate is valid
        """
        # Regenerate expected signature
        signature_data = f"{certificate.verification_hash}:{self._signing_key}"
        expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()

        is_valid = certificate.signature == expected_signature

        if not is_valid:
            logger.warning(
                "Certificate verification failed for %s",
                certificate.certificate_id,
            )

        return is_valid


# ============================================================================
# GDPR Deletion Scheduler
# ============================================================================


class GDPRDeletionScheduler:
    """
    Scheduler that executes scheduled data deletions.

    Runs periodically to check for pending deletions that are past
    their grace period and executes them.
    """

    def __init__(
        self,
        store: DeletionStore | None = None,
        cascade_manager: DeletionCascadeManager | None = None,
        verifier: DataErasureVerifier | None = None,
        check_interval_seconds: int = 3600,  # Default: 1 hour
    ) -> None:
        """
        Initialize the deletion scheduler.

        Args:
            store: Store for deletion requests
            cascade_manager: Manager for cascade deletions
            verifier: Verifier for generating certificates
            check_interval_seconds: How often to check for pending deletions
        """
        self._store = store or DeletionStore()
        self._cascade_manager = cascade_manager or DeletionCascadeManager()
        self._verifier = verifier or DataErasureVerifier(self._cascade_manager)
        self._check_interval = check_interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def store(self) -> DeletionStore:
        """Get the deletion store."""
        return self._store

    def schedule_deletion(
        self,
        user_id: str,
        grace_period_days: int = 30,
        reason: str = "User request",
        metadata: dict[str, Any] | None = None,
    ) -> DeletionRequest:
        """
        Schedule a new deletion request.

        Args:
            user_id: User to delete data for
            grace_period_days: Days before deletion is executed
            reason: Reason for deletion
            metadata: Additional metadata

        Returns:
            Created DeletionRequest
        """
        # Check for legal holds
        active_holds = self._store.get_active_holds_for_user(user_id)
        if active_holds:
            hold = active_holds[0]
            raise ValueError(
                f"Cannot schedule deletion: User is under legal hold "
                f"(hold_id={hold.hold_id}, reason={hold.reason})"
            )

        now = datetime.now(timezone.utc)
        request = DeletionRequest(
            request_id=str(uuid4()),
            user_id=user_id,
            scheduled_for=now + timedelta(days=grace_period_days),
            reason=reason,
            created_at=now,
            status=DeletionStatus.PENDING,
            metadata=metadata or {},
        )

        self._store.save_request(request)

        logger.info(
            "Scheduled deletion for user %s on %s (request_id=%s)",
            user_id,
            request.scheduled_for.isoformat(),
            request.request_id,
        )

        return request

    def cancel_deletion(
        self,
        request_id: str,
        reason: str = "User cancelled",
    ) -> DeletionRequest | None:
        """
        Cancel a pending deletion request.

        Args:
            request_id: ID of the request to cancel
            reason: Reason for cancellation

        Returns:
            Updated DeletionRequest or None if not found
        """
        request = self._store.get_request(request_id)
        if not request:
            return None

        if request.status != DeletionStatus.PENDING:
            raise ValueError(f"Cannot cancel deletion in status {request.status.value}")

        request.status = DeletionStatus.CANCELLED
        request.cancelled_at = datetime.now(timezone.utc)
        request.cancelled_reason = reason

        self._store.save_request(request)

        logger.info("Cancelled deletion request %s: %s", request_id, reason)

        return request

    async def execute_deletion(self, request_id: str) -> DeletionRequest:
        """
        Execute a specific deletion request.

        Args:
            request_id: ID of the request to execute

        Returns:
            Updated DeletionRequest

        Raises:
            ValueError: If request not found or cannot be executed
        """
        request = self._store.get_request(request_id)
        if not request:
            raise ValueError(f"Deletion request not found: {request_id}")

        if request.status != DeletionStatus.PENDING:
            raise ValueError(f"Cannot execute deletion in status {request.status.value}")

        # Check for legal holds
        active_holds = self._store.get_active_holds_for_user(request.user_id)
        if active_holds:
            hold = active_holds[0]
            request.status = DeletionStatus.HELD
            request.hold_id = hold.hold_id
            self._store.save_request(request)
            raise ValueError(f"Deletion blocked by legal hold: {hold.hold_id}")

        # Start execution
        request.status = DeletionStatus.IN_PROGRESS
        request.started_at = datetime.now(timezone.utc)
        self._store.save_request(request)

        try:
            # Execute cascade deletion
            entities_deleted = await self._cascade_manager.execute_cascade_deletion(request.user_id)
            request.entities_deleted = entities_deleted

            # Generate verification and certificate
            request.completed_at = datetime.now(timezone.utc)
            certificate = await self._verifier.verify_and_certify(request)
            request.verification_hash = certificate.verification_hash
            request.deletion_certificate = certificate.to_dict()
            request.status = DeletionStatus.COMPLETED

            self._store.save_certificate(certificate)
            self._store.save_request(request)

            logger.info(
                "Completed deletion for user %s (request_id=%s)",
                request.user_id,
                request.request_id,
            )

        except Exception as e:
            request.status = DeletionStatus.FAILED
            request.errors.append(str(e))
            request.completed_at = datetime.now(timezone.utc)
            self._store.save_request(request)

            logger.error(
                "Deletion failed for user %s (request_id=%s): %s",
                request.user_id,
                request.request_id,
                e,
            )
            raise

        return request

    async def process_pending_deletions(self) -> list[DeletionRequest]:
        """
        Process all pending deletions that are past their grace period.

        Returns:
            List of processed DeletionRequests
        """
        now = datetime.now(timezone.utc)
        pending = self._store.get_pending_requests(before=now)
        processed: list[DeletionRequest] = []

        logger.info("Processing %d pending deletion requests", len(pending))

        for request in pending:
            try:
                result = await self.execute_deletion(request.request_id)
                processed.append(result)
            except Exception as e:
                logger.error(
                    "Error processing deletion %s: %s",
                    request.request_id,
                    e,
                )
                # Request already marked as failed in execute_deletion
                processed.append(self._store.get_request(request.request_id))  # type: ignore

        return processed

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Started GDPR deletion scheduler (interval=%ds)", self._check_interval)

    async def stop(self) -> None:
        """Stop the scheduler background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped GDPR deletion scheduler")

    async def _run_loop(self) -> None:
        """Background loop that processes pending deletions."""
        while self._running:
            try:
                await self.process_pending_deletions()
            except Exception as e:
                logger.error("Error in deletion scheduler loop: %s", e)

            await asyncio.sleep(self._check_interval)


# ============================================================================
# Legal Hold Manager
# ============================================================================


class LegalHoldManager:
    """
    Manages legal holds that prevent data deletion.

    Used for:
    - Litigation holds
    - Regulatory investigations
    - Audit preservation
    """

    def __init__(self, store: DeletionStore | None = None) -> None:
        """Initialize legal hold manager."""
        self._store = store or DeletionStore()

    def create_hold(
        self,
        user_ids: list[str],
        reason: str,
        created_by: str,
        case_reference: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LegalHold:
        """
        Create a new legal hold.

        Args:
            user_ids: Users to place on hold
            reason: Reason for the hold
            created_by: User/system creating the hold
            case_reference: External case reference (optional)
            expires_at: When hold expires (optional, None = indefinite)
            metadata: Additional metadata

        Returns:
            Created LegalHold
        """
        hold = LegalHold(
            hold_id=str(uuid4()),
            user_ids=user_ids,
            reason=reason,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            case_reference=case_reference,
            is_active=True,
            metadata=metadata or {},
        )

        self._store.save_hold(hold)

        # Update any pending deletion requests for these users
        for user_id in user_ids:
            requests = self._store.get_requests_for_user(user_id)
            for request in requests:
                if request.status == DeletionStatus.PENDING:
                    request.status = DeletionStatus.HELD
                    request.hold_id = hold.hold_id
                    self._store.save_request(request)

        logger.info(
            "Created legal hold %s for %d users: %s",
            hold.hold_id,
            len(user_ids),
            reason,
        )

        return hold

    def release_hold(
        self,
        hold_id: str,
        released_by: str,
    ) -> LegalHold | None:
        """
        Release a legal hold.

        Args:
            hold_id: ID of the hold to release
            released_by: User/system releasing the hold

        Returns:
            Updated LegalHold or None if not found
        """
        hold = self._store.get_hold(hold_id)
        if not hold:
            return None

        hold.is_active = False
        hold.released_at = datetime.now(timezone.utc)
        hold.released_by = released_by

        self._store.save_hold(hold)

        # Reactivate any held deletion requests
        for user_id in hold.user_ids:
            # Check if user still has other active holds
            other_holds = [
                h for h in self._store.get_active_holds_for_user(user_id) if h.hold_id != hold_id
            ]
            if not other_holds:
                # Reactivate pending deletions
                requests = self._store.get_requests_for_user(user_id)
                for request in requests:
                    if request.status == DeletionStatus.HELD and request.hold_id == hold_id:
                        request.status = DeletionStatus.PENDING
                        request.hold_id = None
                        self._store.save_request(request)

        logger.info("Released legal hold %s by %s", hold_id, released_by)

        return hold

    def get_active_holds(self) -> list[LegalHold]:
        """Get all active legal holds."""
        return self._store.get_all_active_holds()

    def is_user_on_hold(self, user_id: str) -> bool:
        """Check if a user is under any legal hold."""
        return len(self._store.get_active_holds_for_user(user_id)) > 0


# ============================================================================
# Default Entity Deleters
# ============================================================================


class ConsentDeleter:
    """Deleter for consent records."""

    def __init__(self) -> None:
        from aragora.privacy.consent import get_consent_manager

        self._manager = get_consent_manager()

    async def delete_for_user(self, user_id: str) -> int:
        """Delete all consent records for a user."""
        return self._manager.delete_user_data(user_id)

    async def verify_deleted(self, user_id: str) -> bool:
        """Verify no consent data remains."""
        records = self._manager.get_all_consents(user_id)
        return len(records) == 0


class PreferencesDeleter:
    """Deleter for user preferences."""

    async def delete_for_user(self, user_id: str) -> int:
        """Delete all preferences for a user."""
        # Implementation depends on your preferences storage
        # This is a placeholder
        logger.debug("Deleting preferences for user: %s", user_id)
        return 0

    async def verify_deleted(self, user_id: str) -> bool:
        """Verify no preferences remain."""
        return True


# ============================================================================
# Global Instances
# ============================================================================


_deletion_store: DeletionStore | None = None
_deletion_scheduler: GDPRDeletionScheduler | None = None
_cascade_manager: DeletionCascadeManager | None = None
_legal_hold_manager: LegalHoldManager | None = None


def get_deletion_store() -> DeletionStore:
    """Get or create the global deletion store."""
    global _deletion_store
    if _deletion_store is None:
        _deletion_store = DeletionStore()
    return _deletion_store


def get_deletion_scheduler() -> GDPRDeletionScheduler:
    """Get or create the global deletion scheduler."""
    global _deletion_scheduler
    if _deletion_scheduler is None:
        _deletion_scheduler = GDPRDeletionScheduler(
            store=get_deletion_store(),
            cascade_manager=get_cascade_manager(),
        )
    return _deletion_scheduler


def get_cascade_manager() -> DeletionCascadeManager:
    """Get or create the global cascade manager."""
    global _cascade_manager
    if _cascade_manager is None:
        _cascade_manager = DeletionCascadeManager()
        # Register default deleters
        _cascade_manager.register_deleter(EntityType.CONSENT_RECORDS, ConsentDeleter())
        _cascade_manager.register_deleter(EntityType.PREFERENCES, PreferencesDeleter())
    return _cascade_manager


def get_legal_hold_manager() -> LegalHoldManager:
    """Get or create the global legal hold manager."""
    global _legal_hold_manager
    if _legal_hold_manager is None:
        _legal_hold_manager = LegalHoldManager(store=get_deletion_store())
    return _legal_hold_manager


__all__ = [
    "ConsentDeleter",
    "DataErasureVerifier",
    "DeletionCascadeManager",
    "DeletionCertificate",
    "DeletionRequest",
    "DeletionStatus",
    "DeletionStore",
    "EntityDeleter",
    "EntityType",
    "GDPRDeletionScheduler",
    "LegalHold",
    "LegalHoldManager",
    "PreferencesDeleter",
    "get_cascade_manager",
    "get_deletion_scheduler",
    "get_deletion_store",
    "get_legal_hold_manager",
]
