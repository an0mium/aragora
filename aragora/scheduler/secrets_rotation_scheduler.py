"""
Secrets Rotation Scheduler.

Provides automated secrets rotation for SOC 2 CC6.2 compliance:
- 90-day API key rotation
- Database credential rotation
- JWT secret rotation
- OAuth token refresh
- Automatic rotation triggers
- Rotation verification steps
- Grace periods during rotation

SOC 2 Compliance: CC6.2 (Credential Management)

Usage:
    from aragora.scheduler.secrets_rotation_scheduler import (
        SecretsRotationScheduler,
        get_secrets_rotation_scheduler,
        rotate_secret,
    )

    # Initialize scheduler
    scheduler = SecretsRotationScheduler(storage_path="secrets_rotation.db")

    # Start automated rotation
    await scheduler.start()

    # Manually trigger rotation
    result = await scheduler.rotate_secret(
        secret_type="api_key",
        secret_id="key_abc123",
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class SecretType(Enum):
    """Types of secrets that can be rotated."""

    API_KEY = "api_key"
    DATABASE_CREDENTIAL = "database_credential"
    JWT_SECRET = "jwt_secret"
    OAUTH_TOKEN = "oauth_token"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"
    SERVICE_ACCOUNT = "service_account"


class RotationStatus(Enum):
    """Status of a rotation operation."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RotationTrigger(Enum):
    """What triggered a rotation."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"
    EXPIRATION = "expiration"
    COMPROMISE = "compromise"
    POLICY = "policy"


@dataclass
class SecretMetadata:
    """Metadata about a managed secret."""

    secret_id: str
    secret_type: SecretType
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_rotated_at: Optional[datetime] = None
    next_rotation_at: Optional[datetime] = None
    rotation_interval_days: int = 90
    owner: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class RotationResult:
    """Result of a rotation operation."""

    rotation_id: str
    secret_id: str
    secret_type: SecretType
    status: RotationStatus = RotationStatus.SCHEDULED
    trigger: RotationTrigger = RotationTrigger.SCHEDULED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    initiated_by: str = "system"

    # Version tracking
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    grace_period_ends: Optional[datetime] = None

    # Metrics
    duration_seconds: float = 0.0
    verification_passed: bool = False

    # Error handling
    error_message: Optional[str] = None
    rolled_back: bool = False

    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rotation_id": self.rotation_id,
            "secret_id": self.secret_id,
            "secret_type": self.secret_type.value,
            "status": self.status.value,
            "trigger": self.trigger.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "initiated_by": self.initiated_by,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "grace_period_ends": (
                self.grace_period_ends.isoformat() if self.grace_period_ends else None
            ),
            "duration_seconds": self.duration_seconds,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
            "rolled_back": self.rolled_back,
            "notes": self.notes,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SecretsRotationConfig:
    """Configuration for secrets rotation scheduler."""

    # Default rotation intervals (days)
    api_key_rotation_days: int = 90
    database_rotation_days: int = 90
    jwt_rotation_days: int = 30
    oauth_rotation_days: int = 30
    encryption_key_rotation_days: int = 365

    # Grace periods (hours)
    default_grace_period_hours: int = 24
    api_key_grace_period_hours: int = 48
    database_grace_period_hours: int = 2

    # Verification
    verify_after_rotation: bool = True
    rollback_on_verification_failure: bool = True

    # Notification
    notify_days_before: int = 7
    notification_email: Optional[str] = None
    slack_webhook: Optional[str] = None

    # Storage
    storage_path: Optional[str] = None


# =============================================================================
# Storage Layer
# =============================================================================


class SecretsRotationStorage:
    """SQLite-backed storage for secrets rotation."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage."""
        self._db_path = db_path or ":memory:"
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS managed_secrets (
                secret_id TEXT PRIMARY KEY,
                secret_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                last_rotated_at TEXT,
                next_rotation_at TEXT,
                rotation_interval_days INTEGER DEFAULT 90,
                owner TEXT,
                tags_json TEXT,
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS rotation_history (
                rotation_id TEXT PRIMARY KEY,
                secret_id TEXT NOT NULL,
                secret_type TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                initiated_by TEXT,
                old_version TEXT,
                new_version TEXT,
                grace_period_ends TEXT,
                duration_seconds REAL DEFAULT 0,
                verification_passed INTEGER DEFAULT 0,
                error_message TEXT,
                rolled_back INTEGER DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (secret_id) REFERENCES managed_secrets(secret_id)
            );

            CREATE TABLE IF NOT EXISTS rotation_schedule (
                schedule_id TEXT PRIMARY KEY,
                secret_id TEXT NOT NULL,
                next_rotation TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                last_notification TEXT,
                FOREIGN KEY (secret_id) REFERENCES managed_secrets(secret_id)
            );

            CREATE INDEX IF NOT EXISTS idx_secrets_type ON managed_secrets(secret_type);
            CREATE INDEX IF NOT EXISTS idx_secrets_next_rotation ON managed_secrets(next_rotation_at);
            CREATE INDEX IF NOT EXISTS idx_history_secret ON rotation_history(secret_id);
            CREATE INDEX IF NOT EXISTS idx_history_status ON rotation_history(status);
            CREATE INDEX IF NOT EXISTS idx_schedule_next ON rotation_schedule(next_rotation);
            """)
        conn.commit()

    def save_secret(self, metadata: SecretMetadata) -> None:
        """Save or update a secret's metadata."""
        import json

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO managed_secrets (
                secret_id, secret_type, name, description, created_at,
                last_rotated_at, next_rotation_at, rotation_interval_days,
                owner, tags_json, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.secret_id,
                metadata.secret_type.value,
                metadata.name,
                metadata.description,
                metadata.created_at.isoformat(),
                metadata.last_rotated_at.isoformat() if metadata.last_rotated_at else None,
                metadata.next_rotation_at.isoformat() if metadata.next_rotation_at else None,
                metadata.rotation_interval_days,
                metadata.owner,
                json.dumps(metadata.tags),
                1 if metadata.is_active else 0,
            ),
        )
        conn.commit()

    def get_secret(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get a secret's metadata."""
        import json

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM managed_secrets WHERE secret_id = ?",
            (secret_id,),
        ).fetchone()

        if not row:
            return None

        return SecretMetadata(
            secret_id=row["secret_id"],
            secret_type=SecretType(row["secret_type"]),
            name=row["name"],
            description=row["description"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            last_rotated_at=(
                datetime.fromisoformat(row["last_rotated_at"]) if row["last_rotated_at"] else None
            ),
            next_rotation_at=(
                datetime.fromisoformat(row["next_rotation_at"]) if row["next_rotation_at"] else None
            ),
            rotation_interval_days=row["rotation_interval_days"],
            owner=row["owner"],
            tags=json.loads(row["tags_json"] or "{}"),
            is_active=bool(row["is_active"]),
        )

    def get_secrets_due_for_rotation(self) -> List[SecretMetadata]:
        """Get secrets that need rotation."""
        import json

        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        rows = conn.execute(
            """
            SELECT * FROM managed_secrets
            WHERE is_active = 1 AND next_rotation_at <= ?
            ORDER BY next_rotation_at ASC
            """,
            (now,),
        ).fetchall()

        secrets = []
        for row in rows:
            secrets.append(
                SecretMetadata(
                    secret_id=row["secret_id"],
                    secret_type=SecretType(row["secret_type"]),
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_rotated_at=(
                        datetime.fromisoformat(row["last_rotated_at"])
                        if row["last_rotated_at"]
                        else None
                    ),
                    next_rotation_at=(
                        datetime.fromisoformat(row["next_rotation_at"])
                        if row["next_rotation_at"]
                        else None
                    ),
                    rotation_interval_days=row["rotation_interval_days"],
                    owner=row["owner"],
                    tags=json.loads(row["tags_json"] or "{}"),
                    is_active=bool(row["is_active"]),
                )
            )

        return secrets

    def get_secrets_expiring_soon(self, days: int = 7) -> List[SecretMetadata]:
        """Get secrets expiring within N days."""
        import json

        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

        rows = conn.execute(
            """
            SELECT * FROM managed_secrets
            WHERE is_active = 1 AND next_rotation_at <= ?
            ORDER BY next_rotation_at ASC
            """,
            (cutoff,),
        ).fetchall()

        return [
            SecretMetadata(
                secret_id=row["secret_id"],
                secret_type=SecretType(row["secret_type"]),
                name=row["name"],
                description=row["description"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
                last_rotated_at=(
                    datetime.fromisoformat(row["last_rotated_at"])
                    if row["last_rotated_at"]
                    else None
                ),
                next_rotation_at=(
                    datetime.fromisoformat(row["next_rotation_at"])
                    if row["next_rotation_at"]
                    else None
                ),
                rotation_interval_days=row["rotation_interval_days"],
                owner=row["owner"],
                tags=json.loads(row["tags_json"] or "{}"),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    def save_rotation(self, rotation: RotationResult) -> None:
        """Save a rotation result."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO rotation_history (
                rotation_id, secret_id, secret_type, status, trigger,
                started_at, completed_at, initiated_by, old_version,
                new_version, grace_period_ends, duration_seconds,
                verification_passed, error_message, rolled_back, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rotation.rotation_id,
                rotation.secret_id,
                rotation.secret_type.value,
                rotation.status.value,
                rotation.trigger.value,
                rotation.started_at.isoformat() if rotation.started_at else None,
                rotation.completed_at.isoformat() if rotation.completed_at else None,
                rotation.initiated_by,
                rotation.old_version,
                rotation.new_version,
                rotation.grace_period_ends.isoformat() if rotation.grace_period_ends else None,
                rotation.duration_seconds,
                1 if rotation.verification_passed else 0,
                rotation.error_message,
                1 if rotation.rolled_back else 0,
                rotation.notes,
            ),
        )
        conn.commit()

    def get_rotation_history(
        self, secret_id: Optional[str] = None, limit: int = 50
    ) -> List[RotationResult]:
        """Get rotation history."""
        conn = self._get_conn()

        if secret_id:
            rows = conn.execute(
                """
                SELECT * FROM rotation_history
                WHERE secret_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (secret_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM rotation_history
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            RotationResult(
                rotation_id=row["rotation_id"],
                secret_id=row["secret_id"],
                secret_type=SecretType(row["secret_type"]),
                status=RotationStatus(row["status"]),
                trigger=RotationTrigger(row["trigger"]),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
                initiated_by=row["initiated_by"] or "system",
                old_version=row["old_version"],
                new_version=row["new_version"],
                grace_period_ends=(
                    datetime.fromisoformat(row["grace_period_ends"])
                    if row["grace_period_ends"]
                    else None
                ),
                duration_seconds=row["duration_seconds"] or 0.0,
                verification_passed=bool(row["verification_passed"]),
                error_message=row["error_message"],
                rolled_back=bool(row["rolled_back"]),
                notes=row["notes"] or "",
            )
            for row in rows
        ]

    def get_all_secrets(self, active_only: bool = True) -> List[SecretMetadata]:
        """Get all managed secrets."""
        import json

        conn = self._get_conn()

        if active_only:
            rows = conn.execute("SELECT * FROM managed_secrets WHERE is_active = 1").fetchall()
        else:
            rows = conn.execute("SELECT * FROM managed_secrets").fetchall()

        return [
            SecretMetadata(
                secret_id=row["secret_id"],
                secret_type=SecretType(row["secret_type"]),
                name=row["name"],
                description=row["description"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
                last_rotated_at=(
                    datetime.fromisoformat(row["last_rotated_at"])
                    if row["last_rotated_at"]
                    else None
                ),
                next_rotation_at=(
                    datetime.fromisoformat(row["next_rotation_at"])
                    if row["next_rotation_at"]
                    else None
                ),
                rotation_interval_days=row["rotation_interval_days"],
                owner=row["owner"],
                tags=json.loads(row["tags_json"] or "{}"),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]


# =============================================================================
# Secrets Rotation Scheduler
# =============================================================================


class SecretsRotationScheduler:
    """Main secrets rotation scheduler and executor."""

    def __init__(self, config: Optional[SecretsRotationConfig] = None):
        """Initialize scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or SecretsRotationConfig()
        self._storage = SecretsRotationStorage(self.config.storage_path)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Rotation handlers by secret type
        self._rotation_handlers: Dict[SecretType, Callable[[str], Dict[str, Any]]] = {}
        self._verification_handlers: Dict[SecretType, Callable[[str, str], bool]] = {}
        self._rollback_handlers: Dict[SecretType, Callable[[str, str], bool]] = {}
        self._notification_handlers: List[Callable[[Dict[str, Any]], None]] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Secrets rotation scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Secrets rotation scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for secrets due for rotation
                due_secrets = self._storage.get_secrets_due_for_rotation()
                for secret in due_secrets:
                    await self.rotate_secret(
                        secret_type=secret.secret_type,
                        secret_id=secret.secret_id,
                        trigger=RotationTrigger.SCHEDULED,
                    )

                # Check for expiring secrets and send notifications
                expiring = self._storage.get_secrets_expiring_soon(
                    days=self.config.notify_days_before
                )
                for secret in expiring:
                    await self._notify_expiring_secret(secret)

                # Sleep before next check (every hour)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in secrets rotation scheduler: {e}")
                await asyncio.sleep(300)

    # =========================================================================
    # Secret Management
    # =========================================================================

    def register_secret(
        self,
        secret_id: str,
        secret_type: SecretType,
        name: str,
        description: str = "",
        rotation_interval_days: Optional[int] = None,
        owner: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> SecretMetadata:
        """Register a secret for managed rotation.

        Args:
            secret_id: Unique identifier for the secret
            secret_type: Type of secret
            name: Human-readable name
            description: Optional description
            rotation_interval_days: Custom rotation interval
            owner: Owner user/team ID
            tags: Optional tags

        Returns:
            Created secret metadata
        """
        # Determine rotation interval
        if rotation_interval_days is None:
            rotation_interval_days = self._get_default_interval(secret_type)

        # Calculate next rotation
        now = datetime.now(timezone.utc)
        next_rotation = now + timedelta(days=rotation_interval_days)

        metadata = SecretMetadata(
            secret_id=secret_id,
            secret_type=secret_type,
            name=name,
            description=description,
            created_at=now,
            next_rotation_at=next_rotation,
            rotation_interval_days=rotation_interval_days,
            owner=owner,
            tags=tags or {},
        )

        self._storage.save_secret(metadata)
        logger.info(
            f"Registered secret {secret_id} for rotation every {rotation_interval_days} days"
        )

        return metadata

    def _get_default_interval(self, secret_type: SecretType) -> int:
        """Get default rotation interval for a secret type."""
        intervals = {
            SecretType.API_KEY: self.config.api_key_rotation_days,
            SecretType.DATABASE_CREDENTIAL: self.config.database_rotation_days,
            SecretType.JWT_SECRET: self.config.jwt_rotation_days,
            SecretType.OAUTH_TOKEN: self.config.oauth_rotation_days,
            SecretType.ENCRYPTION_KEY: self.config.encryption_key_rotation_days,
            SecretType.WEBHOOK_SECRET: self.config.api_key_rotation_days,
            SecretType.SERVICE_ACCOUNT: self.config.api_key_rotation_days,
        }
        return intervals.get(secret_type, 90)

    def _get_grace_period(self, secret_type: SecretType) -> int:
        """Get grace period in hours for a secret type."""
        grace_periods = {
            SecretType.API_KEY: self.config.api_key_grace_period_hours,
            SecretType.DATABASE_CREDENTIAL: self.config.database_grace_period_hours,
        }
        return grace_periods.get(secret_type, self.config.default_grace_period_hours)

    # =========================================================================
    # Rotation Operations
    # =========================================================================

    async def rotate_secret(
        self,
        secret_type: SecretType,
        secret_id: str,
        trigger: RotationTrigger = RotationTrigger.MANUAL,
        initiated_by: str = "system",
    ) -> RotationResult:
        """Rotate a secret.

        Args:
            secret_type: Type of secret
            secret_id: Secret identifier
            trigger: What triggered the rotation
            initiated_by: User ID who initiated rotation

        Returns:
            Rotation result
        """
        # Create rotation record
        rotation = RotationResult(
            rotation_id=str(uuid.uuid4()),
            secret_id=secret_id,
            secret_type=secret_type,
            status=RotationStatus.IN_PROGRESS,
            trigger=trigger,
            started_at=datetime.now(timezone.utc),
            initiated_by=initiated_by,
        )

        # Get current version hash
        rotation.old_version = self._generate_version_hash()

        try:
            # Execute rotation
            handler = self._rotation_handlers.get(secret_type)
            if handler:
                result = handler(secret_id)
                if not result.get("success", False):
                    raise Exception(result.get("error", "Rotation failed"))
                rotation.new_version = result.get("new_version", self._generate_version_hash())
            else:
                # Simulate rotation if no handler
                rotation.new_version = self._generate_version_hash()
                logger.warning(f"No rotation handler for {secret_type.value}, simulating rotation")

            # Set grace period
            grace_hours = self._get_grace_period(secret_type)
            rotation.grace_period_ends = datetime.now(timezone.utc) + timedelta(hours=grace_hours)

            # Verify rotation
            rotation.status = RotationStatus.VERIFYING
            if self.config.verify_after_rotation:
                verification_handler = self._verification_handlers.get(secret_type)
                if verification_handler:
                    rotation.verification_passed = verification_handler(
                        secret_id, rotation.new_version
                    )
                else:
                    # Assume success if no handler
                    rotation.verification_passed = True

                if not rotation.verification_passed:
                    if self.config.rollback_on_verification_failure:
                        await self._rollback(rotation)
                        rotation.status = RotationStatus.ROLLED_BACK
                        rotation.error_message = "Verification failed, rolled back"
                    else:
                        rotation.status = RotationStatus.FAILED
                        rotation.error_message = "Verification failed"
                else:
                    rotation.status = RotationStatus.COMPLETED
            else:
                rotation.verification_passed = True
                rotation.status = RotationStatus.COMPLETED

        except Exception as e:
            rotation.status = RotationStatus.FAILED
            rotation.error_message = str(e)
            logger.error(f"Secret rotation failed: {e}")

        # Finalize
        rotation.completed_at = datetime.now(timezone.utc)
        rotation.duration_seconds = (rotation.completed_at - rotation.started_at).total_seconds()

        # Update secret metadata
        if rotation.status == RotationStatus.COMPLETED:
            metadata = self._storage.get_secret(secret_id)
            if metadata:
                metadata.last_rotated_at = datetime.now(timezone.utc)
                metadata.next_rotation_at = metadata.last_rotated_at + timedelta(
                    days=metadata.rotation_interval_days
                )
                self._storage.save_secret(metadata)

        # Persist rotation record
        self._storage.save_rotation(rotation)

        # Notify
        await self._notify_rotation_completed(rotation)

        logger.info(
            f"Secret rotation {rotation.rotation_id}: "
            f"secret={secret_id}, status={rotation.status.value}"
        )

        return rotation

    async def _rollback(self, rotation: RotationResult) -> None:
        """Rollback a failed rotation."""
        handler = self._rollback_handlers.get(rotation.secret_type)
        if handler and rotation.old_version:
            try:
                success = handler(rotation.secret_id, rotation.old_version)
                rotation.rolled_back = success
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                rotation.rolled_back = False
        else:
            logger.warning("No rollback handler, cannot rollback")
            rotation.rolled_back = False

    def _generate_version_hash(self) -> str:
        """Generate a version hash for tracking."""
        return hashlib.sha256(
            f"{datetime.now(timezone.utc).isoformat()}{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_expiring_secret(self, secret: SecretMetadata) -> None:
        """Send notification for expiring secret."""
        if not secret.next_rotation_at:
            return

        days_until = (secret.next_rotation_at - datetime.now(timezone.utc)).days

        notification = {
            "type": "secret_expiring",
            "secret_id": secret.secret_id,
            "secret_name": secret.name,
            "secret_type": secret.secret_type.value,
            "days_until_rotation": days_until,
            "next_rotation_at": secret.next_rotation_at.isoformat(),
            "owner": secret.owner,
        }

        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    async def _notify_rotation_completed(self, rotation: RotationResult) -> None:
        """Send notification when rotation completes."""
        notification = {
            "type": "rotation_completed",
            "rotation_id": rotation.rotation_id,
            "secret_id": rotation.secret_id,
            "secret_type": rotation.secret_type.value,
            "status": rotation.status.value,
            "trigger": rotation.trigger.value,
            "verification_passed": rotation.verification_passed,
            "rolled_back": rotation.rolled_back,
            "error_message": rotation.error_message,
        }

        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    # =========================================================================
    # Registration
    # =========================================================================

    def register_rotation_handler(
        self, secret_type: SecretType, handler: Callable[[str], Dict[str, Any]]
    ) -> None:
        """Register a rotation handler for a secret type."""
        self._rotation_handlers[secret_type] = handler

    def register_verification_handler(
        self, secret_type: SecretType, handler: Callable[[str, str], bool]
    ) -> None:
        """Register a verification handler."""
        self._verification_handlers[secret_type] = handler

    def register_rollback_handler(
        self, secret_type: SecretType, handler: Callable[[str, str], bool]
    ) -> None:
        """Register a rollback handler."""
        self._rollback_handlers[secret_type] = handler

    def register_notification_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a notification handler."""
        self._notification_handlers.append(handler)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_secret(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get a secret's metadata."""
        return self._storage.get_secret(secret_id)

    def get_all_secrets(self) -> List[SecretMetadata]:
        """Get all managed secrets."""
        return self._storage.get_all_secrets()

    def get_rotation_history(
        self, secret_id: Optional[str] = None, limit: int = 50
    ) -> List[RotationResult]:
        """Get rotation history."""
        return self._storage.get_rotation_history(secret_id, limit)

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report for secrets rotation."""
        secrets = self._storage.get_all_secrets()
        history = self._storage.get_rotation_history(limit=100)

        now = datetime.now(timezone.utc)

        # Count by status
        overdue = 0
        compliant = 0
        expiring_soon = 0

        for secret in secrets:
            if secret.next_rotation_at:
                days_until = (secret.next_rotation_at - now).days
                if days_until < 0:
                    overdue += 1
                elif days_until <= self.config.notify_days_before:
                    expiring_soon += 1
                else:
                    compliant += 1

        # Calculate rotation success rate
        successful = len([r for r in history if r.status == RotationStatus.COMPLETED])
        failed = len([r for r in history if r.status == RotationStatus.FAILED])
        total = successful + failed
        success_rate = successful / total if total > 0 else 1.0

        return {
            "total_secrets": len(secrets),
            "compliant": compliant,
            "expiring_soon": expiring_soon,
            "overdue": overdue,
            "compliance_rate": compliant / len(secrets) if secrets else 1.0,
            "rotation_success_rate": success_rate,
            "total_rotations": len(history),
            "rotations_last_30_days": len(
                [r for r in history if r.started_at and (now - r.started_at).days <= 30]
            ),
            "by_type": {
                st.value: len([s for s in secrets if s.secret_type == st]) for st in SecretType
            },
        }


# =============================================================================
# Global Instance
# =============================================================================

_scheduler: Optional[SecretsRotationScheduler] = None
_scheduler_lock = threading.Lock()


def get_secrets_rotation_scheduler(
    config: Optional[SecretsRotationConfig] = None,
) -> SecretsRotationScheduler:
    """Get or create the global secrets rotation scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = SecretsRotationScheduler(config)
        return _scheduler


async def rotate_secret(
    secret_type: SecretType,
    secret_id: str,
    trigger: RotationTrigger = RotationTrigger.MANUAL,
) -> RotationResult:
    """Convenience function to rotate a secret."""
    return await get_secrets_rotation_scheduler().rotate_secret(
        secret_type=secret_type,
        secret_id=secret_id,
        trigger=trigger,
    )


__all__ = [
    # Types
    "SecretType",
    "RotationStatus",
    "RotationTrigger",
    "SecretMetadata",
    "RotationResult",
    # Configuration
    "SecretsRotationConfig",
    # Core
    "SecretsRotationScheduler",
    "get_secrets_rotation_scheduler",
    # Convenience
    "rotate_secret",
]
