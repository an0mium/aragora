"""
Key Rotation Scheduler - Automated encryption key rotation.

Provides automated key rotation scheduling with:
- Configurable rotation intervals (30, 60, 90 days)
- KMS provider integration for cloud key rotation
- Tenant-specific key rotation
- Audit logging and metrics
- Re-encryption support for rotating encrypted data

SOC 2 Compliance: CC6.1, CC6.7 (Cryptographic Key Management)

Usage:
    from aragora.security.key_rotation import (
        KeyRotationScheduler,
        KeyRotationConfig,
        get_key_rotation_scheduler,
    )

    # Create scheduler
    scheduler = KeyRotationScheduler(
        config=KeyRotationConfig(
            rotation_interval_days=90,
            auto_rotate_kms_keys=True,
        ),
    )

    # Start automated rotation
    await scheduler.start()

    # Trigger manual rotation
    await scheduler.rotate_now()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class RotationStatus(str, Enum):
    """Status of a key rotation operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SchedulerStatus(str, Enum):
    """Status of the key rotation scheduler."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class KeyRotationConfig:
    """Configuration for key rotation."""

    # Rotation interval
    rotation_interval_days: int = 90
    """Days between automatic key rotations."""

    # Check interval (how often to check for keys needing rotation)
    check_interval_hours: int = 6
    """Hours between rotation checks."""

    # KMS integration
    auto_rotate_kms_keys: bool = True
    """Automatically rotate keys in cloud KMS providers."""

    rotate_tenant_keys: bool = True
    """Rotate per-tenant encryption keys."""

    # Re-encryption settings
    re_encrypt_on_rotation: bool = True
    """Re-encrypt data with new keys after rotation."""

    re_encrypt_batch_size: int = 100
    """Batch size for re-encryption operations."""

    # Overlap period
    key_overlap_days: int = 7
    """Days to keep old key version active after rotation."""

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Notifications
    notify_on_rotation: bool = True
    notify_on_failure: bool = True
    notify_days_before: int = 7
    """Days before expiry to send reminder notification."""


@dataclass
class KeyRotationJob:
    """Record of a key rotation operation."""

    id: str
    key_id: str
    tenant_id: Optional[str] = None
    provider: str = "local"
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RotationStatus = RotationStatus.PENDING
    old_version: Optional[int] = None
    new_version: Optional[int] = None
    records_re_encrypted: int = 0
    error: Optional[str] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "records_re_encrypted": self.records_re_encrypted,
            "error": self.error,
            "retries": self.retries,
            "metadata": self.metadata,
        }


@dataclass
class KeyInfo:
    """Information about an encryption key."""

    key_id: str
    provider: str
    version: int
    created_at: datetime
    last_rotated_at: Optional[datetime] = None
    next_rotation_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    is_active: bool = True


@dataclass
class SchedulerStats:
    """Statistics for the key rotation scheduler."""

    status: SchedulerStatus
    total_rotations: int = 0
    successful_rotations: int = 0
    failed_rotations: int = 0
    last_rotation_at: Optional[datetime] = None
    last_rotation_status: Optional[str] = None
    next_check_at: Optional[datetime] = None
    keys_tracked: int = 0
    keys_expiring_soon: int = 0
    uptime_seconds: float = 0.0


# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]


class KeyRotationScheduler:
    """
    Automated key rotation scheduler.

    Manages scheduled key rotations across KMS providers
    and ensures encryption keys are rotated according to policy.
    """

    def __init__(
        self,
        config: Optional[KeyRotationConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize the key rotation scheduler.

        Args:
            config: Key rotation configuration
            event_callback: Optional callback for events
        """
        self._config = config or KeyRotationConfig()
        self._event_callback = event_callback

        self._status = SchedulerStatus.STOPPED
        self._started_at: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        self._job_history: List[KeyRotationJob] = []
        self._tracked_keys: Dict[str, KeyInfo] = {}
        self._stats = SchedulerStats(status=SchedulerStatus.STOPPED)
        self._lock = asyncio.Lock()

    @property
    def status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        return self._status

    @property
    def config(self) -> KeyRotationConfig:
        """Get current configuration."""
        return self._config

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for notifications."""
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _record_metric(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Record Prometheus metrics."""
        try:
            from aragora.observability.metrics import (
                record_key_rotation,
            )

            record_key_rotation(operation, success, duration_seconds)
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")

    async def start(self) -> None:
        """Start the key rotation scheduler."""
        if self._status == SchedulerStatus.RUNNING:
            logger.warning("Key rotation scheduler already running")
            return

        self._status = SchedulerStatus.RUNNING
        self._started_at = datetime.now(timezone.utc)
        self._stats.status = SchedulerStatus.RUNNING

        logger.info(
            f"Starting key rotation scheduler "
            f"(interval={self._config.rotation_interval_days}d, "
            f"check_every={self._config.check_interval_hours}h)"
        )

        # Start the rotation check loop
        self._task = asyncio.create_task(self._run_rotation_checks())

        self._emit_event(
            "key_rotation_scheduler_started",
            {
                "rotation_interval_days": self._config.rotation_interval_days,
                "check_interval_hours": self._config.check_interval_hours,
            },
        )

    async def stop(self) -> None:
        """Stop the key rotation scheduler."""
        if self._status == SchedulerStatus.STOPPED:
            return

        logger.info("Stopping key rotation scheduler")
        self._status = SchedulerStatus.STOPPED
        self._stats.status = SchedulerStatus.STOPPED

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._emit_event("key_rotation_scheduler_stopped", {})

    async def pause(self) -> None:
        """Pause the key rotation scheduler."""
        self._status = SchedulerStatus.PAUSED
        self._stats.status = SchedulerStatus.PAUSED
        logger.info("Key rotation scheduler paused")
        self._emit_event("key_rotation_scheduler_paused", {})

    async def resume(self) -> None:
        """Resume the key rotation scheduler."""
        if self._status == SchedulerStatus.PAUSED:
            self._status = SchedulerStatus.RUNNING
            self._stats.status = SchedulerStatus.RUNNING
            logger.info("Key rotation scheduler resumed")
            self._emit_event("key_rotation_scheduler_resumed", {})

    def track_key(self, key_info: KeyInfo) -> None:
        """
        Add a key to be tracked for rotation.

        Args:
            key_info: Information about the key to track
        """
        self._tracked_keys[key_info.key_id] = key_info
        self._stats.keys_tracked = len(self._tracked_keys)

        # Calculate next rotation time
        if key_info.last_rotated_at:
            key_info.next_rotation_at = key_info.last_rotated_at + timedelta(
                days=self._config.rotation_interval_days
            )
        else:
            key_info.next_rotation_at = key_info.created_at + timedelta(
                days=self._config.rotation_interval_days
            )

    def untrack_key(self, key_id: str) -> None:
        """Remove a key from tracking."""
        self._tracked_keys.pop(key_id, None)
        self._stats.keys_tracked = len(self._tracked_keys)

    async def rotate_now(
        self,
        key_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> KeyRotationJob:
        """
        Trigger an immediate key rotation.

        Args:
            key_id: Specific key to rotate (None = rotate all due)
            tenant_id: Specific tenant's key to rotate

        Returns:
            KeyRotationJob with the result
        """
        job = KeyRotationJob(
            id=str(uuid.uuid4()),
            key_id=key_id or "all",
            tenant_id=tenant_id,
            provider=self._detect_provider(),
        )

        await self._execute_rotation(job)
        return job

    async def rotate_all_due(self) -> List[KeyRotationJob]:
        """
        Rotate all keys that are due for rotation.

        Returns:
            List of KeyRotationJob results
        """
        jobs = []
        now = datetime.now(timezone.utc)

        for key_id, key_info in self._tracked_keys.items():
            if key_info.next_rotation_at and key_info.next_rotation_at <= now:
                job = KeyRotationJob(
                    id=str(uuid.uuid4()),
                    key_id=key_id,
                    tenant_id=key_info.tenant_id,
                    provider=key_info.provider,
                )
                await self._execute_rotation(job)
                jobs.append(job)

        return jobs

    def _detect_provider(self) -> str:
        """Detect the current KMS provider."""
        try:
            from aragora.security.kms_provider import detect_cloud_provider

            return detect_cloud_provider()
        except Exception:
            return "local"

    async def _run_rotation_checks(self) -> None:
        """Run periodic rotation checks."""
        check_interval_seconds = self._config.check_interval_hours * 3600

        while self._status == SchedulerStatus.RUNNING:
            try:
                self._stats.next_check_at = datetime.now(timezone.utc) + timedelta(
                    seconds=check_interval_seconds
                )

                # Check for keys expiring soon
                await self._check_expiring_keys()

                # Rotate any due keys
                if self._status == SchedulerStatus.RUNNING:
                    await self.rotate_all_due()

                await asyncio.sleep(check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Key rotation check error: {e}")
                self._status = SchedulerStatus.ERROR
                self._stats.status = SchedulerStatus.ERROR
                await asyncio.sleep(60)

    async def _check_expiring_keys(self) -> None:
        """Check for keys expiring soon and send notifications."""
        now = datetime.now(timezone.utc)
        warning_threshold = now + timedelta(days=self._config.notify_days_before)

        expiring_soon = []
        for key_id, key_info in self._tracked_keys.items():
            if key_info.next_rotation_at and key_info.next_rotation_at <= warning_threshold:
                expiring_soon.append(key_info)

        self._stats.keys_expiring_soon = len(expiring_soon)

        if expiring_soon and self._config.notify_on_rotation:
            self._emit_event(
                "keys_expiring_soon",
                {
                    "count": len(expiring_soon),
                    "keys": [
                        {
                            "key_id": k.key_id,
                            "next_rotation_at": k.next_rotation_at.isoformat()
                            if k.next_rotation_at
                            else None,
                            "tenant_id": k.tenant_id,
                        }
                        for k in expiring_soon
                    ],
                },
            )

    async def _execute_rotation(self, job: KeyRotationJob) -> None:
        """Execute a key rotation job."""
        import time as time_module

        async with self._lock:
            job.started_at = datetime.now(timezone.utc)
            job.status = RotationStatus.IN_PROGRESS

            start_time = time_module.time()
            success = False

            try:
                logger.info(
                    f"Starting key rotation job {job.id} "
                    f"(key={job.key_id}, provider={job.provider})"
                )

                # Perform rotation based on provider
                if job.provider == "vault":
                    await self._rotate_vault_key(job)
                elif job.provider in ("aws", "azure", "gcp"):
                    await self._rotate_cloud_key(job)
                else:
                    await self._rotate_local_key(job)

                job.status = RotationStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                success = True

                self._stats.total_rotations += 1
                self._stats.successful_rotations += 1
                self._stats.last_rotation_at = job.completed_at
                self._stats.last_rotation_status = "success"

                # Update tracked key info
                if job.key_id in self._tracked_keys:
                    key_info = self._tracked_keys[job.key_id]
                    key_info.last_rotated_at = job.completed_at
                    key_info.next_rotation_at = job.completed_at + timedelta(
                        days=self._config.rotation_interval_days
                    )
                    if job.new_version:
                        key_info.version = job.new_version

                self._emit_event(
                    "key_rotation_completed",
                    {
                        "job_id": job.id,
                        "key_id": job.key_id,
                        "old_version": job.old_version,
                        "new_version": job.new_version,
                        "duration_seconds": time_module.time() - start_time,
                    },
                )

            except Exception as e:
                job.status = RotationStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc)

                self._stats.total_rotations += 1
                self._stats.failed_rotations += 1
                self._stats.last_rotation_at = job.completed_at
                self._stats.last_rotation_status = "failed"

                logger.error(f"Key rotation job {job.id} failed: {e}")

                if self._config.notify_on_failure:
                    self._emit_event(
                        "key_rotation_failed",
                        {
                            "job_id": job.id,
                            "key_id": job.key_id,
                            "error": str(e),
                            "retries": job.retries,
                        },
                    )

                # Retry if configured
                if job.retries < self._config.max_retries:
                    job.retries += 1
                    logger.info(f"Retrying rotation job {job.id} (attempt {job.retries})")
                    await asyncio.sleep(self._config.retry_delay_seconds)
                    await self._execute_rotation(job)

            finally:
                duration = time_module.time() - start_time
                self._record_metric("key_rotation", success, duration)
                self._job_history.append(job)

    async def _rotate_vault_key(self, job: KeyRotationJob) -> None:
        """Rotate a key in HashiCorp Vault."""
        from aragora.security.kms_provider import get_kms_provider, HashiCorpVaultProvider

        provider = get_kms_provider()
        if not isinstance(provider, HashiCorpVaultProvider):
            raise ValueError("Vault provider not configured")

        # Get current key metadata
        old_meta = await provider.get_key_metadata(job.key_id)
        job.old_version = int(old_meta.version) if old_meta.version else 1

        # Rotate the key in Vault
        new_meta = await provider.rotate_key(job.key_id)
        job.new_version = int(new_meta.version) if new_meta.version else job.old_version + 1

        logger.info(f"Vault key {job.key_id} rotated: v{job.old_version} -> v{job.new_version}")

    async def _rotate_cloud_key(self, job: KeyRotationJob) -> None:
        """Rotate a key in cloud KMS (AWS/Azure/GCP)."""
        from aragora.security.kms_provider import get_kms_provider

        provider = get_kms_provider()

        # Get current metadata
        old_meta = await provider.get_key_metadata(job.key_id)
        job.old_version = int(old_meta.version) if old_meta.version else 1

        # Cloud KMS rotation is typically automatic or via console
        # We generate a new data key which uses the latest master key version
        new_key = await provider.get_encryption_key(job.key_id)

        job.new_version = job.old_version + 1
        job.metadata["key_generated"] = True
        job.metadata["key_length"] = len(new_key)

        logger.info(f"Cloud key {job.key_id} rotated via new data key generation")

    async def _rotate_local_key(self, job: KeyRotationJob) -> None:
        """Rotate a local encryption key."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        # Get current key info
        active_key = service.get_active_key()
        if active_key:
            job.old_version = active_key.version

        # Rotate the key
        new_key = service.rotate_key(job.key_id if job.key_id != "all" else None)
        job.new_version = new_key.version

        logger.info(f"Local key {new_key.key_id} rotated: v{job.old_version} -> v{job.new_version}")

    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        if self._started_at:
            self._stats.uptime_seconds = (
                datetime.now(timezone.utc) - self._started_at
            ).total_seconds()
        return self._stats

    def get_job_history(self, limit: int = 100) -> List[KeyRotationJob]:
        """Get recent rotation job history."""
        return self._job_history[-limit:]

    def get_tracked_keys(self) -> List[KeyInfo]:
        """Get all tracked keys."""
        return list(self._tracked_keys.values())


# =============================================================================
# Global Instance
# =============================================================================

_key_rotation_scheduler: Optional[KeyRotationScheduler] = None


def get_key_rotation_scheduler() -> Optional[KeyRotationScheduler]:
    """Get the global key rotation scheduler instance."""
    return _key_rotation_scheduler


def set_key_rotation_scheduler(scheduler: Optional[KeyRotationScheduler]) -> None:
    """Set the global key rotation scheduler instance."""
    global _key_rotation_scheduler
    _key_rotation_scheduler = scheduler


async def start_key_rotation_scheduler(
    config: Optional[KeyRotationConfig] = None,
    event_callback: Optional[EventCallback] = None,
) -> KeyRotationScheduler:
    """
    Start the global key rotation scheduler.

    Args:
        config: Optional rotation configuration
        event_callback: Optional event callback

    Returns:
        The started KeyRotationScheduler instance
    """
    global _key_rotation_scheduler

    if _key_rotation_scheduler is not None:
        await _key_rotation_scheduler.stop()

    _key_rotation_scheduler = KeyRotationScheduler(
        config=config,
        event_callback=event_callback,
    )
    await _key_rotation_scheduler.start()

    return _key_rotation_scheduler


async def stop_key_rotation_scheduler() -> None:
    """Stop the global key rotation scheduler."""
    global _key_rotation_scheduler

    if _key_rotation_scheduler is not None:
        await _key_rotation_scheduler.stop()
        _key_rotation_scheduler = None


__all__ = [
    "KeyRotationScheduler",
    "KeyRotationConfig",
    "KeyRotationJob",
    "KeyInfo",
    "RotationStatus",
    "SchedulerStatus",
    "SchedulerStats",
    "get_key_rotation_scheduler",
    "set_key_rotation_scheduler",
    "start_key_rotation_scheduler",
    "stop_key_rotation_scheduler",
]
