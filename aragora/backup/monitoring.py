"""
Backup Monitoring with Prometheus Metrics.

Provides monitoring for backup operations:
- Backup age (time since last backup)
- Backup size
- Restore time measurements
- RPO/RTO compliance status

Usage:
    from aragora.backup.monitoring import (
        record_backup_created,
        record_backup_restored,
        get_backup_age_seconds,
    )

    # Record a backup operation
    record_backup_created(size_bytes=1024*1024, duration_seconds=60)

    # Check backup freshness
    age = get_backup_age_seconds()
    if age > 3600:  # More than 1 hour
        alert("Backup is stale!")

Requirements:
    pip install prometheus-client

SLA Targets (from docs/SLA.md):
    Free:       RTO=24h, RPO=24h
    Pro:        RTO=4h,  RPO=1h
    Enterprise: RTO=1h,  RPO=15m
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Prometheus metrics - initialized lazily
_initialized = False

# Metric instances (will be set during initialization)
BACKUP_CREATED_TOTAL: Any = None
BACKUP_SIZE_BYTES: Any = None
BACKUP_DURATION_SECONDS: Any = None
BACKUP_LAST_TIMESTAMP: Any = None
BACKUP_AGE_SECONDS: Any = None
BACKUP_VERIFICATION_TOTAL: Any = None
BACKUP_VERIFICATION_FAILURES: Any = None
RESTORE_DURATION_SECONDS: Any = None
RESTORE_TOTAL: Any = None
RESTORE_FAILURES: Any = None
RPO_COMPLIANCE: Any = None
RTO_COMPLIANCE: Any = None

# SLA targets in seconds
SLA_TARGETS = {
    "free": {"rto": 24 * 3600, "rpo": 24 * 3600},
    "pro": {"rto": 4 * 3600, "rpo": 1 * 3600},
    "enterprise": {"rto": 1 * 3600, "rpo": 15 * 60},
}


def _init_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global BACKUP_CREATED_TOTAL, BACKUP_SIZE_BYTES, BACKUP_DURATION_SECONDS
    global BACKUP_LAST_TIMESTAMP, BACKUP_AGE_SECONDS
    global BACKUP_VERIFICATION_TOTAL, BACKUP_VERIFICATION_FAILURES
    global RESTORE_DURATION_SECONDS, RESTORE_TOTAL, RESTORE_FAILURES
    global RPO_COMPLIANCE, RTO_COMPLIANCE

    if _initialized:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Backup creation metrics
        BACKUP_CREATED_TOTAL = Counter(
            "aragora_backup_created_total",
            "Total number of backups created",
            ["backup_type"],
        )

        BACKUP_SIZE_BYTES = Gauge(
            "aragora_backup_size_bytes",
            "Size of the latest backup in bytes",
            ["backup_type"],
        )

        BACKUP_DURATION_SECONDS = Histogram(
            "aragora_backup_duration_seconds",
            "Time taken to create a backup",
            ["backup_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        BACKUP_LAST_TIMESTAMP = Gauge(
            "aragora_backup_last_timestamp",
            "Unix timestamp of the last successful backup",
        )

        BACKUP_AGE_SECONDS = Gauge(
            "aragora_backup_age_seconds",
            "Age of the latest backup in seconds",
        )

        # Verification metrics
        BACKUP_VERIFICATION_TOTAL = Counter(
            "aragora_backup_verification_total",
            "Total backup verifications performed",
            ["result"],
        )

        BACKUP_VERIFICATION_FAILURES = Counter(
            "aragora_backup_verification_failures_total",
            "Total backup verification failures",
            ["failure_type"],
        )

        # Restore metrics
        RESTORE_DURATION_SECONDS = Histogram(
            "aragora_restore_duration_seconds",
            "Time taken to restore from backup",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
        )

        RESTORE_TOTAL = Counter(
            "aragora_restore_total",
            "Total restore operations",
            ["result"],
        )

        RESTORE_FAILURES = Counter(
            "aragora_restore_failures_total",
            "Total restore failures",
            ["failure_type"],
        )

        # SLA compliance gauges (1 = compliant, 0 = non-compliant)
        RPO_COMPLIANCE = Gauge(
            "aragora_rpo_compliance",
            "RPO compliance status (1=compliant, 0=non-compliant)",
            ["tier"],
        )

        RTO_COMPLIANCE = Gauge(
            "aragora_rto_compliance",
            "RTO compliance status (1=compliant, 0=non-compliant)",
            ["tier"],
        )

        _initialized = True
        logger.info("Backup monitoring metrics initialized")

    except ImportError:
        logger.warning(
            "prometheus_client not installed, backup metrics disabled. "
            "Install with: pip install prometheus-client"
        )


@dataclass
class BackupMetrics:
    """Current backup metrics snapshot."""

    last_backup_timestamp: Optional[float] = None
    backup_age_seconds: Optional[float] = None
    last_backup_size_bytes: Optional[int] = None
    last_restore_duration: Optional[float] = None
    rpo_compliant: dict[str, bool] = None  # type: ignore
    rto_compliant: dict[str, bool] = None  # type: ignore

    def __post_init__(self):
        if self.rpo_compliant is None:
            self.rpo_compliant = {}
        if self.rto_compliant is None:
            self.rto_compliant = {}


# In-memory state for metrics (used when prometheus_client not available)
_last_backup_timestamp: Optional[float] = None
_last_backup_size: Optional[int] = None
_last_restore_duration: Optional[float] = None


def record_backup_created(
    size_bytes: int,
    duration_seconds: float,
    backup_type: str = "full",
) -> None:
    """Record a successful backup creation.

    Args:
        size_bytes: Size of the backup in bytes
        duration_seconds: Time taken to create the backup
        backup_type: Type of backup (full, incremental, differential)
    """
    global _last_backup_timestamp, _last_backup_size

    _init_metrics()
    _last_backup_timestamp = time.time()
    _last_backup_size = size_bytes

    if BACKUP_CREATED_TOTAL is not None:
        BACKUP_CREATED_TOTAL.labels(backup_type=backup_type).inc()

    if BACKUP_SIZE_BYTES is not None:
        BACKUP_SIZE_BYTES.labels(backup_type=backup_type).set(size_bytes)

    if BACKUP_DURATION_SECONDS is not None:
        BACKUP_DURATION_SECONDS.labels(backup_type=backup_type).observe(duration_seconds)

    if BACKUP_LAST_TIMESTAMP is not None:
        BACKUP_LAST_TIMESTAMP.set(_last_backup_timestamp)

    if BACKUP_AGE_SECONDS is not None:
        BACKUP_AGE_SECONDS.set(0)  # Just created

    # Update compliance status
    _update_compliance_status()

    logger.info(
        f"Backup created: type={backup_type}, size={size_bytes}, duration={duration_seconds:.2f}s"
    )


def record_backup_verified(success: bool, failure_type: Optional[str] = None) -> None:
    """Record a backup verification result.

    Args:
        success: Whether verification succeeded
        failure_type: Type of failure if not successful
    """
    _init_metrics()

    if BACKUP_VERIFICATION_TOTAL is not None:
        result = "success" if success else "failure"
        BACKUP_VERIFICATION_TOTAL.labels(result=result).inc()

    if not success and failure_type and BACKUP_VERIFICATION_FAILURES is not None:
        BACKUP_VERIFICATION_FAILURES.labels(failure_type=failure_type).inc()


def record_restore_completed(
    duration_seconds: float,
    success: bool,
    failure_type: Optional[str] = None,
) -> None:
    """Record a restore operation result.

    Args:
        duration_seconds: Time taken for restore
        success: Whether restore succeeded
        failure_type: Type of failure if not successful
    """
    global _last_restore_duration

    _init_metrics()
    _last_restore_duration = duration_seconds

    if RESTORE_DURATION_SECONDS is not None:
        RESTORE_DURATION_SECONDS.observe(duration_seconds)

    if RESTORE_TOTAL is not None:
        result = "success" if success else "failure"
        RESTORE_TOTAL.labels(result=result).inc()

    if not success and failure_type and RESTORE_FAILURES is not None:
        RESTORE_FAILURES.labels(failure_type=failure_type).inc()

    # Update RTO compliance
    _update_compliance_status()

    logger.info(f"Restore completed: success={success}, duration={duration_seconds:.2f}s")


def update_backup_age() -> Optional[float]:
    """Update and return the current backup age in seconds.

    Returns:
        Backup age in seconds, or None if no backup exists
    """
    _init_metrics()

    if _last_backup_timestamp is None:
        return None

    age = time.time() - _last_backup_timestamp

    if BACKUP_AGE_SECONDS is not None:
        BACKUP_AGE_SECONDS.set(age)

    _update_compliance_status()

    return age


def get_backup_age_seconds() -> Optional[float]:
    """Get the current backup age in seconds.

    Returns:
        Backup age in seconds, or None if no backup exists
    """
    if _last_backup_timestamp is None:
        return None
    return time.time() - _last_backup_timestamp


def get_current_metrics() -> BackupMetrics:
    """Get current backup metrics snapshot.

    Returns:
        BackupMetrics with current state
    """
    backup_age = get_backup_age_seconds()

    rpo_compliant = {}
    rto_compliant = {}

    for tier, targets in SLA_TARGETS.items():
        if backup_age is not None:
            rpo_compliant[tier] = backup_age <= targets["rpo"]
        else:
            rpo_compliant[tier] = False

        if _last_restore_duration is not None:
            rto_compliant[tier] = _last_restore_duration <= targets["rto"]
        else:
            rto_compliant[tier] = True  # Assume compliant if not tested

    return BackupMetrics(
        last_backup_timestamp=_last_backup_timestamp,
        backup_age_seconds=backup_age,
        last_backup_size_bytes=_last_backup_size,
        last_restore_duration=_last_restore_duration,
        rpo_compliant=rpo_compliant,
        rto_compliant=rto_compliant,
    )


def _update_compliance_status() -> None:
    """Update RPO and RTO compliance gauges."""
    if RPO_COMPLIANCE is None or RTO_COMPLIANCE is None:
        return

    backup_age = get_backup_age_seconds()

    for tier, targets in SLA_TARGETS.items():
        # RPO compliance
        if backup_age is not None:
            rpo_ok = 1 if backup_age <= targets["rpo"] else 0
        else:
            rpo_ok = 0  # No backup = non-compliant
        RPO_COMPLIANCE.labels(tier=tier).set(rpo_ok)

        # RTO compliance
        if _last_restore_duration is not None:
            rto_ok = 1 if _last_restore_duration <= targets["rto"] else 0
        else:
            rto_ok = 1  # Not tested = assume compliant
        RTO_COMPLIANCE.labels(tier=tier).set(rto_ok)


def check_rpo_breach(tier: str = "pro") -> bool:
    """Check if RPO is breached for a given tier.

    Args:
        tier: SLA tier to check (free, pro, enterprise)

    Returns:
        True if RPO is breached (backup too old)
    """
    if tier not in SLA_TARGETS:
        raise ValueError(f"Unknown tier: {tier}")

    backup_age = get_backup_age_seconds()
    if backup_age is None:
        return True  # No backup = breached

    return backup_age > SLA_TARGETS[tier]["rpo"]


def check_rto_breach(tier: str = "pro") -> bool:
    """Check if RTO is breached for a given tier.

    Args:
        tier: SLA tier to check (free, pro, enterprise)

    Returns:
        True if last restore exceeded RTO
    """
    if tier not in SLA_TARGETS:
        raise ValueError(f"Unknown tier: {tier}")

    if _last_restore_duration is None:
        return False  # No restore = not breached

    return _last_restore_duration > SLA_TARGETS[tier]["rto"]


# Alerting helpers
def get_alerts() -> list[dict]:
    """Get list of current backup-related alerts.

    Returns:
        List of alert dictionaries with severity and message
    """
    alerts = []

    backup_age = get_backup_age_seconds()

    if backup_age is None:
        alerts.append(
            {
                "severity": "critical",
                "message": "No backups found",
                "metric": "backup_age",
            }
        )
    else:
        # Check against each tier
        for tier, targets in SLA_TARGETS.items():
            if backup_age > targets["rpo"]:
                severity = "critical" if tier == "enterprise" else "warning"
                alerts.append(
                    {
                        "severity": severity,
                        "message": f"Backup age ({backup_age / 3600:.1f}h) exceeds {tier} RPO ({targets['rpo'] / 3600:.1f}h)",
                        "metric": "backup_age",
                        "tier": tier,
                    }
                )
                break  # Only report most severe breach

    return alerts


__all__ = [
    "record_backup_created",
    "record_backup_verified",
    "record_restore_completed",
    "update_backup_age",
    "get_backup_age_seconds",
    "get_current_metrics",
    "check_rpo_breach",
    "check_rto_breach",
    "get_alerts",
    "BackupMetrics",
    "SLA_TARGETS",
]
