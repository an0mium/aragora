"""
Security metrics for Aragora.

Provides metrics for monitoring security operations including:
- Encryption/decryption operations
- Key management and rotation
- Authentication attempts
- RBAC authorization decisions
- Secret access patterns
- Security incidents

Usage:
    from aragora.observability.metrics.security import (
        record_encryption_operation,
        record_auth_attempt,
        record_rbac_decision,
    )

    # Record an encryption operation
    record_encryption_operation("encrypt", success=True, latency=0.05)

    # Record an authentication attempt
    record_auth_attempt(user_id="user_123", success=True, method="jwt")

    # Record an RBAC decision
    record_rbac_decision(permission="debates.create", granted=True)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

# Module-level initialization state
_initialized = False

# Encryption metrics
ENCRYPTION_OPERATIONS_TOTAL: Any = None
ENCRYPTION_OPERATION_LATENCY: Any = None
ENCRYPTION_ERRORS_TOTAL: Any = None
ENCRYPTED_FIELDS_TOTAL: Any = None

# Key management metrics
KEY_OPERATIONS_TOTAL: Any = None
KEY_ROTATION_TOTAL: Any = None
KEY_ROTATION_LATENCY: Any = None
ACTIVE_KEYS_GAUGE: Any = None

# Authentication metrics
AUTH_ATTEMPTS_TOTAL: Any = None
AUTH_FAILURES_TOTAL: Any = None
AUTH_LATENCY: Any = None
ACTIVE_SESSIONS_GAUGE: Any = None

# RBAC metrics
RBAC_DECISIONS_TOTAL: Any = None
RBAC_DENIED_TOTAL: Any = None
RBAC_EVALUATION_LATENCY: Any = None
PERMISSIONS_CHECKED_TOTAL: Any = None

# Secret access metrics
SECRET_ACCESS_TOTAL: Any = None
SECRET_DECRYPTION_TOTAL: Any = None
SENSITIVE_FIELD_OPERATIONS: Any = None

# Security incident metrics
SECURITY_INCIDENTS_TOTAL: Any = None
SECURITY_ALERTS_TOTAL: Any = None
BLOCKED_REQUESTS_TOTAL: Any = None

# Migration metrics
MIGRATION_RECORDS_TOTAL: Any = None
MIGRATION_ERRORS_TOTAL: Any = None
MIGRATION_DURATION: Any = None


def init_security_metrics() -> bool:
    """Initialize security Prometheus metrics."""
    global _initialized
    global ENCRYPTION_OPERATIONS_TOTAL, ENCRYPTION_OPERATION_LATENCY
    global ENCRYPTION_ERRORS_TOTAL, ENCRYPTED_FIELDS_TOTAL
    global KEY_OPERATIONS_TOTAL, KEY_ROTATION_TOTAL
    global KEY_ROTATION_LATENCY, ACTIVE_KEYS_GAUGE
    global AUTH_ATTEMPTS_TOTAL, AUTH_FAILURES_TOTAL
    global AUTH_LATENCY, ACTIVE_SESSIONS_GAUGE
    global RBAC_DECISIONS_TOTAL, RBAC_DENIED_TOTAL
    global RBAC_EVALUATION_LATENCY, PERMISSIONS_CHECKED_TOTAL
    global SECRET_ACCESS_TOTAL, SECRET_DECRYPTION_TOTAL
    global SENSITIVE_FIELD_OPERATIONS
    global SECURITY_INCIDENTS_TOTAL, SECURITY_ALERTS_TOTAL
    global BLOCKED_REQUESTS_TOTAL
    global MIGRATION_RECORDS_TOTAL, MIGRATION_ERRORS_TOTAL, MIGRATION_DURATION

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Encryption metrics
        ENCRYPTION_OPERATIONS_TOTAL = Counter(
            "aragora_security_encryption_operations_total",
            "Total encryption/decryption operations",
            ["operation", "status"],
        )
        ENCRYPTION_OPERATION_LATENCY = Histogram(
            "aragora_security_encryption_latency_seconds",
            "Encryption operation latency in seconds",
            ["operation"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        )
        ENCRYPTION_ERRORS_TOTAL = Counter(
            "aragora_security_encryption_errors_total",
            "Total encryption errors",
            ["operation", "error_type"],
        )
        ENCRYPTED_FIELDS_TOTAL = Counter(
            "aragora_security_encrypted_fields_total",
            "Total fields encrypted",
            ["field_type", "store"],
        )

        # Key management metrics
        KEY_OPERATIONS_TOTAL = Counter(
            "aragora_security_key_operations_total",
            "Total key management operations",
            ["operation", "status"],
        )
        KEY_ROTATION_TOTAL = Counter(
            "aragora_security_key_rotations_total",
            "Total key rotations performed",
            ["key_id", "status"],
        )
        KEY_ROTATION_LATENCY = Histogram(
            "aragora_security_key_rotation_latency_seconds",
            "Key rotation latency in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )
        ACTIVE_KEYS_GAUGE = Gauge(
            "aragora_security_active_keys",
            "Number of active encryption keys",
            ["key_type"],
        )

        # Authentication metrics
        AUTH_ATTEMPTS_TOTAL = Counter(
            "aragora_security_auth_attempts_total",
            "Total authentication attempts",
            ["method", "status"],
        )
        AUTH_FAILURES_TOTAL = Counter(
            "aragora_security_auth_failures_total",
            "Total authentication failures",
            ["method", "reason"],
        )
        AUTH_LATENCY = Histogram(
            "aragora_security_auth_latency_seconds",
            "Authentication latency in seconds",
            ["method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        )
        ACTIVE_SESSIONS_GAUGE = Gauge(
            "aragora_security_active_sessions",
            "Number of active authenticated sessions",
            ["session_type"],
        )

        # RBAC metrics
        RBAC_DECISIONS_TOTAL = Counter(
            "aragora_security_rbac_decisions_total",
            "Total RBAC authorization decisions",
            ["permission", "decision"],
        )
        RBAC_DENIED_TOTAL = Counter(
            "aragora_security_rbac_denied_total",
            "Total RBAC denials by permission",
            ["permission", "role"],
        )
        RBAC_EVALUATION_LATENCY = Histogram(
            "aragora_security_rbac_evaluation_latency_seconds",
            "RBAC policy evaluation latency",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025],
        )
        PERMISSIONS_CHECKED_TOTAL = Counter(
            "aragora_security_permissions_checked_total",
            "Total permissions checked",
            ["permission"],
        )

        # Secret access metrics
        SECRET_ACCESS_TOTAL = Counter(
            "aragora_security_secret_access_total",
            "Total secret access operations",
            ["secret_type", "operation"],
        )
        SECRET_DECRYPTION_TOTAL = Counter(
            "aragora_security_secret_decryption_total",
            "Total secret decryption operations",
            ["store", "field"],
        )
        SENSITIVE_FIELD_OPERATIONS = Counter(
            "aragora_security_sensitive_field_operations_total",
            "Total sensitive field operations",
            ["field", "operation"],
        )

        # Security incident metrics
        SECURITY_INCIDENTS_TOTAL = Counter(
            "aragora_security_incidents_total",
            "Total security incidents detected",
            ["severity", "type"],
        )
        SECURITY_ALERTS_TOTAL = Counter(
            "aragora_security_alerts_total",
            "Total security alerts raised",
            ["alert_type", "destination"],
        )
        BLOCKED_REQUESTS_TOTAL = Counter(
            "aragora_security_blocked_requests_total",
            "Total blocked requests",
            ["reason", "source"],
        )

        # Migration metrics
        MIGRATION_RECORDS_TOTAL = Counter(
            "aragora_security_migration_records_total",
            "Total records migrated during encryption migration",
            ["store", "status"],
        )
        MIGRATION_ERRORS_TOTAL = Counter(
            "aragora_security_migration_errors_total",
            "Total errors during encryption migration",
            ["store", "error_type"],
        )
        MIGRATION_DURATION = Histogram(
            "aragora_security_migration_duration_seconds",
            "Encryption migration duration in seconds",
            ["store"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        _initialized = True
        logger.debug("Security metrics initialized")
        return True

    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global ENCRYPTION_OPERATIONS_TOTAL, ENCRYPTION_OPERATION_LATENCY
    global ENCRYPTION_ERRORS_TOTAL, ENCRYPTED_FIELDS_TOTAL
    global KEY_OPERATIONS_TOTAL, KEY_ROTATION_TOTAL
    global KEY_ROTATION_LATENCY, ACTIVE_KEYS_GAUGE
    global AUTH_ATTEMPTS_TOTAL, AUTH_FAILURES_TOTAL
    global AUTH_LATENCY, ACTIVE_SESSIONS_GAUGE
    global RBAC_DECISIONS_TOTAL, RBAC_DENIED_TOTAL
    global RBAC_EVALUATION_LATENCY, PERMISSIONS_CHECKED_TOTAL
    global SECRET_ACCESS_TOTAL, SECRET_DECRYPTION_TOTAL
    global SENSITIVE_FIELD_OPERATIONS
    global SECURITY_INCIDENTS_TOTAL, SECURITY_ALERTS_TOTAL
    global BLOCKED_REQUESTS_TOTAL
    global MIGRATION_RECORDS_TOTAL, MIGRATION_ERRORS_TOTAL, MIGRATION_DURATION

    noop = NoOpMetric()
    ENCRYPTION_OPERATIONS_TOTAL = noop
    ENCRYPTION_OPERATION_LATENCY = noop
    ENCRYPTION_ERRORS_TOTAL = noop
    ENCRYPTED_FIELDS_TOTAL = noop
    KEY_OPERATIONS_TOTAL = noop
    KEY_ROTATION_TOTAL = noop
    KEY_ROTATION_LATENCY = noop
    ACTIVE_KEYS_GAUGE = noop
    AUTH_ATTEMPTS_TOTAL = noop
    AUTH_FAILURES_TOTAL = noop
    AUTH_LATENCY = noop
    ACTIVE_SESSIONS_GAUGE = noop
    RBAC_DECISIONS_TOTAL = noop
    RBAC_DENIED_TOTAL = noop
    RBAC_EVALUATION_LATENCY = noop
    PERMISSIONS_CHECKED_TOTAL = noop
    SECRET_ACCESS_TOTAL = noop
    SECRET_DECRYPTION_TOTAL = noop
    SENSITIVE_FIELD_OPERATIONS = noop
    SECURITY_INCIDENTS_TOTAL = noop
    SECURITY_ALERTS_TOTAL = noop
    BLOCKED_REQUESTS_TOTAL = noop
    MIGRATION_RECORDS_TOTAL = noop
    MIGRATION_ERRORS_TOTAL = noop
    MIGRATION_DURATION = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_security_metrics()


# =============================================================================
# Encryption Functions
# =============================================================================


def record_encryption_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record an encryption or decryption operation.

    Args:
        operation: "encrypt" or "decrypt"
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
    """
    _ensure_init()
    status = "success" if success else "error"
    ENCRYPTION_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    ENCRYPTION_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def record_encryption_error(operation: str, error_type: str) -> None:
    """Record an encryption error.

    Args:
        operation: "encrypt" or "decrypt"
        error_type: Type of error (e.g., "invalid_key", "decryption_failed")
    """
    _ensure_init()
    ENCRYPTION_ERRORS_TOTAL.labels(operation=operation, error_type=error_type).inc()


def record_encrypted_field(field_type: str, store: str) -> None:
    """Record a field that was encrypted.

    Args:
        field_type: Type of field (e.g., "api_key", "access_token")
        store: Store name (e.g., "integrations", "webhooks")
    """
    _ensure_init()
    ENCRYPTED_FIELDS_TOTAL.labels(field_type=field_type, store=store).inc()


@contextmanager
def track_encryption_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track encryption operations.

    Args:
        operation: "encrypt" or "decrypt"
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_encryption_operation(operation, success, latency)


# =============================================================================
# Key Management Functions
# =============================================================================


def record_key_operation(operation: str, success: bool) -> None:
    """Record a key management operation.

    Args:
        operation: "generate", "load", "delete", etc.
        success: Whether the operation succeeded
    """
    _ensure_init()
    status = "success" if success else "error"
    KEY_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()


def record_key_rotation(key_id: str, success: bool, latency_seconds: float) -> None:
    """Record a key rotation.

    Args:
        key_id: ID of the rotated key
        success: Whether rotation succeeded
        latency_seconds: Rotation latency in seconds
    """
    _ensure_init()
    status = "success" if success else "error"
    KEY_ROTATION_TOTAL.labels(key_id=key_id, status=status).inc()
    KEY_ROTATION_LATENCY.observe(latency_seconds)


def set_active_keys(master: int = 0, session: int = 0, ephemeral: int = 0) -> None:
    """Set the number of active encryption keys by type.

    Args:
        master: Number of master keys
        session: Number of session keys
        ephemeral: Number of ephemeral keys
    """
    _ensure_init()
    ACTIVE_KEYS_GAUGE.labels(key_type="master").set(master)
    ACTIVE_KEYS_GAUGE.labels(key_type="session").set(session)
    ACTIVE_KEYS_GAUGE.labels(key_type="ephemeral").set(ephemeral)


# =============================================================================
# Authentication Functions
# =============================================================================


def record_auth_attempt(
    method: str,
    success: bool,
    latency_seconds: Optional[float] = None,
) -> None:
    """Record an authentication attempt.

    Args:
        method: Auth method (e.g., "jwt", "api_key", "oauth")
        success: Whether authentication succeeded
        latency_seconds: Authentication latency in seconds
    """
    _ensure_init()
    status = "success" if success else "failure"
    AUTH_ATTEMPTS_TOTAL.labels(method=method, status=status).inc()
    if latency_seconds is not None:
        AUTH_LATENCY.labels(method=method).observe(latency_seconds)


def record_auth_failure(method: str, reason: str) -> None:
    """Record an authentication failure with reason.

    Args:
        method: Auth method (e.g., "jwt", "api_key", "oauth")
        reason: Failure reason (e.g., "expired_token", "invalid_signature")
    """
    _ensure_init()
    AUTH_FAILURES_TOTAL.labels(method=method, reason=reason).inc()


def set_active_sessions(jwt: int = 0, api_key: int = 0, oauth: int = 0) -> None:
    """Set the number of active sessions by type.

    Args:
        jwt: Number of active JWT sessions
        api_key: Number of active API key sessions
        oauth: Number of active OAuth sessions
    """
    _ensure_init()
    ACTIVE_SESSIONS_GAUGE.labels(session_type="jwt").set(jwt)
    ACTIVE_SESSIONS_GAUGE.labels(session_type="api_key").set(api_key)
    ACTIVE_SESSIONS_GAUGE.labels(session_type="oauth").set(oauth)


@contextmanager
def track_auth_attempt(method: str) -> Generator[None, None, None]:
    """Context manager to track authentication attempts.

    Args:
        method: Auth method (e.g., "jwt", "api_key", "oauth")
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_auth_attempt(method, success, latency)


# =============================================================================
# RBAC Functions
# =============================================================================


def record_rbac_decision(permission: str, granted: bool) -> None:
    """Record an RBAC authorization decision.

    Args:
        permission: Permission being checked (e.g., "debates.create")
        granted: Whether permission was granted
    """
    _ensure_init()
    decision = "granted" if granted else "denied"
    RBAC_DECISIONS_TOTAL.labels(permission=permission, decision=decision).inc()
    PERMISSIONS_CHECKED_TOTAL.labels(permission=permission).inc()


def record_rbac_denial(permission: str, role: str) -> None:
    """Record an RBAC denial with role context.

    Args:
        permission: Permission that was denied
        role: Role that was denied
    """
    _ensure_init()
    RBAC_DENIED_TOTAL.labels(permission=permission, role=role).inc()


def record_rbac_evaluation_latency(latency_seconds: float) -> None:
    """Record RBAC policy evaluation latency.

    Args:
        latency_seconds: Evaluation latency in seconds
    """
    _ensure_init()
    RBAC_EVALUATION_LATENCY.observe(latency_seconds)


@contextmanager
def track_rbac_evaluation() -> Generator[None, None, None]:
    """Context manager to track RBAC evaluation latency."""
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_rbac_evaluation_latency(latency)


# =============================================================================
# Secret Access Functions
# =============================================================================


def record_secret_access(secret_type: str, operation: str) -> None:
    """Record a secret access operation.

    Args:
        secret_type: Type of secret (e.g., "api_key", "oauth_token")
        operation: Operation type (e.g., "read", "write", "delete")
    """
    _ensure_init()
    SECRET_ACCESS_TOTAL.labels(secret_type=secret_type, operation=operation).inc()


def record_secret_decryption(store: str, field: str) -> None:
    """Record a secret decryption operation.

    Args:
        store: Store name (e.g., "integrations", "webhooks")
        field: Field name (e.g., "api_key", "secret")
    """
    _ensure_init()
    SECRET_DECRYPTION_TOTAL.labels(store=store, field=field).inc()


def record_sensitive_field_operation(field: str, operation: str) -> None:
    """Record an operation on a sensitive field.

    Args:
        field: Field name
        operation: Operation type (e.g., "encrypt", "decrypt", "access")
    """
    _ensure_init()
    SENSITIVE_FIELD_OPERATIONS.labels(field=field, operation=operation).inc()


# =============================================================================
# Security Incident Functions
# =============================================================================


def record_security_incident(severity: str, incident_type: str) -> None:
    """Record a security incident.

    Args:
        severity: "critical", "high", "medium", "low"
        incident_type: Type of incident (e.g., "unauthorized_access", "brute_force")
    """
    _ensure_init()
    SECURITY_INCIDENTS_TOTAL.labels(severity=severity, type=incident_type).inc()


def record_security_alert(alert_type: str, destination: str) -> None:
    """Record a security alert.

    Args:
        alert_type: Type of alert
        destination: Where the alert was sent (e.g., "siem", "email", "slack")
    """
    _ensure_init()
    SECURITY_ALERTS_TOTAL.labels(alert_type=alert_type, destination=destination).inc()


def record_blocked_request(reason: str, source: str) -> None:
    """Record a blocked request.

    Args:
        reason: Reason for blocking (e.g., "rate_limit", "unauthorized", "invalid_token")
        source: Source of the request (e.g., "ip", "user_id")
    """
    _ensure_init()
    BLOCKED_REQUESTS_TOTAL.labels(reason=reason, source=source).inc()


# =============================================================================
# Migration Functions
# =============================================================================


def record_migration_record(store: str, success: bool) -> None:
    """Record a record migrated during encryption migration.

    Args:
        store: Store being migrated (e.g., "integrations", "webhooks")
        success: Whether the record was migrated successfully
    """
    _ensure_init()
    status = "success" if success else "error"
    MIGRATION_RECORDS_TOTAL.labels(store=store, status=status).inc()


def record_migration_error(store: str, error_type: str) -> None:
    """Record an error during encryption migration.

    Args:
        store: Store being migrated
        error_type: Type of error
    """
    _ensure_init()
    MIGRATION_ERRORS_TOTAL.labels(store=store, error_type=error_type).inc()


def record_migration_duration(store: str, duration_seconds: float) -> None:
    """Record the duration of a migration operation.

    Args:
        store: Store that was migrated
        duration_seconds: Duration in seconds
    """
    _ensure_init()
    MIGRATION_DURATION.labels(store=store).observe(duration_seconds)


@contextmanager
def track_migration(store: str) -> Generator[None, None, None]:
    """Context manager to track migration operations.

    Args:
        store: Store being migrated
    """
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_migration_duration(store, duration)
