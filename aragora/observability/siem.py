"""
SIEM (Security Information and Event Management) Integration.

Provides structured security event streaming for enterprise compliance
and threat detection. Supports multiple SIEM backends:

- Splunk (HTTP Event Collector)
- AWS CloudWatch Logs
- Azure Sentinel (Log Analytics)
- Datadog Logs
- Generic Syslog

Configuration via environment variables:
    SIEM_BACKEND: splunk, cloudwatch, sentinel, datadog, syslog, none
    SIEM_ENDPOINT: Backend-specific endpoint URL
    SIEM_TOKEN: Authentication token (for Splunk HEC, Datadog)
    SIEM_INDEX: Index/stream name (for Splunk, CloudWatch)
    SIEM_BATCH_SIZE: Events to batch before sending (default: 10)
    SIEM_FLUSH_INTERVAL: Seconds between flushes (default: 5)

Usage:
    from aragora.observability.siem import (
        emit_security_event,
        emit_auth_event,
        emit_data_access_event,
        get_siem_client,
    )

    # Emit authentication event
    emit_auth_event(
        user_id="user-123",
        action="login_success",
        ip_address="192.168.1.1",
        metadata={"mfa_used": True},
    )

    # Emit data access event (GDPR/compliance)
    emit_data_access_event(
        user_id="user-123",
        resource_type="debate",
        resource_id="debate-456",
        action="read",
        granted=True,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class SIEMBackend(Enum):
    """Supported SIEM backends."""

    NONE = "none"
    SPLUNK = "splunk"
    CLOUDWATCH = "cloudwatch"
    SENTINEL = "sentinel"
    DATADOG = "datadog"
    SYSLOG = "syslog"


@dataclass
class SIEMConfig:
    """SIEM configuration."""

    backend: SIEMBackend = SIEMBackend.NONE
    endpoint: str = ""
    token: str = ""
    index: str = "aragora-security"
    batch_size: int = 10
    flush_interval: float = 5.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "SIEMConfig":
        """Load configuration from environment variables."""
        backend_str = os.environ.get("SIEM_BACKEND", "none").lower()
        try:
            backend = SIEMBackend(backend_str)
        except ValueError:
            logger.warning(f"Unknown SIEM backend: {backend_str}, using none")
            backend = SIEMBackend.NONE

        return cls(
            backend=backend,
            endpoint=os.environ.get("SIEM_ENDPOINT", ""),
            token=os.environ.get("SIEM_TOKEN", ""),
            index=os.environ.get("SIEM_INDEX", "aragora-security"),
            batch_size=int(os.environ.get("SIEM_BATCH_SIZE", "10")),
            flush_interval=float(os.environ.get("SIEM_FLUSH_INTERVAL", "5")),
            enabled=os.environ.get("SIEM_ENABLED", "true").lower() == "true",
        )


# =============================================================================
# Security Event Types
# =============================================================================


class SecurityEventType(Enum):
    """Types of security events."""

    # Authentication
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_MFA_CHALLENGE = "auth.mfa.challenge"
    AUTH_MFA_SUCCESS = "auth.mfa.success"
    AUTH_MFA_FAILURE = "auth.mfa.failure"
    AUTH_PASSWORD_CHANGE = "auth.password.change"
    AUTH_PASSWORD_RESET = "auth.password.reset"

    # Authorization
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_ROLE_CHANGE = "authz.role.change"
    AUTHZ_PERMISSION_CHANGE = "authz.permission.change"

    # Data Access
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_SHARE = "data.share"

    # Privacy
    PRIVACY_CONSENT_GRANTED = "privacy.consent.granted"
    PRIVACY_CONSENT_REVOKED = "privacy.consent.revoked"
    PRIVACY_DATA_REQUEST = "privacy.data.request"
    PRIVACY_DATA_DELETION = "privacy.data.deletion"

    # Security Incidents
    SECURITY_RATE_LIMIT = "security.rate_limit"
    SECURITY_INVALID_TOKEN = "security.invalid_token"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious"
    SECURITY_BRUTE_FORCE = "security.brute_force"

    # API Activity
    API_KEY_CREATED = "api.key.created"
    API_KEY_REVOKED = "api.key.revoked"
    API_KEY_USED = "api.key.used"

    # Admin Actions
    ADMIN_USER_CREATED = "admin.user.created"
    ADMIN_USER_DELETED = "admin.user.deleted"
    ADMIN_USER_SUSPENDED = "admin.user.suspended"
    ADMIN_CONFIG_CHANGE = "admin.config.change"


@dataclass
class SecurityEvent:
    """A security event to be sent to SIEM."""

    event_type: SecurityEventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "severity": self.severity,
            "metadata": self.metadata,
            "source": "aragora",
            "version": "1.0",
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


# =============================================================================
# SIEM Client
# =============================================================================


class SIEMClient:
    """Client for sending events to SIEM backend."""

    def __init__(self, config: Optional[SIEMConfig] = None):
        """Initialize SIEM client.

        Args:
            config: SIEM configuration (loads from env if not provided)
        """
        self.config = config or SIEMConfig.from_env()
        self._queue: Queue[SecurityEvent] = Queue()
        self._shutdown = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if self.config.enabled and self.config.backend != SIEMBackend.NONE:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread."""
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        logger.info(f"SIEM worker started: backend={self.config.backend.value}")

    def _worker_loop(self) -> None:
        """Background worker that batches and sends events."""
        batch: List[SecurityEvent] = []
        last_flush = time.time()

        while not self._shutdown.is_set():
            try:
                # Get event with timeout
                event = self._queue.get(timeout=1.0)
                batch.append(event)
            except Empty:
                pass

            # Flush if batch full or interval elapsed
            now = time.time()
            should_flush = (
                len(batch) >= self.config.batch_size
                or (batch and now - last_flush >= self.config.flush_interval)
            )

            if should_flush and batch:
                self._send_batch(batch)
                batch = []
                last_flush = now

        # Final flush on shutdown
        if batch:
            self._send_batch(batch)

    def _send_batch(self, events: List[SecurityEvent]) -> None:
        """Send a batch of events to the configured backend."""
        if not events:
            return

        try:
            if self.config.backend == SIEMBackend.SPLUNK:
                self._send_to_splunk(events)
            elif self.config.backend == SIEMBackend.CLOUDWATCH:
                self._send_to_cloudwatch(events)
            elif self.config.backend == SIEMBackend.DATADOG:
                self._send_to_datadog(events)
            elif self.config.backend == SIEMBackend.SYSLOG:
                self._send_to_syslog(events)
            else:
                # Log locally if no backend configured
                for event in events:
                    logger.info(f"SIEM event: {event.to_json()}")

            logger.debug(f"Sent {len(events)} events to {self.config.backend.value}")
        except Exception as e:
            logger.error(f"Failed to send SIEM batch: {e}")

    def _send_to_splunk(self, events: List[SecurityEvent]) -> None:
        """Send events to Splunk HTTP Event Collector."""
        import urllib.request

        for event in events:
            data = json.dumps({
                "event": event.to_dict(),
                "index": self.config.index,
                "sourcetype": "aragora:security",
            }).encode("utf-8")

            req = urllib.request.Request(
                self.config.endpoint,
                data=data,
                headers={
                    "Authorization": f"Splunk {self.config.token}",
                    "Content-Type": "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=10)

    def _send_to_cloudwatch(self, events: List[SecurityEvent]) -> None:
        """Send events to AWS CloudWatch Logs."""
        try:
            import boto3

            client = boto3.client("logs")
            log_events = [
                {
                    "timestamp": int(time.time() * 1000),
                    "message": event.to_json(),
                }
                for event in events
            ]

            client.put_log_events(
                logGroupName=self.config.index,
                logStreamName="aragora-security",
                logEvents=log_events,
            )
        except ImportError:
            logger.warning("boto3 not installed, cannot send to CloudWatch")

    def _send_to_datadog(self, events: List[SecurityEvent]) -> None:
        """Send events to Datadog Logs."""
        import urllib.request

        for event in events:
            data = json.dumps({
                **event.to_dict(),
                "ddsource": "aragora",
                "service": "aragora-security",
            }).encode("utf-8")

            req = urllib.request.Request(
                self.config.endpoint or "https://http-intake.logs.datadoghq.com/api/v2/logs",
                data=data,
                headers={
                    "DD-API-KEY": self.config.token,
                    "Content-Type": "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=10)

    def _send_to_syslog(self, events: List[SecurityEvent]) -> None:
        """Send events to syslog."""
        import socket
        import syslog

        for event in events:
            severity = {
                "info": syslog.LOG_INFO,
                "warning": syslog.LOG_WARNING,
                "error": syslog.LOG_ERR,
                "critical": syslog.LOG_CRIT,
            }.get(event.severity, syslog.LOG_INFO)

            syslog.syslog(severity, event.to_json())

    def emit(self, event: SecurityEvent) -> None:
        """Emit a security event (async, non-blocking)."""
        if not self.config.enabled:
            return
        self._queue.put(event)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the client and flush remaining events."""
        self._shutdown.set()
        if self._worker:
            self._worker.join(timeout=timeout)


# =============================================================================
# Global Client & Helper Functions
# =============================================================================

_client: Optional[SIEMClient] = None
_client_lock = threading.Lock()


def get_siem_client() -> SIEMClient:
    """Get or create the global SIEM client."""
    global _client
    with _client_lock:
        if _client is None:
            _client = SIEMClient()
        return _client


def emit_security_event(
    event_type: SecurityEventType,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    outcome: str = "success",
    severity: str = "info",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a security event to SIEM.

    Args:
        event_type: Type of security event
        user_id: ID of the user (if applicable)
        organization_id: ID of the organization (if applicable)
        ip_address: Client IP address
        resource_type: Type of resource accessed
        resource_id: ID of resource accessed
        action: Specific action taken
        outcome: success, failure, or error
        severity: info, warning, error, or critical
        metadata: Additional context
    """
    event = SecurityEvent(
        event_type=event_type,
        user_id=user_id,
        organization_id=organization_id,
        ip_address=ip_address,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        outcome=outcome,
        severity=severity,
        metadata=metadata or {},
    )
    get_siem_client().emit(event)


def emit_auth_event(
    user_id: str,
    action: str,
    ip_address: Optional[str] = None,
    outcome: str = "success",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit an authentication event.

    Args:
        user_id: User ID
        action: login_success, login_failure, logout, etc.
        ip_address: Client IP
        outcome: success or failure
        metadata: Additional context (mfa_used, provider, etc.)
    """
    event_type = {
        "login_success": SecurityEventType.AUTH_LOGIN_SUCCESS,
        "login_failure": SecurityEventType.AUTH_LOGIN_FAILURE,
        "logout": SecurityEventType.AUTH_LOGOUT,
        "mfa_success": SecurityEventType.AUTH_MFA_SUCCESS,
        "mfa_failure": SecurityEventType.AUTH_MFA_FAILURE,
        "password_change": SecurityEventType.AUTH_PASSWORD_CHANGE,
    }.get(action, SecurityEventType.AUTH_LOGIN_SUCCESS)

    severity = "warning" if "failure" in action else "info"

    emit_security_event(
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        action=action,
        outcome=outcome,
        severity=severity,
        metadata=metadata,
    )


def emit_data_access_event(
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str,
    granted: bool = True,
    organization_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a data access event (for compliance/audit).

    Args:
        user_id: User ID
        resource_type: debate, agent, document, etc.
        resource_id: Resource ID
        action: read, write, delete, export, share
        granted: Whether access was granted
        organization_id: Organization ID
        metadata: Additional context
    """
    event_type = {
        "read": SecurityEventType.DATA_READ,
        "write": SecurityEventType.DATA_WRITE,
        "delete": SecurityEventType.DATA_DELETE,
        "export": SecurityEventType.DATA_EXPORT,
        "share": SecurityEventType.DATA_SHARE,
    }.get(action, SecurityEventType.DATA_READ)

    emit_security_event(
        event_type=event_type,
        user_id=user_id,
        organization_id=organization_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        outcome="success" if granted else "denied",
        severity="info" if granted else "warning",
        metadata=metadata,
    )


def emit_privacy_event(
    user_id: str,
    action: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a privacy-related event (GDPR/CCPA).

    Args:
        user_id: User ID
        action: consent_granted, consent_revoked, data_request, data_deletion
        metadata: Additional context
    """
    event_type = {
        "consent_granted": SecurityEventType.PRIVACY_CONSENT_GRANTED,
        "consent_revoked": SecurityEventType.PRIVACY_CONSENT_REVOKED,
        "data_request": SecurityEventType.PRIVACY_DATA_REQUEST,
        "data_deletion": SecurityEventType.PRIVACY_DATA_DELETION,
    }.get(action, SecurityEventType.PRIVACY_DATA_REQUEST)

    emit_security_event(
        event_type=event_type,
        user_id=user_id,
        action=action,
        outcome="success",
        severity="info",
        metadata=metadata,
    )


def shutdown_siem() -> None:
    """Shutdown the SIEM client gracefully."""
    global _client
    with _client_lock:
        if _client:
            _client.shutdown()
            _client = None


__all__ = [
    # Configuration
    "SIEMBackend",
    "SIEMConfig",
    # Event types
    "SecurityEventType",
    "SecurityEvent",
    # Client
    "SIEMClient",
    "get_siem_client",
    # Helper functions
    "emit_security_event",
    "emit_auth_event",
    "emit_data_access_event",
    "emit_privacy_event",
    "shutdown_siem",
]
