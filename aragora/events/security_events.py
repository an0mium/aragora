"""
Security Events Module for Aragora.

Provides event types and handlers for security-related events:
- Vulnerability detection events
- Secrets detection events
- Security scan completion events
- Debate triggering for critical findings

Integration Flow:
    SecurityScanner → Critical Finding → SecurityEvent → Arena.run()
                                                            ↓
                                        Multi-agent debate on remediation
                                                            ↓
                                        ConsensusProof → Recommended action
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events."""

    # Vulnerability events
    VULNERABILITY_DETECTED = "vulnerability_detected"
    CRITICAL_VULNERABILITY = "critical_vulnerability"
    VULNERABILITY_RESOLVED = "vulnerability_resolved"

    # CVE-specific events
    CRITICAL_CVE = "critical_cve"  # CVE with CVSS >= 9.0

    # Secrets events
    SECRET_DETECTED = "secret_detected"  # noqa: S105 -- enum value
    CRITICAL_SECRET = "critical_secret"  # noqa: S105 -- enum value
    SECRET_ROTATED = "secret_rotated"  # noqa: S105 -- enum value

    # SAST events
    SAST_CRITICAL = "sast_critical"  # SAST scanner found critical vulnerability

    # Threat intelligence events
    THREAT_DETECTED = "threat_detected"  # Threat intel match detected

    # Scan events
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETED = "scan_completed"
    SCAN_FAILED = "scan_failed"

    # Debate events
    SECURITY_DEBATE_REQUESTED = "security_debate_requested"
    SECURITY_DEBATE_STARTED = "security_debate_started"
    SECURITY_DEBATE_COMPLETED = "security_debate_completed"


class SecuritySeverity(str, Enum):
    """Security severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SECRET_FINDING_TYPES = frozenset(
    {
        "secret",
        "secrets",
        "credential",
        "credentials",
        "token",
        "api_key",
        "apikey",
    }
)
SECRET_INDICATOR_TERMS = (
    "secret",
    "credential",
    "credentials",
    "token",
    "api key",
    "api_key",
    "apikey",
    "password",
    "private key",
    "access key",
)
SECRET_VALUE_TERMS = (
    "api key",
    "api_key",
    "apikey",
    "api token",
    "access token",
    "auth token",
    "bearer token",
    "github token",
    "gitlab token",
    "slack token",
    "private key",
    "access key",
    "client secret",
    "secret key",
)
SECRET_METADATA_RULE_KEYS = frozenset({"check_id", "rule", "rule_id", "rule_name"})
SECRET_METADATA_CONTENT_KEYS = frozenset({"message", "snippet"})
SECRET_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "api_token",
        "apikey",
        "auth_token",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "github_token",
        "gitlab_token",
        "matched_secret",
        "password",
        "private_key",
        "raw_secret",
        "secret",
        "secret_key",
        "secret_value",
        "slack_token",
        "token",
    }
)
SECRET_CATEGORY_TERMS = ("secret", "secrets", "credential", "credentials")
SECRET_RULE_TERMS = (
    "credential",
    "credentials",
    "api key",
    "api_key",
    "apikey",
    "github token",
    "gitlab token",
    "slack token",
    "private key",
    "access key",
    "client secret",
    "secret key",
)
SECRET_EXPOSURE_TERMS = (
    "hardcoded",
    "hard coded",
    "embedded",
    "plaintext",
    "plain text",
    "committed",
)
SECRET_VALUE_EXPOSURE_TERMS = (
    "exposed",
    "exposure",
    "leaked",
)
SAFE_SECRET_TYPES = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "api_token",
        "auth_token",
        "aws_access_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "github_token",
        "gitlab_token",
        "hardcoded_credential",
        "password",
        "private_key",
        "secret_key",
        "slack_token",
        "token",
        "unknown",
    }
)


SECRET_FINDING_TEXT_KEYS = ("title", "description", "recommendation")


def _finding_dict(finding: Any) -> dict[str, Any]:
    if isinstance(finding, dict):
        return dict(finding)
    to_dict = getattr(finding, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def _finding_metadata(finding: Any, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or _finding_dict(finding)
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = getattr(finding, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata


def _iter_text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_iter_text_values(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for nested in value:
            values.extend(_iter_text_values(nested))
        return values
    if value is None:
        return []
    return [str(value)]


def _has_secret_indicator(value: Any) -> bool:
    for text in _iter_text_values(value):
        normalized = text.lower().replace("-", " ").replace(".", " ").replace("_", " ")
        compact = normalized.replace(" ", "")
        for term in SECRET_INDICATOR_TERMS:
            normalized_term = term.lower().replace("-", " ").replace(".", " ").replace("_", " ")
            if normalized_term in normalized or normalized_term.replace(" ", "") in compact:
                return True
    return False


def _text_has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.lower().replace("-", " ").replace(".", " ").replace("_", " ")
    compact = normalized.replace(" ", "")
    for term in terms:
        normalized_term = term.lower().replace("-", " ").replace(".", " ").replace("_", " ")
        if normalized_term in normalized or normalized_term.replace(" ", "") in compact:
            return True
    return False


def _has_secret_category_signature(value: Any) -> bool:
    return any(_text_has_any_term(text, SECRET_CATEGORY_TERMS) for text in _iter_text_values(value))


def _metadata_key_has_secret_signature(key: Any) -> bool:
    key_name = str(key).lower()
    return key_name in SECRET_SENSITIVE_METADATA_KEYS or _text_has_any_term(
        key_name, SECRET_VALUE_TERMS
    )


def _has_secret_metadata_key_signature(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _metadata_key_has_secret_signature(key):
                return True
            if _has_secret_metadata_key_signature(nested):
                return True
    elif isinstance(value, (list, tuple, set)):
        return any(_has_secret_metadata_key_signature(nested) for nested in value)
    return False


def safe_secret_type(value: Any) -> str:
    """Return an allowlisted secret type label safe for logs and prompts."""
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in SAFE_SECRET_TYPES:
        return normalized
    return "unknown"


def _has_secret_rule_signature(value: Any) -> bool:
    for text in _iter_text_values(value):
        if _text_has_any_term(text, SECRET_RULE_TERMS):
            return True
        if _has_secret_content_signature(text):
            return True
    return False


def _has_secret_content_signature(value: Any) -> bool:
    for text in _iter_text_values(value):
        if _text_has_any_term(text, SECRET_EXPOSURE_TERMS) and _has_secret_indicator(text):
            return True
        if _text_has_any_term(text, SECRET_VALUE_EXPOSURE_TERMS) and _text_has_any_term(
            text, SECRET_VALUE_TERMS
        ):
            return True
        if "=" in text and _text_has_any_term(text, SECRET_VALUE_TERMS):
            return True
    return False


def _has_secret_metadata_signature(metadata: dict[str, Any]) -> bool:
    if "secret_type" in metadata:
        return True
    if _has_secret_metadata_key_signature(metadata):
        return True

    for key, value in metadata.items():
        key_name = str(key).lower()
        if _metadata_key_has_secret_signature(key_name):
            return True
        if key_name in {"category", "categories", "tags"}:
            if _has_secret_category_signature(value):
                return True
            continue
        if key_name in SECRET_METADATA_RULE_KEYS and _has_secret_rule_signature(value):
            return True
        if key_name in SECRET_METADATA_CONTENT_KEYS and _has_secret_content_signature(value):
            return True
    return False


def _has_secret_finding_text_signature(finding: Any, data: dict[str, Any]) -> bool:
    values = []
    for key in SECRET_FINDING_TEXT_KEYS:
        value = data.get(key, getattr(finding, key, None))
        if value:
            values.append(value)
    return _has_secret_content_signature(values)


def redacted_security_metadata_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize event metadata without exposing secret-like material."""
    if not _has_secret_metadata_signature(metadata):
        return metadata
    return {"secret_type": safe_secret_type(metadata.get("secret_type"))}


def is_secret_finding(
    finding: Any,
    *,
    event_metadata: dict[str, Any] | None = None,
) -> bool:
    """Return whether a finding contains secret-like material."""
    data = _finding_dict(finding)
    finding_type = str(data.get("finding_type") or getattr(finding, "finding_type", "")).lower()
    if finding_type in SECRET_FINDING_TYPES:
        return True

    metadata = _finding_metadata(finding, data)
    if _has_secret_metadata_signature(metadata):
        return True

    if event_metadata and _has_secret_metadata_signature(event_metadata):
        return True

    if _has_secret_finding_text_signature(finding, data):
        return True

    return False


def redacted_security_finding_dict(
    finding: Any,
    *,
    event_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a finding while redacting secret-like material."""
    data = _finding_dict(finding)
    if not is_secret_finding(finding, event_metadata=event_metadata):
        return data

    metadata = _finding_metadata(finding, data)
    return {
        "id": data.get("id"),
        "finding_type": "secret",
        "severity": data.get("severity"),
        "title": "Secret finding",
        "description": "[redacted secret finding description]",
        "file_path": data.get("file_path", getattr(finding, "file_path", None)),
        "line_number": data.get("line_number", getattr(finding, "line_number", None)),
        "cve_id": None,
        "package_name": None,
        "package_version": None,
        "recommendation": "Rotate or revoke the exposed credential and remove it from history.",
        "metadata": {"secret_type": safe_secret_type(metadata.get("secret_type"))},
    }


def redacted_security_finding(
    finding: SecurityFinding,
    *,
    event_metadata: dict[str, Any] | None = None,
) -> SecurityFinding:
    """Return a redacted copy when a finding is secret-like."""
    if not is_secret_finding(finding, event_metadata=event_metadata):
        return finding

    data = redacted_security_finding_dict(finding, event_metadata=event_metadata)
    severity_value = data.get("severity") or SecuritySeverity.HIGH.value
    try:
        severity = SecuritySeverity(str(severity_value))
    except ValueError:
        severity = SecuritySeverity.HIGH

    return SecurityFinding(
        id=str(data.get("id") or uuid.uuid4()),
        finding_type="secret",
        severity=severity,
        title=str(data["title"]),
        description=str(data["description"]),
        file_path=data.get("file_path"),
        line_number=data.get("line_number"),
        recommendation=str(data["recommendation"]),
        metadata=dict(data["metadata"]),
    )


@dataclass
class SecurityFinding:
    """Represents a security finding that may trigger a debate."""

    id: str
    finding_type: str  # "vulnerability", "secret", "misconfiguration"
    severity: SecuritySeverity
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    cve_id: str | None = None
    package_name: str | None = None
    package_version: str | None = None
    recommendation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "finding_type": self.finding_type,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "cve_id": self.cve_id,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }
        if is_secret_finding(data):
            return redacted_security_finding_dict(data)
        data["metadata"] = redacted_security_metadata_dict(self.metadata)
        return data


@dataclass
class SecurityEvent:
    """Represents a security event with context for debate triggering."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.VULNERABILITY_DETECTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: SecuritySeverity = SecuritySeverity.MEDIUM

    # Source information - categorizes the origin of the event
    source: str = "sast"  # "sast", "secrets", "dependency", "threat_intel"
    repository: str | None = None
    scan_id: str | None = None
    workspace_id: str | None = None

    # Findings
    findings: list[SecurityFinding] = field(default_factory=list)

    # Debate context
    debate_requested: bool = False
    debate_id: str | None = None
    debate_question: str | None = None

    # Correlation
    correlation_id: str | None = None

    # Metadata for additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        finding_event_metadata = self.metadata if len(self.findings) == 1 else None
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source": self.source,
            "repository": self.repository,
            "scan_id": self.scan_id,
            "workspace_id": self.workspace_id,
            "findings": [
                redacted_security_finding_dict(f, event_metadata=finding_event_metadata)
                for f in self.findings
            ],
            "debate_requested": self.debate_requested,
            "debate_id": self.debate_id,
            "debate_question": self.debate_question,
            "correlation_id": self.correlation_id,
            "metadata": redacted_security_metadata_dict(self.metadata),
        }

    @property
    def is_critical(self) -> bool:
        """Check if event contains critical findings."""
        return self.severity == SecuritySeverity.CRITICAL or any(
            f.severity == SecuritySeverity.CRITICAL for f in self.findings
        )

    @property
    def critical_count(self) -> int:
        """Count of critical findings."""
        return sum(1 for f in self.findings if f.severity == SecuritySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity findings."""
        return sum(1 for f in self.findings if f.severity == SecuritySeverity.HIGH)


# Type alias for event handlers
SecurityEventHandler = Callable[[SecurityEvent], Coroutine[Any, Any, None]]

# Type alias for the domain-coupled security debate runner. Registered by
# aragora.debate.security_response (see register_security_debate_runner) so
# this module can auto-trigger debates without importing aragora.debate or
# aragora.agents directly.
SecurityDebateRunner = Callable[..., Coroutine[Any, Any, str | None]]


class _UnsetRunner:
    """Sentinel type marking a runner registry that was never explicitly set."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET security debate runner>"


_UNSET_RUNNER = _UnsetRunner()

# Tri-state registry: _UNSET_RUNNER (never set -> no domain-side composition
# root has registered the default runner yet), None (explicitly cleared via
# register_security_debate_runner(None) -> auto-debate stays disabled until a
# runner is registered again), or a SecurityDebateRunner callable. This module
# never imports aragora.debate itself (see _trigger_security_debate below);
# the default runner is registered by aragora.debate.security_response and
# its own composition roots, never lazily from here.
_security_debate_runner: SecurityDebateRunner | None | _UnsetRunner = _UNSET_RUNNER


def register_security_debate_runner(runner: SecurityDebateRunner | None) -> None:
    """Register the callback used to run security debates.

    Composition roots (aragora.debate.security_response and its own
    composition roots - see that module's docstring) call this to install the
    default runner, or to install/clear an explicit override. Passing None
    explicitly clears the registry and disables auto-debate until a runner is
    registered again. Default runner registration uses
    _register_default_security_debate_runner so it does not clobber an
    explicit hook or an explicit clear.
    """
    global _security_debate_runner
    _security_debate_runner = runner


def _register_default_security_debate_runner(
    runner: SecurityDebateRunner,
) -> SecurityDebateRunner | None:
    """Register the default runner only when the registry was never set.

    Neither an explicit runner nor an explicit None-clear is clobbered.
    """
    global _security_debate_runner
    if isinstance(_security_debate_runner, _UnsetRunner):
        _security_debate_runner = runner
    return get_security_debate_runner()


def get_security_debate_runner() -> SecurityDebateRunner | None:
    """Get the currently registered security debate runner, if any."""
    if isinstance(_security_debate_runner, _UnsetRunner):
        return None
    return _security_debate_runner


def _accepted_security_debate_runner_kwargs(
    runner: SecurityDebateRunner,
    *,
    confidence_threshold: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Return debate options accepted by a custom runner.

    Older integrations registered ``async def runner(event)`` callbacks. The
    default runner accepts the newer keyword options, but custom callbacks should
    not break critical event delivery merely because they have not adopted them.
    """
    options = {
        "confidence_threshold": confidence_threshold,
        "timeout_seconds": timeout_seconds,
    }
    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return options

    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return options

    return {name: value for name, value in options.items() if name in parameters}


class SecurityEventEmitter:
    """
    Emits security events and optionally triggers debates for critical findings.

    Usage:
        emitter = SecurityEventEmitter()

        # Subscribe to events
        async def on_critical(event: SecurityEvent):
            print(f"Critical finding: {event.findings[0].title}")

        emitter.subscribe(SecurityEventType.CRITICAL_VULNERABILITY, on_critical)

        # Emit event (auto-triggers debate for critical findings if enabled)
        await emitter.emit(event)
    """

    # Minimum severity to trigger automatic debate
    AUTO_DEBATE_THRESHOLD = SecuritySeverity.CRITICAL

    def __init__(
        self,
        enable_auto_debate: bool = True,
        debate_confidence_threshold: float = 0.7,
        debate_timeout_seconds: int = 300,
        workspace_id: str | None = None,
    ):
        """
        Initialize the security event emitter.

        Args:
            enable_auto_debate: Whether to auto-trigger debates for critical findings
            debate_confidence_threshold: Minimum confidence for debate consensus
            workspace_id: Default workspace for events
        """
        self._handlers: dict[SecurityEventType, list[SecurityEventHandler]] = {}
        self._global_handlers: list[SecurityEventHandler] = []
        self._enable_auto_debate = enable_auto_debate
        self._debate_confidence_threshold = debate_confidence_threshold
        self._debate_timeout_seconds = debate_timeout_seconds
        self._workspace_id = workspace_id
        self._pending_debates: dict[str, asyncio.Task] = {}
        self._event_history: list[SecurityEvent] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: SecurityEventType,
        handler: SecurityEventHandler,
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed handler to %s", event_type.value)

    def subscribe_all(self, handler: SecurityEventHandler) -> None:
        """
        Subscribe to all event types.

        Args:
            handler: Async handler function
        """
        self._global_handlers.append(handler)
        logger.debug("Subscribed global handler to all security events")

    def unsubscribe(
        self,
        event_type: SecurityEventType,
        handler: SecurityEventHandler,
    ) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                return True
            except ValueError as e:
                logger.debug("unsubscribe encountered an error: %s", e)
        return False

    async def emit(self, event: SecurityEvent) -> None:
        """
        Emit a security event.

        Notifies all subscribers and optionally triggers a debate
        for critical findings.

        Args:
            event: Security event to emit
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Set workspace if not provided
        if not event.workspace_id and self._workspace_id:
            event.workspace_id = self._workspace_id

        # Notify type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:  # noqa: BLE001 - intentional broad catch for event handler isolation
                logger.warning(
                    "Security event handler failed for %s: %s", event.event_type.value, e
                )

        # Notify global handlers
        for handler in self._global_handlers:
            try:
                await handler(event)
            except Exception as e:  # noqa: BLE001 - intentional broad catch for event handler isolation
                logger.warning("Global security event handler failed: %s", e)

        # Auto-trigger debate for critical findings
        if self._should_trigger_debate(event):
            await self._trigger_security_debate(event)

    def _should_trigger_debate(self, event: SecurityEvent) -> bool:
        """Check if event should trigger an automatic debate."""
        if not self._enable_auto_debate:
            return False

        # Already has a debate
        if event.debate_id or event.debate_requested:
            return False

        # Check severity threshold
        if event.is_critical:
            return True

        # Check for multiple high-severity findings
        if event.high_count >= 3:
            return True

        return False

    async def _trigger_security_debate(self, event: SecurityEvent) -> str | None:
        """
        Trigger a multi-agent debate for remediation recommendations.

        Args:
            event: Security event with findings

        Returns:
            Debate ID if triggered, None otherwise
        """
        try:
            runner = get_security_debate_runner()
            if runner is None:
                logger.warning(
                    "No security debate runner registered; skipping auto-debate for %s. "
                    "A composition root must call "
                    "aragora.events.security_events.register_security_debate_runner() "
                    "(aragora.debate.security_response registers a default runner at "
                    "import time; ensure it or an equivalent composition root has run).",
                    event.id,
                )
                return None

            runner_kwargs = _accepted_security_debate_runner_kwargs(
                runner,
                confidence_threshold=self._debate_confidence_threshold,
                timeout_seconds=self._debate_timeout_seconds,
            )
            if runner_kwargs:
                debate_id = await runner(event, **runner_kwargs)
            else:
                debate_id = await runner(event)

            if debate_id:
                event.debate_requested = True
                event.debate_id = debate_id
                finding_event_metadata = event.metadata if len(event.findings) == 1 else None

                # Emit debate started event
                debate_event = SecurityEvent(
                    event_type=SecurityEventType.SECURITY_DEBATE_STARTED,
                    severity=event.severity,
                    repository=event.repository,
                    scan_id=event.scan_id,
                    workspace_id=event.workspace_id,
                    findings=[
                        redacted_security_finding(f, event_metadata=finding_event_metadata)
                        for f in event.findings
                    ],
                    debate_id=debate_id,
                    correlation_id=event.correlation_id,
                )
                await self.emit(debate_event)

            return debate_id

        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
            logger.exception("Failed to trigger security debate: %s", e)
            return None

    def get_recent_events(
        self,
        event_type: SecurityEventType | None = None,
        severity: SecuritySeverity | None = None,
        limit: int = 100,
    ) -> list[SecurityEvent]:
        """
        Get recent security events with optional filtering.

        Args:
            event_type: Filter by event type
            severity: Filter by minimum severity
            limit: Maximum events to return

        Returns:
            List of matching events (newest first)
        """
        events = self._event_history.copy()
        events.reverse()  # Newest first

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            severity_order = {
                SecuritySeverity.CRITICAL: 0,
                SecuritySeverity.HIGH: 1,
                SecuritySeverity.MEDIUM: 2,
                SecuritySeverity.LOW: 3,
                SecuritySeverity.INFO: 4,
            }
            max_order = severity_order.get(severity, 4)
            events = [e for e in events if severity_order.get(e.severity, 4) <= max_order]

        return events[:limit]

    def get_pending_debates(self) -> dict[str, asyncio.Task]:
        """Get currently pending security debates."""
        return {k: v for k, v in self._pending_debates.items() if not v.done()}


# =============================================================================
# Debate Integration
# =============================================================================


# Storage for debate results (in-memory, replace with database in production)
_security_debate_results: dict[str, dict[str, Any]] = {}


async def _store_security_debate_result(
    debate_id: str,
    event: SecurityEvent,
    result: Any,
) -> None:
    """Store security debate result for later retrieval."""
    _security_debate_results[debate_id] = {
        "debate_id": debate_id,
        "event_id": event.id,
        "repository": event.repository,
        "findings_count": len(event.findings),
        "consensus_reached": getattr(result, "consensus_reached", False),
        "confidence": getattr(result, "confidence", 0.0),
        "final_answer": getattr(result, "final_answer", ""),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_security_debate_result(debate_id: str) -> dict[str, Any] | None:
    """Get a security debate result by ID."""
    return _security_debate_results.get(debate_id)


async def list_security_debates(
    repository: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List security debate results."""
    results = list(_security_debate_results.values())

    if repository:
        results = [r for r in results if r.get("repository") == repository]

    # Sort by completion time descending
    results.sort(key=lambda r: r.get("completed_at", ""), reverse=True)

    return results[:limit]


# =============================================================================
# Convenience Functions
# =============================================================================


def create_vulnerability_event(
    vulnerability: dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> SecurityEvent:
    """
    Create a security event from a vulnerability finding.

    Args:
        vulnerability: Vulnerability data from scanner
        repository: Repository identifier
        scan_id: Scan identifier
        workspace_id: Optional workspace ID

    Returns:
        SecurityEvent ready for emission
    """
    severity_map = {
        "critical": SecuritySeverity.CRITICAL,
        "high": SecuritySeverity.HIGH,
        "medium": SecuritySeverity.MEDIUM,
        "low": SecuritySeverity.LOW,
    }
    severity = severity_map.get(
        vulnerability.get("severity", "").lower(),
        SecuritySeverity.MEDIUM,
    )

    finding = SecurityFinding(
        id=vulnerability.get("id", str(uuid.uuid4())),
        finding_type="vulnerability",
        severity=severity,
        title=vulnerability.get("title", vulnerability.get("cve_id", "Unknown")),
        description=vulnerability.get("description", ""),
        cve_id=vulnerability.get("cve_id"),
        package_name=vulnerability.get("package_name"),
        package_version=vulnerability.get("package_version"),
        recommendation=vulnerability.get("recommendation"),
        metadata=vulnerability,
    )

    event_type = (
        SecurityEventType.CRITICAL_VULNERABILITY
        if severity == SecuritySeverity.CRITICAL
        else SecurityEventType.VULNERABILITY_DETECTED
    )

    return SecurityEvent(
        event_type=event_type,
        severity=severity,
        repository=repository,
        scan_id=scan_id,
        workspace_id=workspace_id,
        findings=[finding],
    )


def create_secret_event(
    secret: dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> SecurityEvent:
    """
    Create a security event from a secret finding.

    Args:
        secret: Secret finding data from scanner
        repository: Repository identifier
        scan_id: Scan identifier
        workspace_id: Optional workspace ID

    Returns:
        SecurityEvent ready for emission
    """
    severity_map = {
        "critical": SecuritySeverity.CRITICAL,
        "high": SecuritySeverity.HIGH,
        "medium": SecuritySeverity.MEDIUM,
        "low": SecuritySeverity.LOW,
    }
    severity = severity_map.get(
        secret.get("severity", "").lower(),
        SecuritySeverity.HIGH,
    )

    finding = SecurityFinding(
        id=secret.get("id", str(uuid.uuid4())),
        finding_type="secret",
        severity=severity,
        title=f"Exposed {secret.get('secret_type', 'secret')}",
        description=secret.get("description", "Hardcoded credential detected"),
        file_path=secret.get("file_path"),
        line_number=secret.get("line_number"),
        recommendation="Rotate the credential immediately and remove from codebase",
        metadata=secret,
    )

    event_type = (
        SecurityEventType.CRITICAL_SECRET
        if severity == SecuritySeverity.CRITICAL
        else SecurityEventType.SECRET_DETECTED
    )

    return SecurityEvent(
        event_type=event_type,
        severity=severity,
        repository=repository,
        scan_id=scan_id,
        workspace_id=workspace_id,
        findings=[finding],
    )


def create_scan_completed_event(
    scan_result: dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> SecurityEvent:
    """
    Create a scan completed event with findings summary.

    Args:
        scan_result: Complete scan result
        repository: Repository identifier
        scan_id: Scan identifier
        workspace_id: Optional workspace ID

    Returns:
        SecurityEvent for scan completion
    """
    # Determine overall severity
    critical_count = scan_result.get("critical_count", 0)
    high_count = scan_result.get("high_count", 0)

    if critical_count > 0:
        severity = SecuritySeverity.CRITICAL
    elif high_count > 0:
        severity = SecuritySeverity.HIGH
    else:
        severity = SecuritySeverity.MEDIUM

    # Build findings list from scan result
    findings = []
    for vuln in scan_result.get("vulnerabilities", [])[:10]:  # Limit to top 10
        findings.append(
            SecurityFinding(
                id=vuln.get("id", str(uuid.uuid4())),
                finding_type="vulnerability",
                severity=SecuritySeverity(vuln.get("severity", "medium")),
                title=vuln.get("title", vuln.get("cve_id", "Unknown")),
                description=vuln.get("description", ""),
                cve_id=vuln.get("cve_id"),
                package_name=vuln.get("package_name"),
                package_version=vuln.get("package_version"),
            )
        )

    return SecurityEvent(
        event_type=SecurityEventType.SCAN_COMPLETED,
        severity=severity,
        repository=repository,
        scan_id=scan_id,
        workspace_id=workspace_id,
        findings=findings,
    )


# =============================================================================
# Singleton Instance
# =============================================================================

_default_emitter: SecurityEventEmitter | None = None


def get_security_emitter() -> SecurityEventEmitter:
    """Get the default security event emitter instance."""
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = SecurityEventEmitter()
    return _default_emitter


def set_security_emitter(emitter: SecurityEventEmitter) -> None:
    """Set the default security event emitter instance."""
    global _default_emitter
    _default_emitter = emitter


__all__ = [
    # Event types
    "SecurityEventType",
    "SecuritySeverity",
    "SecurityFinding",
    "SecurityEvent",
    # Emitter
    "SecurityEventEmitter",
    "SecurityEventHandler",
    "get_security_emitter",
    "set_security_emitter",
    # Debate integration (domain-free hook; runner lives in aragora.debate.security_response)
    "SecurityDebateRunner",
    "register_security_debate_runner",
    "get_security_debate_runner",
    "get_security_debate_result",
    "list_security_debates",
    # Convenience functions
    "create_vulnerability_event",
    "create_secret_event",
    "create_scan_completed_event",
]
