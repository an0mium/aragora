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
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

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
    SECRET_DETECTED = "secret_detected"
    CRITICAL_SECRET = "critical_secret"
    SECRET_ROTATED = "secret_rotated"

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


@dataclass
class SecurityFinding:
    """Represents a security finding that may trigger a debate."""

    id: str
    finding_type: str  # "vulnerability", "secret", "misconfiguration"
    severity: SecuritySeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    package_name: Optional[str] = None
    package_version: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
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


@dataclass
class SecurityEvent:
    """Represents a security event with context for debate triggering."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.VULNERABILITY_DETECTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: SecuritySeverity = SecuritySeverity.MEDIUM

    # Source information - categorizes the origin of the event
    source: str = "sast"  # "sast", "secrets", "dependency", "threat_intel"
    repository: Optional[str] = None
    scan_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Findings
    findings: List[SecurityFinding] = field(default_factory=list)

    # Debate context
    debate_requested: bool = False
    debate_id: Optional[str] = None
    debate_question: Optional[str] = None

    # Correlation
    correlation_id: Optional[str] = None

    # Metadata for additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source": self.source,
            "repository": self.repository,
            "scan_id": self.scan_id,
            "workspace_id": self.workspace_id,
            "findings": [f.to_dict() for f in self.findings],
            "debate_requested": self.debate_requested,
            "debate_id": self.debate_id,
            "debate_question": self.debate_question,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
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
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize the security event emitter.

        Args:
            enable_auto_debate: Whether to auto-trigger debates for critical findings
            debate_confidence_threshold: Minimum confidence for debate consensus
            workspace_id: Default workspace for events
        """
        self._handlers: Dict[SecurityEventType, List[SecurityEventHandler]] = {}
        self._global_handlers: List[SecurityEventHandler] = []
        self._enable_auto_debate = enable_auto_debate
        self._debate_confidence_threshold = debate_confidence_threshold
        self._workspace_id = workspace_id
        self._pending_debates: Dict[str, asyncio.Task] = {}
        self._event_history: List[SecurityEvent] = []
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
        logger.debug(f"Subscribed handler to {event_type.value}")

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
            except ValueError:
                pass
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
            except Exception as e:
                logger.warning(f"Security event handler failed for {event.event_type.value}: {e}")

        # Notify global handlers
        for handler in self._global_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.warning(f"Global security event handler failed: {e}")

        # Auto-trigger debate for critical findings
        if self._should_trigger_debate(event):
            await self._trigger_security_debate(event)

    def _should_trigger_debate(self, event: SecurityEvent) -> bool:
        """Check if event should trigger an automatic debate."""
        if not self._enable_auto_debate:
            return False

        # Already has a debate
        if event.debate_id:
            return False

        # Check severity threshold
        if event.is_critical:
            return True

        # Check for multiple high-severity findings
        if event.high_count >= 3:
            return True

        return False

    async def _trigger_security_debate(self, event: SecurityEvent) -> Optional[str]:
        """
        Trigger a multi-agent debate for remediation recommendations.

        Args:
            event: Security event with findings

        Returns:
            Debate ID if triggered, None otherwise
        """
        try:
            debate_id = await trigger_security_debate(
                event=event,
                confidence_threshold=self._debate_confidence_threshold,
            )

            if debate_id:
                event.debate_requested = True
                event.debate_id = debate_id

                # Emit debate started event
                debate_event = SecurityEvent(
                    event_type=SecurityEventType.SECURITY_DEBATE_STARTED,
                    severity=event.severity,
                    repository=event.repository,
                    scan_id=event.scan_id,
                    workspace_id=event.workspace_id,
                    findings=event.findings,
                    debate_id=debate_id,
                    correlation_id=event.correlation_id,
                )
                await self.emit(debate_event)

            return debate_id

        except Exception as e:
            logger.exception(f"Failed to trigger security debate: {e}")
            return None

    def get_recent_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecuritySeverity] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
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

    def get_pending_debates(self) -> Dict[str, asyncio.Task]:
        """Get currently pending security debates."""
        return {k: v for k, v in self._pending_debates.items() if not v.done()}


# =============================================================================
# Debate Integration
# =============================================================================


def build_security_debate_question(event: SecurityEvent) -> str:
    """
    Build a debate question from security findings.

    Args:
        event: Security event with findings

    Returns:
        Formatted debate question
    """
    findings = event.findings[:5]  # Limit to top 5 findings

    if not findings:
        return f"Analyze and recommend remediation for security findings in {event.repository or 'the codebase'}."

    # Group by type
    vulns = [f for f in findings if f.finding_type == "vulnerability"]
    secrets = [f for f in findings if f.finding_type == "secret"]

    question_parts = []

    if vulns:
        vuln_summary = ", ".join(f"{v.cve_id or v.title} in {v.package_name}" for v in vulns[:3])
        question_parts.append(f"vulnerabilities ({vuln_summary})")

    if secrets:
        secret_types = set(s.metadata.get("secret_type", "unknown") for s in secrets)
        question_parts.append(f"exposed secrets ({', '.join(secret_types)})")

    findings_str = " and ".join(question_parts)

    return (
        f"Analyze the following critical security findings and provide remediation recommendations:\n\n"
        f"Repository: {event.repository or 'Unknown'}\n"
        f"Findings: {findings_str}\n\n"
        f"Details:\n"
        + "\n".join(
            f"- {f.severity.value.upper()}: {f.title} - {f.description[:200]}" for f in findings
        )
        + "\n\n"
        "What is the recommended prioritized remediation plan, considering:\n"
        "1. Immediate mitigations (quick wins)\n"
        "2. Root cause fixes\n"
        "3. Preventive measures for future\n"
        "4. Impact on existing functionality"
    )


async def trigger_security_debate(
    event: SecurityEvent,
    confidence_threshold: float = 0.7,
    agents: Optional[List[Any]] = None,
    timeout_seconds: int = 300,
) -> Optional[str]:
    """
    Trigger a multi-agent debate for security remediation.

    Args:
        event: Security event with findings
        confidence_threshold: Minimum consensus confidence
        agents: Optional list of agents (uses defaults if None)
        timeout_seconds: Maximum debate duration

    Returns:
        Debate ID if triggered, None if failed
    """
    try:
        from aragora.core import Environment, DebateResult
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        # Build debate question
        question = build_security_debate_question(event)
        event.debate_question = question

        # Create environment
        env = Environment(
            task=question,
            context={  # type: ignore[arg-type]
                "security_event_id": event.id,
                "repository": event.repository,
                "scan_id": event.scan_id,
                "findings": [f.to_dict() for f in event.findings],
            },
        )

        # Create protocol for security debates
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            convergence_detection=True,
            convergence_threshold=0.85,
            timeout_seconds=timeout_seconds,
        )

        # Get default agents if none provided
        if agents is None:
            agents = await _get_security_debate_agents()

        if not agents:
            logger.warning("No agents available for security debate")
            return None

        # Generate debate ID
        debate_id = f"security_debate_{uuid.uuid4().hex[:12]}"

        # Run debate
        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            org_id=event.workspace_id or "default",
        )

        logger.info(f"[Security] Starting debate {debate_id} for {len(event.findings)} findings")

        result: DebateResult = await arena.run()

        logger.info(
            f"[Security] Debate {debate_id} completed: "
            f"consensus={result.consensus_reached}, confidence={result.confidence:.2f}"
        )

        # Store result for later retrieval
        await _store_security_debate_result(debate_id, event, result)

        return debate_id

    except ImportError as e:
        logger.warning(f"Arena not available for security debate: {e}")
        return None
    except Exception as e:
        logger.exception(f"Failed to run security debate: {e}")
        return None


async def _get_security_debate_agents() -> List[Any]:
    """Get agents suitable for security debates."""
    try:
        from aragora.agents.factory import get_available_agents

        # Get available agents with security expertise preference
        agents = await get_available_agents(
            capabilities=["security", "code_analysis"],
            min_count=2,
            max_count=4,
        )
        return agents
    except ImportError:
        # Fall back to basic agent creation
        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent as AnthropicAgent
            from aragora.agents.api_agents.openai import OpenAIAPIAgent as OpenAIAgent

            agents = []

            try:
                agents.append(
                    AnthropicAgent(
                        name="claude-security",
                        model="claude-sonnet-4-20250514",
                    )
                )
            except Exception:
                pass

            try:
                agents.append(
                    OpenAIAgent(
                        name="gpt-security",
                        model="gpt-4o",
                    )
                )
            except Exception:
                pass

            return agents
        except ImportError:
            return []
    except Exception:
        return []


# Storage for debate results (in-memory, replace with database in production)
_security_debate_results: Dict[str, Dict[str, Any]] = {}


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


async def get_security_debate_result(debate_id: str) -> Optional[Dict[str, Any]]:
    """Get a security debate result by ID."""
    return _security_debate_results.get(debate_id)


async def list_security_debates(
    repository: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
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
    vulnerability: Dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
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
    secret: Dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
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
    scan_result: Dict[str, Any],
    repository: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
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

_default_emitter: Optional[SecurityEventEmitter] = None


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
    # Debate integration
    "trigger_security_debate",
    "build_security_debate_question",
    "get_security_debate_result",
    "list_security_debates",
    # Convenience functions
    "create_vulnerability_event",
    "create_secret_event",
    "create_scan_completed_event",
]
