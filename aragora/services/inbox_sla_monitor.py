"""
Inbox SLA Monitor - SLA Tracking and Alerting for Shared Inboxes.

Provides SLA monitoring capabilities:
- Track response time SLAs
- Track resolution time SLAs
- Detect at-risk messages approaching SLA breach
- Trigger escalation on SLA violations
- Calculate SLA compliance metrics

Usage:
    from aragora.services.inbox_sla_monitor import get_sla_monitor

    monitor = get_sla_monitor()
    violations = await monitor.check_sla_compliance(inbox_id)
    at_risk = await monitor.get_at_risk_messages(inbox_id, threshold_minutes=15)
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SLAViolationType(str, Enum):
    """Types of SLA violations."""

    FIRST_RESPONSE = "first_response"
    RESOLUTION = "resolution"


class EscalationLevel(str, Enum):
    """Escalation severity levels."""

    WARNING = "warning"  # Approaching breach
    BREACH = "breach"  # SLA breached
    CRITICAL = "critical"  # Extended breach


@dataclass
class EscalationRule:
    """An escalation rule for SLA violations."""

    id: str
    level: EscalationLevel
    threshold_minutes: int  # Minutes before/after SLA to trigger
    notify_channels: List[str]  # ["email", "slack", "webhook"]
    notify_users: List[str]  # User IDs to notify
    reassign_to: Optional[str] = None  # Auto-reassign on breach

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "threshold_minutes": self.threshold_minutes,
            "notify_channels": self.notify_channels,
            "notify_users": self.notify_users,
            "reassign_to": self.reassign_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationRule":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            level=EscalationLevel(data["level"]),
            threshold_minutes=data["threshold_minutes"],
            notify_channels=data.get("notify_channels", []),
            notify_users=data.get("notify_users", []),
            reassign_to=data.get("reassign_to"),
        )


@dataclass
class SLAConfig:
    """SLA configuration for a shared inbox."""

    inbox_id: str
    org_id: str
    response_time_minutes: int = 60  # First response SLA (1 hour default)
    resolution_time_minutes: int = 480  # Full resolution SLA (8 hours default)
    escalation_rules: List[EscalationRule] = field(default_factory=list)
    enabled: bool = True
    business_hours_only: bool = False  # If True, only count business hours
    business_hours_start: int = 9  # 9 AM
    business_hours_end: int = 17  # 5 PM
    business_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inbox_id": self.inbox_id,
            "org_id": self.org_id,
            "response_time_minutes": self.response_time_minutes,
            "resolution_time_minutes": self.resolution_time_minutes,
            "escalation_rules": [r.to_dict() for r in self.escalation_rules],
            "enabled": self.enabled,
            "business_hours_only": self.business_hours_only,
            "business_hours_start": self.business_hours_start,
            "business_hours_end": self.business_hours_end,
            "business_days": self.business_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLAConfig":
        rules = [EscalationRule.from_dict(r) for r in data.get("escalation_rules", [])]
        return cls(
            inbox_id=data["inbox_id"],
            org_id=data["org_id"],
            response_time_minutes=data.get("response_time_minutes", 60),
            resolution_time_minutes=data.get("resolution_time_minutes", 480),
            escalation_rules=rules,
            enabled=data.get("enabled", True),
            business_hours_only=data.get("business_hours_only", False),
            business_hours_start=data.get("business_hours_start", 9),
            business_hours_end=data.get("business_hours_end", 17),
            business_days=data.get("business_days", [0, 1, 2, 3, 4]),
        )


@dataclass
class SLAViolation:
    """A detected SLA violation."""

    id: str
    inbox_id: str
    message_id: str
    violation_type: SLAViolationType
    sla_minutes: int  # What the SLA was
    actual_minutes: int  # How long it actually took/has taken
    breached_at: datetime
    escalation_level: EscalationLevel
    escalation_triggered: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "inbox_id": self.inbox_id,
            "message_id": self.message_id,
            "violation_type": self.violation_type.value,
            "sla_minutes": self.sla_minutes,
            "actual_minutes": self.actual_minutes,
            "breached_at": self.breached_at.isoformat(),
            "escalation_level": self.escalation_level.value,
            "escalation_triggered": self.escalation_triggered,
            "resolved": self.resolved,
        }


@dataclass
class AtRiskMessage:
    """A message at risk of SLA breach."""

    message_id: str
    inbox_id: str
    subject: str
    received_at: datetime
    sla_deadline: datetime
    minutes_remaining: int
    risk_type: SLAViolationType
    assigned_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "inbox_id": self.inbox_id,
            "subject": self.subject,
            "received_at": self.received_at.isoformat(),
            "sla_deadline": self.sla_deadline.isoformat(),
            "minutes_remaining": self.minutes_remaining,
            "risk_type": self.risk_type.value,
            "assigned_to": self.assigned_to,
        }


@dataclass
class SLAMetrics:
    """SLA compliance metrics for an inbox."""

    inbox_id: str
    period_start: datetime
    period_end: datetime
    total_messages: int
    response_sla_met: int
    response_sla_breached: int
    resolution_sla_met: int
    resolution_sla_breached: int
    avg_response_time_minutes: float
    avg_resolution_time_minutes: float
    response_compliance_rate: float  # 0.0 to 1.0
    resolution_compliance_rate: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inbox_id": self.inbox_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_messages": self.total_messages,
            "response_sla_met": self.response_sla_met,
            "response_sla_breached": self.response_sla_breached,
            "resolution_sla_met": self.resolution_sla_met,
            "resolution_sla_breached": self.resolution_sla_breached,
            "avg_response_time_minutes": round(self.avg_response_time_minutes, 2),
            "avg_resolution_time_minutes": round(self.avg_resolution_time_minutes, 2),
            "response_compliance_rate": round(self.response_compliance_rate, 4),
            "resolution_compliance_rate": round(self.resolution_compliance_rate, 4),
        }


class InboxSLAMonitor:
    """
    SLA monitoring and alerting service for shared inboxes.

    Tracks SLA compliance, detects violations, and triggers escalations.
    """

    def __init__(self):
        """Initialize the SLA monitor."""
        self._configs: Dict[str, SLAConfig] = {}  # inbox_id -> config
        self._violations: Dict[str, List[SLAViolation]] = {}  # inbox_id -> violations
        self._lock = threading.Lock()
        self._escalation_handlers: List[Callable] = []

    # =========================================================================
    # Configuration Management
    # =========================================================================

    def set_config(self, config: SLAConfig) -> None:
        """Set SLA configuration for an inbox."""
        with self._lock:
            self._configs[config.inbox_id] = config
        logger.info(f"[SLAMonitor] Set SLA config for inbox {config.inbox_id}")

    def get_config(self, inbox_id: str) -> Optional[SLAConfig]:
        """Get SLA configuration for an inbox."""
        with self._lock:
            return self._configs.get(inbox_id)

    def delete_config(self, inbox_id: str) -> bool:
        """Delete SLA configuration for an inbox."""
        with self._lock:
            if inbox_id in self._configs:
                del self._configs[inbox_id]
                logger.info(f"[SLAMonitor] Deleted SLA config for inbox {inbox_id}")
                return True
            return False

    def register_escalation_handler(
        self, handler: Callable[[SLAViolation, SLAConfig], None]
    ) -> None:
        """Register a callback for escalation events."""
        self._escalation_handlers.append(handler)

    # =========================================================================
    # SLA Checking
    # =========================================================================

    async def check_sla_compliance(
        self,
        inbox_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SLAViolation]:
        """
        Check SLA compliance for messages in an inbox.

        Args:
            inbox_id: Inbox to check
            messages: List of message dicts with received_at, first_response_at,
                     resolved_at, status fields

        Returns:
            List of SLA violations detected
        """
        config = self.get_config(inbox_id)
        if not config or not config.enabled:
            return []

        if messages is None:
            messages = await self._get_inbox_messages(inbox_id)

        violations = []
        now = datetime.now(timezone.utc)

        for msg in messages:
            msg_id = msg.get("id", msg.get("message_id", ""))
            received_at = self._parse_datetime(msg.get("received_at"))
            first_response_at = self._parse_datetime(msg.get("first_response_at"))
            status = msg.get("status", "open")

            if not received_at:
                continue

            # Check first response SLA
            if status not in ("resolved", "closed") and not first_response_at:
                elapsed_minutes = int((now - received_at).total_seconds() / 60)
                if elapsed_minutes > config.response_time_minutes:
                    violation = SLAViolation(
                        id=str(uuid.uuid4()),
                        inbox_id=inbox_id,
                        message_id=msg_id,
                        violation_type=SLAViolationType.FIRST_RESPONSE,
                        sla_minutes=config.response_time_minutes,
                        actual_minutes=elapsed_minutes,
                        breached_at=received_at + timedelta(minutes=config.response_time_minutes),
                        escalation_level=self._determine_escalation_level(
                            elapsed_minutes, config.response_time_minutes
                        ),
                    )
                    violations.append(violation)

            # Check resolution SLA
            if status not in ("resolved", "closed"):
                elapsed_minutes = int((now - received_at).total_seconds() / 60)
                if elapsed_minutes > config.resolution_time_minutes:
                    violation = SLAViolation(
                        id=str(uuid.uuid4()),
                        inbox_id=inbox_id,
                        message_id=msg_id,
                        violation_type=SLAViolationType.RESOLUTION,
                        sla_minutes=config.resolution_time_minutes,
                        actual_minutes=elapsed_minutes,
                        breached_at=received_at + timedelta(minutes=config.resolution_time_minutes),
                        escalation_level=self._determine_escalation_level(
                            elapsed_minutes, config.resolution_time_minutes
                        ),
                    )
                    violations.append(violation)

        # Store violations
        with self._lock:
            self._violations[inbox_id] = violations

        return violations

    async def get_at_risk_messages(
        self,
        inbox_id: str,
        threshold_minutes: int = 15,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AtRiskMessage]:
        """
        Get messages that are approaching SLA breach.

        Args:
            inbox_id: Inbox to check
            threshold_minutes: Minutes before breach to consider "at risk"
            messages: Optional list of message dicts

        Returns:
            List of at-risk messages
        """
        config = self.get_config(inbox_id)
        if not config or not config.enabled:
            return []

        if messages is None:
            messages = await self._get_inbox_messages(inbox_id)

        at_risk = []
        now = datetime.now(timezone.utc)

        for msg in messages:
            msg_id = msg.get("id", msg.get("message_id", ""))
            received_at = self._parse_datetime(msg.get("received_at"))
            first_response_at = self._parse_datetime(msg.get("first_response_at"))
            status = msg.get("status", "open")

            if not received_at or status in ("resolved", "closed"):
                continue

            # Check response SLA risk
            if not first_response_at:
                response_deadline = received_at + timedelta(minutes=config.response_time_minutes)
                minutes_remaining = int((response_deadline - now).total_seconds() / 60)

                if 0 < minutes_remaining <= threshold_minutes:
                    at_risk.append(
                        AtRiskMessage(
                            message_id=msg_id,
                            inbox_id=inbox_id,
                            subject=msg.get("subject", ""),
                            received_at=received_at,
                            sla_deadline=response_deadline,
                            minutes_remaining=minutes_remaining,
                            risk_type=SLAViolationType.FIRST_RESPONSE,
                            assigned_to=msg.get("assigned_to"),
                        )
                    )

            # Check resolution SLA risk
            resolution_deadline = received_at + timedelta(minutes=config.resolution_time_minutes)
            minutes_remaining = int((resolution_deadline - now).total_seconds() / 60)

            if 0 < minutes_remaining <= threshold_minutes:
                at_risk.append(
                    AtRiskMessage(
                        message_id=msg_id,
                        inbox_id=inbox_id,
                        subject=msg.get("subject", ""),
                        received_at=received_at,
                        sla_deadline=resolution_deadline,
                        minutes_remaining=minutes_remaining,
                        risk_type=SLAViolationType.RESOLUTION,
                        assigned_to=msg.get("assigned_to"),
                    )
                )

        # Sort by minutes remaining (most urgent first)
        at_risk.sort(key=lambda x: x.minutes_remaining)
        return at_risk

    async def trigger_escalation(
        self,
        violation: SLAViolation,
    ) -> bool:
        """
        Trigger escalation for an SLA violation.

        Args:
            violation: The SLA violation to escalate

        Returns:
            True if escalation was triggered
        """
        config = self.get_config(violation.inbox_id)
        if not config:
            return False

        # Find applicable escalation rules
        applicable_rules = [
            r for r in config.escalation_rules if r.level == violation.escalation_level
        ]

        if not applicable_rules:
            logger.debug(f"[SLAMonitor] No escalation rules for level {violation.escalation_level}")
            return False

        # Mark as triggered
        violation.escalation_triggered = True

        # Call registered handlers
        for handler in self._escalation_handlers:
            try:
                handler(violation, config)
            except Exception as e:
                logger.warning(f"[SLAMonitor] Escalation handler failed: {e}")

        # Log activity
        try:
            from aragora.storage.inbox_activity_store import (
                InboxActivity,
                InboxActivityAction,
                get_inbox_activity_store,
            )

            store = get_inbox_activity_store()
            activity = InboxActivity(
                inbox_id=violation.inbox_id,
                org_id=config.org_id,
                actor_id="system",
                action=InboxActivityAction.SLA_BREACHED,
                target_id=violation.message_id,
                metadata={
                    "violation_type": violation.violation_type.value,
                    "sla_minutes": violation.sla_minutes,
                    "actual_minutes": violation.actual_minutes,
                    "escalation_level": violation.escalation_level.value,
                },
            )
            store.log_activity(activity)
        except Exception as e:
            logger.debug(f"[SLAMonitor] Failed to log activity: {e}")

        logger.info(
            f"[SLAMonitor] Triggered {violation.escalation_level.value} escalation "
            f"for message {violation.message_id}"
        )
        return True

    # =========================================================================
    # Metrics
    # =========================================================================

    async def get_sla_metrics(
        self,
        inbox_id: str,
        period_days: int = 7,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> SLAMetrics:
        """
        Get SLA compliance metrics for an inbox.

        Args:
            inbox_id: Inbox to get metrics for
            period_days: Number of days to analyze
            messages: Optional list of resolved message dicts

        Returns:
            SLA metrics for the period
        """
        config = self.get_config(inbox_id)
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=period_days)

        if messages is None:
            messages = await self._get_inbox_messages(
                inbox_id, status="resolved", since=period_start
            )

        total = 0
        response_met = 0
        response_breached = 0
        resolution_met = 0
        resolution_breached = 0
        response_times: List[float] = []
        resolution_times: List[float] = []

        for msg in messages:
            received_at = self._parse_datetime(msg.get("received_at"))
            first_response_at = self._parse_datetime(msg.get("first_response_at"))
            resolved_at = self._parse_datetime(msg.get("resolved_at"))

            if not received_at:
                continue

            total += 1

            # Response time metrics
            if first_response_at:
                response_minutes = (first_response_at - received_at).total_seconds() / 60
                response_times.append(response_minutes)
                if config and response_minutes <= config.response_time_minutes:
                    response_met += 1
                else:
                    response_breached += 1

            # Resolution time metrics
            if resolved_at:
                resolution_minutes = (resolved_at - received_at).total_seconds() / 60
                resolution_times.append(resolution_minutes)
                if config and resolution_minutes <= config.resolution_time_minutes:
                    resolution_met += 1
                else:
                    resolution_breached += 1

        # Calculate averages
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0

        # Calculate compliance rates
        response_total = response_met + response_breached
        resolution_total = resolution_met + resolution_breached

        response_rate = response_met / response_total if response_total > 0 else 1.0
        resolution_rate = resolution_met / resolution_total if resolution_total > 0 else 1.0

        return SLAMetrics(
            inbox_id=inbox_id,
            period_start=period_start,
            period_end=period_end,
            total_messages=total,
            response_sla_met=response_met,
            response_sla_breached=response_breached,
            resolution_sla_met=resolution_met,
            resolution_sla_breached=resolution_breached,
            avg_response_time_minutes=avg_response,
            avg_resolution_time_minutes=avg_resolution,
            response_compliance_rate=response_rate,
            resolution_compliance_rate=resolution_rate,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _determine_escalation_level(
        self,
        actual_minutes: int,
        sla_minutes: int,
    ) -> EscalationLevel:
        """Determine escalation level based on how much SLA is exceeded."""
        overage = actual_minutes - sla_minutes
        if overage <= 0:
            return EscalationLevel.WARNING
        elif overage <= sla_minutes * 0.5:  # Up to 50% over
            return EscalationLevel.BREACH
        else:  # More than 50% over
            return EscalationLevel.CRITICAL

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return None

    def _parse_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        """Parse timestamp from string (ISO date or unix timestamp)."""
        if not value:
            return None

        try:
            # Try as unix timestamp
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            pass

        try:
            # Try as ISO date
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

        return None

    async def _get_inbox_messages(
        self,
        inbox_id: str,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get messages from inbox store.

        Args:
            inbox_id: The inbox to fetch messages from
            status: Optional status filter (e.g., "resolved" for metrics)
            since: Optional datetime filter for messages received after this time

        Returns:
            List of message dicts with id, received_at, status, etc.
        """
        try:
            from aragora.storage.email_store import get_email_store

            store = get_email_store()
            if store is None:
                logger.debug("[SLAMonitor] Email store not available")
                return []

            # Fetch messages with optional status filter
            # Use a high limit since we may need to filter by date
            messages = store.list_inbox_messages(
                inbox_id=inbox_id,
                status=status,
                limit=1000,
                offset=0,
            )

            # Filter by since timestamp if provided
            if since is not None:
                filtered = []
                for msg in messages:
                    received_at = self._parse_timestamp(msg.get("received_at"))
                    if received_at and received_at >= since:
                        filtered.append(msg)
                messages = filtered

            return messages
        except Exception as e:
            logger.debug(f"[SLAMonitor] Failed to get messages: {e}")
            return []


# Module-level singleton
_default_monitor: Optional[InboxSLAMonitor] = None
_monitor_lock = threading.Lock()


def get_sla_monitor() -> InboxSLAMonitor:
    """Get or create the default SLA monitor instance."""
    global _default_monitor

    if _default_monitor is None:
        with _monitor_lock:
            if _default_monitor is None:
                _default_monitor = InboxSLAMonitor()
                logger.info("[SLAMonitor] Initialized inbox SLA monitor")

    return _default_monitor


def reset_sla_monitor() -> None:
    """Reset the default monitor instance (for testing)."""
    global _default_monitor
    with _monitor_lock:
        _default_monitor = None


__all__ = [
    "InboxSLAMonitor",
    "SLAConfig",
    "SLAViolation",
    "SLAViolationType",
    "AtRiskMessage",
    "SLAMetrics",
    "EscalationRule",
    "EscalationLevel",
    "get_sla_monitor",
    "reset_sla_monitor",
]
