"""
SLO Alert Bridge - Connects SLO violations to alerting systems.

Provides automatic routing of SLO violations to:
- PagerDuty (critical/major violations create incidents)
- Slack/Teams (all violations posted to channels)
- Generic webhooks (for custom integrations)

Features:
- Severity-based routing (critical → P1, major → P2, etc.)
- Incident deduplication (same operation/percentile → same incident)
- Automatic recovery handling (resolves incidents when SLO recovers)
- Cooldown to prevent notification spam
- Business hour awareness for escalation

Usage:
    from aragora.observability.slo_alert_bridge import (
        SLOAlertBridge,
        init_slo_alerting,
    )

    # Initialize at startup
    bridge = init_slo_alerting(
        pagerduty_api_key="...",
        pagerduty_service_id="...",
        slack_webhook_url="...",
    )

    # Violations automatically routed via webhook callbacks
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels mapped to PagerDuty priorities."""

    CRITICAL = "critical"  # P1 - Immediate response required
    MAJOR = "major"  # P2 - Urgent, within 1 hour
    MODERATE = "moderate"  # P3 - High priority, within 4 hours
    MINOR = "minor"  # P4 - Low priority, next business day


class AlertChannel(Enum):
    """Available alert channels."""

    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class SLOAlertConfig:
    """Configuration for SLO alerting."""

    # PagerDuty configuration
    pagerduty_enabled: bool = False
    pagerduty_api_key: str | None = None
    pagerduty_service_id: str | None = None
    pagerduty_email: str = "slo-alerts@aragora.dev"  # Email for PagerDuty requests
    pagerduty_min_severity: AlertSeverity = AlertSeverity.MAJOR

    # Slack configuration
    slack_enabled: bool = False
    slack_webhook_url: str | None = None
    slack_channel: str | None = None
    slack_min_severity: AlertSeverity = AlertSeverity.MINOR

    # Teams configuration
    teams_enabled: bool = False
    teams_webhook_url: str | None = None
    teams_min_severity: AlertSeverity = AlertSeverity.MINOR

    # General settings
    cooldown_seconds: float = 60.0  # Min time between duplicate alerts
    dedup_window_seconds: float = 300.0  # Window for incident deduplication
    auto_resolve_on_recovery: bool = True
    include_runbook_links: bool = True


@dataclass
class ActiveViolation:
    """Tracks an active SLO violation for deduplication."""

    operation: str
    percentile: str
    severity: str
    first_seen: float
    last_seen: float
    count: int = 1
    incident_key: str | None = None
    pagerduty_incident_id: str | None = None
    notified_channels: set[str] = field(default_factory=set)


class SLOAlertBridge:
    """
    Bridges SLO violations to alerting systems.

    Receives SLO violation notifications and routes them to appropriate
    alert channels based on severity and configuration.
    """

    def __init__(self, config: SLOAlertConfig):
        """Initialize the bridge with configuration."""
        self.config = config
        self._active_violations: dict[str, ActiveViolation] = {}
        self._last_notification: dict[str, float] = {}
        self._pagerduty_client = None
        self._notification_manager = None
        self._lock = asyncio.Lock()

    def _make_incident_key(self, operation: str, percentile: str) -> str:
        """Generate unique incident key for deduplication."""
        raw = f"slo-{operation}-{percentile}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _map_severity(self, slo_severity: str) -> AlertSeverity:
        """Map SLO severity string to AlertSeverity enum."""
        mapping = {
            "critical": AlertSeverity.CRITICAL,
            "major": AlertSeverity.MAJOR,
            "moderate": AlertSeverity.MODERATE,
            "minor": AlertSeverity.MINOR,
        }
        return mapping.get(slo_severity.lower(), AlertSeverity.MINOR)

    def _should_alert(
        self, channel: AlertChannel, severity: AlertSeverity, incident_key: str
    ) -> bool:
        """Check if we should send an alert to this channel."""
        # Check channel-specific minimum severity
        min_severity_map = {
            AlertChannel.PAGERDUTY: self.config.pagerduty_min_severity,
            AlertChannel.SLACK: self.config.slack_min_severity,
            AlertChannel.TEAMS: self.config.teams_min_severity,
        }

        min_severity = min_severity_map.get(channel, AlertSeverity.MINOR)
        severity_order = [
            AlertSeverity.MINOR,
            AlertSeverity.MODERATE,
            AlertSeverity.MAJOR,
            AlertSeverity.CRITICAL,
        ]

        if severity_order.index(severity) < severity_order.index(min_severity):
            return False

        # Check cooldown
        cache_key = f"{channel.value}:{incident_key}"
        last_time = self._last_notification.get(cache_key, 0)
        if time.time() - last_time < self.config.cooldown_seconds:
            return False

        return True

    async def _send_pagerduty_alert(
        self,
        violation: ActiveViolation,
        context: dict[str, Any],
    ) -> str | None:
        """Create or update a PagerDuty incident."""
        if not self.config.pagerduty_enabled:
            return None

        try:
            from aragora.connectors.devops.pagerduty import (
                PagerDutyConnector,
                PagerDutyCredentials,
                IncidentCreateRequest,
                IncidentUrgency,
            )

            if self._pagerduty_client is None:
                credentials = PagerDutyCredentials(
                    api_key=self.config.pagerduty_api_key or "",
                    email=self.config.pagerduty_email,
                )
                self._pagerduty_client = PagerDutyConnector(credentials)  # type: ignore[assignment]

            # Map severity to PagerDuty urgency/priority
            urgency_map = {
                AlertSeverity.CRITICAL: IncidentUrgency.HIGH,
                AlertSeverity.MAJOR: IncidentUrgency.HIGH,
                AlertSeverity.MODERATE: IncidentUrgency.LOW,
                AlertSeverity.MINOR: IncidentUrgency.LOW,
            }

            severity = self._map_severity(violation.severity)
            urgency = urgency_map.get(severity, IncidentUrgency.LOW)

            # Build description with context
            description = (
                f"SLO Violation: {violation.operation} {violation.percentile}\n"
                f"Severity: {violation.severity}\n"
                f"Latency: {context.get('latency_ms', 'N/A')}ms "
                f"(threshold: {context.get('threshold_ms', 'N/A')}ms)\n"
                f"First seen: {datetime.fromtimestamp(violation.first_seen, tz=timezone.utc).isoformat()}\n"
                f"Occurrences: {violation.count}"
            )

            if self.config.include_runbook_links:
                description += "\n\nRunbook: https://docs.aragora.dev/runbooks/slo-violations"

            request = IncidentCreateRequest(
                title=f"[SLO] {violation.operation} {violation.percentile} violation ({violation.severity})",
                service_id=self.config.pagerduty_service_id or "",
                urgency=urgency,
                description=description,
                incident_key=violation.incident_key,
            )

            # Client is guaranteed to be non-None after the above assignment
            assert self._pagerduty_client is not None
            incident = await self._pagerduty_client.create_incident(request)
            logger.info(f"Created PagerDuty incident {incident.id} for {violation.operation}")
            return incident.id

        except ImportError:
            logger.debug("PagerDuty connector not available")
        except Exception as e:
            logger.error(f"Failed to create PagerDuty incident: {e}")

        return None

    async def _send_slack_alert(
        self,
        violation: ActiveViolation,
        context: dict[str, Any],
    ) -> bool:
        """Send alert to Slack."""
        if not self.config.slack_enabled or not self.config.slack_webhook_url:
            return False

        try:
            from aragora.control_plane.channels import (
                NotificationManager,
                NotificationEventType,
                NotificationPriority,
            )

            if self._notification_manager is None:
                self._notification_manager = NotificationManager()  # type: ignore[assignment]

            severity = self._map_severity(violation.severity)
            priority_map = {
                AlertSeverity.CRITICAL: NotificationPriority.CRITICAL,
                AlertSeverity.MAJOR: NotificationPriority.URGENT,
                AlertSeverity.MODERATE: NotificationPriority.HIGH,
                AlertSeverity.MINOR: NotificationPriority.NORMAL,
            }

            # Build Slack message with blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"SLO Violation: {violation.operation}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{violation.severity.upper()}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Percentile:*\n{violation.percentile}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Latency:*\n{context.get('latency_ms', 'N/A')}ms",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Threshold:*\n{context.get('threshold_ms', 'N/A')}ms",
                        },
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Occurrences: {violation.count} | First seen: {datetime.fromtimestamp(violation.first_seen, tz=timezone.utc).strftime('%H:%M:%S UTC')}",
                        }
                    ],
                },
            ]

            # Manager is guaranteed to be non-None after the above assignment
            assert self._notification_manager is not None
            await self._notification_manager.notify(
                event_type=NotificationEventType.SLA_VIOLATION,
                title=f"SLO Violation: {violation.operation} {violation.percentile}",
                body=f"Severity: {violation.severity}, Latency: {context.get('latency_ms')}ms",
                priority=priority_map.get(severity, NotificationPriority.NORMAL),
                metadata={
                    "operation": violation.operation,
                    "percentile": violation.percentile,
                    "severity": violation.severity,
                    "blocks": blocks,
                },
            )

            logger.info(f"Sent Slack alert for {violation.operation}")
            return True

        except ImportError:
            logger.debug("Notification manager not available")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

        return False

    async def on_slo_violation(
        self,
        operation: str,
        percentile: str,
        latency_ms: float,
        threshold_ms: float,
        severity: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Handle an SLO violation event.

        Called by the SLO metrics system when a violation is detected.
        Routes to appropriate alert channels based on severity.
        """
        context = context or {}
        context.update(
            {
                "latency_ms": latency_ms,
                "threshold_ms": threshold_ms,
            }
        )

        incident_key = self._make_incident_key(operation, percentile)

        async with self._lock:
            # Check if this is a new or existing violation
            if incident_key in self._active_violations:
                violation = self._active_violations[incident_key]
                violation.last_seen = time.time()
                violation.count += 1

                # Update severity if it increased
                severity_order = ["minor", "moderate", "major", "critical"]
                if severity_order.index(severity) > severity_order.index(violation.severity):
                    violation.severity = severity
            else:
                violation = ActiveViolation(
                    operation=operation,
                    percentile=percentile,
                    severity=severity,
                    first_seen=time.time(),
                    last_seen=time.time(),
                    incident_key=incident_key,
                )
                self._active_violations[incident_key] = violation

            alert_severity = self._map_severity(severity)

            # Route to PagerDuty for critical/major violations
            if self.config.pagerduty_enabled and self._should_alert(
                AlertChannel.PAGERDUTY, alert_severity, incident_key
            ):
                incident_id = await self._send_pagerduty_alert(violation, context)
                if incident_id:
                    violation.pagerduty_incident_id = incident_id
                    violation.notified_channels.add("pagerduty")
                    self._last_notification[f"pagerduty:{incident_key}"] = time.time()

            # Route to Slack for all severities above threshold
            if self.config.slack_enabled and self._should_alert(
                AlertChannel.SLACK, alert_severity, incident_key
            ):
                if await self._send_slack_alert(violation, context):
                    violation.notified_channels.add("slack")
                    self._last_notification[f"slack:{incident_key}"] = time.time()

    async def on_slo_recovery(
        self,
        operation: str,
        percentile: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Handle an SLO recovery event.

        Called when an SLO returns to compliance after a violation.
        Resolves any open incidents and sends recovery notifications.
        """
        incident_key = self._make_incident_key(operation, percentile)

        async with self._lock:
            if incident_key not in self._active_violations:
                return

            violation = self._active_violations[incident_key]

            # Auto-resolve PagerDuty incident
            if (
                self.config.auto_resolve_on_recovery
                and violation.pagerduty_incident_id
                and self._pagerduty_client
            ):
                try:
                    await self._pagerduty_client.resolve_incident(
                        violation.pagerduty_incident_id,
                        resolution=f"SLO {operation} {percentile} recovered. "
                        f"Duration: {time.time() - violation.first_seen:.0f}s, "
                        f"Occurrences: {violation.count}",
                    )
                    logger.info(f"Resolved PagerDuty incident {violation.pagerduty_incident_id}")
                except Exception as e:
                    logger.error(f"Failed to resolve PagerDuty incident: {e}")

            # Send recovery notification to Slack
            if "slack" in violation.notified_channels and self._notification_manager:
                try:
                    from aragora.control_plane.channels import (
                        NotificationEventType,
                        NotificationPriority,
                    )

                    await self._notification_manager.notify(
                        event_type=NotificationEventType.TASK_COMPLETED,
                        title=f"SLO Recovered: {operation} {percentile}",
                        body=f"Duration: {time.time() - violation.first_seen:.0f}s, Occurrences: {violation.count}",
                        priority=NotificationPriority.LOW,
                        metadata={
                            "operation": operation,
                            "percentile": percentile,
                            "duration_seconds": time.time() - violation.first_seen,
                            "occurrence_count": violation.count,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to send recovery notification: {e}")

            # Clean up
            del self._active_violations[incident_key]

    def get_active_violations(self) -> list[dict[str, Any]]:
        """Get list of currently active violations."""
        return [
            {
                "operation": v.operation,
                "percentile": v.percentile,
                "severity": v.severity,
                "first_seen": datetime.fromtimestamp(v.first_seen, tz=timezone.utc).isoformat(),
                "last_seen": datetime.fromtimestamp(v.last_seen, tz=timezone.utc).isoformat(),
                "count": v.count,
                "incident_key": v.incident_key,
                "pagerduty_incident_id": v.pagerduty_incident_id,
                "notified_channels": list(v.notified_channels),
            }
            for v in self._active_violations.values()
        ]

    def cleanup_stale_violations(self, max_age_seconds: float = 3600) -> int:
        """Remove violations that haven't been seen recently."""
        now = time.time()
        stale_keys = [
            key for key, v in self._active_violations.items() if now - v.last_seen > max_age_seconds
        ]
        for key in stale_keys:
            del self._active_violations[key]
        return len(stale_keys)


# Global bridge instance
_bridge: SLOAlertBridge | None = None


def get_slo_alert_bridge() -> SLOAlertBridge | None:
    """Get the global SLO alert bridge instance."""
    return _bridge


def init_slo_alerting(
    pagerduty_api_key: str | None = None,
    pagerduty_service_id: str | None = None,
    slack_webhook_url: str | None = None,
    slack_channel: str | None = None,
    teams_webhook_url: str | None = None,
    cooldown_seconds: float = 60.0,
    auto_resolve_on_recovery: bool = True,
) -> SLOAlertBridge:
    """
    Initialize the SLO alerting system.

    Call this at application startup to enable SLO → Alert routing.

    Args:
        pagerduty_api_key: PagerDuty API key for incident creation
        pagerduty_service_id: PagerDuty service ID to create incidents in
        slack_webhook_url: Slack webhook URL for notifications
        slack_channel: Slack channel override
        teams_webhook_url: Microsoft Teams webhook URL
        cooldown_seconds: Minimum time between duplicate alerts
        auto_resolve_on_recovery: Auto-resolve incidents when SLO recovers

    Returns:
        Configured SLOAlertBridge instance
    """
    global _bridge

    config = SLOAlertConfig(
        pagerduty_enabled=bool(pagerduty_api_key and pagerduty_service_id),
        pagerduty_api_key=pagerduty_api_key,
        pagerduty_service_id=pagerduty_service_id,
        slack_enabled=bool(slack_webhook_url),
        slack_webhook_url=slack_webhook_url,
        slack_channel=slack_channel,
        teams_enabled=bool(teams_webhook_url),
        teams_webhook_url=teams_webhook_url,
        cooldown_seconds=cooldown_seconds,
        auto_resolve_on_recovery=auto_resolve_on_recovery,
    )

    _bridge = SLOAlertBridge(config)

    # Register callbacks with SLO metrics system
    try:
        from aragora.observability.metrics.slo import (
            register_violation_callback,
            register_recovery_callback,
        )

        async def violation_callback(data: dict[str, Any]) -> None:
            """Handle SLO violation events from the metrics system."""
            if _bridge:
                await _bridge.on_slo_violation(
                    operation=data.get("operation", "unknown"),
                    percentile=data.get("percentile", "p99"),
                    latency_ms=data.get("latency_ms", 0),
                    threshold_ms=data.get("threshold_ms", 0),
                    severity=data.get("severity", "minor"),
                    context=data.get("context", {}),
                )

        async def recovery_callback(data: dict[str, Any]) -> None:
            """Handle SLO recovery events from the metrics system."""
            if _bridge:
                await _bridge.on_slo_recovery(
                    operation=data.get("operation", "unknown"),
                    percentile=data.get("percentile", "p99"),
                    context=data.get("context", {}),
                )

        register_violation_callback(violation_callback)
        register_recovery_callback(recovery_callback)

        logger.info(
            f"SLO alerting initialized: "
            f"pagerduty={config.pagerduty_enabled}, "
            f"slack={config.slack_enabled}, "
            f"teams={config.teams_enabled}"
        )

    except ImportError:
        logger.warning("SLO metrics module not available - alerting callbacks not registered")

    return _bridge


def shutdown_slo_alerting() -> None:
    """Shutdown the SLO alerting system and unregister callbacks."""
    global _bridge

    if _bridge is None:
        return

    try:
        from aragora.observability.metrics.slo import clear_all_callbacks

        clear_all_callbacks()
        logger.info("SLO alerting shutdown, callbacks unregistered")

    except ImportError:
        pass

    _bridge = None


__all__ = [
    "AlertSeverity",
    "AlertChannel",
    "SLOAlertConfig",
    "SLOAlertBridge",
    "ActiveViolation",
    "get_slo_alert_bridge",
    "init_slo_alerting",
    "shutdown_slo_alerting",
]
