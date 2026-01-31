"""
OpenClaw Gateway Audit Events.

Defines audit event types for all OpenClaw gateway operations.
All events are logged through the standard audit infrastructure with HMAC signing.
"""

from __future__ import annotations

from enum import Enum


class OpenClawAuditEvents(str, Enum):
    """Audit event types for OpenClaw gateway operations."""

    # Task lifecycle
    TASK_SUBMITTED = "openclaw.task.submitted"
    TASK_STARTED = "openclaw.task.started"
    TASK_COMPLETED = "openclaw.task.completed"
    TASK_FAILED = "openclaw.task.failed"
    TASK_BLOCKED = "openclaw.task.blocked"
    TASK_TIMEOUT = "openclaw.task.timeout"

    # Device operations
    DEVICE_REGISTERED = "openclaw.device.registered"
    DEVICE_UNREGISTERED = "openclaw.device.unregistered"
    DEVICE_CONNECTED = "openclaw.device.connected"
    DEVICE_DISCONNECTED = "openclaw.device.disconnected"

    # Plugin operations
    PLUGIN_INSTALLED = "openclaw.plugin.installed"
    PLUGIN_UNINSTALLED = "openclaw.plugin.uninstalled"
    PLUGIN_BLOCKED = "openclaw.plugin.blocked"
    PLUGIN_ALLOWED = "openclaw.plugin.allowed"

    # Security events
    CAPABILITY_DENIED = "openclaw.capability.denied"
    CAPABILITY_APPROVED = "openclaw.capability.approved"
    SANDBOX_VIOLATION = "openclaw.sandbox.violation"
    RESOURCE_LIMIT_HIT = "openclaw.resource.limit_hit"
    POLICY_VIOLATION = "openclaw.policy.violation"

    # Gateway lifecycle
    GATEWAY_CONNECTED = "openclaw.gateway.connected"
    GATEWAY_DISCONNECTED = "openclaw.gateway.disconnected"
    GATEWAY_ERROR = "openclaw.gateway.error"


# Event severity levels for alerting
EVENT_SEVERITY = {
    OpenClawAuditEvents.TASK_SUBMITTED: "info",
    OpenClawAuditEvents.TASK_STARTED: "info",
    OpenClawAuditEvents.TASK_COMPLETED: "info",
    OpenClawAuditEvents.TASK_FAILED: "warning",
    OpenClawAuditEvents.TASK_BLOCKED: "warning",
    OpenClawAuditEvents.TASK_TIMEOUT: "warning",
    OpenClawAuditEvents.DEVICE_REGISTERED: "info",
    OpenClawAuditEvents.DEVICE_UNREGISTERED: "info",
    OpenClawAuditEvents.DEVICE_CONNECTED: "info",
    OpenClawAuditEvents.DEVICE_DISCONNECTED: "info",
    OpenClawAuditEvents.PLUGIN_INSTALLED: "info",
    OpenClawAuditEvents.PLUGIN_UNINSTALLED: "info",
    OpenClawAuditEvents.PLUGIN_BLOCKED: "warning",
    OpenClawAuditEvents.PLUGIN_ALLOWED: "info",
    OpenClawAuditEvents.CAPABILITY_DENIED: "warning",
    OpenClawAuditEvents.CAPABILITY_APPROVED: "info",
    OpenClawAuditEvents.SANDBOX_VIOLATION: "critical",
    OpenClawAuditEvents.RESOURCE_LIMIT_HIT: "warning",
    OpenClawAuditEvents.POLICY_VIOLATION: "critical",
    OpenClawAuditEvents.GATEWAY_CONNECTED: "info",
    OpenClawAuditEvents.GATEWAY_DISCONNECTED: "info",
    OpenClawAuditEvents.GATEWAY_ERROR: "error",
}


def get_event_severity(event: OpenClawAuditEvents) -> str:
    """Get severity level for an audit event."""
    return EVENT_SEVERITY.get(event, "info")


__all__ = [
    "OpenClawAuditEvents",
    "EVENT_SEVERITY",
    "get_event_severity",
]
