"""
RBAC Permissions for Autonomous Operations.

Contains permissions related to:
- Autonomous agent triggers (scheduled debates)
- Alert management and thresholds
- Monitoring (trends, anomalies, baselines)
- Continuous learning (ratings, calibration, patterns)
- Approval flows
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# AUTONOMOUS PERMISSIONS
# ============================================================================

PERM_AUTONOMOUS_READ = _permission(
    ResourceType.AUTONOMOUS,
    Action.READ,
    "Read Autonomous Operations",
    "View autonomous triggers, monitoring, and learning data",
)
PERM_AUTONOMOUS_WRITE = _permission(
    ResourceType.AUTONOMOUS,
    Action.WRITE,
    "Write Autonomous Operations",
    "Create/modify triggers, record metrics, manage learning",
)
PERM_AUTONOMOUS_APPROVE = _permission(
    ResourceType.AUTONOMOUS,
    Action.APPROVE,
    "Approve Autonomous Actions",
    "Approve or reject autonomous operation requests",
)

# ============================================================================
# ALERT PERMISSIONS
# ============================================================================

PERM_ALERTS_READ = _permission(
    ResourceType.ALERTS,
    Action.READ,
    "View Alerts",
    "View active alerts and alert status",
)
PERM_ALERTS_WRITE = _permission(
    ResourceType.ALERTS,
    Action.WRITE,
    "Manage Alerts",
    "Acknowledge, resolve, and check alerts",
)
PERM_ALERTS_ADMIN = _permission(
    ResourceType.ALERTS,
    Action.ADMIN_OP,
    "Administer Alerts",
    "Configure alert thresholds and policies",
)

# All autonomous-related permission exports
__all__ = [
    # Autonomous
    "PERM_AUTONOMOUS_READ",
    "PERM_AUTONOMOUS_WRITE",
    "PERM_AUTONOMOUS_APPROVE",
    # Alerts
    "PERM_ALERTS_READ",
    "PERM_ALERTS_WRITE",
    "PERM_ALERTS_ADMIN",
]
