"""
RBAC Permissions for Admin and System resources.

Contains permissions related to:
- Admin operations
- System configuration
- Metrics and monitoring
- Queue management
- Scheduler
- Backup and disaster recovery
- Features and feature flags
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# ADMIN PERMISSIONS
# ============================================================================

PERM_ADMIN_CONFIG = _permission(
    ResourceType.ADMIN, Action.SYSTEM_CONFIG, "System Configuration", "Modify system-wide settings"
)
PERM_ADMIN_METRICS = _permission(
    ResourceType.ADMIN,
    Action.VIEW_METRICS,
    "View System Metrics",
    "Access system performance metrics",
)
PERM_ADMIN_FEATURES = _permission(
    ResourceType.ADMIN, Action.MANAGE_FEATURES, "Manage Feature Flags", "Enable/disable features"
)
PERM_ADMIN_ALL = _permission(
    ResourceType.ADMIN, Action.ALL, "Full Admin Access", "All administrative capabilities"
)
PERM_ADMIN_AUDIT = _permission(
    ResourceType.ADMIN,
    Action.AUDIT,
    "Admin Audit",
    "Access administrative audit functions",
)
PERM_ADMIN_SECURITY = _permission(
    ResourceType.ADMIN,
    Action.SECURITY,
    "Admin Security",
    "Manage security configurations and policies",
)
PERM_ADMIN_SYSTEM = _permission(
    ResourceType.ADMIN,
    Action.SYSTEM,
    "Admin System",
    "System-wide administrative operations",
)

# ============================================================================
# SYSTEM PERMISSIONS
# ============================================================================

PERM_SYSTEM_HEALTH_READ = _permission(
    ResourceType.SYSTEM, Action.READ, "View System Health", "View system health and diagnostics"
)

# ============================================================================
# METRICS PERMISSIONS
# ============================================================================

PERM_METRICS_READ = _permission(
    ResourceType.METRICS, Action.READ, "View Metrics", "Access system and admin metrics"
)

# ============================================================================
# QUEUE PERMISSIONS
# ============================================================================

PERM_QUEUE_READ = _permission(
    ResourceType.QUEUE, Action.READ, "View Queue", "View job queue status and messages"
)
PERM_QUEUE_MANAGE = _permission(
    ResourceType.QUEUE, Action.MANAGE, "Manage Queue", "Submit, retry, and cancel queue jobs"
)
PERM_QUEUE_ADMIN = _permission(
    ResourceType.QUEUE,
    Action.ADMIN_OP,
    "Administer Queue",
    "Full queue administration including DLQ",
)

# ============================================================================
# SCHEDULER PERMISSIONS
# ============================================================================

PERM_SCHEDULER_READ = _permission(
    ResourceType.SCHEDULER,
    Action.READ,
    "View Scheduler",
    "View scheduled jobs and their status",
)
PERM_SCHEDULER_CREATE = _permission(
    ResourceType.SCHEDULER,
    Action.CREATE,
    "Create Schedules",
    "Create new scheduled jobs",
)
PERM_SCHEDULER_EXECUTE = _permission(
    ResourceType.SCHEDULER,
    Action.EXECUTE,
    "Execute Schedules",
    "Manually trigger scheduled jobs",
)
PERM_SCHEDULER_UPDATE = _permission(
    ResourceType.SCHEDULER, Action.UPDATE, "Update Scheduler", "Modify scheduled jobs"
)
PERM_SCHEDULER_DELETE = _permission(
    ResourceType.SCHEDULER, Action.DELETE, "Delete Scheduler", "Remove scheduled jobs"
)

# ============================================================================
# BACKUP & DR PERMISSIONS
# ============================================================================

PERM_BACKUP_CREATE = _permission(
    ResourceType.BACKUP,
    Action.CREATE,
    "Create Backups",
    "Create system backups",
)
PERM_BACKUP_READ = _permission(
    ResourceType.BACKUP,
    Action.READ,
    "Read Backups",
    "View backup status and metadata",
)
PERM_BACKUP_RESTORE = _permission(
    ResourceType.BACKUP,
    Action.RESTORE,
    "Restore Backups",
    "Restore system from backup (irreversible)",
)
PERM_BACKUP_DELETE = _permission(
    ResourceType.BACKUP,
    Action.DELETE,
    "Delete Backups",
    "Remove backup archives",
)
PERM_DR_READ = _permission(
    ResourceType.DISASTER_RECOVERY,
    Action.READ,
    "Read DR Status",
    "View disaster recovery configuration and status",
)
PERM_DR_EXECUTE = _permission(
    ResourceType.DISASTER_RECOVERY,
    Action.EXECUTE,
    "Execute DR Procedures",
    "Execute disaster recovery procedures",
)
PERM_DR_ALIAS_READ = _permission(
    ResourceType.DR, Action.READ, "View DR Status", "View disaster recovery status (alias)"
)
PERM_DR_DRILL = _permission(
    ResourceType.DR, Action.DRILL, "Run DR Drill", "Execute disaster recovery drills"
)

# ============================================================================
# FEATURES PERMISSIONS
# ============================================================================

PERM_FEATURES_READ = _permission(
    ResourceType.FEATURES, Action.READ, "View Features", "View feature flag status"
)
PERM_FEATURES_WRITE = _permission(
    ResourceType.FEATURES, Action.WRITE, "Manage Features", "Enable/disable feature flags"
)
PERM_FEATURES_DELETE = _permission(
    ResourceType.FEATURES, Action.DELETE, "Delete Features", "Remove feature flag configurations"
)

# All admin-related permission exports
__all__ = [
    # Admin
    "PERM_ADMIN_CONFIG",
    "PERM_ADMIN_METRICS",
    "PERM_ADMIN_FEATURES",
    "PERM_ADMIN_ALL",
    "PERM_ADMIN_AUDIT",
    "PERM_ADMIN_SECURITY",
    "PERM_ADMIN_SYSTEM",
    # System
    "PERM_SYSTEM_HEALTH_READ",
    # Metrics
    "PERM_METRICS_READ",
    # Queue
    "PERM_QUEUE_READ",
    "PERM_QUEUE_MANAGE",
    "PERM_QUEUE_ADMIN",
    # Scheduler
    "PERM_SCHEDULER_READ",
    "PERM_SCHEDULER_CREATE",
    "PERM_SCHEDULER_EXECUTE",
    "PERM_SCHEDULER_UPDATE",
    "PERM_SCHEDULER_DELETE",
    # Backup & DR
    "PERM_BACKUP_CREATE",
    "PERM_BACKUP_READ",
    "PERM_BACKUP_RESTORE",
    "PERM_BACKUP_DELETE",
    "PERM_DR_READ",
    "PERM_DR_EXECUTE",
    "PERM_DR_ALIAS_READ",
    "PERM_DR_DRILL",
    # Features
    "PERM_FEATURES_READ",
    "PERM_FEATURES_WRITE",
    "PERM_FEATURES_DELETE",
]
