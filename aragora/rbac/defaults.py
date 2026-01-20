"""
RBAC Default Roles and Permissions.

Defines the system-wide roles and permissions available in Aragora.
Organizations can create custom roles based on these templates.
"""

from __future__ import annotations


from .models import Action, Permission, ResourceType, Role


# ============================================================================
# SYSTEM PERMISSIONS
# ============================================================================


def _permission(
    resource: ResourceType,
    action: Action,
    name: str = "",
    description: str = "",
) -> Permission:
    """Helper to create a permission with auto-generated ID."""
    key = f"{resource.value}.{action.value}"
    return Permission(
        id=key,  # Use key as ID for simplicity
        name=name or key.replace(".", " ").replace("_", " ").title(),
        resource=resource,
        action=action,
        description=description,
    )


# Debate permissions
PERM_DEBATE_CREATE = _permission(
    ResourceType.DEBATE, Action.CREATE, "Create Debates", "Create new debates"
)
PERM_DEBATE_READ = _permission(
    ResourceType.DEBATE, Action.READ, "View Debates", "View debate details and history"
)
PERM_DEBATE_UPDATE = _permission(
    ResourceType.DEBATE, Action.UPDATE, "Update Debates", "Modify debate settings"
)
PERM_DEBATE_DELETE = _permission(
    ResourceType.DEBATE, Action.DELETE, "Delete Debates", "Delete debates permanently"
)
PERM_DEBATE_RUN = _permission(
    ResourceType.DEBATE, Action.RUN, "Run Debates", "Start and execute debates"
)
PERM_DEBATE_STOP = _permission(
    ResourceType.DEBATE, Action.STOP, "Stop Debates", "Stop running debates"
)
PERM_DEBATE_FORK = _permission(
    ResourceType.DEBATE, Action.FORK, "Fork Debates", "Create branches from existing debates"
)

# Agent permissions
PERM_AGENT_CREATE = _permission(
    ResourceType.AGENT, Action.CREATE, "Create Agents", "Create custom agent configurations"
)
PERM_AGENT_READ = _permission(
    ResourceType.AGENT, Action.READ, "View Agents", "View agent details and statistics"
)
PERM_AGENT_UPDATE = _permission(
    ResourceType.AGENT, Action.UPDATE, "Update Agents", "Modify agent configurations"
)
PERM_AGENT_DELETE = _permission(
    ResourceType.AGENT, Action.DELETE, "Delete Agents", "Remove agent configurations"
)
PERM_AGENT_DEPLOY = _permission(
    ResourceType.AGENT, Action.DEPLOY, "Deploy Agents", "Deploy agents to production"
)

# User management permissions
PERM_USER_READ = _permission(
    ResourceType.USER, Action.READ, "View Users", "View user profiles in organization"
)
PERM_USER_INVITE = _permission(
    ResourceType.USER, Action.INVITE, "Invite Users", "Invite new users to organization"
)
PERM_USER_REMOVE = _permission(
    ResourceType.USER, Action.REMOVE, "Remove Users", "Remove users from organization"
)
PERM_USER_CHANGE_ROLE = _permission(
    ResourceType.USER, Action.CHANGE_ROLE, "Change User Roles", "Modify user role assignments"
)
PERM_USER_IMPERSONATE = _permission(
    ResourceType.USER, Action.IMPERSONATE, "Impersonate Users", "Act on behalf of other users"
)

# Organization permissions
PERM_ORG_READ = _permission(
    ResourceType.ORGANIZATION, Action.READ, "View Organization", "View organization settings"
)
PERM_ORG_UPDATE = _permission(
    ResourceType.ORGANIZATION, Action.UPDATE, "Update Organization", "Modify organization settings"
)
PERM_ORG_BILLING = _permission(
    ResourceType.ORGANIZATION,
    Action.MANAGE_BILLING,
    "Manage Billing",
    "Manage organization billing and subscriptions",
)
PERM_ORG_AUDIT = _permission(
    ResourceType.ORGANIZATION,
    Action.VIEW_AUDIT,
    "View Audit Logs",
    "Access organization audit trail",
)
PERM_ORG_EXPORT = _permission(
    ResourceType.ORGANIZATION, Action.EXPORT_DATA, "Export Data", "Export organization data"
)

# API permissions
PERM_API_GENERATE_KEY = _permission(
    ResourceType.API, Action.GENERATE_KEY, "Generate API Keys", "Create new API keys"
)
PERM_API_REVOKE_KEY = _permission(
    ResourceType.API, Action.REVOKE_KEY, "Revoke API Keys", "Revoke existing API keys"
)

# Memory permissions
PERM_MEMORY_READ = _permission(
    ResourceType.MEMORY, Action.READ, "View Memory", "View memory contents and analytics"
)
PERM_MEMORY_UPDATE = _permission(
    ResourceType.MEMORY, Action.UPDATE, "Update Memory", "Modify memory contents"
)
PERM_MEMORY_DELETE = _permission(
    ResourceType.MEMORY, Action.DELETE, "Delete Memory", "Clear memory contents"
)

# Workflow permissions
PERM_WORKFLOW_CREATE = _permission(
    ResourceType.WORKFLOW, Action.CREATE, "Create Workflows", "Create new workflows"
)
PERM_WORKFLOW_READ = _permission(
    ResourceType.WORKFLOW, Action.READ, "View Workflows", "View workflow definitions and executions"
)
PERM_WORKFLOW_RUN = _permission(
    ResourceType.WORKFLOW, Action.RUN, "Run Workflows", "Execute workflows"
)
PERM_WORKFLOW_DELETE = _permission(
    ResourceType.WORKFLOW, Action.DELETE, "Delete Workflows", "Delete workflow definitions"
)

# Analytics permissions
PERM_ANALYTICS_READ = _permission(
    ResourceType.ANALYTICS, Action.READ, "View Analytics", "Access analytics dashboards"
)
PERM_ANALYTICS_EXPORT = _permission(
    ResourceType.ANALYTICS, Action.EXPORT_DATA, "Export Analytics", "Export analytics data"
)

# Training permissions
PERM_TRAINING_READ = _permission(
    ResourceType.TRAINING, Action.READ, "View Training Data", "Access training data exports"
)
PERM_TRAINING_CREATE = _permission(
    ResourceType.TRAINING,
    Action.CREATE,
    "Create Training Exports",
    "Generate training data exports",
)

# Evidence permissions
PERM_EVIDENCE_READ = _permission(
    ResourceType.EVIDENCE, Action.READ, "View Evidence", "Access evidence and citations"
)
PERM_EVIDENCE_CREATE = _permission(
    ResourceType.EVIDENCE, Action.CREATE, "Add Evidence", "Add new evidence sources"
)

# Connector permissions
PERM_CONNECTOR_READ = _permission(
    ResourceType.CONNECTOR, Action.READ, "View Connectors", "View connector configurations"
)
PERM_CONNECTOR_CREATE = _permission(
    ResourceType.CONNECTOR, Action.CREATE, "Create Connectors", "Configure new data connectors"
)
PERM_CONNECTOR_DELETE = _permission(
    ResourceType.CONNECTOR, Action.DELETE, "Delete Connectors", "Remove connector configurations"
)

# Admin permissions
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

# Webhook permissions
PERM_WEBHOOK_READ = _permission(
    ResourceType.WEBHOOK, Action.READ, "View Webhooks", "View webhook configurations"
)
PERM_WEBHOOK_CREATE = _permission(
    ResourceType.WEBHOOK, Action.CREATE, "Create Webhooks", "Create new webhooks"
)
PERM_WEBHOOK_DELETE = _permission(
    ResourceType.WEBHOOK, Action.DELETE, "Delete Webhooks", "Remove webhook configurations"
)

# Checkpoint permissions
PERM_CHECKPOINT_READ = _permission(
    ResourceType.CHECKPOINT, Action.READ, "View Checkpoints", "View saved checkpoints"
)
PERM_CHECKPOINT_CREATE = _permission(
    ResourceType.CHECKPOINT, Action.CREATE, "Create Checkpoints", "Save debate checkpoints"
)
PERM_CHECKPOINT_DELETE = _permission(
    ResourceType.CHECKPOINT, Action.DELETE, "Delete Checkpoints", "Remove saved checkpoints"
)


# All permissions as a dictionary for easy lookup
SYSTEM_PERMISSIONS: dict[str, Permission] = {
    p.key: p
    for p in [
        # Debates
        PERM_DEBATE_CREATE,
        PERM_DEBATE_READ,
        PERM_DEBATE_UPDATE,
        PERM_DEBATE_DELETE,
        PERM_DEBATE_RUN,
        PERM_DEBATE_STOP,
        PERM_DEBATE_FORK,
        # Agents
        PERM_AGENT_CREATE,
        PERM_AGENT_READ,
        PERM_AGENT_UPDATE,
        PERM_AGENT_DELETE,
        PERM_AGENT_DEPLOY,
        # Users
        PERM_USER_READ,
        PERM_USER_INVITE,
        PERM_USER_REMOVE,
        PERM_USER_CHANGE_ROLE,
        PERM_USER_IMPERSONATE,
        # Organization
        PERM_ORG_READ,
        PERM_ORG_UPDATE,
        PERM_ORG_BILLING,
        PERM_ORG_AUDIT,
        PERM_ORG_EXPORT,
        # API
        PERM_API_GENERATE_KEY,
        PERM_API_REVOKE_KEY,
        # Memory
        PERM_MEMORY_READ,
        PERM_MEMORY_UPDATE,
        PERM_MEMORY_DELETE,
        # Workflows
        PERM_WORKFLOW_CREATE,
        PERM_WORKFLOW_READ,
        PERM_WORKFLOW_RUN,
        PERM_WORKFLOW_DELETE,
        # Analytics
        PERM_ANALYTICS_READ,
        PERM_ANALYTICS_EXPORT,
        # Training
        PERM_TRAINING_READ,
        PERM_TRAINING_CREATE,
        # Evidence
        PERM_EVIDENCE_READ,
        PERM_EVIDENCE_CREATE,
        # Connectors
        PERM_CONNECTOR_READ,
        PERM_CONNECTOR_CREATE,
        PERM_CONNECTOR_DELETE,
        # Admin
        PERM_ADMIN_CONFIG,
        PERM_ADMIN_METRICS,
        PERM_ADMIN_FEATURES,
        PERM_ADMIN_ALL,
        # Webhooks
        PERM_WEBHOOK_READ,
        PERM_WEBHOOK_CREATE,
        PERM_WEBHOOK_DELETE,
        # Checkpoints
        PERM_CHECKPOINT_READ,
        PERM_CHECKPOINT_CREATE,
        PERM_CHECKPOINT_DELETE,
    ]
}


# ============================================================================
# SYSTEM ROLES
# ============================================================================

# Owner - Full control over organization
ROLE_OWNER = Role(
    id="owner",
    name="owner",
    display_name="Owner",
    description="Full control over the organization. Can manage billing, users, and all resources.",
    permissions={p.key for p in SYSTEM_PERMISSIONS.values()},
    priority=100,
    is_system=True,
)

# Admin - Administrative access without billing
ROLE_ADMIN = Role(
    id="admin",
    name="admin",
    display_name="Administrator",
    description="Manage users and resources. Cannot manage billing.",
    permissions={
        # All debate operations
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_UPDATE.key,
        PERM_DEBATE_DELETE.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # All agent operations
        PERM_AGENT_CREATE.key,
        PERM_AGENT_READ.key,
        PERM_AGENT_UPDATE.key,
        PERM_AGENT_DELETE.key,
        PERM_AGENT_DEPLOY.key,
        # User management
        PERM_USER_READ.key,
        PERM_USER_INVITE.key,
        PERM_USER_REMOVE.key,
        PERM_USER_CHANGE_ROLE.key,
        # Organization (no billing)
        PERM_ORG_READ.key,
        PERM_ORG_UPDATE.key,
        PERM_ORG_AUDIT.key,
        PERM_ORG_EXPORT.key,
        # API keys
        PERM_API_GENERATE_KEY.key,
        PERM_API_REVOKE_KEY.key,
        # All memory
        PERM_MEMORY_READ.key,
        PERM_MEMORY_UPDATE.key,
        PERM_MEMORY_DELETE.key,
        # All workflows
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        PERM_WORKFLOW_DELETE.key,
        # Analytics
        PERM_ANALYTICS_READ.key,
        PERM_ANALYTICS_EXPORT.key,
        # Training
        PERM_TRAINING_READ.key,
        PERM_TRAINING_CREATE.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Connectors
        PERM_CONNECTOR_READ.key,
        PERM_CONNECTOR_CREATE.key,
        PERM_CONNECTOR_DELETE.key,
        # Webhooks
        PERM_WEBHOOK_READ.key,
        PERM_WEBHOOK_CREATE.key,
        PERM_WEBHOOK_DELETE.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        PERM_CHECKPOINT_DELETE.key,
        # Admin (limited)
        PERM_ADMIN_METRICS.key,
    },
    parent_roles=[],
    priority=80,
    is_system=True,
)

# Debate Creator - Can create and manage debates
ROLE_DEBATE_CREATOR = Role(
    id="debate_creator",
    name="debate_creator",
    display_name="Debate Creator",
    description="Create, run, and manage debates. Cannot manage users or billing.",
    permissions={
        # Debate operations
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_UPDATE.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # Agent (read only + configure)
        PERM_AGENT_READ.key,
        # Memory
        PERM_MEMORY_READ.key,
        PERM_MEMORY_UPDATE.key,
        # Workflows
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Analytics (read)
        PERM_ANALYTICS_READ.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        # User (self only - enforced at resource level)
        PERM_USER_READ.key,
        # Org (read only)
        PERM_ORG_READ.key,
        # API keys for self
        PERM_API_GENERATE_KEY.key,
    },
    priority=50,
    is_system=True,
)

# Analyst - Read access to data and analytics
ROLE_ANALYST = Role(
    id="analyst",
    name="analyst",
    display_name="Analyst",
    description="View debates, analytics, and reports. Cannot create or modify resources.",
    permissions={
        # Read-only debate access
        PERM_DEBATE_READ.key,
        # Agent read
        PERM_AGENT_READ.key,
        # Memory read
        PERM_MEMORY_READ.key,
        # Workflow read
        PERM_WORKFLOW_READ.key,
        # Analytics (full)
        PERM_ANALYTICS_READ.key,
        PERM_ANALYTICS_EXPORT.key,
        # Training read
        PERM_TRAINING_READ.key,
        # Evidence read
        PERM_EVIDENCE_READ.key,
        # Checkpoint read
        PERM_CHECKPOINT_READ.key,
        # User/Org read
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
    },
    priority=30,
    is_system=True,
)

# Viewer - Minimal read-only access
ROLE_VIEWER = Role(
    id="viewer",
    name="viewer",
    display_name="Viewer",
    description="View debates and basic information. No modification rights.",
    permissions={
        PERM_DEBATE_READ.key,
        PERM_AGENT_READ.key,
        PERM_ORG_READ.key,
    },
    priority=10,
    is_system=True,
)

# Member - Default role for organization members (backward compatibility)
ROLE_MEMBER = Role(
    id="member",
    name="member",
    display_name="Member",
    description="Default organization member with standard access.",
    permissions={
        # Debate (create + run)
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # Agent read
        PERM_AGENT_READ.key,
        # Memory
        PERM_MEMORY_READ.key,
        # Workflow
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Analytics read
        PERM_ANALYTICS_READ.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        # Basic access
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
        # API key for self
        PERM_API_GENERATE_KEY.key,
    },
    parent_roles=[],
    priority=40,
    is_system=True,
)

# All system roles
SYSTEM_ROLES: dict[str, Role] = {
    r.name: r
    for r in [
        ROLE_OWNER,
        ROLE_ADMIN,
        ROLE_DEBATE_CREATOR,
        ROLE_ANALYST,
        ROLE_VIEWER,
        ROLE_MEMBER,
    ]
}

# Role hierarchy (for inheritance resolution)
ROLE_HIERARCHY: dict[str, list[str]] = {
    "owner": ["admin"],
    "admin": ["debate_creator", "analyst"],
    "debate_creator": ["member"],
    "analyst": ["viewer"],
    "member": ["viewer"],
    "viewer": [],
}


def get_permission(key: str) -> Permission | None:
    """Get a permission by its key."""
    return SYSTEM_PERMISSIONS.get(key)


def get_role(name: str) -> Role | None:
    """Get a role by its name."""
    return SYSTEM_ROLES.get(name)


def get_role_permissions(role_name: str, include_inherited: bool = True) -> set[str]:
    """
    Get all permissions for a role, optionally including inherited permissions.

    Args:
        role_name: Name of the role
        include_inherited: Whether to include permissions from parent roles

    Returns:
        Set of permission keys
    """
    role = get_role(role_name)
    if not role:
        return set()

    permissions = set(role.permissions)

    if include_inherited:
        for parent_name in ROLE_HIERARCHY.get(role_name, []):
            permissions |= get_role_permissions(parent_name, include_inherited=True)

    return permissions


def create_custom_role(
    name: str,
    display_name: str,
    description: str,
    permission_keys: set[str],
    org_id: str,
    base_role: str | None = None,
) -> Role:
    """
    Create a custom role for an organization.

    Args:
        name: Role name (must be unique within org)
        display_name: Human-readable name
        description: Role description
        permission_keys: Set of permission keys to grant
        org_id: Organization ID
        base_role: Optional base role to inherit from

    Returns:
        New custom Role instance
    """
    # Start with base role permissions if specified
    permissions = set()
    parent_roles = []

    if base_role:
        base = get_role(base_role)
        if base:
            permissions = get_role_permissions(base_role)
            parent_roles = [base_role]

    # Add specified permissions
    permissions |= permission_keys

    # Validate all permissions exist
    for key in permissions:
        if key not in SYSTEM_PERMISSIONS and not key.endswith(".*"):
            raise ValueError(f"Unknown permission: {key}")

    return Role(
        id=f"{org_id}:{name}",
        name=name,
        display_name=display_name,
        description=description,
        permissions=permissions,
        parent_roles=parent_roles,
        is_system=False,
        is_custom=True,
        org_id=org_id,
        priority=45,  # Between member and debate_creator
    )


# Predefined role templates for quick setup
ROLE_TEMPLATES = {
    "engineering": {
        "base": "debate_creator",
        "add": {PERM_AGENT_CREATE.key, PERM_AGENT_UPDATE.key, PERM_CONNECTOR_CREATE.key},
        "description": "Engineering team with agent management",
    },
    "research": {
        "base": "analyst",
        "add": {PERM_TRAINING_CREATE.key, PERM_DEBATE_CREATE.key, PERM_DEBATE_RUN.key},
        "description": "Research team with training data access",
    },
    "support": {
        "base": "viewer",
        "add": {PERM_USER_READ.key, PERM_ORG_AUDIT.key},
        "description": "Support team with user visibility",
    },
    "external": {
        "base": "viewer",
        "add": set(),
        "description": "External collaborators with minimal access",
    },
}
