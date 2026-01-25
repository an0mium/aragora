"""
RBAC Models - Core data structures for role-based access control.

Implements:
- Permission: Individual access rights
- Role: Collection of permissions with hierarchy support
- RoleAssignment: User-role bindings with org scope and expiration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ResourceType(str, Enum):
    """Resource types that can be protected by permissions."""

    DEBATE = "debates"
    AGENT = "agents"
    USER = "users"
    ORGANIZATION = "organization"
    API = "api"
    MEMORY = "memory"
    WORKFLOW = "workflows"
    EVIDENCE = "evidence"
    TRAINING = "training"
    ANALYTICS = "analytics"
    ADMIN = "admin"
    BILLING = "billing"
    CONNECTOR = "connectors"
    WEBHOOK = "webhooks"
    CHECKPOINT = "checkpoints"
    GAUNTLET = "gauntlet"  # Adversarial stress-testing
    MARKETPLACE = "marketplace"  # Template marketplace
    EXPLAINABILITY = "explainability"  # Decision explanations
    FINDINGS = "findings"  # Audit findings management
    DECISION = "decisions"  # Unified decision routing

    # Governance and orchestration
    POLICY = "policies"  # Governance policies
    COMPLIANCE = "compliance"  # Compliance management
    CONTROL_PLANE = "control_plane"  # Control plane orchestration

    # Enterprise data governance
    DATA_CLASSIFICATION = "data_classification"  # Data sensitivity classification
    DATA_RETENTION = "data_retention"  # Data retention policies
    DATA_LINEAGE = "data_lineage"  # Data provenance tracking
    PII = "pii"  # Personally identifiable information

    # Compliance and regulatory
    COMPLIANCE_POLICY = "compliance_policy"  # Compliance rules (SOC2, GDPR, HIPAA)
    AUDIT_LOG = "audit_log"  # Audit trail management
    VENDOR = "vendor"  # Third-party vendor management

    # Team/group management
    TEAM = "team"  # Team-based access control

    # Cost and quota management
    QUOTA = "quota"  # Rate limits and quotas
    COST_CENTER = "cost_center"  # Cost tracking and chargeback
    BUDGET = "budget"  # Budget limits and alerts

    # Session and authentication
    SESSION = "session"  # Active session management
    AUTHENTICATION = "authentication"  # Auth policy management

    # Approval workflows
    APPROVAL = "approval"  # Access request approvals

    # Enterprise infrastructure
    BACKUP = "backup"  # Backup management
    DISASTER_RECOVERY = "disaster_recovery"  # DR procedures
    ROLE = "role"  # Custom role management
    API_KEY = "api_key"  # API key management
    TEMPLATE = "template"  # Workflow template management


class Action(str, Enum):
    """Actions that can be performed on resources."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Debate-specific
    RUN = "run"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    FORK = "fork"

    # Agent-specific
    DEPLOY = "deploy"
    CONFIGURE = "configure"

    # User management
    INVITE = "invite"
    REMOVE = "remove"
    CHANGE_ROLE = "change_role"
    IMPERSONATE = "impersonate"

    # Organization
    MANAGE_BILLING = "manage_billing"
    VIEW_AUDIT = "view_audit"
    EXPORT_DATA = "export_data"

    # API
    GENERATE_KEY = "generate_key"
    REVOKE_KEY = "revoke_key"

    # Admin
    SYSTEM_CONFIG = "system_config"
    VIEW_METRICS = "view_metrics"
    MANAGE_FEATURES = "manage_features"

    # Gauntlet-specific
    SIGN = "sign"  # Sign receipts cryptographically
    COMPARE = "compare"  # Compare gauntlet runs

    # Marketplace-specific
    PUBLISH = "publish"  # Publish to marketplace
    IMPORT = "import"  # Import from marketplace
    RATE = "rate"  # Rate templates
    REVIEW = "review"  # Write reviews

    # Explainability-specific
    BATCH = "batch"  # Run batch operations

    # Findings-specific
    ASSIGN = "assign"  # Assign to users
    BULK = "bulk"  # Bulk operations

    # Data governance actions
    CLASSIFY = "classify"  # Classify data sensitivity
    REDACT = "redact"  # Redact sensitive data
    MASK = "mask"  # Apply data masking rules

    # Compliance actions
    ENFORCE = "enforce"  # Enforce compliance policies
    STREAM = "stream"  # Stream to external systems (SIEM)
    SEARCH = "search"  # Advanced search capabilities
    CHECK = "check"  # Run compliance checks

    # Control plane actions
    SUBMIT = "submit"  # Submit tasks/requests
    CANCEL = "cancel"  # Cancel pending operations
    DELIBERATE = "deliberate"  # Start deliberation process

    # Connector lifecycle actions
    AUTHORIZE = "authorize"  # Grant OAuth/API credentials
    ROTATE = "rotate"  # Rotate credentials
    TEST = "test"  # Test connection health
    ROLLBACK = "rollback"  # Revert failed operations

    # Team management actions
    ADD_MEMBER = "add_member"  # Add user to team
    REMOVE_MEMBER = "remove_member"  # Remove user from team
    SHARE = "share"  # Share resource with team

    # Quota and cost actions
    SET_LIMIT = "set_limit"  # Set quotas/limits
    CHARGEBACK = "chargeback"  # Assign costs to cost center

    # Session and auth actions
    REVOKE = "revoke"  # Revoke sessions/credentials
    LIST_ACTIVE = "list_active"  # List active sessions
    RESET_PASSWORD = "reset_password"  # Reset user password
    REQUIRE_MFA = "require_mfa"  # Enforce MFA

    # Approval workflow actions
    REQUEST = "request"  # Request access/approval
    GRANT = "grant"  # Grant approval
    DENY = "deny"  # Deny approval

    # Enterprise sensitive operations
    OVERRIDE = "override"  # Override quotas/limits
    DISSOLVE = "dissolve"  # Dissolve teams/groups
    LIST_ALL = "list_all"  # List all items (not just own)
    EXPORT_SECRET = "export_secret"  # Export secrets/credentials
    EXPORT_HISTORY = "export_history"  # Export historical data
    RESTORE = "restore"  # Restore from backup
    EXECUTE = "execute"  # Execute procedures (DR, migrations)

    # Wildcard
    ALL = "*"


@dataclass
class Permission:
    """
    Individual permission representing access to perform an action on a resource.

    Attributes:
        id: Unique identifier
        name: Human-readable name (e.g., "Create Debates")
        resource: Resource type this permission applies to
        action: Action allowed by this permission
        description: Detailed description for documentation
        conditions: Optional conditions for ABAC (attribute-based access control)
    """

    id: str
    name: str
    resource: ResourceType
    action: Action
    description: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Generate permission key in format 'resource.action'."""
        return f"{self.resource.value}.{self.action.value}"

    @classmethod
    def from_key(cls, key: str, name: str = "", description: str = "") -> Permission:
        """Create permission from key string like 'debates.create'."""
        resource_str, action_str = key.split(".", 1)
        return cls(
            id=str(uuid4()),
            name=name or key.replace(".", " ").title(),
            resource=ResourceType(resource_str),
            action=Action(action_str),
            description=description,
        )

    def matches(self, resource: ResourceType, action: Action) -> bool:
        """Check if this permission matches the requested resource and action."""
        # Wildcard action matches all
        if self.action == Action.ALL:
            return self.resource == resource
        # Exact match
        return self.resource == resource and self.action == action


@dataclass
class Role:
    """
    Collection of permissions assigned to users.

    Supports hierarchy where a role can inherit from parent roles.

    Attributes:
        id: Unique identifier
        name: Role name (e.g., "admin", "debate_creator")
        display_name: Human-readable display name
        description: Role description
        permissions: Set of permission IDs granted by this role
        parent_roles: Roles this role inherits from
        is_system: Whether this is a built-in system role
        is_custom: Whether this is a custom org-defined role
        org_id: Organization ID for custom roles (None for system roles)
        priority: Role priority for conflict resolution (higher = more privileged)
        metadata: Additional role configuration
    """

    id: str
    name: str
    display_name: str = ""
    description: str = ""
    permissions: set[str] = field(default_factory=set)
    parent_roles: list[str] = field(default_factory=list)
    is_system: bool = True
    is_custom: bool = False
    org_id: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def has_permission(self, permission_id: str) -> bool:
        """Check if role directly has a permission (not including inheritance)."""
        return permission_id in self.permissions

    def add_permission(self, permission_id: str) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission_id)

    def remove_permission(self, permission_id: str) -> None:
        """Remove a permission from this role."""
        self.permissions.discard(permission_id)


@dataclass
class RoleAssignment:
    """
    Assignment of a role to a user, scoped to an organization.

    Attributes:
        id: Unique identifier
        user_id: User receiving the role
        role_id: Role being assigned
        org_id: Organization scope (None for platform-wide roles)
        assigned_by: User who made the assignment
        assigned_at: When the assignment was made
        expires_at: When the assignment expires (None = never)
        is_active: Whether the assignment is currently active
        conditions: Additional conditions for the assignment
        metadata: Additional assignment data
    """

    id: str
    user_id: str
    role_id: str
    org_id: str | None = None
    assigned_by: str | None = None
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    is_active: bool = True
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the assignment is currently valid."""
        return self.is_active and not self.is_expired


@dataclass
class APIKeyScope:
    """
    Scope definition for API keys to limit their permissions.

    Attributes:
        permissions: Set of permission keys allowed for this key
        resources: Specific resource IDs the key can access (None = all)
        rate_limit: Custom rate limit for this key
        expires_at: Key expiration time
        ip_whitelist: Allowed IP addresses (None = all)
    """

    permissions: set[str] = field(default_factory=set)
    resources: dict[ResourceType, set[str]] | None = None
    rate_limit: int | None = None
    expires_at: datetime | None = None
    ip_whitelist: set[str] | None = None

    def allows_permission(self, permission_key: str) -> bool:
        """Check if this scope allows a permission."""
        # Empty permissions = full access
        if not self.permissions:
            return True
        # Check for wildcard
        if "*" in self.permissions:
            return True
        # Check exact match
        if permission_key in self.permissions:
            return True
        # Check resource wildcard (e.g., "debates.*")
        resource = permission_key.split(".")[0]
        if f"{resource}.*" in self.permissions:
            return True
        return False

    def allows_resource(self, resource_type: ResourceType, resource_id: str) -> bool:
        """Check if this scope allows access to a specific resource."""
        if self.resources is None:
            return True
        if resource_type not in self.resources:
            return True
        return resource_id in self.resources[resource_type]


@dataclass
class AuthorizationContext:
    """
    Context for authorization decisions.

    Attributes:
        user_id: ID of the user making the request
        user_email: Email of the user (optional, for display/audit)
        org_id: Organization context
        workspace_id: Workspace context for multi-tenant workspaces
        roles: User's active roles
        permissions: Resolved permissions from roles
        api_key_scope: Scope if using API key (None for session auth)
        ip_address: Request IP address
        user_agent: Request user agent
        request_id: Unique request identifier for tracing
        timestamp: When the authorization context was created
    """

    user_id: str
    user_email: str | None = None
    org_id: str | None = None
    workspace_id: str | None = None
    roles: set[str] = field(default_factory=set)
    permissions: set[str] = field(default_factory=set)
    api_key_scope: APIKeyScope | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def has_permission(self, permission_key: str) -> bool:
        """Check if context has a permission."""
        # Check API key scope first if present
        if self.api_key_scope and not self.api_key_scope.allows_permission(permission_key):
            return False
        # Check resolved permissions
        if permission_key in self.permissions:
            return True
        # Check for wildcard
        resource = permission_key.split(".")[0]
        if f"{resource}.*" in self.permissions:
            return True
        if "*" in self.permissions:
            return True
        return False

    def has_role(self, role_name: str) -> bool:
        """Check if context has a specific role."""
        return role_name in self.roles

    def has_any_role(self, *role_names: str) -> bool:
        """Check if context has any of the specified roles."""
        return bool(self.roles & set(role_names))


@dataclass
class AuthorizationDecision:
    """
    Result of an authorization check.

    Attributes:
        allowed: Whether access is allowed
        reason: Explanation of the decision
        permission_key: Permission that was checked
        resource_id: Specific resource if applicable
        context: Authorization context used
        checked_at: When the check was performed
        cached: Whether the decision was from cache
    """

    allowed: bool
    reason: str
    permission_key: str
    resource_id: str | None = None
    context: AuthorizationContext | None = None
    checked_at: datetime = field(default_factory=datetime.utcnow)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "permission_key": self.permission_key,
            "resource_id": self.resource_id,
            "user_id": self.context.user_id if self.context else None,
            "org_id": self.context.org_id if self.context else None,
            "request_id": self.context.request_id if self.context else None,
            "checked_at": self.checked_at.isoformat(),
            "cached": self.cached,
        }
