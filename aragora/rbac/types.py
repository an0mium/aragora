"""
RBAC Types.

Role-based access control types for the enterprise control plane.
Supports hierarchical permissions: global -> org -> workspace -> resource.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class ResourceType(str, Enum):
    """Types of resources that can be protected by RBAC."""

    # Core resources
    DEBATE = "debate"
    AGENT = "agent"
    WORKFLOW = "workflow"
    DOCUMENT = "document"

    # Knowledge resources
    MEMORY = "memory"
    CULTURE = "culture"
    KNOWLEDGE_NODE = "knowledge_node"

    # Audit resources
    AUDIT_SESSION = "audit_session"
    AUDIT_FINDING = "audit_finding"

    # Training resources
    TRAINING_JOB = "training_job"
    SPECIALIST_MODEL = "specialist_model"

    # Administrative resources
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    BILLING = "billing"
    API_KEY = "api_key"


class Action(str, Enum):
    """Actions that can be performed on resources."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    SHARE = "share"
    ADMIN = "admin"


class Scope(str, Enum):
    """Scope levels for permissions."""

    GLOBAL = "global"  # System-wide permissions
    ORGANIZATION = "organization"  # Organization-level
    WORKSPACE = "workspace"  # Workspace-level
    RESOURCE = "resource"  # Specific resource instance


@dataclass
class Permission:
    """
    A single permission granting access to perform an action on a resource.

    Permissions can be scoped at different levels and may include
    conditions that further restrict when the permission applies.
    """

    resource: ResourceType
    action: Action
    scope: Scope = Scope.WORKSPACE
    conditions: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.resource.value, self.action.value, self.scope.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permission):
            return False
        return (
            self.resource == other.resource
            and self.action == other.action
            and self.scope == other.scope
        )

    def matches(
        self,
        resource: ResourceType,
        action: Action,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if this permission grants the requested access."""
        if self.resource != resource:
            return False
        if self.action != action and self.action != Action.ADMIN:
            return False

        # Check conditions if present
        if self.conditions and context:
            for key, expected in self.conditions.items():
                actual = context.get(key)
                if actual != expected:
                    return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource.value,
            "action": self.action.value,
            "scope": self.scope.value,
            "conditions": self.conditions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Permission:
        """Create from dictionary."""
        return cls(
            resource=ResourceType(data["resource"]),
            action=Action(data["action"]),
            scope=Scope(data.get("scope", "workspace")),
            conditions=data.get("conditions", {}),
        )


@dataclass
class Role:
    """
    A role that bundles permissions together.

    Roles can be scoped at different levels:
    - Global: System-wide roles (e.g., superadmin)
    - Organization: Org-level roles (e.g., org_admin)
    - Workspace: Workspace-specific roles (e.g., workspace_editor)
    """

    id: str
    name: str
    description: str
    permissions: set[Permission]
    scope: Scope
    is_system: bool = False  # System roles cannot be deleted
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(
        self,
        resource: ResourceType,
        action: Action,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if this role grants the requested permission."""
        for perm in self.permissions:
            if perm.matches(resource, action, context):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": [p.to_dict() for p in self.permissions],
            "scope": self.scope.value,
            "is_system": self.is_system,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class RoleAssignment:
    """
    Assignment of a role to an actor (user or service).

    The scope_id specifies where the role applies:
    - For org-scoped roles: organization_id
    - For workspace-scoped roles: workspace_id
    - For global roles: empty or "*"
    """

    actor_id: str
    role_id: str
    scope: Scope
    scope_id: str  # org_id or workspace_id depending on scope
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    assigned_by: str = ""
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actor_id": self.actor_id,
            "role_id": self.role_id,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "assigned_at": self.assigned_at.isoformat(),
            "assigned_by": self.assigned_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class IsolationContext:
    """
    Context for threading isolation information through request handling.

    This context is created at request entry and passed through all
    service calls to ensure proper workspace/org isolation.
    """

    # Actor information
    actor_id: str
    actor_type: Literal["user", "service", "agent"] = "user"

    # Scope information
    organization_id: str | None = None
    workspace_id: str | None = None

    # Request metadata
    request_id: str = ""
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cached role assignments (populated by RBACEnforcer)
    _role_assignments: list[RoleAssignment] = field(default_factory=list)
    _effective_permissions: set[Permission] = field(default_factory=set)

    def with_workspace(self, workspace_id: str) -> IsolationContext:
        """Create a new context scoped to a specific workspace."""
        return IsolationContext(
            actor_id=self.actor_id,
            actor_type=self.actor_type,
            organization_id=self.organization_id,
            workspace_id=workspace_id,
            request_id=self.request_id,
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
        )

    def with_organization(self, organization_id: str) -> IsolationContext:
        """Create a new context scoped to a specific organization."""
        return IsolationContext(
            actor_id=self.actor_id,
            actor_type=self.actor_type,
            organization_id=organization_id,
            workspace_id=self.workspace_id,
            request_id=self.request_id,
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


# Pre-defined system roles
SYSTEM_ROLES: dict[str, Role] = {
    "superadmin": Role(
        id="role_superadmin",
        name="Superadmin",
        description="Full system access",
        permissions={
            Permission(resource=r, action=Action.ADMIN, scope=Scope.GLOBAL) for r in ResourceType
        },
        scope=Scope.GLOBAL,
        is_system=True,
    ),
    "org_admin": Role(
        id="role_org_admin",
        name="Organization Admin",
        description="Full organization access",
        permissions={
            Permission(resource=r, action=Action.ADMIN, scope=Scope.ORGANIZATION)
            for r in ResourceType
        },
        scope=Scope.ORGANIZATION,
        is_system=True,
    ),
    "workspace_admin": Role(
        id="role_workspace_admin",
        name="Workspace Admin",
        description="Full workspace access",
        permissions={
            Permission(resource=r, action=Action.ADMIN, scope=Scope.WORKSPACE) for r in ResourceType
        },
        scope=Scope.WORKSPACE,
        is_system=True,
    ),
    "workspace_editor": Role(
        id="role_workspace_editor",
        name="Workspace Editor",
        description="Can create and modify resources in workspace",
        permissions={
            Permission(resource=r, action=a, scope=Scope.WORKSPACE)
            for r in [
                ResourceType.DEBATE,
                ResourceType.DOCUMENT,
                ResourceType.WORKFLOW,
                ResourceType.MEMORY,
            ]
            for a in [Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE]
        },
        scope=Scope.WORKSPACE,
        is_system=True,
    ),
    "workspace_viewer": Role(
        id="role_workspace_viewer",
        name="Workspace Viewer",
        description="Read-only access to workspace",
        permissions={
            Permission(resource=r, action=Action.READ, scope=Scope.WORKSPACE) for r in ResourceType
        },
        scope=Scope.WORKSPACE,
        is_system=True,
    ),
    "auditor": Role(
        id="role_auditor",
        name="Auditor",
        description="Can run and view audits",
        permissions={
            Permission(resource=ResourceType.AUDIT_SESSION, action=a, scope=Scope.WORKSPACE)
            for a in [Action.CREATE, Action.READ, Action.EXECUTE]
        }
        | {
            Permission(
                resource=ResourceType.AUDIT_FINDING, action=Action.READ, scope=Scope.WORKSPACE
            ),
            Permission(resource=ResourceType.DOCUMENT, action=Action.READ, scope=Scope.WORKSPACE),
        },
        scope=Scope.WORKSPACE,
        is_system=True,
    ),
    "ml_engineer": Role(
        id="role_ml_engineer",
        name="ML Engineer",
        description="Can train and manage specialist models",
        permissions={
            Permission(resource=ResourceType.TRAINING_JOB, action=a, scope=Scope.ORGANIZATION)
            for a in [Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE]
        }
        | {
            Permission(resource=ResourceType.SPECIALIST_MODEL, action=a, scope=Scope.ORGANIZATION)
            for a in [Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE]
        },
        scope=Scope.ORGANIZATION,
        is_system=True,
    ),
}


__all__ = [
    "ResourceType",
    "Action",
    "Scope",
    "Permission",
    "Role",
    "RoleAssignment",
    "IsolationContext",
    "SYSTEM_ROLES",
]
