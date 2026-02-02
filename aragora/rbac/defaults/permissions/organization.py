"""
RBAC Permissions for Organization, User, Team, and Workspace resources.

Contains permissions related to:
- Organization management
- User management
- Team management
- Workspace management
- Role management
- Session and authentication
- Approval workflows
"""

from __future__ import annotations

from aragora.rbac.models import Action, Permission, ResourceType

from ._helpers import _permission

# ============================================================================
# USER MANAGEMENT PERMISSIONS
# ============================================================================

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

# ============================================================================
# ORGANIZATION PERMISSIONS
# ============================================================================

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
PERM_ORG_INVITE = _permission(
    ResourceType.ORGANIZATION,
    Action.INVITE,
    "Invite to Organization",
    "Invite new users to the organization",
)
PERM_ORG_DELETE = _permission(
    ResourceType.ORGANIZATION,
    Action.DELETE,
    "Delete Organization",
    "Permanently delete an organization (irreversible)",
)
PERM_ORG_USAGE_READ = _permission(
    ResourceType.ORGANIZATION, Action.READ, "View Org Usage", "View organization usage metrics"
)
PERM_ORG_MEMBERS = _permission(
    ResourceType.ORGANIZATION, Action.READ, "View Org Members", "View organization members"
)

# ============================================================================
# TEAM PERMISSIONS
# ============================================================================

PERM_TEAM_CREATE = _permission(ResourceType.TEAM, Action.CREATE, "Create Teams", "Create new teams")
PERM_TEAM_READ = _permission(
    ResourceType.TEAM, Action.READ, "View Teams", "View team membership and details"
)
PERM_TEAM_UPDATE = _permission(
    ResourceType.TEAM, Action.UPDATE, "Update Teams", "Modify team settings"
)
PERM_TEAM_DELETE = _permission(ResourceType.TEAM, Action.DELETE, "Delete Teams", "Remove teams")
PERM_TEAM_ADD_MEMBER = _permission(
    ResourceType.TEAM, Action.ADD_MEMBER, "Add Team Members", "Invite users to teams"
)
PERM_TEAM_REMOVE_MEMBER = _permission(
    ResourceType.TEAM,
    Action.REMOVE_MEMBER,
    "Remove Team Members",
    "Remove users from teams",
)
PERM_TEAM_SHARE = _permission(
    ResourceType.TEAM, Action.SHARE, "Share with Team", "Grant team access to resources"
)
PERM_TEAM_DISSOLVE = _permission(
    ResourceType.TEAM,
    Action.DISSOLVE,
    "Dissolve Teams",
    "Dissolve teams with resource reallocation",
)

# ============================================================================
# WORKSPACE PERMISSIONS
# ============================================================================

PERM_WORKSPACE_CREATE = _permission(
    ResourceType.WORKSPACE, Action.CREATE, "Create Workspaces", "Create new workspaces"
)
PERM_WORKSPACE_READ = _permission(
    ResourceType.WORKSPACE, Action.READ, "View Workspaces", "View workspace details and settings"
)
PERM_WORKSPACE_UPDATE = _permission(
    ResourceType.WORKSPACE, Action.UPDATE, "Update Workspaces", "Modify workspace settings"
)
PERM_WORKSPACE_DELETE = _permission(
    ResourceType.WORKSPACE, Action.DELETE, "Delete Workspaces", "Remove workspaces"
)
PERM_WORKSPACE_MEMBER_ADD = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.ADD_MEMBER,
    "Add Workspace Members",
    "Invite users to workspaces",
)
PERM_WORKSPACE_MEMBER_REMOVE = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.REMOVE_MEMBER,
    "Remove Workspace Members",
    "Remove users from workspaces",
)
PERM_WORKSPACE_MEMBER_CHANGE_ROLE = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.CHANGE_ROLE,
    "Change Member Roles",
    "Modify member roles within workspaces",
)
PERM_WORKSPACE_SHARE = _permission(
    ResourceType.WORKSPACE,
    Action.SHARE,
    "Share with Workspace",
    "Grant workspace access to resources",
)

# ============================================================================
# ROLE PERMISSIONS
# ============================================================================

PERM_ROLE_CREATE = _permission(
    ResourceType.ROLE,
    Action.CREATE,
    "Create Roles",
    "Create custom roles",
)
PERM_ROLE_READ = _permission(
    ResourceType.ROLE,
    Action.READ,
    "Read Roles",
    "View custom role definitions",
)
PERM_ROLE_UPDATE = _permission(
    ResourceType.ROLE,
    Action.UPDATE,
    "Update Roles",
    "Modify custom role permissions",
)
PERM_ROLE_DELETE = _permission(
    ResourceType.ROLE,
    Action.DELETE,
    "Delete Roles",
    "Permanently delete custom roles",
)

# ============================================================================
# SESSION & AUTH PERMISSIONS
# ============================================================================

PERM_SESSION_READ = _permission(
    ResourceType.SESSION, Action.LIST_ACTIVE, "View Active Sessions", "List active user sessions"
)
PERM_SESSION_REVOKE = _permission(
    ResourceType.SESSION, Action.REVOKE, "Revoke Sessions", "Force logout of user sessions"
)
PERM_SESSION_CREATE = _permission(
    ResourceType.SESSION,
    Action.CREATE,
    "Create Sessions",
    "Create sessions on behalf of users",
)
PERM_AUTH_RESET_PASSWORD = _permission(
    ResourceType.AUTHENTICATION,
    Action.RESET_PASSWORD,
    "Reset Passwords",
    "Reset user passwords",
)
PERM_AUTH_REQUIRE_MFA = _permission(
    ResourceType.AUTHENTICATION,
    Action.REQUIRE_MFA,
    "Require MFA",
    "Enforce MFA for users or operations",
)
PERM_AUTH_READ = _permission(
    ResourceType.AUTHENTICATION,
    Action.READ,
    "View Auth Info",
    "View own authentication info and settings",
)
PERM_AUTH_CREATE = _permission(
    ResourceType.AUTHENTICATION,
    Action.CREATE,
    "Setup Auth",
    "Setup authentication methods (MFA)",
)
PERM_AUTH_UPDATE = _permission(
    ResourceType.AUTHENTICATION,
    Action.UPDATE,
    "Update Auth",
    "Update authentication settings (enable/disable MFA, link accounts)",
)
PERM_AUTH_REVOKE = _permission(
    ResourceType.AUTHENTICATION,
    Action.REVOKE,
    "Revoke Auth",
    "Revoke sessions and tokens (logout)",
)

# ============================================================================
# APPROVAL PERMISSIONS
# ============================================================================

PERM_APPROVAL_REQUEST = _permission(
    ResourceType.APPROVAL, Action.REQUEST, "Request Access", "Request elevated access or approvals"
)
PERM_APPROVAL_GRANT = _permission(
    ResourceType.APPROVAL,
    Action.GRANT,
    "Grant Approvals",
    "Approve access requests",
)
PERM_APPROVAL_READ = _permission(
    ResourceType.APPROVAL,
    Action.READ,
    "View Approval History",
    "View past approval decisions",
)

# ============================================================================
# ONBOARDING PERMISSIONS
# ============================================================================

PERM_ONBOARDING_READ = _permission(
    ResourceType.ONBOARDING, Action.READ, "View Onboarding", "View onboarding status"
)
PERM_ONBOARDING_CREATE = _permission(
    ResourceType.ONBOARDING, Action.CREATE, "Create Onboarding", "Start onboarding flows"
)
PERM_ONBOARDING_UPDATE = _permission(
    ResourceType.ONBOARDING, Action.UPDATE, "Update Onboarding", "Modify onboarding progress"
)

# ============================================================================
# PARTNER PERMISSIONS
# ============================================================================

PERM_PARTNER_READ = _permission(
    ResourceType.PARTNER, Action.READ, "View Partners", "View partner information"
)

# ============================================================================
# HR PERMISSIONS
# ============================================================================

PERM_HR_READ = _permission(ResourceType.HR, Action.READ, "View HR", "View human resources data")

# All organization-related permission exports
__all__ = [
    # User
    "PERM_USER_READ",
    "PERM_USER_INVITE",
    "PERM_USER_REMOVE",
    "PERM_USER_CHANGE_ROLE",
    "PERM_USER_IMPERSONATE",
    # Organization
    "PERM_ORG_READ",
    "PERM_ORG_UPDATE",
    "PERM_ORG_BILLING",
    "PERM_ORG_AUDIT",
    "PERM_ORG_EXPORT",
    "PERM_ORG_INVITE",
    "PERM_ORG_DELETE",
    "PERM_ORG_USAGE_READ",
    "PERM_ORG_MEMBERS",
    # Team
    "PERM_TEAM_CREATE",
    "PERM_TEAM_READ",
    "PERM_TEAM_UPDATE",
    "PERM_TEAM_DELETE",
    "PERM_TEAM_ADD_MEMBER",
    "PERM_TEAM_REMOVE_MEMBER",
    "PERM_TEAM_SHARE",
    "PERM_TEAM_DISSOLVE",
    # Workspace
    "PERM_WORKSPACE_CREATE",
    "PERM_WORKSPACE_READ",
    "PERM_WORKSPACE_UPDATE",
    "PERM_WORKSPACE_DELETE",
    "PERM_WORKSPACE_MEMBER_ADD",
    "PERM_WORKSPACE_MEMBER_REMOVE",
    "PERM_WORKSPACE_MEMBER_CHANGE_ROLE",
    "PERM_WORKSPACE_SHARE",
    # Role
    "PERM_ROLE_CREATE",
    "PERM_ROLE_READ",
    "PERM_ROLE_UPDATE",
    "PERM_ROLE_DELETE",
    # Session & Auth
    "PERM_SESSION_READ",
    "PERM_SESSION_REVOKE",
    "PERM_SESSION_CREATE",
    "PERM_AUTH_RESET_PASSWORD",
    "PERM_AUTH_REQUIRE_MFA",
    "PERM_AUTH_READ",
    "PERM_AUTH_CREATE",
    "PERM_AUTH_UPDATE",
    "PERM_AUTH_REVOKE",
    # Approval
    "PERM_APPROVAL_REQUEST",
    "PERM_APPROVAL_GRANT",
    "PERM_APPROVAL_READ",
    # Onboarding
    "PERM_ONBOARDING_READ",
    "PERM_ONBOARDING_CREATE",
    "PERM_ONBOARDING_UPDATE",
    # Partner
    "PERM_PARTNER_READ",
    # HR
    "PERM_HR_READ",
]
