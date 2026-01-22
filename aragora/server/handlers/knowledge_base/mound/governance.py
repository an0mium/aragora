"""
Governance operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for RBAC and audit trail:
- POST /api/knowledge/mound/governance/roles - Create a role
- GET /api/knowledge/mound/governance/roles - List roles
- POST /api/knowledge/mound/governance/roles/assign - Assign role to user
- POST /api/knowledge/mound/governance/roles/revoke - Revoke role from user
- GET /api/knowledge/mound/governance/permissions/:user_id - Get user permissions
- POST /api/knowledge/mound/governance/permissions/check - Check specific permission
- GET /api/knowledge/mound/governance/audit - Query audit trail
- GET /api/knowledge/mound/governance/audit/user/:user_id - Get user activity
- GET /api/knowledge/mound/governance/stats - Get governance stats

Phase A2 - Workspace Governance
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from ...base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class GovernanceOperationsMixin:
    """Mixin providing governance (RBAC + audit) API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @rate_limit(requests_per_minute=20)
    async def create_role(
        self,
        name: str,
        permissions: list[str],
        description: str = "",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> HandlerResult:
        """
        Create a new role.

        POST /api/knowledge/mound/governance/roles
        {
            "name": "Custom Editor",
            "permissions": ["read", "create", "update"],
            "description": "Can read and edit items",
            "workspace_id": "...",  // optional, None for global
            "created_by": "admin_user_id"
        }

        Args:
            name: Role name
            permissions: List of permission strings
            description: Role description
            workspace_id: Optional workspace scope
            created_by: User creating the role

        Returns:
            Created role object
        """
        from aragora.knowledge.mound.ops.governance import Permission

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not name:
            return error_response("name is required", status=400)

        if not permissions:
            return error_response("permissions list is required", status=400)

        # Convert permission strings to Permission enum
        try:
            perm_set: Set[Permission] = set()
            for p in permissions:
                perm_set.add(Permission(p))
        except ValueError as e:
            valid_perms = [p.value for p in Permission]
            return error_response(
                f"Invalid permission. Valid permissions: {valid_perms}. Error: {e}",
                status=400,
            )

        try:
            role = await mound.create_role(
                name=name,
                permissions=perm_set,
                description=description,
                workspace_id=workspace_id,
                created_by=created_by,
            )

            return json_response(
                {
                    "success": True,
                    "role": role.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error creating role: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=20)
    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
        assigned_by: Optional[str] = None,
    ) -> HandlerResult:
        """
        Assign a role to a user.

        POST /api/knowledge/mound/governance/roles/assign
        {
            "user_id": "...",
            "role_id": "...",
            "workspace_id": "...",  // optional
            "assigned_by": "admin_id"
        }

        Args:
            user_id: User to assign role to
            role_id: Role to assign
            workspace_id: Optional workspace scope
            assigned_by: User making the assignment

        Returns:
            Role assignment record
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not user_id or not role_id:
            return error_response("user_id and role_id are required", status=400)

        try:
            assignment = await mound.assign_role(
                user_id=user_id,
                role_id=role_id,
                workspace_id=workspace_id,
                assigned_by=assigned_by,
            )

            return json_response(
                {
                    "success": True,
                    "assignment": assignment.to_dict(),
                }
            )
        except ValueError as e:
            return error_response(str(e), status=404)
        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=20)
    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
    ) -> HandlerResult:
        """
        Revoke a role from a user.

        POST /api/knowledge/mound/governance/roles/revoke
        {
            "user_id": "...",
            "role_id": "...",
            "workspace_id": "..."  // optional
        }

        Args:
            user_id: User to revoke from
            role_id: Role to revoke
            workspace_id: Optional workspace scope

        Returns:
            Success status
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not user_id or not role_id:
            return error_response("user_id and role_id are required", status=400)

        try:
            success = await mound.revoke_role(
                user_id=user_id,
                role_id=role_id,
                workspace_id=workspace_id,
            )

            if not success:
                return error_response("Role assignment not found", status=404)

            return json_response(
                {
                    "success": True,
                    "message": f"Role {role_id} revoked from user {user_id}",
                }
            )
        except Exception as e:
            logger.error(f"Error revoking role: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=60)
    async def get_user_permissions(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> HandlerResult:
        """
        Get all permissions for a user.

        GET /api/knowledge/mound/governance/permissions/:user_id?workspace_id=...

        Args:
            user_id: User to get permissions for
            workspace_id: Optional workspace filter

        Returns:
            List of permissions
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not user_id:
            return error_response("user_id is required", status=400)

        try:
            permissions = await mound.get_user_permissions(
                user_id=user_id,
                workspace_id=workspace_id,
            )

            return json_response(
                {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "permissions": [p.value for p in permissions],
                }
            )
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=100)
    async def check_permission(
        self,
        user_id: str,
        permission: str,
        workspace_id: Optional[str] = None,
    ) -> HandlerResult:
        """
        Check if user has a specific permission.

        POST /api/knowledge/mound/governance/permissions/check
        {
            "user_id": "...",
            "permission": "create",
            "workspace_id": "..."  // optional
        }

        Args:
            user_id: User to check
            permission: Permission to check
            workspace_id: Optional workspace context

        Returns:
            Boolean indicating if user has permission
        """
        from aragora.knowledge.mound.ops.governance import Permission

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not user_id or not permission:
            return error_response("user_id and permission are required", status=400)

        try:
            perm = Permission(permission)
        except ValueError:
            valid_perms = [p.value for p in Permission]
            return error_response(
                f"Invalid permission. Valid permissions: {valid_perms}", status=400
            )

        try:
            has_permission = await mound.check_permission(
                user_id=user_id,
                permission=perm,
                workspace_id=workspace_id,
            )

            return json_response(
                {
                    "user_id": user_id,
                    "permission": permission,
                    "workspace_id": workspace_id,
                    "has_permission": has_permission,
                }
            )
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=30)
    async def query_audit_trail(
        self,
        actor_id: Optional[str] = None,
        action: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> HandlerResult:
        """
        Query audit trail entries.

        GET /api/knowledge/mound/governance/audit?actor_id=...&action=...&limit=100

        Args:
            actor_id: Filter by actor
            action: Filter by action type
            workspace_id: Filter by workspace
            limit: Maximum results

        Returns:
            List of audit entries
        """
        from aragora.knowledge.mound.ops.governance import AuditAction

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        # Convert action string to enum if provided
        action_enum = None
        if action:
            try:
                action_enum = AuditAction(action)
            except ValueError:
                valid_actions = [a.value for a in AuditAction]
                return error_response(f"Invalid action. Valid actions: {valid_actions}", status=400)

        try:
            entries = await mound.query_audit(
                actor_id=actor_id,
                action=action_enum,
                workspace_id=workspace_id,
                limit=limit,
            )

            return json_response(
                {
                    "filters": {
                        "actor_id": actor_id,
                        "action": action,
                        "workspace_id": workspace_id,
                    },
                    "count": len(entries),
                    "entries": [e.to_dict() for e in entries],
                }
            )
        except Exception as e:
            logger.error(f"Error querying audit trail: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=30)
    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> HandlerResult:
        """
        Get activity summary for a user.

        GET /api/knowledge/mound/governance/audit/user/:user_id?days=30

        Args:
            user_id: User to get activity for
            days: Number of days to look back

        Returns:
            Activity summary
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not user_id:
            return error_response("user_id is required", status=400)

        try:
            activity = await mound.get_user_activity(
                user_id=user_id,
                days=days,
            )

            return json_response(activity)
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=60)
    async def get_governance_stats(self) -> HandlerResult:
        """
        Get governance statistics.

        GET /api/knowledge/mound/governance/stats

        Returns:
            Governance statistics including audit summary
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            stats = mound.get_governance_stats()
            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting governance stats: {e}")
            return error_response(safe_error_message(e), status=500)
