"""
Data Isolation Manager.

Ensures strict data isolation between workspaces/tenants.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class WorkspacePermission(str, Enum):
    """Permissions for workspace access."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXPORT = "export"


class AccessDeniedException(Exception):
    """Raised when access to a workspace is denied."""

    def __init__(
        self,
        message: str,
        workspace_id: str,
        actor: str,
        action: str,
    ):
        super().__init__(message)
        self.workspace_id = workspace_id
        self.actor = actor
        self.action = action


@dataclass
class IsolationConfig:
    """Configuration for data isolation."""

    # Encryption
    enable_encryption_at_rest: bool = True
    encryption_algorithm: str = "AES-256-GCM"

    # Access control
    require_workspace_membership: bool = True
    default_permissions: list[WorkspacePermission] = field(
        default_factory=lambda: [WorkspacePermission.READ]
    )

    # Audit
    log_all_access: bool = True
    log_retention_days: int = 365

    # Storage
    workspace_data_root: str = "/var/aragora/workspaces"
    use_separate_databases: bool = False


@dataclass
class WorkspaceMember:
    """A member of a workspace with permissions."""

    user_id: str
    permissions: list[WorkspacePermission]
    added_at: datetime = field(default_factory=datetime.utcnow)
    added_by: str = ""


@dataclass
class Workspace:
    """An isolated workspace/tenant."""

    id: str
    organization_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

    # Encryption
    encryption_key_id: str = ""
    encrypted: bool = False

    # Membership
    members: dict[str, WorkspaceMember] = field(default_factory=dict)

    # Settings
    retention_days: int = 90
    sensitivity_level: str = "internal"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Statistics
    document_count: int = 0
    storage_bytes: int = 0
    last_accessed: datetime | None = None

    def has_permission(self, user_id: str, permission: WorkspacePermission) -> bool:
        """Check if a user has a specific permission."""
        member = self.members.get(user_id)
        if not member:
            return False

        return permission in member.permissions or WorkspacePermission.ADMIN in member.permissions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "encrypted": self.encrypted,
            "member_count": len(self.members),
            "retention_days": self.retention_days,
            "sensitivity_level": self.sensitivity_level,
            "document_count": self.document_count,
            "storage_bytes": self.storage_bytes,
        }


class DataIsolationManager:
    """
    Manages data isolation between workspaces.

    Ensures:
    - Documents are only accessible within their workspace
    - Cross-workspace data leakage is prevented
    - Access is properly logged
    """

    def __init__(self, config: IsolationConfig | None = None):
        self.config = config or IsolationConfig()
        self._workspaces: dict[str, Workspace] = {}
        self._audit_log: list[dict] = []

    async def create_workspace(
        self,
        organization_id: str,
        name: str,
        created_by: str,
        initial_members: list[str] | None = None,
    ) -> Workspace:
        """
        Create a new isolated workspace.

        Args:
            organization_id: Organization owning the workspace
            name: Display name for the workspace
            created_by: User ID creating the workspace
            initial_members: Optional list of initial member user IDs

        Returns:
            Created workspace
        """
        workspace_id = str(uuid4())

        workspace = Workspace(
            id=workspace_id,
            organization_id=organization_id,
            name=name,
            created_by=created_by,
        )

        # Add creator as admin
        workspace.members[created_by] = WorkspaceMember(
            user_id=created_by,
            permissions=list(WorkspacePermission),
            added_by=created_by,
        )

        # Add initial members with default permissions
        for member_id in initial_members or []:
            if member_id != created_by:
                workspace.members[member_id] = WorkspaceMember(
                    user_id=member_id,
                    permissions=list(self.config.default_permissions),
                    added_by=created_by,
                )

        # Setup encryption if enabled
        if self.config.enable_encryption_at_rest:
            workspace.encryption_key_id = await self._generate_encryption_key(workspace_id)
            workspace.encrypted = True

        # Create workspace storage directory
        await self._setup_workspace_storage(workspace)

        self._workspaces[workspace_id] = workspace

        # Log creation
        await self._log_access(
            workspace_id=workspace_id,
            actor=created_by,
            action="create_workspace",
            outcome="success",
        )

        logger.info(f"Created workspace {workspace_id} for org {organization_id}")

        return workspace

    async def get_workspace(
        self,
        workspace_id: str,
        actor: str,
    ) -> Workspace:
        """
        Get a workspace by ID.

        Args:
            workspace_id: Workspace ID
            actor: User requesting access

        Returns:
            Workspace if accessible

        Raises:
            AccessDeniedException: If user cannot access workspace
        """
        workspace = self._workspaces.get(workspace_id)
        if not workspace:
            raise AccessDeniedException(
                f"Workspace not found: {workspace_id}",
                workspace_id=workspace_id,
                actor=actor,
                action="get_workspace",
            )

        # Check access
        if self.config.require_workspace_membership:
            if not workspace.has_permission(actor, WorkspacePermission.READ):
                await self._log_access(
                    workspace_id=workspace_id,
                    actor=actor,
                    action="get_workspace",
                    outcome="denied",
                )
                raise AccessDeniedException(
                    f"Access denied to workspace {workspace_id}",
                    workspace_id=workspace_id,
                    actor=actor,
                    action="get_workspace",
                )

        workspace.last_accessed = datetime.utcnow()

        if self.config.log_all_access:
            await self._log_access(
                workspace_id=workspace_id,
                actor=actor,
                action="get_workspace",
                outcome="success",
            )

        return workspace

    async def check_access(
        self,
        workspace_id: str,
        actor: str,
        permission: WorkspacePermission,
    ) -> bool:
        """
        Check if an actor has permission for a workspace.

        Args:
            workspace_id: Workspace to check
            actor: User to check
            permission: Required permission

        Returns:
            True if access is allowed
        """
        workspace = self._workspaces.get(workspace_id)
        if not workspace:
            return False

        return workspace.has_permission(actor, permission)

    async def require_access(
        self,
        workspace_id: str,
        actor: str,
        permission: WorkspacePermission,
        action: str = "access",
    ) -> None:
        """
        Require that an actor has permission, raising if not.

        Args:
            workspace_id: Workspace to check
            actor: User to check
            permission: Required permission
            action: Action being performed (for logging)

        Raises:
            AccessDeniedException: If access is denied
        """
        has_access = await self.check_access(workspace_id, actor, permission)

        if not has_access:
            await self._log_access(
                workspace_id=workspace_id,
                actor=actor,
                action=action,
                outcome="denied",
                details={"required_permission": permission.value},
            )
            raise AccessDeniedException(
                f"Permission {permission.value} required for {action}",
                workspace_id=workspace_id,
                actor=actor,
                action=action,
            )

        await self._log_access(
            workspace_id=workspace_id,
            actor=actor,
            action=action,
            outcome="success",
        )

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        permissions: list[WorkspacePermission],
        added_by: str,
    ) -> None:
        """Add a member to a workspace."""
        await self.require_access(
            workspace_id,
            added_by,
            WorkspacePermission.ADMIN,
            "add_member",
        )

        workspace = self._workspaces[workspace_id]
        workspace.members[user_id] = WorkspaceMember(
            user_id=user_id,
            permissions=permissions,
            added_by=added_by,
        )

        logger.info(f"Added member {user_id} to workspace {workspace_id}")

    async def remove_member(
        self,
        workspace_id: str,
        user_id: str,
        removed_by: str,
    ) -> None:
        """Remove a member from a workspace."""
        await self.require_access(
            workspace_id,
            removed_by,
            WorkspacePermission.ADMIN,
            "remove_member",
        )

        workspace = self._workspaces[workspace_id]
        if user_id in workspace.members:
            del workspace.members[user_id]
            logger.info(f"Removed member {user_id} from workspace {workspace_id}")

    async def delete_workspace(
        self,
        workspace_id: str,
        deleted_by: str,
        force: bool = False,
    ) -> None:
        """
        Delete a workspace and all its data.

        Args:
            workspace_id: Workspace to delete
            deleted_by: User performing deletion
            force: Skip confirmation checks
        """
        await self.require_access(
            workspace_id,
            deleted_by,
            WorkspacePermission.ADMIN,
            "delete_workspace",
        )

        workspace = self._workspaces.get(workspace_id)
        if not workspace:
            return

        # Clean up storage
        await self._cleanup_workspace_storage(workspace)

        # Delete encryption key
        if workspace.encryption_key_id:
            await self._delete_encryption_key(workspace.encryption_key_id)

        del self._workspaces[workspace_id]

        await self._log_access(
            workspace_id=workspace_id,
            actor=deleted_by,
            action="delete_workspace",
            outcome="success",
        )

        logger.info(f"Deleted workspace {workspace_id}")

    async def encrypt_data(
        self,
        workspace_id: str,
        data: bytes,
    ) -> bytes:
        """
        Encrypt data for a workspace.

        Args:
            workspace_id: Workspace owning the data
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        workspace = self._workspaces.get(workspace_id)
        if not workspace or not workspace.encrypted:
            return data

        # Use workspace's encryption key
        try:
            from aragora.security.encryption import encrypt_data

            return encrypt_data(data, workspace.encryption_key_id)
        except ImportError:
            logger.warning("Encryption module not available")
            return data

    async def decrypt_data(
        self,
        workspace_id: str,
        data: bytes,
    ) -> bytes:
        """
        Decrypt data for a workspace.

        Args:
            workspace_id: Workspace owning the data
            data: Data to decrypt

        Returns:
            Decrypted data
        """
        workspace = self._workspaces.get(workspace_id)
        if not workspace or not workspace.encrypted:
            return data

        try:
            from aragora.security.encryption import decrypt_data

            return decrypt_data(data, workspace.encryption_key_id)
        except ImportError:
            logger.warning("Encryption module not available")
            return data

    async def list_workspaces(
        self,
        actor: str,
        organization_id: str | None = None,
    ) -> list[Workspace]:
        """List workspaces accessible to an actor."""
        accessible = []

        for workspace in self._workspaces.values():
            if organization_id and workspace.organization_id != organization_id:
                continue

            if workspace.has_permission(actor, WorkspacePermission.READ):
                accessible.append(workspace)

        return accessible

    async def _generate_encryption_key(self, workspace_id: str) -> str:
        """Generate an encryption key for a workspace."""
        key_id = f"wsk_{workspace_id}_{secrets.token_hex(8)}"
        # In production, this would integrate with a KMS
        logger.debug(f"Generated encryption key {key_id}")
        return key_id

    async def _delete_encryption_key(self, key_id: str) -> None:
        """Delete an encryption key."""
        # In production, this would integrate with a KMS
        logger.debug(f"Deleted encryption key {key_id}")

    async def _setup_workspace_storage(self, workspace: Workspace) -> None:
        """Setup storage directory for a workspace."""
        storage_path = Path(self.config.workspace_data_root) / workspace.id
        storage_path.mkdir(parents=True, exist_ok=True)

    async def _cleanup_workspace_storage(self, workspace: Workspace) -> None:
        """Clean up storage for a deleted workspace."""
        import shutil

        storage_path = Path(self.config.workspace_data_root) / workspace.id
        if storage_path.exists():
            shutil.rmtree(storage_path)

    async def _log_access(
        self,
        workspace_id: str,
        actor: str,
        action: str,
        outcome: str,
        details: dict | None = None,
    ) -> None:
        """Log an access event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "workspace_id": workspace_id,
            "actor": actor,
            "action": action,
            "outcome": outcome,
            "details": details or {},
        }
        self._audit_log.append(entry)

        # Trim old entries
        max_entries = 10000
        if len(self._audit_log) > max_entries:
            self._audit_log = self._audit_log[-max_entries:]


# Global instance
_isolation_manager: DataIsolationManager | None = None


def get_isolation_manager(
    config: IsolationConfig | None = None,
) -> DataIsolationManager:
    """Get or create the global isolation manager."""
    global _isolation_manager
    if _isolation_manager is None:
        _isolation_manager = DataIsolationManager(config)
    return _isolation_manager


__all__ = [
    "DataIsolationManager",
    "IsolationConfig",
    "Workspace",
    "WorkspacePermission",
    "WorkspaceMember",
    "AccessDeniedException",
    "get_isolation_manager",
]
