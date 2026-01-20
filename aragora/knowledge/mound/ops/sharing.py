"""
Cross-Workspace Knowledge Sharing Mixin for Knowledge Mound.

Provides operations for sharing knowledge across workspaces:
- share_with_workspace: Share an item with another workspace
- share_with_user: Share an item with a specific user
- get_shared_with_me: Get items shared with the current workspace/user
- revoke_share: Revoke a sharing grant
- get_share_grants: Get all sharing grants for an item

This integrates with the CrossWorkspaceCoordinator for federation
and audit logging of data sharing consent.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        AccessGrant,
        AccessGrantType,
        KnowledgeItem,
        MoundConfig,
    )

logger = logging.getLogger(__name__)


class SharingProtocol(Protocol):
    """Protocol defining expected interface for Sharing mixin."""

    config: "MoundConfig"
    workspace_id: str
    _meta_store: Optional[Any]
    _cache: Optional[Any]
    _initialized: bool

    def _ensure_initialized(self) -> None: ...

    async def get(self, node_id: str, workspace_id: Optional[str] = None) -> Optional["KnowledgeItem"]: ...


class KnowledgeSharingMixin:
    """Mixin providing cross-workspace sharing for KnowledgeMound."""

    async def share_with_workspace(
        self: SharingProtocol,
        item_id: str,
        from_workspace_id: str,
        to_workspace_id: str,
        shared_by: str,
        permissions: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> "AccessGrant":
        """
        Share a knowledge item with another workspace.

        Creates an access grant allowing the target workspace to view
        (and optionally edit) the knowledge item.

        Args:
            item_id: ID of the item to share
            from_workspace_id: Workspace that owns the item
            to_workspace_id: Workspace to share with
            shared_by: User ID who is sharing
            permissions: List of permissions (default: ["read"])
            expires_at: Optional expiration datetime

        Returns:
            The created AccessGrant

        Raises:
            ValueError: If item not found or sharing with self
        """
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        self._ensure_initialized()

        if from_workspace_id == to_workspace_id:
            raise ValueError("Cannot share item with the same workspace")

        # Verify item exists
        item = await self.get(item_id, workspace_id=from_workspace_id)
        if not item:
            raise ValueError(f"Item {item_id} not found in workspace {from_workspace_id}")

        grant = AccessGrant(
            id=f"grant_{uuid4().hex[:12]}",
            item_id=item_id,
            grantee_type=AccessGrantType.WORKSPACE,
            grantee_id=to_workspace_id,
            permissions=permissions or ["read"],
            granted_by=shared_by,
            granted_at=datetime.now(),
            expires_at=expires_at,
        )

        # Persist grant
        if hasattr(self._meta_store, "save_access_grant_async"):
            await self._meta_store.save_access_grant_async(grant)
        else:
            logger.warning("Store does not support access grants, grant not persisted")

        # Record in federation coordinator if available
        await self._record_sharing_consent(
            from_workspace_id=from_workspace_id,
            to_workspace_id=to_workspace_id,
            scope="knowledge_item",
            operations=permissions or ["read"],
            granted_by=shared_by,
        )

        logger.info(
            f"Shared item {item_id} from {from_workspace_id} "
            f"to workspace {to_workspace_id} by {shared_by}"
        )

        return grant

    async def share_with_user(
        self: SharingProtocol,
        item_id: str,
        from_workspace_id: str,
        user_id: str,
        shared_by: str,
        permissions: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> "AccessGrant":
        """
        Share a knowledge item with a specific user.

        Creates an access grant allowing the user to view (and optionally edit)
        the knowledge item, regardless of their workspace.

        Args:
            item_id: ID of the item to share
            from_workspace_id: Workspace that owns the item
            user_id: User to share with
            shared_by: User ID who is sharing
            permissions: List of permissions (default: ["read"])
            expires_at: Optional expiration datetime

        Returns:
            The created AccessGrant
        """
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        self._ensure_initialized()

        # Verify item exists
        item = await self.get(item_id, workspace_id=from_workspace_id)
        if not item:
            raise ValueError(f"Item {item_id} not found in workspace {from_workspace_id}")

        grant = AccessGrant(
            id=f"grant_{uuid4().hex[:12]}",
            item_id=item_id,
            grantee_type=AccessGrantType.USER,
            grantee_id=user_id,
            permissions=permissions or ["read"],
            granted_by=shared_by,
            granted_at=datetime.now(),
            expires_at=expires_at,
        )

        # Persist grant
        if hasattr(self._meta_store, "save_access_grant_async"):
            await self._meta_store.save_access_grant_async(grant)
        else:
            logger.warning("Store does not support access grants, grant not persisted")

        logger.info(
            f"Shared item {item_id} from {from_workspace_id} "
            f"to user {user_id} by {shared_by}"
        )

        return grant

    async def get_shared_with_me(
        self: SharingProtocol,
        workspace_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List["KnowledgeItem"]:
        """
        Get knowledge items shared with this workspace or user.

        Args:
            workspace_id: Current workspace ID
            user_id: Optional user ID for user-specific shares
            limit: Maximum number of items to return

        Returns:
            List of shared KnowledgeItems
        """
        from aragora.knowledge.mound.types import AccessGrantType

        self._ensure_initialized()

        items: List["KnowledgeItem"] = []
        seen_ids = set()

        if hasattr(self._meta_store, "get_grants_for_grantee_async"):
            # Get workspace grants
            workspace_grants = await self._meta_store.get_grants_for_grantee_async(
                workspace_id, AccessGrantType.WORKSPACE
            )

            # Get user grants if user_id provided
            user_grants = []
            if user_id:
                user_grants = await self._meta_store.get_grants_for_grantee_async(
                    user_id, AccessGrantType.USER
                )

            all_grants = workspace_grants + user_grants

            # Fetch items
            for grant in all_grants:
                if grant.is_expired():
                    continue
                if grant.item_id in seen_ids:
                    continue
                seen_ids.add(grant.item_id)

                item = await self.get(grant.item_id)
                if item:
                    items.append(item)

                if len(items) >= limit:
                    break

        return items

    async def revoke_share(
        self: SharingProtocol,
        item_id: str,
        grantee_id: str,
        revoked_by: str,
    ) -> bool:
        """
        Revoke a sharing grant.

        Args:
            item_id: ID of the shared item
            grantee_id: ID of the grantee (user or workspace)
            revoked_by: User ID who is revoking

        Returns:
            True if grant was revoked, False if not found
        """
        self._ensure_initialized()

        if hasattr(self._meta_store, "delete_access_grant_async"):
            result = await self._meta_store.delete_access_grant_async(item_id, grantee_id)
            if result:
                logger.info(
                    f"Revoked share for item {item_id} from {grantee_id} by {revoked_by}"
                )
            return result

        logger.warning("Store does not support access grants")
        return False

    async def get_share_grants(
        self: SharingProtocol,
        item_id: str,
    ) -> List["AccessGrant"]:
        """
        Get all sharing grants for an item.

        Args:
            item_id: ID of the item

        Returns:
            List of AccessGrant objects
        """
        self._ensure_initialized()

        if hasattr(self._meta_store, "get_access_grants_async"):
            return await self._meta_store.get_access_grants_async(item_id)

        return []

    async def update_share_permissions(
        self: SharingProtocol,
        item_id: str,
        grantee_id: str,
        new_permissions: List[str],
        updated_by: str,
    ) -> Optional["AccessGrant"]:
        """
        Update permissions for an existing share grant.

        Args:
            item_id: ID of the shared item
            grantee_id: ID of the grantee
            new_permissions: New permissions list
            updated_by: User ID making the update

        Returns:
            Updated AccessGrant or None if not found
        """
        self._ensure_initialized()

        # Get existing grant
        grants = await self.get_share_grants(item_id)
        existing = next((g for g in grants if g.grantee_id == grantee_id), None)

        if not existing:
            return None

        # Create updated grant
        from aragora.knowledge.mound.types import AccessGrant

        updated_grant = AccessGrant(
            id=existing.id,
            item_id=existing.item_id,
            grantee_type=existing.grantee_type,
            grantee_id=existing.grantee_id,
            permissions=new_permissions,
            granted_by=updated_by,
            granted_at=datetime.now(),
            expires_at=existing.expires_at,
        )

        if hasattr(self._meta_store, "save_access_grant_async"):
            await self._meta_store.save_access_grant_async(updated_grant)

        logger.info(
            f"Updated permissions for item {item_id} grantee {grantee_id}: {new_permissions}"
        )

        return updated_grant

    async def _record_sharing_consent(
        self: SharingProtocol,
        from_workspace_id: str,
        to_workspace_id: str,
        scope: str,
        operations: List[str],
        granted_by: str,
    ) -> None:
        """
        Record sharing consent with the CrossWorkspaceCoordinator.

        This provides audit logging for data sharing compliance.
        """
        try:
            from aragora.coordination.cross_workspace import (
                CrossWorkspaceCoordinator,
                DataSharingConsent,
            )

            coordinator = CrossWorkspaceCoordinator()
            consent = DataSharingConsent(
                from_workspace_id=from_workspace_id,
                to_workspace_id=to_workspace_id,
                scope=scope,
                operations=operations,
                granted_by=granted_by,
            )
            await coordinator.record_consent(consent)
        except ImportError:
            # CrossWorkspaceCoordinator not available
            pass
        except Exception as e:
            logger.warning(f"Failed to record sharing consent: {e}")
