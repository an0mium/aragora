"""
Visibility Operations Mixin for Knowledge Mound Handler.

Provides visibility and access control operations:
- Set item visibility level
- Grant/revoke access to items
- List access grants for an item
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async
from aragora.server.metrics import track_access_grant, track_visibility_change

from ...base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class VisibilityHandlerProtocol(Protocol):
    """Protocol for handlers that use VisibilityOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class VisibilityOperationsMixin:
    """Mixin providing visibility operations for KnowledgeMoundHandler."""

    @handle_errors("set visibility")
    def _handle_set_visibility(
        self: VisibilityHandlerProtocol, node_id: str, handler: Any
    ) -> HandlerResult:
        """Handle PUT /api/knowledge/mound/nodes/:id/visibility - Set item visibility."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        from aragora.knowledge.mound.types import VisibilityLevel

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        visibility_str = data.get("visibility")
        if not visibility_str:
            return error_response("visibility is required", 400)

        try:
            visibility = VisibilityLevel(visibility_str)
        except ValueError:
            valid_levels = [v.value for v in VisibilityLevel]
            return error_response(
                f"Invalid visibility level: {visibility_str}. Valid: {valid_levels}", 400
            )

        is_discoverable = data.get("is_discoverable", True)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        # Get user ID from auth
        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            _run_async(
                mound.set_visibility(
                    item_id=node_id,
                    visibility=visibility,
                    set_by=user_id,
                    is_discoverable=is_discoverable,
                )
            )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to set visibility: {e}")
            return error_response(f"Failed to set visibility: {e}", 500)

        # Track metrics
        workspace_id = getattr(user, "workspace_id", None) or "unknown"
        track_visibility_change(
            node_id=node_id,
            from_level="unknown",  # We don't fetch old value for efficiency
            to_level=visibility.value,
            workspace_id=workspace_id,
        )

        return json_response(
            {
                "success": True,
                "item_id": node_id,
                "visibility": visibility.value,
                "is_discoverable": is_discoverable,
                "set_by": user_id,
            }
        )

    @handle_errors("get visibility")
    def _handle_get_visibility(self: VisibilityHandlerProtocol, node_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/mound/nodes/:id/visibility - Get item visibility."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            node = _run_async(mound.get_node(node_id))
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return error_response(f"Failed to get node: {e}", 500)

        if not node:
            return error_response(f"Node not found: {node_id}", 404)

        # Extract visibility from node metadata
        metadata = node.metadata or {}
        visibility = metadata.get("visibility", "workspace")
        visibility_set_by = metadata.get("visibility_set_by")
        is_discoverable = metadata.get("is_discoverable", True)

        return json_response(
            {
                "item_id": node_id,
                "visibility": visibility,
                "visibility_set_by": visibility_set_by,
                "is_discoverable": is_discoverable,
            }
        )

    @handle_errors("grant access")
    def _handle_grant_access(
        self: VisibilityHandlerProtocol, node_id: str, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/knowledge/mound/nodes/:id/access - Grant access to item."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        from aragora.knowledge.mound.types import AccessGrantType

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        grantee_type_str = data.get("grantee_type")
        grantee_id = data.get("grantee_id")

        if not grantee_type_str or not grantee_id:
            return error_response("grantee_type and grantee_id are required", 400)

        try:
            grantee_type = AccessGrantType(grantee_type_str)
        except ValueError:
            valid_types = [t.value for t in AccessGrantType]
            return error_response(
                f"Invalid grantee_type: {grantee_type_str}. Valid: {valid_types}", 400
            )

        permissions = data.get("permissions", ["read"])
        expires_at_str = data.get("expires_at")
        expires_at = None
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid expires_at format. Use ISO format.", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            grant = _run_async(
                mound.grant_access(
                    item_id=node_id,
                    grantee_type=grantee_type,
                    grantee_id=grantee_id,
                    permissions=permissions,
                    granted_by=user_id,
                    expires_at=expires_at,
                )
            )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to grant access: {e}")
            return error_response(f"Failed to grant access: {e}", 500)

        # Track metrics
        workspace_id = getattr(user, "workspace_id", None) or "unknown"
        track_access_grant(
            action="grant",
            grantee_type=grantee_type.value,
            workspace_id=workspace_id,
        )

        return json_response(
            {
                "success": True,
                "grant": (
                    grant.to_dict()
                    if hasattr(grant, "to_dict")
                    else {
                        "item_id": node_id,
                        "grantee_type": grantee_type.value,
                        "grantee_id": grantee_id,
                        "permissions": permissions,
                        "granted_by": user_id,
                    }
                ),
            },
            status=201,
        )

    @handle_errors("revoke access")
    def _handle_revoke_access(
        self: VisibilityHandlerProtocol, node_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/knowledge/mound/nodes/:id/access - Revoke access."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        grantee_id = data.get("grantee_id")
        if not grantee_id:
            return error_response("grantee_id is required", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            _run_async(
                mound.revoke_access(
                    item_id=node_id,
                    grantee_id=grantee_id,
                    revoked_by=user_id,
                )
            )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return error_response(f"Failed to revoke access: {e}", 500)

        # Track metrics
        workspace_id = getattr(user, "workspace_id", None) or "unknown"
        track_access_grant(
            action="revoke",
            grantee_type="unknown",  # We don't know the type after revocation
            workspace_id=workspace_id,
        )

        return json_response(
            {
                "success": True,
                "item_id": node_id,
                "grantee_id": grantee_id,
                "revoked_by": user_id,
            }
        )

    @handle_errors("list access grants")
    def _handle_list_access_grants(
        self: VisibilityHandlerProtocol, node_id: str, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/nodes/:id/access - List access grants."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            grants = _run_async(mound.get_access_grants(item_id=node_id))
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to list access grants: {e}")
            return error_response(f"Failed to list access grants: {e}", 500)

        return json_response(
            {
                "item_id": node_id,
                "grants": [g.to_dict() if hasattr(g, "to_dict") else g for g in grants],
                "count": len(grants),
            }
        )
