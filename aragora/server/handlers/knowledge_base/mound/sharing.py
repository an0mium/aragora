"""
Sharing Operations Mixin for Knowledge Mound Handler.

Provides cross-workspace sharing operations:
- Share item with workspace/user
- List items shared with me
- Revoke shares
- Update share permissions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission
from aragora.server.http_utils import run_async as _run_async
from aragora.server.metrics import track_share

from ...base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class SharingHandlerProtocol(Protocol):
    """Protocol for handlers that use SharingOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class SharingOperationsMixin:
    """Mixin providing cross-workspace sharing operations for KnowledgeMoundHandler."""

    @require_permission("sharing:create")
    @rate_limit(rpm=30, limiter_name="knowledge_share")
    @handle_errors("share item")
    def _handle_share_item(self: SharingHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/share - Share item with workspace/user."""
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

        item_id = data.get("item_id")
        target_type = data.get("target_type")  # "workspace" or "user"
        target_id = data.get("target_id")

        if not item_id:
            return error_response("item_id is required", 400)
        if not target_type or target_type not in ("workspace", "user"):
            return error_response("target_type must be 'workspace' or 'user'", 400)
        if not target_id:
            return error_response("target_id is required", 400)

        permissions = data.get("permissions", ["read"])
        expires_at_str = data.get("expires_at")
        expires_at = None
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid expires_at format. Use ISO format.", 400)

        message = data.get("message")  # Optional sharing message

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")
        workspace_id = data.get("from_workspace_id", "default")

        try:
            if target_type == "workspace":
                _run_async(
                    mound.share_with_workspace(  # type: ignore[misc]
                        item_id=item_id,
                        from_workspace_id=workspace_id,
                        to_workspace_id=target_id,
                        shared_by=user_id,
                        permissions=permissions,
                        expires_at=expires_at,
                    )
                )
            else:
                _run_async(
                    mound.share_with_user(  # type: ignore[misc,call-arg]
                        item_id=item_id,
                        from_workspace_id=workspace_id,
                        to_user_id=target_id,
                        shared_by=user_id,
                        permissions=permissions,
                        expires_at=expires_at,
                    )
                )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to share item: {e}")
            return error_response(f"Failed to share item: {e}", 500)

        # Track metrics
        track_share(action="share", target_type=target_type)

        return json_response(
            {
                "success": True,
                "share": {
                    "item_id": item_id,
                    "target_type": target_type,
                    "target_id": target_id,
                    "permissions": permissions,
                    "shared_by": user_id,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "message": message,
                },
            },
            status=201,
        )

    @require_permission("sharing:read")
    @handle_errors("list shared with me")
    def _handle_shared_with_me(
        self: SharingHandlerProtocol, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/shared-with-me - Get items shared with workspace/user."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)
        include_expired = query_params.get("include_expired", ["false"])[0].lower() == "true"

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            items = _run_async(
                mound.get_shared_with_me(  # type: ignore[misc,call-arg]
                    workspace_id=workspace_id,
                    user_id=user_id,
                    limit=limit,
                    include_expired=include_expired,
                )
            )
        except Exception as e:
            logger.error(f"Failed to get shared items: {e}")
            return error_response(f"Failed to get shared items: {e}", 500)

        return json_response(
            {
                "items": [
                    (
                        item.to_dict()
                        if hasattr(item, "to_dict")
                        else {
                            "id": getattr(item, "id", "unknown"),
                            "content": getattr(item, "content", ""),
                        }
                    )
                    for item in items[offset : offset + limit]
                ],
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        )

    @require_permission("sharing:create")
    @rate_limit(rpm=30, limiter_name="knowledge_share")
    @handle_errors("revoke share")
    def _handle_revoke_share(self: SharingHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle DELETE /api/knowledge/mound/share - Revoke a share."""
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

        item_id = data.get("item_id")
        grantee_id = data.get("grantee_id")

        if not item_id or not grantee_id:
            return error_response("item_id and grantee_id are required", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            _run_async(
                mound.revoke_share(  # type: ignore[misc]
                    item_id=item_id,
                    grantee_id=grantee_id,
                    revoked_by=user_id,
                )
            )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to revoke share: {e}")
            return error_response(f"Failed to revoke share: {e}", 500)

        return json_response(
            {
                "success": True,
                "item_id": item_id,
                "grantee_id": grantee_id,
                "revoked_by": user_id,
            }
        )

    @require_permission("sharing:read")
    @handle_errors("list my shares")
    def _handle_my_shares(
        self: SharingHandlerProtocol, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/my-shares - List items I've shared."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            grants = _run_async(
                mound.get_share_grants(  # type: ignore[misc,call-arg]
                    shared_by=user_id,
                    workspace_id=workspace_id,
                )
            )
        except Exception as e:
            logger.error(f"Failed to list shares: {e}")
            return error_response(f"Failed to list shares: {e}", 500)

        return json_response(
            {
                "grants": [
                    g.to_dict() if hasattr(g, "to_dict") else g
                    for g in grants[offset : offset + limit]
                ],
                "count": len(grants),
                "limit": limit,
                "offset": offset,
            }
        )

    @require_permission("sharing:create")
    @handle_errors("update share permissions")
    def _handle_update_share(self: SharingHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle PATCH /api/knowledge/mound/share - Update share permissions."""
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

        item_id = data.get("item_id")
        grantee_id = data.get("grantee_id")
        permissions = data.get("permissions")
        expires_at_str = data.get("expires_at")

        if not item_id or not grantee_id:
            return error_response("item_id and grantee_id are required", 400)
        if not permissions and not expires_at_str:
            return error_response("permissions or expires_at is required", 400)

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
            updated_grant = _run_async(
                mound.update_share_permissions(  # type: ignore[misc,call-arg]
                    item_id=item_id,
                    grantee_id=grantee_id,
                    permissions=permissions,
                    expires_at=expires_at,
                    updated_by=user_id,
                )
            )
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to update share: {e}")
            return error_response(f"Failed to update share: {e}", 500)

        return json_response(
            {
                "success": True,
                "grant": (
                    updated_grant.to_dict()
                    if hasattr(updated_grant, "to_dict")
                    else {
                        "item_id": item_id,
                        "grantee_id": grantee_id,
                        "permissions": permissions,
                    }
                ),
            }
        )
