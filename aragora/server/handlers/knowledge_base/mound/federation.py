"""
Federation Operations Mixin for Knowledge Mound Handler.

Provides multi-region federation operations:
- Register federated region
- Sync to/from regions
- Get federation status
- Configure federation policies
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async
from aragora.server.metrics import track_federation_sync, track_federation_regions

from ...base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class FederationHandlerProtocol(Protocol):
    """Protocol for handlers that use FederationOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...
    def require_admin_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class FederationOperationsMixin:
    """Mixin providing federation operations for KnowledgeMoundHandler."""

    @rate_limit(rpm=10, limiter_name="federation_admin")
    @handle_errors("register federated region")
    def _handle_register_region(self: FederationHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/federation/regions - Register a federated region."""
        # Admin only
        user, err = self.require_admin_or_error(handler)
        if err:
            return err

        from aragora.knowledge.mound.ops.federation import FederationMode, SyncScope

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        region_id = data.get("region_id")
        endpoint_url = data.get("endpoint_url")
        api_key = data.get("api_key")

        if not region_id:
            return error_response("region_id is required", 400)
        if not endpoint_url:
            return error_response("endpoint_url is required", 400)
        if not api_key:
            return error_response("api_key is required", 400)

        mode_str = data.get("mode", "bidirectional")
        sync_scope_str = data.get("sync_scope", "summary")

        try:
            mode = FederationMode(mode_str)
        except ValueError:
            valid_modes = [m.value for m in FederationMode]
            return error_response(f"Invalid mode: {mode_str}. Valid: {valid_modes}", 400)

        try:
            sync_scope = SyncScope(sync_scope_str)
        except ValueError:
            valid_scopes = [s.value for s in SyncScope]
            return error_response(
                f"Invalid sync_scope: {sync_scope_str}. Valid: {valid_scopes}", 400
            )

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            region = _run_async(
                mound.register_federated_region(
                    region_id=region_id,
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    mode=mode,
                    sync_scope=sync_scope,
                )
            )
        except Exception as e:
            logger.error(f"Failed to register region: {e}")
            return error_response(f"Failed to register region: {e}", 500)

        return json_response(
            {
                "success": True,
                "region": {
                    "region_id": region.region_id,
                    "endpoint_url": region.endpoint_url,
                    "mode": region.mode.value,
                    "sync_scope": region.sync_scope.value,
                    "enabled": region.enabled,
                },
            },
            status=201,
        )

    @handle_errors("unregister region")
    def _handle_unregister_region(
        self: FederationHandlerProtocol, region_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/knowledge/mound/federation/regions/:id - Unregister a region."""
        user, err = self.require_admin_or_error(handler)
        if err:
            return err

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            success = _run_async(mound.unregister_federated_region(region_id))
        except Exception as e:
            logger.error(f"Failed to unregister region: {e}")
            return error_response(f"Failed to unregister region: {e}", 500)

        if not success:
            return error_response(f"Region not found: {region_id}", 404)

        return json_response(
            {
                "success": True,
                "region_id": region_id,
            }
        )

    @rate_limit(rpm=5, limiter_name="federation_sync")
    @handle_errors("sync to region")
    def _handle_sync_to_region(self: FederationHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/federation/sync/push - Sync to a region."""
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

        region_id = data.get("region_id")
        workspace_id = data.get("workspace_id")
        since_str = data.get("since")
        visibility_levels = data.get("visibility_levels")

        if not region_id:
            return error_response("region_id is required", 400)

        since = None
        if since_str:
            try:
                since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid since format. Use ISO format.", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        with track_federation_sync(region_id, "push") as metrics_ctx:
            try:
                result = _run_async(
                    mound.sync_to_region(
                        region_id=region_id,
                        workspace_id=workspace_id,
                        since=since,
                        visibility_levels=visibility_levels,
                    )
                )
                metrics_ctx["nodes_synced"] = result.nodes_synced
                metrics_ctx["status"] = "success" if result.success else "failed"
            except Exception as e:
                metrics_ctx["status"] = "error"
                logger.error(f"Failed to sync to region: {e}")
                return error_response(f"Failed to sync to region: {e}", 500)

        return json_response(
            {
                "success": result.success,
                "region_id": result.region_id,
                "direction": result.direction,
                "nodes_synced": result.nodes_synced,
                "nodes_skipped": result.nodes_skipped,
                "nodes_failed": result.nodes_failed,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        )

    @rate_limit(rpm=5, limiter_name="federation_sync")
    @handle_errors("pull from region")
    def _handle_pull_from_region(self: FederationHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/federation/sync/pull - Pull from a region."""
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

        region_id = data.get("region_id")
        workspace_id = data.get("workspace_id")
        since_str = data.get("since")

        if not region_id:
            return error_response("region_id is required", 400)

        since = None
        if since_str:
            try:
                since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid since format. Use ISO format.", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        with track_federation_sync(region_id, "pull") as metrics_ctx:
            try:
                result = _run_async(
                    mound.pull_from_region(
                        region_id=region_id,
                        workspace_id=workspace_id,
                        since=since,
                    )
                )
                metrics_ctx["nodes_synced"] = result.nodes_synced
                metrics_ctx["status"] = "success" if result.success else "failed"
            except Exception as e:
                metrics_ctx["status"] = "error"
                logger.error(f"Failed to pull from region: {e}")
                return error_response(f"Failed to pull from region: {e}", 500)

        return json_response(
            {
                "success": result.success,
                "region_id": result.region_id,
                "direction": result.direction,
                "nodes_synced": result.nodes_synced,
                "nodes_failed": result.nodes_failed,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        )

    @handle_errors("sync all regions")
    def _handle_sync_all_regions(self: FederationHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/federation/sync/all - Sync with all regions."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id")
        since_str = data.get("since")

        since = None
        if since_str:
            try:
                since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid since format. Use ISO format.", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            results = _run_async(
                mound.sync_all_regions(
                    workspace_id=workspace_id,
                    since=since,
                )
            )
        except Exception as e:
            logger.error(f"Failed to sync all regions: {e}")
            return error_response(f"Failed to sync all regions: {e}", 500)

        return json_response(
            {
                "results": [
                    {
                        "region_id": r.region_id,
                        "direction": r.direction,
                        "success": r.success,
                        "nodes_synced": r.nodes_synced,
                        "nodes_failed": r.nodes_failed,
                        "error": r.error,
                    }
                    for r in results
                ],
                "total_regions": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            }
        )

    @handle_errors("get federation status")
    def _handle_get_federation_status(
        self: FederationHandlerProtocol, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/federation/status - Get federation status."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            status = _run_async(mound.get_federation_status())
        except Exception as e:
            logger.error(f"Failed to get federation status: {e}")
            return error_response(f"Failed to get federation status: {e}", 500)

        # Track region counts
        enabled_count = sum(1 for r in status.values() if r.get("enabled", False))
        disabled_count = len(status) - enabled_count
        healthy_count = sum(1 for r in status.values() if r.get("healthy", False))
        unhealthy_count = len(status) - healthy_count
        track_federation_regions(
            enabled=enabled_count,
            disabled=disabled_count,
            healthy=healthy_count,
            unhealthy=unhealthy_count,
        )

        return json_response(
            {
                "regions": status,
                "total_regions": len(status),
                "enabled_regions": enabled_count,
            }
        )

    @handle_errors("list federated regions")
    def _handle_list_regions(self: FederationHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/federation/regions - List federated regions."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            status = _run_async(mound.get_federation_status())
        except Exception as e:
            logger.error(f"Failed to list regions: {e}")
            return error_response(f"Failed to list regions: {e}", 500)

        regions = [
            {
                "region_id": region_id,
                **region_data,
            }
            for region_id, region_data in status.items()
        ]

        return json_response(
            {
                "regions": regions,
                "count": len(regions),
            }
        )
