"""
Knowledge Mound Checkpoint HTTP Handler.

Provides REST endpoints for managing KM checkpoints:
- GET /api/km/checkpoints - List all checkpoints
- POST /api/km/checkpoints - Create a checkpoint
- GET /api/km/checkpoints/{name} - Get checkpoint details
- DELETE /api/km/checkpoints/{name} - Delete a checkpoint
- POST /api/km/checkpoints/{name}/restore - Restore from checkpoint
- GET /api/km/checkpoints/{name}/compare - Compare with current state
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.utils.rate_limit import (
    RateLimiter,
    get_client_ip,
    rate_limit,
)
from aragora.observability.metrics import (
    record_checkpoint_operation,
    track_checkpoint_operation,
)

logger = logging.getLogger(__name__)

# Rate limiter for checkpoint endpoints (20 requests per minute)
_checkpoint_limiter = RateLimiter(requests_per_minute=20)

# More restrictive rate limiter for write operations (5 per minute)
_checkpoint_write_limiter = RateLimiter(requests_per_minute=5)


class KMCheckpointHandler(BaseHandler):
    """Handler for Knowledge Mound checkpoint management endpoints."""

    routes = [
        "/api/km/checkpoints",
        "/api/km/checkpoints/compare",
    ]

    # Dynamic routes handled via pattern matching in handle_*
    dynamic_routes = [
        "/api/km/checkpoints/{name}",
        "/api/km/checkpoints/{name}/restore",
        "/api/km/checkpoints/{name}/compare",
        "/api/km/checkpoints/{name}/download",
    ]

    def __init__(self):
        super().__init__()
        self._checkpoint_store: Optional["KMCheckpointStore"] = None

    def _get_checkpoint_store(self) -> "KMCheckpointStore":
        """Get or create the checkpoint store instance."""
        if self._checkpoint_store is None:
            try:
                from aragora.knowledge.mound.checkpoint import get_checkpoint_store

                self._checkpoint_store = get_checkpoint_store()
            except ImportError:
                raise RuntimeError("KM checkpoint module not available")
        return self._checkpoint_store

    def _check_auth(self, handler) -> tuple[Optional[dict], Optional[HandlerResult]]:
        """Check authentication for checkpoint operations.

        Returns:
            Tuple of (user_context, error_response) - one will be None
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return None, err
        return user, None

    @rate_limit(rpm=20)
    def _list_checkpoints(self, handler) -> HandlerResult:
        """List all KM checkpoints.

        GET /api/km/checkpoints

        Query params:
            limit: Maximum number to return (default: 20, max: 100)
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        try:
            store = self._get_checkpoint_store()

            # Get limit parameter
            limit = int(self.get_query_param(handler, "limit", "20"))
            limit = min(max(1, limit), 100)

            checkpoints = store.list_checkpoints(limit=limit)

            return success_response(
                {
                    "checkpoints": [
                        {
                            "name": cp.name,
                            "description": cp.description,
                            "created_at": cp.created_at.isoformat(),
                            "node_count": cp.node_count,
                            "size_bytes": cp.size_bytes,
                            "compressed": cp.compressed,
                            "tags": cp.tags,
                        }
                        for cp in checkpoints
                    ],
                    "total": len(checkpoints),
                }
            )
        except RuntimeError as e:
            logger.error("Checkpoint store not available: %s", e)
            return error_response("Checkpoint service unavailable", status=503)
        except (OSError, IOError) as e:
            logger.error("IO error listing checkpoints: %s", e)
            return error_response("Failed to list checkpoints", status=500)

    @rate_limit(rpm=5, limiter_name="km_checkpoint_write")
    def _create_checkpoint(self, handler) -> HandlerResult:
        """Create a new KM checkpoint.

        POST /api/km/checkpoints

        Body:
            name: Checkpoint name (required)
            description: Optional description
            tags: Optional list of tags
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        start_time = time.perf_counter()
        success = False

        try:
            body = self.get_request_body(handler)
            name = body.get("name")
            if not name:
                return error_response("Checkpoint name is required", status=400)

            description = body.get("description", "")
            tags = body.get("tags", [])

            store = self._get_checkpoint_store()

            with track_checkpoint_operation("create") as ctx:
                metadata = store.create_checkpoint(
                    name=name,
                    description=description,
                    tags=tags,
                )
                ctx["size_bytes"] = metadata.size_bytes

            success = True
            return success_response(
                {
                    "name": metadata.name,
                    "description": metadata.description,
                    "created_at": metadata.created_at.isoformat(),
                    "node_count": metadata.node_count,
                    "size_bytes": metadata.size_bytes,
                    "compressed": metadata.compressed,
                    "tags": metadata.tags,
                },
                status=201,
            )
        except ValueError as e:
            logger.warning("Invalid checkpoint request: %s", e)
            return error_response(str(e), status=400)
        except FileExistsError:
            return error_response("Checkpoint with this name already exists", status=409)
        except RuntimeError as e:
            logger.error("Checkpoint creation failed: %s", e)
            return error_response("Failed to create checkpoint", status=500)
        finally:
            latency = time.perf_counter() - start_time
            record_checkpoint_operation("create", success, latency)

    @rate_limit(rpm=30)
    def _get_checkpoint(self, handler, name: str) -> HandlerResult:
        """Get checkpoint details by name.

        GET /api/km/checkpoints/{name}
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        try:
            store = self._get_checkpoint_store()
            metadata = store.get_checkpoint(name)

            if metadata is None:
                return error_response(f"Checkpoint '{name}' not found", status=404)

            return success_response(
                {
                    "name": metadata.name,
                    "description": metadata.description,
                    "created_at": metadata.created_at.isoformat(),
                    "node_count": metadata.node_count,
                    "size_bytes": metadata.size_bytes,
                    "compressed": metadata.compressed,
                    "tags": metadata.tags,
                    "checksum": metadata.checksum,
                }
            )
        except RuntimeError as e:
            logger.error("Failed to get checkpoint: %s", e)
            return error_response("Checkpoint service unavailable", status=503)

    @rate_limit(rpm=5, limiter_name="km_checkpoint_write")
    def _delete_checkpoint(self, handler, name: str) -> HandlerResult:
        """Delete a checkpoint.

        DELETE /api/km/checkpoints/{name}
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        start_time = time.perf_counter()
        success = False

        try:
            store = self._get_checkpoint_store()

            if not store.delete_checkpoint(name):
                return error_response(f"Checkpoint '{name}' not found", status=404)

            success = True
            return success_response({"deleted": name})
        except RuntimeError as e:
            logger.error("Failed to delete checkpoint: %s", e)
            return error_response("Failed to delete checkpoint", status=500)
        finally:
            latency = time.perf_counter() - start_time
            record_checkpoint_operation("delete", success, latency)

    @rate_limit(rpm=3, limiter_name="km_checkpoint_restore")
    def _restore_checkpoint(self, handler, name: str) -> HandlerResult:
        """Restore KM state from a checkpoint.

        POST /api/km/checkpoints/{name}/restore

        Body:
            strategy: "merge" (default) or "replace"
            skip_duplicates: boolean (default: True)
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        start_time = time.perf_counter()
        success = False

        try:
            body = self.get_request_body(handler) or {}
            strategy = body.get("strategy", "merge")
            skip_duplicates = body.get("skip_duplicates", True)

            if strategy not in ("merge", "replace"):
                return error_response("Invalid strategy. Use 'merge' or 'replace'", status=400)

            store = self._get_checkpoint_store()
            result = store.restore_checkpoint(
                name=name,
                strategy=strategy,
                skip_duplicates=skip_duplicates,
            )

            if result is None:
                return error_response(f"Checkpoint '{name}' not found", status=404)

            success = True

            from aragora.observability.metrics import record_checkpoint_restore_result

            record_checkpoint_restore_result(
                nodes_restored=result.nodes_restored,
                nodes_skipped=result.nodes_skipped,
                errors=len(result.errors),
            )

            return success_response(
                {
                    "checkpoint_name": result.checkpoint_name,
                    "strategy": strategy,
                    "nodes_restored": result.nodes_restored,
                    "nodes_skipped": result.nodes_skipped,
                    "errors": result.errors[:10],  # Limit error list
                    "error_count": len(result.errors),
                }
            )
        except ValueError as e:
            logger.warning("Invalid restore request: %s", e)
            return error_response(str(e), status=400)
        except RuntimeError as e:
            logger.error("Restore failed: %s", e)
            return error_response("Failed to restore checkpoint", status=500)
        finally:
            latency = time.perf_counter() - start_time
            record_checkpoint_operation("restore", success, latency)

    @rate_limit(rpm=30)
    def _compare_checkpoint(self, handler, name: str) -> HandlerResult:
        """Compare checkpoint with current KM state.

        GET /api/km/checkpoints/{name}/compare
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        start_time = time.perf_counter()
        success = False

        try:
            store = self._get_checkpoint_store()
            comparison = store.compare_with_current(name)

            if comparison is None:
                return error_response(f"Checkpoint '{name}' not found", status=404)

            success = True
            return success_response(comparison)
        except RuntimeError as e:
            logger.error("Comparison failed: %s", e)
            return error_response("Failed to compare checkpoint", status=500)
        finally:
            latency = time.perf_counter() - start_time
            record_checkpoint_operation("compare", success, latency)

    @rate_limit(rpm=10, limiter_name="km_checkpoint_compare")
    def _compare_checkpoints(self, handler) -> HandlerResult:
        """Compare two checkpoints.

        POST /api/km/checkpoints/compare

        Body:
            checkpoint_a: First checkpoint name
            checkpoint_b: Second checkpoint name
        """
        user, err = self._check_auth(handler)
        if err:
            return err

        try:
            body = self.get_request_body(handler)
            checkpoint_a = body.get("checkpoint_a")
            checkpoint_b = body.get("checkpoint_b")

            if not checkpoint_a or not checkpoint_b:
                return error_response("Both checkpoint_a and checkpoint_b are required", status=400)

            store = self._get_checkpoint_store()
            comparison = store.compare_checkpoints(checkpoint_a, checkpoint_b)

            if comparison is None:
                return error_response("One or both checkpoints not found", status=404)

            return success_response(comparison)
        except RuntimeError as e:
            logger.error("Checkpoint comparison failed: %s", e)
            return error_response("Failed to compare checkpoints", status=500)

    def handle_get(self, handler) -> HandlerResult:
        """Handle GET requests."""
        path = handler.path.split("?")[0]

        if path == "/api/km/checkpoints":
            return self._list_checkpoints(handler)

        # Handle /api/km/checkpoints/{name} or /api/km/checkpoints/{name}/compare
        if path.startswith("/api/km/checkpoints/"):
            parts = path.replace("/api/km/checkpoints/", "").split("/")
            name = parts[0]

            if len(parts) == 1:
                return self._get_checkpoint(handler, name)
            elif len(parts) == 2 and parts[1] == "compare":
                return self._compare_checkpoint(handler, name)

        return error_response("Not found", status=404)

    def handle_post(self, handler) -> HandlerResult:
        """Handle POST requests."""
        path = handler.path.split("?")[0]

        if path == "/api/km/checkpoints":
            return self._create_checkpoint(handler)

        if path == "/api/km/checkpoints/compare":
            return self._compare_checkpoints(handler)

        # Handle /api/km/checkpoints/{name}/restore
        if path.startswith("/api/km/checkpoints/") and path.endswith("/restore"):
            name = path.replace("/api/km/checkpoints/", "").replace("/restore", "")
            return self._restore_checkpoint(handler, name)

        return error_response("Not found", status=404)

    def handle_delete(self, handler) -> HandlerResult:
        """Handle DELETE requests."""
        path = handler.path.split("?")[0]

        # Handle /api/km/checkpoints/{name}
        if path.startswith("/api/km/checkpoints/"):
            name = path.replace("/api/km/checkpoints/", "")
            if "/" not in name:  # Ensure it's just the name
                return self._delete_checkpoint(handler, name)

        return error_response("Not found", status=404)
