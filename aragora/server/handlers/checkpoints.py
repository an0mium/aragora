"""
Checkpoint management endpoint handlers.

Endpoints:
- GET /api/checkpoints - List all checkpoints
- GET /api/checkpoints/{id} - Get checkpoint details
- POST /api/checkpoints/{id}/resume - Resume debate from checkpoint
- DELETE /api/checkpoints/{id} - Delete checkpoint
- GET /api/debates/{id}/checkpoints - List checkpoints for a debate
- POST /api/debates/{id}/checkpoint - Create checkpoint for running debate
- POST /api/debates/{id}/pause - Pause debate and create checkpoint
"""

from __future__ import annotations

__all__ = [
    "CheckpointHandler",
]

import logging
from typing import Any, Dict, Optional

from aragora.debate.checkpoint import (
    CheckpointManager,
    DatabaseCheckpointStore,
)

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    handle_errors,
    json_response,
    safe_json_parse,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for checkpoint endpoints (30 requests per minute)
_checkpoint_limiter = RateLimiter(requests_per_minute=30)


class CheckpointHandler(BaseHandler):
    """Handler for checkpoint management endpoints."""

    ROUTES = [
        "/api/v1/checkpoints",
        "/api/v1/checkpoints/resumable",
    ]

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self._checkpoint_manager: Optional[CheckpointManager] = None

    def _get_checkpoint_manager(self) -> CheckpointManager:
        """Get or create checkpoint manager instance."""
        if self._checkpoint_manager is None:
            # Use database store for persistence
            store = DatabaseCheckpointStore()
            self._checkpoint_manager = CheckpointManager(store=store)
        return self._checkpoint_manager

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return (
            path == "/api/v1/checkpoints"
            or path == "/api/v1/checkpoints/resumable"
            or path.startswith("/api/v1/checkpoints/")
            or (path.startswith("/api/v1/debates/") and "/checkpoint" in path)
        )

    @handle_errors("checkpoint handling")  # type: ignore[arg-type]
    async def handle(  # type: ignore[override]
        self,
        path: str,
        query_params: Dict[str, str],
        handler: Any,
        body: Optional[bytes] = None,
    ) -> HandlerResult:
        """Route checkpoint requests to appropriate method."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _checkpoint_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for checkpoint endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        method = handler.command

        # GET /api/checkpoints
        if path == "/api/v1/checkpoints" and method == "GET":
            return await self.list_checkpoints(query_params)

        # GET /api/checkpoints/resumable
        if path == "/api/v1/checkpoints/resumable" and method == "GET":
            return await self.list_resumable_debates()

        # /api/checkpoints/{id}...
        if path.startswith("/api/v1/checkpoints/") and not path.startswith(
            "/api/v1/checkpoints/resumable"
        ):
            parts = path.split("/")
            if len(parts) >= 4:
                checkpoint_id = parts[3]

                # GET /api/checkpoints/{id}
                if method == "GET" and len(parts) == 4:
                    return await self.get_checkpoint(checkpoint_id)

                # POST /api/checkpoints/{id}/resume
                if method == "POST" and len(parts) >= 5 and parts[4] == "resume":
                    return await self.resume_checkpoint(checkpoint_id, body)

                # DELETE /api/checkpoints/{id}
                if method == "DELETE" and len(parts) == 4:
                    return await self.delete_checkpoint(checkpoint_id)

                # POST /api/checkpoints/{id}/intervention
                if method == "POST" and len(parts) >= 5 and parts[4] == "intervention":
                    return await self.add_intervention(checkpoint_id, body)

        # /api/debates/{id}/checkpoint...
        if path.startswith("/api/v1/debates/") and "/checkpoint" in path:
            parts = path.split("/")
            if len(parts) >= 4:
                debate_id = parts[3]

                # GET /api/debates/{id}/checkpoints
                if method == "GET" and len(parts) >= 5 and parts[4] == "checkpoints":
                    return await self.list_debate_checkpoints(debate_id, query_params)

                # POST /api/debates/{id}/checkpoint
                if method == "POST" and len(parts) == 5 and parts[4] == "checkpoint":
                    return await self.create_checkpoint(debate_id, body)

                # POST /api/debates/{id}/checkpoint/pause
                if (
                    method == "POST"
                    and len(parts) >= 6
                    and parts[4] == "checkpoint"
                    and parts[5] == "pause"
                ):
                    return await self.pause_debate(debate_id, body)

        return error_response("Not found", 404)

    async def list_checkpoints(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        GET /api/checkpoints

        List all checkpoints with optional filtering.

        Query params:
        - debate_id: Filter by debate ID
        - status: Filter by status (complete, resuming, corrupted, expired)
        - limit: Maximum results (default 50)
        - offset: Pagination offset
        """
        manager = self._get_checkpoint_manager()

        debate_id = query_params.get("debate_id")
        status_filter = query_params.get("status")
        limit = get_int_param(query_params, "limit", 50)
        offset = get_int_param(query_params, "offset", 0)

        # List checkpoints from store
        all_checkpoints = await manager.store.list_checkpoints(debate_id=debate_id)

        # Filter by status if requested
        if status_filter:
            all_checkpoints = [cp for cp in all_checkpoints if cp.get("status") == status_filter]

        # Apply pagination
        total = len(all_checkpoints)
        checkpoints = all_checkpoints[offset : offset + limit]

        return json_response(
            {
                "checkpoints": checkpoints,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    async def list_resumable_debates(self) -> HandlerResult:
        """
        GET /api/checkpoints/resumable

        List all debates that have resumable checkpoints.
        """
        manager = self._get_checkpoint_manager()
        debates = await manager.list_debates_with_checkpoints()

        return json_response(
            {
                "debates": debates,
                "total": len(debates),
            }
        )

    async def get_checkpoint(self, checkpoint_id: str) -> HandlerResult:
        """
        GET /api/checkpoints/{id}

        Get detailed information about a specific checkpoint.
        """
        manager = self._get_checkpoint_manager()
        checkpoint = await manager.store.load(checkpoint_id)

        if not checkpoint:
            return error_response(f"Checkpoint not found: {checkpoint_id}", 404)

        # Include integrity status
        checkpoint_dict = checkpoint.to_dict()
        checkpoint_dict["integrity_valid"] = checkpoint.verify_integrity()

        return json_response({"checkpoint": checkpoint_dict})

    async def resume_checkpoint(self, checkpoint_id: str, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/checkpoints/{id}/resume

        Resume a debate from a checkpoint.

        Request body (optional):
        {
            "resumed_by": "user_id",
            "modifications": {
                "task": "Modified task description",
                "additional_rounds": 2
            }
        }
        """
        manager = self._get_checkpoint_manager()

        # Parse request body
        data = safe_json_parse(body) or {}
        resumed_by = data.get("resumed_by", "api")

        # Resume from checkpoint
        resumed = await manager.resume_from_checkpoint(
            checkpoint_id=checkpoint_id,
            resumed_by=resumed_by,
        )

        if not resumed:
            return error_response(
                f"Checkpoint not found or corrupted: {checkpoint_id}",
                404,
            )

        return json_response(
            {
                "message": "Debate resumed from checkpoint",
                "resumed_debate": {
                    "original_debate_id": resumed.original_debate_id,
                    "checkpoint_id": checkpoint_id,
                    "resumed_at": resumed.resumed_at,
                    "resumed_by": resumed.resumed_by,
                    "current_round": resumed.checkpoint.current_round,
                    "total_rounds": resumed.checkpoint.total_rounds,
                    "task": resumed.checkpoint.task,
                    "message_count": len(resumed.messages),
                    "vote_count": len(resumed.votes),
                },
            },
            status=200,
        )

    async def delete_checkpoint(self, checkpoint_id: str) -> HandlerResult:
        """
        DELETE /api/checkpoints/{id}

        Delete a checkpoint.
        """
        manager = self._get_checkpoint_manager()

        # Check if checkpoint exists
        checkpoint = await manager.store.load(checkpoint_id)
        if not checkpoint:
            return error_response(f"Checkpoint not found: {checkpoint_id}", 404)

        # Delete
        success = await manager.store.delete(checkpoint_id)

        if success:
            return json_response(
                {"message": f"Checkpoint deleted: {checkpoint_id}"},
                status=200,
            )
        else:
            return error_response("Failed to delete checkpoint", 500)

    async def add_intervention(self, checkpoint_id: str, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/checkpoints/{id}/intervention

        Add a human intervention note to a checkpoint.

        Request body:
        {
            "note": "Human review: The debate is stuck on...",
            "by": "reviewer_name"
        }
        """
        data = safe_json_parse(body) or {}
        note = data.get("note")
        by = data.get("by", "human")

        if not note:
            return error_response("Missing required field: note", 400)

        manager = self._get_checkpoint_manager()
        success = await manager.add_intervention(
            checkpoint_id=checkpoint_id,
            note=note,
            by=by,
        )

        if success:
            return json_response(
                {
                    "message": "Intervention note added",
                    "checkpoint_id": checkpoint_id,
                }
            )
        else:
            return error_response(f"Checkpoint not found: {checkpoint_id}", 404)

    async def list_debate_checkpoints(
        self, debate_id: str, query_params: Dict[str, str]
    ) -> HandlerResult:
        """
        GET /api/debates/{id}/checkpoints

        List all checkpoints for a specific debate.
        """
        manager = self._get_checkpoint_manager()

        checkpoints = await manager.store.list_checkpoints(debate_id=debate_id)

        # Sort by creation time (most recent first)
        checkpoints.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return json_response(
            {
                "debate_id": debate_id,
                "checkpoints": checkpoints,
                "total": len(checkpoints),
            }
        )

    async def create_checkpoint(self, debate_id: str, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/debates/{id}/checkpoint

        Manually trigger checkpoint creation for a running debate.

        Note: This is primarily for administrative use. Checkpoints are
        automatically created during debate execution.
        """
        # This would need to connect to the running debate's Arena instance
        # For now, return a message indicating how to use this endpoint
        return json_response(
            {
                "message": "Manual checkpoint creation requires an active debate session",
                "hint": "Checkpoints are automatically created during debate execution. "
                "Use /api/checkpoints/resumable to find debates with existing checkpoints.",
            },
            status=501,
        )

    async def pause_debate(self, debate_id: str, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/debates/{id}/checkpoint/pause

        Pause a running debate and create a checkpoint.

        Note: This requires integration with the debate lifecycle manager.
        """
        # This would need to signal the running debate to pause
        # For now, return implementation status
        return json_response(
            {
                "message": "Debate pause functionality requires lifecycle manager integration",
                "hint": "To pause a debate, cancel the running debate and it will "
                "automatically checkpoint its current state if checkpointing is enabled.",
            },
            status=501,
        )
