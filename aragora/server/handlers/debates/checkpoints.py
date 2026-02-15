"""
Debate Checkpoint Handler.

Provides endpoints for checkpoint-based pause/resume:
- POST /api/v1/debates/:id/checkpoint/pause  - Create a checkpoint and pause
- POST /api/v1/debates/:id/checkpoint/resume - Resume from a checkpoint
- GET  /api/v1/debates/:id/checkpoints       - List checkpoints for a debate
"""

import json
import logging
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.utils.auth import UnauthorizedError, get_auth_context

logger = logging.getLogger(__name__)

# In-memory checkpoint tracking for paused debates
# In production, CheckpointManager persists to database/file/S3
_paused_debates: dict[str, dict[str, Any]] = {}


def _get_checkpoint_manager(server_context: dict[str, Any] | None = None) -> Any:
    """Get CheckpointManager from server context or create a default."""
    if server_context and "checkpoint_manager" in server_context:
        return server_context["checkpoint_manager"]
    try:
        from aragora.debate.checkpoint import CheckpointManager, DatabaseCheckpointStore

        return CheckpointManager(store=DatabaseCheckpointStore())
    except (ImportError, TypeError, ValueError, OSError):
        return None


@require_permission("debates:write")
async def handle_checkpoint_pause(
    debate_id: str,
    context: AuthorizationContext,
    checkpoint_manager: Any = None,
) -> HandlerResult:
    """Pause a debate by creating a durable checkpoint.

    Creates a checkpoint of the current debate state, allowing it to be
    resumed later from exactly this point.

    Args:
        debate_id: ID of the debate to pause.
        context: Authorization context.
        checkpoint_manager: Optional CheckpointManager instance.
    """
    if not checkpoint_manager:
        return error_response("Checkpoint manager not configured", 503)

    try:
        # Get the latest state for the debate and create a checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task=f"Paused debate {debate_id}",
            current_round=0,
            total_rounds=0,
            phase="paused",
            messages=[],
            critiques=[],
            votes=[],
            agents=[],
            current_consensus=None,
        )

        _paused_debates[debate_id] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "paused_at": datetime.now().isoformat(),
            "paused_by": context.user_id,
        }

        logger.info(
            "Debate %s paused with checkpoint %s",
            debate_id,
            checkpoint.checkpoint_id,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "paused_at": _paused_debates[debate_id]["paused_at"],
                "message": "Debate paused and checkpoint created",
            }
        )

    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.warning("Failed to create pause checkpoint for debate %s: %s", debate_id, e)
        return error_response("Checkpoint creation failed", 500)


@require_permission("debates:write")
async def handle_checkpoint_resume(
    debate_id: str,
    context: AuthorizationContext,
    checkpoint_id: str | None = None,
    checkpoint_manager: Any = None,
) -> HandlerResult:
    """Resume a debate from a checkpoint.

    Loads the checkpoint and prepares the debate for continuation.

    Args:
        debate_id: ID of the debate to resume.
        context: Authorization context.
        checkpoint_id: Optional specific checkpoint ID. If not provided,
            uses the latest checkpoint for the debate.
        checkpoint_manager: Optional CheckpointManager instance.
    """
    if not checkpoint_manager:
        return error_response("Checkpoint manager not configured", 503)

    try:
        # If no specific checkpoint, use the latest
        if not checkpoint_id:
            latest = await checkpoint_manager.get_latest(debate_id)
            if not latest:
                return error_response(f"No checkpoints found for debate {debate_id}", 404)
            checkpoint_id = latest.checkpoint_id

        resumed = await checkpoint_manager.resume_from_checkpoint(
            checkpoint_id=checkpoint_id,
            resumed_by=context.user_id or "api",
        )

        if not resumed:
            return error_response(f"Checkpoint {checkpoint_id} not found or corrupted", 404)

        # Clear paused state
        _paused_debates.pop(debate_id, None)

        logger.info(
            "Debate %s resumed from checkpoint %s at round %d",
            debate_id,
            checkpoint_id,
            resumed.checkpoint.current_round,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "checkpoint_id": checkpoint_id,
                "resumed_at": resumed.resumed_at,
                "resumed_from_round": resumed.checkpoint.current_round,
                "total_rounds": resumed.checkpoint.total_rounds,
                "message_count": len(resumed.messages),
                "message": "Debate resumed from checkpoint",
            }
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning("Failed to resume debate %s: %s", debate_id, e)
        return error_response("Checkpoint resume failed", 500)


@require_permission("debates:read")
async def handle_list_checkpoints(
    debate_id: str,
    context: AuthorizationContext,
    limit: int = 50,
    checkpoint_manager: Any = None,
) -> HandlerResult:
    """List all checkpoints for a debate.

    Args:
        debate_id: ID of the debate.
        context: Authorization context.
        limit: Maximum number of checkpoints to return.
        checkpoint_manager: Optional CheckpointManager instance.
    """
    if not checkpoint_manager:
        return error_response("Checkpoint manager not configured", 503)

    try:
        checkpoints = await checkpoint_manager.store.list_checkpoints(
            debate_id=debate_id,
            limit=min(limit, 100),
        )

        is_paused = debate_id in _paused_debates
        paused_info = _paused_debates.get(debate_id)

        return json_response(
            {
                "debate_id": debate_id,
                "is_paused": is_paused,
                "paused_at": paused_info["paused_at"] if paused_info else None,
                "total_checkpoints": len(checkpoints),
                "checkpoints": checkpoints,
            }
        )

    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to list checkpoints for debate %s: %s", debate_id, e)
        return error_response("Failed to list checkpoints", 500)


def register_checkpoint_routes(router: Any) -> None:
    """Register checkpoint routes with the server router."""

    async def _require_context(
        handler: Any,
    ) -> tuple[AuthorizationContext | None, HandlerResult | None]:
        try:
            return await get_auth_context(handler, require_auth=True), None
        except UnauthorizedError as e:
            return None, error_response(e.message, 401)

    def _get_manager(request: Any) -> Any:
        """Extract checkpoint manager from request app state."""
        app_state = getattr(request, "app", None)
        if app_state:
            state = getattr(app_state, "state", {})
            if isinstance(state, dict):
                return state.get("checkpoint_manager")
        return _get_checkpoint_manager()

    async def pause_debate(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        context, err = await _require_context(request)
        if err:
            return err
        manager = _get_manager(request)
        return await handle_checkpoint_pause(debate_id, context, checkpoint_manager=manager)

    async def resume_debate(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        body = await request.body()
        data = json.loads(body) if body else {}
        context, err = await _require_context(request)
        if err:
            return err
        manager = _get_manager(request)
        return await handle_checkpoint_resume(
            debate_id,
            context,
            checkpoint_id=data.get("checkpoint_id"),
            checkpoint_manager=manager,
        )

    async def list_checkpoints(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        limit = int(request.query_params.get("limit", 50))
        context, err = await _require_context(request)
        if err:
            return err
        manager = _get_manager(request)
        return await handle_list_checkpoints(
            debate_id, context, limit=limit, checkpoint_manager=manager
        )

    # Register routes
    router.add_route(
        "POST",
        "/api/v1/debates/{debate_id}/checkpoint/pause",
        pause_debate,
    )
    router.add_route(
        "POST",
        "/api/v1/debates/{debate_id}/checkpoint/resume",
        resume_debate,
    )
    router.add_route(
        "GET",
        "/api/v1/debates/{debate_id}/checkpoints",
        list_checkpoints,
    )


__all__ = [
    "handle_checkpoint_pause",
    "handle_checkpoint_resume",
    "handle_list_checkpoints",
    "register_checkpoint_routes",
]
