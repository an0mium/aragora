"""
Intervention operations handler mixin.

Provides mid-debate control endpoints:
- Pause/resume debate execution
- Inject user arguments
- Adjust agent weights
- Modify consensus threshold
- Get intervention state and logs

All interventions are logged to the audit trail for compliance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission

from ..base import (
    HandlerResult,
    json_response,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# In-memory state for active debates
# In production, this would be in Redis/database
_debate_state: dict[str, dict[str, Any]] = {}
_intervention_log: list[dict[str, Any]] = []


def _get_debate_state(debate_id: str) -> dict[str, Any]:
    """Get or create debate state."""
    if debate_id not in _debate_state:
        _debate_state[debate_id] = {
            "is_paused": False,
            "agent_weights": {},
            "consensus_threshold": 0.75,
            "injected_arguments": [],
            "follow_up_questions": [],
        }
    return _debate_state[debate_id]


def _log_intervention(
    debate_id: str,
    intervention_type: str,
    data: dict[str, Any],
    user_id: str | None = None,
) -> None:
    """Log intervention to audit trail."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "debate_id": debate_id,
        "type": intervention_type,
        "data": data,
        "user_id": user_id,
    }
    _intervention_log.append(entry)
    logger.info("Intervention logged: %s for debate %s", intervention_type, debate_id)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by InterventionMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: dict[str, Any]

    def get_current_user(self, handler: Any) -> Any | None:
        """Get current authenticated user from handler."""
        ...


class InterventionMixin:
    """Mixin providing intervention operations for DebatesHandler."""

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{debate_id}/intervention/pause",
        summary="Pause a debate",
        description="Pause an active debate. Stops agent responses but preserves state.",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {
                "description": "Debate paused successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "debate_id": {"type": "string"},
                                "is_paused": {"type": "boolean"},
                                "paused_at": {"type": "string", "format": "date-time"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:write")
    def _pause_debate(
        self: _DebatesHandlerProtocol, debate_id: str, user_id: str | None = None
    ) -> HandlerResult:
        """Pause an active debate.

        Pausing a debate stops agent responses but preserves state.
        The debate can be resumed at any point.
        """
        state = _get_debate_state(debate_id)

        if state["is_paused"]:
            return json_response(
                {
                    "success": False,
                    "error": "Debate is already paused",
                    "debate_id": debate_id,
                }
            )

        state["is_paused"] = True
        state["paused_at"] = datetime.now().isoformat()

        _log_intervention(debate_id, "pause", {"paused_at": state["paused_at"]}, user_id)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "is_paused": True,
                "paused_at": state["paused_at"],
                "message": "Debate paused successfully",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{debate_id}/intervention/resume",
        summary="Resume a debate",
        description="Resume a paused debate. Continues agent responses from where they left off.",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {
                "description": "Debate resumed successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "debate_id": {"type": "string"},
                                "is_paused": {"type": "boolean"},
                                "resumed_at": {"type": "string", "format": "date-time"},
                                "pause_duration_seconds": {"type": "number"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:write")
    def _resume_debate(
        self: _DebatesHandlerProtocol, debate_id: str, user_id: str | None = None
    ) -> HandlerResult:
        """Resume a paused debate.

        Resumes agent responses from where they left off.
        """
        state = _get_debate_state(debate_id)

        if not state["is_paused"]:
            return json_response(
                {
                    "success": False,
                    "error": "Debate is not paused",
                    "debate_id": debate_id,
                }
            )

        paused_at = state.get("paused_at")
        state["is_paused"] = False
        state["paused_at"] = None
        state["resumed_at"] = datetime.now().isoformat()

        # Calculate pause duration
        pause_duration = None
        if paused_at:
            pause_start = datetime.fromisoformat(paused_at)
            pause_duration = (datetime.now() - pause_start).total_seconds()

        _log_intervention(
            debate_id,
            "resume",
            {"resumed_at": state["resumed_at"], "pause_duration_seconds": pause_duration},
            user_id,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "is_paused": False,
                "resumed_at": state["resumed_at"],
                "pause_duration_seconds": pause_duration,
                "message": "Debate resumed successfully",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{debate_id}/intervention/inject",
        summary="Inject argument into debate",
        description="Inject a user argument or follow-up question into the debate. Will be included in the next round's context.",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The argument or question to inject"},
                            "type": {"type": "string", "enum": ["argument", "follow_up"], "default": "argument"},
                            "source": {"type": "string", "default": "user"},
                        },
                        "required": ["content"],
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Argument injected successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "debate_id": {"type": "string"},
                                "injection_id": {"type": "string"},
                                "type": {"type": "string"},
                                "message": {"type": "string"},
                                "will_appear_in": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "400": {"description": "Invalid request"},
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:write")
    def _inject_argument(
        self: _DebatesHandlerProtocol,
        debate_id: str,
        content: str,
        injection_type: str = "argument",
        source: str = "user",
        user_id: str | None = None,
    ) -> HandlerResult:
        """Inject a user argument into the debate.

        The argument will be included in the next round's context
        and considered by all agents.
        """
        if not content or not content.strip():
            return json_response(
                {
                    "success": False,
                    "error": "Content cannot be empty",
                },
                status=400,
            )

        state = _get_debate_state(debate_id)

        injection = {
            "id": f"inj_{len(state['injected_arguments']) + 1}",
            "content": content.strip(),
            "type": injection_type,
            "source": source,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "processed": False,
        }

        if injection_type == "follow_up":
            state["follow_up_questions"].append(injection)
        else:
            state["injected_arguments"].append(injection)

        _log_intervention(
            debate_id,
            f"inject_{injection_type}",
            {
                "content_preview": content[:100],
                "full_length": len(content),
                "source": source,
            },
            user_id,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "injection_id": injection["id"],
                "type": injection_type,
                "message": f"{injection_type.title()} injected successfully",
                "will_appear_in": "next_round",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{debate_id}/intervention/weights",
        summary="Update agent weight",
        description="Update an agent's influence weight. Weight affects how much the agent's vote counts in consensus (0.0=muted, 1.0=normal, 2.0=double).",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "agent": {"type": "string", "description": "Agent name/ID"},
                            "weight": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                        },
                        "required": ["agent", "weight"],
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Weight updated successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "debate_id": {"type": "string"},
                                "agent": {"type": "string"},
                                "old_weight": {"type": "number"},
                                "new_weight": {"type": "number"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "400": {"description": "Invalid weight value"},
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:write")
    def _update_agent_weights(
        self: _DebatesHandlerProtocol,
        debate_id: str,
        agent: str,
        weight: float,
        user_id: str | None = None,
    ) -> HandlerResult:
        """Update an agent's influence weight.

        Weight affects how much the agent's vote counts in consensus:
        - 0.0 = muted (agent's vote doesn't count)
        - 1.0 = normal influence
        - 2.0 = double influence
        """
        if not (0.0 <= weight <= 2.0):
            return json_response(
                {
                    "success": False,
                    "error": "Weight must be between 0.0 and 2.0",
                },
                status=400,
            )

        state = _get_debate_state(debate_id)
        old_weight = state["agent_weights"].get(agent, 1.0)
        state["agent_weights"][agent] = weight

        _log_intervention(
            debate_id,
            "weight_change",
            {
                "agent": agent,
                "old_weight": old_weight,
                "new_weight": weight,
            },
            user_id,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "agent": agent,
                "old_weight": old_weight,
                "new_weight": weight,
                "message": f"Agent {agent} weight updated to {weight}",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{debate_id}/intervention/threshold",
        summary="Update consensus threshold",
        description="Update the consensus threshold. Threshold is the minimum agreement level required (0.5=majority, 0.75=strong majority, 1.0=unanimous).",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                        },
                        "required": ["threshold"],
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Threshold updated successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "debate_id": {"type": "string"},
                                "old_threshold": {"type": "number"},
                                "new_threshold": {"type": "number"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "400": {"description": "Invalid threshold value"},
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:write")
    def _update_consensus_threshold(
        self: _DebatesHandlerProtocol,
        debate_id: str,
        threshold: float,
        user_id: str | None = None,
    ) -> HandlerResult:
        """Update the consensus threshold.

        Threshold is the minimum agreement level required for consensus:
        - 0.5 = simple majority
        - 0.75 = strong majority (default)
        - 1.0 = unanimous
        """
        if not (0.5 <= threshold <= 1.0):
            return json_response(
                {
                    "success": False,
                    "error": "Threshold must be between 0.5 and 1.0",
                },
                status=400,
            )

        state = _get_debate_state(debate_id)
        old_threshold = state["consensus_threshold"]
        state["consensus_threshold"] = threshold

        _log_intervention(
            debate_id,
            "threshold_change",
            {
                "old_threshold": old_threshold,
                "new_threshold": threshold,
            },
            user_id,
        )

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "old_threshold": old_threshold,
                "new_threshold": threshold,
                "message": f"Consensus threshold updated to {threshold:.0%}",
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{debate_id}/intervention/state",
        summary="Get intervention state",
        description="Get the current intervention state for a debate including pause status, weights, threshold, and pending injections.",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {
                "description": "Intervention state returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "is_paused": {"type": "boolean"},
                                "paused_at": {"type": "string", "format": "date-time", "nullable": True},
                                "consensus_threshold": {"type": "number"},
                                "agent_weights": {"type": "object", "additionalProperties": {"type": "number"}},
                                "pending_injections": {"type": "integer"},
                                "pending_follow_ups": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:read")
    def _get_intervention_state(
        self: _DebatesHandlerProtocol, debate_id: str
    ) -> HandlerResult:
        """Get the current intervention state for a debate.

        Returns pause status, weights, threshold, and pending injections.
        """
        state = _get_debate_state(debate_id)

        return json_response(
            {
                "debate_id": debate_id,
                "is_paused": state["is_paused"],
                "paused_at": state.get("paused_at"),
                "consensus_threshold": state["consensus_threshold"],
                "agent_weights": state["agent_weights"],
                "pending_injections": len(
                    [i for i in state["injected_arguments"] if not i["processed"]]
                ),
                "pending_follow_ups": len(
                    [q for q in state["follow_up_questions"] if not q["processed"]]
                ),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{debate_id}/intervention/log",
        summary="Get intervention log",
        description="Get the intervention log for a debate. Returns all interventions with timestamps for audit purposes.",
        tags=["Debates", "Intervention"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}, "required": False},
        ],
        responses={
            "200": {
                "description": "Intervention log returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "total_interventions": {"type": "integer"},
                                "interventions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "debate_id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "data": {"type": "object"},
                                            "user_id": {"type": "string", "nullable": True},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
        },
    )
    @require_permission("debates:read")
    def _get_intervention_log(
        self: _DebatesHandlerProtocol, debate_id: str, limit: int = 50
    ) -> HandlerResult:
        """Get the intervention log for a debate.

        Returns all interventions with timestamps for audit purposes.
        """
        debate_logs = [log for log in _intervention_log if log["debate_id"] == debate_id]

        # Sort by timestamp descending
        debate_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return json_response(
            {
                "debate_id": debate_id,
                "total_interventions": len(debate_logs),
                "interventions": debate_logs[:limit],
            }
        )


__all__ = ["InterventionMixin"]
