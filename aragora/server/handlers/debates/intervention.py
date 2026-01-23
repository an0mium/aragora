"""
Debate Intervention Handler.

Provides endpoints for mid-debate control:
- Pause/resume debate execution
- Inject user arguments
- Adjust agent weights
- Modify consensus threshold
- Add follow-up questions

All interventions are logged to the audit trail for compliance.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from aragora.server.handlers.base import HandlerResult, json_response

logger = logging.getLogger(__name__)

# In-memory state for active debates
# In production, this would be in Redis/database
_debate_state: dict[str, dict[str, Any]] = {}
_intervention_log: list[dict[str, Any]] = []


def get_debate_state(debate_id: str) -> dict[str, Any]:
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


def log_intervention(
    debate_id: str,
    intervention_type: str,
    data: dict[str, Any],
    user_id: Optional[str] = None,
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
    logger.info(f"Intervention logged: {intervention_type} for debate {debate_id}")


async def handle_pause_debate(debate_id: str) -> HandlerResult:
    """Pause an active debate.

    Pausing a debate stops agent responses but preserves state.
    The debate can be resumed at any point.
    """
    state = get_debate_state(debate_id)

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

    log_intervention(debate_id, "pause", {"paused_at": state["paused_at"]})

    return json_response(
        {
            "success": True,
            "debate_id": debate_id,
            "is_paused": True,
            "paused_at": state["paused_at"],
            "message": "Debate paused successfully",
        }
    )


async def handle_resume_debate(debate_id: str) -> HandlerResult:
    """Resume a paused debate.

    Resumes agent responses from where they left off.
    """
    state = get_debate_state(debate_id)

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

    log_intervention(
        debate_id,
        "resume",
        {"resumed_at": state["resumed_at"], "pause_duration_seconds": pause_duration},
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


async def handle_inject_argument(
    debate_id: str,
    content: str,
    injection_type: str = "argument",
    source: str = "user",
    user_id: Optional[str] = None,
) -> HandlerResult:
    """Inject a user argument into the debate.

    The argument will be included in the next round's context
    and considered by all agents.

    Args:
        debate_id: Active debate ID
        content: The argument or question to inject
        injection_type: "argument" or "follow_up"
        source: Source identifier (usually "user")
        user_id: Optional user ID for attribution
    """
    if not content or not content.strip():
        return json_response(
            {
                "success": False,
                "error": "Content cannot be empty",
            },
            status=400,
        )

    state = get_debate_state(debate_id)

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

    log_intervention(
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


async def handle_update_weights(
    debate_id: str,
    agent: str,
    weight: float,
    user_id: Optional[str] = None,
) -> HandlerResult:
    """Update an agent's influence weight.

    Weight affects how much the agent's vote counts in consensus:
    - 0.0 = muted (agent's vote doesn't count)
    - 1.0 = normal influence
    - 2.0 = double influence

    Args:
        debate_id: Active debate ID
        agent: Agent name/ID
        weight: New weight (0.0 to 2.0)
    """
    if not (0.0 <= weight <= 2.0):
        return json_response(
            {
                "success": False,
                "error": "Weight must be between 0.0 and 2.0",
            },
            status=400,
        )

    state = get_debate_state(debate_id)
    old_weight = state["agent_weights"].get(agent, 1.0)
    state["agent_weights"][agent] = weight

    log_intervention(
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


async def handle_update_threshold(
    debate_id: str,
    threshold: float,
    user_id: Optional[str] = None,
) -> HandlerResult:
    """Update the consensus threshold.

    Threshold is the minimum agreement level required for consensus:
    - 0.5 = simple majority
    - 0.75 = strong majority (default)
    - 1.0 = unanimous

    Args:
        debate_id: Active debate ID
        threshold: New threshold (0.5 to 1.0)
    """
    if not (0.5 <= threshold <= 1.0):
        return json_response(
            {
                "success": False,
                "error": "Threshold must be between 0.5 and 1.0",
            },
            status=400,
        )

    state = get_debate_state(debate_id)
    old_threshold = state["consensus_threshold"]
    state["consensus_threshold"] = threshold

    log_intervention(
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


async def handle_get_intervention_state(
    debate_id: str,
) -> HandlerResult:
    """Get the current intervention state for a debate.

    Returns pause status, weights, threshold, and pending injections.
    """
    state = get_debate_state(debate_id)

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


async def handle_get_intervention_log(
    debate_id: str,
    limit: int = 50,
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


def register_intervention_routes(router: Any) -> None:
    """Register intervention routes with the server router."""

    async def pause_debate(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        return await handle_pause_debate(debate_id)

    async def resume_debate(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        return await handle_resume_debate(debate_id)

    async def inject_argument(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        body = await request.body()
        data = json.loads(body) if body else {}
        return await handle_inject_argument(
            debate_id,
            content=data.get("content", ""),
            injection_type=data.get("type", "argument"),
            source=data.get("source", "user"),
            user_id=data.get("user_id"),
        )

    async def update_weights(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        body = await request.body()
        data = json.loads(body) if body else {}
        return await handle_update_weights(
            debate_id,
            agent=data.get("agent", ""),
            weight=float(data.get("weight", 1.0)),
            user_id=data.get("user_id"),
        )

    async def update_threshold(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        body = await request.body()
        data = json.loads(body) if body else {}
        return await handle_update_threshold(
            debate_id,
            threshold=float(data.get("threshold", 0.75)),
            user_id=data.get("user_id"),
        )

    async def get_state(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        return await handle_get_intervention_state(debate_id)

    async def get_log(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        limit = int(request.query_params.get("limit", 50))
        return await handle_get_intervention_log(debate_id, limit)

    # Register routes
    router.add_route("POST", "/api/debates/{debate_id}/intervention/pause", pause_debate)
    router.add_route("POST", "/api/debates/{debate_id}/intervention/resume", resume_debate)
    router.add_route("POST", "/api/debates/{debate_id}/intervention/inject", inject_argument)
    router.add_route("POST", "/api/debates/{debate_id}/intervention/weights", update_weights)
    router.add_route("POST", "/api/debates/{debate_id}/intervention/threshold", update_threshold)
    router.add_route("GET", "/api/debates/{debate_id}/intervention/state", get_state)
    router.add_route("GET", "/api/debates/{debate_id}/intervention/log", get_log)


__all__ = [
    "handle_pause_debate",
    "handle_resume_debate",
    "handle_inject_argument",
    "handle_update_weights",
    "handle_update_threshold",
    "handle_get_intervention_state",
    "handle_get_intervention_log",
    "get_debate_state",
    "register_intervention_routes",
]
