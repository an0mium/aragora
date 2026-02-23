"""
Debate Result Router.

Routes debate results back to originating chat channels when debates complete.
Hooks into the debate lifecycle to automatically notify users.

Usage:
    from aragora.server.result_router import register_result_router_hooks

    # Register with hook manager (typically done at startup)
    register_result_router_hooks(hook_manager)

    # Or manually route a result
    from aragora.server.result_router import route_result
    await route_result(debate_id, result)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def route_result(debate_id: str, result: dict[str, Any]) -> bool:
    """Route a debate result to its originating channel.

    Args:
        debate_id: Debate identifier
        result: Debate result dictionary

    Returns:
        True if routed successfully
    """
    try:
        from aragora.server.debate_origin import route_debate_result

        return await route_debate_result(debate_id, result)
    except ImportError:
        logger.debug("debate_origin module not available")
        return False
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to route result: %s", e)
        return False


async def route_plan_outcome(
    debate_id: str,
    plan_id: str,
    outcome: dict[str, Any],
) -> bool:
    """Route a plan outcome notification to the originating channel.

    Args:
        debate_id: Original debate identifier
        plan_id: Plan identifier
        outcome: Plan outcome dictionary

    Returns:
        True if routed successfully
    """
    try:
        from aragora.server.debate_origin import get_debate_origin
        from aragora.server.debate_origin.router import route_plan_result

        # Look up the origin by debate_id (plans originate from debates)
        origin = get_debate_origin(debate_id)
        if not origin:
            logger.debug("No origin found for debate %s, skipping plan delivery", debate_id)
            return False

        # Format the outcome as a user-friendly message
        message = _format_plan_outcome_message(plan_id, outcome)

        # Add plan_id and formatted message to outcome for routing
        outcome_with_message = {
            **outcome,
            "plan_id": plan_id,
            "formatted_message": message,
        }

        # Route to the originating channel
        return await route_plan_result(debate_id, outcome_with_message)

    except ImportError as e:
        logger.debug("debate_origin module not available for plan routing: %s", e)
        return False
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to route plan outcome: %s", e)
        return False


def _format_plan_outcome_message(plan_id: str, outcome: dict[str, Any]) -> str:
    """Format a plan outcome as a user-friendly message.

    Args:
        plan_id: Plan identifier
        outcome: Plan outcome dictionary

    Returns:
        Formatted message string
    """
    success = outcome.get("success", False)
    task = outcome.get("task", "Unknown task")[:200]
    tasks_completed = outcome.get("tasks_completed", 0)
    tasks_total = outcome.get("tasks_total", 0)
    verification_passed = outcome.get("verification_passed", 0)
    verification_total = outcome.get("verification_total", 0)
    error = outcome.get("error")

    if success:
        status = "Completed Successfully"
        icon = "check"
    elif tasks_completed > 0:
        status = "Partially Completed"
        icon = "warning"
    else:
        status = "Failed"
        icon = "error"

    lines = [
        f"**Decision Plan {status}** ({icon})",
        f"Plan: `{plan_id[:8]}...`",
        f"Task: {task}",
        "",
        f"- Tasks: {tasks_completed}/{tasks_total} completed",
    ]

    if verification_total > 0:
        lines.append(f"- Verification: {verification_passed}/{verification_total} passed")

    if error:
        lines.append(f"- Error: {error}")

    # Include receipt if present
    receipt_id = outcome.get("receipt_id")
    if receipt_id:
        lines.append(f"- Receipt: `{receipt_id[:8]}...`")

    # Include lessons if present
    lessons = outcome.get("lessons", [])
    if lessons:
        lines.append("")
        lines.append("**Lessons learned:**")
        for lesson in lessons[:3]:  # Limit to 3 lessons
            lines.append(f"- {lesson}")

    return "\n".join(lines)


def _on_post_debate(**kwargs: Any) -> None:
    """POST_DEBATE hook handler that routes results to originating channels."""
    result = kwargs.get("result")
    if not result:
        return

    # Extract debate_id from result
    debate_id = getattr(result, "debate_id", None)
    if not debate_id:
        debate_id = kwargs.get("debate_id")

    if not debate_id:
        logger.debug("No debate_id in POST_DEBATE hook, skipping result routing")
        return

    # Convert result to dict if needed
    if hasattr(result, "to_dict"):
        result_dict = result.to_dict()
    elif hasattr(result, "__dict__"):
        result_dict = {
            "debate_id": debate_id,
            "consensus_reached": getattr(result, "consensus_reached", False),
            "final_answer": getattr(result, "final_answer", ""),
            "confidence": getattr(result, "confidence", 0.0),
            "participants": getattr(result, "participants", []),
            "task": getattr(result, "task", ""),
        }
    else:
        result_dict = {"debate_id": debate_id}

    # Route in background
    try:
        asyncio.get_running_loop()
        asyncio.create_task(_route_async(debate_id, result_dict))
    except RuntimeError:
        # No event loop available, try to run directly
        asyncio.run(_route_async(debate_id, result_dict))


async def _route_async(debate_id: str, result: dict[str, Any]) -> None:
    """Async wrapper for result routing."""
    try:
        success = await route_result(debate_id, result)
        if success:
            logger.info("Result routed for debate %s", debate_id)
        else:
            logger.debug("No origin found or routing failed for debate %s", debate_id)
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Async result routing failed: %s", e)


def register_result_router_hooks(manager: Any) -> None:
    """Register result router hooks with a HookManager.

    Args:
        manager: HookManager instance
    """
    try:
        from aragora.debate.hooks import HookPriority, HookType

        # Register POST_DEBATE hook at CLEANUP priority (runs last)
        manager.register(
            HookType.POST_DEBATE,
            _on_post_debate,
            priority=HookPriority.CLEANUP,
            name="result_router",
        )

        logger.info("Result router hooks registered")

    except ImportError as e:
        logger.warning("Could not register result router hooks: %s", e)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Failed to register result router hooks: %s", e)


def setup_result_routing() -> None:
    """Set up result routing in the global hook manager.

    Call this at application startup to enable automatic result routing.
    """
    try:
        from aragora.debate.hooks import create_hook_manager

        manager = create_hook_manager()
        register_result_router_hooks(manager)
        logger.info("Result routing enabled globally")

    except ImportError as e:
        logger.warning("Could not set up result routing: %s", e)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Failed to set up result routing: %s", e)


__all__ = [
    "route_result",
    "route_plan_outcome",
    "register_result_router_hooks",
    "setup_result_routing",
]
