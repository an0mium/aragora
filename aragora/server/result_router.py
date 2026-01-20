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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def route_result(debate_id: str, result: Dict[str, Any]) -> bool:
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
        logger.warning("debate_origin module not available")
        return False
    except Exception as e:
        logger.error(f"Failed to route result: {e}")
        return False


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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_route_async(debate_id, result_dict))
        else:
            asyncio.run(_route_async(debate_id, result_dict))
    except RuntimeError:
        # No event loop available, try to run directly
        asyncio.run(_route_async(debate_id, result_dict))


async def _route_async(debate_id: str, result: Dict[str, Any]) -> None:
    """Async wrapper for result routing."""
    try:
        success = await route_result(debate_id, result)
        if success:
            logger.info(f"Result routed for debate {debate_id}")
        else:
            logger.debug(f"No origin found or routing failed for debate {debate_id}")
    except Exception as e:
        logger.error(f"Async result routing failed: {e}")


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
        logger.warning(f"Could not register result router hooks: {e}")
    except Exception as e:
        logger.error(f"Failed to register result router hooks: {e}")


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
        logger.warning(f"Could not set up result routing: {e}")
    except Exception as e:
        logger.error(f"Failed to set up result routing: {e}")


__all__ = [
    "route_result",
    "register_result_router_hooks",
    "setup_result_routing",
]
