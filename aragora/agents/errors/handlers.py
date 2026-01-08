"""
Exception handler utilities for agent operations.

Provides reusable error handling patterns for async agent operations,
implementing the "Autonomic layer" that keeps debates alive by gracefully
handling agent failures.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from aragora.agents.types import T
from .classifier import ErrorClassifier

logger = logging.getLogger(__name__)


async def handle_agent_operation(
    operation: Callable[[], Any],
    agent_name: str,
    operation_name: str = "operation",
    fallback_value: T = None,
    fallback_message: Optional[str] = None,
) -> T:
    """Execute an async agent operation with autonomic error handling.

    Implements the "Autonomic layer" pattern - catches all exceptions to keep
    debates alive. Logs appropriately based on error type.

    Args:
        operation: Async callable to execute (e.g., lambda: agent.generate(prompt))
        agent_name: Name of the agent for logging
        operation_name: Name of the operation for logging (e.g., "generate", "critique")
        fallback_value: Value to return on error (default: None)
        fallback_message: If provided, return this string message on error
                         (overrides fallback_value for string operations)

    Returns:
        Result of operation, or fallback_value/fallback_message on error

    Example:
        # For generate (returns string message on error)
        result = await handle_agent_operation(
            lambda: agent.generate(prompt, context),
            agent.name,
            "generate",
            fallback_message=f"[System: Agent {agent.name} error - skipping]"
        )

        # For critique/vote (returns None on error)
        critique = await handle_agent_operation(
            lambda: agent.critique(proposal, task, context),
            agent.name,
            "critique",
            fallback_value=None
        )
    """
    try:
        return await operation()

    except asyncio.TimeoutError:
        logger.warning(f"[Autonomic] Agent {agent_name} {operation_name} timed out")
        return fallback_message if fallback_message else fallback_value  # type: ignore[return-value]

    except (ConnectionError, OSError) as e:
        logger.warning(f"[Autonomic] Agent {agent_name} {operation_name} connection error: {e}")
        return fallback_message if fallback_message else fallback_value  # type: ignore[return-value]

    except Exception as e:
        # Use ErrorClassifier for more detailed categorization
        _, category = ErrorClassifier.classify_error(e)
        logger.exception(
            f"[Autonomic] Agent {agent_name} {operation_name} failed ({category}): "
            f"{type(e).__name__}: {e}"
        )
        return fallback_message if fallback_message else fallback_value  # type: ignore[return-value]


class AgentErrorHandler:
    """Context manager for agent error handling with automatic fallback.

    Provides a cleaner interface for wrapping agent operations.

    Example:
        async with AgentErrorHandler(agent.name, "generate") as handler:
            result = await agent.generate(prompt, context)
            handler.set_result(result)

        # If error occurred, handler.result is the fallback value
        output = handler.result or "[System: Error occurred]"
    """

    def __init__(
        self,
        agent_name: str,
        operation_name: str = "operation",
        fallback_value: Any = None,
    ):
        self.agent_name = agent_name
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.result = fallback_value
        self.error: Optional[Exception] = None

    def set_result(self, value: Any) -> None:
        """Set the successful result."""
        self.result = value

    async def __aenter__(self) -> "AgentErrorHandler":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            return False

        self.error = exc_val

        if exc_type is asyncio.TimeoutError:
            logger.warning(
                f"[Autonomic] Agent {self.agent_name} {self.operation_name} timed out"
            )
            return True  # Suppress exception

        if issubclass(exc_type, (ConnectionError, OSError)):
            logger.warning(
                f"[Autonomic] Agent {self.agent_name} {self.operation_name} "
                f"connection error: {exc_val}"
            )
            return True

        # General exception
        _, category = ErrorClassifier.classify_error(exc_val)
        logger.exception(
            f"[Autonomic] Agent {self.agent_name} {self.operation_name} "
            f"failed ({category}): {type(exc_val).__name__}: {exc_val}"
        )
        return True  # Suppress exception and use fallback


def make_fallback_message(agent_name: str, operation: str = "turn") -> str:
    """Generate a standardized system fallback message.

    Args:
        agent_name: Name of the failing agent
        operation: What was being attempted

    Returns:
        Formatted system message for inclusion in debate context
    """
    return f"[System: Agent {agent_name} encountered an error - skipping this {operation}]"


__all__ = [
    "handle_agent_operation",
    "AgentErrorHandler",
    "make_fallback_message",
]
