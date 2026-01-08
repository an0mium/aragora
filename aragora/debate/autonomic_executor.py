"""
Autonomic executor for safe agent operations.

Provides error handling and timeout management for agent generation,
critique, and voting operations. Implements the "autonomic layer" pattern
that catches all exceptions to keep debates running even when individual
agents fail.

Extracted from Arena orchestrator to improve testability and separation
of concerns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Optional, TypeVar

from aragora.resilience import CircuitBreaker

T = TypeVar("T")
from aragora.debate.sanitization import OutputSanitizer

if TYPE_CHECKING:
    from aragora.core import Agent, Critique, Message, Vote

logger = logging.getLogger(__name__)


class AutonomicExecutor:
    """
    Executes agent operations with automatic error handling.

    The autonomic layer ensures that individual agent failures don't
    crash the entire debate. Errors are caught, logged, and converted
    to graceful fallback responses.

    Usage:
        executor = AutonomicExecutor(circuit_breaker)
        response = await executor.generate(agent, prompt, context)
        critique = await executor.critique(agent, proposal, task, context)
        vote = await executor.vote(agent, proposals, task)
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        default_timeout: float = 90.0,
    ):
        """
        Initialize the autonomic executor.

        Args:
            circuit_breaker: Optional circuit breaker for failure tracking
            default_timeout: Default timeout for agent operations in seconds
        """
        self.circuit_breaker = circuit_breaker
        self.default_timeout = default_timeout

    async def with_timeout(
        self,
        coro: Awaitable[T],
        agent_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> T:
        """
        Wrap coroutine with per-agent timeout.

        If the agent times out, records a circuit breaker failure and
        raises TimeoutError. This prevents a single stalled agent from
        blocking the entire debate.

        Args:
            coro: Coroutine to execute
            agent_name: Agent name for logging and circuit breaker
            timeout_seconds: Timeout in seconds (uses default if None)

        Returns:
            Result of the coroutine

        Raises:
            TimeoutError: If the operation times out
        """
        timeout = timeout_seconds or self.default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(agent_name)
            logger.warning(f"Agent {agent_name} timed out after {timeout}s")
            raise TimeoutError(f"Agent {agent_name} timed out after {timeout}s")

    async def generate(
        self,
        agent: "Agent",
        prompt: str,
        context: list["Message"],
    ) -> str:
        """
        Generate response with an agent, handling errors and sanitizing output.

        Implements the "autonomic layer" - catches all exceptions to keep
        the debate alive even when individual agents fail.

        Args:
            agent: Agent to generate response
            prompt: Prompt for the agent
            context: Conversation context

        Returns:
            Generated response (or system message on failure)
        """
        try:
            raw_output = await agent.generate(prompt, context)
            return OutputSanitizer.sanitize_agent_output(raw_output, agent.name)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} timed out")
            return f"[System: Agent {agent.name} timed out - skipping this turn]"
        except (ConnectionError, OSError) as e:
            # Network/OS errors - log without full traceback
            logger.warning(f"[Autonomic] Agent {agent.name} connection error: {e}")
            return f"[System: Agent {agent.name} connection failed - skipping this turn]"
        except Exception as e:
            # Autonomic containment: convert crashes to valid responses
            logger.exception(f"[Autonomic] Agent {agent.name} failed: {type(e).__name__}: {e}")
            return f"[System: Agent {agent.name} encountered an error - skipping this turn]"

    async def critique(
        self,
        agent: "Agent",
        proposal: str,
        task: str,
        context: list["Message"],
    ) -> Optional["Critique"]:
        """
        Get critique from an agent with autonomic error handling.

        Args:
            agent: Agent to provide critique
            proposal: Proposal to critique
            task: Task description
            context: Conversation context

        Returns:
            Critique object or None on failure
        """
        try:
            return await agent.critique(proposal, task, context)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} critique timed out")
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} critique connection error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} critique failed: {e}")
            return None

    async def vote(
        self,
        agent: "Agent",
        proposals: dict[str, str],
        task: str,
    ) -> Optional["Vote"]:
        """
        Get vote from an agent with autonomic error handling.

        Args:
            agent: Agent to vote
            proposals: Dict of agent_name -> proposal text
            task: Task description

        Returns:
            Vote object or None on failure
        """
        try:
            return await agent.vote(proposals, task)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} vote timed out")
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} vote connection error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} vote failed: {e}")
            return None


__all__ = ["AutonomicExecutor"]
