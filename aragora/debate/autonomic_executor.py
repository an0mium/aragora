"""
Autonomic executor for safe agent operations.

Provides error handling and timeout management for agent generation,
critique, and voting operations. Implements the "autonomic layer" pattern
that catches all exceptions to keep debates running even when individual
agents fail.

Features:
- Timeout escalation: retries use progressively longer timeouts
- Fallback agents: automatic substitution when primary agent fails
- Streaming buffer: capture partial content from timed-out streams
- Circuit breaker integration: track and avoid failing agents

Extracted from Arena orchestrator to improve testability and separation
of concerns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Awaitable, Optional, TypeVar

from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.resilience import CircuitBreaker

T = TypeVar("T")
from aragora.debate.sanitization import OutputSanitizer
from aragora.debate.schemas import validate_agent_response

# Lazy import for telemetry to avoid circular imports
_telemetry_initialized = False


def _ensure_telemetry_collectors() -> None:
    """Initialize default telemetry collectors (once)."""
    global _telemetry_initialized
    if _telemetry_initialized:
        return
    try:
        from aragora.agents.telemetry import setup_default_collectors

        setup_default_collectors()
        _telemetry_initialized = True
    except ImportError:
        pass


if TYPE_CHECKING:
    from aragora.agents.performance_monitor import AgentPerformanceMonitor
    from aragora.core import Agent, Critique, Message, Vote
    from aragora.debate.chaos_theater import ChaosDirector
    from aragora.debate.immune_system import TransparentImmuneSystem
    from aragora.insights.store import InsightStore

logger = logging.getLogger(__name__)


class StreamingContentBuffer:
    """
    Buffer for capturing partial streaming responses.

    When an agent times out during streaming, this buffer preserves
    any content received before the timeout, allowing partial responses
    to be recovered rather than lost entirely.

    Thread-safe via per-agent locks.
    """

    def __init__(self) -> None:
        self._buffer: dict[str, str] = defaultdict(str)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def append(self, agent_name: str, chunk: str) -> None:
        """Append chunk to agent's buffer."""
        async with self._locks[agent_name]:
            self._buffer[agent_name] += chunk

    def get_partial(self, agent_name: str) -> str:
        """Get accumulated partial content (non-async for error handlers)."""
        return self._buffer.get(agent_name, "")

    async def get_partial_async(self, agent_name: str) -> str:
        """Get accumulated partial content with lock."""
        async with self._locks[agent_name]:
            return self._buffer.get(agent_name, "")

    async def clear(self, agent_name: str) -> None:
        """Clear agent's buffer."""
        async with self._locks[agent_name]:
            self._buffer.pop(agent_name, None)

    def clear_sync(self, agent_name: str) -> None:
        """Clear agent's buffer (non-async)."""
        self._buffer.pop(agent_name, None)


class AutonomicExecutor:
    """
    Executes agent operations with automatic error handling.

    The autonomic layer ensures that individual agent failures don't
    crash the entire debate. Errors are caught, logged, and converted
    to graceful fallback responses.

    Features:
        - Timeout escalation: each retry gets 1.5x more time (configurable)
        - Fallback agents: automatic substitution when primary agent fails
        - Streaming buffer: capture partial content from timed-out streams
        - Circuit breaker: track and avoid persistently failing agents

    Usage:
        executor = AutonomicExecutor(circuit_breaker)
        response = await executor.generate(agent, prompt, context)
        critique = await executor.critique(agent, proposal, task, context)
        vote = await executor.vote(agent, proposals, task)

        # With fallback agents
        response = await executor.generate_with_fallback(
            agent, prompt, context, fallback_agents=[backup1, backup2]
        )
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        default_timeout: Optional[float] = None,  # Uses AGENT_TIMEOUT_SECONDS if not specified
        timeout_escalation_factor: float = 1.5,
        max_timeout: float = 600.0,  # Max timeout cap
        streaming_buffer: Optional[StreamingContentBuffer] = None,
        wisdom_store: Optional["InsightStore"] = None,
        loop_id: Optional[str] = None,
        immune_system: Optional["TransparentImmuneSystem"] = None,
        chaos_director: Optional["ChaosDirector"] = None,
        performance_monitor: Optional["AgentPerformanceMonitor"] = None,
        enable_telemetry: bool = False,
        event_hooks: Optional[dict] = None,  # Optional hooks for emitting events
    ):
        """
        Initialize the autonomic executor.

        Args:
            circuit_breaker: Optional circuit breaker for failure tracking
            default_timeout: Default timeout for agent operations in seconds
            timeout_escalation_factor: Multiplier for timeout on each retry (default 1.5x)
            max_timeout: Maximum timeout cap in seconds (default 300s / 5 min)
            streaming_buffer: Optional buffer for capturing partial streaming content
            wisdom_store: Optional InsightStore for audience wisdom fallback
            loop_id: Current loop/debate ID for wisdom retrieval
            immune_system: Optional TransparentImmuneSystem for health monitoring
            chaos_director: Optional ChaosDirector for theatrical failure messages
            performance_monitor: Optional AgentPerformanceMonitor for telemetry
            enable_telemetry: Enable Prometheus/Blackbox telemetry emission
            event_hooks: Optional dict of hooks for emitting agent events (on_agent_error, etc.)
        """
        self.circuit_breaker = circuit_breaker
        self.event_hooks = event_hooks or {}
        # Use AGENT_TIMEOUT_SECONDS from config if not explicitly specified
        self.default_timeout = (
            default_timeout if default_timeout is not None else float(AGENT_TIMEOUT_SECONDS)
        )
        self.immune_system = immune_system
        self.chaos_director = chaos_director
        self.timeout_escalation_factor = timeout_escalation_factor
        self.max_timeout = max_timeout
        self.streaming_buffer = streaming_buffer or StreamingContentBuffer()
        self.wisdom_store = wisdom_store
        self.loop_id = loop_id
        self.performance_monitor = performance_monitor
        self.enable_telemetry = enable_telemetry
        # Track retry counts per agent for timeout escalation
        self._retry_counts: dict[str, int] = defaultdict(int)

        # Initialize telemetry collectors if enabled
        if enable_telemetry:
            _ensure_telemetry_collectors()
            logger.debug("[telemetry] Prometheus/Blackbox collectors initialized")

    def set_loop_id(self, loop_id: str) -> None:
        """Set the current loop/debate ID for wisdom retrieval."""
        self.loop_id = loop_id

    def _emit_agent_error(
        self,
        agent_name: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        phase: str = "",
    ) -> None:
        """Emit an agent error event via hooks if available.

        This notifies the frontend about agent failures so users understand
        why an agent produced placeholder/error output.
        """
        on_agent_error = self.event_hooks.get("on_agent_error")
        if on_agent_error:
            try:
                on_agent_error(
                    agent=agent_name,
                    error_type=error_type,
                    message=message,
                    recoverable=recoverable,
                    phase=phase,
                )
            except Exception as e:
                logger.debug(f"[Autonomic] Failed to emit agent error event: {e}")

    @staticmethod
    def _is_empty_critique(result: "Critique | None") -> bool:
        """Return True if a critique is empty or only contains placeholder content."""
        if result is None:
            return True
        issues = [i.strip() for i in result.issues if isinstance(i, str) and i.strip()]
        suggestions = [s.strip() for s in result.suggestions if isinstance(s, str) and s.strip()]
        if not issues and not suggestions:
            return True
        if len(issues) == 1 and issues[0].strip().lower() == "agent response was empty":
            return not suggestions
        return False

    def _emit_agent_telemetry(
        self,
        agent_name: str,
        operation: str,
        start_time: float,
        success: bool,
        error: Exception | None = None,
        output: str | None = None,
        input_text: str | None = None,
    ) -> None:
        """Emit telemetry for an agent operation if enabled."""
        if not self.enable_telemetry:
            return

        try:
            from aragora.agents.telemetry import AgentTelemetry, _emit_telemetry

            telemetry = AgentTelemetry(
                agent_name=agent_name,
                operation=operation,
                start_time=start_time,
            )

            # Set input/output tokens
            if input_text:
                telemetry.input_chars = len(input_text)
                telemetry.input_tokens = AgentTelemetry.estimate_tokens(input_text)
            if output:
                telemetry.output_chars = len(output)
                telemetry.output_tokens = AgentTelemetry.estimate_tokens(output)

            telemetry.complete(success=success, error=error)
            _emit_telemetry(telemetry)
        except ImportError:
            pass  # Telemetry not available
        except (TypeError, ValueError, OSError) as e:
            # Expected telemetry issues: serialization, I/O
            logger.debug(f"[telemetry] Emission failed: {e}")
        except Exception as e:
            # Unexpected errors - log at warning level
            logger.warning(f"[telemetry] Unexpected emission error: {type(e).__name__}: {e}")

    def _get_wisdom_fallback(self, failed_agent: str) -> Optional[str]:
        """
        Get audience wisdom as fallback when agent fails.

        Returns formatted wisdom response if available, None otherwise.
        """
        if not self.wisdom_store or not self.loop_id:
            return None

        try:
            wisdom_list = self.wisdom_store.get_relevant_wisdom(self.loop_id, limit=1)
            if not wisdom_list:
                return None

            wisdom = wisdom_list[0]
            self.wisdom_store.mark_wisdom_used(wisdom["id"])

            logger.info(f"[wisdom] Injecting audience wisdom for failed agent {failed_agent}")

            return (
                f"[Audience Wisdom - submitted by {wisdom['submitter_id']}]\n\n"
                f"{wisdom['text']}\n\n"
                f"[System: This response was provided by the audience after "
                f"{failed_agent} failed to respond]"
            )
        except (KeyError, OSError, IOError) as e:
            # Expected database/storage issues
            logger.warning(f"[wisdom] Failed to retrieve wisdom: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log with more detail
            logger.error(f"[wisdom] Unexpected error retrieving wisdom: {type(e).__name__}: {e}")
            return None

    def get_escalated_timeout(self, agent_name: str, base_timeout: Optional[float] = None) -> float:
        """
        Calculate escalated timeout based on retry count.

        Each retry increases the timeout by the escalation factor,
        up to max_timeout. This gives slow agents more time on retries
        while keeping initial attempts fast.

        Args:
            agent_name: Agent name for retry tracking
            base_timeout: Base timeout (uses default if None)

        Returns:
            Escalated timeout in seconds
        """
        base = base_timeout or self.default_timeout
        retry_count = self._retry_counts[agent_name]
        escalated = base * (self.timeout_escalation_factor**retry_count)
        return min(escalated, self.max_timeout)

    def record_retry(self, agent_name: str) -> int:
        """
        Record a retry attempt for an agent.

        Args:
            agent_name: Agent that is being retried

        Returns:
            New retry count
        """
        self._retry_counts[agent_name] += 1
        return self._retry_counts[agent_name]

    def reset_retries(self, agent_name: str) -> None:
        """Reset retry count for an agent after success."""
        self._retry_counts.pop(agent_name, None)

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
        phase: str = "",
        round_num: int = 0,
    ) -> str:
        """
        Generate response with an agent, handling errors and sanitizing output.

        Implements the "autonomic layer" - catches all exceptions to keep
        the debate alive even when individual agents fail.

        Args:
            agent: Agent to generate response
            prompt: Prompt for the agent
            context: Conversation context
            phase: Current debate phase (for telemetry)
            round_num: Current round number (for telemetry)

        Returns:
            Generated response (or system message on failure)
        """
        start_time = time.time()

        # Start performance tracking
        tracking_id = None
        if self.performance_monitor:
            tracking_id = self.performance_monitor.track_agent_call(
                agent.name, "generate", phase=phase, round_num=round_num
            )

        # Notify immune system that agent started
        if self.immune_system:
            self.immune_system.agent_started(agent.name, task=prompt[:100])

        try:
            raw_output = await agent.generate(prompt, context)
            response_ms = (time.time() - start_time) * 1000

            # Notify immune system of successful completion
            if self.immune_system:
                self.immune_system.agent_completed(agent.name, response_ms, success=True)

            sanitized = OutputSanitizer.sanitize_agent_output(raw_output, agent.name)
            empty_output = sanitized == "(Agent produced empty output)"

            # Retry once on empty output (qwen and other agents sometimes produce empty responses)
            if empty_output:
                logger.warning(
                    f"[Autonomic] Agent {agent.name} produced empty output, retrying once..."
                )
                retry_raw = await agent.generate(prompt, context)
                retry_sanitized = OutputSanitizer.sanitize_agent_output(retry_raw, agent.name)
                if retry_sanitized != "(Agent produced empty output)":
                    logger.info(f"[Autonomic] Agent {agent.name} retry succeeded")
                    sanitized = retry_sanitized
                    empty_output = False
                else:
                    logger.warning(
                        f"[Autonomic] Agent {agent.name} retry also produced empty output"
                    )

            if empty_output:
                if tracking_id and self.performance_monitor:
                    self.performance_monitor.record_completion(
                        tracking_id, success=False, error="empty output"
                    )

                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(agent.name)

                if self.immune_system:
                    self.immune_system.agent_failed(agent.name, "empty output", recoverable=True)

                self._emit_agent_telemetry(
                    agent.name,
                    "generate",
                    start_time,
                    success=False,
                    error="empty output",
                    input_text=prompt,
                )
                self._emit_agent_error(
                    agent.name,
                    error_type="empty",
                    message="Agent produced empty output",
                    recoverable=True,
                    phase=phase,
                )
                return sanitized

            # Validate response schema for type safety and size limits
            validation_result = validate_agent_response(
                content=sanitized,
                agent_name=agent.name,
                role=getattr(agent, "role", "proposer"),
                round_number=round_num,
            )
            if not validation_result.is_valid:
                logger.warning(
                    f"[Autonomic] Agent {agent.name} response validation failed: "
                    f"{validation_result.errors}"
                )
            elif validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.info(f"[Autonomic] Agent {agent.name} response warning: {warning}")

            # Record successful completion
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=True, response=sanitized
                )

            # Emit telemetry
            self._emit_agent_telemetry(
                agent.name,
                "generate",
                start_time,
                success=True,
                output=sanitized,
                input_text=prompt,
            )

            return sanitized
        except asyncio.TimeoutError as e:
            timeout_seconds = time.time() - start_time
            logger.warning(f"[Autonomic] Agent {agent.name} timed out")

            # Record timeout failure
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"timeout after {timeout_seconds:.1f}s"
                )

            # Notify immune system of timeout
            if self.immune_system:
                self.immune_system.agent_timeout(agent.name, timeout_seconds)

            # Emit telemetry for timeout
            self._emit_agent_telemetry(
                agent.name, "generate", start_time, success=False, error=e, input_text=prompt
            )

            # Emit agent error event for frontend visibility
            self._emit_agent_error(
                agent.name,
                error_type="timeout",
                message=f"Agent timed out after {timeout_seconds:.1f}s",
                recoverable=True,
                phase=phase,
            )

            # Use theatrical message if chaos director available
            if self.chaos_director:
                return self.chaos_director.timeout_response(agent.name, timeout_seconds).message
            return f"[System: Agent {agent.name} timed out - skipping this turn]"

        except (ConnectionError, OSError) as e:
            # Network/OS errors - log without full traceback
            logger.warning(f"[Autonomic] Agent {agent.name} connection error: {e}")

            # Record connection failure
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"connection error: {e}"
                )

            # Notify immune system of failure
            if self.immune_system:
                self.immune_system.agent_failed(agent.name, str(e), recoverable=True)

            # Emit telemetry for connection error
            self._emit_agent_telemetry(
                agent.name, "generate", start_time, success=False, error=e, input_text=prompt
            )

            # Emit agent error event for frontend visibility
            self._emit_agent_error(
                agent.name,
                error_type="connection",
                message=f"Connection failed: {e}",
                recoverable=True,
                phase=phase,
            )

            # Use theatrical message if chaos director available
            if self.chaos_director:
                return self.chaos_director.connection_response(agent.name).message
            return f"[System: Agent {agent.name} connection failed - skipping this turn]"

        except Exception as e:
            # Autonomic containment: convert crashes to valid responses
            logger.exception(f"[Autonomic] Agent {agent.name} failed: {type(e).__name__}: {e}")

            # Record exception failure
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"{type(e).__name__}: {e}"
                )

            # Notify immune system of failure
            if self.immune_system:
                self.immune_system.agent_failed(agent.name, str(e), recoverable=False)

            # Emit telemetry for exception
            self._emit_agent_telemetry(
                agent.name, "generate", start_time, success=False, error=e, input_text=prompt
            )

            # Emit agent error event for frontend visibility
            self._emit_agent_error(
                agent.name,
                error_type="internal",
                message=f"Internal error: {type(e).__name__}",
                recoverable=False,
                phase=phase,
            )

            # Use theatrical message if chaos director available
            if self.chaos_director:
                return self.chaos_director.internal_error_response(agent.name).message
            return f"[System: Agent {agent.name} encountered an error - skipping this turn]"

    async def critique(
        self,
        agent: "Agent",
        proposal: str,
        task: str,
        context: list["Message"],
        phase: str = "",
        round_num: int = 0,
        target_agent: Optional[str] = None,
    ) -> Optional["Critique"]:
        """
        Get critique from an agent with autonomic error handling.

        Args:
            agent: Agent to provide critique
            proposal: Proposal to critique
            task: Task description
            context: Conversation context
            phase: Current debate phase (for telemetry)
            round_num: Current round number (for telemetry)
            target_agent: Name of the agent being critiqued (for fallback messages)

        Returns:
            Critique object or None on failure
        """
        start_time = time.time()
        tracking_id = None
        if self.performance_monitor:
            tracking_id = self.performance_monitor.track_agent_call(
                agent.name, "critique", phase=phase, round_num=round_num
            )

        try:
            result = await agent.critique(proposal, task, context, target_agent=target_agent)
            if self._is_empty_critique(result):
                logger.warning(
                    f"[Autonomic] Agent {agent.name} returned empty critique, retrying once..."
                )
                retry_result = await agent.critique(
                    proposal, task, context, target_agent=target_agent
                )
                if not self._is_empty_critique(retry_result):
                    result = retry_result
                else:
                    logger.warning(
                        f"[Autonomic] Agent {agent.name} retry also returned empty critique"
                    )
                    if tracking_id and self.performance_monitor:
                        self.performance_monitor.record_completion(
                            tracking_id, success=False, error="empty critique"
                        )
                    self._emit_agent_telemetry(
                        agent.name,
                        "critique",
                        start_time,
                        success=False,
                        output=None,
                        input_text=proposal,
                    )
                    self._emit_agent_error(
                        agent.name,
                        error_type="empty",
                        message="Agent returned empty critique",
                        recoverable=True,
                        phase=phase,
                    )
                    return None
            if result is None:
                if tracking_id and self.performance_monitor:
                    self.performance_monitor.record_completion(
                        tracking_id, success=False, error="empty critique"
                    )
                self._emit_agent_telemetry(
                    agent.name,
                    "critique",
                    start_time,
                    success=False,
                    output=None,
                    input_text=proposal,
                )
                self._emit_agent_error(
                    agent.name,
                    error_type="empty",
                    message="Agent returned no critique",
                    recoverable=True,
                    phase=phase,
                )
                return None
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=True, response=str(result) if result else None
                )
            # Emit telemetry
            self._emit_agent_telemetry(
                agent.name,
                "critique",
                start_time,
                success=True,
                output=str(result) if result else None,
                input_text=proposal,
            )
            return result
        except asyncio.TimeoutError as e:
            logger.warning(f"[Autonomic] Agent {agent.name} critique timed out")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error="timeout"
                )
            self._emit_agent_telemetry(
                agent.name, "critique", start_time, success=False, error=e, input_text=proposal
            )
            self._emit_agent_error(
                agent.name,
                error_type="timeout",
                message="Critique timed out",
                recoverable=True,
                phase=phase,
            )
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} critique connection error: {e}")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"connection error: {e}"
                )
            self._emit_agent_telemetry(
                agent.name, "critique", start_time, success=False, error=e, input_text=proposal
            )
            self._emit_agent_error(
                agent.name,
                error_type="connection",
                message=f"Critique connection error: {type(e).__name__}",
                recoverable=True,
                phase=phase,
            )
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} critique failed: {e}")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"{type(e).__name__}: {e}"
                )
            self._emit_agent_telemetry(
                agent.name, "critique", start_time, success=False, error=e, input_text=proposal
            )
            self._emit_agent_error(
                agent.name,
                error_type="internal",
                message=f"Critique failed: {type(e).__name__}",
                recoverable=False,
                phase=phase,
            )
            return None

    async def vote(
        self,
        agent: "Agent",
        proposals: dict[str, str],
        task: str,
        phase: str = "",
        round_num: int = 0,
    ) -> Optional["Vote"]:
        """
        Get vote from an agent with autonomic error handling.

        Args:
            agent: Agent to vote
            proposals: Dict of agent_name -> proposal text
            task: Task description
            phase: Current debate phase (for telemetry)
            round_num: Current round number (for telemetry)

        Returns:
            Vote object or None on failure
        """
        start_time = time.time()
        input_text = f"{task}\n{str(proposals)}"
        tracking_id = None
        if self.performance_monitor:
            tracking_id = self.performance_monitor.track_agent_call(
                agent.name, "vote", phase=phase, round_num=round_num
            )

        try:
            result = await agent.vote(proposals, task)
            if result is None:
                if tracking_id and self.performance_monitor:
                    self.performance_monitor.record_completion(
                        tracking_id, success=False, error="empty vote"
                    )
                self._emit_agent_telemetry(
                    agent.name,
                    "vote",
                    start_time,
                    success=False,
                    output=None,
                    input_text=input_text,
                )
                self._emit_agent_error(
                    agent.name,
                    error_type="empty",
                    message="Agent returned no vote",
                    recoverable=True,
                    phase=phase,
                )
                return None
            if not str(getattr(result, "choice", "")).strip():
                if tracking_id and self.performance_monitor:
                    self.performance_monitor.record_completion(
                        tracking_id, success=False, error="empty vote choice"
                    )
                self._emit_agent_telemetry(
                    agent.name,
                    "vote",
                    start_time,
                    success=False,
                    output=None,
                    input_text=input_text,
                )
                self._emit_agent_error(
                    agent.name,
                    error_type="empty",
                    message="Agent returned empty vote choice",
                    recoverable=True,
                    phase=phase,
                )
                return None
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=True, response=str(result) if result else None
                )
            # Emit telemetry
            self._emit_agent_telemetry(
                agent.name,
                "vote",
                start_time,
                success=True,
                output=str(result) if result else None,
                input_text=input_text,
            )
            return result
        except asyncio.TimeoutError as e:
            logger.warning(f"[Autonomic] Agent {agent.name} vote timed out")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error="timeout"
                )
            self._emit_agent_telemetry(
                agent.name, "vote", start_time, success=False, error=e, input_text=input_text
            )
            self._emit_agent_error(
                agent.name,
                error_type="timeout",
                message="Vote timed out",
                recoverable=True,
                phase=phase,
            )
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} vote connection error: {e}")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"connection error: {e}"
                )
            self._emit_agent_telemetry(
                agent.name, "vote", start_time, success=False, error=e, input_text=input_text
            )
            self._emit_agent_error(
                agent.name,
                error_type="connection",
                message=f"Vote connection error: {type(e).__name__}",
                recoverable=True,
                phase=phase,
            )
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} vote failed: {e}")
            if tracking_id and self.performance_monitor:
                self.performance_monitor.record_completion(
                    tracking_id, success=False, error=f"{type(e).__name__}: {e}"
                )
            self._emit_agent_telemetry(
                agent.name, "vote", start_time, success=False, error=e, input_text=input_text
            )
            self._emit_agent_error(
                agent.name,
                error_type="internal",
                message=f"Vote failed: {type(e).__name__}",
                recoverable=False,
                phase=phase,
            )
            return None

    async def generate_with_fallback(
        self,
        agent: "Agent",
        prompt: str,
        context: list["Message"],
        fallback_agents: Optional[list["Agent"]] = None,
        max_retries: int = 2,
    ) -> str:
        """
        Generate response with automatic fallback to alternative agents.

        Tries the primary agent first. If it fails, tries fallback agents
        in order. Each retry gets an escalated timeout. If all agents fail,
        returns partial content from streaming buffer if available.

        Args:
            agent: Primary agent to generate response
            prompt: Prompt for the agent
            context: Conversation context
            fallback_agents: List of backup agents to try on failure
            max_retries: Maximum retries per agent before moving to fallback

        Returns:
            Generated response (or system message on total failure)
        """
        fallback_agents = fallback_agents or []
        all_agents = [agent] + fallback_agents
        last_error = None

        for current_agent in all_agents:
            # Skip agents that are circuit-broken
            if self.circuit_breaker and not self.circuit_breaker.is_available(current_agent.name):
                logger.info(f"[Autonomic] Skipping circuit-broken agent {current_agent.name}")
                continue

            for attempt in range(max_retries):
                try:
                    timeout = self.get_escalated_timeout(current_agent.name)
                    logger.debug(
                        f"[Autonomic] {current_agent.name} attempt {attempt + 1}/{max_retries}, "
                        f"timeout={timeout:.1f}s"
                    )

                    # Clear streaming buffer before attempt
                    self.streaming_buffer.clear_sync(current_agent.name)

                    raw_output = await asyncio.wait_for(
                        current_agent.generate(prompt, context),
                        timeout=timeout,
                    )

                    # Success - reset retry count and return
                    self.reset_retries(current_agent.name)
                    return OutputSanitizer.sanitize_agent_output(raw_output, current_agent.name)

                except asyncio.TimeoutError:
                    self.record_retry(current_agent.name)
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure(current_agent.name)

                    # Check for partial content
                    partial = self.streaming_buffer.get_partial(current_agent.name)
                    if partial and len(partial) > 100:
                        logger.warning(
                            f"[Autonomic] {current_agent.name} timed out but has "
                            f"{len(partial)} chars of partial content"
                        )
                        # Could use partial content as fallback

                    logger.warning(
                        f"[Autonomic] {current_agent.name} timed out on attempt "
                        f"{attempt + 1}/{max_retries}"
                    )
                    last_error = f"timeout after {timeout:.1f}s"

                except (ConnectionError, OSError) as e:
                    self.record_retry(current_agent.name)
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure(current_agent.name)
                    logger.warning(
                        f"[Autonomic] {current_agent.name} connection error on attempt "
                        f"{attempt + 1}: {e}"
                    )
                    last_error = str(e)

                except Exception as e:
                    self.record_retry(current_agent.name)
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure(current_agent.name)
                    logger.exception(
                        f"[Autonomic] {current_agent.name} failed on attempt "
                        f"{attempt + 1}: {type(e).__name__}: {e}"
                    )
                    last_error = str(e)
                    # Don't retry on unexpected errors
                    break

            # Agent exhausted retries, try next fallback
            logger.info(f"[Autonomic] Moving to fallback after {current_agent.name} failed")

        # All agents failed - check for any partial content
        for tried_agent in all_agents:
            partial = self.streaming_buffer.get_partial(tried_agent.name)
            if partial and len(partial) > 200:
                logger.info(
                    f"[Autonomic] Using partial content ({len(partial)} chars) "
                    f"from {tried_agent.name}"
                )
                sanitized = OutputSanitizer.sanitize_agent_output(partial, tried_agent.name)
                return f"{sanitized}\n\n[System: Response truncated due to timeout]"

        # Try audience wisdom as final fallback
        wisdom_response = self._get_wisdom_fallback(agent.name)
        if wisdom_response:
            return wisdom_response

        # Total failure
        tried_names = [a.name for a in all_agents]
        return f"[System: All agents failed ({', '.join(tried_names)}). Last error: {last_error}]"


__all__ = ["AutonomicExecutor", "StreamingContentBuffer"]
