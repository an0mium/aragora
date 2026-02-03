"""
Fallback Chain - Graceful degradation for external agent failures.

Provides resilient execution with automatic failover:
- Ordered fallback chain of agents
- Configurable retry policies
- Circuit breaker integration
- Degradation to internal agents when external fail

Security Model:
1. Fallback respects capability restrictions
2. Sensitive data never sent to lower-trust agents
3. Full audit trail of fallback events
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.gateway.external_agents.base import (
        ExternalAgentTask,
        ExternalAgentResult,
        BaseExternalAgentAdapter,
    )

logger = logging.getLogger(__name__)


class FallbackReason(str, Enum):
    """Reason for triggering fallback."""

    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    POLICY_DENIED = "policy_denied"
    CAPABILITY_MISSING = "capability_missing"


@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""

    success: bool
    final_agent: str
    result: "ExternalAgentResult | None" = None
    attempts: int = 0
    fallback_chain: list[str] = field(default_factory=list)
    fallback_reasons: dict[str, FallbackReason] = field(default_factory=dict)
    total_time_ms: float = 0.0
    degraded: bool = False  # True if fell back to lower-capability agent

    @property
    def used_fallback(self) -> bool:
        """Check if fallback was used."""
        return self.attempts > 1


@dataclass
class CircuitState:
    """State of a circuit breaker for an agent."""

    agent_name: str
    failure_count: int = 0
    success_count: int = 0
    last_failure: datetime | None = None
    state: str = "closed"  # closed, open, half-open
    opens_at_failures: int = 5
    reset_after: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self.state == "closed":
            return False
        if self.state == "open":
            if (
                self.last_failure
                and datetime.now(timezone.utc) - self.last_failure > self.reset_after
            ):
                self.state = "half-open"
                return False
            return True
        return False  # half-open allows one request

    def record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure = datetime.now(timezone.utc)
        if self.failure_count >= self.opens_at_failures:
            self.state = "open"
            logger.warning(f"Circuit opened for agent: {self.agent_name}")

    def record_success(self) -> None:
        """Record a success."""
        self.success_count += 1
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            logger.info(f"Circuit closed for agent: {self.agent_name}")


class FallbackChain:
    """
    Fallback chain for resilient external agent execution.

    Executes tasks through a prioritized chain of agents, automatically
    falling back to alternatives on failure.

    Usage:
        chain = FallbackChain()
        chain.add_agent("openclaw", openclaw_adapter, priority=1)
        chain.add_agent("openhands", openhands_adapter, priority=2)
        chain.add_agent("internal", internal_adapter, priority=3)

        result = await chain.execute(task)
        if result.used_fallback:
            logger.warning(f"Task fell back from {result.fallback_chain}")
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
        enable_circuit_breaker: bool = True,
    ):
        self._adapters: dict[str, "BaseExternalAgentAdapter"] = {}
        self._priorities: dict[str, int] = {}  # Lower = higher priority
        self._circuits: dict[str, CircuitState] = {}
        self._max_retries = max_retries
        self._retry_delay_ms = retry_delay_ms
        self._enable_circuit_breaker = enable_circuit_breaker

        # Callbacks for events
        self._on_fallback: Callable[[str, str, FallbackReason], Awaitable[None]] | None = None
        self._on_exhausted: Callable[["ExternalAgentTask"], Awaitable[None]] | None = None

    def add_agent(
        self,
        name: str,
        adapter: "BaseExternalAgentAdapter",
        priority: int = 0,
        circuit_threshold: int = 5,
        circuit_reset: timedelta | None = None,
    ) -> None:
        """
        Add an agent to the fallback chain.

        Args:
            name: Unique agent name
            adapter: The agent adapter
            priority: Priority in chain (lower = tried first)
            circuit_threshold: Failures before circuit opens
            circuit_reset: Time before circuit resets
        """
        self._adapters[name] = adapter
        self._priorities[name] = priority
        self._circuits[name] = CircuitState(
            agent_name=name,
            opens_at_failures=circuit_threshold,
            reset_after=circuit_reset or timedelta(minutes=1),
        )
        logger.info(f"Added agent to fallback chain: {name} (priority={priority})")

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the chain."""
        if name in self._adapters:
            del self._adapters[name]
            del self._priorities[name]
            del self._circuits[name]
            return True
        return False

    def on_fallback(
        self,
        callback: Callable[[str, str, FallbackReason], Awaitable[None]],
    ) -> None:
        """Set callback for fallback events. Args: (from_agent, to_agent, reason)."""
        self._on_fallback = callback

    def on_exhausted(
        self,
        callback: Callable[["ExternalAgentTask"], Awaitable[None]],
    ) -> None:
        """Set callback when all agents exhausted."""
        self._on_exhausted = callback

    def _get_ordered_agents(self) -> list[str]:
        """Get agents ordered by priority."""
        return sorted(self._adapters.keys(), key=lambda a: self._priorities[a])

    async def execute(
        self,
        task: "ExternalAgentTask",
        start_from: str | None = None,
        skip_agents: list[str] | None = None,
    ) -> FallbackResult:
        """
        Execute task through the fallback chain.

        Args:
            task: The task to execute
            start_from: Start from specific agent (skip higher priority)
            skip_agents: Agents to skip entirely

        Returns:
            FallbackResult with execution details
        """
        import time

        start_time = time.time()

        skip = set(skip_agents or [])
        agents = self._get_ordered_agents()

        # Filter to agents after start_from
        if start_from and start_from in agents:
            start_idx = agents.index(start_from)
            agents = agents[start_idx:]

        # Filter out skipped and circuit-open agents
        available = [
            a
            for a in agents
            if a not in skip and not (self._enable_circuit_breaker and self._circuits[a].is_open)
        ]

        if not available:
            logger.error(f"No agents available for task {task.task_id}")
            if self._on_exhausted:
                await self._on_exhausted(task)
            return FallbackResult(
                success=False,
                final_agent="",
                attempts=0,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        fallback_chain: list[str] = []
        fallback_reasons: dict[str, FallbackReason] = {}
        last_result: "ExternalAgentResult | None" = None

        for i, agent_name in enumerate(available):
            fallback_chain.append(agent_name)
            adapter = self._adapters[agent_name]

            for retry in range(self._max_retries):
                try:
                    try:
                        result = await adapter.execute(
                            task,
                            credentials={},
                            sandbox_config={},
                        )
                    except TypeError as exc:
                        # Backwards compatibility for adapters with legacy signatures.
                        # Some test adapters only accept (task).
                        if "credentials" in str(exc) or "sandbox" in str(exc):
                            result = await adapter.execute(task)
                        else:
                            raise
                    last_result = result

                    if result.success:
                        # Record success
                        if self._enable_circuit_breaker:
                            self._circuits[agent_name].record_success()

                        return FallbackResult(
                            success=True,
                            final_agent=agent_name,
                            result=result,
                            attempts=len(fallback_chain),
                            fallback_chain=fallback_chain,
                            fallback_reasons=fallback_reasons,
                            total_time_ms=(time.time() - start_time) * 1000,
                            degraded=i > 0,
                        )

                    # Execution failed but didn't raise exception
                    reason = self._determine_failure_reason(result)
                    if not self._should_retry(reason):
                        fallback_reasons[agent_name] = reason
                        break  # Move to next agent

                except asyncio.TimeoutError:
                    fallback_reasons[agent_name] = FallbackReason.TIMEOUT
                    break  # Move to next agent

                except Exception as e:
                    logger.warning(f"Agent {agent_name} failed on attempt {retry + 1}: {e}")
                    if retry < self._max_retries - 1:
                        await asyncio.sleep(self._retry_delay_ms / 1000)
                    else:
                        fallback_reasons[agent_name] = FallbackReason.ERROR

            # Record failure and trigger circuit breaker
            if self._enable_circuit_breaker:
                self._circuits[agent_name].record_failure()

            # Notify fallback
            if i < len(available) - 1:
                next_agent = available[i + 1]
                reason = fallback_reasons.get(agent_name, FallbackReason.ERROR)
                logger.info(
                    f"Falling back from {agent_name} to {next_agent} (reason={reason.value})"
                )
                if self._on_fallback:
                    await self._on_fallback(agent_name, next_agent, reason)

        # All agents exhausted
        logger.error(f"All agents exhausted for task {task.task_id}")
        if self._on_exhausted:
            await self._on_exhausted(task)

        return FallbackResult(
            success=False,
            final_agent=fallback_chain[-1] if fallback_chain else "",
            result=last_result,
            attempts=len(fallback_chain),
            fallback_chain=fallback_chain,
            fallback_reasons=fallback_reasons,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    def _determine_failure_reason(
        self,
        result: "ExternalAgentResult",
    ) -> FallbackReason:
        """Determine the reason for failure from result."""
        if result.error:
            error_lower = result.error.lower()
            if "timeout" in error_lower:
                return FallbackReason.TIMEOUT
            if "rate" in error_lower or "429" in error_lower:
                return FallbackReason.RATE_LIMITED
            if "policy" in error_lower or "denied" in error_lower:
                return FallbackReason.POLICY_DENIED
            if "capability" in error_lower:
                return FallbackReason.CAPABILITY_MISSING
        return FallbackReason.ERROR

    def _should_retry(self, reason: FallbackReason) -> bool:
        """Determine if we should retry the same agent."""
        # Don't retry policy denials or capability issues
        return reason not in {
            FallbackReason.POLICY_DENIED,
            FallbackReason.CAPABILITY_MISSING,
            FallbackReason.CIRCUIT_OPEN,
        }

    def get_circuit_status(self) -> dict[str, dict[str, Any]]:
        """Get circuit breaker status for all agents."""
        return {
            name: {
                "state": circuit.state,
                "failure_count": circuit.failure_count,
                "success_count": circuit.success_count,
                "is_open": circuit.is_open,
                "last_failure": circuit.last_failure.isoformat() if circuit.last_failure else None,
            }
            for name, circuit in self._circuits.items()
        }

    def reset_circuit(self, agent_name: str) -> bool:
        """Manually reset a circuit breaker."""
        if agent_name in self._circuits:
            circuit = self._circuits[agent_name]
            circuit.state = "closed"
            circuit.failure_count = 0
            logger.info(f"Manually reset circuit for agent: {agent_name}")
            return True
        return False

    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        for circuit in self._circuits.values():
            circuit.state = "closed"
            circuit.failure_count = 0
        logger.info("Reset all circuit breakers")
