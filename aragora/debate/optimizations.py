"""Debate latency optimization utilities.

Provides opt-in performance utilities for the debate execution hot path:

- ``BatchedAgentCaller``: Execute agent calls with bounded concurrency and
  per-call timeout, returning results as they complete.
- ``CachedTeamSelector``: Wraps ``TeamSelector`` to cache selection results
  for repeated same-domain queries within a single debate.
- ``LatencyProfiler``: Async context manager that measures and logs
  per-phase timing with structured output.

These are non-invasive -- they wrap existing APIs without changing them.

Usage::

    from aragora.debate.optimizations import (
        BatchedAgentCaller,
        CachedTeamSelector,
        LatencyProfiler,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# BatchedAgentCaller
# ---------------------------------------------------------------------------


@dataclass
class AgentCallResult:
    """Result from a single agent call within a batch."""

    agent_name: str
    result: Any = None
    error: Exception | None = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


class BatchedAgentCaller:
    """Execute agent calls concurrently with configurable limits.

    Wraps an async agent-call function and dispatches calls through an
    ``asyncio.Semaphore`` to enforce a maximum concurrency level.  Results
    are collected as they complete, enabling the caller to process early
    finishers while slower agents are still running.

    Args:
        max_concurrency: Maximum number of simultaneous agent calls.
        timeout_seconds: Per-call timeout.  ``None`` disables the timeout.

    Example::

        caller = BatchedAgentCaller(max_concurrency=5, timeout_seconds=60)
        results = await caller.call_all(
            agents=agents,
            call_fn=lambda agent: generate(agent, prompt, context),
        )
        for r in results:
            if r.success:
                process(r.agent_name, r.result)
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        timeout_seconds: float | None = 60.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max_concurrency = max_concurrency
        self._timeout_seconds = timeout_seconds
        self._semaphore: asyncio.Semaphore | None = None

    async def call_all(
        self,
        agents: list[Any],
        call_fn: Callable[[Any], Awaitable[Any]],
    ) -> list[AgentCallResult]:
        """Invoke *call_fn* for every agent with bounded concurrency.

        Args:
            agents: Agent objects (must have a ``.name`` attribute).
            call_fn: Async callable accepting a single agent and returning
                     a result.

        Returns:
            List of ``AgentCallResult`` in completion order.
        """
        if not agents:
            return []

        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        results: list[AgentCallResult] = []

        async def _bounded_call(agent: Any) -> AgentCallResult:
            name = getattr(agent, "name", str(agent))
            start = time.perf_counter()
            async with self._semaphore:  # type: ignore[union-attr]
                try:
                    coro = call_fn(agent)
                    if self._timeout_seconds is not None:
                        value = await asyncio.wait_for(coro, timeout=self._timeout_seconds)
                    else:
                        value = await coro
                    elapsed = (time.perf_counter() - start) * 1000
                    return AgentCallResult(agent_name=name, result=value, duration_ms=elapsed)
                except asyncio.TimeoutError:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.warning(
                        "batched_call_timeout agent=%s timeout=%.1fs",
                        name,
                        self._timeout_seconds,
                    )
                    return AgentCallResult(
                        agent_name=name,
                        error=asyncio.TimeoutError(
                            f"Agent {name} timed out after {self._timeout_seconds}s"
                        ),
                        duration_ms=elapsed,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.warning(
                        "batched_call_error agent=%s error=%s",
                        name,
                        exc,
                    )
                    return AgentCallResult(agent_name=name, error=exc, duration_ms=elapsed)

        tasks = [
            asyncio.create_task(_bounded_call(agent), name=f"batch_{getattr(agent, 'name', i)}")
            for i, agent in enumerate(agents)
        ]

        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# CachedTeamSelector
# ---------------------------------------------------------------------------


class CachedTeamSelector:
    """Caching wrapper around ``TeamSelector.select()``.

    Within a single debate, team selection for the same domain and task is
    often repeated (e.g., across rounds).  This wrapper caches the result
    keyed by ``(domain, task)`` and returns the cached list on subsequent
    calls, avoiding the full multi-factor scoring pipeline.

    The cache is scoped per debate: call ``clear()`` between debates or
    create a new instance.

    Args:
        selector: The underlying ``TeamSelector`` to delegate to.
        max_entries: Maximum cache entries before oldest is evicted.

    Example::

        cached = CachedTeamSelector(team_selector)
        team1 = cached.select(agents, domain="code", task="Review PR")
        team2 = cached.select(agents, domain="code", task="Review PR")
        assert team1 is team2  # Cache hit
    """

    def __init__(
        self,
        selector: Any,
        max_entries: int = 32,
    ) -> None:
        self._selector = selector
        self._max_entries = max_entries
        self._cache: dict[tuple[str, str], list[Any]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def select(
        self,
        agents: list[Any],
        domain: str = "general",
        task: str = "",
        **kwargs: Any,
    ) -> list[Any]:
        """Select agents, returning cached result when available.

        Cache key is ``(domain, task)``.  Additional keyword arguments
        (``context``, ``required_hierarchy_roles``, ``debate_id``) are
        forwarded to the underlying selector on cache misses.

        Args:
            agents: Candidate agent list.
            domain: Task domain for context-aware selection.
            task: Task description for delegation-based routing.
            **kwargs: Extra arguments forwarded to the underlying selector.

        Returns:
            Sorted list of agents.
        """
        cache_key = (domain, task)

        if cache_key in self._cache:
            self._hits += 1
            logger.debug(
                "team_selection_cache_hit domain=%s hits=%d misses=%d",
                domain,
                self._hits,
                self._misses,
            )
            return self._cache[cache_key]

        self._misses += 1

        # Delegate to underlying selector
        result = self._selector.select(agents, domain=domain, task=task, **kwargs)

        # Evict oldest entry if cache is full
        if len(self._cache) >= self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result
        return result

    def clear(self) -> None:
        """Clear the selection cache.  Call between debates."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    # Proxy common TeamSelector attributes/methods to the underlying selector
    def __getattr__(self, name: str) -> Any:
        return getattr(self._selector, name)


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimingRecord:
    """Timing data for a single profiled phase."""

    phase_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LatencyProfiler:
    """Async context manager that measures and logs per-phase timing.

    Designed to wrap existing phase execution calls non-invasively.
    Collects timing data for each phase and emits structured log lines
    at the end of the debate or when ``report()`` is called.

    Usage::

        profiler = LatencyProfiler()

        async with profiler.phase("context_init"):
            await context_initializer.initialize(ctx)

        async with profiler.phase("proposals"):
            await proposal_phase.execute(ctx)

        profiler.report()  # logs structured timing summary

    Or wrap a ``PhaseExecutor`` with :meth:`wrap_executor`.
    """

    def __init__(self) -> None:
        self._records: list[PhaseTimingRecord] = []
        self._debate_start: float = 0.0

    def phase(self, name: str, **metadata: Any) -> _PhaseContext:
        """Return an async context manager that times the enclosed block.

        Args:
            name: Phase name (e.g. ``"proposals"``, ``"consensus"``).
            **metadata: Extra key/value pairs stored on the timing record.

        Returns:
            Async context manager.
        """
        record = PhaseTimingRecord(phase_name=name, metadata=metadata)
        self._records.append(record)
        if not self._debate_start:
            self._debate_start = time.perf_counter()
        return _PhaseContext(record)

    def report(self) -> dict[str, Any]:
        """Log and return a structured timing summary.

        Returns:
            Dictionary with per-phase durations, total time, and slowest
            phase identification.
        """
        total_ms = sum(r.duration_ms for r in self._records)
        wall_ms = (
            (time.perf_counter() - self._debate_start) * 1000 if self._debate_start else total_ms
        )
        phase_durations = {r.phase_name: round(r.duration_ms, 2) for r in self._records}
        slowest = max(self._records, key=lambda r: r.duration_ms) if self._records else None

        summary = {
            "phase_durations_ms": phase_durations,
            "total_phase_ms": round(total_ms, 2),
            "wall_clock_ms": round(wall_ms, 2),
            "overhead_ms": round(wall_ms - total_ms, 2),
            "phase_count": len(self._records),
            "slowest_phase": slowest.phase_name if slowest else None,
            "slowest_phase_ms": round(slowest.duration_ms, 2) if slowest else 0.0,
        }

        logger.info(
            "latency_profile total_ms=%.1f wall_ms=%.1f slowest=%s slowest_ms=%.1f phases=%s",
            total_ms,
            wall_ms,
            summary["slowest_phase"],
            summary["slowest_phase_ms"],
            phase_durations,
        )

        return summary

    @property
    def records(self) -> list[PhaseTimingRecord]:
        """Return all collected timing records."""
        return list(self._records)

    def clear(self) -> None:
        """Reset all collected timing data."""
        self._records.clear()
        self._debate_start = 0.0


class _PhaseContext:
    """Async context manager that records timing for a single phase."""

    def __init__(self, record: PhaseTimingRecord) -> None:
        self._record = record

    async def __aenter__(self) -> PhaseTimingRecord:
        self._record.start_time = time.perf_counter()
        return self._record

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self._record.end_time = time.perf_counter()
        self._record.duration_ms = (self._record.end_time - self._record.start_time) * 1000
        if exc_type is not None:
            self._record.metadata["error"] = exc_type.__name__


__all__ = [
    "AgentCallResult",
    "BatchedAgentCaller",
    "CachedTeamSelector",
    "LatencyProfiler",
    "PhaseTimingRecord",
]
