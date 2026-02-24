"""Tests for debate latency optimization utilities.

Tests BatchedAgentCaller, CachedTeamSelector, and LatencyProfiler
from aragora.debate.optimizations.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.debate.optimizations import (
    AgentCallResult,
    BatchedAgentCaller,
    CachedTeamSelector,
    LatencyProfiler,
    PhaseTimingRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str) -> SimpleNamespace:
    """Create a minimal agent-like object."""
    return SimpleNamespace(name=name)


# ---------------------------------------------------------------------------
# BatchedAgentCaller
# ---------------------------------------------------------------------------


class TestBatchedAgentCaller:
    """Tests for BatchedAgentCaller."""

    @pytest.mark.asyncio
    async def test_call_all_returns_results_for_every_agent(self):
        agents = [_make_agent("a1"), _make_agent("a2"), _make_agent("a3")]

        async def call_fn(agent):
            return f"result_{agent.name}"

        caller = BatchedAgentCaller(max_concurrency=2)
        results = await caller.call_all(agents, call_fn)

        assert len(results) == 3
        names = {r.agent_name for r in results}
        assert names == {"a1", "a2", "a3"}
        for r in results:
            assert r.success
            assert r.result == f"result_{r.agent_name}"

    @pytest.mark.asyncio
    async def test_call_all_empty_agents_returns_empty(self):
        caller = BatchedAgentCaller(max_concurrency=3)
        results = await caller.call_all([], lambda a: asyncio.sleep(0))
        assert results == []

    @pytest.mark.asyncio
    async def test_call_all_respects_concurrency_limit(self):
        """Verify that no more than max_concurrency calls run simultaneously."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_call(agent):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return agent.name

        agents = [_make_agent(f"agent_{i}") for i in range(6)]
        caller = BatchedAgentCaller(max_concurrency=2)
        results = await caller.call_all(agents, tracked_call)

        assert len(results) == 6
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_call_all_handles_timeout(self):
        async def slow_call(agent):
            await asyncio.sleep(5)
            return "done"

        caller = BatchedAgentCaller(max_concurrency=2, timeout_seconds=0.1)
        results = await caller.call_all([_make_agent("slow")], slow_call)

        assert len(results) == 1
        assert not results[0].success
        assert isinstance(results[0].error, asyncio.TimeoutError)
        assert results[0].duration_ms > 0

    @pytest.mark.asyncio
    async def test_call_all_handles_exception(self):
        async def failing_call(agent):
            raise ValueError("test error")

        caller = BatchedAgentCaller(max_concurrency=2)
        results = await caller.call_all([_make_agent("fail")], failing_call)

        assert len(results) == 1
        assert not results[0].success
        assert isinstance(results[0].error, ValueError)

    @pytest.mark.asyncio
    async def test_call_all_mixed_success_and_failure(self):
        async def mixed_call(agent):
            if agent.name == "bad":
                raise RuntimeError("boom")
            return "ok"

        agents = [_make_agent("good"), _make_agent("bad")]
        caller = BatchedAgentCaller(max_concurrency=5)
        results = await caller.call_all(agents, mixed_call)

        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0].agent_name == "bad"

    @pytest.mark.asyncio
    async def test_call_all_records_duration(self):
        async def timed_call(agent):
            await asyncio.sleep(0.05)
            return "done"

        caller = BatchedAgentCaller(max_concurrency=1)
        results = await caller.call_all([_make_agent("timed")], timed_call)

        assert results[0].duration_ms >= 40  # at least ~50ms sleep

    def test_invalid_max_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            BatchedAgentCaller(max_concurrency=0)

    @pytest.mark.asyncio
    async def test_no_timeout_when_none(self):
        async def fast_call(agent):
            return "fast"

        caller = BatchedAgentCaller(max_concurrency=2, timeout_seconds=None)
        results = await caller.call_all([_make_agent("x")], fast_call)
        assert results[0].success


# ---------------------------------------------------------------------------
# CachedTeamSelector
# ---------------------------------------------------------------------------


class TestCachedTeamSelector:
    """Tests for CachedTeamSelector."""

    def _make_selector(self):
        """Create a mock TeamSelector."""
        selector = MagicMock()
        selector.select.return_value = [_make_agent("a1"), _make_agent("a2")]
        return selector

    def test_cache_miss_delegates_to_selector(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1"), _make_agent("a2")]
        result = cached.select(agents, domain="code", task="review")

        selector.select.assert_called_once_with(
            agents, domain="code", task="review"
        )
        assert len(result) == 2

    def test_cache_hit_avoids_delegate(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="code", task="review")
        result2 = cached.select(agents, domain="code", task="review")

        # Underlying selector called only once
        assert selector.select.call_count == 1
        assert result2 is cached._cache[("code", "review")]

    def test_different_domain_is_cache_miss(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="code", task="review")
        cached.select(agents, domain="research", task="review")

        assert selector.select.call_count == 2

    def test_different_task_is_cache_miss(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="code", task="review PR")
        cached.select(agents, domain="code", task="fix bug")

        assert selector.select.call_count == 2

    def test_clear_resets_cache(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="code", task="review")
        assert cached.stats["size"] == 1

        cached.clear()
        assert cached.stats == {"hits": 0, "misses": 0, "size": 0}

    def test_stats_tracking(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="code", task="t1")  # miss
        cached.select(agents, domain="code", task="t1")  # hit
        cached.select(agents, domain="code", task="t2")  # miss
        cached.select(agents, domain="code", task="t1")  # hit

        assert cached.stats == {"hits": 2, "misses": 2, "size": 2}

    def test_max_entries_eviction(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector, max_entries=2)

        agents = [_make_agent("a1")]
        cached.select(agents, domain="d1", task="t")
        cached.select(agents, domain="d2", task="t")
        cached.select(agents, domain="d3", task="t")

        # Only 2 entries should remain (oldest evicted)
        assert cached.stats["size"] == 2
        assert ("d1", "t") not in cached._cache
        assert ("d2", "t") in cached._cache
        assert ("d3", "t") in cached._cache

    def test_kwargs_forwarded_on_miss(self):
        selector = self._make_selector()
        cached = CachedTeamSelector(selector)

        agents = [_make_agent("a1")]
        cached.select(
            agents, domain="code", task="review", debate_id="d1", context=None
        )

        selector.select.assert_called_once_with(
            agents, domain="code", task="review", debate_id="d1", context=None
        )

    def test_proxy_attribute_access(self):
        selector = self._make_selector()
        selector.config = SimpleNamespace(elo_weight=0.3)
        cached = CachedTeamSelector(selector)

        assert cached.config.elo_weight == 0.3


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------


class TestLatencyProfiler:
    """Tests for LatencyProfiler."""

    @pytest.mark.asyncio
    async def test_phase_records_timing(self):
        profiler = LatencyProfiler()

        async with profiler.phase("test_phase") as record:
            await asyncio.sleep(0.05)

        assert len(profiler.records) == 1
        assert profiler.records[0].phase_name == "test_phase"
        assert profiler.records[0].duration_ms >= 40

    @pytest.mark.asyncio
    async def test_multiple_phases(self):
        profiler = LatencyProfiler()

        async with profiler.phase("phase_a"):
            await asyncio.sleep(0.02)
        async with profiler.phase("phase_b"):
            await asyncio.sleep(0.01)

        assert len(profiler.records) == 2
        names = [r.phase_name for r in profiler.records]
        assert names == ["phase_a", "phase_b"]

    @pytest.mark.asyncio
    async def test_report_returns_summary(self):
        profiler = LatencyProfiler()

        async with profiler.phase("slow"):
            await asyncio.sleep(0.05)
        async with profiler.phase("fast"):
            await asyncio.sleep(0.01)

        summary = profiler.report()

        assert "phase_durations_ms" in summary
        assert "slow" in summary["phase_durations_ms"]
        assert "fast" in summary["phase_durations_ms"]
        assert summary["phase_count"] == 2
        assert summary["slowest_phase"] == "slow"
        assert summary["total_phase_ms"] > 0
        assert summary["wall_clock_ms"] >= summary["total_phase_ms"]

    @pytest.mark.asyncio
    async def test_phase_metadata(self):
        profiler = LatencyProfiler()

        async with profiler.phase("test", agents=3, domain="code"):
            pass

        assert profiler.records[0].metadata == {"agents": 3, "domain": "code"}

    @pytest.mark.asyncio
    async def test_phase_records_error_metadata(self):
        profiler = LatencyProfiler()

        with pytest.raises(ValueError):
            async with profiler.phase("failing"):
                raise ValueError("boom")

        assert profiler.records[0].metadata.get("error") == "ValueError"
        assert profiler.records[0].duration_ms >= 0

    def test_clear_resets_state(self):
        profiler = LatencyProfiler()
        profiler._records.append(
            PhaseTimingRecord(phase_name="x", duration_ms=100)
        )
        profiler._debate_start = 1.0

        profiler.clear()

        assert profiler.records == []
        assert profiler._debate_start == 0.0

    @pytest.mark.asyncio
    async def test_report_empty_profiler(self):
        profiler = LatencyProfiler()
        summary = profiler.report()

        assert summary["phase_count"] == 0
        assert summary["slowest_phase"] is None
        assert summary["total_phase_ms"] == 0


# ---------------------------------------------------------------------------
# AgentCallResult
# ---------------------------------------------------------------------------


class TestAgentCallResult:
    """Tests for AgentCallResult dataclass."""

    def test_success_when_no_error(self):
        r = AgentCallResult(agent_name="a1", result="ok")
        assert r.success

    def test_not_success_when_error(self):
        r = AgentCallResult(agent_name="a1", error=ValueError("bad"))
        assert not r.success

    def test_duration_default_zero(self):
        r = AgentCallResult(agent_name="a1")
        assert r.duration_ms == 0.0
