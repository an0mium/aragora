#!/usr/bin/env python3
"""Performance benchmarks for aragora-debate package.

Measures latency and memory for debate operations with mock agents.
Runnable as both a script and pytest tests.

Usage:
    python tests/benchmarks/test_debate_perf.py
    pytest tests/benchmarks/test_debate_perf.py -v
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

import pytest

pytest.importorskip("aragora_debate", reason="aragora_debate package is required for debate benchmarks")

from aragora_debate import Debate, create_agent
from aragora_debate.receipt import ReceiptBuilder


# ── Helpers ────────────────────────────────────────────────────────────


def _timed_runs(fn: Any, iterations: int) -> list[float]:
    """Run fn() `iterations` times and return durations in ms."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if asyncio.iscoroutine(result):
            asyncio.get_event_loop().run_until_complete(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return times


async def _async_timed_runs(coro_fn: Any, iterations: int) -> list[float]:
    """Run async fn() `iterations` times and return durations in ms."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        await coro_fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return times


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile from sorted data."""
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_d):
        return sorted_d[f]
    return sorted_d[f] + (k - f) * (sorted_d[c] - sorted_d[f])


def _stats(times: list[float]) -> dict[str, float]:
    """Compute min/p50/p95/p99/max from timing data."""
    return {
        "min": min(times),
        "p50": _percentile(times, 50),
        "p95": _percentile(times, 95),
        "p99": _percentile(times, 99),
        "max": max(times),
    }


def _print_row(name: str, stats: dict[str, float]) -> None:
    """Print a formatted benchmark row."""
    print(
        f"  {name:<40} {stats['min']:>8.2f} {stats['p50']:>8.2f} "
        f"{stats['p95']:>8.2f} {stats['p99']:>8.2f} {stats['max']:>8.2f}"
    )


def _make_debate(n_agents: int = 3, rounds: int = 3, trickster: bool = False) -> Debate:
    """Create a debate with n mock agents."""
    debate = Debate(
        topic="Should we adopt microservices architecture?",
        rounds=rounds,
        consensus="majority",
        enable_trickster=trickster,
        trickster_sensitivity=0.5,
    )
    for i in range(n_agents):
        debate.add_agent(
            create_agent(
                "mock",
                name=f"agent-{i}",
                proposal=f"Agent {i} proposes approach {i} with detailed reasoning.",
                vote_for="agent-0",
            )
        )
    return debate


# ── Micro benchmarks ──────────────────────────────────────────────────


class TestMicroBenchmarks:
    """Micro benchmarks for individual operations."""

    def test_agent_creation(self) -> None:
        """Benchmark mock agent creation."""
        times = _timed_runs(
            lambda: create_agent("mock", name="bench", proposal="test"),
            iterations=500,
        )
        stats = _stats(times)
        assert stats["p95"] < 5.0, f"Agent creation too slow: p95={stats['p95']:.2f}ms"

    @pytest.mark.asyncio
    async def test_receipt_generation(self) -> None:
        """Benchmark receipt generation via full debate + receipt extraction."""
        async def run_and_extract() -> None:
            debate = _make_debate(n_agents=3, rounds=1)
            result = await debate.run()
            assert result.receipt is not None
            ReceiptBuilder.to_json(result.receipt)

        times = await _async_timed_runs(run_and_extract, iterations=50)
        stats = _stats(times)
        assert stats["p95"] < 50.0, f"Receipt gen too slow: p95={stats['p95']:.2f}ms"

    @pytest.mark.asyncio
    async def test_consensus_detection(self) -> None:
        """Benchmark consensus detection across methods."""
        for method in ["majority", "supermajority", "unanimous"]:
            debate = Debate(
                topic="Benchmark consensus",
                rounds=1,
                consensus=method,
            )
            for i in range(5):
                debate.add_agent(create_agent("mock", name=f"a-{i}", vote_for="a-0"))

            times = await _async_timed_runs(
                lambda: Debate(
                    topic="Benchmark", rounds=1, consensus=method
                ).add_agent(create_agent("mock", name="x", vote_for="x"))
                .add_agent(create_agent("mock", name="y", vote_for="x"))
                .run(),
                iterations=50,
            )
            stats = _stats(times)
            assert stats["p95"] < 50.0, f"{method} consensus too slow: p95={stats['p95']:.2f}ms"


# ── Debate benchmarks ─────────────────────────────────────────────────


class TestDebateBenchmarks:
    """End-to-end debate benchmarks at various scales."""

    @pytest.mark.asyncio
    async def test_small_debate(self) -> None:
        """3-round, 3-agent debate."""
        times = await _async_timed_runs(
            lambda: _make_debate(n_agents=3, rounds=3).run(),
            iterations=20,
        )
        stats = _stats(times)
        assert stats["p95"] < 100.0, f"Small debate too slow: p95={stats['p95']:.2f}ms"

    @pytest.mark.asyncio
    async def test_medium_debate(self) -> None:
        """3-round, 5-agent debate."""
        times = await _async_timed_runs(
            lambda: _make_debate(n_agents=5, rounds=3).run(),
            iterations=20,
        )
        stats = _stats(times)
        assert stats["p95"] < 200.0, f"Medium debate too slow: p95={stats['p95']:.2f}ms"

    @pytest.mark.asyncio
    async def test_large_debate(self) -> None:
        """5-round, 10-agent debate."""
        times = await _async_timed_runs(
            lambda: _make_debate(n_agents=10, rounds=5).run(),
            iterations=10,
        )
        stats = _stats(times)
        assert stats["p95"] < 500.0, f"Large debate too slow: p95={stats['p95']:.2f}ms"

    @pytest.mark.asyncio
    async def test_trickster_debate(self) -> None:
        """3-round, 5-agent debate with trickster enabled."""
        times = await _async_timed_runs(
            lambda: _make_debate(n_agents=5, rounds=3, trickster=True).run(),
            iterations=20,
        )
        stats = _stats(times)
        assert stats["p95"] < 300.0, f"Trickster debate too slow: p95={stats['p95']:.2f}ms"


# ── Memory benchmark ──────────────────────────────────────────────────


class TestMemoryBenchmark:
    """Memory usage benchmarks."""

    @pytest.mark.asyncio
    async def test_sequential_debates_memory(self) -> None:
        """Run 50 debates and check memory growth stays bounded."""
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(50):
            debate = _make_debate(n_agents=5, rounds=2)
            await debate.run()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats_before = snapshot_before.statistics("lineno")
        stats_after = snapshot_after.statistics("lineno")

        mem_before = sum(s.size for s in stats_before)
        mem_after = sum(s.size for s in stats_after)
        growth_mb = (mem_after - mem_before) / (1024 * 1024)

        # Memory growth should be <10MB for 50 debates
        assert growth_mb < 10.0, f"Memory growth too high: {growth_mb:.1f}MB for 50 debates"


# ── CLI runner ─────────────────────────────────────────────────────────


async def _run_all_benchmarks() -> None:
    """Run all benchmarks and print results table."""
    print()
    print("=" * 88)
    print("  Aragora Debate Performance Benchmarks")
    print("=" * 88)
    print()
    print(f"  {'Benchmark':<40} {'min':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    # Agent creation (500 iterations)
    times = _timed_runs(lambda: create_agent("mock", name="b", proposal="x"), 500)
    _print_row("Agent creation (500x)", _stats(times))

    # Receipt generation via debate (50 iterations)
    async def run_and_receipt() -> None:
        d = _make_debate(n_agents=3, rounds=1)
        r = await d.run()
        ReceiptBuilder.to_json(r.receipt)

    times = await _async_timed_runs(run_and_receipt, 50)
    _print_row("Receipt via debate (50x)", _stats(times))

    # Debates at various scales
    for label, agents, rounds, trickster, iters in [
        ("Debate 3-agent 3-round (20x)", 3, 3, False, 20),
        ("Debate 5-agent 3-round (20x)", 5, 3, False, 20),
        ("Debate 10-agent 5-round (10x)", 10, 5, False, 10),
        ("Debate 5-agent trickster (20x)", 5, 3, True, 20),
    ]:
        times = await _async_timed_runs(
            lambda a=agents, r=rounds, t=trickster: _make_debate(a, r, t).run(),
            iterations=iters,
        )
        _print_row(label, _stats(times))

    # Memory benchmark
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(50):
        await _make_debate(5, 2).run()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = (sum(s.size for s in snap2.statistics("lineno")) -
              sum(s.size for s in snap1.statistics("lineno"))) / (1024 * 1024)

    print()
    print(f"  Memory growth (50 debates): {growth:.2f} MB")
    print()
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(_run_all_benchmarks())
