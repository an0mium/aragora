#!/usr/bin/env python3
"""Debate engine load-testing benchmark.

Measures throughput and latency of the aragora-debate Arena using MockAgent
and StyledMockAgent, so no API keys are required.

Usage::

    # Default benchmark (3 agents, 2 rounds, 10 concurrent)
    python scripts/benchmark_debate.py

    # Custom parameters
    python scripts/benchmark_debate.py --agents 5 --rounds 3 --concurrent 25

    # Machine-readable output
    python scripts/benchmark_debate.py --json

    # Quick smoke test
    python scripts/benchmark_debate.py --concurrent 5 --rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import json as json_mod
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

from aragora_debate import Arena, DebateConfig, MockAgent
from aragora_debate.styled_mock import StyledMockAgent


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SingleDebateMetrics:
    """Metrics for a single debate run."""

    total_time: float = 0.0
    time_per_round: float = 0.0
    rounds_used: int = 0
    num_agents: int = 0
    num_proposals: int = 0
    num_critiques: int = 0
    num_votes: int = 0
    consensus_reached: bool = False


@dataclass
class ConcurrentMetrics:
    """Metrics for a batch of concurrent debates."""

    num_debates: int = 0
    total_wall_time: float = 0.0
    debates_per_sec: float = 0.0
    avg_time_per_debate: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    median_time: float = 0.0
    p95_time: float = 0.0
    failures: int = 0


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    single_debate: SingleDebateMetrics = field(default_factory=SingleDebateMetrics)
    concurrent_batches: dict[int, ConcurrentMetrics] = field(default_factory=dict)
    large_panel: SingleDebateMetrics = field(default_factory=SingleDebateMetrics)
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_STYLES = ["supportive", "critical", "balanced", "contrarian"]


def _make_agents(n: int) -> list[MockAgent | StyledMockAgent]:
    """Create *n* mock agents with varied styles."""
    agents: list[MockAgent | StyledMockAgent] = []
    for i in range(n):
        style = _STYLES[i % len(_STYLES)]
        agents.append(
            StyledMockAgent(
                name=f"agent-{i}",
                style=style,  # type: ignore[arg-type]
            )
        )
    return agents


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


async def _run_single_debate(
    question: str,
    num_agents: int,
    num_rounds: int,
) -> SingleDebateMetrics:
    """Run one debate and collect per-phase timing."""
    agents = _make_agents(num_agents)
    config = DebateConfig(
        rounds=num_rounds,
        early_stopping=False,  # always run all rounds for consistent measurement
    )
    arena = Arena(
        question=question,
        agents=agents,
        config=config,
    )

    start = time.perf_counter()
    result = await arena.run()
    elapsed = time.perf_counter() - start

    return SingleDebateMetrics(
        total_time=elapsed,
        time_per_round=elapsed / max(result.rounds_used, 1),
        rounds_used=result.rounds_used,
        num_agents=num_agents,
        num_proposals=len(result.proposals),
        num_critiques=len(result.critiques),
        num_votes=len(result.votes),
        consensus_reached=result.consensus_reached,
    )


async def _run_concurrent_debates(
    num_debates: int,
    num_agents: int,
    num_rounds: int,
) -> ConcurrentMetrics:
    """Run *num_debates* debates concurrently and measure throughput."""
    questions = [
        f"Benchmark question #{i}: Should the system adopt approach {i}?"
        for i in range(num_debates)
    ]

    async def _single(q: str) -> float:
        agents = _make_agents(num_agents)
        config = DebateConfig(
            rounds=num_rounds,
            early_stopping=False,
        )
        arena = Arena(question=q, agents=agents, config=config)
        t0 = time.perf_counter()
        await arena.run()
        return time.perf_counter() - t0

    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_single(q) for q in questions],
        return_exceptions=True,
    )
    wall_time = time.perf_counter() - wall_start

    durations: list[float] = []
    failures = 0
    for r in results:
        if isinstance(r, Exception):
            failures += 1
        else:
            durations.append(r)

    if not durations:
        return ConcurrentMetrics(
            num_debates=num_debates,
            total_wall_time=wall_time,
            failures=failures,
        )

    durations.sort()
    p95_idx = max(0, int(len(durations) * 0.95) - 1)

    return ConcurrentMetrics(
        num_debates=num_debates,
        total_wall_time=wall_time,
        debates_per_sec=len(durations) / wall_time if wall_time > 0 else 0.0,
        avg_time_per_debate=statistics.mean(durations),
        min_time=durations[0],
        max_time=durations[-1],
        median_time=statistics.median(durations),
        p95_time=durations[p95_idx],
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    num_agents: int = 3,
    num_rounds: int = 2,
    concurrent_levels: list[int] | None = None,
    large_panel_agents: int = 10,
    large_panel_rounds: int = 3,
) -> BenchmarkResults:
    """Execute the full benchmark suite."""
    if concurrent_levels is None:
        concurrent_levels = [5, 10, 25, 50]

    results = BenchmarkResults(
        config={
            "num_agents": num_agents,
            "num_rounds": num_rounds,
            "concurrent_levels": concurrent_levels,
            "large_panel_agents": large_panel_agents,
            "large_panel_rounds": large_panel_rounds,
        },
    )

    # -- 1. Single debate --
    print(f"  Running single debate ({num_agents} agents, {num_rounds} rounds)...", flush=True)
    results.single_debate = await _run_single_debate(
        question="Should our team adopt a microservices architecture?",
        num_agents=num_agents,
        num_rounds=num_rounds,
    )

    # -- 2. Concurrent debates --
    for level in concurrent_levels:
        print(f"  Running {level} concurrent debates...", flush=True)
        results.concurrent_batches[level] = await _run_concurrent_debates(
            num_debates=level,
            num_agents=num_agents,
            num_rounds=num_rounds,
        )

    # -- 3. Large panel --
    print(
        f"  Running large panel ({large_panel_agents} agents, {large_panel_rounds} rounds)...",
        flush=True,
    )
    results.large_panel = await _run_single_debate(
        question="What is the optimal caching strategy for a multi-region deployment?",
        num_agents=large_panel_agents,
        num_rounds=large_panel_rounds,
    )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_human(results: BenchmarkResults) -> None:
    """Print results in human-readable table format."""
    s = results.single_debate
    lp = results.large_panel

    print()
    print("=" * 60)
    print("  Aragora Debate Engine Benchmark")
    print("=" * 60)

    print(f"\nSingle Debate ({s.num_agents} agents, {s.rounds_used} rounds):")
    print(f"  Total time:          {s.total_time:.4f}s")
    print(f"  Time per round:      {s.time_per_round:.4f}s")
    print(f"  Proposals:           {s.num_proposals}")
    print(f"  Critiques:           {s.num_critiques}")
    print(f"  Votes:               {s.num_votes}")
    print(f"  Consensus reached:   {s.consensus_reached}")

    print("\nConcurrent Debates:")
    # Table header
    print(
        f"  {'Debates':>8}  {'Wall time':>10}  {'Avg/debate':>11}  "
        f"{'Median':>8}  {'P95':>8}  {'Throughput':>12}  {'Fails':>5}"
    )
    print(
        f"  {'-------':>8}  {'--------':>10}  {'---------':>11}  "
        f"{'------':>8}  {'---':>8}  {'----------':>12}  {'-----':>5}"
    )
    for level in sorted(results.concurrent_batches):
        m = results.concurrent_batches[level]
        print(
            f"  {m.num_debates:>8}  "
            f"{m.total_wall_time:>9.4f}s  "
            f"{m.avg_time_per_debate:>10.4f}s  "
            f"{m.median_time:>7.4f}s  "
            f"{m.p95_time:>7.4f}s  "
            f"{m.debates_per_sec:>8.1f} d/s  "
            f"{m.failures:>5}"
        )

    print(f"\nLarge Panel ({lp.num_agents} agents, {lp.rounds_used} rounds):")
    print(f"  Total time:          {lp.total_time:.4f}s")
    print(f"  Time per round:      {lp.time_per_round:.4f}s")
    print(f"  Proposals:           {lp.num_proposals}")
    print(f"  Critiques:           {lp.num_critiques}")
    print(f"  Votes:               {lp.num_votes}")
    print(f"  Consensus reached:   {lp.consensus_reached}")

    print()
    print("=" * 60)
    print()


def _print_json(results: BenchmarkResults) -> None:
    """Print results as machine-readable JSON."""
    out = {
        "config": results.config,
        "single_debate": asdict(results.single_debate),
        "concurrent_batches": {str(k): asdict(v) for k, v in results.concurrent_batches.items()},
        "large_panel": asdict(results.large_panel),
    }
    print(json_mod.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the aragora-debate engine (no API keys required).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents per debate (default: 3)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help="Run a single concurrency level instead of the default [5,10,25,50]",
    )
    parser.add_argument(
        "--large-panel-agents",
        type=int,
        default=10,
        help="Number of agents in the large panel scenario (default: 10)",
    )
    parser.add_argument(
        "--large-panel-rounds",
        type=int,
        default=3,
        help="Number of rounds in the large panel scenario (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.concurrent is not None:
        concurrent_levels = [args.concurrent]
    else:
        concurrent_levels = [5, 10, 25, 50]

    if not args.json:
        print("\nStarting debate engine benchmark...")
        print(
            f"  Agents: {args.agents}  |  Rounds: {args.rounds}  "
            f"|  Concurrency levels: {concurrent_levels}"
        )
        print()

    results = asyncio.run(
        run_benchmark(
            num_agents=args.agents,
            num_rounds=args.rounds,
            concurrent_levels=concurrent_levels,
            large_panel_agents=args.large_panel_agents,
            large_panel_rounds=args.large_panel_rounds,
        )
    )

    if args.json:
        _print_json(results)
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
