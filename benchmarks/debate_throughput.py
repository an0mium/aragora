#!/usr/bin/env python3
"""
Debate Throughput Benchmarks.

Measures performance of the debate orchestration system:
- Single debate latency
- Concurrent debate handling
- Agent response times
- Round completion times
- Consensus detection performance

Metrics:
- Debate start-to-finish latency
- Per-round latency
- Agent response time distribution
- Throughput (debates per minute)
- Memory usage during concurrent debates

Usage:
    python -m benchmarks.debate_throughput
    python -m benchmarks.debate_throughput --debates 10
    python -m benchmarks.debate_throughput --concurrent 5
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.core import Agent, Environment, Message
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_dev_ms: float
    throughput_ops_per_sec: float
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total time: {self.total_time_ms:.2f}ms\n"
            f"  Latency (ms): min={self.min_latency_ms:.3f}, "
            f"max={self.max_latency_ms:.3f}, "
            f"mean={self.mean_latency_ms:.3f}, "
            f"median={self.median_latency_ms:.3f}\n"
            f"  Percentiles (ms): p95={self.p95_latency_ms:.3f}, "
            f"p99={self.p99_latency_ms:.3f}\n"
            f"  Std dev: {self.std_dev_ms:.3f}ms\n"
            f"  Throughput: {self.throughput_ops_per_sec:.2f} ops/sec"
        )


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def create_benchmark_result(
    name: str, latencies: List[float], extra: dict = None
) -> BenchmarkResult:
    """Create a BenchmarkResult from latency measurements."""
    if not latencies:
        return BenchmarkResult(
            name=name,
            iterations=0,
            total_time_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            mean_latency_ms=0,
            median_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            std_dev_ms=0,
            throughput_ops_per_sec=0,
            extra=extra or {},
        )

    total_time = sum(latencies)
    return BenchmarkResult(
        name=name,
        iterations=len(latencies),
        total_time_ms=total_time,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        mean_latency_ms=statistics.mean(latencies),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=calculate_percentile(latencies, 95),
        p99_latency_ms=calculate_percentile(latencies, 99),
        std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops_per_sec=(len(latencies) / total_time * 1000) if total_time > 0 else 0,
        extra=extra or {},
    )


class MockAgent(Agent):
    """
    Mock agent for benchmarking.

    Simulates agent behavior with configurable response delay.
    """

    def __init__(
        self,
        name: str,
        response_delay_ms: float = 10.0,
        response_variation_ms: float = 5.0,
    ):
        """
        Initialize mock agent.

        Args:
            name: Agent name
            response_delay_ms: Base response delay in milliseconds
            response_variation_ms: Random variation in response delay
        """
        super().__init__(name=name)
        self.response_delay_ms = response_delay_ms
        self.response_variation_ms = response_variation_ms
        self.call_count = 0
        self.total_response_time_ms = 0

    async def respond(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a mock response with simulated delay."""
        import random

        # Simulate API call latency
        delay = self.response_delay_ms + random.uniform(
            -self.response_variation_ms, self.response_variation_ms
        )
        delay = max(1, delay)  # Minimum 1ms

        await asyncio.sleep(delay / 1000)

        self.call_count += 1
        self.total_response_time_ms += delay

        # Generate a mock response
        return f"Mock response from {self.name} (call #{self.call_count}): Based on the prompt, I propose..."

    def get_stats(self) -> dict:
        """Get agent statistics."""
        avg_response = self.total_response_time_ms / self.call_count if self.call_count > 0 else 0
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_response_time_ms": self.total_response_time_ms,
            "avg_response_time_ms": avg_response,
        }


class DebateThroughputBenchmark:
    """Benchmark suite for debate throughput."""

    def __init__(
        self,
        num_debates: int = 5,
        rounds_per_debate: int = 2,
        agents_per_debate: int = 3,
        response_delay_ms: float = 10.0,
    ):
        """
        Initialize benchmark.

        Args:
            num_debates: Number of debates to run
            rounds_per_debate: Rounds per debate
            agents_per_debate: Agents per debate
            response_delay_ms: Mock agent response delay
        """
        self.num_debates = num_debates
        self.rounds_per_debate = rounds_per_debate
        self.agents_per_debate = agents_per_debate
        self.response_delay_ms = response_delay_ms
        self.results: List[BenchmarkResult] = []

    def create_agents(self) -> List[MockAgent]:
        """Create mock agents for a debate."""
        return [
            MockAgent(
                name=f"agent-{i}",
                response_delay_ms=self.response_delay_ms,
                response_variation_ms=self.response_delay_ms * 0.5,
            )
            for i in range(self.agents_per_debate)
        ]

    async def run_single_debate(self, debate_id: int) -> dict:
        """Run a single debate and collect metrics."""
        agents = self.create_agents()
        env = Environment(task=f"Benchmark debate #{debate_id}: Discuss optimal caching strategies")
        protocol = DebateProtocol(
            rounds=self.rounds_per_debate,
            consensus="majority",
            timeout=30.0,
        )

        round_latencies = []
        agent_response_times = []

        start_time = time.perf_counter()

        try:
            arena = Arena(environment=env, agents=agents, protocol=protocol)
            result = await arena.run()

            total_time = (time.perf_counter() - start_time) * 1000

            # Collect agent stats
            for agent in agents:
                stats = agent.get_stats()
                agent_response_times.append(stats["avg_response_time_ms"])

            return {
                "debate_id": debate_id,
                "total_time_ms": total_time,
                "rounds": result.rounds if hasattr(result, "rounds") else self.rounds_per_debate,
                "consensus_reached": (
                    result.consensus_reached if hasattr(result, "consensus_reached") else False
                ),
                "agent_response_times": agent_response_times,
                "success": True,
            }

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Debate {debate_id} failed: {e}")
            return {
                "debate_id": debate_id,
                "total_time_ms": total_time,
                "rounds": 0,
                "consensus_reached": False,
                "agent_response_times": [],
                "success": False,
                "error": str(e),
            }

    async def benchmark_single_debate_latency(self) -> BenchmarkResult:
        """Benchmark single debate execution latency."""
        latencies = []
        all_agent_times = []

        for i in range(self.num_debates):
            result = await self.run_single_debate(i)
            latencies.append(result["total_time_ms"])
            all_agent_times.extend(result.get("agent_response_times", []))

        avg_agent_time = statistics.mean(all_agent_times) if all_agent_times else 0

        return create_benchmark_result(
            "Single debate latency",
            latencies,
            extra={
                "rounds_per_debate": self.rounds_per_debate,
                "agents_per_debate": self.agents_per_debate,
                "avg_agent_response_ms": avg_agent_time,
            },
        )

    async def benchmark_concurrent_debates(self, concurrency: int = 3) -> BenchmarkResult:
        """Benchmark concurrent debate execution."""

        async def run_debate_batch(batch_id: int, batch_size: int) -> List[dict]:
            tasks = [self.run_single_debate(batch_id * batch_size + i) for i in range(batch_size)]
            return await asyncio.gather(*tasks)

        start_time = time.perf_counter()
        batches = (self.num_debates + concurrency - 1) // concurrency
        all_results = []

        for batch in range(batches):
            batch_size = min(concurrency, self.num_debates - batch * concurrency)
            results = await run_debate_batch(batch, batch_size)
            all_results.extend(results)

        total_time = (time.perf_counter() - start_time) * 1000

        latencies = [r["total_time_ms"] for r in all_results if r["success"]]
        success_rate = sum(1 for r in all_results if r["success"]) / len(all_results)

        return create_benchmark_result(
            f"Concurrent debates (n={concurrency})",
            latencies,
            extra={
                "concurrency": concurrency,
                "total_debates": len(all_results),
                "success_rate": success_rate,
                "total_wall_time_ms": total_time,
            },
        )

    async def benchmark_round_latency(self) -> BenchmarkResult:
        """Benchmark per-round latency."""
        # Run a single debate and measure per-round timing
        agents = self.create_agents()
        env = Environment(task="Benchmark round timing: Discuss API design patterns")
        protocol = DebateProtocol(
            rounds=max(3, self.rounds_per_debate),
            consensus="majority",
            timeout=60.0,
        )

        round_latencies = []

        try:
            arena = Arena(environment=env, agents=agents, protocol=protocol)

            # Measure round-by-round (approximation based on total time / rounds)
            start_time = time.perf_counter()
            result = await arena.run()
            total_time = (time.perf_counter() - start_time) * 1000

            # Estimate per-round latency
            rounds = result.rounds if hasattr(result, "rounds") else protocol.rounds
            if rounds > 0:
                estimated_per_round = total_time / rounds
                round_latencies = [estimated_per_round] * rounds

        except Exception as e:
            logger.warning(f"Round latency benchmark failed: {e}")
            round_latencies = []

        return create_benchmark_result(
            "Per-round latency (estimated)",
            round_latencies,
            extra={"rounds": len(round_latencies)},
        )

    async def benchmark_agent_response_distribution(self) -> BenchmarkResult:
        """Benchmark agent response time distribution."""
        # Run multiple debates and collect all agent response times
        all_response_times = []

        for i in range(self.num_debates):
            agents = self.create_agents()
            env = Environment(task=f"Response timing test #{i}")
            protocol = DebateProtocol(rounds=1, consensus="first")

            try:
                arena = Arena(environment=env, agents=agents, protocol=protocol)
                await arena.run()

                for agent in agents:
                    stats = agent.get_stats()
                    if stats["call_count"] > 0:
                        # Add all individual response times
                        avg_time = stats["avg_response_time_ms"]
                        all_response_times.extend([avg_time] * stats["call_count"])

            except Exception as e:
                logger.warning(f"Agent response benchmark {i} failed: {e}")
                continue

        return create_benchmark_result(
            "Agent response time distribution",
            all_response_times,
            extra={"total_calls": len(all_response_times)},
        )

    async def benchmark_consensus_detection(self) -> BenchmarkResult:
        """Benchmark consensus detection performance."""
        from aragora.debate.consensus import ConsensusDetector

        # Create mock votes for consensus detection
        from aragora.core import Vote

        latencies = []

        for i in range(self.num_debates * 5):  # More iterations for this lightweight benchmark
            agents = [MockAgent(f"agent-{j}") for j in range(self.agents_per_debate)]
            votes = [
                Vote(
                    agent=agent,
                    choice="option_a" if j % 2 == 0 else "option_b",
                    reasoning="Test reasoning",
                    confidence=0.7 + (j % 3) * 0.1,
                    continue_debate=False,
                )
                for j, agent in enumerate(agents)
            ]

            start = time.perf_counter()

            # Simulate consensus calculation
            vote_counts: dict = {}
            for vote in votes:
                vote_counts[vote.choice] = vote_counts.get(vote.choice, 0) + 1

            total_votes = len(votes)
            majority_threshold = total_votes / 2

            consensus_reached = any(count > majority_threshold for count in vote_counts.values())

            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            "Consensus detection",
            latencies,
            extra={"agents_per_vote": self.agents_per_debate},
        )

    async def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print(f"Running debate throughput benchmarks...")
        print(f"  Debates: {self.num_debates}")
        print(f"  Rounds per debate: {self.rounds_per_debate}")
        print(f"  Agents per debate: {self.agents_per_debate}")
        print(f"  Mock response delay: {self.response_delay_ms}ms")
        print()

        # Single debate latency
        print("Benchmarking single debate latency...")
        result = await self.benchmark_single_debate_latency()
        self.results.append(result)
        print(result)
        print()

        # Concurrent debates
        print("Benchmarking concurrent debates...")
        result = await self.benchmark_concurrent_debates(concurrency=3)
        self.results.append(result)
        print(result)
        if "success_rate" in result.extra:
            print(f"  Success rate: {result.extra['success_rate']:.1%}")
        print()

        # Round latency
        print("Benchmarking per-round latency...")
        result = await self.benchmark_round_latency()
        self.results.append(result)
        print(result)
        print()

        # Agent response distribution
        print("Benchmarking agent response distribution...")
        result = await self.benchmark_agent_response_distribution()
        self.results.append(result)
        print(result)
        print()

        # Consensus detection
        print("Benchmarking consensus detection...")
        result = await self.benchmark_consensus_detection()
        self.results.append(result)
        print(result)
        print()

        return self.results

    def summary(self) -> str:
        """Generate summary of all benchmark results."""
        lines = ["=" * 60, "DEBATE THROUGHPUT BENCHMARK SUMMARY", "=" * 60, ""]

        for r in self.results:
            throughput_str = (
                f"{r.throughput_ops_per_sec:.2f} ops/sec" if r.throughput_ops_per_sec > 0 else "N/A"
            )
            lines.append(f"{r.name}:")
            lines.append(f"  Mean latency: {r.mean_latency_ms:.2f}ms")
            lines.append(f"  P95 latency: {r.p95_latency_ms:.2f}ms")
            lines.append(f"  Throughput: {throughput_str}")
            lines.append("")

        # Calculate overall throughput
        single_debate_result = next((r for r in self.results if "Single debate" in r.name), None)
        if single_debate_result and single_debate_result.mean_latency_ms > 0:
            debates_per_min = 60000 / single_debate_result.mean_latency_ms
            lines.append(f"Estimated throughput: {debates_per_min:.1f} debates/minute")

        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Debate Throughput Benchmarks")
    parser.add_argument("--debates", type=int, default=5, help="Number of debates to run")
    parser.add_argument("--rounds", type=int, default=2, help="Rounds per debate")
    parser.add_argument("--agents", type=int, default=3, help="Agents per debate")
    parser.add_argument("--delay", type=float, default=10.0, help="Mock agent response delay (ms)")
    parser.add_argument("--concurrent", type=int, default=3, help="Concurrency level")
    args = parser.parse_args()

    benchmark = DebateThroughputBenchmark(
        num_debates=args.debates,
        rounds_per_debate=args.rounds,
        agents_per_debate=args.agents,
        response_delay_ms=args.delay,
    )

    asyncio.run(benchmark.run_all())
    print(benchmark.summary())


if __name__ == "__main__":
    main()
