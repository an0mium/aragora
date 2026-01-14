"""
Aragora bench command - Benchmark agents.

Run performance benchmarks on agents to measure response times,
token usage, and quality metrics.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import cast


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    agent: str
    task: str
    iterations: int
    response_times: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    errors: int = 0
    success_rate: float = 0.0

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p50_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def p95_response_time(self) -> float:
        if len(self.response_times) < 2:
            return self.avg_response_time
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def avg_tokens(self) -> float:
        if not self.token_counts:
            return 0.0
        return statistics.mean(self.token_counts)


BENCHMARK_TASKS = [
    "Explain the concept of recursion in one sentence.",
    "What is the time complexity of binary search?",
    "Name three design patterns and their use cases.",
    "What is the difference between a stack and a queue?",
    "Explain what a hash table is and when to use it.",
]


async def benchmark_agent(
    agent_type: str,
    task: str,
    iterations: int = 5,
    timeout: float = 30.0,
) -> BenchmarkResult:
    """Run benchmark on a single agent."""
    from aragora.agents.base import AgentType, create_agent

    result = BenchmarkResult(
        agent=agent_type,
        task=task[:50],
        iterations=iterations,
    )

    try:
        agent = create_agent(
            model_type=cast(AgentType, agent_type),
            name=f"bench_{agent_type}",
            role="proposer",
        )
    except Exception as e:
        print(f"  Error creating agent {agent_type}: {e}")
        result.errors = iterations
        return result

    for i in range(iterations):
        try:
            start = time.time()

            # Run with timeout
            await asyncio.wait_for(
                agent.generate(task),
                timeout=timeout,
            )

            elapsed = time.time() - start
            result.response_times.append(elapsed)

            # Try to get token count
            if hasattr(agent, "get_token_usage"):
                usage = agent.get_token_usage()
                if usage:
                    result.token_counts.append(usage.get("total_tokens", 0))

        except asyncio.TimeoutError:
            result.errors += 1
        except Exception:
            result.errors += 1

    result.success_rate = (iterations - result.errors) / iterations
    return result


def print_result(result: BenchmarkResult) -> None:
    """Print a benchmark result."""
    print(f"\n  Agent: {result.agent}")
    print(f"  Task: {result.task}...")
    print(f"  Iterations: {result.iterations}")
    print(f"  Success Rate: {result.success_rate * 100:.1f}%")

    if result.response_times:
        print("  Response Time:")
        print(f"    Avg: {result.avg_response_time:.2f}s")
        print(f"    P50: {result.p50_response_time:.2f}s")
        print(f"    P95: {result.p95_response_time:.2f}s")

    if result.token_counts:
        print(f"  Avg Tokens: {result.avg_tokens:.0f}")

    if result.errors > 0:
        print(f"  Errors: {result.errors}")


def cmd_bench(args) -> None:
    """Handle 'bench' command."""
    agents_str = getattr(args, "agents", "anthropic-api,openai-api")
    iterations = getattr(args, "iterations", 3)
    task = getattr(args, "task", None)
    quick = getattr(args, "quick", False)

    if quick:
        iterations = 1

    agents = [a.strip() for a in agents_str.split(",")]

    # Use custom task or default
    tasks = [task] if task else BENCHMARK_TASKS[:2]

    print("\nAragora Agent Benchmark")
    print("=" * 60)
    print(f"Agents: {', '.join(agents)}")
    print(f"Iterations: {iterations}")
    print(f"Tasks: {len(tasks)}")
    print("=" * 60)

    all_results: list[BenchmarkResult] = []

    async def run_benchmarks():
        for agent_type in agents:
            print(f"\nBenchmarking {agent_type}...")

            for task_text in tasks:
                result = await benchmark_agent(
                    agent_type=agent_type,
                    task=task_text,
                    iterations=iterations,
                )
                all_results.append(result)
                print_result(result)

    try:
        asyncio.run(run_benchmarks())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted.")
        return

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Group by agent
    agent_results: dict[str, list[BenchmarkResult]] = {}
    for r in all_results:
        if r.agent not in agent_results:
            agent_results[r.agent] = []
        agent_results[r.agent].append(r)

    for agent, results in agent_results.items():
        all_times = []
        all_tokens = []
        total_errors = 0
        total_iters = 0

        for r in results:
            all_times.extend(r.response_times)
            all_tokens.extend(r.token_counts)
            total_errors += r.errors
            total_iters += r.iterations

        print(f"\n{agent}:")
        if all_times:
            print(f"  Avg Response: {statistics.mean(all_times):.2f}s")
        if all_tokens:
            print(f"  Avg Tokens: {statistics.mean(all_tokens):.0f}")
        print(f"  Success Rate: {(total_iters - total_errors) / total_iters * 100:.1f}%")

    print()
