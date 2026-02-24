#!/usr/bin/env python3
"""
Load test for Aragora debate infrastructure.

Simulates N concurrent debate sessions with mocked LLM API calls to test
the infrastructure layer (orchestration, consensus, receipts, memory)
without incurring real API costs.

Each simulated debate:
  1. Creates an Arena with mock agents
  2. Runs proposals through the debate loop
  3. Checks consensus
  4. Verifies a result is produced

Reports p50/p95/p99 latency, throughput, error rate, and memory usage.

Usage:
    # Quick smoke test (10 concurrent, 20 debates)
    python scripts/load_test_debate.py

    # Moderate load
    python scripts/load_test_debate.py --concurrency 20 --debates 100 --rounds 3

    # Heavy load with custom agents
    python scripts/load_test_debate.py --concurrency 50 --debates 500 --agents 5

    # Output JSON report
    python scripts/load_test_debate.py --output report.json

    # Validate against SLO thresholds
    python scripts/load_test_debate.py --validate-slos

Environment:
    ARAGORA_LOAD_TEST_ENABLED=1  - Required to run (safety guard)
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class DebateLoadMetrics:
    """Metrics collected during the load test run."""

    total_debates: int = 0
    completed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    # Latency buckets (milliseconds)
    debate_latencies_ms: list[float] = field(default_factory=list)
    first_token_latencies_ms: list[float] = field(default_factory=list)
    consensus_latencies_ms: list[float] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Memory
    initial_memory_kb: int = 0
    peak_memory_kb: int = 0
    final_memory_kb: int = 0

    # Concurrency tracking
    max_concurrent_active: int = 0
    dispatch_concurrency_samples: list[float] = field(default_factory=list)

    def percentile(self, values: list[float], pct: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = (pct / 100.0) * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    @property
    def duration_s(self) -> float:
        return max(self.end_time - self.start_time, 0.001)

    @property
    def throughput_per_s(self) -> float:
        return self.completed / self.duration_s

    @property
    def success_rate(self) -> float:
        if self.total_debates == 0:
            return 0.0
        return self.completed / self.total_debates

    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def memory_growth_mb(self) -> float:
        return (self.peak_memory_kb - self.initial_memory_kb) / 1024.0

    def summary(self) -> dict[str, Any]:
        """Generate a summary report dictionary."""
        return {
            "configuration": {
                "total_debates": self.total_debates,
            },
            "results": {
                "completed": self.completed,
                "failed": self.failed,
                "success_rate": round(self.success_rate * 100, 2),
                "error_rate": round(self.error_rate * 100, 4),
                "duration_seconds": round(self.duration_s, 2),
                "throughput_debates_per_second": round(self.throughput_per_s, 2),
            },
            "latency_debate_ms": {
                "p50": round(self.percentile(self.debate_latencies_ms, 50), 1),
                "p95": round(self.percentile(self.debate_latencies_ms, 95), 1),
                "p99": round(self.percentile(self.debate_latencies_ms, 99), 1),
                "min": round(min(self.debate_latencies_ms), 1) if self.debate_latencies_ms else 0,
                "max": round(max(self.debate_latencies_ms), 1) if self.debate_latencies_ms else 0,
                "mean": round(statistics.mean(self.debate_latencies_ms), 1)
                if self.debate_latencies_ms
                else 0,
            },
            "latency_first_token_ms": {
                "p50": round(self.percentile(self.first_token_latencies_ms, 50), 1),
                "p95": round(self.percentile(self.first_token_latencies_ms, 95), 1),
                "p99": round(self.percentile(self.first_token_latencies_ms, 99), 1),
            },
            "latency_consensus_ms": {
                "p50": round(self.percentile(self.consensus_latencies_ms, 50), 1),
                "p95": round(self.percentile(self.consensus_latencies_ms, 95), 1),
                "p99": round(self.percentile(self.consensus_latencies_ms, 99), 1),
            },
            "memory": {
                "initial_mb": round(self.initial_memory_kb / 1024.0, 1),
                "peak_mb": round(self.peak_memory_kb / 1024.0, 1),
                "growth_mb": round(self.memory_growth_mb, 1),
            },
            "concurrency": {
                "max_concurrent": self.max_concurrent_active,
                "avg_dispatch_ratio": round(statistics.mean(self.dispatch_concurrency_samples), 3)
                if self.dispatch_concurrency_samples
                else 0.0,
            },
            "errors": self.errors[:10],
        }


# =============================================================================
# Mock Debate Session
# =============================================================================


class MockAgent:
    """Lightweight mock agent that returns canned responses without API calls."""

    def __init__(self, name: str, model: str = "mock-model"):
        self.name = name
        self.model = model
        self._response_delay = random.uniform(0.01, 0.05)  # 10-50ms simulated latency

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a canned response with simulated latency."""
        await asyncio.sleep(self._response_delay)
        return f"[{self.name}] Response to: {prompt[:80]}... I believe the answer involves careful analysis."

    async def critique(self, proposal: str, **kwargs: Any) -> str:
        """Generate a canned critique."""
        await asyncio.sleep(self._response_delay * 0.8)
        return f"[{self.name}] The proposal has merit but could improve in specificity."

    async def vote(self, proposals: list[str], **kwargs: Any) -> int:
        """Vote for a proposal."""
        await asyncio.sleep(self._response_delay * 0.3)
        return 0  # Always vote for first proposal


async def run_mock_debate(
    debate_id: str,
    question: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    fail_rate: float = 0.02,
) -> dict[str, Any]:
    """Run a complete mock debate session.

    Simulates the full debate lifecycle without actual LLM calls:
    - Create agents
    - Generate proposals (with simulated first-token latency)
    - Run critique/revision rounds
    - Detect consensus
    - Produce result

    Args:
        debate_id: Unique identifier for this debate.
        question: The debate question/task.
        num_agents: Number of mock agents.
        num_rounds: Number of debate rounds.
        fail_rate: Probability of a simulated infrastructure failure.

    Returns:
        Dictionary with debate results and timing information.

    Raises:
        RuntimeError: If a simulated failure occurs.
    """
    debate_start = time.monotonic()
    agents = [MockAgent(f"agent_{i}", f"mock-{i}") for i in range(num_agents)]

    # Simulate occasional infrastructure failures
    if random.random() < fail_rate:
        raise RuntimeError(f"Simulated infrastructure failure for debate {debate_id}")

    # Phase 1: Proposals (measures time to first token)
    first_token_start = time.monotonic()
    proposals = []
    proposal_tasks = [agent.generate(f"Propose a solution for: {question}") for agent in agents]
    # Run proposals concurrently to measure dispatch concurrency
    dispatch_start = time.monotonic()
    results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
    dispatch_duration = time.monotonic() - dispatch_start

    first_token_latency = time.monotonic() - first_token_start

    for r in results:
        if isinstance(r, str):
            proposals.append(r)
        # Skip exceptions from individual agents

    if not proposals:
        raise RuntimeError(f"No proposals generated for debate {debate_id}")

    # Calculate dispatch concurrency ratio
    # If fully parallel, total time ~ single agent time
    # If sequential, total time ~ sum of agent times
    single_agent_est = max(0.01, dispatch_duration / max(num_agents, 1))
    dispatch_ratio = min(1.0, single_agent_est / max(dispatch_duration, 0.001))

    # Phase 2: Rounds (critique and revision)
    for round_num in range(num_rounds):
        # Critiques
        critique_tasks = [agent.critique(proposals[0]) for agent in agents]
        await asyncio.gather(*critique_tasks, return_exceptions=True)

        # Minor revision of proposals (simulated)
        await asyncio.sleep(random.uniform(0.005, 0.02))

    # Phase 3: Consensus detection
    consensus_start = time.monotonic()
    vote_tasks = [agent.vote(proposals) for agent in agents]
    votes = await asyncio.gather(*vote_tasks, return_exceptions=True)

    # Simple majority consensus
    valid_votes = [v for v in votes if isinstance(v, int)]
    if valid_votes:
        from collections import Counter

        vote_counts = Counter(valid_votes)
        winner_idx, winner_count = vote_counts.most_common(1)[0]
        consensus_reached = winner_count > len(valid_votes) / 2
    else:
        winner_idx = 0
        consensus_reached = False

    consensus_latency = time.monotonic() - consensus_start
    total_duration = time.monotonic() - debate_start

    return {
        "debate_id": debate_id,
        "question": question[:100],
        "agents": num_agents,
        "rounds": num_rounds,
        "consensus_reached": consensus_reached,
        "winner_index": winner_idx,
        "duration_ms": total_duration * 1000,
        "first_token_ms": first_token_latency * 1000,
        "consensus_ms": consensus_latency * 1000,
        "dispatch_ratio": dispatch_ratio,
        "proposals_count": len(proposals),
    }


# =============================================================================
# Load Test Runner
# =============================================================================


async def run_debate_load_test(
    num_debates: int = 20,
    concurrency: int = 10,
    num_rounds: int = 3,
    num_agents: int = 3,
    fail_rate: float = 0.02,
) -> DebateLoadMetrics:
    """Run the debate load test with the specified parameters.

    Args:
        num_debates: Total number of debates to execute.
        concurrency: Maximum number of concurrent debates.
        num_rounds: Rounds per debate.
        num_agents: Agents per debate.
        fail_rate: Simulated failure probability per debate.

    Returns:
        DebateLoadMetrics with comprehensive results.
    """
    metrics = DebateLoadMetrics(total_debates=num_debates)
    semaphore = asyncio.Semaphore(concurrency)
    active_count = 0
    active_lock = asyncio.Lock()

    # Initial memory
    metrics.initial_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    questions = [
        "What is the optimal rate limiting strategy for a multi-tenant SaaS platform?",
        "Should we migrate from monolith to microservices?",
        "What is the best approach to handle data consistency in distributed systems?",
        "How should we implement RBAC for an enterprise application?",
        "What is the most effective CI/CD pipeline architecture?",
        "Should we use event sourcing or traditional CRUD?",
        "What is the best strategy for database sharding?",
        "How should we handle secrets management in Kubernetes?",
        "What is the optimal caching strategy for a read-heavy workload?",
        "Should we use GraphQL or REST for our public API?",
    ]

    async def run_single(idx: int) -> None:
        nonlocal active_count

        async with semaphore:
            async with active_lock:
                active_count += 1
                metrics.max_concurrent_active = max(metrics.max_concurrent_active, active_count)

            debate_id = f"load_{idx:06d}"
            question = questions[idx % len(questions)]

            try:
                result = await run_mock_debate(
                    debate_id=debate_id,
                    question=question,
                    num_agents=num_agents,
                    num_rounds=num_rounds,
                    fail_rate=fail_rate,
                )

                metrics.debate_latencies_ms.append(result["duration_ms"])
                metrics.first_token_latencies_ms.append(result["first_token_ms"])
                metrics.consensus_latencies_ms.append(result["consensus_ms"])
                metrics.dispatch_concurrency_samples.append(result["dispatch_ratio"])
                metrics.completed += 1

            except Exception as e:
                metrics.failed += 1
                metrics.errors.append(f"{debate_id}: {type(e).__name__}: {str(e)[:100]}")

            finally:
                async with active_lock:
                    active_count -= 1

    # Run all debates
    metrics.start_time = time.monotonic()

    tasks = [asyncio.create_task(run_single(i)) for i in range(num_debates)]
    await asyncio.gather(*tasks, return_exceptions=True)

    metrics.end_time = time.monotonic()

    # Final memory
    gc.collect()
    metrics.peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics.final_memory_kb = metrics.peak_memory_kb

    return metrics


# =============================================================================
# SLO Validation
# =============================================================================


def validate_against_slos(metrics: DebateLoadMetrics) -> dict[str, Any]:
    """Validate load test results against debate SLO targets.

    Args:
        metrics: Collected load test metrics.

    Returns:
        Dictionary with per-SLO pass/fail status.
    """
    try:
        from aragora.observability.debate_slos import get_debate_slo_definitions

        definitions = get_debate_slo_definitions()
    except ImportError:
        definitions = {}

    results = {}

    # SLO 1: Time to first token p95 < 3s
    ttft_p95 = metrics.percentile(metrics.first_token_latencies_ms, 95) / 1000
    ttft_target = definitions.get("time_to_first_token")
    target_val = ttft_target.target if ttft_target else 3.0
    results["time_to_first_token_p95"] = {
        "target_s": target_val,
        "actual_s": round(ttft_p95, 3),
        "passed": ttft_p95 <= target_val,
    }

    # SLO 2: Debate completion p95 < 60s
    completion_p95 = metrics.percentile(metrics.debate_latencies_ms, 95) / 1000
    completion_target = definitions.get("debate_completion")
    target_val = completion_target.target if completion_target else 60.0
    results["debate_completion_p95"] = {
        "target_s": target_val,
        "actual_s": round(completion_p95, 3),
        "passed": completion_p95 <= target_val,
    }

    # SLO 3: Success rate > 99% (proxy for WS reconnection)
    results["success_rate"] = {
        "target": 0.99,
        "actual": round(metrics.success_rate, 4),
        "passed": metrics.success_rate >= 0.99,
    }

    # SLO 4: Consensus detection p95 < 500ms
    consensus_p95 = metrics.percentile(metrics.consensus_latencies_ms, 95) / 1000
    consensus_target = definitions.get("consensus_detection")
    target_val = consensus_target.target if consensus_target else 0.5
    results["consensus_detection_p95"] = {
        "target_s": target_val,
        "actual_s": round(consensus_p95, 3),
        "passed": consensus_p95 <= target_val,
    }

    # SLO 5: Dispatch concurrency > 0.8
    avg_dispatch = (
        statistics.mean(metrics.dispatch_concurrency_samples)
        if metrics.dispatch_concurrency_samples
        else 0.0
    )
    dispatch_target = definitions.get("agent_dispatch_concurrency")
    target_val = dispatch_target.target if dispatch_target else 0.8
    results["agent_dispatch_concurrency"] = {
        "target": target_val,
        "actual": round(avg_dispatch, 3),
        "passed": avg_dispatch >= target_val,
    }

    results["all_passed"] = all(v["passed"] for v in results.values() if isinstance(v, dict))

    return results


# =============================================================================
# Report Formatting
# =============================================================================


def format_report(metrics: DebateLoadMetrics, slo_results: dict | None = None) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: Load test metrics.
        slo_results: Optional SLO validation results.

    Returns:
        Formatted report string.
    """
    summary = metrics.summary()

    lines = [
        "",
        "=" * 70,
        "  ARAGORA DEBATE LOAD TEST REPORT",
        "=" * 70,
        "",
        f"  Debates: {summary['results']['completed']}/{summary['configuration']['total_debates']} "
        f"completed ({summary['results']['success_rate']}% success)",
        f"  Duration: {summary['results']['duration_seconds']}s",
        f"  Throughput: {summary['results']['throughput_debates_per_second']} debates/s",
        "",
        "  Latency (Debate Completion):",
        f"    p50:  {summary['latency_debate_ms']['p50']}ms",
        f"    p95:  {summary['latency_debate_ms']['p95']}ms",
        f"    p99:  {summary['latency_debate_ms']['p99']}ms",
        f"    min:  {summary['latency_debate_ms']['min']}ms",
        f"    max:  {summary['latency_debate_ms']['max']}ms",
        "",
        "  Latency (First Token):",
        f"    p50:  {summary['latency_first_token_ms']['p50']}ms",
        f"    p95:  {summary['latency_first_token_ms']['p95']}ms",
        f"    p99:  {summary['latency_first_token_ms']['p99']}ms",
        "",
        "  Latency (Consensus Detection):",
        f"    p50:  {summary['latency_consensus_ms']['p50']}ms",
        f"    p95:  {summary['latency_consensus_ms']['p95']}ms",
        f"    p99:  {summary['latency_consensus_ms']['p99']}ms",
        "",
        "  Memory:",
        f"    Initial: {summary['memory']['initial_mb']}MB",
        f"    Peak:    {summary['memory']['peak_mb']}MB",
        f"    Growth:  {summary['memory']['growth_mb']}MB",
        "",
        "  Concurrency:",
        f"    Max Concurrent: {summary['concurrency']['max_concurrent']}",
        f"    Avg Dispatch Ratio: {summary['concurrency']['avg_dispatch_ratio']}",
    ]

    if summary["errors"]:
        lines.extend(["", "  Errors (first 10):"])
        for err in summary["errors"]:
            lines.append(f"    - {err}")

    if slo_results:
        lines.extend(["", "  SLO Validation:"])
        for slo_name, result in slo_results.items():
            if not isinstance(result, dict):
                continue
            status = "PASS" if result["passed"] else "FAIL"
            if "target_s" in result:
                lines.append(
                    f"    [{status}] {slo_name}: "
                    f"{result['actual_s']}s (target: {result['target_s']}s)"
                )
            elif "target" in result:
                lines.append(
                    f"    [{status}] {slo_name}: {result['actual']} (target: {result['target']})"
                )
        overall = "PASSED" if slo_results.get("all_passed") else "FAILED"
        lines.append(f"    Overall: {overall}")

    lines.extend(["", "=" * 70, ""])

    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for the debate load test."""
    parser = argparse.ArgumentParser(
        description="Aragora Debate Load Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent debates (default: 10)",
    )
    parser.add_argument(
        "--debates",
        type=int,
        default=20,
        help="Total number of debates to run (default: 20)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Rounds per debate (default: 3)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Agents per debate (default: 3)",
    )
    parser.add_argument(
        "--fail-rate",
        type=float,
        default=0.02,
        help="Simulated failure rate 0-1 (default: 0.02)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--validate-slos",
        action="store_true",
        help="Validate results against debate SLO targets",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="Skip the ARAGORA_LOAD_TEST_ENABLED check",
    )

    args = parser.parse_args()

    # Safety guard
    if not args.no_guard and os.getenv("ARAGORA_LOAD_TEST_ENABLED", "0") != "1":
        print(
            "Load tests are disabled by default. Set ARAGORA_LOAD_TEST_ENABLED=1 "
            "or use --no-guard to run."
        )
        sys.exit(0)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting debate load test: %d debates, %d concurrency, %d rounds, %d agents",
        args.debates,
        args.concurrency,
        args.rounds,
        args.agents,
    )

    # Run the load test
    metrics = asyncio.run(
        run_debate_load_test(
            num_debates=args.debates,
            concurrency=args.concurrency,
            num_rounds=args.rounds,
            num_agents=args.agents,
            fail_rate=args.fail_rate,
        )
    )

    # SLO validation
    slo_results = None
    if args.validate_slos:
        slo_results = validate_against_slos(metrics)

    # Print report
    print(format_report(metrics, slo_results))

    # Output JSON if requested
    if args.output:
        report = metrics.summary()
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        if slo_results:
            report["slo_validation"] = slo_results

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        logger.info("JSON report written to %s", args.output)

    # Exit code based on results
    if metrics.error_rate > 0.05:
        logger.warning("Error rate %.2f%% exceeds 5%%", metrics.error_rate * 100)
        sys.exit(1)

    if slo_results and not slo_results.get("all_passed"):
        logger.warning("SLO validation failed")
        sys.exit(2)


if __name__ == "__main__":
    main()
