#!/usr/bin/env python3
"""
Streaming SLO baseline load test for Aragora.

Simulates concurrent debate sessions with streaming metrics collection
and validates results against four streaming SLOs:

  1. First byte latency: p95 < 500ms
  2. Message throughput: >= 10 messages/sec/debate
  3. Reconnection success rate: >= 99%
  4. End-to-end debate completion: p99 < 30s

Uses mock agents to avoid real API costs while exercising the full
debate lifecycle (proposals, critiques, consensus, result streaming).

Usage:
    # Quick baseline (default: 60s, 50 concurrent debates)
    python scripts/load_test_baseline.py

    # Custom duration and concurrency
    python scripts/load_test_baseline.py --duration 120 --concurrency 100

    # Output JSON report to file
    python scripts/load_test_baseline.py --output baseline_report.json

    # Strict mode: exit non-zero on any SLO failure
    python scripts/load_test_baseline.py --strict

    # Verbose logging
    python scripts/load_test_baseline.py -v

Environment Variables:
    SLO_FIRST_BYTE_P95_MS: Override first byte p95 target (default: 500)
    SLO_MSG_THROUGHPUT_MIN: Override min messages/sec/debate (default: 10)
    SLO_RECONNECT_RATE_MIN: Override reconnect success rate (default: 0.99)
    SLO_COMPLETION_P99_S: Override debate completion p99 target (default: 30)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Add project root to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# SLO Definitions
# =============================================================================


@dataclass(frozen=True)
class StreamingSLO:
    """Definition of a single streaming SLO target."""

    slo_id: str
    name: str
    target: float
    unit: str
    comparison: str  # "lte" = value must be <= target, "gte" = value must be >= target
    description: str


def get_streaming_slo_targets() -> dict[str, StreamingSLO]:
    """Get the four streaming SLO targets, with environment overrides.

    Returns:
        Dictionary mapping slo_id to StreamingSLO.
    """
    return {
        "first_byte_latency": StreamingSLO(
            slo_id="first_byte_latency",
            name="First Byte Latency p95",
            target=float(os.getenv("SLO_FIRST_BYTE_P95_MS", "500")),
            unit="ms",
            comparison="lte",
            description="p95 latency from debate start to first streamed byte",
        ),
        "message_throughput": StreamingSLO(
            slo_id="message_throughput",
            name="Message Throughput",
            target=float(os.getenv("SLO_MSG_THROUGHPUT_MIN", "10")),
            unit="messages/sec/debate",
            comparison="gte",
            description="Minimum sustained message throughput per debate",
        ),
        "reconnection_success_rate": StreamingSLO(
            slo_id="reconnection_success_rate",
            name="Reconnection Success Rate",
            target=float(os.getenv("SLO_RECONNECT_RATE_MIN", "0.99")),
            unit="ratio",
            comparison="gte",
            description="Percentage of simulated WebSocket reconnections that succeed",
        ),
        "debate_completion": StreamingSLO(
            slo_id="debate_completion",
            name="End-to-End Debate Completion p99",
            target=float(os.getenv("SLO_COMPLETION_P99_S", "30")),
            unit="seconds",
            comparison="lte",
            description="p99 end-to-end debate completion time",
        ),
    }


# =============================================================================
# Percentile Calculation
# =============================================================================


def percentile(values: list[float], pct: float) -> float:
    """Calculate a percentile using linear interpolation.

    Args:
        values: List of numeric values.
        pct: Percentile to compute (0-100).

    Returns:
        The percentile value, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


# =============================================================================
# SLO Evaluation
# =============================================================================


@dataclass
class SLOCheckResult:
    """Result of checking a single SLO against measured data."""

    slo_id: str
    name: str
    target: float
    actual: float
    unit: str
    passed: bool
    description: str
    margin: float = 0.0  # positive = headroom, negative = overshoot

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "slo_id": self.slo_id,
            "name": self.name,
            "target": self.target,
            "actual": round(self.actual, 4),
            "unit": self.unit,
            "passed": self.passed,
            "margin": round(self.margin, 4),
            "description": self.description,
        }


def evaluate_slo(slo: StreamingSLO, actual: float) -> SLOCheckResult:
    """Evaluate a measured value against an SLO target.

    Args:
        slo: The SLO definition.
        actual: The measured value.

    Returns:
        SLOCheckResult with pass/fail status.
    """
    if slo.comparison == "lte":
        passed = actual <= slo.target
        margin = slo.target - actual
    else:  # gte
        passed = actual >= slo.target
        margin = actual - slo.target

    return SLOCheckResult(
        slo_id=slo.slo_id,
        name=slo.name,
        target=slo.target,
        actual=actual,
        unit=slo.unit,
        passed=passed,
        description=slo.description,
        margin=margin,
    )


# =============================================================================
# Mock Debate Simulation
# =============================================================================


@dataclass
class DebateMetrics:
    """Metrics collected from a single simulated debate."""

    debate_id: str
    first_byte_ms: float = 0.0
    total_messages: int = 0
    duration_s: float = 0.0
    messages_per_sec: float = 0.0
    reconnect_attempts: int = 0
    reconnect_successes: int = 0
    completed: bool = False
    error: str | None = None


class MockStreamingAgent:
    """Lightweight mock agent that simulates streaming message delivery."""

    def __init__(self, name: str, latency_range: tuple[float, float] = (0.005, 0.03)):
        self.name = name
        self._latency_range = latency_range

    async def stream_proposal(self, prompt: str, num_tokens: int = 20) -> list[float]:
        """Simulate streaming a proposal, returning per-token timestamps.

        Args:
            prompt: The debate prompt (unused in mock).
            num_tokens: Number of tokens to stream.

        Returns:
            List of monotonic timestamps for each token.
        """
        timestamps: list[float] = []
        for _ in range(num_tokens):
            await asyncio.sleep(random.uniform(*self._latency_range))
            timestamps.append(time.monotonic())
        return timestamps

    async def critique(self, proposal: str) -> str:
        """Simulate a critique response."""
        await asyncio.sleep(random.uniform(0.01, 0.04))
        return f"[{self.name}] Critique: The proposal could improve in specificity."

    async def vote(self, proposals: list[str]) -> int:
        """Simulate a vote."""
        await asyncio.sleep(random.uniform(0.003, 0.01))
        return random.randint(0, max(0, len(proposals) - 1))


async def simulate_reconnection(fail_probability: float = 0.005) -> bool:
    """Simulate a WebSocket reconnection attempt.

    Args:
        fail_probability: Probability of reconnection failure.

    Returns:
        True if reconnection succeeded, False otherwise.
    """
    await asyncio.sleep(random.uniform(0.001, 0.05))
    return random.random() > fail_probability


async def run_single_debate(
    debate_id: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    tokens_per_proposal: int = 20,
    reconnect_probability: float = 0.1,
    fail_rate: float = 0.005,
) -> DebateMetrics:
    """Run a single simulated debate session with streaming metrics.

    Simulates the full debate lifecycle:
      1. Agents stream proposals (measures first-byte latency)
      2. Critique rounds (generates message throughput)
      3. Voting and consensus
      4. Occasional reconnection attempts

    Args:
        debate_id: Unique identifier for this debate.
        num_agents: Number of mock agents.
        num_rounds: Number of debate rounds.
        tokens_per_proposal: Tokens per proposal stream.
        reconnect_probability: Probability of triggering a reconnect per round.
        fail_rate: Probability of debate infrastructure failure.

    Returns:
        DebateMetrics with timing and throughput data.
    """
    metrics = DebateMetrics(debate_id=debate_id)
    agents = [
        MockStreamingAgent(f"agent_{i}", latency_range=(0.005, 0.03))
        for i in range(num_agents)
    ]

    # Simulate rare infrastructure failure
    if random.random() < fail_rate:
        metrics.error = f"Simulated infrastructure failure for {debate_id}"
        return metrics

    debate_start = time.monotonic()
    total_messages = 0

    try:
        # Phase 1: Proposal streaming (measures first-byte latency)
        proposal_tasks = [
            agent.stream_proposal(f"Debate question for {debate_id}", tokens_per_proposal)
            for agent in agents
        ]
        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        # First byte = earliest timestamp across all agent streams
        earliest_token = float("inf")
        for result in proposal_results:
            if isinstance(result, list) and result:
                earliest_token = min(earliest_token, result[0])
                total_messages += len(result)
            elif isinstance(result, BaseException):
                metrics.error = f"Proposal failed: {type(result).__name__}"
                return metrics

        if earliest_token == float("inf"):
            metrics.error = "No proposal tokens generated"
            return metrics

        metrics.first_byte_ms = (earliest_token - debate_start) * 1000.0

        # Phase 2: Debate rounds (critiques, revisions, messages)
        for round_num in range(num_rounds):
            # Critiques
            critique_tasks = [
                agent.critique(f"Proposal from round {round_num}")
                for agent in agents
            ]
            critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
            total_messages += sum(1 for c in critiques if isinstance(c, str))

            # Simulate occasional reconnection
            if random.random() < reconnect_probability:
                metrics.reconnect_attempts += 1
                success = await simulate_reconnection()
                if success:
                    metrics.reconnect_successes += 1

            # Brief revision delay
            await asyncio.sleep(random.uniform(0.005, 0.015))
            total_messages += 1  # Revision message

        # Phase 3: Voting and consensus
        vote_tasks = [agent.vote(["proposal_a", "proposal_b"]) for agent in agents]
        votes = await asyncio.gather(*vote_tasks, return_exceptions=True)
        total_messages += sum(1 for v in votes if isinstance(v, int))

        # Final result message
        total_messages += 1

        debate_end = time.monotonic()
        metrics.duration_s = debate_end - debate_start
        metrics.total_messages = total_messages
        metrics.messages_per_sec = (
            total_messages / metrics.duration_s if metrics.duration_s > 0 else 0.0
        )
        metrics.completed = True

    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)[:100]}"
        metrics.duration_s = time.monotonic() - debate_start

    return metrics


# =============================================================================
# Load Test Runner
# =============================================================================


@dataclass
class BaselineResult:
    """Aggregated results from the baseline load test."""

    # Configuration
    concurrency: int = 0
    duration_seconds: float = 0.0
    total_debates: int = 0

    # Per-debate metrics
    debate_metrics: list[DebateMetrics] = field(default_factory=list)

    # Timing
    actual_duration_s: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    # SLO check results
    slo_results: dict[str, SLOCheckResult] = field(default_factory=dict)
    all_slos_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        completed = [m for m in self.debate_metrics if m.completed]
        failed = [m for m in self.debate_metrics if not m.completed]

        first_byte_latencies = [m.first_byte_ms for m in completed]
        throughputs = [m.messages_per_sec for m in completed]
        durations = [m.duration_s for m in completed]

        total_reconnect_attempts = sum(m.reconnect_attempts for m in self.debate_metrics)
        total_reconnect_successes = sum(m.reconnect_successes for m in self.debate_metrics)

        return {
            "configuration": {
                "concurrency": self.concurrency,
                "target_duration_seconds": self.duration_seconds,
                "total_debates": self.total_debates,
            },
            "timing": {
                "actual_duration_seconds": round(self.actual_duration_s, 2),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "results": {
                "completed_debates": len(completed),
                "failed_debates": len(failed),
                "success_rate": round(
                    len(completed) / max(len(self.debate_metrics), 1), 4
                ),
            },
            "metrics": {
                "first_byte_latency_ms": {
                    "p50": round(percentile(first_byte_latencies, 50), 2),
                    "p95": round(percentile(first_byte_latencies, 95), 2),
                    "p99": round(percentile(first_byte_latencies, 99), 2),
                    "min": round(min(first_byte_latencies), 2) if first_byte_latencies else 0,
                    "max": round(max(first_byte_latencies), 2) if first_byte_latencies else 0,
                },
                "message_throughput_per_sec": {
                    "mean": round(statistics.mean(throughputs), 2) if throughputs else 0,
                    "min": round(min(throughputs), 2) if throughputs else 0,
                    "p5": round(percentile(throughputs, 5), 2),
                },
                "reconnection": {
                    "total_attempts": total_reconnect_attempts,
                    "total_successes": total_reconnect_successes,
                    "success_rate": round(
                        1.0 if total_reconnect_attempts == 0
                        else total_reconnect_successes / total_reconnect_attempts,
                        4,
                    ),
                },
                "debate_completion_s": {
                    "p50": round(percentile(durations, 50), 3),
                    "p95": round(percentile(durations, 95), 3),
                    "p99": round(percentile(durations, 99), 3),
                    "min": round(min(durations), 3) if durations else 0,
                    "max": round(max(durations), 3) if durations else 0,
                },
            },
            "slo_validation": {
                slo_id: result.to_dict()
                for slo_id, result in self.slo_results.items()
            },
            "all_slos_passed": self.all_slos_passed,
            "errors": [m.error for m in failed if m.error][:20],
        }


async def run_baseline(
    duration_seconds: float = 60.0,
    concurrency: int = 50,
    num_agents: int = 3,
    num_rounds: int = 3,
) -> BaselineResult:
    """Run the streaming SLO baseline load test.

    Launches concurrent debate simulations for the specified duration
    and collects metrics for SLO validation.

    Args:
        duration_seconds: How long to keep launching new debates.
        concurrency: Maximum concurrent debates.
        num_agents: Agents per debate.
        num_rounds: Rounds per debate.

    Returns:
        BaselineResult with aggregated metrics and SLO results.
    """
    result = BaselineResult(
        concurrency=concurrency,
        duration_seconds=duration_seconds,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    semaphore = asyncio.Semaphore(concurrency)
    debate_counter = 0
    all_metrics: list[DebateMetrics] = []
    metrics_lock = asyncio.Lock()

    test_start = time.monotonic()
    test_end = test_start + duration_seconds

    async def run_one(idx: int) -> None:
        async with semaphore:
            debate_id = f"baseline_{idx:06d}"
            metrics = await run_single_debate(
                debate_id=debate_id,
                num_agents=num_agents,
                num_rounds=num_rounds,
                tokens_per_proposal=20,
                reconnect_probability=0.15,
                fail_rate=0.005,
            )
            async with metrics_lock:
                all_metrics.append(metrics)

    # Launch debates continuously until duration expires
    tasks: list[asyncio.Task] = []
    while time.monotonic() < test_end:
        task = asyncio.create_task(run_one(debate_counter))
        tasks.append(task)
        debate_counter += 1
        # Brief pause between launches to avoid thundering herd
        await asyncio.sleep(max(0.01, 1.0 / max(concurrency, 1)))

    # Wait for all in-flight debates to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    actual_duration = time.monotonic() - test_start
    result.actual_duration_s = actual_duration
    result.total_debates = debate_counter
    result.debate_metrics = all_metrics
    result.completed_at = datetime.now(timezone.utc).isoformat()

    # Evaluate SLOs
    result.slo_results = evaluate_all_slos(all_metrics)
    result.all_slos_passed = all(r.passed for r in result.slo_results.values())

    return result


def evaluate_all_slos(
    debate_metrics: list[DebateMetrics],
) -> dict[str, SLOCheckResult]:
    """Evaluate all four streaming SLOs against collected metrics.

    Args:
        debate_metrics: List of per-debate metrics.

    Returns:
        Dictionary mapping slo_id to SLOCheckResult.
    """
    slos = get_streaming_slo_targets()
    completed = [m for m in debate_metrics if m.completed]
    results: dict[str, SLOCheckResult] = {}

    # SLO 1: First byte latency p95 < 500ms
    first_byte_values = [m.first_byte_ms for m in completed]
    first_byte_p95 = percentile(first_byte_values, 95)
    results["first_byte_latency"] = evaluate_slo(
        slos["first_byte_latency"], first_byte_p95
    )

    # SLO 2: Message throughput >= 10 messages/sec/debate (use p5 as floor)
    throughput_values = [m.messages_per_sec for m in completed]
    # Use p5 (5th percentile) as the effective floor - 95% of debates must meet this
    throughput_p5 = percentile(throughput_values, 5)
    results["message_throughput"] = evaluate_slo(
        slos["message_throughput"], throughput_p5
    )

    # SLO 3: Reconnection success rate >= 99%
    total_attempts = sum(m.reconnect_attempts for m in debate_metrics)
    total_successes = sum(m.reconnect_successes for m in debate_metrics)
    # No reconnection attempts means no failures, so treat as 100% success
    reconnect_rate = 1.0 if total_attempts == 0 else total_successes / total_attempts
    results["reconnection_success_rate"] = evaluate_slo(
        slos["reconnection_success_rate"], reconnect_rate
    )

    # SLO 4: End-to-end debate completion p99 < 30s
    duration_values = [m.duration_s for m in completed]
    completion_p99 = percentile(duration_values, 99)
    results["debate_completion"] = evaluate_slo(
        slos["debate_completion"], completion_p99
    )

    return results


# =============================================================================
# Report Formatting
# =============================================================================


def format_report(result: BaselineResult) -> str:
    """Format baseline results as a human-readable report.

    Args:
        result: The baseline test results.

    Returns:
        Formatted report string.
    """
    report = result.to_dict()
    lines = [
        "",
        "=" * 70,
        "  ARAGORA STREAMING SLO BASELINE REPORT",
        "=" * 70,
        "",
        f"  Configuration: {report['configuration']['concurrency']} concurrent, "
        f"{report['timing']['actual_duration_seconds']}s duration",
        f"  Debates: {report['results']['completed_debates']}/"
        f"{report['configuration']['total_debates']} completed "
        f"({report['results']['success_rate'] * 100:.1f}% success)",
        "",
        "  First Byte Latency (ms):",
        f"    p50:  {report['metrics']['first_byte_latency_ms']['p50']}",
        f"    p95:  {report['metrics']['first_byte_latency_ms']['p95']}",
        f"    p99:  {report['metrics']['first_byte_latency_ms']['p99']}",
        "",
        "  Message Throughput (messages/sec/debate):",
        f"    mean: {report['metrics']['message_throughput_per_sec']['mean']}",
        f"    min:  {report['metrics']['message_throughput_per_sec']['min']}",
        f"    p5:   {report['metrics']['message_throughput_per_sec']['p5']}",
        "",
        "  Reconnection:",
        f"    attempts:  {report['metrics']['reconnection']['total_attempts']}",
        f"    successes: {report['metrics']['reconnection']['total_successes']}",
        f"    rate:      {report['metrics']['reconnection']['success_rate'] * 100:.2f}%",
        "",
        "  Debate Completion (s):",
        f"    p50:  {report['metrics']['debate_completion_s']['p50']}",
        f"    p95:  {report['metrics']['debate_completion_s']['p95']}",
        f"    p99:  {report['metrics']['debate_completion_s']['p99']}",
        "",
        "-" * 70,
        "  SLO Validation:",
    ]

    for slo_id, slo_result in report["slo_validation"].items():
        status = "PASS" if slo_result["passed"] else "FAIL"
        margin_label = "headroom" if slo_result["margin"] >= 0 else "overshoot"
        lines.append(
            f"    [{status}] {slo_result['name']}: "
            f"{slo_result['actual']} {slo_result['unit']} "
            f"(target: {slo_result['target']} {slo_result['unit']}, "
            f"{margin_label}: {abs(slo_result['margin']):.4f})"
        )

    overall = "ALL PASSED" if report["all_slos_passed"] else "FAILED"
    lines.extend([
        "",
        f"  Overall: {overall}",
    ])

    if report["errors"]:
        lines.extend(["", "  Errors (first 20):"])
        for err in report["errors"]:
            lines.append(f"    - {err}")

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for the streaming SLO baseline test."""
    parser = argparse.ArgumentParser(
        description="Aragora Streaming SLO Baseline Load Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum concurrent debates (default: 50)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Agents per debate (default: 3)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Rounds per debate (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for JSON report (default: stdout JSON + report)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any SLO fails",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON (no human-readable report)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting streaming SLO baseline: %ds duration, %d concurrency, "
        "%d agents, %d rounds",
        args.duration,
        args.concurrency,
        args.agents,
        args.rounds,
    )

    # Run the baseline test
    result = asyncio.run(
        run_baseline(
            duration_seconds=args.duration,
            concurrency=args.concurrency,
            num_agents=args.agents,
            num_rounds=args.rounds,
        )
    )

    # Output results
    report_dict = result.to_dict()

    if args.json_only:
        print(json.dumps(report_dict, indent=2))
    else:
        print(format_report(result))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
            f.write("\n")
        logger.info("JSON report written to %s", args.output)

    # Exit code
    if args.strict and not result.all_slos_passed:
        failed_slos = [
            r.name for r in result.slo_results.values() if not r.passed
        ]
        logger.warning("SLO validation failed: %s", ", ".join(failed_slos))
        sys.exit(1)


if __name__ == "__main__":
    main()
