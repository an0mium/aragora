"""
Extended Debate Benchmarks.

Measures:
- Memory usage for 50+ round debates
- Context management efficiency
- Checkpoint/resume performance
- Streaming event throughput
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    times_ms: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        idx = int(len(self.times_ms) * 0.95)
        return sorted(self.times_ms)[idx]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0
        idx = int(len(self.times_ms) * 0.99)
        return sorted(self.times_ms)[idx]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "iterations": self.iterations,
            "latency_ms": {
                "p50": round(self.p50_ms, 3),
                "p95": round(self.p95_ms, 3),
                "p99": round(self.p99_ms, 3),
            },
            "memory_mb": {
                "avg": round(statistics.mean(self.memory_mb), 2) if self.memory_mb else 0,
                "peak": round(max(self.memory_mb), 2) if self.memory_mb else 0,
            },
        }
        for key, values in self.custom_metrics.items():
            if values:
                result[key] = {
                    "avg": round(statistics.mean(values), 3),
                    "max": round(max(values), 3),
                }
        return result


async def benchmark_extended_debate_memory(iterations: int = 10) -> BenchmarkResult:
    """Benchmark memory usage for 50+ round debates."""
    from aragora.debate.extended_rounds import ExtendedDebateConfig, RLMContextManager

    result = BenchmarkResult(name="extended_debate_memory", iterations=iterations)
    result.custom_metrics["rounds_completed"] = []
    result.custom_metrics["context_size_kb"] = []

    for _ in range(iterations):
        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()

        config = ExtendedDebateConfig(max_rounds=55, compression_threshold=0.7)
        context_manager = RLMContextManager(config)

        # Simulate 55 rounds
        for round_num in range(55):
            responses = [
                {"agent": "claude", "content": f"Round {round_num} response from claude. " * 20},
                {"agent": "gpt4", "content": f"Round {round_num} response from gpt4. " * 20},
                {"agent": "gemini", "content": f"Round {round_num} response from gemini. " * 20},
            ]
            await context_manager.add_round(round_num, responses)

        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        context = context_manager.get_context()
        context_size = sys.getsizeof(context) / 1024

        result.times_ms.append(elapsed)
        result.memory_mb.append(peak / 1024 / 1024)
        result.custom_metrics["rounds_completed"].append(55)
        result.custom_metrics["context_size_kb"].append(context_size)

    return result


async def benchmark_context_compression(iterations: int = 50) -> BenchmarkResult:
    """Benchmark context compression during extended debates."""
    from aragora.debate.extended_rounds import RLMContextManager

    result = BenchmarkResult(name="context_compression", iterations=iterations)
    result.custom_metrics["compression_ratio"] = []

    context_manager = RLMContextManager()

    for i in range(iterations):
        gc.collect()

        # Add 10 rounds
        initial_size = 0
        for round_num in range(10):
            responses = [
                {"agent": f"agent_{j}", "content": f"Response {round_num} " * 50} for j in range(3)
            ]
            if round_num == 0:
                initial_size = sum(len(r["content"]) for r in responses)
            await context_manager.add_round(i * 10 + round_num, responses)

        start = time.perf_counter()
        compressed = await context_manager.compress()
        elapsed = (time.perf_counter() - start) * 1000

        final_size = len(compressed) if compressed else 0
        ratio = 1 - (final_size / (initial_size * 10)) if initial_size > 0 else 0

        result.times_ms.append(elapsed)
        result.custom_metrics["compression_ratio"].append(ratio * 100)

    return result


async def benchmark_checkpoint_resume(iterations: int = 50) -> BenchmarkResult:
    """Benchmark debate checkpoint and resume performance."""
    from aragora.debate.extended_rounds import RLMContextManager

    result = BenchmarkResult(name="checkpoint_resume", iterations=iterations)
    result.custom_metrics["checkpoint_size_kb"] = []

    for _ in range(iterations):
        gc.collect()

        context_manager = RLMContextManager()

        # Build up 20 rounds
        for round_num in range(20):
            responses = [
                {"agent": f"agent_{j}", "content": f"Response {round_num} " * 30} for j in range(3)
            ]
            await context_manager.add_round(round_num, responses)

        # Checkpoint
        start = time.perf_counter()
        checkpoint = context_manager.checkpoint()
        checkpoint_time = (time.perf_counter() - start) * 1000

        checkpoint_size = sys.getsizeof(str(checkpoint)) / 1024

        # Resume
        new_manager = RLMContextManager()
        resume_start = time.perf_counter()
        new_manager.restore(checkpoint)
        resume_time = (time.perf_counter() - resume_start) * 1000

        total_time = checkpoint_time + resume_time

        result.times_ms.append(total_time)
        result.custom_metrics["checkpoint_size_kb"].append(checkpoint_size)

    return result


async def benchmark_event_streaming(iterations: int = 100) -> BenchmarkResult:
    """Benchmark debate event streaming throughput."""
    result = BenchmarkResult(name="event_streaming", iterations=iterations)
    result.custom_metrics["events_per_second"] = []

    for _ in range(iterations):
        events_sent = 0
        events: List[Dict[str, Any]] = []

        async def event_handler(event: Dict[str, Any]):
            nonlocal events_sent
            events_sent += 1
            events.append(event)

        gc.collect()

        start = time.perf_counter()

        # Simulate 100 events
        for i in range(100):
            await event_handler(
                {
                    "type": "agent_message",
                    "round": i // 3,
                    "agent": f"agent_{i % 3}",
                    "content": f"Message {i}",
                }
            )

        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000

        events_per_sec = events_sent / elapsed if elapsed > 0 else 0

        result.times_ms.append(elapsed_ms)
        result.custom_metrics["events_per_second"].append(events_per_sec)

    return result


async def run_debate_benchmarks(iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
    """Run all extended debate benchmarks."""

    results = {}

    # Warmup
    await benchmark_event_streaming(warmup)

    benchmarks = [
        ("Extended Debate Memory", benchmark_extended_debate_memory),
        ("Context Compression", benchmark_context_compression),
        ("Checkpoint Resume", benchmark_checkpoint_resume),
        ("Event Streaming", benchmark_event_streaming),
    ]

    for name, bench_func in benchmarks:
        try:
            # Use fewer iterations for memory-intensive tests
            iters = iterations // 10 if "Memory" in name else iterations
            result = await bench_func(iters)
            results[result.name] = result.to_dict()
        except Exception as e:
            results[name.lower().replace(" ", "_")] = {"error": str(e)}

    return results


if __name__ == "__main__":
    results = asyncio.run(run_debate_benchmarks())
