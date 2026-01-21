"""
RLM Compression Benchmarks.

Measures:
- Compression ratio at different context sizes
- Token savings per operation
- Compression/decompression latency
- Memory usage during compression
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

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

    @property
    def avg_memory_mb(self) -> float:
        return statistics.mean(self.memory_mb) if self.memory_mb else 0

    @property
    def peak_memory_mb(self) -> float:
        return max(self.memory_mb) if self.memory_mb else 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "iterations": self.iterations,
            "latency_ms": {
                "p50": round(self.p50_ms, 3),
                "p95": round(self.p95_ms, 3),
                "p99": round(self.p99_ms, 3),
                "min": round(min(self.times_ms), 3) if self.times_ms else 0,
                "max": round(max(self.times_ms), 3) if self.times_ms else 0,
            },
            "memory_mb": {
                "avg": round(self.avg_memory_mb, 2),
                "peak": round(self.peak_memory_mb, 2),
            },
        }

        for key, values in self.custom_metrics.items():
            if values:
                result[key] = {
                    "avg": round(statistics.mean(values), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                }

        return result


async def benchmark_compression_ratio(iterations: int = 100) -> BenchmarkResult:
    """Benchmark compression ratio at various context sizes."""
    from aragora.rlm.compressor import AragoraRLM

    result = BenchmarkResult(name="compression_ratio", iterations=iterations)
    result.custom_metrics["compression_ratio"] = []
    result.custom_metrics["tokens_saved"] = []

    rlm = AragoraRLM()

    # Generate test contexts of varying sizes
    context_sizes = [1000, 5000, 10000, 50000, 100000]

    for size in context_sizes:
        context = "This is sample debate context. " * (size // 30)

        for _ in range(iterations // len(context_sizes)):
            gc.collect()
            tracemalloc.start()

            start = time.perf_counter()
            compressed = await rlm.compress(context)
            elapsed = (time.perf_counter() - start) * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result.times_ms.append(elapsed)
            result.memory_mb.append(peak / 1024 / 1024)

            # Calculate compression ratio
            original_tokens = len(context.split())
            compressed_tokens = len(compressed.split()) if compressed else 0
            ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

            result.custom_metrics["compression_ratio"].append(ratio * 100)
            result.custom_metrics["tokens_saved"].append(original_tokens - compressed_tokens)

    return result


async def benchmark_hierarchical_compression(iterations: int = 100) -> BenchmarkResult:
    """Benchmark hierarchical multi-level compression."""
    from aragora.rlm.compressor import AragoraRLM

    result = BenchmarkResult(name="hierarchical_compression", iterations=iterations)
    result.custom_metrics["levels_compressed"] = []

    rlm = AragoraRLM()

    # Simulate multi-round debate context
    rounds = [
        f"Round {i}: Agent discusses point about topic with detailed analysis. " * 10
        for i in range(20)
    ]

    for _ in range(iterations):
        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()
        # Compress with hierarchical levels
        compressed = await rlm.compress_hierarchical(rounds, levels=3)
        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.times_ms.append(elapsed)
        result.memory_mb.append(peak / 1024 / 1024)
        result.custom_metrics["levels_compressed"].append(len(compressed) if compressed else 0)

    return result


async def benchmark_incremental_compression(iterations: int = 100) -> BenchmarkResult:
    """Benchmark incremental context updates."""
    from aragora.rlm.compressor import AragoraRLM

    result = BenchmarkResult(name="incremental_compression", iterations=iterations)

    rlm = AragoraRLM()

    # Start with initial context
    context = "Initial debate context with some discussion points. " * 100

    for _ in range(iterations):
        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()
        # Add new round
        new_content = "New round with additional arguments and counter-points. " * 10
        compressed = await rlm.compress_incremental(context, new_content)
        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.times_ms.append(elapsed)
        result.memory_mb.append(peak / 1024 / 1024)

        # Update context for next iteration
        context = compressed if compressed else context

    return result


async def benchmark_query_latency(iterations: int = 100) -> BenchmarkResult:
    """Benchmark RLM query response latency."""
    from aragora.rlm.compressor import AragoraRLM

    result = BenchmarkResult(name="query_latency", iterations=iterations)

    rlm = AragoraRLM()

    # Build context
    context = "Debate about software architecture patterns. " * 500

    # Compress once
    compressed = await rlm.compress(context)

    queries = [
        "What are the main arguments?",
        "What consensus was reached?",
        "What are the key disagreements?",
        "Summarize the discussion",
    ]

    for i in range(iterations):
        query = queries[i % len(queries)]

        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()
        await rlm.query(compressed, query)
        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.times_ms.append(elapsed)
        result.memory_mb.append(peak / 1024 / 1024)

    return result


async def run_rlm_benchmarks(iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
    """Run all RLM benchmarks."""

    results = {}

    # Warmup
    await benchmark_compression_ratio(warmup)

    benchmarks = [
        ("Compression Ratio", benchmark_compression_ratio),
        ("Hierarchical Compression", benchmark_hierarchical_compression),
        ("Incremental Compression", benchmark_incremental_compression),
        ("Query Latency", benchmark_query_latency),
    ]

    for name, bench_func in benchmarks:
        try:
            result = await bench_func(iterations)
            results[result.name] = result.to_dict()
        except Exception as e:
            results[name.lower().replace(" ", "_")] = {"error": str(e)}

    return results


if __name__ == "__main__":
    results = asyncio.run(run_rlm_benchmarks())
