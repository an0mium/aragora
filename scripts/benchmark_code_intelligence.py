#!/usr/bin/env python3
"""
Benchmark Script for Code Intelligence Performance Testing.

Measures performance of code intelligence components on repositories
of various sizes. Useful for identifying bottlenecks and regressions.

Usage:
    python scripts/benchmark_code_intelligence.py [path] [--iterations N]

Examples:
    # Quick benchmark
    python scripts/benchmark_code_intelligence.py . --iterations 3

    # Full benchmark on a specific repo
    python scripts/benchmark_code_intelligence.py /path/to/large/repo
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iterations: int
    times_ms: List[float] = field(default_factory=list)
    errors: int = 0
    memory_mb: float = 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    path: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    file_count: int = 0
    line_count: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "file_count": self.file_count,
            "line_count": self.line_count,
            "results": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "mean_ms": round(r.mean_ms, 2),
                    "median_ms": round(r.median_ms, 2),
                    "std_ms": round(r.std_ms, 2),
                    "min_ms": round(r.min_ms, 2),
                    "max_ms": round(r.max_ms, 2),
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }


def count_files_and_lines(path: str) -> tuple[int, int]:
    """Count source files and lines in a directory."""
    extensions = {".py", ".js", ".ts", ".tsx", ".go", ".java", ".rs"}
    exclude_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}

    file_count = 0
    line_count = 0

    for p in Path(path).rglob("*"):
        if p.is_file() and p.suffix in extensions:
            if not any(ex in p.parts for ex in exclude_dirs):
                file_count += 1
                try:
                    line_count += sum(1 for _ in open(p, "r", encoding="utf-8", errors="ignore"))
                except (OSError, IOError):
                    pass

    return file_count, line_count


async def benchmark_codebase_indexing(path: str, iterations: int) -> BenchmarkResult:
    """Benchmark codebase indexing performance."""
    result = BenchmarkResult(name="codebase_indexing", iterations=iterations)

    try:
        from aragora.agents.codebase_agent import CodebaseUnderstandingAgent
    except ImportError:
        result.errors = iterations
        return result

    for i in range(iterations):
        gc.collect()
        agent = CodebaseUnderstandingAgent(root_path=path, enable_debate=False)

        start = time.perf_counter()
        try:
            await agent.index_codebase(force=True)
            elapsed = (time.perf_counter() - start) * 1000
            result.times_ms.append(elapsed)
        except Exception as e:
            logger.warning(f"Indexing error: {e}")
            result.errors += 1

    return result


async def benchmark_understanding_query(path: str, iterations: int) -> BenchmarkResult:
    """Benchmark understanding query performance."""
    result = BenchmarkResult(name="understanding_query", iterations=iterations)

    try:
        from aragora.agents.codebase_agent import CodebaseUnderstandingAgent
    except ImportError:
        result.errors = iterations
        return result

    agent = CodebaseUnderstandingAgent(root_path=path, enable_debate=False)
    await agent.index_codebase()

    queries = [
        "What are the main modules?",
        "How does authentication work?",
        "Where are errors handled?",
    ]

    for i in range(iterations):
        gc.collect()
        query = queries[i % len(queries)]

        start = time.perf_counter()
        try:
            await agent.understand(query, max_files=5)
            elapsed = (time.perf_counter() - start) * 1000
            result.times_ms.append(elapsed)
        except Exception as e:
            logger.warning(f"Query error: {e}")
            result.errors += 1

    return result


async def benchmark_security_scan(path: str, iterations: int) -> BenchmarkResult:
    """Benchmark security scanning performance."""
    result = BenchmarkResult(name="security_scan", iterations=iterations)

    try:
        from aragora.audit.security_scanner import SecurityScanner
    except ImportError:
        result.errors = iterations
        return result

    scanner = SecurityScanner()

    for i in range(iterations):
        gc.collect()

        start = time.perf_counter()
        try:
            scanner.scan_directory(
                path,
                exclude_patterns=["__pycache__", ".git", "node_modules", ".venv"],
            )
            elapsed = (time.perf_counter() - start) * 1000
            result.times_ms.append(elapsed)
        except Exception as e:
            logger.warning(f"Scan error: {e}")
            result.errors += 1

    return result


async def benchmark_code_review(iterations: int) -> BenchmarkResult:
    """Benchmark code review performance."""
    result = BenchmarkResult(name="code_review", iterations=iterations)

    try:
        from aragora.agents.code_reviewer import CodeReviewOrchestrator
    except ImportError:
        result.errors = iterations
        return result

    sample_code = """
import os
import subprocess

def execute_command(user_input):
    result = subprocess.run(user_input, shell=True, capture_output=True)
    return result.stdout

def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
"""

    orchestrator = CodeReviewOrchestrator()

    for i in range(iterations):
        gc.collect()

        start = time.perf_counter()
        try:
            await orchestrator.review_code(code=sample_code, file_path="sample.py")
            elapsed = (time.perf_counter() - start) * 1000
            result.times_ms.append(elapsed)
        except Exception as e:
            logger.warning(f"Review error: {e}")
            result.errors += 1

    return result


async def run_benchmarks(
    path: str,
    iterations: int = 5,
    benchmarks: Optional[List[str]] = None,
) -> BenchmarkSuite:
    """Run all benchmarks."""
    suite = BenchmarkSuite(path=path)

    # Count files
    print("Counting files...")
    suite.file_count, suite.line_count = count_files_and_lines(path)
    print(f"Found {suite.file_count} source files ({suite.line_count:,} lines)")

    all_benchmarks = {
        "indexing": lambda: benchmark_codebase_indexing(path, iterations),
        "query": lambda: benchmark_understanding_query(path, iterations),
        "security": lambda: benchmark_security_scan(path, iterations),
        "review": lambda: benchmark_code_review(iterations),
    }

    benchmarks = benchmarks or list(all_benchmarks.keys())

    for name in benchmarks:
        if name not in all_benchmarks:
            print(f"Unknown benchmark: {name}")
            continue

        print(f"\nRunning {name} benchmark ({iterations} iterations)...")
        result = await all_benchmarks[name]()
        suite.results.append(result)

        if result.times_ms:
            print(f"  Mean: {result.mean_ms:.1f}ms")
            print(f"  Median: {result.median_ms:.1f}ms")
            print(f"  Min: {result.min_ms:.1f}ms, Max: {result.max_ms:.1f}ms")
            if result.errors:
                print(f"  Errors: {result.errors}")
        else:
            print("  All iterations failed or component not available")

    suite.completed_at = datetime.now()
    return suite


def print_summary(suite: BenchmarkSuite) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print(" BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nPath: {suite.path}")
    print(f"Files: {suite.file_count}")
    print(f"Lines: {suite.line_count:,}")
    elapsed = (suite.completed_at - suite.started_at).total_seconds()
    print(f"Duration: {elapsed:.1f}s")

    print("\n" + "-" * 60)
    print(f"{'Benchmark':<25} {'Mean (ms)':<12} {'Median (ms)':<12} {'Errors':<8}")
    print("-" * 60)

    for r in suite.results:
        print(f"{r.name:<25} {r.mean_ms:<12.1f} {r.median_ms:<12.1f} {r.errors:<8}")

    print("-" * 60)

    # Performance ratings
    print("\nPerformance Ratings:")
    for r in suite.results:
        if r.mean_ms < 100:
            rating = "Excellent"
        elif r.mean_ms < 500:
            rating = "Good"
        elif r.mean_ms < 2000:
            rating = "Acceptable"
        else:
            rating = "Needs optimization"
        print(f"  {r.name}: {rating} ({r.mean_ms:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark code intelligence performance",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to codebase (default: current directory)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per benchmark (default: 5)",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        choices=["indexing", "query", "security", "review"],
        help="Specific benchmarks to run (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    print("=" * 60)
    print(" CODE INTELLIGENCE BENCHMARK")
    print("=" * 60)

    suite = asyncio.run(
        run_benchmarks(
            str(path),
            iterations=args.iterations,
            benchmarks=args.benchmarks,
        )
    )

    print_summary(suite)

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
