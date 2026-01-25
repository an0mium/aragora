"""
Benchmark fixtures for Aragora performance tests.

These benchmarks measure:
- Debate round latency
- Memory tier operations
- ELO calculations
- WebSocket throughput
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Generator, Callable
from unittest.mock import AsyncMock

import pytest

from aragora.core import Agent, Vote, Critique, Environment

pytestmark = pytest.mark.benchmark


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)


# =============================================================================
# Simple Timing Decorator (works without pytest-benchmark)
# =============================================================================


class SimpleBenchmark:
    """Simple benchmark helper when pytest-benchmark is not available."""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []

    def __call__(self, func: Callable) -> float:
        """Run function and record time."""
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        self.times.append(elapsed)
        return result

    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0


# =============================================================================
# Mock Agent for Benchmarks
# =============================================================================


class BenchmarkAgent(Agent):
    """Minimal agent for benchmark testing."""

    def __init__(self, name: str = "bench_agent", delay: float = 0.0):
        super().__init__(name, "benchmark-model", "proposer")
        self.agent_type = "benchmark"
        self._delay = delay

    async def generate(self, prompt: str, context: list = None) -> str:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return f"Response from {self.name}"

    async def critique(
        self, proposal: str, task: str, context: list = None, target_agent: str = None
    ) -> Critique:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return Critique(
            agent=self.name,
            target_agent=target_agent or "target",
            target_content=proposal[:100],
            issues=[],
            suggestions=[],
            severity=0.1,
            reasoning="Benchmark critique",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Benchmark vote",
            confidence=0.9,
            continue_debate=False,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def benchmark_agent() -> BenchmarkAgent:
    """Fast agent for benchmarking."""
    return BenchmarkAgent()


@pytest.fixture
def benchmark_agents() -> list[BenchmarkAgent]:
    """Set of fast agents for benchmarking."""
    return [BenchmarkAgent(f"agent_{i}") for i in range(3)]


@pytest.fixture
def temp_benchmark_db() -> Generator[Path, None, None]:
    """Temporary database for benchmark tests."""
    import gc
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "benchmark.db"
        # Force garbage collection to release SQLite connections
        gc.collect()
        # On Windows, give a brief moment for file handles to release
        if sys.platform == "win32":
            import time

            time.sleep(0.1)


@pytest.fixture
def simple_benchmark():
    """Simple benchmark helper."""
    return SimpleBenchmark


@pytest.fixture
def benchmark_environment() -> Environment:
    """Simple environment for benchmarking."""
    return Environment(task="Benchmark task", context="")
