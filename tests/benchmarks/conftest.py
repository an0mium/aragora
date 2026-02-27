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
from collections.abc import Generator, Callable
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


@pytest.fixture(autouse=True)
def _disable_post_debate_steps(monkeypatch):
    """Prevent PostDebateCoordinator from blocking on real async calls."""
    try:
        from aragora.debate import post_debate_coordinator as pdc_mod

        def _noop(self, *a, **kw):
            return None

        monkeypatch.setattr(pdc_mod.PostDebateCoordinator, "_step_llm_judge", _noop)
        monkeypatch.setattr(pdc_mod.PostDebateCoordinator, "_step_outcome_feedback", _noop)
    except (ImportError, AttributeError):
        pass

    # Reset circuit breakers so failed API calls in one test don't
    # trip breakers for subsequent tests.
    try:
        from aragora.resilience.registry import reset_all_circuit_breakers

        reset_all_circuit_breakers()
    except ImportError:
        pass

    yield

    try:
        from aragora.resilience.registry import reset_all_circuit_breakers

        reset_all_circuit_breakers()
    except ImportError:
        pass


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


# =============================================================================
# Decision Receipt Fixtures
# =============================================================================


def _make_receipt_findings(count: int = 10) -> list:
    """Create a list of ReceiptFinding objects for benchmarking."""
    from aragora.export.decision_receipt import ReceiptFinding

    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    return [
        ReceiptFinding(
            id=f"finding-{i}",
            severity=severities[i % len(severities)],
            category=f"category-{i % 4}",
            title=f"Finding {i}: Potential issue in component",
            description=f"Detailed description of finding {i} with enough text to be realistic.",
            mitigation=f"Apply fix {i} to resolve this issue.",
            source=f"agent-{i % 3}",
            verified=i % 2 == 0,
        )
        for i in range(count)
    ]


def _make_receipt_dissents(count: int = 2) -> list:
    """Create a list of ReceiptDissent objects for benchmarking."""
    from aragora.export.decision_receipt import ReceiptDissent

    return [
        ReceiptDissent(
            agent=f"agent-{i}",
            type="philosophical",
            severity=0.6 + i * 0.1,
            reasons=[f"Reason {j} for dissent" for j in range(2)],
            alternative=f"Alternative approach {i}",
        )
        for i in range(count)
    ]


def _make_receipt_verifications(count: int = 3) -> list:
    """Create a list of ReceiptVerification objects for benchmarking."""
    from aragora.export.decision_receipt import ReceiptVerification

    return [
        ReceiptVerification(
            claim=f"Claim {i}: The system handles edge case correctly",
            verified=i % 2 == 0,
            method="formal" if i % 2 == 0 else "heuristic",
            proof_hash=f"sha256:{i:064x}" if i % 2 == 0 else None,
        )
        for i in range(count)
    ]


@pytest.fixture
def sample_receipt():
    """Pre-built DecisionReceipt with 10 findings for benchmark use."""
    from aragora.export.decision_receipt import DecisionReceipt

    findings = _make_receipt_findings(10)
    return DecisionReceipt(
        receipt_id="rcpt_bench_0001",
        gauntlet_id="gauntlet_bench_0001",
        timestamp="2026-02-11T00:00:00Z",
        input_summary="Benchmark test input for performance measurement",
        input_type="spec",
        verdict="approved_with_conditions",
        confidence=0.82,
        risk_level="MEDIUM",
        risk_score=0.18,
        robustness_score=0.85,
        coverage_score=0.90,
        verification_coverage=0.75,
        findings=findings,
        critical_count=3,
        high_count=3,
        medium_count=2,
        low_count=2,
        mitigations=["Apply security patch", "Add input validation", "Enable rate limiting"],
        dissenting_views=_make_receipt_dissents(2),
        unresolved_tensions=["Tension between performance and security"],
        verified_claims=_make_receipt_verifications(3),
        unverified_claims=["Claim about scalability", "Claim about fault tolerance"],
        agents_involved=["agent-0", "agent-1", "agent-2"],
        rounds_completed=3,
        duration_seconds=12.5,
    )


@pytest.fixture
def mock_review_findings() -> dict:
    """Mock review findings dict matching the shape of extract_review_findings output."""
    return {
        "unanimous_critiques": [
            "SQL injection vulnerability in user search",
            "Missing input validation on file upload endpoint",
        ],
        "split_opinions": [
            ("Add request rate limiting", ["anthropic-api", "openai-api"], ["gemini-api"]),
        ],
        "risk_areas": [
            "Error handling in payment flow may expose sensitive data",
        ],
        "agreement_score": 0.75,
        "agent_alignment": {
            "anthropic-api": {"openai-api": 0.8},
        },
        "critical_issues": [
            {
                "agent": "anthropic-api",
                "issue": "SQL injection in search_users()",
                "target": "api/users.py:45",
                "suggestions": ["Use parameterized queries"],
            },
        ],
        "high_issues": [
            {
                "agent": "openai-api",
                "issue": "Missing CSRF protection on POST endpoints",
                "target": "api/routes.py",
                "suggestions": ["Add CSRF token middleware"],
            },
        ],
        "medium_issues": [
            {
                "agent": "gemini-api",
                "issue": "Unbounded query results - add pagination",
                "target": "api/products.py:102",
                "suggestions": [],
            },
        ],
        "low_issues": [
            {
                "agent": "openai-api",
                "issue": "Consider adding debug logging",
                "target": "api/utils.py",
                "suggestions": [],
            },
        ],
        "all_critiques": [],
        "final_summary": "Multi-agent review found 1 critical and 1 high severity issue.",
        "agents_used": ["anthropic-api", "openai-api", "gemini-api"],
    }
