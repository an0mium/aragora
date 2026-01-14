"""
API Benchmark Suite.

Comprehensive performance benchmarks for Aragora API endpoints.

Run with:
    pytest tests/benchmarks/test_api_benchmarks.py -v --benchmark-only
    pytest tests/benchmarks/test_api_benchmarks.py -v --benchmark-json=benchmark.json

Requires pytest-benchmark:
    pip install pytest-benchmark
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if benchmark plugin is available
try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

# Skip benchmark-specific tests if plugin not installed
requires_benchmark = pytest.mark.skipif(
    not HAS_BENCHMARK, reason="pytest-benchmark not installed"
)


class TestHealthEndpointBenchmarks:
    """Benchmark health check endpoint."""

    @pytest.fixture
    def mock_app(self):
        """Create mock application."""
        from aiohttp import web

        app = web.Application()

        async def health_handler(request):
            return web.json_response({"status": "healthy", "version": "1.0.0"})

        app.router.add_get("/api/health", health_handler)
        return app

    @pytest.mark.asyncio
    async def test_health_endpoint_latency(self, benchmark, mock_app):
        """Benchmark health endpoint latency."""
        from aiohttp.test_utils import AioHTTPTestCase, TestClient

        async def health_request():
            from aiohttp import web
            from aiohttp.test_utils import TestClient

            async with TestClient(mock_app) as client:
                resp = await client.get("/api/health")
                return resp.status

        # Run benchmark
        result = benchmark(lambda: asyncio.get_event_loop().run_until_complete(health_request()))
        assert result == 200


class TestUsageTrackingBenchmarks:
    """Benchmark usage tracking system."""

    @pytest.fixture
    def usage_tracker(self, tmp_path):
        """Create usage tracker with temp database."""
        from aragora.billing.usage import UsageTracker

        db_path = tmp_path / "usage.db"
        return UsageTracker(db_path=db_path)

    def test_usage_record_latency(self, benchmark, usage_tracker):
        """Benchmark recording a usage event."""
        from aragora.billing.usage import UsageEvent, UsageEventType

        def record_event():
            event = UsageEvent(
                user_id="user_123",
                org_id="org_456",
                event_type=UsageEventType.DEBATE,
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-3",
            )
            usage_tracker.record(event)

        benchmark(record_event)

    def test_usage_summary_latency(self, benchmark, usage_tracker):
        """Benchmark generating usage summary."""
        from aragora.billing.usage import UsageEvent, UsageEventType

        # Seed with some data
        for i in range(100):
            event = UsageEvent(
                user_id="user_123",
                org_id="org_test",
                event_type=UsageEventType.DEBATE,
                tokens_in=1000 + i,
                tokens_out=500 + i,
                provider="anthropic",
                model="claude-3",
            )
            usage_tracker.record(event)

        def get_summary():
            return usage_tracker.get_summary("org_test")

        result = benchmark(get_summary)
        assert result.total_debates == 100


class TestAuditLogBenchmarks:
    """Benchmark audit log system."""

    @pytest.fixture
    def audit_log(self, tmp_path):
        """Create audit log with temp database."""
        from aragora.audit import AuditLog

        db_path = tmp_path / "audit.db"
        return AuditLog(db_path=db_path)

    def test_audit_log_latency(self, benchmark, audit_log):
        """Benchmark logging an audit event."""
        from aragora.audit import AuditCategory, AuditEvent

        counter = [0]

        def log_event():
            counter[0] += 1
            event = AuditEvent(
                category=AuditCategory.AUTH,
                action="login",
                actor_id=f"user_{counter[0]}",
                resource_type="session",
                resource_id=f"sess_{counter[0]}",
            )
            return audit_log.log(event)

        benchmark(log_event)

    def test_audit_query_latency(self, benchmark, audit_log):
        """Benchmark querying audit events."""
        from aragora.audit import AuditCategory, AuditEvent, AuditQuery

        # Seed with data
        for i in range(200):
            event = AuditEvent(
                category=AuditCategory.AUTH,
                action="login" if i % 2 == 0 else "logout",
                actor_id=f"user_{i % 10}",
            )
            audit_log.log(event)

        def query_events():
            query = AuditQuery(category=AuditCategory.AUTH, limit=50)
            return audit_log.query(query)

        result = benchmark(query_events)
        assert len(result) == 50

    def test_audit_integrity_verification(self, benchmark, audit_log):
        """Benchmark integrity verification."""
        from aragora.audit import AuditCategory, AuditEvent

        # Seed with data
        for i in range(100):
            event = AuditEvent(
                category=AuditCategory.AUTH,
                action="test",
                actor_id=f"user_{i}",
            )
            audit_log.log(event)

        def verify_integrity():
            return audit_log.verify_integrity()

        is_valid, errors = benchmark(verify_integrity)
        assert is_valid
        assert len(errors) == 0


class TestPersonasBenchmarks:
    """Benchmark persona system."""

    @pytest.fixture
    def persona_manager(self, tmp_path):
        """Create persona manager with temp database."""
        from aragora.agents.personas import PersonaManager

        db_path = tmp_path / "personas.db"
        return PersonaManager(db_path=str(db_path))

    def test_persona_get_latency(self, benchmark, persona_manager):
        """Benchmark getting a persona."""
        # Create some personas
        persona_manager.create_persona(
            "test_agent",
            description="Test agent",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.8, "performance": 0.6},
        )

        def get_persona():
            return persona_manager.get_persona("test_agent")

        result = benchmark(get_persona)
        assert result is not None
        assert result.agent_name == "test_agent"

    def test_persona_create_latency(self, benchmark, persona_manager):
        """Benchmark creating a persona."""
        counter = [0]

        def create_persona():
            counter[0] += 1
            return persona_manager.create_persona(
                f"agent_{counter[0]}",
                description=f"Test agent {counter[0]}",
                traits=["thorough"],
                expertise={"security": 0.5},
            )

        benchmark(create_persona)


class TestTokenCostCalculation:
    """Benchmark token cost calculations."""

    def test_cost_calculation_latency(self, benchmark):
        """Benchmark token cost calculation."""
        from aragora.billing.usage import calculate_token_cost

        def calculate():
            return calculate_token_cost(
                provider="anthropic",
                model="claude-opus-4",
                tokens_in=10000,
                tokens_out=5000,
            )

        result = benchmark(calculate)
        assert result > 0


class TestJSONSerializationBenchmarks:
    """Benchmark JSON serialization performance."""

    def test_audit_event_serialization(self, benchmark):
        """Benchmark audit event to dict conversion."""
        from aragora.audit import AuditCategory, AuditEvent

        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="user_123",
            resource_type="session",
            resource_id="sess_456",
            details={"ip": "127.0.0.1", "user_agent": "Mozilla/5.0"},
        )

        def serialize():
            return event.to_dict()

        result = benchmark(serialize)
        assert "id" in result

    def test_usage_summary_serialization(self, benchmark):
        """Benchmark usage summary serialization."""
        from datetime import datetime
        from decimal import Decimal

        from aragora.billing.usage import UsageSummary

        summary = UsageSummary(
            org_id="org_123",
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_debates=100,
            total_tokens_in=1000000,
            total_tokens_out=500000,
            total_cost_usd=Decimal("15.50"),
            cost_by_provider={"anthropic": Decimal("10.00"), "openai": Decimal("5.50")},
        )

        def serialize():
            return summary.to_dict()

        result = benchmark(serialize)
        assert result["total_debates"] == 100


class TestConcurrencyBenchmarks:
    """Benchmark concurrent operations."""

    @pytest.fixture
    def audit_log(self, tmp_path):
        """Create audit log."""
        from aragora.audit import AuditLog

        return AuditLog(db_path=tmp_path / "audit.db")

    def test_concurrent_audit_writes(self, benchmark, audit_log):
        """Benchmark concurrent audit log writes."""
        from aragora.audit import AuditCategory, AuditEvent

        import threading

        def concurrent_writes():
            threads = []
            for i in range(10):

                def write(idx):
                    for j in range(10):
                        event = AuditEvent(
                            category=AuditCategory.AUTH,
                            action=f"action_{idx}_{j}",
                            actor_id=f"user_{idx}",
                        )
                        audit_log.log(event)

                t = threading.Thread(target=write, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        benchmark(concurrent_writes)


class TestBasicFunctionality:
    """Basic functionality tests that don't require benchmark plugin."""

    def test_usage_tracker_initialization(self, tmp_path):
        """Test usage tracker can be initialized."""
        from aragora.billing.usage import UsageTracker

        tracker = UsageTracker(db_path=tmp_path / "usage.db")
        assert tracker is not None

    def test_audit_log_initialization(self, tmp_path):
        """Test audit log can be initialized."""
        from aragora.audit import AuditLog

        audit = AuditLog(db_path=tmp_path / "audit.db")
        assert audit is not None

    def test_audit_log_write_and_query(self, tmp_path):
        """Test audit log write and query."""
        from aragora.audit import AuditCategory, AuditEvent, AuditLog, AuditQuery

        audit = AuditLog(db_path=tmp_path / "audit.db")

        # Write
        event = AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id="test_user",
        )
        event_id = audit.log(event)
        assert event_id is not None

        # Query
        query = AuditQuery(category=AuditCategory.AUTH, limit=10)
        events = audit.query(query)
        assert len(events) >= 1

    def test_usage_tracking_flow(self, tmp_path):
        """Test usage tracking end-to-end."""
        from aragora.billing.usage import UsageEvent, UsageEventType, UsageTracker

        tracker = UsageTracker(db_path=tmp_path / "usage.db")

        # Record event
        event = UsageEvent(
            user_id="user_123",
            org_id="org_456",
            event_type=UsageEventType.DEBATE,
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-3",
        )
        tracker.record(event)

        # Get summary
        summary = tracker.get_summary("org_456")
        assert summary.total_debates >= 1

    def test_token_cost_calculation(self):
        """Test token cost calculation."""
        from aragora.billing.usage import calculate_token_cost

        cost = calculate_token_cost(
            provider="anthropic",
            model="claude-opus-4",
            tokens_in=10000,
            tokens_out=5000,
        )
        assert cost > 0

    def test_audit_integrity_verification(self, tmp_path):
        """Test audit log integrity verification."""
        from aragora.audit import AuditCategory, AuditEvent, AuditLog

        audit = AuditLog(db_path=tmp_path / "audit.db")

        # Write some events
        for i in range(10):
            event = AuditEvent(
                category=AuditCategory.AUTH,
                action=f"action_{i}",
                actor_id="test_user",
            )
            audit.log(event)

        # Verify integrity
        is_valid, errors = audit.verify_integrity()
        assert is_valid
        assert len(errors) == 0


# Performance thresholds for CI/CD gates
PERFORMANCE_THRESHOLDS = {
    "health_endpoint_p99": 50,  # 50ms
    "usage_record": 5,  # 5ms
    "usage_summary": 50,  # 50ms
    "audit_log": 5,  # 5ms
    "audit_query": 100,  # 100ms
    "persona_get": 5,  # 5ms
    "cost_calculation": 0.1,  # 0.1ms
}


def get_performance_report(benchmark_results: dict[str, float]) -> dict[str, Any]:
    """
    Generate performance report comparing results to thresholds.

    Args:
        benchmark_results: Dict of benchmark name to latency in ms

    Returns:
        Report with pass/fail status and details
    """
    report = {
        "passed": True,
        "results": [],
        "summary": {
            "total": len(benchmark_results),
            "passed": 0,
            "failed": 0,
        },
    }

    for name, latency in benchmark_results.items():
        threshold = PERFORMANCE_THRESHOLDS.get(name)
        if threshold is None:
            status = "skip"
        elif latency <= threshold:
            status = "pass"
            report["summary"]["passed"] += 1
        else:
            status = "fail"
            report["summary"]["failed"] += 1
            report["passed"] = False

        report["results"].append(
            {
                "name": name,
                "latency_ms": latency,
                "threshold_ms": threshold,
                "status": status,
            }
        )

    return report
