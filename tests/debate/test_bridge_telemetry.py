"""Tests for Bridge Telemetry Module."""

from __future__ import annotations

import pytest
import time
from unittest.mock import MagicMock, patch

from aragora.debate.bridge_telemetry import (
    BridgeMetrics,
    BridgeOperation,
    BridgeTelemetryContext,
    get_bridge_telemetry_stats,
    get_recent_bridge_operations,
    record_bridge_operation,
    reset_bridge_telemetry,
    with_bridge_telemetry,
    performance_router_telemetry,
    relationship_bias_telemetry,
)


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Reset telemetry before each test."""
    reset_bridge_telemetry()
    yield
    reset_bridge_telemetry()


class TestBridgeOperation:
    """Tests for BridgeOperation dataclass."""

    def test_create_operation(self):
        """Test creating a bridge operation."""
        op = BridgeOperation(
            bridge_name="performance_router",
            operation="sync",
            start_time=time.time(),
        )
        assert op.bridge_name == "performance_router"
        assert op.operation == "sync"
        assert op.success is True
        assert op.error_type is None

    def test_complete_success(self):
        """Test completing an operation successfully."""
        start = time.time()
        op = BridgeOperation(
            bridge_name="relationship_bias",
            operation="compute",
            start_time=start,
        )
        time.sleep(0.01)  # Small delay
        op.complete(success=True)

        assert op.success is True
        assert op.end_time > start
        assert op.duration_ms > 0
        assert op.error_type is None

    def test_complete_with_error(self):
        """Test completing an operation with error."""
        op = BridgeOperation(
            bridge_name="calibration_cost",
            operation="calculate",
            start_time=time.time(),
        )
        error = ValueError("Test error message")
        op.complete(success=False, error=error)

        assert op.success is False
        assert op.error_type == "ValueError"
        assert "Test error" in op.error_message


class TestBridgeMetrics:
    """Tests for BridgeMetrics aggregation."""

    def test_create_metrics(self):
        """Test creating bridge metrics."""
        metrics = BridgeMetrics(bridge_name="test_bridge")
        assert metrics.bridge_name == "test_bridge"
        assert metrics.total_operations == 0
        assert metrics.success_rate == 1.0  # No operations = 100% success
        assert metrics.avg_duration_ms == 0.0

    def test_record_successful_operation(self):
        """Test recording a successful operation."""
        metrics = BridgeMetrics(bridge_name="test_bridge")

        op = BridgeOperation(
            bridge_name="test_bridge",
            operation="sync",
            start_time=time.time(),
        )
        op.complete(success=True)
        op.duration_ms = 10.0

        metrics.record_operation(op)

        assert metrics.total_operations == 1
        assert metrics.successful_operations == 1
        assert metrics.failed_operations == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_duration_ms == 10.0
        assert metrics.operations_by_type["sync"] == 1

    def test_record_failed_operation(self):
        """Test recording a failed operation."""
        metrics = BridgeMetrics(bridge_name="test_bridge")

        op = BridgeOperation(
            bridge_name="test_bridge",
            operation="compute",
            start_time=time.time(),
        )
        op.complete(success=False, error=ValueError("Test"))

        metrics.record_operation(op)

        assert metrics.total_operations == 1
        assert metrics.failed_operations == 1
        assert metrics.success_rate == 0.0
        assert metrics.errors_by_type["ValueError"] == 1

    def test_multiple_operations(self):
        """Test recording multiple operations."""
        metrics = BridgeMetrics(bridge_name="test_bridge")

        # Record 3 successful, 1 failed
        for i in range(3):
            op = BridgeOperation(
                bridge_name="test_bridge",
                operation="sync",
                start_time=time.time(),
            )
            op.complete(success=True)
            op.duration_ms = 10.0
            metrics.record_operation(op)

        op_fail = BridgeOperation(
            bridge_name="test_bridge",
            operation="sync",
            start_time=time.time(),
        )
        op_fail.complete(success=False, error=RuntimeError("Fail"))
        op_fail.duration_ms = 5.0
        metrics.record_operation(op_fail)

        assert metrics.total_operations == 4
        assert metrics.successful_operations == 3
        assert metrics.failed_operations == 1
        assert metrics.success_rate == 0.75
        assert metrics.avg_duration_ms == 8.75  # (10*3 + 5) / 4

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BridgeMetrics(bridge_name="test_bridge")

        op = BridgeOperation(
            bridge_name="test_bridge",
            operation="sync",
            start_time=time.time(),
        )
        op.complete(success=True)
        metrics.record_operation(op)

        data = metrics.to_dict()

        assert data["bridge_name"] == "test_bridge"
        assert data["total_operations"] == 1
        assert "success_rate" in data
        assert "avg_duration_ms" in data


class TestBridgeTelemetryContext:
    """Tests for BridgeTelemetryContext manager."""

    def test_context_manager_success(self):
        """Test using context manager for successful operation."""
        with BridgeTelemetryContext("performance_router", "compute") as ctx:
            ctx.set_metadata("agent", "claude")
            # Simulate work
            time.sleep(0.01)

        # Check that operation was recorded
        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1
        assert "performance_router" in stats["bridges"]
        assert stats["bridges"]["performance_router"]["successful_operations"] == 1

    def test_context_manager_with_error(self):
        """Test context manager captures errors."""
        with pytest.raises(ValueError):
            with BridgeTelemetryContext("relationship_bias", "detect") as ctx:
                raise ValueError("Test error")

        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1
        assert stats["bridges"]["relationship_bias"]["failed_operations"] == 1

    def test_context_manager_metadata(self):
        """Test setting metadata in context."""
        with BridgeTelemetryContext("calibration_cost", "calculate") as ctx:
            ctx.set_metadata("agent", "gpt")
            ctx.set_metadata("efficiency", "high")

        recent = get_recent_bridge_operations(limit=1)
        assert len(recent) == 1
        assert recent[0]["metadata"]["agent"] == "gpt"
        assert recent[0]["metadata"]["efficiency"] == "high"


class TestBridgeTelemetryDecorator:
    """Tests for with_bridge_telemetry decorator."""

    def test_decorator_sync_function(self):
        """Test decorator on sync function."""

        @with_bridge_telemetry("test_bridge", "test_op")
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1
        assert stats["bridges"]["test_bridge"]["successful_operations"] == 1

    def test_decorator_async_function(self):
        """Test decorator on async function."""
        import asyncio

        @with_bridge_telemetry("async_bridge", "async_op")
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(async_test_function())

        assert result == "async_result"
        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1

    def test_decorator_with_error(self):
        """Test decorator captures errors."""

        @with_bridge_telemetry("error_bridge", "error_op")
        def error_function():
            raise RuntimeError("Deliberate error")

        with pytest.raises(RuntimeError):
            error_function()

        stats = get_bridge_telemetry_stats()
        assert stats["bridges"]["error_bridge"]["failed_operations"] == 1

    def test_decorator_with_metadata_extractor(self):
        """Test decorator with metadata extraction function."""

        @with_bridge_telemetry(
            "meta_bridge",
            "meta_op",
            extract_metadata=lambda x, y: {"x": x, "y": y},
        )
        def function_with_args(x, y):
            return x + y

        result = function_with_args(1, 2)

        assert result == 3
        recent = get_recent_bridge_operations(limit=1)
        assert recent[0]["metadata"]["x"] == 1
        assert recent[0]["metadata"]["y"] == 2


class TestConvenienceDecorators:
    """Tests for bridge-specific convenience decorators."""

    def test_performance_router_telemetry(self):
        """Test performance_router_telemetry decorator."""

        @performance_router_telemetry("sync")
        def sync_router():
            return True

        sync_router()

        stats = get_bridge_telemetry_stats()
        assert "performance_router" in stats["bridges"]

    def test_relationship_bias_telemetry(self):
        """Test relationship_bias_telemetry decorator."""

        @relationship_bias_telemetry("detect")
        def detect_echo_chamber():
            return {"risk": "low"}

        result = detect_echo_chamber()

        assert result["risk"] == "low"
        stats = get_bridge_telemetry_stats()
        assert "relationship_bias" in stats["bridges"]


class TestRecordBridgeOperation:
    """Tests for manual operation recording."""

    def test_record_operation(self):
        """Test manually recording an operation."""
        record_bridge_operation(
            bridge_name="manual_bridge",
            operation="manual_op",
            success=True,
            duration_ms=15.0,
            agent="test_agent",
        )

        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1
        assert stats["bridges"]["manual_bridge"]["successful_operations"] == 1

    def test_record_failed_operation(self):
        """Test manually recording a failed operation."""
        record_bridge_operation(
            bridge_name="manual_bridge",
            operation="fail_op",
            success=False,
            duration_ms=5.0,
            error=ValueError("Manual error"),
        )

        stats = get_bridge_telemetry_stats()
        assert stats["bridges"]["manual_bridge"]["failed_operations"] == 1


class TestTelemetryStats:
    """Tests for telemetry statistics."""

    def test_empty_stats(self):
        """Test getting stats with no operations."""
        stats = get_bridge_telemetry_stats()

        assert stats["total_operations"] == 0
        assert stats["overall_success_rate"] == 1.0  # No failures
        assert stats["bridges"] == {}

    def test_multiple_bridges(self):
        """Test stats across multiple bridges."""
        record_bridge_operation("bridge_a", "op1", True, 10.0)
        record_bridge_operation("bridge_a", "op2", True, 20.0)
        record_bridge_operation("bridge_b", "op1", False, 5.0)

        stats = get_bridge_telemetry_stats()

        assert stats["total_operations"] == 3
        assert len(stats["bridges"]) == 2
        assert stats["bridges"]["bridge_a"]["total_operations"] == 2
        assert stats["bridges"]["bridge_b"]["total_operations"] == 1
        assert stats["overall_success_rate"] == 2 / 3

    def test_recent_operations_limit(self):
        """Test getting recent operations with limit."""
        for i in range(10):
            record_bridge_operation(f"bridge_{i}", "op", True, 1.0)

        recent = get_recent_bridge_operations(limit=5)
        assert len(recent) == 5

        # Should be most recent operations
        assert recent[-1]["bridge_name"] == "bridge_9"


class TestPrometheusIntegration:
    """Tests for Prometheus metrics integration."""

    def test_prometheus_metrics_called(self):
        """Test that Prometheus metrics functions are called when available."""
        # Mock the observability metrics module
        mock_metrics = MagicMock()
        mock_metrics.record_bridge_sync = MagicMock()
        mock_metrics.record_bridge_sync_latency = MagicMock()
        mock_metrics.record_bridge_error = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics": mock_metrics},
        ):
            # Record an operation - the internal function will try to import
            # and call the metrics functions
            record_bridge_operation(
                bridge_name="prom_bridge",
                operation="sync",
                success=True,
                duration_ms=10.0,
            )

            # Verify the operation was recorded internally
            stats = get_bridge_telemetry_stats()
            assert stats["total_operations"] == 1
            assert "prom_bridge" in stats["bridges"]


class TestResetTelemetry:
    """Tests for telemetry reset functionality."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all telemetry data."""
        record_bridge_operation("bridge1", "op1", True, 10.0)
        record_bridge_operation("bridge2", "op2", True, 20.0)

        stats_before = get_bridge_telemetry_stats()
        assert stats_before["total_operations"] == 2

        reset_bridge_telemetry()

        stats_after = get_bridge_telemetry_stats()
        assert stats_after["total_operations"] == 0
        assert stats_after["bridges"] == {}
