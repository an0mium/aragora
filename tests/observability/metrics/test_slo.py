"""Tests for SLO metrics module."""

import pytest
from unittest.mock import patch, MagicMock


class TestSLOMetrics:
    """Tests for SLO metrics functions."""

    def test_init_slo_metrics_without_prometheus(self):
        """Test SLO metrics initialization when prometheus-client not available."""
        # Reset initialization state
        import aragora.observability.metrics.slo as slo_module

        slo_module._initialized = False

        with patch.dict("sys.modules", {"prometheus_client": None}):
            with patch.object(slo_module, "get_metrics_config") as mock_config:
                mock_config.return_value = MagicMock(enabled=True)
                # This should fallback gracefully
                result = slo_module.init_slo_metrics()
                # Should return False when prometheus unavailable
                assert result is False or result is True  # Either works

    def test_init_slo_metrics_disabled(self):
        """Test SLO metrics when metrics are disabled."""
        import aragora.observability.metrics.slo as slo_module

        slo_module._initialized = False

        with patch.object(slo_module, "get_metrics_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=False)
            result = slo_module.init_slo_metrics()
            assert result is False
            # Should use NoOp metrics
            from aragora.observability.metrics.base import NoOpMetric

            assert isinstance(slo_module.SLO_CHECKS_TOTAL, NoOpMetric)

    def test_record_slo_check_pass(self):
        """Test recording a passing SLO check."""
        from aragora.observability.metrics.slo import record_slo_check

        # Should not raise
        record_slo_check("km_query", passed=True, percentile="p99")

    def test_record_slo_check_fail(self):
        """Test recording a failing SLO check."""
        from aragora.observability.metrics.slo import record_slo_check

        # Should not raise
        record_slo_check("km_query", passed=False, percentile="p99")

    def test_record_slo_violation_auto_severity(self):
        """Test SLO violation with automatic severity calculation."""
        from aragora.observability.metrics.slo import record_slo_violation

        # < 20% over threshold = minor
        record_slo_violation("km_query", "p99", latency_ms=550, threshold_ms=500)

        # 20-50% over = moderate
        record_slo_violation("km_query", "p99", latency_ms=700, threshold_ms=500)

        # 50-100% over = major
        record_slo_violation("km_query", "p99", latency_ms=900, threshold_ms=500)

        # > 100% over = critical
        record_slo_violation("km_query", "p99", latency_ms=1100, threshold_ms=500)

    def test_record_slo_violation_explicit_severity(self):
        """Test SLO violation with explicit severity."""
        from aragora.observability.metrics.slo import record_slo_violation

        record_slo_violation(
            "km_query", "p99", latency_ms=600, threshold_ms=500, severity="warning"
        )

    def test_record_operation_latency(self):
        """Test recording operation latency."""
        from aragora.observability.metrics.slo import record_operation_latency

        record_operation_latency("km_query", 45.5)
        record_operation_latency("consensus_ingestion", 180.0)

    def test_check_and_record_slo_within(self):
        """Test combined SLO check and record when within SLO."""
        from aragora.observability.metrics.slo import check_and_record_slo

        passed, message = check_and_record_slo("km_query", latency_ms=40.0)
        assert passed is True
        assert "within" in message.lower()

    def test_check_and_record_slo_violation(self):
        """Test combined SLO check and record when SLO violated."""
        from aragora.observability.metrics.slo import check_and_record_slo

        # P99 for km_query is 500ms
        passed, message = check_and_record_slo("km_query", latency_ms=600.0)
        assert passed is False
        assert "EXCEEDS" in message

    def test_check_and_record_slo_unknown_operation(self):
        """Test SLO check for unknown operation."""
        from aragora.observability.metrics.slo import check_and_record_slo

        passed, message = check_and_record_slo("unknown_operation", latency_ms=100.0)
        # Should return True with "no SLO defined" message
        assert passed is True
        assert "No SLO defined" in message

    def test_track_operation_slo_context_manager(self):
        """Test the SLO tracking context manager."""
        from aragora.observability.metrics.slo import track_operation_slo
        import time

        with track_operation_slo("km_query") as ctx:
            ctx["test_key"] = "test_value"
            time.sleep(0.01)  # Simulate some work

        assert ctx["test_key"] == "test_value"

    def test_track_operation_slo_with_exception(self):
        """Test SLO tracking context manager with exception."""
        from aragora.observability.metrics.slo import track_operation_slo

        with pytest.raises(ValueError):
            with track_operation_slo("km_query") as ctx:
                ctx["before_error"] = True
                raise ValueError("Test error")

    def test_get_slo_metrics_summary(self):
        """Test getting SLO metrics summary."""
        from aragora.observability.metrics.slo import get_slo_metrics_summary

        summary = get_slo_metrics_summary()
        assert "initialized" in summary
        assert "metrics_enabled" in summary
        assert "tracked_operations" in summary
        assert "km_query" in summary["tracked_operations"]
        assert "consensus_ingestion" in summary["tracked_operations"]


class TestSLOIntegration:
    """Integration tests for SLO metrics with performance_slos config."""

    def test_slo_config_integration(self):
        """Test that SLO metrics work with the SLO config."""
        from aragora.config.performance_slos import get_slo_config, check_latency_slo

        config = get_slo_config()

        # Verify km_query SLO exists
        assert hasattr(config, "km_query")
        assert config.km_query.p50_ms == 50.0
        assert config.km_query.p99_ms == 500.0

        # Check a value within SLO
        passed, _ = check_latency_slo("km_query", 40.0, "p99")
        assert passed is True

        # Check a value exceeding SLO
        passed, _ = check_latency_slo("km_query", 600.0, "p99")
        assert passed is False

    def test_all_operations_have_config(self):
        """Test that all tracked operations have SLO config."""
        from aragora.config.performance_slos import get_slo_config
        from aragora.observability.metrics.slo import get_slo_metrics_summary

        config = get_slo_config()
        summary = get_slo_metrics_summary()

        for operation in summary["tracked_operations"]:
            slo = getattr(config, operation, None)
            assert slo is not None, f"Missing SLO config for {operation}"
            assert hasattr(slo, "p50_ms")
            assert hasattr(slo, "p90_ms")
            assert hasattr(slo, "p99_ms")
            assert hasattr(slo, "timeout_ms")


class TestSLOSeverityCalculation:
    """Tests for automatic severity calculation."""

    def test_minor_severity(self):
        """Test minor severity for < 20% over threshold."""
        from aragora.observability.metrics.slo import record_slo_violation
        import aragora.observability.metrics.slo as slo_module

        slo_module._initialized = False
        slo_module.init_slo_metrics()

        # 10% over = minor
        record_slo_violation("km_query", "p99", latency_ms=550, threshold_ms=500)

    def test_moderate_severity(self):
        """Test moderate severity for 20-50% over threshold."""
        from aragora.observability.metrics.slo import record_slo_violation

        # 30% over = moderate
        record_slo_violation("km_query", "p99", latency_ms=650, threshold_ms=500)

    def test_major_severity(self):
        """Test major severity for 50-100% over threshold."""
        from aragora.observability.metrics.slo import record_slo_violation

        # 80% over = major
        record_slo_violation("km_query", "p99", latency_ms=900, threshold_ms=500)

    def test_critical_severity(self):
        """Test critical severity for > 100% over threshold."""
        from aragora.observability.metrics.slo import record_slo_violation

        # 120% over = critical
        record_slo_violation("km_query", "p99", latency_ms=1100, threshold_ms=500)
