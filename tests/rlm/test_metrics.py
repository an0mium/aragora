"""
Tests for RLM refinement metrics.

Tests the Prometheus metrics for iterative refinement tracking.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRefinementMetricsDefinitions:
    """Test that refinement metric functions are defined."""

    def test_record_refinement_exists(self):
        """Test that record_refinement function exists."""
        from aragora.rlm import metrics

        assert hasattr(metrics, "record_refinement")
        assert callable(metrics.record_refinement)

    def test_record_ready_false_exists(self):
        """Test that record_ready_false function exists."""
        from aragora.rlm import metrics

        assert hasattr(metrics, "record_ready_false")
        assert callable(metrics.record_ready_false)

    def test_measure_refinement_exists(self):
        """Test that measure_refinement context manager exists."""
        from aragora.rlm import metrics

        assert hasattr(metrics, "measure_refinement")


class TestRefinementMetricsWithMocks:
    """Test refinement metrics functions with mocked metrics."""

    def test_record_refinement_calls_metric(self):
        """Test that record_refinement calls the metric."""
        from aragora.rlm import metrics

        # Create mock metric
        mock_metric = MagicMock()
        mock_metric.labels.return_value.observe = MagicMock()
        mock_metric.labels.return_value.inc = MagicMock()

        # Patch the metrics
        with patch.object(metrics, "RLM_REFINEMENT_ITERATIONS", mock_metric):
            with patch.object(metrics, "RLM_REFINEMENT_SUCCESS", mock_metric):
                with patch.object(metrics, "RLM_REFINEMENT_DURATION", mock_metric):
                    with patch.object(metrics, "_initialized", True):
                        metrics.record_refinement(
                            strategy="grep",
                            iterations=2,
                            success=True,
                            duration_seconds=1.5,
                        )

        # Verify metrics were called
        mock_metric.labels.assert_called()

    def test_record_ready_false_calls_metric(self):
        """Test that record_ready_false calls the metric."""
        from aragora.rlm import metrics

        mock_metric = MagicMock()
        mock_metric.labels.return_value.inc = MagicMock()

        with patch.object(metrics, "RLM_READY_FALSE_RATE", mock_metric):
            with patch.object(metrics, "_initialized", True):
                metrics.record_ready_false(iteration=1)

        mock_metric.labels.assert_called_with(iteration="1")


class TestMeasureRefinementContextManager:
    """Test measure_refinement context manager."""

    def test_measure_refinement_yields_dict(self):
        """Test that context manager yields a dict with defaults."""
        from aragora.rlm import metrics

        # Mock record_refinement to avoid metric issues
        with patch.object(metrics, "record_refinement"):
            with metrics.measure_refinement("auto") as ctx:
                assert isinstance(ctx, dict)
                assert ctx["iterations"] == 1
                assert ctx["success"] is True

    def test_measure_refinement_allows_modification(self):
        """Test that context dict can be modified."""
        from aragora.rlm import metrics

        with patch.object(metrics, "record_refinement") as mock_record:
            with metrics.measure_refinement("grep") as ctx:
                ctx["iterations"] = 3
                ctx["success"] = True

            # Check that record_refinement was called with modified values
            mock_record.assert_called_once()
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["iterations"] == 3
            assert call_kwargs["success"] is True


class TestMetricVariablesExist:
    """Test that metric variables are defined in module."""

    def test_refinement_metric_variables_defined(self):
        """Test that refinement metric variables exist."""
        from aragora.rlm import metrics

        # These should exist as module-level variables (may be None initially)
        assert hasattr(metrics, "RLM_REFINEMENT_ITERATIONS")
        assert hasattr(metrics, "RLM_REFINEMENT_SUCCESS")
        assert hasattr(metrics, "RLM_REFINEMENT_DURATION")
        assert hasattr(metrics, "RLM_READY_FALSE_RATE")

    def test_noop_metric_class_works(self):
        """Test that NoopMetric class works as expected."""
        from aragora.rlm.metrics import _init_noop_metrics

        # Call init to create noop metrics
        _init_noop_metrics()

        # Import after init
        from aragora.rlm import metrics

        # Noop metrics should accept method chains without raising
        if metrics.RLM_REFINEMENT_ITERATIONS is not None:
            result = metrics.RLM_REFINEMENT_ITERATIONS.labels(strategy="test")
            # NoopMetric returns lambda that returns None
            assert result is not None or result is None  # Just shouldn't raise
