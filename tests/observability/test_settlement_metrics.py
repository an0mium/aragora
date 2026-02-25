"""Tests for settlement, calibration, and intervention observability metrics."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.metrics import settlement as mod


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset settlement metrics module state between tests."""
    mod._initialized = False
    mod.SETTLEMENT_CAPTURED_TOTAL = None
    mod.SETTLEMENT_REVIEW_DUE_TOTAL = None
    mod.SETTLEMENT_STATUS_TRANSITIONS = None
    mod.SETTLEMENT_CONFIDENCE = None
    mod.SETTLEMENT_FALSIFIER_COUNT = None
    mod.CALIBRATION_BRIER_SCORE = None
    mod.CALIBRATION_OUTCOMES_TOTAL = None
    mod.INTERVENTION_TOTAL = None
    mod.INTERVENTION_PAUSE_DURATION = None
    yield


class TestInitSettlementMetrics:
    """Tests for metrics initialization."""

    def test_init_creates_noop_when_disabled(self):
        """When metrics are disabled, all metrics are NoOpMetric."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        assert mod._initialized is True
        from aragora.observability.metrics.base import NoOpMetric

        assert isinstance(mod.SETTLEMENT_CAPTURED_TOTAL, NoOpMetric)
        assert isinstance(mod.CALIBRATION_BRIER_SCORE, NoOpMetric)
        assert isinstance(mod.INTERVENTION_TOTAL, NoOpMetric)

    def test_init_idempotent(self):
        """Calling init twice does not re-initialize."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
            first = mod.SETTLEMENT_CAPTURED_TOTAL
            mod.init_settlement_metrics()
            assert mod.SETTLEMENT_CAPTURED_TOTAL is first

    def test_init_with_prometheus_import_error(self):
        """Falls back to noop on ImportError."""
        with (
            patch.object(mod, "get_metrics_enabled", return_value=True),
            patch.dict("sys.modules", {"prometheus_client": None}),
        ):
            # Force reimport to trigger ImportError path
            mod._initialized = False
            mod.init_settlement_metrics()
        assert mod._initialized is True
        from aragora.observability.metrics.base import NoOpMetric

        assert isinstance(mod.SETTLEMENT_CAPTURED_TOTAL, NoOpMetric)


class TestSettlementRecording:
    """Tests for settlement metric recording functions."""

    def test_record_settlement_captured(self):
        """Records a settlement capture event."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        # NoOp metrics accept any call without error
        mod.record_settlement_captured("settled")
        mod.record_settlement_captured("pending")

    def test_record_settlement_review_due(self):
        """Records the due review gauge."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_settlement_review_due(5)

    def test_record_settlement_transition(self):
        """Records status transitions."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_settlement_transition("settled", "due_review")
        mod.record_settlement_transition("due_review", "confirmed")

    def test_record_settlement_confidence(self):
        """Records confidence score histogram."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_settlement_confidence(0.85)

    def test_record_settlement_falsifiers(self):
        """Records falsifier count histogram."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_settlement_falsifiers(3)


class TestCalibrationRecording:
    """Tests for calibration metric recording functions."""

    def test_record_calibration_brier(self):
        """Records Brier score for an agent."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_calibration_brier("claude", 0.15)
        mod.record_calibration_brier("gpt4", 0.22)

    def test_record_calibration_outcome(self):
        """Records calibration outcome types."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_calibration_outcome("correct")
        mod.record_calibration_outcome("incorrect")


class TestInterventionRecording:
    """Tests for intervention metric recording functions."""

    def test_record_intervention(self):
        """Records operator intervention actions."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_intervention("pause")
        mod.record_intervention("resume")
        mod.record_intervention("restart")

    def test_record_intervention_pause_duration(self):
        """Records pause duration."""
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.init_settlement_metrics()
        mod.record_intervention_pause_duration(30.5)


class TestAutoInit:
    """Tests for _ensure_init lazy initialization."""

    def test_ensure_init_called_on_record(self):
        """Recording functions auto-init if not already initialized."""
        assert mod._initialized is False
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.record_settlement_captured()
        assert mod._initialized is True

    def test_ensure_init_called_on_calibration_record(self):
        """Calibration recording auto-inits."""
        assert mod._initialized is False
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.record_calibration_brier("agent", 0.1)
        assert mod._initialized is True

    def test_ensure_init_called_on_intervention_record(self):
        """Intervention recording auto-inits."""
        assert mod._initialized is False
        with patch.object(mod, "get_metrics_enabled", return_value=False):
            mod.record_intervention("pause")
        assert mod._initialized is True


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """All items in __all__ are actually defined in the module."""
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_key_metrics_in_all(self):
        """Key metrics are exported."""
        expected = {
            "SETTLEMENT_CAPTURED_TOTAL",
            "CALIBRATION_BRIER_SCORE",
            "INTERVENTION_TOTAL",
            "init_settlement_metrics",
            "record_settlement_captured",
            "record_calibration_brier",
            "record_intervention",
        }
        assert expected.issubset(set(mod.__all__))
