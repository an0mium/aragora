"""
Tests for CalibrationHandler endpoints.

Endpoints tested:
- GET /api/agent/{name}/calibration-curve - Get calibration curve
- GET /api/agent/{name}/calibration-summary - Get calibration summary metrics
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.agents import CalibrationHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_calibration_bucket():
    """Create a mock calibration bucket."""

    def create_bucket(range_start, range_end, total, correct):
        bucket = Mock()
        bucket.range_start = range_start
        bucket.range_end = range_end
        bucket.total_predictions = total
        bucket.correct_predictions = correct
        bucket.accuracy = correct / total if total > 0 else 0
        bucket.brier_score = 0.15
        return bucket

    return create_bucket


@pytest.fixture
def mock_calibration_summary():
    """Create a mock calibration summary."""
    summary = Mock()
    summary.agent = "claude"
    summary.total_predictions = 100
    summary.total_correct = 75
    summary.accuracy = 0.75
    summary.brier_score = 0.18
    summary.ece = 0.05  # Expected Calibration Error
    summary.is_overconfident = False
    summary.is_underconfident = True
    return summary


@pytest.fixture
def mock_calibration_tracker(mock_calibration_bucket, mock_calibration_summary):
    """Create a mock CalibrationTracker."""
    tracker = Mock()

    # Create calibration curve with 5 buckets
    curve = [
        mock_calibration_bucket(0.0, 0.2, 20, 3),
        mock_calibration_bucket(0.2, 0.4, 25, 8),
        mock_calibration_bucket(0.4, 0.6, 30, 15),
        mock_calibration_bucket(0.6, 0.8, 15, 11),
        mock_calibration_bucket(0.8, 1.0, 10, 9),
    ]
    tracker.get_calibration_curve.return_value = curve
    tracker.get_calibration_summary.return_value = mock_calibration_summary
    return tracker


@pytest.fixture
def calibration_handler():
    """Create a CalibrationHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return CalibrationHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestCalibrationRouting:
    """Tests for route matching."""

    def test_can_handle_calibration_curve(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/claude/calibration-curve") is True

    def test_can_handle_calibration_summary(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/claude/calibration-summary") is True

    def test_can_handle_with_hyphenated_agent(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/gpt-4/calibration-curve") is True
        assert calibration_handler.can_handle("/api/v1/agent/gpt-4/calibration-summary") is True

    def test_cannot_handle_unrelated_routes(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/claude") is False
        assert calibration_handler.can_handle("/api/v1/agent/claude/persona") is False
        assert calibration_handler.can_handle("/api/v1/calibration") is False
        assert calibration_handler.can_handle("/api/v1/agents/claude/calibration-curve") is False


# ============================================================================
# GET /api/agent/{name}/calibration-curve Tests
# ============================================================================


class TestCalibrationCurve:
    """Tests for GET /api/agent/{name}/calibration-curve endpoint."""

    def test_calibration_curve_module_unavailable(self, calibration_handler):
        import aragora.server.handlers.calibration as mod

        original = mod.CALIBRATION_AVAILABLE
        mod.CALIBRATION_AVAILABLE = False
        try:
            result = calibration_handler.handle("/api/agent/claude/calibration-curve", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.CALIBRATION_AVAILABLE = original

    def test_calibration_curve_success(self, calibration_handler, mock_calibration_tracker):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle("/api/agent/claude/calibration-curve", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["agent"] == "claude"
            assert "buckets" in data
            assert len(data["buckets"]) == 5
            assert data["count"] == 5

    def test_calibration_curve_with_buckets_param(
        self, calibration_handler, mock_calibration_tracker
    ):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-curve", {"buckets": "15"}, None
            )

            assert result is not None
            assert result.status_code == 200
            # Verify tracker was called with correct buckets
            mock_calibration_tracker.get_calibration_curve.assert_called_with(
                "claude", num_buckets=15, domain=None
            )

    def test_calibration_curve_buckets_clamped_min(
        self, calibration_handler, mock_calibration_tracker
    ):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            # Request 3 buckets, should be clamped to 5
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-curve", {"buckets": "3"}, None
            )

            assert result is not None
            assert result.status_code == 200
            mock_calibration_tracker.get_calibration_curve.assert_called_with(
                "claude", num_buckets=5, domain=None
            )

    def test_calibration_curve_buckets_clamped_max(
        self, calibration_handler, mock_calibration_tracker
    ):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            # Request 50 buckets, should be clamped to 20
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-curve", {"buckets": "50"}, None
            )

            assert result is not None
            assert result.status_code == 200
            mock_calibration_tracker.get_calibration_curve.assert_called_with(
                "claude", num_buckets=20, domain=None
            )

    def test_calibration_curve_with_domain(self, calibration_handler, mock_calibration_tracker):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-curve", {"domain": "coding"}, None
            )

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["domain"] == "coding"

    def test_calibration_curve_invalid_agent(self, calibration_handler):
        result = calibration_handler.handle("/api/agent/test..admin/calibration-curve", {}, None)
        assert result is not None
        assert result.status_code == 400


# ============================================================================
# GET /api/agent/{name}/calibration-summary Tests
# ============================================================================


class TestCalibrationSummary:
    """Tests for GET /api/agent/{name}/calibration-summary endpoint."""

    def test_calibration_summary_module_unavailable(self, calibration_handler):
        import aragora.server.handlers.calibration as mod

        original = mod.CALIBRATION_AVAILABLE
        mod.CALIBRATION_AVAILABLE = False
        try:
            result = calibration_handler.handle("/api/agent/claude/calibration-summary", {}, None)
            assert result is not None
            assert result.status_code == 503
        finally:
            mod.CALIBRATION_AVAILABLE = original

    def test_calibration_summary_success(self, calibration_handler, mock_calibration_tracker):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle("/api/agent/claude/calibration-summary", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["agent"] == "claude"
            assert data["total_predictions"] == 100
            assert data["total_correct"] == 75
            assert data["accuracy"] == 0.75
            assert "brier_score" in data
            assert "ece" in data
            assert "is_overconfident" in data
            assert "is_underconfident" in data

    def test_calibration_summary_with_domain(self, calibration_handler, mock_calibration_tracker):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-summary", {"domain": "reasoning"}, None
            )

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["domain"] == "reasoning"

    def test_calibration_summary_invalid_agent(self, calibration_handler):
        result = calibration_handler.handle("/api/agent/<script>/calibration-summary", {}, None)
        assert result is not None
        assert result.status_code == 400


# ============================================================================
# Security Tests
# ============================================================================


class TestCalibrationSecurity:
    """Security tests for calibration endpoints."""

    def test_path_traversal_blocked_curve(self, calibration_handler):
        result = calibration_handler.handle("/api/agent/test..admin/calibration-curve", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_summary(self, calibration_handler):
        result = calibration_handler.handle(
            "/api/agent/..%2F..%2Fetc/calibration-summary", {}, None
        )
        assert result.status_code == 400

    def test_sql_injection_blocked(self, calibration_handler):
        result = calibration_handler.handle(
            "/api/agent/'; DROP TABLE agents;--/calibration-curve", {}, None
        )
        assert result.status_code == 400

    def test_xss_blocked(self, calibration_handler):
        result = calibration_handler.handle(
            "/api/agent/<img src=x onerror=alert(1)>/calibration-summary", {}, None
        )
        assert result.status_code == 400


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCalibrationErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, calibration_handler):
        result = calibration_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_calibration_curve_exception(self, calibration_handler):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        mock_tracker = Mock()
        mock_tracker.get_calibration_curve.side_effect = Exception("Database error")

        with patch.object(mod, "CalibrationTracker", return_value=mock_tracker):
            result = calibration_handler.handle("/api/agent/claude/calibration-curve", {}, None)

            assert result is not None
            assert result.status_code == 500

    def test_calibration_summary_exception(self, calibration_handler):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        mock_tracker = Mock()
        mock_tracker.get_calibration_summary.side_effect = Exception("Database error")

        with patch.object(mod, "CalibrationTracker", return_value=mock_tracker):
            result = calibration_handler.handle("/api/agent/claude/calibration-summary", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================


class TestCalibrationEdgeCases:
    """Tests for edge cases."""

    def test_empty_agent_name(self, calibration_handler):
        result = calibration_handler.handle("/api/agent//calibration-curve", {}, None)
        # Empty name should fail validation or return None
        assert result is None or result.status_code == 400

    def test_very_long_agent_name(self, calibration_handler):
        long_name = "a" * 1000
        result = calibration_handler.handle(f"/api/agent/{long_name}/calibration-curve", {}, None)
        # Should handle gracefully
        assert result is not None

    def test_unicode_agent_name(self, calibration_handler):
        result = calibration_handler.handle("/api/agent/测试/calibration-curve", {}, None)
        # Should either accept or reject gracefully
        assert result is not None

    def test_calibration_curve_bucket_data_structure(
        self, calibration_handler, mock_calibration_tracker
    ):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            result = calibration_handler.handle("/api/agent/claude/calibration-curve", {}, None)

            assert result is not None
            data = json.loads(result.body)
            # Verify bucket structure
            bucket = data["buckets"][0]
            assert "range_start" in bucket
            assert "range_end" in bucket
            assert "total_predictions" in bucket
            assert "correct_predictions" in bucket
            assert "accuracy" in bucket
            assert "expected_accuracy" in bucket
            assert "brier_score" in bucket

    def test_invalid_buckets_param(self, calibration_handler, mock_calibration_tracker):
        import aragora.server.handlers.calibration as mod

        if not mod.CALIBRATION_AVAILABLE:
            pytest.skip("Calibration module not available")

        with patch.object(mod, "CalibrationTracker", return_value=mock_calibration_tracker):
            # Invalid buckets should default to 10
            result = calibration_handler.handle(
                "/api/agent/claude/calibration-curve", {"buckets": "invalid"}, None
            )

            assert result is not None
            assert result.status_code == 200
