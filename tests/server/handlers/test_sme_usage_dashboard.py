"""
Tests for SME Usage Dashboard API Handler.

Tests coverage for:
- GET /api/v1/usage/summary - Unified usage metrics
- GET /api/v1/usage/breakdown - Detailed breakdown
- GET /api/v1/usage/roi - ROI analysis
- GET /api/v1/usage/export - CSV/JSON export
- GET /api/v1/usage/budget-status - Budget utilization
- GET /api/v1/usage/forecast - Usage forecast
- GET /api/v1/usage/benchmarks - Industry benchmarks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest


class TestSMEUsageDashboardHandler:
    """Tests for SMEUsageDashboardHandler."""

    @pytest.fixture
    def dashboard_handler(self, mock_server_context):
        """Create SMEUsageDashboardHandler with mocked context."""
        from aragora.server.handlers.sme_usage_dashboard import SMEUsageDashboardHandler

        return SMEUsageDashboardHandler(mock_server_context)

    def test_can_handle_valid_routes(self, dashboard_handler):
        """Handler recognizes valid routes."""
        assert dashboard_handler.can_handle("/api/v1/usage/summary") is True
        assert dashboard_handler.can_handle("/api/v1/usage/breakdown") is True
        assert dashboard_handler.can_handle("/api/v1/usage/roi") is True
        assert dashboard_handler.can_handle("/api/v1/usage/export") is True
        assert dashboard_handler.can_handle("/api/v1/usage/budget-status") is True
        assert dashboard_handler.can_handle("/api/v1/usage/forecast") is True
        assert dashboard_handler.can_handle("/api/v1/usage/benchmarks") is True

    def test_can_handle_invalid_routes(self, dashboard_handler):
        """Handler rejects invalid routes."""
        assert dashboard_handler.can_handle("/api/v1/billing/usage") is False
        assert dashboard_handler.can_handle("/api/v1/debates") is False
        assert dashboard_handler.can_handle("/usage/summary") is False
        assert dashboard_handler.can_handle("/api/v2/usage/summary") is False


class TestSMEUsageDashboardRouting:
    """Tests for request routing."""

    @pytest.fixture
    def dashboard_handler(self, mock_server_context):
        """Create handler with mocked context."""
        from aragora.server.handlers.sme_usage_dashboard import SMEUsageDashboardHandler

        return SMEUsageDashboardHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self, mock_http_handler):
        """Create mock HTTP handler."""
        return mock_http_handler(method="GET")

    def test_handle_routes_to_summary(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/summary to _get_summary."""
        with patch.object(
            dashboard_handler, "_get_summary", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/summary", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_breakdown(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/breakdown to _get_breakdown."""
        with patch.object(
            dashboard_handler, "_get_breakdown", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/breakdown", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_roi(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/roi to _get_roi."""
        with patch.object(
            dashboard_handler, "_get_roi", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/roi", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_budget_status(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/budget-status to _get_budget_status."""
        with patch.object(
            dashboard_handler,
            "_get_budget_status",
            return_value=MagicMock(status_code=200),
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/budget-status", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_forecast(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/forecast to _get_forecast."""
        with patch.object(
            dashboard_handler, "_get_forecast", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/forecast", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_benchmarks(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/benchmarks to _get_benchmarks."""
        with patch.object(
            dashboard_handler, "_get_benchmarks", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/benchmarks", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_export(self, dashboard_handler, mock_http):
        """handle() routes /api/v1/usage/export to _export_usage."""
        with patch.object(
            dashboard_handler, "_export_usage", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch(
                "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                dashboard_handler.handle("/api/v1/usage/export", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_rate_limit_exceeded(self, dashboard_handler, mock_http):
        """handle() returns 429 when rate limit exceeded."""
        with patch(
            "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = dashboard_handler.handle("/api/v1/usage/summary", {}, mock_http, "GET")
            assert result.status_code == 429

    def test_handle_method_not_allowed(self, dashboard_handler):
        """handle() returns 405 for unsupported method."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"
        mock_http.client_address = ("127.0.0.1", 12345)
        mock_http.headers = {}

        with patch(
            "aragora.server.handlers.sme_usage_dashboard._dashboard_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = dashboard_handler.handle("/api/v1/usage/summary", {}, mock_http, "DELETE")
            assert result.status_code == 405


class TestROICalculator:
    """Tests for ROI Calculator module."""

    def test_roi_calculator_initialization(self):
        """ROI calculator initializes with defaults."""
        from aragora.billing.roi_calculator import ROICalculator, IndustryBenchmark

        calc = ROICalculator()
        assert calc._benchmark == IndustryBenchmark.SME

    def test_roi_calculator_with_benchmark(self):
        """ROI calculator accepts different benchmarks."""
        from aragora.billing.roi_calculator import ROICalculator, IndustryBenchmark

        calc = ROICalculator(benchmark=IndustryBenchmark.ENTERPRISE)
        assert calc._benchmark == IndustryBenchmark.ENTERPRISE

    def test_calculate_single_debate_roi(self):
        """Single debate ROI calculation works."""
        from aragora.billing.roi_calculator import ROICalculator, DebateROIInput

        calc = ROICalculator()
        debate = DebateROIInput(
            debate_id="test-1",
            duration_seconds=300,  # 5 minutes
            cost_usd=Decimal("0.50"),
            reached_consensus=True,
            confidence_score=0.85,
            agent_count=3,
            round_count=3,
            completed=True,
        )

        result = calc.calculate_single_debate_roi(debate)

        assert "debate_id" in result
        assert result["debate_id"] == "test-1"
        assert "cost_savings_usd" in result
        assert "hours_saved" in result
        assert "roi_percentage" in result
        assert result["consensus_achieved"] is True

    def test_calculate_period_roi(self):
        """Period ROI calculation aggregates correctly."""
        from aragora.billing.roi_calculator import ROICalculator, DebateROIInput

        calc = ROICalculator()
        debates = [
            DebateROIInput(
                debate_id=f"test-{i}",
                duration_seconds=300,
                cost_usd=Decimal("0.50"),
                reached_consensus=True,
                confidence_score=0.85,
                completed=True,
            )
            for i in range(5)
        ]

        now = datetime.now(timezone.utc)
        metrics = calc.calculate_period_roi(
            debates=debates,
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert metrics.total_debates == 5
        assert metrics.completed_debates == 5
        assert metrics.consensus_reached_count == 5
        assert metrics.total_aragora_cost_usd == Decimal("2.50")
        assert metrics.consensus_rate == 1.0

    def test_roi_metrics_to_dict(self):
        """ROI metrics serialization works."""
        from aragora.billing.roi_calculator import ROIMetrics

        metrics = ROIMetrics(
            total_debates=10,
            completed_debates=8,
            consensus_rate=0.75,
            roi_percentage=150.0,
        )

        result = metrics.to_dict()

        assert result["usage"]["total_debates"] == 10
        assert result["usage"]["completed_debates"] == 8
        assert result["quality"]["consensus_rate"] == 75.0
        assert result["roi"]["roi_percentage"] == 150.0

    def test_estimate_future_savings(self):
        """Future savings estimation works."""
        from aragora.billing.roi_calculator import ROICalculator

        calc = ROICalculator()
        projections = calc.estimate_future_savings(
            projected_debates_per_month=50,
            current_cost_per_debate=Decimal("0.50"),
        )

        assert "projections" in projections
        assert "monthly" in projections["projections"]
        assert "annual" in projections["projections"]
        assert "assumptions" in projections

    def test_get_benchmark_comparison(self):
        """Benchmark comparison returns all benchmarks."""
        from aragora.billing.roi_calculator import ROICalculator

        calc = ROICalculator()
        comparison = calc.get_benchmark_comparison()

        assert "benchmarks" in comparison
        assert "current_selection" in comparison
        assert "sme" in comparison["benchmarks"]
        assert "enterprise" in comparison["benchmarks"]
        assert "tech_startup" in comparison["benchmarks"]
        assert "consulting" in comparison["benchmarks"]


class TestROICalculatorBenchmarks:
    """Tests for industry benchmark data."""

    def test_all_benchmarks_have_required_fields(self):
        """All benchmarks have required data fields."""
        from aragora.billing.roi_calculator import BENCHMARK_COSTS, IndustryBenchmark

        required_fields = [
            "avg_decision_cost_usd",
            "avg_hours_per_decision",
            "avg_participants",
            "hourly_rate_usd",
        ]

        for benchmark in IndustryBenchmark:
            data = BENCHMARK_COSTS[benchmark]
            for field_name in required_fields:
                assert field_name in data, f"Missing {field_name} in {benchmark}"

    def test_benchmark_values_are_reasonable(self):
        """Benchmark values are within reasonable ranges."""
        from aragora.billing.roi_calculator import BENCHMARK_COSTS, IndustryBenchmark

        for benchmark in IndustryBenchmark:
            data = BENCHMARK_COSTS[benchmark]
            assert data["avg_hours_per_decision"] >= 1
            assert data["avg_hours_per_decision"] <= 10
            assert data["avg_participants"] >= 1
            assert data["avg_participants"] <= 10
            assert data["hourly_rate_usd"] >= Decimal("20")
            assert data["hourly_rate_usd"] <= Decimal("500")


class TestROICalculatorEdgeCases:
    """Tests for edge cases in ROI calculations."""

    def test_empty_debates_list(self):
        """Period ROI handles empty debate list."""
        from aragora.billing.roi_calculator import ROICalculator

        calc = ROICalculator()
        metrics = calc.calculate_period_roi(debates=[])

        assert metrics.total_debates == 0
        assert metrics.completed_debates == 0
        assert metrics.roi_percentage == 0.0

    def test_zero_cost_debate(self):
        """Single debate ROI handles zero cost."""
        from aragora.billing.roi_calculator import ROICalculator, DebateROIInput

        calc = ROICalculator()
        debate = DebateROIInput(
            debate_id="free-1",
            cost_usd=Decimal("0"),
            completed=True,
        )

        result = calc.calculate_single_debate_roi(debate)
        # With zero cost, ROI should be calculated as 0 (can't divide by zero)
        assert result["roi_percentage"] == 0.0

    def test_incomplete_debates_filtered(self):
        """Only completed debates are counted in period ROI."""
        from aragora.billing.roi_calculator import ROICalculator, DebateROIInput

        calc = ROICalculator()
        debates = [
            DebateROIInput(debate_id="complete-1", completed=True),
            DebateROIInput(debate_id="incomplete-1", completed=False),
            DebateROIInput(debate_id="complete-2", completed=True),
        ]

        metrics = calc.calculate_period_roi(debates=debates)

        assert metrics.total_debates == 3
        assert metrics.completed_debates == 2

    def test_custom_hourly_rate_override(self):
        """Hourly rate override affects calculations."""
        from aragora.billing.roi_calculator import ROICalculator, DebateROIInput

        calc_default = ROICalculator()
        calc_custom = ROICalculator(hourly_rate_override=Decimal("200"))

        debate = DebateROIInput(
            debate_id="test-1",
            duration_seconds=300,
            cost_usd=Decimal("1.00"),
            completed=True,
        )

        result_default = calc_default.calculate_single_debate_roi(debate)
        result_custom = calc_custom.calculate_single_debate_roi(debate)

        # Custom rate should produce higher manual equivalent cost
        default_manual = Decimal(result_default["manual_equivalent_cost_usd"])
        custom_manual = Decimal(result_custom["manual_equivalent_cost_usd"])
        assert custom_manual > default_manual


class TestGlobalROICalculator:
    """Tests for global calculator instance."""

    def test_get_roi_calculator_returns_instance(self):
        """get_roi_calculator returns a calculator."""
        from aragora.billing.roi_calculator import get_roi_calculator

        calc = get_roi_calculator()
        assert calc is not None

    def test_get_roi_calculator_uses_default_benchmark(self):
        """get_roi_calculator uses SME benchmark by default."""
        from aragora.billing.roi_calculator import get_roi_calculator, IndustryBenchmark

        calc = get_roi_calculator()
        assert calc._benchmark == IndustryBenchmark.SME

    def test_get_roi_calculator_respects_benchmark_param(self):
        """get_roi_calculator uses specified benchmark."""
        from aragora.billing.roi_calculator import get_roi_calculator, IndustryBenchmark

        calc = get_roi_calculator(benchmark=IndustryBenchmark.CONSULTING)
        assert calc._benchmark == IndustryBenchmark.CONSULTING
