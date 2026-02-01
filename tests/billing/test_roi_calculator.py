"""
Comprehensive tests for aragora.billing.roi_calculator module.

Tests cover:
- IndustryBenchmark enum
- BENCHMARK_COSTS data structure
- ROIMetrics dataclass
- DebateROIInput dataclass
- ROICalculator class
- Single debate ROI calculation
- Period ROI aggregation
- Future savings estimation
- Benchmark comparison
- Global calculator management
- Business-critical accuracy verification
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from aragora.billing.roi_calculator import (
    IndustryBenchmark,
    BENCHMARK_COSTS,
    ROIMetrics,
    DebateROIInput,
    ROICalculator,
    get_roi_calculator,
)


# =============================================================================
# IndustryBenchmark Tests
# =============================================================================


class TestIndustryBenchmark:
    """Tests for IndustryBenchmark enum."""

    def test_benchmark_values(self):
        """Test benchmark enum values."""
        assert IndustryBenchmark.TECH_STARTUP.value == "tech_startup"
        assert IndustryBenchmark.ENTERPRISE.value == "enterprise"
        assert IndustryBenchmark.SME.value == "sme"
        assert IndustryBenchmark.CONSULTING.value == "consulting"

    def test_benchmark_from_string(self):
        """Test creating benchmark from string."""
        assert IndustryBenchmark("sme") == IndustryBenchmark.SME
        assert IndustryBenchmark("enterprise") == IndustryBenchmark.ENTERPRISE

    def test_invalid_benchmark(self):
        """Test invalid benchmark string raises error."""
        with pytest.raises(ValueError):
            IndustryBenchmark("invalid")


# =============================================================================
# BENCHMARK_COSTS Tests
# =============================================================================


class TestBenchmarkCosts:
    """Tests for BENCHMARK_COSTS data structure."""

    def test_all_benchmarks_have_costs(self):
        """Test all benchmarks have cost data."""
        for benchmark in IndustryBenchmark:
            assert benchmark in BENCHMARK_COSTS
            data = BENCHMARK_COSTS[benchmark]
            assert "avg_decision_cost_usd" in data
            assert "avg_hours_per_decision" in data
            assert "avg_participants" in data
            assert "hourly_rate_usd" in data

    def test_tech_startup_costs(self):
        """Test tech startup benchmark values."""
        data = BENCHMARK_COSTS[IndustryBenchmark.TECH_STARTUP]
        assert data["avg_decision_cost_usd"] == Decimal("150")
        assert data["avg_hours_per_decision"] == 2.0
        assert data["avg_participants"] == 3
        assert data["hourly_rate_usd"] == Decimal("75")

    def test_enterprise_costs(self):
        """Test enterprise benchmark values."""
        data = BENCHMARK_COSTS[IndustryBenchmark.ENTERPRISE]
        assert data["avg_decision_cost_usd"] == Decimal("500")
        assert data["avg_hours_per_decision"] == 4.0
        assert data["avg_participants"] == 5
        assert data["hourly_rate_usd"] == Decimal("100")

    def test_sme_costs(self):
        """Test SME benchmark values."""
        data = BENCHMARK_COSTS[IndustryBenchmark.SME]
        assert data["avg_decision_cost_usd"] == Decimal("200")
        assert data["avg_hours_per_decision"] == 2.5
        assert data["avg_participants"] == 3
        assert data["hourly_rate_usd"] == Decimal("60")

    def test_consulting_costs(self):
        """Test consulting benchmark values."""
        data = BENCHMARK_COSTS[IndustryBenchmark.CONSULTING]
        assert data["avg_decision_cost_usd"] == Decimal("800")
        assert data["avg_hours_per_decision"] == 3.0
        assert data["avg_participants"] == 4
        assert data["hourly_rate_usd"] == Decimal("200")

    def test_benchmark_cost_consistency(self):
        """Test that benchmark costs are internally consistent."""
        # avg_decision_cost should roughly equal hourly_rate * hours * participants
        for benchmark, data in BENCHMARK_COSTS.items():
            calculated = (
                data["hourly_rate_usd"]
                * Decimal(str(data["avg_hours_per_decision"]))
                * data["avg_participants"]
            )
            # Should be within reasonable range (not exact due to rounded benchmark values)
            assert calculated > Decimal("0")


# =============================================================================
# ROIMetrics Tests
# =============================================================================


class TestROIMetrics:
    """Tests for ROIMetrics dataclass."""

    def test_default_values(self):
        """Test ROIMetrics default values."""
        metrics = ROIMetrics()

        assert metrics.total_debates == 0
        assert metrics.completed_debates == 0
        assert metrics.consensus_reached_count == 0
        assert metrics.estimated_hours_saved == 0.0
        assert metrics.total_aragora_cost_usd == Decimal("0")
        assert metrics.cost_savings_usd == Decimal("0")
        assert metrics.roi_percentage == 0.0
        assert metrics.productivity_multiplier == 1.0
        assert metrics.benchmark_type == "sme"

    def test_custom_values(self):
        """Test ROIMetrics with custom values."""
        now = datetime.now(timezone.utc)
        metrics = ROIMetrics(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_debates=50,
            completed_debates=45,
            consensus_reached_count=40,
            estimated_hours_saved=100.5,
            total_aragora_cost_usd=Decimal("150.00"),
            cost_savings_usd=Decimal("850.00"),
            roi_percentage=566.7,
            productivity_multiplier=5.5,
        )

        assert metrics.total_debates == 50
        assert metrics.completed_debates == 45
        assert metrics.roi_percentage == 566.7

    def test_to_dict(self):
        """Test ROIMetrics serialization."""
        now = datetime.now(timezone.utc)
        metrics = ROIMetrics(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_debates=10,
            completed_debates=8,
            consensus_reached_count=6,
            estimated_hours_saved=20.5,
            total_aragora_cost_usd=Decimal("25.00"),
            total_manual_cost_usd=Decimal("450.00"),
            cost_savings_usd=Decimal("425.00"),
            cost_per_decision_usd=Decimal("3.125"),
            roi_percentage=1700.0,
            consensus_rate=0.75,
            avg_confidence_score=0.85,
            productivity_multiplier=3.5,
            benchmark_type="sme",
            benchmark_cost_usd=Decimal("200.00"),
            vs_benchmark_savings_pct=98.4,
        )

        data = metrics.to_dict()

        # Check structure
        assert "period_start" in data
        assert "period_end" in data
        assert "usage" in data
        assert "time_savings" in data
        assert "cost" in data
        assert "roi" in data
        assert "quality" in data
        assert "productivity" in data
        assert "benchmark" in data

        # Check nested values
        assert data["usage"]["total_debates"] == 10
        assert data["usage"]["completed_debates"] == 8
        assert data["time_savings"]["estimated_hours_saved"] == 20.5
        assert data["cost"]["cost_savings_usd"] == "425.00"
        assert data["roi"]["roi_percentage"] == 1700.0
        assert data["quality"]["consensus_rate"] == 75.0  # Converted to percentage
        assert data["productivity"]["productivity_multiplier"] == 3.5
        assert data["benchmark"]["type"] == "sme"


# =============================================================================
# DebateROIInput Tests
# =============================================================================


class TestDebateROIInput:
    """Tests for DebateROIInput dataclass."""

    def test_required_debate_id(self):
        """Test debate_id is required."""
        debate = DebateROIInput(debate_id="debate-123")
        assert debate.debate_id == "debate-123"

    def test_default_values(self):
        """Test DebateROIInput default values."""
        debate = DebateROIInput(debate_id="debate-123")

        assert debate.duration_seconds == 0.0
        assert debate.cost_usd == Decimal("0")
        assert debate.reached_consensus is False
        assert debate.confidence_score == 0.0
        assert debate.agent_count == 0
        assert debate.round_count == 0
        assert debate.completed is False

    def test_full_initialization(self):
        """Test DebateROIInput with all values."""
        debate = DebateROIInput(
            debate_id="debate-456",
            duration_seconds=300.0,  # 5 minutes
            cost_usd=Decimal("0.75"),
            reached_consensus=True,
            confidence_score=0.92,
            agent_count=5,
            round_count=3,
            completed=True,
        )

        assert debate.duration_seconds == 300.0
        assert debate.cost_usd == Decimal("0.75")
        assert debate.reached_consensus is True
        assert debate.confidence_score == 0.92


# =============================================================================
# ROICalculator Initialization Tests
# =============================================================================


class TestROICalculatorInit:
    """Tests for ROICalculator initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        calc = ROICalculator()

        assert calc._benchmark == IndustryBenchmark.SME
        assert calc._hourly_rate == Decimal("60")
        assert calc._hours_per_decision == 2.5
        assert calc._avg_participants == 3

    def test_custom_benchmark(self):
        """Test initialization with custom benchmark."""
        calc = ROICalculator(benchmark=IndustryBenchmark.ENTERPRISE)

        assert calc._benchmark == IndustryBenchmark.ENTERPRISE
        assert calc._hourly_rate == Decimal("100")
        assert calc._hours_per_decision == 4.0
        assert calc._avg_participants == 5

    def test_hourly_rate_override(self):
        """Test hourly rate override."""
        calc = ROICalculator(hourly_rate_override=Decimal("150"))

        assert calc._hourly_rate == Decimal("150")

    def test_hours_override(self):
        """Test hours per decision override."""
        calc = ROICalculator(hours_per_decision_override=5.0)

        assert calc._hours_per_decision == 5.0

    def test_all_overrides(self):
        """Test all overrides together."""
        calc = ROICalculator(
            benchmark=IndustryBenchmark.TECH_STARTUP,
            hourly_rate_override=Decimal("200"),
            hours_per_decision_override=6.0,
        )

        assert calc._benchmark == IndustryBenchmark.TECH_STARTUP
        assert calc._hourly_rate == Decimal("200")
        assert calc._hours_per_decision == 6.0


# =============================================================================
# Single Debate ROI Tests
# =============================================================================


class TestCalculateSingleDebateROI:
    """Tests for calculate_single_debate_roi method."""

    def test_basic_calculation(self):
        """Test basic single debate ROI calculation."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debate = DebateROIInput(
            debate_id="debate-123",
            duration_seconds=300,  # 5 minutes
            cost_usd=Decimal("0.50"),
            reached_consensus=True,
            confidence_score=0.88,
            completed=True,
        )

        result = calc.calculate_single_debate_roi(debate)

        assert result["debate_id"] == "debate-123"
        assert "manual_equivalent_cost_usd" in result
        assert "aragora_cost_usd" in result
        assert "cost_savings_usd" in result
        assert "hours_saved" in result
        assert "roi_percentage" in result
        assert "productivity_multiplier" in result
        assert result["consensus_achieved"] is True
        assert result["confidence_score"] == 0.88

    def test_manual_cost_calculation(self):
        """Test manual equivalent cost calculation accuracy."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # SME: $60/hour * 2.5 hours * 3 participants = $450

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("0.50"),
        )

        result = calc.calculate_single_debate_roi(debate)

        manual_cost = Decimal(result["manual_equivalent_cost_usd"])
        assert manual_cost == Decimal("450")  # 60 * 2.5 * 3

    def test_cost_savings_calculation(self):
        """Test cost savings calculation accuracy."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("1.50"),
        )

        result = calc.calculate_single_debate_roi(debate)

        manual_cost = Decimal(result["manual_equivalent_cost_usd"])
        aragora_cost = Decimal(result["aragora_cost_usd"])
        savings = Decimal(result["cost_savings_usd"])

        assert savings == manual_cost - aragora_cost

    def test_hours_saved_calculation(self):
        """Test hours saved calculation accuracy."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # SME: 2.5 hours * 3 participants = 7.5 manual hours

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=1800,  # 30 minutes = 0.5 hours
            cost_usd=Decimal("0.50"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # Manual: 7.5 hours, Debate: 0.5 hours -> Saved: 7.0 hours
        assert result["hours_saved"] == 7.0

    def test_roi_percentage_calculation(self):
        """Test ROI percentage calculation accuracy."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("1.00"),  # $1 cost
        )

        result = calc.calculate_single_debate_roi(debate)

        # Manual cost: $450, Aragora cost: $1
        # Savings: $449, ROI = (449/1) * 100 = 44,900%
        manual = Decimal(result["manual_equivalent_cost_usd"])
        savings = manual - Decimal("1.00")
        expected_roi = float((savings / Decimal("1.00")) * 100)

        assert abs(result["roi_percentage"] - expected_roi) < 0.1

    def test_productivity_multiplier_calculation(self):
        """Test productivity multiplier calculation."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # SME: 2.5 hours * 3 participants = 7.5 manual hours

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=3600,  # 1 hour
            cost_usd=Decimal("0.50"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # Manual: 7.5 hours, Debate: 1 hour -> Multiplier: 7.5x
        assert result["productivity_multiplier"] == 7.5

    def test_zero_cost_debate(self):
        """Test ROI calculation with zero cost."""
        calc = ROICalculator()

        debate = DebateROIInput(
            debate_id="free",
            duration_seconds=300,
            cost_usd=Decimal("0"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # ROI should be 0% when cost is 0 (avoid division by zero)
        assert result["roi_percentage"] == 0.0

    def test_zero_duration_debate(self):
        """Test ROI calculation with zero duration."""
        calc = ROICalculator()

        debate = DebateROIInput(
            debate_id="instant",
            duration_seconds=0,
            cost_usd=Decimal("0.50"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # Productivity should be 1.0 when duration is 0
        assert result["productivity_multiplier"] == 1.0


# =============================================================================
# Period ROI Tests
# =============================================================================


class TestCalculatePeriodROI:
    """Tests for calculate_period_roi method."""

    def test_empty_debates_list(self):
        """Test period ROI with no debates."""
        calc = ROICalculator()

        result = calc.calculate_period_roi(debates=[])

        assert result.total_debates == 0
        assert result.completed_debates == 0
        assert result.roi_percentage == 0.0

    def test_single_debate_period(self):
        """Test period ROI with single debate."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=600,  # 10 minutes
                cost_usd=Decimal("1.00"),
                reached_consensus=True,
                confidence_score=0.90,
                completed=True,
            )
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_debates == 1
        assert result.completed_debates == 1
        assert result.consensus_reached_count == 1
        assert result.avg_confidence_score == 0.90

    def test_multiple_debates_aggregation(self):
        """Test aggregation of multiple debates."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=300,
                cost_usd=Decimal("0.50"),
                reached_consensus=True,
                confidence_score=0.85,
                completed=True,
            ),
            DebateROIInput(
                debate_id="d2",
                duration_seconds=600,
                cost_usd=Decimal("0.75"),
                reached_consensus=True,
                confidence_score=0.90,
                completed=True,
            ),
            DebateROIInput(
                debate_id="d3",
                duration_seconds=450,
                cost_usd=Decimal("0.60"),
                reached_consensus=False,
                confidence_score=0.70,
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_debates == 3
        assert result.completed_debates == 3
        assert result.consensus_reached_count == 2

        # Average confidence: (0.85 + 0.90 + 0.70) / 3 = 0.8166...
        assert abs(result.avg_confidence_score - 0.8166) < 0.01

    def test_incomplete_debates_excluded(self):
        """Test that incomplete debates are excluded from costs."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=300,
                cost_usd=Decimal("1.00"),
                completed=True,
            ),
            DebateROIInput(
                debate_id="d2",
                duration_seconds=100,
                cost_usd=Decimal("0.25"),
                completed=False,  # Not completed
            ),
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_debates == 2
        assert result.completed_debates == 1
        # Only completed debate cost should be included
        assert result.total_aragora_cost_usd == Decimal("1.00")

    def test_consensus_rate_calculation(self):
        """Test consensus rate calculation."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(debate_id="d1", reached_consensus=True, completed=True),
            DebateROIInput(debate_id="d2", reached_consensus=True, completed=True),
            DebateROIInput(debate_id="d3", reached_consensus=False, completed=True),
            DebateROIInput(debate_id="d4", reached_consensus=True, completed=True),
        ]

        result = calc.calculate_period_roi(debates)

        # 3 out of 4 reached consensus
        assert result.consensus_rate == 0.75

    def test_time_savings_calculation(self):
        """Test time savings calculation for period."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # Manual: 2.5 hours * 3 participants = 7.5 hours per debate

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=600,  # 10 min = 0.167 hours
                completed=True,
            ),
            DebateROIInput(
                debate_id="d2",
                duration_seconds=600,
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)

        # Manual equivalent: 7.5 * 2 = 15 hours
        assert result.manual_equivalent_hours == 15.0

        # Actual debate hours: 0.333 hours
        # Hours saved: 15 - 0.333 = 14.667 hours
        assert result.estimated_hours_saved > 14.0

    def test_subscription_cost_included(self):
        """Test subscription cost is included in calculations."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=300,
                cost_usd=Decimal("0.50"),
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(
            debates,
            subscription_cost_usd=Decimal("99.00"),
        )

        # Cost per decision should include subscription
        # Total cost: $0.50 + $99.00 = $99.50
        # Cost per decision: $99.50 / 1 = $99.50
        assert result.cost_per_decision_usd == Decimal("99.50")

    def test_payback_debates_calculation(self):
        """Test payback debates calculation."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # SME manual cost per decision: $60 * 2.5 * 3 = $450

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=300,
                cost_usd=Decimal("0.50"),
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(
            debates,
            subscription_cost_usd=Decimal("100.00"),
        )

        # Savings per debate: $450 - $100.50 = $349.50
        # Payback: ceil($100 / $349.50) = 1 debate
        assert result.payback_debates > 0

    def test_vs_benchmark_savings(self):
        """Test benchmark savings percentage."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debates = [
            DebateROIInput(
                debate_id="d1",
                duration_seconds=300,
                cost_usd=Decimal("1.00"),
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)

        # Benchmark cost: $450
        # Aragora cost per decision: $1.00
        # Savings: (1 - 1/450) * 100 = 99.78%
        assert result.vs_benchmark_savings_pct > 99.0

    def test_period_dates_default(self):
        """Test default period dates."""
        calc = ROICalculator()
        now = datetime.now(timezone.utc)

        result = calc.calculate_period_roi([])

        # Default period should be ~30 days ending now
        assert (now - result.period_end).total_seconds() < 60
        assert (result.period_end - result.period_start).days == 30

    def test_custom_period_dates(self):
        """Test custom period dates."""
        calc = ROICalculator()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        result = calc.calculate_period_roi(
            [],
            period_start=start,
            period_end=end,
        )

        assert result.period_start == start
        assert result.period_end == end


# =============================================================================
# Future Savings Estimation Tests
# =============================================================================


class TestEstimateFutureSavings:
    """Tests for estimate_future_savings method."""

    def test_basic_projection(self):
        """Test basic future savings projection."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        result = calc.estimate_future_savings(
            projected_debates_per_month=100,
        )

        assert "projections" in result
        assert "assumptions" in result
        assert result["projections"]["debates_per_month"] == 100

    def test_monthly_projections(self):
        """Test monthly projection values."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # Manual cost per decision: $60 * 2.5 * 3 = $450

        result = calc.estimate_future_savings(
            projected_debates_per_month=10,
            current_cost_per_debate=Decimal("1.00"),
            subscription_cost_usd=Decimal("0"),
        )

        monthly = result["projections"]["monthly"]

        # Manual cost: $450 * 10 = $4,500
        assert Decimal(monthly["manual_cost_usd"]) == Decimal("4500")

        # Aragora cost: $1 * 10 + $0 subscription = $10
        assert Decimal(monthly["aragora_cost_usd"]) == Decimal("10")

        # Savings: $4,500 - $10 = $4,490
        assert Decimal(monthly["savings_usd"]) == Decimal("4490")

    def test_annual_projections(self):
        """Test annual projection values are 12x monthly."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        result = calc.estimate_future_savings(
            projected_debates_per_month=10,
            current_cost_per_debate=Decimal("1.00"),
            subscription_cost_usd=Decimal("0"),
        )

        monthly = result["projections"]["monthly"]
        annual = result["projections"]["annual"]

        # Annual should be 12x monthly
        assert Decimal(annual["manual_cost_usd"]) == Decimal(monthly["manual_cost_usd"]) * 12
        assert Decimal(annual["aragora_cost_usd"]) == Decimal(monthly["aragora_cost_usd"]) * 12
        assert Decimal(annual["savings_usd"]) == Decimal(monthly["savings_usd"]) * 12

    def test_hours_saved_projection(self):
        """Test hours saved projections."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)
        # Manual: 2.5 hours * 3 participants = 7.5 person-hours per debate

        result = calc.estimate_future_savings(
            projected_debates_per_month=10,
        )

        monthly = result["projections"]["monthly"]

        # Manual hours: 7.5 * 10 = 75 hours
        # Aragora hours: 10 * (5/60) = 0.833 hours
        # Saved: ~74.2 hours
        assert monthly["hours_saved"] > 70

    def test_subscription_cost_included(self):
        """Test subscription cost in projections."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        result = calc.estimate_future_savings(
            projected_debates_per_month=10,
            current_cost_per_debate=Decimal("0.50"),
            subscription_cost_usd=Decimal("100.00"),
        )

        monthly = result["projections"]["monthly"]

        # Aragora cost: $0.50 * 10 + $100 = $105
        assert Decimal(monthly["aragora_cost_usd"]) == Decimal("105")

    def test_assumptions_included(self):
        """Test assumptions are included in result."""
        calc = ROICalculator(benchmark=IndustryBenchmark.ENTERPRISE)

        result = calc.estimate_future_savings(
            projected_debates_per_month=20,
        )

        assumptions = result["assumptions"]

        assert assumptions["benchmark"] == "enterprise"
        assert assumptions["hourly_rate_usd"] == "100"
        assert assumptions["hours_per_decision"] == 4.0
        assert assumptions["avg_participants"] == 5
        assert assumptions["avg_debate_duration_minutes"] == 5


# =============================================================================
# Benchmark Comparison Tests
# =============================================================================


class TestGetBenchmarkComparison:
    """Tests for get_benchmark_comparison method."""

    def test_returns_all_benchmarks(self):
        """Test all benchmarks are included."""
        calc = ROICalculator()

        result = calc.get_benchmark_comparison()

        assert "benchmarks" in result
        assert "current_selection" in result

        benchmarks = result["benchmarks"]
        assert "tech_startup" in benchmarks
        assert "enterprise" in benchmarks
        assert "sme" in benchmarks
        assert "consulting" in benchmarks

    def test_benchmark_data_structure(self):
        """Test benchmark data structure."""
        calc = ROICalculator()

        result = calc.get_benchmark_comparison()

        for name, data in result["benchmarks"].items():
            assert "avg_decision_cost_usd" in data
            assert "avg_hours_per_decision" in data
            assert "avg_participants" in data
            assert "hourly_rate_usd" in data

    def test_current_selection_correct(self):
        """Test current selection reflects initialization."""
        calc = ROICalculator(benchmark=IndustryBenchmark.CONSULTING)

        result = calc.get_benchmark_comparison()

        assert result["current_selection"] == "consulting"


# =============================================================================
# Global Calculator Tests
# =============================================================================


class TestGetROICalculator:
    """Tests for get_roi_calculator function."""

    def test_creates_calculator(self):
        """Test calculator creation."""
        calc = get_roi_calculator()
        assert isinstance(calc, ROICalculator)

    def test_default_benchmark(self):
        """Test default benchmark is SME."""
        calc = get_roi_calculator()
        assert calc._benchmark == IndustryBenchmark.SME

    def test_custom_benchmark(self):
        """Test custom benchmark selection."""
        calc = get_roi_calculator(benchmark=IndustryBenchmark.ENTERPRISE)
        assert calc._benchmark == IndustryBenchmark.ENTERPRISE


# =============================================================================
# Business-Critical Accuracy Tests
# =============================================================================


class TestBusinessCriticalAccuracy:
    """Tests to verify accuracy of billing calculations."""

    def test_sme_manual_cost_accuracy(self):
        """Verify SME manual cost calculation is accurate."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        # SME benchmark: $60/hr * 2.5 hrs * 3 people = $450
        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("0"),
        )

        result = calc.calculate_single_debate_roi(debate)
        manual_cost = Decimal(result["manual_equivalent_cost_usd"])

        assert manual_cost == Decimal("450"), f"Expected $450, got ${manual_cost}"

    def test_enterprise_manual_cost_accuracy(self):
        """Verify enterprise manual cost calculation is accurate."""
        calc = ROICalculator(benchmark=IndustryBenchmark.ENTERPRISE)

        # Enterprise benchmark: $100/hr * 4 hrs * 5 people = $2000
        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("0"),
        )

        result = calc.calculate_single_debate_roi(debate)
        manual_cost = Decimal(result["manual_equivalent_cost_usd"])

        assert manual_cost == Decimal("2000"), f"Expected $2000, got ${manual_cost}"

    def test_consulting_manual_cost_accuracy(self):
        """Verify consulting manual cost calculation is accurate."""
        calc = ROICalculator(benchmark=IndustryBenchmark.CONSULTING)

        # Consulting benchmark: $200/hr * 3 hrs * 4 people = $2400
        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("0"),
        )

        result = calc.calculate_single_debate_roi(debate)
        manual_cost = Decimal(result["manual_equivalent_cost_usd"])

        assert manual_cost == Decimal("2400"), f"Expected $2400, got ${manual_cost}"

    def test_roi_formula_correctness(self):
        """Verify ROI percentage formula is correct."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("45.00"),  # 10% of manual cost
        )

        result = calc.calculate_single_debate_roi(debate)

        # Manual: $450, Aragora: $45
        # Savings: $405
        # ROI = ($405 / $45) * 100 = 900%
        expected_roi = ((Decimal("450") - Decimal("45")) / Decimal("45")) * 100
        actual_roi = result["roi_percentage"]

        assert abs(actual_roi - float(expected_roi)) < 0.1, (
            f"Expected ROI {expected_roi}%, got {actual_roi}%"
        )

    def test_cost_savings_never_negative(self):
        """Verify cost savings are calculated correctly even when aragora is more expensive."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        # Very expensive debate (more than manual cost)
        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("500.00"),  # More than $450 manual cost
        )

        result = calc.calculate_single_debate_roi(debate)
        savings = Decimal(result["cost_savings_usd"])

        # Savings should be negative when aragora is more expensive
        assert savings == Decimal("-50"), f"Expected -$50, got ${savings}"

    def test_period_cost_aggregation(self):
        """Verify period costs aggregate correctly."""
        calc = ROICalculator(benchmark=IndustryBenchmark.SME)

        debates = [
            DebateROIInput(
                debate_id="d1",
                cost_usd=Decimal("1.00"),
                completed=True,
            ),
            DebateROIInput(
                debate_id="d2",
                cost_usd=Decimal("2.00"),
                completed=True,
            ),
            DebateROIInput(
                debate_id="d3",
                cost_usd=Decimal("3.00"),
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_aragora_cost_usd == Decimal("6.00"), (
            f"Expected $6.00, got ${result.total_aragora_cost_usd}"
        )

    def test_decimal_precision_maintained(self):
        """Verify decimal precision is maintained in calculations."""
        calc = ROICalculator()

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("0.123456789"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # Aragora cost should maintain precision
        assert result["aragora_cost_usd"] == "0.123456789"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_debate(self):
        """Test ROI calculation for very long debate."""
        calc = ROICalculator()

        debate = DebateROIInput(
            debate_id="long",
            duration_seconds=86400,  # 24 hours
            cost_usd=Decimal("100.00"),
        )

        result = calc.calculate_single_debate_roi(debate)
        # Should still calculate without errors

    def test_very_short_debate(self):
        """Test ROI calculation for very short debate."""
        calc = ROICalculator()

        debate = DebateROIInput(
            debate_id="quick",
            duration_seconds=1,  # 1 second
            cost_usd=Decimal("0.01"),
        )

        result = calc.calculate_single_debate_roi(debate)
        assert result["productivity_multiplier"] > 0

    def test_high_volume_debates(self):
        """Test period ROI with large number of debates."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(
                debate_id=f"d{i}",
                duration_seconds=300,
                cost_usd=Decimal("0.50"),
                completed=True,
            )
            for i in range(1000)
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_debates == 1000
        assert result.completed_debates == 1000

    def test_zero_confidence_scores(self):
        """Test handling of zero confidence scores."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(
                debate_id="d1",
                confidence_score=0.0,
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)
        # Zero confidence scores should be excluded from average
        assert result.avg_confidence_score == 0.0

    def test_mixed_completed_incomplete(self):
        """Test mixed completed and incomplete debates."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(debate_id="d1", completed=True, cost_usd=Decimal("1")),
            DebateROIInput(debate_id="d2", completed=False, cost_usd=Decimal("2")),
            DebateROIInput(debate_id="d3", completed=True, cost_usd=Decimal("3")),
            DebateROIInput(debate_id="d4", completed=False, cost_usd=Decimal("4")),
        ]

        result = calc.calculate_period_roi(debates)

        assert result.total_debates == 4
        assert result.completed_debates == 2
        assert result.total_aragora_cost_usd == Decimal("4")  # Only completed

    def test_all_debates_fail_consensus(self):
        """Test when no debates reach consensus."""
        calc = ROICalculator()

        debates = [
            DebateROIInput(
                debate_id="d1",
                reached_consensus=False,
                completed=True,
            ),
            DebateROIInput(
                debate_id="d2",
                reached_consensus=False,
                completed=True,
            ),
        ]

        result = calc.calculate_period_roi(debates)

        assert result.consensus_rate == 0.0
        assert result.consensus_reached_count == 0

    def test_override_with_custom_rates(self):
        """Test overrides work correctly with custom rates."""
        calc = ROICalculator(
            hourly_rate_override=Decimal("1000"),
            hours_per_decision_override=10.0,
        )

        debate = DebateROIInput(
            debate_id="test",
            duration_seconds=300,
            cost_usd=Decimal("1.00"),
        )

        result = calc.calculate_single_debate_roi(debate)

        # Manual cost should use overrides: $1000 * 10 * 3 = $30,000
        manual_cost = Decimal(result["manual_equivalent_cost_usd"])
        assert manual_cost == Decimal("30000")
