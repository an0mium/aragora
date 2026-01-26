"""
ROI Calculator for Aragora SME Usage Dashboard.

Calculates return on investment metrics for AI-assisted decision making:
- Time savings from automated debates vs manual processes
- Cost per decision compared to industry benchmarks
- Quality indicators based on consensus rates
- Productivity multipliers for team evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IndustryBenchmark(str, Enum):
    """Industry benchmarks for decision-making costs."""

    # Based on industry research for decision meeting costs
    TECH_STARTUP = "tech_startup"
    ENTERPRISE = "enterprise"
    SME = "sme"
    CONSULTING = "consulting"


# Industry benchmark data (avg cost per decision in USD)
BENCHMARK_COSTS: Dict[IndustryBenchmark, Dict[str, Any]] = {
    IndustryBenchmark.TECH_STARTUP: {
        "avg_decision_cost_usd": Decimal("150"),
        "avg_hours_per_decision": 2.0,
        "avg_participants": 3,
        "hourly_rate_usd": Decimal("75"),
    },
    IndustryBenchmark.ENTERPRISE: {
        "avg_decision_cost_usd": Decimal("500"),
        "avg_hours_per_decision": 4.0,
        "avg_participants": 5,
        "hourly_rate_usd": Decimal("100"),
    },
    IndustryBenchmark.SME: {
        "avg_decision_cost_usd": Decimal("200"),
        "avg_hours_per_decision": 2.5,
        "avg_participants": 3,
        "hourly_rate_usd": Decimal("60"),
    },
    IndustryBenchmark.CONSULTING: {
        "avg_decision_cost_usd": Decimal("800"),
        "avg_hours_per_decision": 3.0,
        "avg_participants": 4,
        "hourly_rate_usd": Decimal("200"),
    },
}


@dataclass
class ROIMetrics:
    """Comprehensive ROI metrics for a time period."""

    # Time period
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Usage metrics
    total_debates: int = 0
    completed_debates: int = 0
    consensus_reached_count: int = 0

    # Time savings
    estimated_hours_saved: float = 0.0
    avg_debate_duration_minutes: float = 0.0
    manual_equivalent_hours: float = 0.0

    # Cost metrics
    total_aragora_cost_usd: Decimal = Decimal("0")
    total_manual_cost_usd: Decimal = Decimal("0")
    cost_savings_usd: Decimal = Decimal("0")
    cost_per_decision_usd: Decimal = Decimal("0")

    # ROI calculations
    roi_percentage: float = 0.0
    payback_debates: int = 0  # Debates needed to break even

    # Quality metrics
    consensus_rate: float = 0.0
    avg_confidence_score: float = 0.0

    # Productivity
    productivity_multiplier: float = 1.0  # How many times faster than manual

    # Comparison
    benchmark_type: str = "sme"
    benchmark_cost_usd: Decimal = Decimal("0")
    vs_benchmark_savings_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "usage": {
                "total_debates": self.total_debates,
                "completed_debates": self.completed_debates,
                "consensus_reached_count": self.consensus_reached_count,
            },
            "time_savings": {
                "estimated_hours_saved": round(self.estimated_hours_saved, 1),
                "avg_debate_duration_minutes": round(self.avg_debate_duration_minutes, 1),
                "manual_equivalent_hours": round(self.manual_equivalent_hours, 1),
            },
            "cost": {
                "total_aragora_cost_usd": str(self.total_aragora_cost_usd),
                "total_manual_cost_usd": str(self.total_manual_cost_usd),
                "cost_savings_usd": str(self.cost_savings_usd),
                "cost_per_decision_usd": str(self.cost_per_decision_usd),
            },
            "roi": {
                "roi_percentage": round(self.roi_percentage, 1),
                "payback_debates": self.payback_debates,
            },
            "quality": {
                "consensus_rate": round(self.consensus_rate * 100, 1),
                "avg_confidence_score": round(self.avg_confidence_score, 2),
            },
            "productivity": {
                "productivity_multiplier": round(self.productivity_multiplier, 1),
            },
            "benchmark": {
                "type": self.benchmark_type,
                "cost_usd": str(self.benchmark_cost_usd),
                "savings_vs_benchmark_pct": round(self.vs_benchmark_savings_pct, 1),
            },
        }


@dataclass
class DebateROIInput:
    """Input data for a single debate's ROI calculation."""

    debate_id: str
    duration_seconds: float = 0.0
    cost_usd: Decimal = Decimal("0")
    reached_consensus: bool = False
    confidence_score: float = 0.0
    agent_count: int = 0
    round_count: int = 0
    completed: bool = False


class ROICalculator:
    """
    Calculates ROI metrics for Aragora usage.

    Compares AI-assisted debate costs against manual decision-making
    processes to quantify value delivered.
    """

    def __init__(
        self,
        benchmark: IndustryBenchmark = IndustryBenchmark.SME,
        hourly_rate_override: Optional[Decimal] = None,
        hours_per_decision_override: Optional[float] = None,
    ):
        """
        Initialize ROI calculator.

        Args:
            benchmark: Industry benchmark to compare against
            hourly_rate_override: Override default hourly rate
            hours_per_decision_override: Override default hours per decision
        """
        self._benchmark = benchmark
        self._benchmark_data = BENCHMARK_COSTS[benchmark]

        # Use overrides or defaults
        self._hourly_rate = hourly_rate_override or self._benchmark_data["hourly_rate_usd"]
        self._hours_per_decision = (
            hours_per_decision_override or self._benchmark_data["avg_hours_per_decision"]
        )
        self._avg_participants = self._benchmark_data["avg_participants"]

    def calculate_single_debate_roi(
        self,
        debate: DebateROIInput,
    ) -> Dict[str, Any]:
        """
        Calculate ROI for a single debate.

        Args:
            debate: Debate input data

        Returns:
            ROI metrics for the debate
        """
        # Calculate manual equivalent cost
        manual_cost = self._hourly_rate * Decimal(str(self._hours_per_decision))
        manual_cost *= self._avg_participants  # Multiple participants

        # Time saved
        debate_hours = debate.duration_seconds / 3600
        manual_hours = self._hours_per_decision * self._avg_participants
        hours_saved = manual_hours - debate_hours

        # Cost savings
        cost_savings = manual_cost - debate.cost_usd

        # ROI percentage
        roi_pct = 0.0
        if debate.cost_usd > 0:
            roi_pct = float((cost_savings / debate.cost_usd) * 100)

        # Productivity multiplier
        productivity = 1.0
        if debate_hours > 0:
            productivity = manual_hours / debate_hours

        return {
            "debate_id": debate.debate_id,
            "manual_equivalent_cost_usd": str(manual_cost),
            "aragora_cost_usd": str(debate.cost_usd),
            "cost_savings_usd": str(cost_savings),
            "hours_saved": round(hours_saved, 2),
            "roi_percentage": round(roi_pct, 1),
            "productivity_multiplier": round(productivity, 1),
            "consensus_achieved": debate.reached_consensus,
            "confidence_score": debate.confidence_score,
        }

    def calculate_period_roi(
        self,
        debates: List[DebateROIInput],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        subscription_cost_usd: Decimal = Decimal("0"),
    ) -> ROIMetrics:
        """
        Calculate aggregate ROI for a period.

        Args:
            debates: List of debate input data
            period_start: Start of period
            period_end: End of period
            subscription_cost_usd: Fixed subscription cost for period

        Returns:
            Aggregate ROI metrics
        """
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        metrics = ROIMetrics(
            period_start=period_start,
            period_end=period_end,
            benchmark_type=self._benchmark.value,
        )

        if not debates:
            return metrics

        # Aggregate metrics
        total_duration_seconds = 0.0
        total_confidence = 0.0
        confidence_count = 0
        completed_debates = []

        for debate in debates:
            metrics.total_debates += 1

            if debate.completed:
                metrics.completed_debates += 1
                completed_debates.append(debate)
                total_duration_seconds += debate.duration_seconds
                metrics.total_aragora_cost_usd += debate.cost_usd

                if debate.reached_consensus:
                    metrics.consensus_reached_count += 1

                if debate.confidence_score > 0:
                    total_confidence += debate.confidence_score
                    confidence_count += 1

        # Calculate averages
        if metrics.completed_debates > 0:
            metrics.avg_debate_duration_minutes = (
                total_duration_seconds / metrics.completed_debates / 60
            )

        if confidence_count > 0:
            metrics.avg_confidence_score = total_confidence / confidence_count

        # Consensus rate
        if metrics.completed_debates > 0:
            metrics.consensus_rate = metrics.consensus_reached_count / metrics.completed_debates

        # Calculate manual equivalent costs
        manual_cost_per_decision = (
            self._hourly_rate * Decimal(str(self._hours_per_decision)) * self._avg_participants
        )
        metrics.total_manual_cost_usd = manual_cost_per_decision * metrics.completed_debates
        metrics.benchmark_cost_usd = manual_cost_per_decision

        # Manual equivalent hours (total person-hours)
        metrics.manual_equivalent_hours = (
            self._hours_per_decision * self._avg_participants * metrics.completed_debates
        )

        # Actual debate hours
        actual_hours = total_duration_seconds / 3600

        # Time saved
        metrics.estimated_hours_saved = metrics.manual_equivalent_hours - actual_hours

        # Include subscription cost in total
        total_cost = metrics.total_aragora_cost_usd + subscription_cost_usd

        # Cost savings
        metrics.cost_savings_usd = metrics.total_manual_cost_usd - total_cost

        # Cost per decision
        if metrics.completed_debates > 0:
            metrics.cost_per_decision_usd = total_cost / metrics.completed_debates

        # ROI percentage
        if total_cost > 0:
            metrics.roi_percentage = float((metrics.cost_savings_usd / total_cost) * 100)

        # Payback debates (how many debates to break even on subscription)
        if subscription_cost_usd > 0 and manual_cost_per_decision > metrics.cost_per_decision_usd:
            savings_per_debate = manual_cost_per_decision - metrics.cost_per_decision_usd
            if savings_per_debate > 0:
                metrics.payback_debates = int(subscription_cost_usd / savings_per_debate) + 1

        # Productivity multiplier
        if actual_hours > 0:
            metrics.productivity_multiplier = metrics.manual_equivalent_hours / actual_hours

        # Benchmark comparison
        if metrics.benchmark_cost_usd > 0 and metrics.completed_debates > 0:
            metrics.vs_benchmark_savings_pct = float(
                (1 - (metrics.cost_per_decision_usd / metrics.benchmark_cost_usd)) * 100
            )

        return metrics

    def estimate_future_savings(
        self,
        projected_debates_per_month: int,
        current_cost_per_debate: Decimal = Decimal("0.50"),
        subscription_cost_usd: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """
        Estimate future savings based on projected usage.

        Args:
            projected_debates_per_month: Expected debates per month
            current_cost_per_debate: Average cost per debate
            subscription_cost_usd: Monthly subscription cost

        Returns:
            Savings projections
        """
        manual_cost_per_decision = (
            self._hourly_rate * Decimal(str(self._hours_per_decision)) * self._avg_participants
        )

        # Monthly projections
        monthly_manual_cost = manual_cost_per_decision * projected_debates_per_month
        monthly_aragora_cost = (
            current_cost_per_debate * projected_debates_per_month + subscription_cost_usd
        )
        monthly_savings = monthly_manual_cost - monthly_aragora_cost

        # Time savings
        manual_hours_monthly = (
            self._hours_per_decision * self._avg_participants * projected_debates_per_month
        )
        # Estimate 5 min avg per debate
        aragora_hours_monthly = projected_debates_per_month * (5 / 60)
        hours_saved_monthly = manual_hours_monthly - aragora_hours_monthly

        return {
            "projections": {
                "debates_per_month": projected_debates_per_month,
                "monthly": {
                    "manual_cost_usd": str(monthly_manual_cost),
                    "aragora_cost_usd": str(monthly_aragora_cost),
                    "savings_usd": str(monthly_savings),
                    "hours_saved": round(hours_saved_monthly, 1),
                },
                "annual": {
                    "manual_cost_usd": str(monthly_manual_cost * 12),
                    "aragora_cost_usd": str(monthly_aragora_cost * 12),
                    "savings_usd": str(monthly_savings * 12),
                    "hours_saved": round(hours_saved_monthly * 12, 1),
                },
            },
            "assumptions": {
                "benchmark": self._benchmark.value,
                "hourly_rate_usd": str(self._hourly_rate),
                "hours_per_decision": self._hours_per_decision,
                "avg_participants": self._avg_participants,
                "avg_debate_duration_minutes": 5,
            },
        }

    def get_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Get benchmark comparison data for all industry types.

        Returns:
            Comparison data for each benchmark
        """
        comparisons = {}

        for benchmark, data in BENCHMARK_COSTS.items():
            comparisons[benchmark.value] = {
                "avg_decision_cost_usd": str(data["avg_decision_cost_usd"]),
                "avg_hours_per_decision": data["avg_hours_per_decision"],
                "avg_participants": data["avg_participants"],
                "hourly_rate_usd": str(data["hourly_rate_usd"]),
            }

        return {
            "benchmarks": comparisons,
            "current_selection": self._benchmark.value,
        }


# Global calculator instance with SME defaults
_roi_calculator: Optional[ROICalculator] = None


def get_roi_calculator(
    benchmark: IndustryBenchmark = IndustryBenchmark.SME,
) -> ROICalculator:
    """Get or create the global ROI calculator."""
    global _roi_calculator
    if _roi_calculator is None or _roi_calculator._benchmark != benchmark:
        _roi_calculator = ROICalculator(benchmark=benchmark)
    return _roi_calculator


__all__ = [
    "ROICalculator",
    "ROIMetrics",
    "DebateROIInput",
    "IndustryBenchmark",
    "BENCHMARK_COSTS",
    "get_roi_calculator",
]
