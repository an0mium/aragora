"""
Aragora Analytics Module.

Provides aggregated metrics, trend analysis, and compliance reporting
for enterprise audit management.

Usage:
    from aragora.analytics import (
        AnalyticsDashboard,
        get_analytics_dashboard,
        TimeRange,
        Granularity,
    )

    dashboard = get_analytics_dashboard()
    summary = await dashboard.get_summary("ws-123")
"""

from .dashboard import (
    AnalyticsDashboard,
    TimeRange,
    Granularity,
    FindingTrend,
    RemediationMetrics,
    AgentMetrics,
    AuditCostMetrics,
    ComplianceScore,
    RiskHeatmapCell,
    DashboardSummary,
    get_analytics_dashboard,
)
from .debate_analytics import (
    DebateAnalytics,
    DebateTimeGranularity,
    DebateMetricType,
    DebateStats,
    AgentPerformance,
    UsageTrendPoint,
    CostBreakdown,
    DebateDashboardSummary,
    get_debate_analytics,
)

__all__ = [
    "AnalyticsDashboard",
    "TimeRange",
    "Granularity",
    "FindingTrend",
    "RemediationMetrics",
    "AgentMetrics",
    "AuditCostMetrics",
    "ComplianceScore",
    "RiskHeatmapCell",
    "DashboardSummary",
    "get_analytics_dashboard",
    # Debate Analytics
    "DebateAnalytics",
    "DebateTimeGranularity",
    "DebateMetricType",
    "DebateStats",
    "AgentPerformance",
    "UsageTrendPoint",
    "CostBreakdown",
    "DebateDashboardSummary",
    "get_debate_analytics",
]
