# Aragora Analytics Module

The analytics module provides aggregated metrics, trend analysis, and compliance reporting for enterprise audit management and debate performance tracking.

## Architecture

```
analytics/
├── __init__.py           # Module exports
├── dashboard.py          # AnalyticsDashboard for audit/compliance
└── debate_analytics.py   # DebateAnalytics for debate metrics
```

## Components

### AnalyticsDashboard

Enterprise audit analytics with compliance scoring, risk heatmaps, and cost tracking.

```python
from aragora.analytics import get_analytics_dashboard, TimeRange, Granularity

dashboard = get_analytics_dashboard()

# Get dashboard summary for a workspace
summary = await dashboard.get_summary(
    workspace_id="ws-123",
    time_range=TimeRange.LAST_30_DAYS
)

# Access metrics
print(f"Compliance Score: {summary.compliance_score.overall}")
print(f"Open Findings: {summary.finding_trends.open_count}")
print(f"Total Audit Cost: ${summary.cost_metrics.total_cost}")
```

**Key Features:**
- **Finding Trends**: Track security findings over time
- **Remediation Metrics**: MTTR, closure rates, SLA compliance
- **Agent Metrics**: Per-agent performance and utilization
- **Audit Cost Metrics**: Cost tracking with forecasting
- **Compliance Score**: Aggregate compliance health
- **Risk Heatmap**: Visual risk distribution by category

### DebateAnalytics

Debate performance analytics with usage trends and agent performance.

```python
from aragora.analytics import get_debate_analytics, DebateTimeGranularity

analytics = get_debate_analytics()

# Get debate statistics
stats = await analytics.get_debate_stats(
    workspace_id="ws-123",
    time_range=TimeRange.LAST_7_DAYS,
    granularity=DebateTimeGranularity.DAILY
)

# Access metrics
print(f"Total Debates: {stats.total_debates}")
print(f"Consensus Rate: {stats.consensus_rate}%")
print(f"Average Duration: {stats.avg_duration_seconds}s")
```

**Key Features:**
- **Debate Stats**: Count, consensus rate, duration metrics
- **Agent Performance**: Per-agent debate contributions
- **Usage Trends**: Debate volume over time
- **Cost Breakdown**: Token usage and API costs

## Data Types

### Time Ranges

```python
from aragora.analytics import TimeRange

TimeRange.LAST_24_HOURS
TimeRange.LAST_7_DAYS
TimeRange.LAST_30_DAYS
TimeRange.LAST_90_DAYS
TimeRange.CUSTOM  # Use with start_date/end_date
```

### Granularity

```python
from aragora.analytics import Granularity, DebateTimeGranularity

# Audit analytics
Granularity.HOURLY
Granularity.DAILY
Granularity.WEEKLY
Granularity.MONTHLY

# Debate analytics
DebateTimeGranularity.HOURLY
DebateTimeGranularity.DAILY
DebateTimeGranularity.WEEKLY
```

## Response Models

### DashboardSummary

```python
@dataclass
class DashboardSummary:
    workspace_id: str
    time_range: TimeRange
    finding_trends: FindingTrend
    remediation_metrics: RemediationMetrics
    agent_metrics: list[AgentMetrics]
    cost_metrics: AuditCostMetrics
    compliance_score: ComplianceScore
    risk_heatmap: list[RiskHeatmapCell]
    generated_at: datetime
```

### DebateDashboardSummary

```python
@dataclass
class DebateDashboardSummary:
    workspace_id: str
    time_range: TimeRange
    stats: DebateStats
    agent_performance: list[AgentPerformance]
    usage_trends: list[UsageTrendPoint]
    cost_breakdown: CostBreakdown
    generated_at: datetime
```

## Integration with Handlers

The analytics module powers these HTTP endpoints:

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /api/v2/analytics/dashboard` | AnalyticsHandler | Main dashboard summary |
| `GET /api/v2/analytics/debates` | AnalyticsHandler | Debate-specific analytics |
| `GET /api/v2/analytics/metrics` | AnalyticsMetricsHandler | Raw metrics export |
| `GET /api/v2/analytics/trends` | AnalyticsDashboardHandler | Trend analysis |

## Caching

Analytics queries are cached for performance:

```python
# Default cache TTL by granularity
CACHE_TTL = {
    Granularity.HOURLY: 300,    # 5 minutes
    Granularity.DAILY: 900,     # 15 minutes
    Granularity.WEEKLY: 3600,   # 1 hour
    Granularity.MONTHLY: 7200,  # 2 hours
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANALYTICS_CACHE_ENABLED` | Enable query caching | `true` |
| `ANALYTICS_CACHE_TTL` | Default cache TTL (seconds) | `900` |
| `ANALYTICS_MAX_TIME_RANGE_DAYS` | Maximum query range | `365` |

## Related Modules

- `aragora/observability/` - Prometheus metrics collection
- `aragora/audit/` - Audit event logging
- `aragora/server/handlers/_analytics_impl.py` - Handler implementations
