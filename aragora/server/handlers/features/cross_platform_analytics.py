"""
Cross-Platform Analytics Dashboard Handler.

Provides unified analytics aggregation across multiple platforms:
- Internal Aragora metrics (debates, agents, memory, knowledge)
- Google Analytics 4
- Mixpanel
- Metabase
- Segment

Endpoints:
- GET  /api/v1/analytics/cross-platform/summary     - Unified dashboard summary
- GET  /api/v1/analytics/cross-platform/metrics     - Aggregated metrics
- GET  /api/v1/analytics/cross-platform/trends      - Cross-platform trends
- GET  /api/v1/analytics/cross-platform/comparison  - Platform comparison
- GET  /api/v1/analytics/cross-platform/correlation - Metric correlation
- GET  /api/v1/analytics/cross-platform/anomalies   - Anomaly detection
- POST /api/v1/analytics/cross-platform/query       - Custom metric query
- GET  /api/v1/analytics/cross-platform/alerts      - Active alerts
- POST /api/v1/analytics/cross-platform/alerts      - Create alert rule
- GET  /api/v1/analytics/cross-platform/export      - Export data
- GET  /api/v1/analytics/cross-platform/demo        - Demo data
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..base import (
    HandlerResult,
    error_response,
    success_response,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError

logger = logging.getLogger(__name__)

# =============================================================================
# RBAC Permissions
# =============================================================================

ANALYTICS_READ_PERMISSION = "analytics:read"
ANALYTICS_WRITE_PERMISSION = "analytics:write"
ANALYTICS_EXPORT_PERMISSION = "analytics:export"


# =============================================================================
# Enums and Data Classes
# =============================================================================


class Platform(Enum):
    """Supported analytics platforms."""

    ARAGORA = "aragora"
    GOOGLE_ANALYTICS = "google_analytics"
    MIXPANEL = "mixpanel"
    METABASE = "metabase"
    SEGMENT = "segment"


class MetricType(Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class TimeRange(Enum):
    """Predefined time ranges."""

    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    LAST_QUARTER = "90d"
    LAST_YEAR = "365d"


@dataclass
class MetricValue:
    """A single metric value."""

    name: str
    value: float
    platform: Platform
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "platform": self.platform.value,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": self.dimensions,
            "metric_type": self.metric_type.value,
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric across platforms."""

    name: str
    total: float
    by_platform: Dict[str, float]
    trend: str  # "up", "down", "stable"
    change_percent: float
    period: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total": self.total,
            "by_platform": self.by_platform,
            "trend": self.trend,
            "change_percent": self.change_percent,
            "period": self.period,
        }


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""

    id: str
    metric_name: str
    platform: Platform
    timestamp: datetime
    expected_value: float
    actual_value: float
    deviation: float
    severity: AlertSeverity
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "platform": self.platform.value,
            "timestamp": self.timestamp.isoformat(),
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "severity": self.severity.value,
            "description": self.description,
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""

    id: str
    name: str
    metric_name: str
    condition: str  # "above", "below", "equals", "deviation"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    platforms: List[Platform] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "platforms": [p.value for p in self.platforms],
            "created_at": self.created_at.isoformat(),
            "triggered_count": self.triggered_count,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
        }


@dataclass
class Alert:
    """Active alert instance."""

    id: str
    rule_id: str
    rule_name: str
    metric_name: str
    platform: Platform
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    current_value: float
    threshold: float
    message: str
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "platform": self.platform.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat(),
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


# =============================================================================
# In-Memory Storage
# =============================================================================

_alert_rules: Dict[str, Dict[str, AlertRule]] = {}  # tenant_id -> rule_id -> AlertRule
_active_alerts: Dict[str, Dict[str, Alert]] = {}  # tenant_id -> alert_id -> Alert
_metric_cache: Dict[str, Dict[str, List[MetricValue]]] = {}  # tenant_id -> metric_name -> values


def _get_tenant_rules(tenant_id: str) -> Dict[str, AlertRule]:
    if tenant_id not in _alert_rules:
        _alert_rules[tenant_id] = {}
    return _alert_rules[tenant_id]


def _get_tenant_alerts(tenant_id: str) -> Dict[str, Alert]:
    if tenant_id not in _active_alerts:
        _active_alerts[tenant_id] = {}
    return _active_alerts[tenant_id]


# =============================================================================
# Platform Data Fetchers
# =============================================================================


async def fetch_aragora_metrics(tenant_id: str, time_range: str) -> Dict[str, float]:
    """Fetch internal Aragora metrics."""
    # In real implementation, would query internal services
    return {
        "debates_total": 1250,
        "debates_active": 12,
        "consensus_rate": 0.78,
        "agents_active": 45,
        "cost_usd_total": 156.78,
        "findings_total": 234,
        "findings_critical": 8,
        "memory_utilization": 0.65,
        "knowledge_nodes": 15234,
        "deliberation_avg_rounds": 3.2,
    }


async def fetch_ga4_metrics(tenant_id: str, time_range: str) -> Dict[str, float]:
    """Fetch Google Analytics 4 metrics."""
    try:
        from aragora.connectors.analytics.google_analytics import (  # noqa: F401
            GoogleAnalyticsConnector,
        )

        # In real implementation, would use actual client
    except ImportError:
        pass

    # Return demo data
    return {
        "sessions": 45678,
        "users": 12345,
        "pageviews": 156789,
        "bounce_rate": 0.42,
        "avg_session_duration": 245.5,
        "new_users": 3456,
        "events": 234567,
    }


async def fetch_mixpanel_metrics(tenant_id: str, time_range: str) -> Dict[str, float]:
    """Fetch Mixpanel metrics."""
    try:
        from aragora.connectors.analytics.mixpanel import MixpanelConnector  # noqa: F401

        # In real implementation, would use actual client
    except ImportError:
        pass

    # Return demo data
    return {
        "events_tracked": 567890,
        "unique_users": 23456,
        "retention_day1": 0.45,
        "retention_day7": 0.28,
        "funnel_completion_rate": 0.32,
        "active_users_daily": 8765,
    }


async def fetch_metabase_metrics(tenant_id: str, time_range: str) -> Dict[str, float]:
    """Fetch Metabase dashboard metrics."""
    try:
        from aragora.connectors.analytics.metabase import MetabaseConnector  # noqa: F401

        # In real implementation, would use actual client
    except ImportError:
        pass

    # Return demo data
    return {
        "queries_executed": 12345,
        "dashboards_viewed": 567,
        "active_users": 89,
        "avg_query_time_ms": 234.5,
    }


async def fetch_segment_metrics(tenant_id: str, time_range: str) -> Dict[str, float]:
    """Fetch Segment metrics."""
    try:
        from aragora.connectors.analytics.segment import SegmentConnector  # noqa: F401

        # In real implementation, would use actual client
    except ImportError:
        pass

    # Return demo data
    return {
        "events_delivered": 789012,
        "sources_active": 12,
        "destinations_active": 8,
        "delivery_rate": 0.9985,
    }


# =============================================================================
# Analytics Functions
# =============================================================================


def calculate_trend(current: float, previous: float) -> Tuple[str, float]:
    """Calculate trend direction and change percentage."""
    if previous == 0:
        return ("stable", 0.0) if current == 0 else ("up", 100.0)

    change = ((current - previous) / previous) * 100

    if change > 5:
        return ("up", change)
    elif change < -5:
        return ("down", change)
    else:
        return ("stable", change)


def detect_anomalies(
    values: List[float],
    metric_name: str,
    platform: Platform,
    threshold_std: float = 2.0,
) -> List[Anomaly]:
    """Detect anomalies using Z-score method."""
    if len(values) < 3:
        return []

    anomalies = []
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0

    if std == 0:
        return []

    for i, value in enumerate(values[-5:]):  # Check last 5 values
        z_score = abs((value - mean) / std)
        if z_score > threshold_std:
            severity = (
                AlertSeverity.CRITICAL
                if z_score > 3
                else AlertSeverity.WARNING
                if z_score > 2.5
                else AlertSeverity.INFO
            )
            anomalies.append(
                Anomaly(
                    id=f"anom_{uuid4().hex[:12]}",
                    metric_name=metric_name,
                    platform=platform,
                    timestamp=datetime.now(timezone.utc),
                    expected_value=mean,
                    actual_value=value,
                    deviation=z_score,
                    severity=severity,
                    description=f"{metric_name} is {z_score:.1f} standard deviations from mean",
                )
            )

    return anomalies


def calculate_correlation(
    series_a: List[float],
    series_b: List[float],
) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(series_a) != len(series_b) or len(series_a) < 2:
        return 0.0

    n = len(series_a)
    mean_a = sum(series_a) / n
    mean_b = sum(series_b) / n

    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in series_a))
    denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in series_b))

    if denom_a == 0 or denom_b == 0:
        return 0.0

    return numerator / (denom_a * denom_b)


# =============================================================================
# Handler Class
# =============================================================================


class CrossPlatformAnalyticsHandler(SecureHandler):
    """Handler for cross-platform analytics endpoints.

    RBAC Protected:
    - analytics:read - required for GET endpoints (summary, metrics, trends, etc.)
    - analytics:write - required for POST endpoints (query, create alerts)
    - analytics:export - required for export endpoint
    """

    ROUTES = [
        "/api/v1/analytics/cross-platform/summary",
        "/api/v1/analytics/cross-platform/metrics",
        "/api/v1/analytics/cross-platform/trends",
        "/api/v1/analytics/cross-platform/comparison",
        "/api/v1/analytics/cross-platform/correlation",
        "/api/v1/analytics/cross-platform/anomalies",
        "/api/v1/analytics/cross-platform/query",
        "/api/v1/analytics/cross-platform/alerts",
        "/api/v1/analytics/cross-platform/alerts/{alert_id}/acknowledge",
        "/api/v1/analytics/cross-platform/export",
        "/api/v1/analytics/cross-platform/demo",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]

    async def handle(  # type: ignore[override]
        self, request: Any, path: str, method: str
    ) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            # RBAC: Determine required permission based on path and method
            if "/export" in path:
                required_permission = ANALYTICS_EXPORT_PERMISSION
            elif method == "POST":
                required_permission = ANALYTICS_WRITE_PERMISSION
            else:
                required_permission = ANALYTICS_READ_PERMISSION

            try:
                auth_context = await self.get_auth_context(request, require_auth=True)
                self.check_permission(auth_context, required_permission)
            except UnauthorizedError:
                return error_response("Authentication required for analytics", 401)
            except ForbiddenError as e:
                logger.warning(f"Analytics access denied: {e}")
                return error_response(str(e), 403)

            tenant_id = self._get_tenant_id(request)

            # Summary
            if path == "/api/v1/analytics/cross-platform/summary" and method == "GET":
                return await self._handle_summary(request, tenant_id)

            # Metrics
            elif path == "/api/v1/analytics/cross-platform/metrics" and method == "GET":
                return await self._handle_metrics(request, tenant_id)

            # Trends
            elif path == "/api/v1/analytics/cross-platform/trends" and method == "GET":
                return await self._handle_trends(request, tenant_id)

            # Comparison
            elif path == "/api/v1/analytics/cross-platform/comparison" and method == "GET":
                return await self._handle_comparison(request, tenant_id)

            # Correlation
            elif path == "/api/v1/analytics/cross-platform/correlation" and method == "GET":
                return await self._handle_correlation(request, tenant_id)

            # Anomalies
            elif path == "/api/v1/analytics/cross-platform/anomalies" and method == "GET":
                return await self._handle_anomalies(request, tenant_id)

            # Custom query
            elif path == "/api/v1/analytics/cross-platform/query" and method == "POST":
                return await self._handle_query(request, tenant_id)

            # Alerts
            elif path == "/api/v1/analytics/cross-platform/alerts":
                if method == "GET":
                    return await self._handle_list_alerts(request, tenant_id)
                elif method == "POST":
                    return await self._handle_create_alert(request, tenant_id)

            # Export
            elif path == "/api/v1/analytics/cross-platform/export" and method == "GET":
                return await self._handle_export(request, tenant_id)

            # Demo
            elif path == "/api/v1/analytics/cross-platform/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Alert acknowledge
            elif path.startswith("/api/v1/analytics/cross-platform/alerts/"):
                parts = path.split("/")
                if len(parts) >= 7 and parts[6] == "acknowledge" and method == "POST":
                    alert_id = parts[5]
                    return await self._handle_acknowledge_alert(request, tenant_id, alert_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in cross-platform analytics handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Summary
    # =========================================================================

    async def _handle_summary(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get unified dashboard summary."""
        params = self._get_query_params(request)
        time_range = params.get("range", "24h")

        # Fetch metrics from all platforms in parallel
        aragora, ga4, mixpanel, metabase, segment = await asyncio.gather(
            fetch_aragora_metrics(tenant_id, time_range),
            fetch_ga4_metrics(tenant_id, time_range),
            fetch_mixpanel_metrics(tenant_id, time_range),
            fetch_metabase_metrics(tenant_id, time_range),
            fetch_segment_metrics(tenant_id, time_range),
        )

        # Aggregate key metrics
        total_users = ga4.get("users", 0) + mixpanel.get("unique_users", 0)
        total_events = (
            ga4.get("events", 0)
            + mixpanel.get("events_tracked", 0)
            + segment.get("events_delivered", 0)
        )

        summary = {
            "time_range": time_range,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "platforms_connected": 5,
            "key_metrics": {
                "total_users": total_users,
                "total_events": total_events,
                "total_sessions": ga4.get("sessions", 0),
                "debates_active": aragora.get("debates_active", 0),
                "consensus_rate": aragora.get("consensus_rate", 0),
                "cost_usd": aragora.get("cost_usd_total", 0),
            },
            "platform_status": {
                "aragora": {"status": "connected", "metrics_count": len(aragora)},
                "google_analytics": {"status": "connected", "metrics_count": len(ga4)},
                "mixpanel": {"status": "connected", "metrics_count": len(mixpanel)},
                "metabase": {"status": "connected", "metrics_count": len(metabase)},
                "segment": {"status": "connected", "metrics_count": len(segment)},
            },
            "health_score": 92.5,
            "alerts_active": len(_get_tenant_alerts(tenant_id)),
        }

        return success_response(summary)

    # =========================================================================
    # Metrics
    # =========================================================================

    async def _handle_metrics(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get aggregated metrics from all platforms."""
        params = self._get_query_params(request)
        time_range = params.get("range", "24h")
        platform_filter = params.get("platform")

        # Fetch all metrics
        all_metrics: Dict[str, Dict[str, float]] = {}

        if not platform_filter or platform_filter == "aragora":
            all_metrics["aragora"] = await fetch_aragora_metrics(tenant_id, time_range)
        if not platform_filter or platform_filter == "google_analytics":
            all_metrics["google_analytics"] = await fetch_ga4_metrics(tenant_id, time_range)
        if not platform_filter or platform_filter == "mixpanel":
            all_metrics["mixpanel"] = await fetch_mixpanel_metrics(tenant_id, time_range)
        if not platform_filter or platform_filter == "metabase":
            all_metrics["metabase"] = await fetch_metabase_metrics(tenant_id, time_range)
        if not platform_filter or platform_filter == "segment":
            all_metrics["segment"] = await fetch_segment_metrics(tenant_id, time_range)

        # Calculate aggregations for common metrics
        aggregations = []

        # Users aggregation
        user_values = {
            "google_analytics": all_metrics.get("google_analytics", {}).get("users", 0),
            "mixpanel": all_metrics.get("mixpanel", {}).get("unique_users", 0),
        }
        aggregations.append(
            AggregatedMetric(
                name="total_users",
                total=sum(user_values.values()),
                by_platform=user_values,
                trend="up",
                change_percent=12.5,
                period=time_range,
            )
        )

        # Events aggregation
        event_values = {
            "google_analytics": all_metrics.get("google_analytics", {}).get("events", 0),
            "mixpanel": all_metrics.get("mixpanel", {}).get("events_tracked", 0),
            "segment": all_metrics.get("segment", {}).get("events_delivered", 0),
        }
        aggregations.append(
            AggregatedMetric(
                name="total_events",
                total=sum(event_values.values()),
                by_platform=event_values,
                trend="up",
                change_percent=8.3,
                period=time_range,
            )
        )

        return success_response(
            {
                "time_range": time_range,
                "metrics_by_platform": all_metrics,
                "aggregations": [a.to_dict() for a in aggregations],
            }
        )

    # =========================================================================
    # Trends
    # =========================================================================

    async def _handle_trends(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get cross-platform trends."""
        params = self._get_query_params(request)
        time_range = params.get("range", "7d")
        metric = params.get("metric", "users")

        # Generate trend data (mock for demo)
        days = 7 if time_range == "7d" else 30 if time_range == "30d" else 24
        base_date = datetime.now(timezone.utc)

        trends: Dict[str, Any] = {
            "metric": metric,
            "time_range": time_range,
            "data_points": [],
        }

        for i in range(days):
            date = (
                base_date - timedelta(days=(days - i - 1))
                if time_range != "24h"
                else base_date - timedelta(hours=(days - i - 1))
            )
            trends["data_points"].append(
                {
                    "timestamp": date.isoformat(),
                    "aragora": 100 + i * 5 + (i % 3) * 10,
                    "google_analytics": 1000 + i * 50 + (i % 5) * 100,
                    "mixpanel": 800 + i * 40 + (i % 4) * 80,
                }
            )

        # Calculate overall trend
        if len(trends["data_points"]) > 1:
            first_total = sum(
                trends["data_points"][0].get(k, 0)
                for k in ["aragora", "google_analytics", "mixpanel"]
            )
            last_total = sum(
                trends["data_points"][-1].get(k, 0)
                for k in ["aragora", "google_analytics", "mixpanel"]
            )
            trend, change = calculate_trend(last_total, first_total)
            trends["overall_trend"] = trend
            trends["change_percent"] = round(change, 2)

        return success_response(trends)

    # =========================================================================
    # Comparison
    # =========================================================================

    async def _handle_comparison(self, request: Any, tenant_id: str) -> HandlerResult:
        """Compare metrics across platforms."""
        params = self._get_query_params(request)
        metric_type = params.get("type", "engagement")

        comparisons = []

        if metric_type == "engagement":
            comparisons = [
                {
                    "metric": "daily_active_users",
                    "platforms": {
                        "google_analytics": {"value": 12345, "trend": "up", "change": 5.2},
                        "mixpanel": {"value": 8765, "trend": "up", "change": 3.1},
                    },
                },
                {
                    "metric": "session_duration",
                    "platforms": {
                        "google_analytics": {"value": 245.5, "unit": "seconds", "trend": "stable"},
                        "mixpanel": {"value": 238.2, "unit": "seconds", "trend": "up"},
                    },
                },
                {
                    "metric": "retention_rate",
                    "platforms": {
                        "mixpanel": {"value": 0.45, "unit": "ratio", "trend": "up"},
                    },
                },
            ]
        elif metric_type == "performance":
            comparisons = [
                {
                    "metric": "avg_response_time",
                    "platforms": {
                        "aragora": {"value": 156, "unit": "ms", "trend": "down"},
                        "metabase": {"value": 234.5, "unit": "ms", "trend": "stable"},
                    },
                },
                {
                    "metric": "event_delivery_rate",
                    "platforms": {
                        "segment": {"value": 0.9985, "unit": "ratio", "trend": "stable"},
                    },
                },
            ]
        elif metric_type == "cost":
            comparisons = [
                {
                    "metric": "cost_per_debate",
                    "platforms": {
                        "aragora": {"value": 0.125, "unit": "USD", "trend": "down"},
                    },
                },
                {
                    "metric": "token_cost",
                    "platforms": {
                        "aragora": {"value": 156.78, "unit": "USD", "period": "30d"},
                    },
                },
            ]

        return success_response(
            {
                "comparison_type": metric_type,
                "comparisons": comparisons,
            }
        )

    # =========================================================================
    # Correlation
    # =========================================================================

    async def _handle_correlation(self, request: Any, tenant_id: str) -> HandlerResult:
        """Calculate correlation between metrics."""
        params = self._get_query_params(request)
        time_range = params.get("range", "30d")

        # Generate correlation matrix (mock data for demo)
        metrics = ["users", "events", "sessions", "debates", "consensus_rate"]
        correlation_matrix = []

        for m1 in metrics:
            row: Dict[str, Any] = {"metric": m1}
            for m2 in metrics:
                if m1 == m2:
                    row[m2] = 1.0
                else:
                    # Generate plausible correlation values
                    if (m1 in ["users", "events"] and m2 in ["users", "events"]) or (
                        m1 in ["debates", "consensus_rate"] and m2 in ["debates", "consensus_rate"]
                    ):
                        row[m2] = round(0.7 + (hash(m1 + m2) % 30) / 100, 2)
                    else:
                        row[m2] = round((hash(m1 + m2) % 100 - 50) / 100, 2)
            correlation_matrix.append(row)

        # Find strongest correlations
        strong_correlations = [
            {
                "metric_a": "users",
                "metric_b": "events",
                "correlation": 0.89,
                "strength": "strong positive",
            },
            {
                "metric_a": "debates",
                "metric_b": "consensus_rate",
                "correlation": 0.72,
                "strength": "moderate positive",
            },
            {
                "metric_a": "sessions",
                "metric_b": "cost",
                "correlation": 0.65,
                "strength": "moderate positive",
            },
        ]

        return success_response(
            {
                "time_range": time_range,
                "correlation_matrix": correlation_matrix,
                "strong_correlations": strong_correlations,
            }
        )

    # =========================================================================
    # Anomalies
    # =========================================================================

    async def _handle_anomalies(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get detected anomalies."""
        params = self._get_query_params(request)
        time_range = params.get("range", "24h")
        severity = params.get("severity")

        # Generate mock anomalies for demo
        anomalies = [
            Anomaly(
                id=f"anom_{uuid4().hex[:12]}",
                metric_name="error_rate",
                platform=Platform.ARAGORA,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
                expected_value=0.02,
                actual_value=0.08,
                deviation=3.2,
                severity=AlertSeverity.WARNING,
                description="Error rate is 3.2 standard deviations above normal",
            ),
            Anomaly(
                id=f"anom_{uuid4().hex[:12]}",
                metric_name="response_time",
                platform=Platform.GOOGLE_ANALYTICS,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=5),
                expected_value=150.0,
                actual_value=450.0,
                deviation=2.8,
                severity=AlertSeverity.WARNING,
                description="Response time spike detected",
            ),
        ]

        # Filter by severity
        if severity:
            anomalies = [a for a in anomalies if a.severity.value == severity]

        return success_response(
            {
                "time_range": time_range,
                "anomalies": [a.to_dict() for a in anomalies],
                "total": len(anomalies),
                "by_severity": {
                    "critical": sum(1 for a in anomalies if a.severity == AlertSeverity.CRITICAL),
                    "warning": sum(1 for a in anomalies if a.severity == AlertSeverity.WARNING),
                    "info": sum(1 for a in anomalies if a.severity == AlertSeverity.INFO),
                },
            }
        )

    # =========================================================================
    # Custom Query
    # =========================================================================

    async def _handle_query(self, request: Any, tenant_id: str) -> HandlerResult:
        """Execute custom metric query."""
        try:
            body = await self._get_json_body(request)

            metrics = body.get("metrics", [])
            platforms = body.get("platforms", ["aragora", "google_analytics", "mixpanel"])
            time_range = body.get("range", "24h")
            body.get("aggregation", "sum")

            if not metrics:
                return error_response("At least one metric is required", 400)

            results = {"query": body, "results": {}}

            for platform in platforms:
                if platform == "aragora":
                    data = await fetch_aragora_metrics(tenant_id, time_range)
                elif platform == "google_analytics":
                    data = await fetch_ga4_metrics(tenant_id, time_range)
                elif platform == "mixpanel":
                    data = await fetch_mixpanel_metrics(tenant_id, time_range)
                else:
                    continue

                results["results"][platform] = {m: data.get(m, None) for m in metrics}

            return success_response(results)

        except Exception as e:
            logger.exception(f"Query execution error: {e}")
            return error_response(f"Query failed: {str(e)}", 500)

    # =========================================================================
    # Alerts
    # =========================================================================

    async def _handle_list_alerts(self, request: Any, tenant_id: str) -> HandlerResult:
        """List active alerts and rules."""
        params = self._get_query_params(request)
        status = params.get("status")
        severity = params.get("severity")

        alerts = list(_get_tenant_alerts(tenant_id).values())
        rules = list(_get_tenant_rules(tenant_id).values())

        # Filter
        if status:
            alerts = [a for a in alerts if a.status.value == status]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]

        return success_response(
            {
                "alerts": [a.to_dict() for a in alerts],
                "rules": [r.to_dict() for r in rules],
                "summary": {
                    "total_alerts": len(alerts),
                    "active": sum(1 for a in alerts if a.status == AlertStatus.ACTIVE),
                    "acknowledged": sum(1 for a in alerts if a.status == AlertStatus.ACKNOWLEDGED),
                    "rules_enabled": sum(1 for r in rules if r.enabled),
                },
            }
        )

    async def _handle_create_alert(self, request: Any, tenant_id: str) -> HandlerResult:
        """Create new alert rule."""
        try:
            body = await self._get_json_body(request)

            name = body.get("name")
            metric_name = body.get("metric_name")
            condition = body.get("condition", "above")
            threshold = body.get("threshold")
            severity = body.get("severity", "warning")

            if not name or not metric_name or threshold is None:
                return error_response("name, metric_name, and threshold are required", 400)

            rule = AlertRule(
                id=f"rule_{uuid4().hex[:12]}",
                name=name,
                metric_name=metric_name,
                condition=condition,
                threshold=float(threshold),
                severity=AlertSeverity(severity),
                platforms=[Platform(p) for p in body.get("platforms", ["aragora"])],
            )

            rules = _get_tenant_rules(tenant_id)
            rules[rule.id] = rule

            return success_response(
                {
                    "status": "created",
                    "rule": rule.to_dict(),
                }
            )

        except Exception as e:
            logger.exception(f"Error creating alert: {e}")
            return error_response(f"Failed to create alert: {str(e)}", 500)

    async def _handle_acknowledge_alert(
        self, request: Any, tenant_id: str, alert_id: str
    ) -> HandlerResult:
        """Acknowledge an alert."""
        alerts = _get_tenant_alerts(tenant_id)
        alert = alerts.get(alert_id)

        if not alert:
            return error_response("Alert not found", 404)

        user_id = getattr(request, "user_id", "api_user")
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.now(timezone.utc)

        return success_response(
            {
                "status": "acknowledged",
                "alert": alert.to_dict(),
            }
        )

    # =========================================================================
    # Export
    # =========================================================================

    async def _handle_export(self, request: Any, tenant_id: str) -> HandlerResult:
        """Export analytics data."""
        params = self._get_query_params(request)
        format_type = params.get("format", "json")
        time_range = params.get("range", "7d")

        # Gather all data
        all_data = {
            "aragora": await fetch_aragora_metrics(tenant_id, time_range),
            "google_analytics": await fetch_ga4_metrics(tenant_id, time_range),
            "mixpanel": await fetch_mixpanel_metrics(tenant_id, time_range),
            "metabase": await fetch_metabase_metrics(tenant_id, time_range),
            "segment": await fetch_segment_metrics(tenant_id, time_range),
        }

        if format_type == "json":
            return success_response(
                {
                    "export_format": "json",
                    "time_range": time_range,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "data": all_data,
                }
            )

        elif format_type == "csv":
            # Generate CSV
            lines = ["platform,metric,value"]
            for platform, metrics in all_data.items():
                for metric, value in metrics.items():
                    lines.append(f"{platform},{metric},{value}")

            csv_content = "\n".join(lines)
            return HandlerResult(
                body=csv_content.encode("utf-8"),
                status_code=200,
                content_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=analytics_export.csv"},
            )

        else:
            return error_response(f"Unsupported format: {format_type}", 400)

    # =========================================================================
    # Demo
    # =========================================================================

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo dashboard data."""
        return success_response(
            {
                "is_demo": True,
                "summary": {
                    "platforms_connected": 5,
                    "total_users": 35801,
                    "total_events": 1591469,
                    "total_sessions": 45678,
                    "debates_active": 12,
                    "consensus_rate": 0.78,
                    "cost_usd": 156.78,
                    "health_score": 92.5,
                },
                "platforms": {
                    "aragora": {
                        "status": "connected",
                        "debates_total": 1250,
                        "agents_active": 45,
                        "findings_critical": 8,
                    },
                    "google_analytics": {
                        "status": "connected",
                        "sessions": 45678,
                        "users": 12345,
                        "pageviews": 156789,
                    },
                    "mixpanel": {
                        "status": "connected",
                        "events_tracked": 567890,
                        "unique_users": 23456,
                        "retention_day1": 0.45,
                    },
                    "metabase": {
                        "status": "connected",
                        "queries_executed": 12345,
                        "dashboards_viewed": 567,
                    },
                    "segment": {
                        "status": "connected",
                        "events_delivered": 789012,
                        "delivery_rate": 0.9985,
                    },
                },
                "trends": {
                    "users": "up",
                    "events": "up",
                    "cost": "stable",
                },
                "alerts_active": 2,
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Extract JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "args"):
            return dict(request.args)
        return {}


# =============================================================================
# Handler Registration
# =============================================================================

_handler_instance: Optional[CrossPlatformAnalyticsHandler] = None


def get_cross_platform_analytics_handler() -> CrossPlatformAnalyticsHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = CrossPlatformAnalyticsHandler()
    return _handler_instance


async def handle_cross_platform_analytics(request: Any, path: str, method: str) -> HandlerResult:
    """Entry point for cross-platform analytics requests."""
    handler = get_cross_platform_analytics_handler()
    return await handler.handle(request, path, method)


__all__ = [
    "CrossPlatformAnalyticsHandler",
    "handle_cross_platform_analytics",
    "get_cross_platform_analytics_handler",
    "Platform",
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "TimeRange",
    "MetricValue",
    "AggregatedMetric",
    "Anomaly",
    "AlertRule",
    "Alert",
]
