"""
Analytics and Business Intelligence Connectors.

Integrations for analytics and BI platforms:
- Metabase (BI dashboards, SQL queries)
- Google Analytics (GA4 reporting)
- Mixpanel (product analytics)
- Amplitude - planned
- Segment - planned
"""

from aragora.connectors.analytics.metabase import (
    MetabaseConnector,
    MetabaseCredentials,
    Database,
    Table,
    Collection,
    Card,
    Dashboard,
    DashCard,
    QueryResult,
    MetabaseError,
    DisplayType,
    CollectionType,
    get_mock_card,
    get_mock_dashboard,
)
from aragora.connectors.analytics.google_analytics import (
    GoogleAnalyticsConnector,
    GoogleAnalyticsCredentials,
    Dimension,
    Metric,
    DateRange,
    FilterExpression,
    DimensionHeader,
    MetricHeader,
    Row,
    ReportResponse,
    RealTimeReport,
    GoogleAnalyticsError,
    MetricAggregation,
    OrderType,
    get_mock_report,
)
from aragora.connectors.analytics.mixpanel import (
    MixpanelConnector,
    MixpanelCredentials,
    Event,
    UserProfile,
    InsightResult,
    FunnelResult,
    RetentionResult,
    Cohort,
    MixpanelError,
    Unit,
    EventType,
    get_mock_event,
    get_mock_insight_result,
)

__all__ = [
    # Metabase
    "MetabaseConnector",
    "MetabaseCredentials",
    "Database",
    "Table",
    "Collection",
    "Card",
    "Dashboard",
    "DashCard",
    "QueryResult",
    "MetabaseError",
    "DisplayType",
    "CollectionType",
    "get_mock_card",
    "get_mock_dashboard",
    # Google Analytics
    "GoogleAnalyticsConnector",
    "GoogleAnalyticsCredentials",
    "Dimension",
    "Metric",
    "DateRange",
    "FilterExpression",
    "DimensionHeader",
    "MetricHeader",
    "Row",
    "ReportResponse",
    "RealTimeReport",
    "GoogleAnalyticsError",
    "MetricAggregation",
    "OrderType",
    "get_mock_report",
    # Mixpanel
    "MixpanelConnector",
    "MixpanelCredentials",
    "Event",
    "UserProfile",
    "InsightResult",
    "FunnelResult",
    "RetentionResult",
    "Cohort",
    "MixpanelError",
    "Unit",
    "EventType",
    "get_mock_event",
    "get_mock_insight_result",
]
