"""Debate analytics: overview, trends, topics, outcomes.

Endpoints handled:
- GET /api/analytics/summary         - Dashboard summary (cached: 60s)
- GET /api/analytics/trends/findings - Finding trends over time (cached: 300s)
- GET /api/analytics/remediation     - Remediation metrics (cached: 300s)
- GET /api/analytics/compliance      - Compliance scorecard
- GET /api/analytics/heatmap         - Risk heatmap data
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from ._shared import (
    HandlerResult,
    cached_analytics,
    error_response,
    handle_errors,
    json_response,
    logger,
    require_user_auth,
    safe_error_message,
)


# Access _run_async through the package module so that test patches on
# ``aragora.server.handlers.analytics_dashboard._run_async`` take effect.
def _get_run_async():
    return sys.modules[__package__]._run_async


class DebateAnalyticsMixin:
    """Mixin providing debate analytics endpoint methods."""

    @require_user_auth
    @handle_errors("get analytics summary")
    @cached_analytics("summary", workspace_key="workspace_id", time_range_key="time_range")
    def _get_summary(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get dashboard summary with key metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d

        Caching: 60s TTL, scoped by workspace_id + time_range
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            summary = _get_run_async()(dashboard.get_summary(workspace_id, time_range))

            return json_response(summary.to_dict())

        except ValueError as e:
            logger.warning(f"Invalid analytics summary parameter: {e}")
            return error_response(
                f"Invalid time_range: {time_range_str}", 400, code="INVALID_TIME_RANGE"
            )
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in analytics summary: {e}")
            return error_response(
                safe_error_message(e, "analytics summary"), 400, code="DATA_ERROR"
            )
        except (ImportError, RuntimeError, OSError) as e:
            logger.exception(f"Unexpected error getting analytics summary: {e}")
            return error_response(
                safe_error_message(e, "analytics summary"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get finding trends")
    @cached_analytics(
        "trends",
        workspace_key="workspace_id",
        time_range_key="time_range",
        extra_keys=["granularity"],
    )
    def _get_finding_trends(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get finding trends over time.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Time bucket size (hour, day, week, month) - default day

        Caching: 300s TTL, scoped by workspace_id + time_range + granularity
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400, code="MISSING_WORKSPACE_ID")

        time_range_str = query_params.get("time_range", "30d")
        granularity_str = query_params.get("granularity", "day")

        try:
            from aragora.analytics import (
                get_analytics_dashboard,
                TimeRange,
                Granularity,
            )

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)
            granularity = Granularity(granularity_str)

            trends = asyncio.run(
                dashboard.get_finding_trends(workspace_id, time_range, granularity)
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "granularity": granularity_str,
                    "trends": [t.to_dict() for t in trends],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid finding trends parameter: {e}")
            return error_response("Invalid parameter", 400, code="INVALID_PARAMETER")
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in finding trends: {e}")
            return error_response(safe_error_message(e, "finding trends"), 400, code="DATA_ERROR")
        except (ImportError, RuntimeError, OSError) as e:
            logger.exception(f"Unexpected error getting finding trends: {e}")
            return error_response(
                safe_error_message(e, "finding trends"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get remediation metrics")
    @cached_analytics("remediation", workspace_key="workspace_id", time_range_key="time_range")
    def _get_remediation_metrics(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get remediation performance metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Caching: 300s TTL, scoped by workspace_id + time_range
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400, code="MISSING_WORKSPACE_ID")

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _get_run_async()(dashboard.get_remediation_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    **metrics.to_dict(),
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid remediation metrics parameter: {e}")
            return error_response("Invalid parameter", 400, code="INVALID_PARAMETER")
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in remediation metrics: {e}")
            return error_response(
                safe_error_message(e, "remediation metrics"), 400, code="DATA_ERROR"
            )
        except (ImportError, RuntimeError, OSError) as e:
            logger.exception(f"Unexpected error getting remediation metrics: {e}")
            return error_response(
                safe_error_message(e, "remediation metrics"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get compliance scorecard")
    def _get_compliance_scorecard(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get compliance scorecard for specified frameworks.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - frameworks: Comma-separated list (SOC2,GDPR,HIPAA,PCI-DSS)
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400, code="MISSING_WORKSPACE_ID")

        frameworks_str = query_params.get("frameworks", "SOC2,GDPR,HIPAA,PCI-DSS")
        frameworks = [f.strip() for f in frameworks_str.split(",")]

        try:
            from aragora.analytics import get_analytics_dashboard

            dashboard = get_analytics_dashboard()

            scores = _get_run_async()(dashboard.get_compliance_scorecard(workspace_id, frameworks))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "scores": [s.to_dict() for s in scores],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid compliance scorecard parameter: {e}")
            return error_response("Invalid parameter", 400, code="INVALID_PARAMETER")
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in compliance scorecard: {e}")
            return error_response(
                safe_error_message(e, "compliance scorecard"), 400, code="DATA_ERROR"
            )
        except (ImportError, RuntimeError, OSError) as e:
            logger.exception(f"Unexpected error getting compliance scorecard: {e}")
            return error_response(
                safe_error_message(e, "compliance scorecard"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get risk heatmap")
    def _get_risk_heatmap(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get risk heatmap data (category x severity).

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400, code="MISSING_WORKSPACE_ID")

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            cells = _get_run_async()(dashboard.get_risk_heatmap(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "cells": [c.to_dict() for c in cells],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid risk heatmap parameter: {e}")
            return error_response("Invalid parameter", 400, code="INVALID_PARAMETER")
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in risk heatmap: {e}")
            return error_response(safe_error_message(e, "risk heatmap"), 400, code="DATA_ERROR")
        except (ImportError, RuntimeError, OSError) as e:
            logger.exception(f"Unexpected error getting risk heatmap: {e}")
            return error_response(safe_error_message(e, "risk heatmap"), 500, code="INTERNAL_ERROR")
