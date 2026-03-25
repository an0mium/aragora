from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.fastapi.routes import analytics as analytics_routes


def _auth() -> SimpleNamespace:
    return SimpleNamespace(user_id="user-1", email="user@example.com")


def _analytics_module(dashboard: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(
        get_analytics_dashboard=lambda: dashboard,
        TimeRange=lambda value: f"time:{value}",
        Granularity=lambda value: f"gran:{value}",
    )


def test_get_summary_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_summary = AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"debates": 12}))

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_summary(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                auth=_auth(),
            )
        )

    assert response.data["debates"] == 12
    dashboard.get_summary.assert_awaited_once_with("ws-1", "time:30d")


def test_get_finding_trends_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_finding_trends = AsyncMock(
        return_value=[SimpleNamespace(to_dict=lambda: {"bucket": "2026-03-24", "count": 3})]
    )

    with patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}):
        response = asyncio.run(
            analytics_routes.get_finding_trends(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                granularity=analytics_routes.GranularityEnum.day,
                auth=_auth(),
            )
        )

    assert response.trends[0]["count"] == 3
    dashboard.get_finding_trends.assert_awaited_once_with("ws-1", "time:30d", "gran:day")


def test_get_remediation_metrics_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_remediation_metrics = AsyncMock(
        return_value=SimpleNamespace(to_dict=lambda: {"total_findings": 4})
    )

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_remediation_metrics(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                auth=_auth(),
            )
        )

    assert response.total_findings == 4
    dashboard.get_remediation_metrics.assert_awaited_once_with("ws-1", "time:30d")


def test_get_agent_metrics_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_agent_metrics = AsyncMock(
        return_value=[SimpleNamespace(to_dict=lambda: {"agent": "gemini", "debates": 7})]
    )

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_agent_metrics(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                auth=_auth(),
            )
        )

    assert response.agents[0]["agent"] == "gemini"
    dashboard.get_agent_metrics.assert_awaited_once_with("ws-1", "time:30d")


def test_get_risk_heatmap_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_risk_heatmap = AsyncMock(
        return_value=[
            SimpleNamespace(
                to_dict=lambda: {"category": "security", "severity": "high", "value": 2}
            )
        ]
    )

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_risk_heatmap(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                auth=_auth(),
            )
        )

    assert response.cells[0]["category"] == "security"
    dashboard.get_risk_heatmap.assert_awaited_once_with("ws-1", "time:30d")


def test_get_cost_metrics_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_cost_metrics = AsyncMock(
        return_value=SimpleNamespace(to_dict=lambda: {"total_cost_usd": "1.23"})
    )

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_cost_metrics(
                workspace_id="ws-1",
                time_range=analytics_routes.TimeRangeEnum.d30,
                auth=_auth(),
            )
        )

    assert response.total_cost_usd == "1.23"
    dashboard.get_cost_metrics.assert_awaited_once_with("ws-1", "time:30d")


def test_get_compliance_scorecard_awaits_dashboard_call() -> None:
    dashboard = MagicMock()
    dashboard.get_compliance_scorecard = AsyncMock(
        return_value=[SimpleNamespace(to_dict=lambda: {"framework": "SOC2", "score": 0.9})]
    )

    with (
        patch.dict(sys.modules, {"aragora.analytics": _analytics_module(dashboard)}),
        patch(
            "aragora.server.http_utils.run_async",
            side_effect=AssertionError("sync bridge should not be used"),
        ),
    ):
        response = asyncio.run(
            analytics_routes.get_compliance_scorecard(
                workspace_id="ws-1",
                frameworks="SOC2,GDPR",
                auth=_auth(),
            )
        )

    assert response.scores[0]["framework"] == "SOC2"
    dashboard.get_compliance_scorecard.assert_awaited_once_with("ws-1", ["SOC2", "GDPR"])
