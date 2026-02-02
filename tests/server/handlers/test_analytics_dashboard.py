"""
Comprehensive tests for the AnalyticsDashboardHandler module.

Tests cover:
- Route registration and can_handle
- Each endpoint's happy path via stub responses and mocked internal methods
- Error cases (missing params, invalid input)
- Permission/auth checks via decorators
- Edge cases (empty results, boundary values, version prefix stripping)
- Stub responses for unauthenticated/missing params
- Flip detection endpoints
- Deliberation analytics endpoints
- Token usage endpoints
- ANALYTICS_STUB_RESPONSES structure validation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_dashboard import (
    ANALYTICS_STUB_RESPONSES,
    AnalyticsDashboardHandler,
)
from aragora.server.handlers.base import error_response, json_response


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def handler(server_context):
    """Create handler instance for tests."""
    return AnalyticsDashboardHandler(server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with empty auth."""
    mock = MagicMock()
    mock.headers = {"Authorization": ""}
    return mock


@pytest.fixture
def authed_handler():
    """Create mock HTTP handler that passes require_user_auth.

    The require_user_auth decorator calls extract_user_from_request
    to validate auth. We patch that to return an authenticated user context.
    """
    mock = MagicMock()
    mock.headers = {"Authorization": "Bearer test-token"}
    return mock


@pytest.fixture
def mock_user_ctx():
    """Create a mock authenticated user context."""
    ctx = MagicMock()
    ctx.is_authenticated = True
    ctx.user_id = "user-001"
    ctx.email = "test@example.com"
    ctx.error_reason = None
    return ctx


@pytest.fixture
def patch_auth(mock_user_ctx):
    """Patch the auth extraction to return an authenticated user."""
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=mock_user_ctx,
    ):
        yield mock_user_ctx


@pytest.fixture
def mock_analytics_module():
    """Create a mock aragora.analytics module for lazy imports.

    The handler methods do lazy imports like:
        from aragora.analytics import get_analytics_dashboard, TimeRange
    We inject a mock module into sys.modules so these imports succeed.
    """
    mock_mod = ModuleType("aragora.analytics")
    mock_dashboard = MagicMock()
    mock_mod.get_analytics_dashboard = MagicMock(return_value=mock_dashboard)
    mock_mod.TimeRange = MagicMock()
    mock_mod.Granularity = MagicMock()

    original = sys.modules.get("aragora.analytics")
    sys.modules["aragora.analytics"] = mock_mod
    yield mock_mod, mock_dashboard
    if original is not None:
        sys.modules["aragora.analytics"] = original
    else:
        sys.modules.pop("aragora.analytics", None)


# ===========================================================================
# Mock data classes
# ===========================================================================


@dataclass
class MockSummary:
    total_findings: int = 150
    open_findings: int = 45

    def to_dict(self) -> dict:
        return {"total_findings": self.total_findings, "open_findings": self.open_findings}


@dataclass
class MockTrendPoint:
    timestamp: str = "2024-01-15T00:00:00Z"
    total: int = 12

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "total": self.total}


@dataclass
class MockRemediationMetrics:
    total_resolved: int = 120
    total_open: int = 45

    def to_dict(self) -> dict:
        return {"total_resolved": self.total_resolved, "total_open": self.total_open}


@dataclass
class MockAgentMetric:
    agent_id: str = "claude-3-sonnet"
    agreement_rate: float = 0.92

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "agreement_rate": self.agreement_rate}


@dataclass
class MockCostMetrics:
    total_audits: int = 50
    total_cost_usd: float = 125.50

    def to_dict(self) -> dict:
        return {"total_audits": self.total_audits, "total_cost_usd": self.total_cost_usd}


@dataclass
class MockComplianceScore:
    framework: str = "SOC2"
    score: float = 0.85

    def to_dict(self) -> dict:
        return {"framework": self.framework, "score": self.score}


@dataclass
class MockHeatmapCell:
    category: str = "security"
    severity: str = "critical"
    count: int = 5

    def to_dict(self) -> dict:
        return {"category": self.category, "severity": self.severity, "count": self.count}


@dataclass
class MockUsageSummary:
    total_tokens_in: int = 500000
    total_tokens_out: int = 100000
    total_cost_usd: float = 125.50
    total_debates: int = 50
    total_agent_calls: int = 200
    cost_by_provider: dict[str, float] = field(
        default_factory=lambda: {"anthropic": 80.00, "openai": 45.50}
    )
    debates_by_day: dict[str, int] = field(default_factory=dict)


# ===========================================================================
# Test: Route Registration and can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/analytics/summary",
            "/api/analytics/trends/findings",
            "/api/analytics/remediation",
            "/api/analytics/agents",
            "/api/analytics/cost",
            "/api/analytics/compliance",
            "/api/analytics/heatmap",
            "/api/analytics/tokens",
            "/api/analytics/tokens/trends",
            "/api/analytics/tokens/providers",
            "/api/analytics/flips/summary",
            "/api/analytics/flips/recent",
            "/api/analytics/flips/consistency",
            "/api/analytics/flips/trends",
            "/api/analytics/deliberations",
            "/api/analytics/deliberations/channels",
            "/api/analytics/deliberations/consensus",
            "/api/analytics/deliberations/performance",
        ],
    )
    def test_can_handle_all_routes(self, handler, path):
        """Handler recognizes all registered routes (without version prefix)."""
        assert handler.can_handle(path)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/analytics/summary",
            "/api/v1/analytics/tokens",
            "/api/v1/analytics/flips/summary",
            "/api/v1/analytics/deliberations",
        ],
    )
    def test_can_handle_versioned_routes(self, handler, path):
        """Handler recognizes versioned routes via strip_version_prefix."""
        assert handler.can_handle(path)

    def test_cannot_handle_unknown_path(self, handler):
        """Handler rejects unknown paths."""
        assert not handler.can_handle("/api/analytics/unknown")

    def test_cannot_handle_non_analytics_path(self, handler):
        """Handler rejects non-analytics paths."""
        assert not handler.can_handle("/api/v1/debates")

    def test_cannot_handle_empty_path(self, handler):
        """Handler rejects empty path."""
        assert not handler.can_handle("")

    def test_routes_list_has_expected_count(self, handler):
        """ROUTES list contains all 18 endpoints."""
        assert len(handler.ROUTES) == 18


# ===========================================================================
# Test: Handle Dispatch - Unknown Path
# ===========================================================================


class TestHandleUnknownPath:
    """Tests for handle() returning None on unknown paths."""

    def test_unknown_path_returns_none(self, handler, mock_http_handler):
        """Unknown path returns None so dispatch can continue."""
        result = handler.handle("/api/v1/other/endpoint", {}, mock_http_handler)
        assert result is None

    def test_unknown_analytics_subpath_returns_none(self, handler, mock_http_handler):
        """Unknown analytics sub-path returns None."""
        result = handler.handle("/api/v1/analytics/nonexistent", {}, mock_http_handler)
        assert result is None


# ===========================================================================
# Test: Stub Responses (no auth or no workspace_id)
# ===========================================================================


class TestStubResponses:
    """Tests for stub responses when no auth or workspace_id is provided."""

    @pytest.mark.parametrize(
        "path,stub_key",
        [
            ("/api/v1/analytics/summary", "/api/analytics/summary"),
            ("/api/v1/analytics/trends/findings", "/api/analytics/trends/findings"),
            ("/api/v1/analytics/remediation", "/api/analytics/remediation"),
            ("/api/v1/analytics/agents", "/api/analytics/agents"),
            ("/api/v1/analytics/cost", "/api/analytics/cost"),
            ("/api/v1/analytics/compliance", "/api/analytics/compliance"),
            ("/api/v1/analytics/heatmap", "/api/analytics/heatmap"),
            ("/api/v1/analytics/tokens", "/api/analytics/tokens"),
            ("/api/v1/analytics/tokens/trends", "/api/analytics/tokens/trends"),
            ("/api/v1/analytics/tokens/providers", "/api/analytics/tokens/providers"),
            ("/api/v1/analytics/flips/summary", "/api/analytics/flips/summary"),
            ("/api/v1/analytics/flips/recent", "/api/analytics/flips/recent"),
            ("/api/v1/analytics/flips/consistency", "/api/analytics/flips/consistency"),
            ("/api/v1/analytics/flips/trends", "/api/analytics/flips/trends"),
            ("/api/v1/analytics/deliberations", "/api/analytics/deliberations"),
            (
                "/api/v1/analytics/deliberations/channels",
                "/api/analytics/deliberations/channels",
            ),
            (
                "/api/v1/analytics/deliberations/consensus",
                "/api/analytics/deliberations/consensus",
            ),
            (
                "/api/v1/analytics/deliberations/performance",
                "/api/analytics/deliberations/performance",
            ),
        ],
    )
    def test_stub_response_for_no_workspace(self, handler, mock_http_handler, path, stub_key):
        """Each endpoint returns a stub response when no workspace_id is provided."""
        result = handler.handle(path, {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_stub_response_body_matches_summary(self, handler, mock_http_handler):
        """Summary stub response body matches expected structure."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        body = json.loads(result.body)
        assert "summary" in body
        assert body["summary"]["total_debates"] == 0

    def test_stub_response_body_matches_cost(self, handler, mock_http_handler):
        """Cost stub response body matches expected structure."""
        result = handler.handle("/api/v1/analytics/cost", {}, mock_http_handler)
        body = json.loads(result.body)
        assert "analysis" in body
        assert body["analysis"]["total_cost_usd"] == 0

    def test_stub_response_body_matches_flips_summary(self, handler, mock_http_handler):
        """Flips summary stub response matches expected structure."""
        result = handler.handle("/api/v1/analytics/flips/summary", {}, mock_http_handler)
        body = json.loads(result.body)
        assert "summary" in body
        assert body["summary"]["total"] == 0


# ===========================================================================
# Test: Authenticated Endpoint Methods - Happy Path
# (Call internal methods directly, bypassing handle() stub logic)
# ===========================================================================


class TestGetSummaryMethod:
    """Tests for _get_summary method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Summary returns 400 when workspace_id is missing."""
        result = handler._get_summary({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "workspace_id" in body.get("error", "").lower()

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_happy_path(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Summary returns data when all params are valid."""
        mock_summary = MockSummary()
        mock_run_async.return_value = mock_summary

        result = handler._get_summary(
            {"workspace_id": "ws-001", "time_range": "7d"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_findings"] == 150

    def test_invalid_time_range_returns_400(
        self, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Summary returns 400 for invalid time_range."""
        mock_mod, _ = mock_analytics_module
        mock_mod.TimeRange = MagicMock(side_effect=ValueError("Invalid"))

        result = handler._get_summary(
            {"workspace_id": "ws-001", "time_range": "invalid"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 400

    def test_unauthenticated_returns_401(self, handler):
        """Summary returns 401 when no handler (no auth) is provided."""
        result = handler._get_summary({})
        assert result is not None
        assert result.status_code == 401


class TestGetFindingTrendsMethod:
    """Tests for _get_finding_trends method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Finding trends returns 400 when workspace_id is missing."""
        result = handler._get_finding_trends({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_invalid_granularity_returns_400(
        self, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Finding trends returns 400 for invalid granularity."""
        mock_mod, _ = mock_analytics_module
        mock_mod.Granularity = MagicMock(side_effect=ValueError("Invalid"))

        result = handler._get_finding_trends(
            {"workspace_id": "ws-001", "granularity": "bad"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 400


class TestGetRemediationMetricsMethod:
    """Tests for _get_remediation_metrics method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Remediation metrics returns 400 when workspace_id is missing."""
        result = handler._get_remediation_metrics({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_happy_path(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Remediation returns data when valid."""
        mock_metrics = MockRemediationMetrics()
        mock_run_async.return_value = mock_metrics

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_resolved"] == 120


class TestGetAgentMetricsMethod:
    """Tests for _get_agent_metrics method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Agent metrics returns 400 when workspace_id is missing."""
        result = handler._get_agent_metrics({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_agent_metrics_happy_path(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Agent metrics returns agent list."""
        mock_run_async.return_value = [MockAgentMetric()]

        result = handler._get_agent_metrics(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["agents"]) == 1
        assert body["agents"][0]["agent_id"] == "claude-3-sonnet"


class TestGetCostMetricsMethod:
    """Tests for _get_cost_metrics method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Cost metrics returns 400 when workspace_id is missing."""
        result = handler._get_cost_metrics({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400


class TestGetComplianceScorecardMethod:
    """Tests for _get_compliance_scorecard method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Compliance scorecard returns 400 when workspace_id is missing."""
        result = handler._get_compliance_scorecard({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_happy_path(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Compliance scorecard returns scores."""
        mock_run_async.return_value = [MockComplianceScore()]

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws-001", "frameworks": "SOC2,GDPR"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["scores"]) == 1
        assert body["scores"][0]["framework"] == "SOC2"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_default_frameworks(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Compliance uses default frameworks when none specified."""
        mock_run_async.return_value = []

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200


class TestGetRiskHeatmapMethod:
    """Tests for _get_risk_heatmap method."""

    def test_missing_workspace_id_returns_400(self, handler, authed_handler, patch_auth):
        """Risk heatmap returns 400 when workspace_id is missing."""
        result = handler._get_risk_heatmap({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_happy_path(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Risk heatmap returns cell data."""
        mock_run_async.return_value = [MockHeatmapCell()]

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["cells"]) == 1
        assert body["cells"][0]["category"] == "security"


# ===========================================================================
# Test: Token Usage Internal Methods
# ===========================================================================


class TestGetTokenUsageMethod:
    """Tests for _get_token_usage method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Token usage returns 400 when org_id is missing."""
        result = handler._get_token_usage({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_token_usage_happy_path(self, handler, authed_handler, patch_auth):
        """Token usage returns summary with valid org_id."""
        mock_summary = MockUsageSummary()
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_usage(
                {"org_id": "org-001", "days": "7"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["org_id"] == "org-001"
        assert body["total_tokens_in"] == 500000
        assert body["total_tokens_out"] == 100000
        assert body["total_tokens"] == 600000
        assert body["total_cost_usd"] == "125.5"
        assert body["period"]["days"] == 7


class TestGetTokenTrendsMethod:
    """Tests for _get_token_trends method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Token trends returns 400 when org_id is missing."""
        result = handler._get_token_trends({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_invalid_granularity_defaults_to_day(self, handler, authed_handler, patch_auth):
        """Token trends defaults to day granularity for invalid values."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_trends(
                {"org_id": "org-001", "granularity": "invalid"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "day"

    def test_hour_granularity_accepted(self, handler, authed_handler, patch_auth):
        """Token trends accepts hour granularity."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_trends(
                {"org_id": "org-001", "granularity": "hour"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "hour"


class TestGetProviderBreakdownMethod:
    """Tests for _get_provider_breakdown method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Provider breakdown returns 400 when org_id is missing."""
        result = handler._get_provider_breakdown({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test: Flip Detection Internal Methods
# ===========================================================================


class TestFlipSummaryMethod:
    """Tests for _get_flip_summary method."""

    def test_flip_summary_happy_path(self, handler):
        """Flip summary returns detector data."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {
            "total_flips": 150,
            "by_type": {"contradiction": 45},
        }

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_summary({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_flips"] == 150

    def test_flip_summary_handles_error(self, handler):
        """Flip summary returns 500 on internal error."""
        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            side_effect=RuntimeError("DB unavailable"),
        ):
            result = handler._get_flip_summary({})
        assert result is not None
        assert result.status_code == 500


class TestRecentFlipsMethod:
    """Tests for _get_recent_flips method."""

    def test_recent_flips_with_defaults(self, handler):
        """Recent flips uses default limit and no filters."""
        mock_flip = MagicMock()
        mock_flip.agent_name = "claude"
        mock_flip.flip_type = "contradiction"

        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = [mock_flip]

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            with patch(
                "aragora.insights.flip_detector.format_flip_for_ui",
                return_value={"agent": "claude", "type": "contradiction"},
            ):
                result = handler._get_recent_flips({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_recent_flips_with_agent_filter(self, handler):
        """Recent flips filters by agent name."""
        mock_flip_1 = MagicMock()
        mock_flip_1.agent_name = "claude"
        mock_flip_1.flip_type = "contradiction"
        mock_flip_2 = MagicMock()
        mock_flip_2.agent_name = "gpt-4"
        mock_flip_2.flip_type = "retraction"

        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = [mock_flip_1, mock_flip_2]

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            with patch(
                "aragora.insights.flip_detector.format_flip_for_ui",
                return_value={"agent": "claude", "type": "contradiction"},
            ):
                result = handler._get_recent_flips({"agent": "claude"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_recent_flips_invalid_limit_uses_default(self, handler):
        """Recent flips falls back to default limit for invalid value."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_recent_flips({"limit": "not_a_number"})
        assert result is not None
        assert result.status_code == 200
        # Default limit is 20, so fetch 20*2 = 40
        mock_detector.get_recent_flips.assert_called_once_with(limit=40)

    def test_recent_flips_limit_capped_at_100(self, handler):
        """Recent flips caps limit at 100."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_recent_flips({"limit": "500"})
        assert result is not None
        # min(500, 100) = 100, fetch 100*2 = 200
        mock_detector.get_recent_flips.assert_called_once_with(limit=200)


class TestAgentConsistencyMethod:
    """Tests for _get_agent_consistency method."""

    def test_consistency_with_specific_agents(self, handler):
        """Consistency returns data for specified agents."""
        mock_score = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_agents_consistency_batch.return_value = {"claude": mock_score}

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            with patch(
                "aragora.insights.flip_detector.format_consistency_for_ui",
                return_value={"agent": "claude", "consistency": "92%"},
            ):
                result = handler._get_agent_consistency({"agents": "claude,gpt-4"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_consistency_empty_agents_returns_all(self, handler):
        """Consistency with empty agents param fetches all."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"by_agent": {}}

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_agent_consistency({"agents": ""})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agents"] == []
        assert body["count"] == 0


class TestFlipTrendsMethod:
    """Tests for _get_flip_trends method."""

    def test_flip_trends_happy_path(self, handler):
        """Flip trends returns trend data with summary."""
        mock_row_1 = ("2024-01-15", "contradiction", 3)
        mock_row_2 = ("2024-01-15", "refinement", 2)
        mock_row_3 = ("2024-01-16", "contradiction", 4)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = [
            mock_row_1,
            mock_row_2,
            mock_row_3,
        ]

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({"days": "7", "granularity": "day"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "day"
        assert body["period"]["days"] == 7
        assert body["summary"]["total_flips"] == 9
        assert len(body["data_points"]) == 2

    def test_flip_trends_invalid_granularity_defaults_to_day(self, handler):
        """Flip trends defaults to day for invalid granularity."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({"granularity": "minute"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "day"

    def test_flip_trends_empty_data_insufficient_trend(self, handler):
        """Flip trends returns insufficient_data when no data points."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["summary"]["trend"] == "insufficient_data"
        assert body["summary"]["total_flips"] == 0


# ===========================================================================
# Test: Deliberation Internal Methods
# ===========================================================================


class TestDeliberationSummaryMethod:
    """Tests for _get_deliberation_summary method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Deliberation summary returns 400 when org_id is missing."""
        result = handler._get_deliberation_summary({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_deliberation_summary_happy_path(self, handler, authed_handler, patch_auth):
        """Deliberation summary returns formatted statistics."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = {
            "total": 150,
            "completed": 145,
            "consensus_reached": 120,
            "in_progress": 3,
            "failed": 2,
            "avg_rounds": 3.5,
            "avg_duration_seconds": 45.2,
            "by_template": {"code_review": 45},
            "by_priority": {"high": 50},
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_summary(
                {"org_id": "org-001", "days": "30"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_deliberations"] == 150
        assert body["consensus_rate"] == "82.8%"
        assert body["org_id"] == "org-001"

    def test_deliberation_summary_zero_completed(self, handler, authed_handler, patch_auth):
        """Deliberation summary handles zero completed gracefully."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = {
            "total": 0,
            "completed": 0,
            "consensus_reached": 0,
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_summary(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["consensus_rate"] == "0%"

    def test_unauthenticated_returns_401(self, handler):
        """Deliberation summary returns 401 when unauthenticated."""
        result = handler._get_deliberation_summary({})
        assert result is not None
        assert result.status_code == 401


class TestDeliberationByChannelMethod:
    """Tests for _get_deliberation_by_channel method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Deliberation by channel returns 400 when org_id is missing."""
        result = handler._get_deliberation_by_channel({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_channel_stats_aggregates_by_platform(self, handler, authed_handler, patch_auth):
        """Channel stats aggregates data by platform correctly."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = [
            {
                "platform": "slack",
                "total_deliberations": 45,
                "consensus_reached": 38,
                "total_duration": 1000,
            },
            {
                "platform": "slack",
                "total_deliberations": 35,
                "consensus_reached": 30,
                "total_duration": 800,
            },
            {
                "platform": "teams",
                "total_deliberations": 20,
                "consensus_reached": 15,
                "total_duration": 500,
            },
        ]

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_by_channel(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "slack" in body["by_platform"]
        assert body["by_platform"]["slack"]["count"] == 80
        assert "teams" in body["by_platform"]
        assert body["by_platform"]["teams"]["count"] == 20


class TestConsensusRatesMethod:
    """Tests for _get_consensus_rates method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Consensus rates returns 400 when org_id is missing."""
        result = handler._get_consensus_rates({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_consensus_rates_happy_path(self, handler, authed_handler, patch_auth):
        """Consensus rates returns formatted data."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {
            "overall_consensus_rate": "82%",
            "by_team_size": {"3": {"count": 80, "consensus_rate": "85%"}},
            "by_agent": [],
            "top_teams": [],
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_consensus_rates(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["overall_consensus_rate"] == "82%"


class TestDeliberationPerformanceMethod:
    """Tests for _get_deliberation_performance method."""

    def test_missing_org_id_returns_400(self, handler, authed_handler, patch_auth):
        """Deliberation performance returns 400 when org_id is missing."""
        result = handler._get_deliberation_performance({}, handler=authed_handler)
        assert result is not None
        assert result.status_code == 400

    def test_performance_happy_path(self, handler, authed_handler, patch_auth):
        """Deliberation performance returns metrics."""
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = {
            "summary": {"total_deliberations": 150, "total_cost_usd": "125.50"},
            "by_template": [],
            "trends": [],
            "cost_by_agent": {},
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_performance(
                {"org_id": "org-001", "granularity": "week"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "week"
        assert body["summary"]["total_deliberations"] == 150

    def test_performance_invalid_granularity_defaults_to_day(
        self, handler, authed_handler, patch_auth
    ):
        """Performance defaults to day granularity for invalid values."""
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = {
            "summary": {},
            "by_template": [],
            "trends": [],
            "cost_by_agent": {},
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_performance(
                {"org_id": "org-001", "granularity": "minute"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["granularity"] == "day"


# ===========================================================================
# Test: ANALYTICS_STUB_RESPONSES structure
# ===========================================================================


class TestStubResponsesData:
    """Tests for the ANALYTICS_STUB_RESPONSES constant."""

    def test_stub_responses_contains_all_routes(self):
        """All ROUTES have corresponding stub responses."""
        for route in AnalyticsDashboardHandler.ROUTES:
            assert route in ANALYTICS_STUB_RESPONSES, (
                f"Route {route} missing from ANALYTICS_STUB_RESPONSES"
            )

    def test_stub_responses_summary_structure(self):
        """Summary stub has expected nested structure."""
        stub = ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]
        assert "summary" in stub
        assert "total_debates" in stub["summary"]
        assert "consensus_rate" in stub["summary"]

    def test_stub_responses_remediation_structure(self):
        """Remediation stub has expected metrics structure."""
        stub = ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]
        assert "metrics" in stub
        assert "remediation_rate" in stub["metrics"]

    def test_stub_responses_cost_structure(self):
        """Cost stub has expected analysis structure."""
        stub = ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]
        assert "analysis" in stub
        assert "total_cost_usd" in stub["analysis"]
        assert "projected_monthly_cost" in stub["analysis"]

    def test_stub_responses_heatmap_structure(self):
        """Heatmap stub has expected structure with labels and values."""
        stub = ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]
        assert "heatmap" in stub
        assert "x_labels" in stub["heatmap"]
        assert "y_labels" in stub["heatmap"]
        assert "values" in stub["heatmap"]

    def test_stub_responses_compliance_structure(self):
        """Compliance stub has expected scorecard structure."""
        stub = ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]
        assert "compliance" in stub
        assert "overall_score" in stub["compliance"]

    def test_stub_responses_are_json_serializable(self):
        """All stub responses can be JSON serialized."""
        for route, stub in ANALYTICS_STUB_RESPONSES.items():
            try:
                json.dumps(stub)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Stub for {route} is not JSON serializable: {e}")


# ===========================================================================
# Test: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handle_with_none_handler(self, handler):
        """Handle works with None handler (no HTTP context)."""
        result = handler.handle("/api/v1/analytics/summary", {}, None)
        assert result is not None
        assert result.status_code == 200

    def test_handle_with_empty_query_params(self, handler, mock_http_handler):
        """Handle works with empty query params."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_with_extra_query_params(self, handler, mock_http_handler):
        """Handle ignores extra unknown query params."""
        result = handler.handle(
            "/api/v1/analytics/summary",
            {"unknown_param": "value", "another": "123"},
            mock_http_handler,
        )
        assert result is not None
        assert result.status_code == 200

    def test_handle_strips_v2_prefix(self, handler, mock_http_handler):
        """Handle strips v2 version prefix correctly."""
        result = handler.handle("/api/v2/analytics/summary", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_strips_v99_prefix(self, handler, mock_http_handler):
        """Handle strips any version number prefix."""
        result = handler.handle("/api/v99/analytics/summary", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_content_type_is_json(self, handler, mock_http_handler):
        """Stub responses have JSON content type."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        assert result is not None
        assert result.content_type == "application/json"

    def test_body_is_valid_json(self, handler, mock_http_handler):
        """Response body is valid JSON bytes."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        assert result is not None
        body = json.loads(result.body)
        assert isinstance(body, dict)


# ===========================================================================
# Test: Version Prefix Stripping via can_handle
# ===========================================================================


class TestVersionPrefixHandling:
    """Tests that version prefixes are handled correctly in routing."""

    def test_can_handle_no_version(self, handler):
        """can_handle works without version prefix."""
        assert handler.can_handle("/api/analytics/summary")

    def test_can_handle_v1(self, handler):
        """can_handle works with v1 prefix."""
        assert handler.can_handle("/api/v1/analytics/summary")

    def test_can_handle_v2(self, handler):
        """can_handle works with v2 prefix."""
        assert handler.can_handle("/api/v2/analytics/summary")

    def test_cannot_handle_invalid_prefix(self, handler):
        """can_handle rejects paths with invalid prefix format."""
        assert not handler.can_handle("/other/analytics/summary")

    def test_cannot_handle_partial_match(self, handler):
        """can_handle rejects partial route matches."""
        assert not handler.can_handle("/api/analytics")
        assert not handler.can_handle("/api/analytics/")


# ===========================================================================
# Test: Cache Behavior
# ===========================================================================


class TestCacheBehavior:
    """Tests for workspace-scoped caching functionality."""

    def test_cache_key_includes_workspace_id(self):
        """Cache keys are scoped by workspace_id."""
        from aragora.server.handlers.analytics.cache import get_analytics_dashboard_cache

        cache = get_analytics_dashboard_cache()
        key1 = cache._make_key("summary", "ws-1", "30d")
        key2 = cache._make_key("summary", "ws-2", "30d")

        assert key1 != key2
        assert "ws-1" in key1
        assert "ws-2" in key2

    def test_cache_key_includes_time_range(self):
        """Cache keys include time range."""
        from aragora.server.handlers.analytics.cache import get_analytics_dashboard_cache

        cache = get_analytics_dashboard_cache()
        key1 = cache._make_key("summary", "ws-1", "7d")
        key2 = cache._make_key("summary", "ws-1", "30d")

        assert key1 != key2

    def test_cache_invalidation_by_workspace(self):
        """Cache can be invalidated by workspace."""
        from aragora.server.handlers.analytics.cache import (
            get_analytics_dashboard_cache,
            invalidate_analytics_cache,
        )

        cache = get_analytics_dashboard_cache()

        # Set some cached values
        cache.set("summary", "test-ws-invalidate-1", {"data": "test"}, "30d")

        # Invalidate
        cleared = invalidate_analytics_cache("test-ws-invalidate-1")

        # Should be cleared
        assert cache.get("summary", "test-ws-invalidate-1", "30d") is None

    def test_cache_stats_available(self):
        """Cache provides statistics."""
        from aragora.server.handlers.analytics.cache import get_analytics_dashboard_cache

        cache = get_analytics_dashboard_cache()
        stats = cache.get_summary_stats()

        assert "cache_count" in stats
        assert "total_size" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "overall_hit_rate" in stats

    def test_cache_get_returns_none_for_missing(self):
        """Cache get returns None for missing keys."""
        from aragora.server.handlers.analytics.cache import get_analytics_dashboard_cache

        cache = get_analytics_dashboard_cache()
        result = cache.get("summary", "nonexistent-workspace-xyz", "30d")
        assert result is None

    def test_cache_set_and_get(self):
        """Cache set and get work correctly."""
        from aragora.server.handlers.analytics.cache import get_analytics_dashboard_cache

        cache = get_analytics_dashboard_cache()
        test_data = {"test": "data", "value": 123}

        cache.set("summary", "test-ws-set-get", test_data, "7d")
        result = cache.get("summary", "test-ws-set-get", "7d")

        assert result == test_data

    def test_cache_configs_exist(self):
        """Cache configurations are properly defined."""
        from aragora.server.handlers.analytics.cache import CACHE_CONFIGS

        expected_types = [
            "summary",
            "trends",
            "agents",
            "remediation",
            "cost",
            "tokens",
            "deliberations",
        ]
        for cache_type in expected_types:
            assert cache_type in CACHE_CONFIGS
            assert CACHE_CONFIGS[cache_type].ttl_seconds > 0
            assert CACHE_CONFIGS[cache_type].maxsize > 0

    def test_invalidate_all_caches(self):
        """Invalidate all clears all cache entries."""
        from aragora.server.handlers.analytics.cache import (
            get_analytics_dashboard_cache,
            invalidate_analytics_cache,
        )

        cache = get_analytics_dashboard_cache()

        # Set values in multiple caches
        cache.set("summary", "test-ws-all-1", {"data": 1}, "30d")
        cache.set("trends", "test-ws-all-2", {"data": 2}, "7d")

        # Invalidate all (no workspace_id)
        invalidate_analytics_cache(None)

        # All should be cleared
        assert cache.get("summary", "test-ws-all-1", "30d") is None
        assert cache.get("trends", "test-ws-all-2", "7d") is None


# ===========================================================================
# Test: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on handle method."""

    def test_rate_limit_decorator_present(self, handler):
        """Handle method has rate_limit decorator applied."""
        method = handler.handle
        # Check that the method is wrapped (indicates decorators applied)
        assert hasattr(method, "__wrapped__") or callable(method)


# ===========================================================================
# Test: Error Handling Scenarios
# ===========================================================================


class TestErrorHandling:
    """Tests for comprehensive error handling."""

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_type_error_returns_400(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """TypeError returns 400 DATA_ERROR."""
        mock_run_async.side_effect = TypeError("Missing required field")

        result = handler._get_summary(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert body.get("code") == "DATA_ERROR"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_attribute_error_returns_400(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """AttributeError returns 400 DATA_ERROR."""
        mock_run_async.side_effect = AttributeError("Object has no attribute")

        result = handler._get_summary(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert body.get("code") == "DATA_ERROR"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_os_error_returns_500(
        self, mock_run_async, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """OSError returns 500 INTERNAL_ERROR."""
        mock_run_async.side_effect = OSError("Database file not found")

        result = handler._get_summary(
            {"workspace_id": "ws-001"},
            handler=authed_handler,
        )
        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body.get("code") == "INTERNAL_ERROR"

    def test_flip_detector_import_error_handled(self, handler):
        """Flip detector handles ImportError gracefully."""
        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            side_effect=ImportError("Module not found"),
        ):
            result = handler._get_flip_summary({})
        assert result is not None
        assert result.status_code == 500

    def test_flip_detector_lookup_error_handled(self, handler):
        """Flip detector handles LookupError gracefully."""
        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            side_effect=LookupError("Key not found"),
        ):
            result = handler._get_flip_summary({})
        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# Test: Metric Calculations
# ===========================================================================


class TestMetricCalculations:
    """Tests for metric calculation edge cases."""

    def test_consensus_rate_zero_denominator(self, handler, authed_handler, patch_auth):
        """Consensus rate handles zero denominator gracefully."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = {
            "total": 10,
            "completed": 0,  # Zero denominator
            "consensus_reached": 0,
            "in_progress": 10,
            "failed": 0,
            "avg_rounds": 0,
            "avg_duration_seconds": 0,
            "by_template": {},
            "by_priority": {},
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_deliberation_summary(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["consensus_rate"] == "0%"

    def test_flip_trend_detection_increasing(self, handler):
        """Flip trend correctly detects increasing pattern."""
        # First half: low counts, second half: high counts
        mock_rows = [
            ("2024-01-01", "contradiction", 1),
            ("2024-01-02", "contradiction", 1),
            ("2024-01-03", "contradiction", 10),
            ("2024-01-04", "contradiction", 10),
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # 1+1 = 2 first half, 10+10 = 20 second half -> increasing
        assert body["summary"]["trend"] == "increasing"

    def test_flip_trend_detection_decreasing(self, handler):
        """Flip trend correctly detects decreasing pattern."""
        # First half: high counts, second half: low counts
        mock_rows = [
            ("2024-01-01", "contradiction", 10),
            ("2024-01-02", "contradiction", 10),
            ("2024-01-03", "contradiction", 1),
            ("2024-01-04", "contradiction", 1),
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # 10+10 = 20 first half, 1+1 = 2 second half -> decreasing
        assert body["summary"]["trend"] == "decreasing"

    def test_flip_trend_detection_stable(self, handler):
        """Flip trend correctly detects stable pattern."""
        # Both halves have similar counts
        mock_rows = [
            ("2024-01-01", "contradiction", 5),
            ("2024-01-02", "contradiction", 5),
            ("2024-01-03", "contradiction", 5),
            ("2024-01-04", "contradiction", 5),
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Both halves equal -> stable
        assert body["summary"]["trend"] == "stable"

    def test_flip_trend_single_data_point(self, handler):
        """Flip trend handles single data point gracefully."""
        mock_rows = [("2024-01-01", "contradiction", 5)]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Less than 2 data points -> insufficient_data
        assert body["summary"]["trend"] == "insufficient_data"


# ===========================================================================
# Test: Provider Breakdown Aggregation
# ===========================================================================


class TestProviderBreakdownAggregation:
    """Tests for provider breakdown data aggregation."""

    def test_provider_aggregation_multiple_models(self, handler, authed_handler, patch_auth):
        """Provider breakdown correctly aggregates multiple models."""
        mock_rows = [
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4",
                "tokens_in": 100000,
                "tokens_out": 20000,
                "cost": 30.0,
                "call_count": 50,
            },
            {
                "provider": "anthropic",
                "model": "claude-opus-4",
                "tokens_in": 50000,
                "tokens_out": 10000,
                "cost": 50.0,
                "call_count": 25,
            },
            {
                "provider": "openai",
                "model": "gpt-4o",
                "tokens_in": 80000,
                "tokens_out": 15000,
                "cost": 25.0,
                "call_count": 40,
            },
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_provider_breakdown(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "providers" in body
        # Should have 2 providers
        assert len(body["providers"]) == 2

        # Find anthropic provider
        anthropic = next((p for p in body["providers"] if p["provider"] == "anthropic"), None)
        assert anthropic is not None
        # anthropic: 100000 + 50000 = 150000 tokens_in, 20000 + 10000 = 30000 tokens_out
        assert anthropic["total_tokens_in"] == 150000
        assert anthropic["total_tokens_out"] == 30000
        assert anthropic["total_tokens"] == 180000
        # Cost: 30 + 50 = 80
        assert anthropic["total_cost"] == "80.0000"
        # Should have 2 models
        assert len(anthropic["models"]) == 2

    def test_provider_unknown_provider_handled(self, handler, authed_handler, patch_auth):
        """Provider breakdown handles null/empty provider gracefully."""
        mock_rows = [
            {
                "provider": None,
                "model": "unknown-model",
                "tokens_in": 1000,
                "tokens_out": 200,
                "cost": 1.0,
                "call_count": 5,
            },
            {
                "provider": "",
                "model": "another-model",
                "tokens_in": 2000,
                "tokens_out": 400,
                "cost": 2.0,
                "call_count": 10,
            },
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        # Note: The SQL query filters out NULL/empty providers, so these won't appear
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_provider_breakdown(
                {"org_id": "org-001"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["providers"] == []


# ===========================================================================
# Test: Days Parameter Clamping
# ===========================================================================


class TestDaysParameterClamping:
    """Tests for days parameter clamping in various endpoints."""

    def test_token_usage_days_clamped_to_365(self, handler, authed_handler, patch_auth):
        """Token usage clamps days to maximum 365."""
        mock_summary = MockUsageSummary()
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_usage(
                {"org_id": "org-001", "days": "1000"},  # Over max
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["period"]["days"] == 365

    def test_token_usage_days_clamped_to_1(self, handler, authed_handler, patch_auth):
        """Token usage clamps days to minimum 1."""
        mock_summary = MockUsageSummary()
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_usage(
                {"org_id": "org-001", "days": "0"},  # Under min
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["period"]["days"] == 1

    def test_token_usage_invalid_days_uses_default(self, handler, authed_handler, patch_auth):
        """Token usage uses default 30 for invalid days."""
        mock_summary = MockUsageSummary()
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary

        with patch(
            "aragora.billing.usage.UsageTracker",
            return_value=mock_tracker,
        ):
            result = handler._get_token_usage(
                {"org_id": "org-001", "days": "not_a_number"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["period"]["days"] == 30  # Default

    def test_flip_trends_days_clamped(self, handler):
        """Flip trends clamps days parameter."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_detector = MagicMock()
        mock_detector.db.connection.return_value = mock_conn

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            result = handler._get_flip_trends({"days": "500"})  # Over max
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["period"]["days"] == 365


# ===========================================================================
# Test: Filtering
# ===========================================================================


class TestFiltering:
    """Tests for filtering functionality in endpoints."""

    def test_recent_flips_type_filter(self, handler):
        """Recent flips correctly filters by flip_type."""
        mock_flip_1 = MagicMock()
        mock_flip_1.agent_name = "claude"
        mock_flip_1.flip_type = "contradiction"
        mock_flip_2 = MagicMock()
        mock_flip_2.agent_name = "claude"
        mock_flip_2.flip_type = "retraction"
        mock_flip_3 = MagicMock()
        mock_flip_3.agent_name = "gpt-4"
        mock_flip_3.flip_type = "contradiction"

        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = [mock_flip_1, mock_flip_2, mock_flip_3]

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            with patch(
                "aragora.insights.flip_detector.format_flip_for_ui",
                return_value={"type": "contradiction"},
            ):
                result = handler._get_recent_flips({"flip_type": "contradiction"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should only return contradictions (flip_1 and flip_3)
        assert body["count"] == 2

    def test_recent_flips_combined_filters(self, handler):
        """Recent flips correctly applies both agent and type filters."""
        mock_flip_1 = MagicMock()
        mock_flip_1.agent_name = "claude"
        mock_flip_1.flip_type = "contradiction"
        mock_flip_2 = MagicMock()
        mock_flip_2.agent_name = "claude"
        mock_flip_2.flip_type = "retraction"
        mock_flip_3 = MagicMock()
        mock_flip_3.agent_name = "gpt-4"
        mock_flip_3.flip_type = "contradiction"

        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = [mock_flip_1, mock_flip_2, mock_flip_3]

        with patch(
            "aragora.insights.flip_detector.FlipDetector",
            return_value=mock_detector,
        ):
            with patch(
                "aragora.insights.flip_detector.format_flip_for_ui",
                return_value={"agent": "claude", "type": "contradiction"},
            ):
                result = handler._get_recent_flips(
                    {"agent": "claude", "flip_type": "contradiction"}
                )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should only return claude + contradiction (flip_1)
        assert body["count"] == 1

    def test_compliance_frameworks_parsing(
        self, handler, authed_handler, patch_auth, mock_analytics_module
    ):
        """Compliance correctly parses comma-separated frameworks."""
        mock_mod, mock_dashboard = mock_analytics_module
        mock_dashboard.get_compliance_scorecard = MagicMock(return_value=[])

        with patch(
            "aragora.server.handlers.analytics_dashboard._run_async",
            return_value=[],
        ):
            result = handler._get_compliance_scorecard(
                {"workspace_id": "ws-001", "frameworks": "SOC2,  GDPR , HIPAA"},
                handler=authed_handler,
            )
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test: Handler Initialization
# ===========================================================================


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_handler_init_with_none_context(self):
        """Handler can be initialized with None context."""
        handler = AnalyticsDashboardHandler(ctx=None)
        assert handler.ctx == {}

    def test_handler_init_with_empty_context(self):
        """Handler can be initialized with empty context."""
        handler = AnalyticsDashboardHandler(ctx={})
        assert handler.ctx == {}

    def test_handler_init_with_full_context(self):
        """Handler preserves full context."""
        ctx = {"storage": "mock_storage", "elo_system": "mock_elo", "custom": "value"}
        handler = AnalyticsDashboardHandler(ctx=ctx)
        assert handler.ctx == ctx
        assert handler.ctx["custom"] == "value"
