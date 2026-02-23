"""Comprehensive tests for DebateAnalyticsMixin endpoints.

Tests the four debate analytics mixin methods from
aragora/server/handlers/_analytics_metrics_debates.py:

- _get_debates_overview: Total debates, consensus rate, growth, avg metrics
- _get_debates_trends: Debate counts grouped by time granularity
- _get_debates_topics: Topic distribution with consensus rates
- _get_debates_outcomes: Outcome classification and confidence bucketing

Also tests routing via the AnalyticsMetricsHandler.handle() async method.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics import AnalyticsMetricsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Minimal mock HTTP handler for handle() routing tests."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 54321)
        self.headers: dict[str, str] = {"User-Agent": "test"}
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Debate factory helpers
# ---------------------------------------------------------------------------


def _make_debate(
    debate_id: str = "debate-1",
    task: str = "Test task",
    domain: str = "",
    consensus_reached: bool = True,
    confidence: float = 0.85,
    rounds_used: int = 3,
    agents: list[str] | None = None,
    outcome_type: str = "",
    created_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a debate dict matching what storage.list_debates returns."""
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now.isoformat()
    elif isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    result: dict[str, Any] = {
        "rounds_used": rounds_used,
        "confidence": confidence,
    }
    if domain:
        result["domain"] = domain
    if outcome_type:
        result["outcome_type"] = outcome_type

    debate = {
        "id": debate_id,
        "task": task,
        "consensus_reached": consensus_reached,
        "result": result,
        "agents": agents or ["agent-a", "agent-b"],
        "created_at": created_at,
    }
    if domain:
        debate["domain"] = domain
    return debate


def _recent_time(days_ago: int = 0) -> datetime:
    """Return a timezone-aware datetime *days_ago* days before now."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AnalyticsMetricsHandler with empty context."""
    return AnalyticsMetricsHandler({})


@pytest.fixture
def mock_storage():
    """Create a MagicMock storage that returns an empty debate list by default."""
    storage = MagicMock()
    storage.list_debates.return_value = []
    return storage


@pytest.fixture
def five_debates():
    """Five recent debates with varying attributes."""
    now = datetime.now(timezone.utc)
    return [
        _make_debate(
            debate_id=f"d-{i}",
            task=f"Task {i}",
            consensus_reached=(i % 3 != 0),  # d-0 and d-3 have no consensus
            confidence=0.6 + i * 0.08,
            rounds_used=2 + i,
            agents=[f"agent-{j}" for j in range(2 + i % 3)],
            created_at=(now - timedelta(days=i)).isoformat(),
        )
        for i in range(5)
    ]


@pytest.fixture
def http_handler():
    """Mock HTTP handler for async handle() tests."""
    return MockHTTPHandler()


# ============================================================================
# _get_debates_overview
# ============================================================================


class TestDebatesOverview:
    """Tests for _get_debates_overview."""

    def test_success_with_debates(self, handler, mock_storage, five_debates):
        """Overview returns correct aggregate metrics."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert _status(result) == 200
        assert body["time_range"] == "30d"
        assert body["total_debates"] == 5
        assert body["debates_this_period"] > 0
        assert "consensus_rate" in body
        assert "avg_rounds" in body
        assert "avg_agents_per_debate" in body
        assert "avg_confidence" in body
        assert "generated_at" in body

    def test_no_storage_returns_zeros(self, handler):
        """When storage is None, return zero-filled overview."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert _status(result) == 200
        assert body["total_debates"] == 0
        assert body["consensus_rate"] == 0.0
        assert body["avg_rounds"] == 0.0
        assert body["avg_agents_per_debate"] == 0.0
        assert body["avg_confidence"] == 0.0

    def test_empty_debate_list(self, handler, mock_storage):
        """Empty debate list returns zero metrics."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({})

        body = _body(result)
        assert body["total_debates"] == 0
        assert body["debates_this_period"] == 0

    def test_invalid_time_range_defaults_to_30d(self, handler, mock_storage, five_debates):
        """Invalid time_range silently defaults to 30d."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "garbage"})

        body = _body(result)
        assert body["time_range"] == "30d"
        assert _status(result) == 200

    def test_all_valid_time_ranges(self, handler, mock_storage, five_debates):
        """Every valid time range returns 200."""
        mock_storage.list_debates.return_value = five_debates
        for tr in ("7d", "14d", "30d", "90d", "180d", "365d", "all"):
            with patch.object(handler, "get_storage", return_value=mock_storage):
                result = handler._get_debates_overview({"time_range": tr})
            assert _status(result) == 200, f"Failed for {tr}"
            assert _body(result)["time_range"] == tr

    def test_time_range_all_includes_everything(self, handler, mock_storage):
        """time_range=all includes debates regardless of age."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        debates = [_make_debate(debate_id="old", created_at=old_date)]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "all"})

        body = _body(result)
        assert body["total_debates"] == 1
        assert body["debates_this_period"] == 1

    def test_growth_rate_calculation(self, handler, mock_storage):
        """Growth rate is (this - prev) / prev * 100."""
        now = datetime.now(timezone.utc)
        # 3 debates in last 30d, 2 debates in 30-60d ago
        debates = [
            _make_debate("r1", created_at=(now - timedelta(days=5)).isoformat()),
            _make_debate("r2", created_at=(now - timedelta(days=10)).isoformat()),
            _make_debate("r3", created_at=(now - timedelta(days=15)).isoformat()),
            _make_debate("p1", created_at=(now - timedelta(days=35)).isoformat()),
            _make_debate("p2", created_at=(now - timedelta(days=40)).isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        # 3 this period, 2 previous => growth = (3-2)/2*100 = 50.0
        assert body["debates_this_period"] == 3
        assert body["debates_previous_period"] == 2
        assert body["growth_rate"] == 50.0

    def test_growth_rate_zero_previous(self, handler, mock_storage):
        """Growth rate is 0 when no previous period debates."""
        debates = [_make_debate("r1", created_at=_recent_time(1).isoformat())]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["growth_rate"] == 0.0

    def test_consensus_rate_computation(self, handler, mock_storage):
        """Consensus rate = consensus_count / total * 100."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, created_at=now.isoformat()),
            _make_debate("d2", consensus_reached=True, created_at=now.isoformat()),
            _make_debate("d3", consensus_reached=False, created_at=now.isoformat()),
            _make_debate("d4", consensus_reached=True, created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["consensus_reached"] == 3
        assert body["consensus_rate"] == 75.0

    def test_avg_confidence_skips_zero(self, handler, mock_storage):
        """Only debates with confidence > 0 contribute to avg_confidence."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", confidence=0.9, created_at=now.isoformat()),
            _make_debate("d2", confidence=0.0, created_at=now.isoformat()),
            _make_debate("d3", confidence=0.8, created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        # avg of 0.9 and 0.8 = 0.85
        assert body["avg_confidence"] == 0.85

    def test_avg_agents_per_debate(self, handler, mock_storage):
        """Average agents counted from 'agents' list length."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", agents=["a", "b"], created_at=now.isoformat()),
            _make_debate("d2", agents=["a", "b", "c", "d"], created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["avg_agents_per_debate"] == 3.0  # (2+4)/2

    def test_debate_as_object_not_dict(self, handler, mock_storage):
        """Debates returned as objects (not dicts) are converted via vars()."""
        now = datetime.now(timezone.utc)

        class DebateObj:
            def __init__(self):
                self.id = "obj-debate"
                self.task = "Object task"
                self.consensus_reached = True
                self.result = {"rounds_used": 4, "confidence": 0.9}
                self.agents = ["x", "y", "z"]
                self.created_at = now.isoformat()

        mock_storage.list_debates.return_value = [DebateObj()]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 1
        assert body["avg_rounds"] == 4.0
        assert body["avg_agents_per_debate"] == 3.0

    def test_invalid_created_at_skipped(self, handler, mock_storage):
        """Debates with unparseable created_at are still counted in all_debates."""
        debates = [_make_debate("d1", created_at="not-a-date")]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        # The debate is in all_debates but not period_debates
        assert body["total_debates"] == 1
        assert body["debates_this_period"] == 0

    def test_empty_created_at_not_filtered_by_time(self, handler, mock_storage):
        """Debates with empty created_at not added to period_debates when start_time set."""
        debates = [_make_debate("d1", created_at="")]
        # Manually set created_at to empty string
        debates[0]["created_at"] = ""
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 1
        assert body["debates_this_period"] == 0

    def test_datetime_object_in_created_at(self, handler, mock_storage):
        """created_at can be a datetime object directly."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1")
        debate["created_at"] = now  # datetime object, not string
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["debates_this_period"] == 1

    def test_org_access_denied_returns_error(self, handler, mock_storage):
        """When org access validation fails, returns the error."""

        class NonAdminAuth:
            org_id = "org-user"
            roles = set()

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview(
                {"org_id": "org-other"}, auth_context=NonAdminAuth()
            )

        assert _status(result) == 403

    def test_admin_can_access_any_org(self, handler, mock_storage, five_debates):
        """Admin role bypasses org check."""

        class AdminAuth:
            org_id = "org-admin"
            roles = {"admin"}

        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview(
                {"org_id": "org-other"}, auth_context=AdminAuth()
            )

        assert _status(result) == 200

    def test_rounds_from_result_rounds_fallback(self, handler, mock_storage):
        """rounds_used falls back to result.rounds."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now.isoformat())
        debate["result"] = {"rounds": 7, "confidence": 0.8}  # 'rounds' not 'rounds_used'
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["avg_rounds"] == 7.0

    def test_result_not_dict_ignored(self, handler, mock_storage):
        """Non-dict result is silently ignored for rounds/confidence."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now.isoformat())
        debate["result"] = "not-a-dict"
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["avg_rounds"] == 0.0
        assert body["avg_confidence"] == 0.0

    def test_agents_not_list_ignored(self, handler, mock_storage):
        """Non-list agents field is ignored for agent counting."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now.isoformat())
        debate["agents"] = "not-a-list"
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        assert body["avg_agents_per_debate"] == 0.0


# ============================================================================
# _get_debates_trends
# ============================================================================


class TestDebatesTrends:
    """Tests for _get_debates_trends."""

    def test_success_with_data(self, handler, mock_storage, five_debates):
        """Trends return data_points grouped by granularity."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends(
                {"time_range": "30d", "granularity": "daily"}
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["time_range"] == "30d"
        assert body["granularity"] == "daily"
        assert isinstance(body["data_points"], list)
        assert "generated_at" in body

    def test_no_storage_returns_empty(self, handler):
        """No storage returns empty data_points."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_trends({})

        body = _body(result)
        assert body["data_points"] == []

    def test_invalid_granularity_defaults_to_daily(self, handler, mock_storage, five_debates):
        """Invalid granularity defaults to daily."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"granularity": "hourly"})

        body = _body(result)
        assert body["granularity"] == "daily"

    def test_invalid_time_range_defaults_to_30d(self, handler, mock_storage, five_debates):
        """Invalid time_range defaults to 30d."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"time_range": "xyz"})

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_weekly_granularity(self, handler, mock_storage, five_debates):
        """Weekly granularity groups data into weeks."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"granularity": "weekly"})

        body = _body(result)
        assert body["granularity"] == "weekly"
        for dp in body["data_points"]:
            assert "-W" in dp["period"]

    def test_monthly_granularity(self, handler, mock_storage, five_debates):
        """Monthly granularity groups data by year-month."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"granularity": "monthly"})

        body = _body(result)
        assert body["granularity"] == "monthly"
        for dp in body["data_points"]:
            # YYYY-MM format
            assert len(dp["period"]) == 7

    def test_data_point_metrics(self, handler, mock_storage):
        """Each data point has total, consensus_reached, consensus_rate, avg_rounds."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, rounds_used=3, created_at=now.isoformat()),
            _make_debate("d2", consensus_reached=False, rounds_used=5, created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends(
                {"time_range": "7d", "granularity": "daily"}
            )

        body = _body(result)
        assert len(body["data_points"]) >= 1
        dp = body["data_points"][0]
        assert dp["total"] == 2
        assert dp["consensus_reached"] == 1
        assert dp["consensus_rate"] == 50.0
        assert dp["avg_rounds"] == 4.0

    def test_unparseable_created_at_skipped(self, handler, mock_storage):
        """Debates with bad created_at are silently skipped."""
        debates = [
            _make_debate("d1", created_at="not-a-date"),
            _make_debate("d2", created_at=_recent_time(0).isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"time_range": "7d"})

        body = _body(result)
        total_in_points = sum(dp["total"] for dp in body["data_points"])
        assert total_in_points == 1

    def test_org_access_denied(self, handler, mock_storage):
        """Org access error propagated."""

        class UserAuth:
            org_id = "org-a"
            roles = set()

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends(
                {"org_id": "org-b"}, auth_context=UserAuth()
            )

        assert _status(result) == 403

    def test_time_range_all_no_filter(self, handler, mock_storage):
        """time_range=all includes all debates."""
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
        debates = [_make_debate("old", created_at=old)]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"time_range": "all"})

        body = _body(result)
        total = sum(dp["total"] for dp in body["data_points"])
        assert total == 1

    def test_empty_created_at_skipped(self, handler, mock_storage):
        """Debate with empty created_at is skipped in trends."""
        debate = _make_debate("d1")
        debate["created_at"] = ""
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"time_range": "30d"})

        body = _body(result)
        assert body["data_points"] == []

    def test_debate_object_converted_via_vars(self, handler, mock_storage):
        """Non-dict debates converted via vars()."""
        now = datetime.now(timezone.utc)

        class DebateObj:
            def __init__(self):
                self.id = "obj-1"
                self.task = "Task"
                self.consensus_reached = True
                self.result = {"rounds_used": 2, "confidence": 0.7}
                self.created_at = now.isoformat()

        mock_storage.list_debates.return_value = [DebateObj()]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"time_range": "30d"})

        body = _body(result)
        total = sum(dp["total"] for dp in body["data_points"])
        assert total == 1


# ============================================================================
# _get_debates_topics
# ============================================================================


class TestDebatesTopics:
    """Tests for _get_debates_topics."""

    def test_success_with_topics(self, handler, mock_storage):
        """Topics extracted from debate domain/task."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", domain="security", consensus_reached=True, created_at=now.isoformat()),
            _make_debate("d2", domain="security", consensus_reached=False, created_at=now.isoformat()),
            _make_debate("d3", domain="performance", consensus_reached=True, created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert _status(result) == 200
        assert body["total_debates"] == 3
        assert len(body["topics"]) == 2

        # security should be first (count=2)
        security = body["topics"][0]
        assert security["topic"] == "security"
        assert security["count"] == 2
        assert security["consensus_rate"] == 50.0

        perf = body["topics"][1]
        assert perf["topic"] == "performance"
        assert perf["count"] == 1
        assert perf["consensus_rate"] == 100.0

    def test_no_storage_returns_empty(self, handler):
        """No storage returns empty topics list."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_topics({})

        body = _body(result)
        assert body["topics"] == []
        assert body["total_debates"] == 0

    def test_limit_parameter(self, handler, mock_storage):
        """Limit restricts number of topics returned."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(f"d{i}", domain=f"topic-{i}", created_at=now.isoformat())
            for i in range(10)
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"limit": "3"})

        body = _body(result)
        assert len(body["topics"]) <= 3

    def test_default_limit_is_20(self, handler, mock_storage):
        """Default limit returns up to 20 topics."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(f"d{i}", domain=f"topic-{i}", created_at=now.isoformat())
            for i in range(25)
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({})

        body = _body(result)
        assert len(body["topics"]) == 20

    def test_topic_from_task_when_no_domain(self, handler, mock_storage):
        """When no domain, topic is first word of task (lowered)."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", domain="", task="Refactor the module", created_at=now.isoformat())
        debate.pop("domain", None)
        debate["result"] = {"confidence": 0.8}
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert body["topics"][0]["topic"] == "refactor"

    def test_topic_general_when_no_task_no_domain(self, handler, mock_storage):
        """When no domain and no task, topic defaults to 'general'."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", domain="", task="", created_at=now.isoformat())
        debate.pop("domain", None)
        debate["result"] = {"confidence": 0.8}
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert body["topics"][0]["topic"] == "general"

    def test_domain_from_result_metadata(self, handler, mock_storage):
        """Domain in result takes precedence over top-level domain."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", domain="top-level", created_at=now.isoformat())
        debate["result"]["domain"] = "from-result"
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert body["topics"][0]["topic"] == "from-result"

    def test_percentage_calculation(self, handler, mock_storage):
        """Percentage is (count / total_debates * 100)."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", domain="a", created_at=now.isoformat()),
            _make_debate("d2", domain="a", created_at=now.isoformat()),
            _make_debate("d3", domain="b", created_at=now.isoformat()),
            _make_debate("d4", domain="b", created_at=now.isoformat()),
            _make_debate("d5", domain="c", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        # a and b both have 40%, c has 20%
        for topic in body["topics"]:
            if topic["topic"] == "c":
                assert topic["percentage"] == 20.0
            else:
                assert topic["percentage"] == 40.0

    def test_invalid_time_range_defaults(self, handler, mock_storage):
        """Invalid time_range defaults to 30d."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "bogus"})

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_old_debates_filtered_by_time_range(self, handler, mock_storage):
        """Debates older than time range are excluded."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()
        recent = now.isoformat()
        debates = [
            _make_debate("old", domain="old-topic", created_at=old),
            _make_debate("new", domain="new-topic", created_at=recent),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 1
        assert body["topics"][0]["topic"] == "new-topic"

    def test_unparseable_created_at_skipped(self, handler, mock_storage):
        """Debate with bad date is skipped when time range filtering."""
        debates = [_make_debate("d1", domain="x", created_at="bad-date")]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 0
        assert body["topics"] == []

    def test_org_access_denied(self, handler, mock_storage):
        """Non-admin cannot access other org."""

        class UserAuth:
            org_id = "org-mine"
            roles = set()

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics(
                {"org_id": "org-other"}, auth_context=UserAuth()
            )

        assert _status(result) == 403

    def test_topic_lowercased(self, handler, mock_storage):
        """Domain is lowercased for topic aggregation."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", domain="Security", created_at=now.isoformat()),
            _make_debate("d2", domain="SECURITY", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        assert len(body["topics"]) == 1
        assert body["topics"][0]["topic"] == "security"
        assert body["topics"][0]["count"] == 2

    def test_consensus_rate_per_topic(self, handler, mock_storage):
        """Consensus rate is computed per-topic."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", domain="x", consensus_reached=True, created_at=now.isoformat()),
            _make_debate("d2", domain="x", consensus_reached=True, created_at=now.isoformat()),
            _make_debate("d3", domain="x", consensus_reached=False, created_at=now.isoformat()),
            _make_debate("d4", domain="y", consensus_reached=False, created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"time_range": "30d"})

        body = _body(result)
        topics_by_name = {t["topic"]: t for t in body["topics"]}
        # x: 2 consensus / 3 total = 66.7%
        assert topics_by_name["x"]["consensus_rate"] == pytest.approx(66.7, abs=0.1)
        # y: 0 consensus / 1 total = 0%
        assert topics_by_name["y"]["consensus_rate"] == 0.0


# ============================================================================
# _get_debates_outcomes
# ============================================================================


class TestDebatesOutcomes:
    """Tests for _get_debates_outcomes."""

    def test_success_with_outcomes(self, handler, mock_storage):
        """Outcomes are classified correctly."""
        now = datetime.now(timezone.utc)
        debates = [
            # consensus outcome: consensus_reached + confidence >= 0.8
            _make_debate("d1", consensus_reached=True, confidence=0.9,
                         outcome_type="consensus", created_at=now.isoformat()),
            # majority: consensus_reached + confidence >= 0.5
            _make_debate("d2", consensus_reached=True, confidence=0.6,
                         outcome_type="majority", created_at=now.isoformat()),
            # dissent: not consensus + confidence >= 0.3
            _make_debate("d3", consensus_reached=False, confidence=0.4,
                         outcome_type="dissent", created_at=now.isoformat()),
            # no_resolution: not consensus + confidence < 0.3
            _make_debate("d4", consensus_reached=False, confidence=0.1,
                         outcome_type="", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert _status(result) == 200
        assert body["outcomes"]["consensus"] == 1
        assert body["outcomes"]["majority"] == 1
        assert body["outcomes"]["dissent"] == 1
        assert body["outcomes"]["no_resolution"] == 1
        assert body["total_debates"] == 4

    def test_no_storage_returns_zero_outcomes(self, handler):
        """No storage returns zeroed outcomes."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_outcomes({})

        body = _body(result)
        assert _status(result) == 200
        assert body["outcomes"] == {
            "consensus": 0, "majority": 0, "dissent": 0, "no_resolution": 0
        }
        assert body["total_debates"] == 0
        assert body["by_confidence"] == {}

    def test_invalid_time_range_defaults_to_30d(self, handler, mock_storage):
        """Invalid time_range defaults to 30d."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "bad"})

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_confidence_bucketing(self, handler, mock_storage):
        """Debates bucketed by confidence: high >= 0.8, medium >= 0.5, low < 0.5."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, confidence=0.9,
                         outcome_type="consensus", created_at=now.isoformat()),
            _make_debate("d2", consensus_reached=True, confidence=0.85,
                         outcome_type="consensus", created_at=now.isoformat()),
            _make_debate("d3", consensus_reached=False, confidence=0.6,
                         outcome_type="majority", created_at=now.isoformat()),
            _make_debate("d4", consensus_reached=False, confidence=0.2,
                         outcome_type="", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        by_conf = body["by_confidence"]
        # high: 2 debates (0.9, 0.85), both consensus_reached
        assert by_conf["high"]["count"] == 2
        assert by_conf["high"]["consensus_rate"] == 100.0
        # medium: 1 debate (0.6), not consensus_reached
        assert by_conf["medium"]["count"] == 1
        assert by_conf["medium"]["consensus_rate"] == 0.0
        # low: 1 debate (0.2)
        assert by_conf["low"]["count"] == 1

    def test_empty_confidence_buckets_omitted(self, handler, mock_storage):
        """Only non-empty confidence buckets are in by_confidence."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, confidence=0.95,
                         outcome_type="consensus", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        # Only "high" should be present
        assert "high" in body["by_confidence"]
        assert "medium" not in body["by_confidence"]
        assert "low" not in body["by_confidence"]

    def test_result_not_dict_consensus_fallback(self, handler, mock_storage):
        """Non-dict result: consensus_reached -> consensus, else no_resolution."""
        now = datetime.now(timezone.utc)
        d1 = _make_debate("d1", consensus_reached=True, created_at=now.isoformat())
        d1["result"] = "not-dict"
        d2 = _make_debate("d2", consensus_reached=False, created_at=now.isoformat())
        d2["result"] = "not-dict"
        mock_storage.list_debates.return_value = [d1, d2]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["consensus"] == 1
        assert body["outcomes"]["no_resolution"] == 1

    def test_consensus_by_high_confidence_without_outcome_type(self, handler, mock_storage):
        """consensus_reached + confidence >= 0.8 -> consensus (no outcome_type)."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=True, confidence=0.9,
                              outcome_type="", created_at=now.isoformat())
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["consensus"] == 1

    def test_majority_by_medium_confidence_without_outcome_type(self, handler, mock_storage):
        """consensus_reached + confidence >= 0.5 but < 0.8 -> majority (no outcome_type)."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=True, confidence=0.6,
                              outcome_type="", created_at=now.isoformat())
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["majority"] == 1

    def test_dissent_with_moderate_confidence(self, handler, mock_storage):
        """Not consensus + confidence >= 0.3 -> dissent."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=False, confidence=0.35,
                              outcome_type="", created_at=now.isoformat())
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["dissent"] == 1

    def test_no_resolution_low_confidence_no_consensus(self, handler, mock_storage):
        """Not consensus + confidence < 0.3 -> no_resolution."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=False, confidence=0.1,
                              outcome_type="", created_at=now.isoformat())
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["no_resolution"] == 1

    def test_old_debates_filtered(self, handler, mock_storage):
        """Old debates are filtered out for 7d time range."""
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        debates = [_make_debate("old", created_at=old)]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "7d"})

        body = _body(result)
        assert body["total_debates"] == 0

    def test_time_range_all_includes_all(self, handler, mock_storage):
        """time_range=all includes all debates regardless of age."""
        old = (datetime.now(timezone.utc) - timedelta(days=999)).isoformat()
        debates = [_make_debate("old", created_at=old)]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "all"})

        body = _body(result)
        assert body["total_debates"] == 1

    def test_org_access_denied(self, handler, mock_storage):
        """Non-admin denied access to other org."""

        class UserAuth:
            org_id = "org-x"
            roles = set()

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes(
                {"org_id": "org-y"}, auth_context=UserAuth()
            )

        assert _status(result) == 403

    def test_datetime_object_in_created_at(self, handler, mock_storage):
        """created_at as datetime object works."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=True, confidence=0.9,
                              outcome_type="consensus")
        debate["created_at"] = now
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 1

    def test_unparseable_created_at_skipped(self, handler, mock_storage):
        """Bad created_at skipped in outcomes."""
        debates = [_make_debate("d1", created_at="nope")]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["total_debates"] == 0

    def test_explicit_dissent_outcome_type(self, handler, mock_storage):
        """outcome_type='dissent' categorized as dissent even with higher confidence."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", consensus_reached=False, confidence=0.7,
                              outcome_type="dissent", created_at=now.isoformat())
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert body["outcomes"]["dissent"] == 1

    def test_generated_at_present(self, handler, mock_storage):
        """Response always includes generated_at timestamp."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        assert "generated_at" in body


# ============================================================================
# Async handle() routing tests
# ============================================================================


class TestHandleRouting:
    """Tests for routing through the async handle() method."""

    @pytest.mark.asyncio
    async def test_route_debates_overview(self, handler, mock_storage, five_debates, http_handler):
        """handle() routes /api/v1/analytics/debates/overview to _get_debates_overview."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/overview",
                {"time_range": "30d"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "total_debates" in body

    @pytest.mark.asyncio
    async def test_route_debates_trends(self, handler, mock_storage, five_debates, http_handler):
        """handle() routes /api/v1/analytics/debates/trends."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/trends",
                {"time_range": "7d", "granularity": "daily"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "data_points" in body

    @pytest.mark.asyncio
    async def test_route_debates_topics(self, handler, mock_storage, five_debates, http_handler):
        """handle() routes /api/v1/analytics/debates/topics."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/topics",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "topics" in body

    @pytest.mark.asyncio
    async def test_route_debates_outcomes(self, handler, mock_storage, five_debates, http_handler):
        """handle() routes /api/v1/analytics/debates/outcomes."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/outcomes",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "outcomes" in body

    @pytest.mark.asyncio
    async def test_route_unversioned_path(self, handler, mock_storage, five_debates, http_handler):
        """handle() accepts unversioned /api/analytics/debates/overview."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_route_returns_none(self, handler, http_handler):
        """handle() returns None for unrecognized routes."""
        result = await handler.handle(
            "/api/v1/analytics/debates/unknown-endpoint",
            {},
            http_handler,
        )

        assert result is None


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_all_debate_routes(self, handler):
        """All debate analytics routes are recognized."""
        routes = [
            "/api/analytics/debates/overview",
            "/api/analytics/debates/trends",
            "/api/analytics/debates/topics",
            "/api/analytics/debates/outcomes",
            "/api/v1/analytics/debates/overview",
            "/api/v1/analytics/debates/trends",
            "/api/v1/analytics/debates/topics",
            "/api/v1/analytics/debates/outcomes",
        ]
        for route in routes:
            assert handler.can_handle(route), f"can_handle failed for {route}"

    def test_unknown_route_not_handled(self, handler):
        """Unrecognized routes return False."""
        assert not handler.can_handle("/api/v1/analytics/debates/unknown")
        assert not handler.can_handle("/api/v1/other/endpoint")
        assert not handler.can_handle("/random/path")


# ============================================================================
# Edge cases and integration scenarios
# ============================================================================


class TestEdgeCases:
    """Additional edge-case and integration tests."""

    def test_z_suffix_datetime_parsing(self, handler, mock_storage):
        """ISO dates ending with Z are parsed correctly."""
        debate = _make_debate("d1", created_at="2026-02-01T12:00:00Z")
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 1

    def test_large_debate_set(self, handler, mock_storage):
        """Handler processes a large number of debates without error."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(
                f"d-{i}",
                domain=f"topic-{i % 10}",
                consensus_reached=(i % 2 == 0),
                confidence=0.5 + (i % 5) * 0.1,
                rounds_used=2 + i % 4,
                created_at=(now - timedelta(hours=i)).isoformat(),
            )
            for i in range(200)
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            overview = handler._get_debates_overview({"time_range": "30d"})
            trends = handler._get_debates_trends({"time_range": "30d"})
            topics = handler._get_debates_topics({"time_range": "30d"})
            outcomes = handler._get_debates_outcomes({"time_range": "30d"})

        assert _status(overview) == 200
        assert _status(trends) == 200
        assert _status(topics) == 200
        assert _status(outcomes) == 200

        assert _body(overview)["total_debates"] == 200
        assert _body(topics)["total_debates"] == 200
        assert _body(outcomes)["total_debates"] == 200

    def test_missing_query_params_use_defaults(self, handler, mock_storage):
        """Empty query_params use defaults: time_range=30d, granularity=daily."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            overview = handler._get_debates_overview({})
            trends = handler._get_debates_trends({})
            topics = handler._get_debates_topics({})
            outcomes = handler._get_debates_outcomes({})

        assert _body(overview)["time_range"] == "30d"
        assert _body(trends)["time_range"] == "30d"
        assert _body(trends)["granularity"] == "daily"
        assert _body(topics)["time_range"] == "30d"
        assert _body(outcomes)["time_range"] == "30d"

    def test_storage_list_debates_receives_org_id(self, handler, mock_storage):
        """Storage receives org_id from _validate_org_access."""

        class AdminAuth:
            org_id = "admin-org"
            roles = {"admin"}

        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            handler._get_debates_overview(
                {"org_id": "target-org"}, auth_context=AdminAuth()
            )

        mock_storage.list_debates.assert_called_with(limit=10000, org_id="target-org")

    def test_no_org_id_uses_user_org(self, handler, mock_storage):
        """When no org_id in query, user's own org_id is used."""

        class UserAuth:
            org_id = "user-org"
            roles = set()

        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            handler._get_debates_overview({}, auth_context=UserAuth())

        mock_storage.list_debates.assert_called_with(limit=10000, org_id="user-org")

    def test_consensus_reached_false_default(self, handler, mock_storage):
        """Missing consensus_reached defaults to False in outcomes."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now.isoformat())
        del debate["consensus_reached"]
        mock_storage.list_debates.return_value = [debate]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({"time_range": "30d"})

        body = _body(result)
        # no consensus_reached field -> False -> not consensus, check confidence
        assert body["total_debates"] == 1

    def test_rounding_precision(self, handler, mock_storage):
        """Numeric fields are rounded to expected precision."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, confidence=0.777,
                         rounds_used=3, agents=["a", "b", "c"],
                         created_at=now.isoformat()),
            _make_debate("d2", consensus_reached=False, confidence=0.333,
                         rounds_used=5, agents=["a"],
                         created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

        body = _body(result)
        # growth_rate, consensus_rate, avg_rounds, avg_agents: 1 decimal
        assert isinstance(body["growth_rate"], float)
        assert isinstance(body["consensus_rate"], float)
        assert isinstance(body["avg_rounds"], float)
        assert isinstance(body["avg_agents_per_debate"], float)
        # avg_confidence: 2 decimals
        assert isinstance(body["avg_confidence"], float)
