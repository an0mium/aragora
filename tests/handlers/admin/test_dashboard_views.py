"""Tests for DashboardViewsMixin in dashboard_views.py.

Comprehensive coverage of all 10 view endpoints:
- GET /api/v1/dashboard/overview        (_get_overview)
- GET /api/v1/dashboard/debates         (_get_dashboard_debates)
- GET /api/v1/dashboard/debates/{id}    (_get_dashboard_debate)
- GET /api/v1/dashboard/stats           (_get_dashboard_stats)
- GET /api/v1/dashboard/stat-cards      (_get_stat_cards)
- GET /api/v1/dashboard/team-performance          (_get_team_performance)
- GET /api/v1/dashboard/team-performance/{id}     (_get_team_performance_detail)
- GET /api/v1/dashboard/top-senders     (_get_top_senders)
- GET /api/v1/dashboard/labels          (_get_labels)
- GET /api/v1/dashboard/activity        (_get_activity)
- GET /api/v1/dashboard/inbox-summary   (_get_inbox_summary)

Also covers:
- Storage absence (get_storage returns None)
- SQL errors in storage queries
- Agent performance aggregation from ELO
- Team grouping by provider prefix
- Pagination (limit/offset)
- Error handling (exception paths)
- TTL cache interaction
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.admin.cache import clear_cache
from aragora.server.handlers.admin.dashboard_views import DashboardViewsMixin
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


# ===========================================================================
# In-memory SQLite storage for realistic SQL-level testing
# ===========================================================================


class InMemoryStorage:
    """Minimal storage with a real SQLite debates table."""

    def __init__(self, rows: list[tuple] | None = None):
        self._conn = sqlite3.connect(":memory:")
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                domain TEXT,
                status TEXT,
                consensus_reached INTEGER,
                confidence REAL,
                created_at TEXT
            )"""
        )
        if rows:
            cur.executemany(
                "INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        self._conn.commit()

    @contextmanager
    def connection(self):
        yield self._conn


class ErrorStorage:
    """Storage whose connection raises on cursor ops."""

    @contextmanager
    def connection(self):
        raise OSError("disk failure")


# ===========================================================================
# Testable concrete class wiring the mixin
# ===========================================================================


class TestableHandler(DashboardViewsMixin):
    """Concrete handler exposing mixin methods with controllable deps."""

    def __init__(
        self,
        storage: Any = None,
        agent_perf: dict[str, Any] | None = None,
        perf_metrics: dict[str, Any] | None = None,
        summary_metrics: dict[str, Any] | None = None,
    ):
        self._storage = storage
        self._agent_perf = agent_perf or {
            "top_performers": [],
            "total_agents": 0,
            "avg_elo": 0,
        }
        self._perf_metrics = perf_metrics or {
            "agents": {},
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "total_calls": 0,
        }
        self._summary_metrics = summary_metrics or {
            "total_debates": 0,
            "consensus_rate": 0.0,
            "avg_confidence": 0.0,
        }

    def get_storage(self):
        return self._storage

    def _get_summary_metrics_sql(self, storage, domain):
        return self._summary_metrics

    def _get_agent_performance(self, limit):
        return self._agent_perf

    def _get_performance_metrics(self):
        return self._perf_metrics


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_ttl_cache():
    """Clear TTL cache before every test to avoid cross-test pollution."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def handler():
    """A handler with no storage."""
    return TestableHandler()


@pytest.fixture
def mock_http():
    """Minimal mock HTTP handler."""
    h = MagicMock()
    h.path = "/api/v1/dashboard/overview"
    h.command = "GET"
    h.headers = {"Content-Type": "application/json"}
    h.client_address = ("127.0.0.1", 12345)
    return h


SAMPLE_ROWS = [
    ("d1", "finance", "completed", 1, 0.92, "2026-02-23T10:00:00"),
    ("d2", "tech", "completed", 0, 0.45, "2026-02-23T11:00:00"),
    ("d3", "finance", "in_progress", 0, 0.60, "2026-02-22T08:00:00"),
    ("d4", "legal", "completed", 1, 0.88, "2026-02-21T09:00:00"),
    ("d5", None, "completed", 1, 0.75, "2026-02-20T12:00:00"),
]


AGENT_PERF = {
    "top_performers": [
        {"name": "claude-opus", "elo": 1250, "debates_count": 20, "win_rate": 0.7, "wins": 14, "losses": 6, "draws": 0},
        {"name": "claude-sonnet", "elo": 1180, "debates_count": 15, "win_rate": 0.6, "wins": 9, "losses": 6, "draws": 0},
        {"name": "gpt-4", "elo": 1200, "debates_count": 18, "win_rate": 0.65, "wins": 12, "losses": 6, "draws": 0},
        {"name": "gpt-3.5", "elo": 1050, "debates_count": 10, "win_rate": 0.4, "wins": 4, "losses": 6, "draws": 0},
        {"name": "mistral-large", "elo": 1100, "debates_count": 12, "win_rate": 0.5, "wins": 6, "losses": 6, "draws": 0},
    ],
    "total_agents": 5,
    "avg_elo": 1156,
}


# ===========================================================================
# Tests: _get_overview
# ===========================================================================


class TestGetOverview:
    """Tests for the overview endpoint."""

    def test_overview_no_storage(self, handler, mock_http):
        result = handler._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["system_health"] == "healthy"
        assert body["active_debates"] == 0
        assert body["total_debates_today"] == 0
        assert body["consensus_rate"] == 0.0
        assert isinstance(body["stats"], list)
        assert isinstance(body["recent_debates"], list)
        assert "last_updated" in body

    def test_overview_with_storage(self, mock_http):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(
            storage=storage,
            summary_metrics={"consensus_rate": 0.85, "total_debates": 5},
        )
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["consensus_rate"] == 0.85

    def test_overview_today_count(self, mock_http):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        # today_count depends on current date vs sample data timestamps;
        # the key must be an integer regardless
        assert isinstance(body["total_debates_today"], int)

    def test_overview_stat_cards(self, mock_http):
        h = TestableHandler(agent_perf={"total_agents": 3, "avg_elo": 1100})
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert len(body["stats"]) == 2
        labels = {s["label"] for s in body["stats"]}
        assert "Total Agents" in labels
        assert "Avg ELO" in labels

    def test_overview_storage_connection_error(self, mock_http):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        # Graceful degradation: returns defaults
        assert body["total_debates_today"] == 0

    def test_overview_summary_metrics_error(self, mock_http):
        """When _get_summary_metrics_sql raises, overview returns defaults."""
        h = TestableHandler(storage=InMemoryStorage())
        h._get_summary_metrics_sql = MagicMock(side_effect=ValueError("bad"))
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["consensus_rate"] == 0.0

    def test_overview_agent_perf_error(self, mock_http):
        """When _get_agent_performance raises, overview still returns."""
        h = TestableHandler(storage=InMemoryStorage())
        h._get_agent_performance = MagicMock(side_effect=TypeError("bad"))
        result = h._get_overview({}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["stats"] == []


# ===========================================================================
# Tests: _get_dashboard_debates
# ===========================================================================


class TestGetDashboardDebates:
    """Tests for the debate list endpoint."""

    def test_empty_no_storage(self, handler):
        result = handler._get_dashboard_debates(10, 0, None)
        body = _body(result)
        assert _status(result) == 200
        assert body["debates"] == []
        assert body["total"] == 0

    def test_returns_debates(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(10, 0, None)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 5
        assert len(body["debates"]) == 5
        # Ordered by created_at DESC
        assert body["debates"][0]["id"] == "d2"

    def test_filter_by_status(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(50, 0, "completed")
        body = _body(result)
        assert body["total"] == 4
        assert all(d["status"] == "completed" for d in body["debates"])

    def test_filter_by_status_in_progress(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(50, 0, "in_progress")
        body = _body(result)
        assert body["total"] == 1
        assert body["debates"][0]["id"] == "d3"

    def test_pagination_limit(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(2, 0, None)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["debates"]) == 2

    def test_pagination_offset(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(2, 2, None)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["debates"]) == 2

    def test_pagination_past_end(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(10, 100, None)
        body = _body(result)
        assert body["total"] == 5
        assert body["debates"] == []

    def test_debate_fields(self):
        storage = InMemoryStorage(SAMPLE_ROWS[:1])
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(10, 0, None)
        d = _body(result)["debates"][0]
        assert d["id"] == "d1"
        assert d["domain"] == "finance"
        assert d["status"] == "completed"
        assert d["consensus_reached"] is True
        assert d["confidence"] == 0.92
        assert d["created_at"] == "2026-02-23T10:00:00"

    def test_consensus_reached_false(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(50, 0, None)
        body = _body(result)
        d2 = next(d for d in body["debates"] if d["id"] == "d2")
        assert d2["consensus_reached"] is False

    def test_storage_error_graceful(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_dashboard_debates(10, 0, None)
        body = _body(result)
        assert _status(result) == 200
        assert body["debates"] == []
        assert body["total"] == 0


# ===========================================================================
# Tests: _get_dashboard_debate (single debate detail)
# ===========================================================================


class TestGetDashboardDebate:
    """Tests for the single debate detail endpoint."""

    def test_returns_debate_id(self, handler):
        result = handler._get_dashboard_debate("abc-123")
        body = _body(result)
        assert _status(result) == 200
        assert body["debate_id"] == "abc-123"

    def test_empty_debate_id(self, handler):
        result = handler._get_dashboard_debate("")
        assert _status(result) == 400
        body = _body(result)
        assert "required" in body.get("error", "").lower()

    def test_special_characters_in_id(self, handler):
        result = handler._get_dashboard_debate("test/special%id")
        body = _body(result)
        assert _status(result) == 200
        assert body["debate_id"] == "test/special%id"


# ===========================================================================
# Tests: _get_dashboard_stats
# ===========================================================================


class TestGetDashboardStats:
    """Tests for the stats endpoint."""

    def test_stats_no_storage(self, handler):
        result = handler._get_dashboard_stats()
        body = _body(result)
        assert _status(result) == 200
        assert body["debates"]["total"] == 0
        assert body["debates"]["today"] == 0
        assert body["agents"]["total"] == 0
        assert body["performance"]["consensus_rate"] == 0.0
        assert body["usage"]["api_calls_today"] == 0

    def test_stats_with_storage(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(
            storage=storage,
            summary_metrics={"total_debates": 5, "consensus_rate": 0.8},
            agent_perf=AGENT_PERF,
            perf_metrics={"avg_latency_ms": 150.0, "success_rate": 0.95},
        )
        result = h._get_dashboard_stats()
        body = _body(result)
        assert body["debates"]["total"] == 5
        assert body["performance"]["consensus_rate"] == 0.8
        assert body["agents"]["total"] == 5
        assert body["agents"]["active"] == 5
        assert body["performance"]["avg_response_time_ms"] == 150.0
        assert body["performance"]["success_rate"] == 0.95

    def test_stats_error_rate_calculation(self):
        h = TestableHandler(
            perf_metrics={"avg_latency_ms": 0, "success_rate": 0.9},
        )
        result = h._get_dashboard_stats()
        body = _body(result)
        assert body["performance"]["error_rate"] == 0.1

    def test_stats_zero_success_rate_no_error_rate(self):
        h = TestableHandler(
            perf_metrics={"avg_latency_ms": 0, "success_rate": 0.0},
        )
        result = h._get_dashboard_stats()
        body = _body(result)
        # success_rate 0 means we skip the error_rate calc
        assert body["performance"]["error_rate"] == 0.0

    def test_stats_by_status(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_stats()
        body = _body(result)
        by_status = body["debates"]["by_status"]
        assert "completed" in by_status
        assert by_status["completed"] == 4
        assert by_status.get("in_progress", 0) == 1

    def test_stats_week_and_month_counts(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_stats()
        body = _body(result)
        # All sample rows are within the last 30 days
        assert isinstance(body["debates"]["this_week"], int)
        assert isinstance(body["debates"]["this_month"], int)

    def test_stats_storage_connection_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_dashboard_stats()
        body = _body(result)
        assert _status(result) == 200
        assert body["debates"]["today"] == 0

    def test_stats_agent_perf_error(self):
        h = TestableHandler()
        h._get_agent_performance = MagicMock(side_effect=KeyError("boom"))
        result = h._get_dashboard_stats()
        body = _body(result)
        assert _status(result) == 200
        assert body["agents"]["total"] == 0

    def test_stats_structure(self, handler):
        result = handler._get_dashboard_stats()
        body = _body(result)
        assert set(body.keys()) == {"debates", "agents", "performance", "usage"}
        assert set(body["debates"].keys()) == {"total", "today", "this_week", "this_month", "by_status"}
        assert set(body["agents"].keys()) == {"total", "active", "by_provider"}
        assert set(body["performance"].keys()) == {"avg_response_time_ms", "success_rate", "consensus_rate", "error_rate"}
        assert set(body["usage"].keys()) == {"api_calls_today", "tokens_used_today", "storage_used_bytes"}


# ===========================================================================
# Tests: _get_stat_cards
# ===========================================================================


class TestGetStatCards:
    """Tests for the stat cards endpoint."""

    def test_cards_no_storage(self, handler):
        result = handler._get_stat_cards()
        body = _body(result)
        assert _status(result) == 200
        # Without storage, only agent cards
        card_ids = [c["id"] for c in body["cards"]]
        assert "active_agents" in card_ids
        assert "avg_elo" in card_ids

    def test_cards_with_storage(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(
            storage=storage,
            summary_metrics={"total_debates": 10, "consensus_rate": 0.75, "avg_confidence": 0.82},
            agent_perf=AGENT_PERF,
        )
        result = h._get_stat_cards()
        body = _body(result)
        assert _status(result) == 200
        card_ids = [c["id"] for c in body["cards"]]
        assert "total_debates" in card_ids
        assert "consensus_rate" in card_ids
        assert "avg_confidence" in card_ids
        assert "active_agents" in card_ids
        assert "avg_elo" in card_ids
        assert len(body["cards"]) == 5

    def test_card_values(self):
        h = TestableHandler(
            storage=InMemoryStorage(),
            summary_metrics={"total_debates": 42, "consensus_rate": 0.6, "avg_confidence": 0.77},
            agent_perf={"total_agents": 3, "avg_elo": 1100, "top_performers": []},
        )
        result = h._get_stat_cards()
        cards = {c["id"]: c for c in _body(result)["cards"]}
        assert cards["total_debates"]["value"] == 42
        assert cards["consensus_rate"]["value"] == "60.0%"
        assert cards["avg_confidence"]["value"] == "0.77"
        assert cards["active_agents"]["value"] == 3
        assert cards["avg_elo"]["value"] == 1100

    def test_card_icons(self):
        h = TestableHandler(
            storage=InMemoryStorage(),
            summary_metrics={"total_debates": 0, "consensus_rate": 0, "avg_confidence": 0},
        )
        result = h._get_stat_cards()
        icons = {c["id"]: c["icon"] for c in _body(result)["cards"]}
        assert icons["total_debates"] == "message-circle"
        assert icons["consensus_rate"] == "check-circle"
        assert icons["avg_confidence"] == "trending-up"
        assert icons["active_agents"] == "users"
        assert icons["avg_elo"] == "award"

    def test_cards_error(self):
        h = TestableHandler()
        h._get_agent_performance = MagicMock(side_effect=ValueError("bad"))
        result = h._get_stat_cards()
        body = _body(result)
        assert _status(result) == 200
        assert isinstance(body["cards"], list)


# ===========================================================================
# Tests: _get_team_performance
# ===========================================================================


class TestGetTeamPerformance:
    """Tests for the team performance endpoint."""

    def test_empty_agents(self, handler):
        result = handler._get_team_performance(10, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["teams"] == []
        assert body["total"] == 0

    def test_teams_grouped_by_provider(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        body = _body(result)
        team_ids = {t["team_id"] for t in body["teams"]}
        assert "claude" in team_ids
        assert "gpt" in team_ids
        assert "mistral" in team_ids
        assert body["total"] == 3

    def test_team_sorted_by_avg_elo_desc(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        teams = _body(result)["teams"]
        elos = [t["avg_elo"] for t in teams]
        assert elos == sorted(elos, reverse=True)

    def test_team_member_count(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        teams = {t["team_id"]: t for t in _body(result)["teams"]}
        assert teams["claude"]["member_count"] == 2
        assert teams["gpt"]["member_count"] == 2
        assert teams["mistral"]["member_count"] == 1

    def test_team_total_debates(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        teams = {t["team_id"]: t for t in _body(result)["teams"]}
        assert teams["claude"]["total_debates"] == 35  # 20 + 15
        assert teams["gpt"]["total_debates"] == 28  # 18 + 10
        assert teams["mistral"]["total_debates"] == 12

    def test_team_avg_win_rate(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        teams = {t["team_id"]: t for t in _body(result)["teams"]}
        assert teams["claude"]["avg_win_rate"] == round((0.7 + 0.6) / 2, 3)
        assert teams["mistral"]["avg_win_rate"] == 0.5

    def test_team_name_titlecased(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 0)
        teams = _body(result)["teams"]
        for t in teams:
            assert t["team_name"] == t["team_id"].title()

    def test_team_pagination_limit(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(1, 0)
        body = _body(result)
        assert len(body["teams"]) == 1
        assert body["total"] == 3

    def test_team_pagination_offset(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 1)
        body = _body(result)
        assert len(body["teams"]) == 2
        assert body["total"] == 3

    def test_team_pagination_past_end(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance(10, 100)
        body = _body(result)
        assert body["teams"] == []
        assert body["total"] == 3

    def test_team_agent_no_dash(self):
        """Agent name without dash uses full name as provider."""
        perf = {
            "top_performers": [
                {"name": "grok", "elo": 1100, "debates_count": 5, "win_rate": 0.5},
            ],
            "total_agents": 1,
            "avg_elo": 1100,
        }
        h = TestableHandler(agent_perf=perf)
        result = h._get_team_performance(10, 0)
        teams = _body(result)["teams"]
        assert teams[0]["team_id"] == "grok"

    def test_team_perf_error(self):
        h = TestableHandler()
        h._get_agent_performance = MagicMock(side_effect=TypeError("bad"))
        result = h._get_team_performance(10, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["teams"] == []


# ===========================================================================
# Tests: _get_team_performance_detail
# ===========================================================================


class TestGetTeamPerformanceDetail:
    """Tests for team performance detail endpoint."""

    def test_empty_team_id(self, handler):
        result = handler._get_team_performance_detail("")
        assert _status(result) == 400
        body = _body(result)
        assert "required" in body.get("error", "").lower()

    def test_known_team(self):
        h = TestableHandler(
            agent_perf=AGENT_PERF,
            perf_metrics={"avg_latency_ms": 120.0},
        )
        result = h._get_team_performance_detail("claude")
        body = _body(result)
        assert _status(result) == 200
        assert body["team_id"] == "claude"
        assert body["team_name"] == "Claude"
        assert body["member_count"] == 2
        assert body["debates_participated"] == 35
        assert body["avg_response_time_ms"] == 120.0
        assert body["consensus_contribution_rate"] == round((0.7 + 0.6) / 2, 3)
        assert body["quality_score"] == round((1250 + 1180) / 2 / 1000, 2)
        assert len(body["members"]) == 2

    def test_unknown_team(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance_detail("nonexistent")
        body = _body(result)
        assert _status(result) == 200
        assert body["member_count"] == 0
        assert body["members"] == []
        assert body["debates_participated"] == 0

    def test_single_member_team(self):
        h = TestableHandler(agent_perf=AGENT_PERF)
        result = h._get_team_performance_detail("mistral")
        body = _body(result)
        assert body["member_count"] == 1
        assert body["debates_participated"] == 12
        assert body["quality_score"] == round(1100 / 1000, 2)

    def test_detail_perf_error(self):
        h = TestableHandler()
        h._get_agent_performance = MagicMock(side_effect=KeyError("bad"))
        result = h._get_team_performance_detail("claude")
        body = _body(result)
        assert _status(result) == 200
        assert body["member_count"] == 0


# ===========================================================================
# Tests: _get_top_senders
# ===========================================================================


class TestGetTopSenders:
    """Tests for the top senders endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_top_senders(10, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["senders"] == []
        assert body["total"] == 0

    def test_top_senders_grouped(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_top_senders(10, 0)
        body = _body(result)
        assert body["total"] > 0
        domains = [s["domain"] for s in body["senders"]]
        assert "finance" in domains

    def test_null_domain_becomes_general(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_top_senders(10, 0)
        senders = _body(result)["senders"]
        general = [s for s in senders if s["domain"] == "general"]
        assert len(general) == 1
        assert general[0]["debate_count"] == 1

    def test_senders_ordered_by_count_desc(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_top_senders(10, 0)
        senders = _body(result)["senders"]
        counts = [s["debate_count"] for s in senders]
        assert counts == sorted(counts, reverse=True)

    def test_senders_limit(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_top_senders(2, 0)
        body = _body(result)
        assert len(body["senders"]) == 2

    def test_senders_offset(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        all_result = h._get_top_senders(10, 0)
        offset_result = h._get_top_senders(10, 1)
        all_senders = _body(all_result)["senders"]
        offset_senders = _body(offset_result)["senders"]
        assert len(offset_senders) == len(all_senders) - 1

    def test_senders_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_top_senders(10, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["senders"] == []


# ===========================================================================
# Tests: _get_labels
# ===========================================================================


class TestGetLabels:
    """Tests for the labels endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_labels()
        body = _body(result)
        assert _status(result) == 200
        assert body["labels"] == []

    def test_labels_from_storage(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_labels()
        body = _body(result)
        assert len(body["labels"]) > 0
        names = [lbl["name"] for lbl in body["labels"]]
        assert "finance" in names

    def test_null_domain_becomes_general(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_labels()
        labels = _body(result)["labels"]
        general = [l for l in labels if l["name"] == "general"]
        assert len(general) == 1

    def test_labels_ordered_by_count_desc(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_labels()
        labels = _body(result)["labels"]
        counts = [l["count"] for l in labels]
        assert counts == sorted(counts, reverse=True)

    def test_labels_max_20(self):
        """Labels are limited to 20."""
        rows = [(f"d{i}", f"dom{i}", "completed", 1, 0.5, "2026-01-01T00:00:00") for i in range(25)]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_labels()
        labels = _body(result)["labels"]
        assert len(labels) <= 20

    def test_labels_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_labels()
        body = _body(result)
        assert _status(result) == 200
        assert body["labels"] == []


# ===========================================================================
# Tests: _get_activity
# ===========================================================================


class TestGetActivity:
    """Tests for the activity feed endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_activity(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["activity"] == []
        assert body["total"] == 0

    def test_activity_entries(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_activity(20, 0)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["activity"]) == 5

    def test_activity_fields(self):
        storage = InMemoryStorage(SAMPLE_ROWS[:1])
        h = TestableHandler(storage=storage)
        result = h._get_activity(20, 0)
        entry = _body(result)["activity"][0]
        assert entry["type"] == "debate"
        assert entry["debate_id"] == "d1"
        assert entry["domain"] == "finance"
        assert entry["consensus_reached"] is True
        assert entry["confidence"] == 0.92
        assert entry["created_at"] == "2026-02-23T10:00:00"

    def test_activity_ordered_desc(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_activity(20, 0)
        entries = _body(result)["activity"]
        dates = [e["created_at"] for e in entries]
        assert dates == sorted(dates, reverse=True)

    def test_activity_pagination(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_activity(2, 0)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["activity"]) == 2

    def test_activity_offset(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_activity(2, 3)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["activity"]) == 2

    def test_activity_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_activity(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["activity"] == []


# ===========================================================================
# Tests: _get_inbox_summary
# ===========================================================================


class TestGetInboxSummary:
    """Tests for the inbox summary endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_inbox_summary()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_messages"] == 0
        assert body["unread_messages"] == 0
        assert body["response_rate"] == 0.0
        assert body["by_importance"] == {"high": 0, "medium": 0, "low": 0}

    def test_inbox_with_storage(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(
            storage=storage,
            summary_metrics={"total_debates": 5, "consensus_rate": 0.8},
        )
        result = h._get_inbox_summary()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_messages"] == 5
        assert body["response_rate"] == 0.8

    def test_inbox_today_count(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_inbox_summary()
        body = _body(result)
        assert isinstance(body["today_count"], int)

    def test_inbox_structure(self, handler):
        result = handler._get_inbox_summary()
        body = _body(result)
        expected_keys = {
            "total_messages", "unread_messages", "urgent_count",
            "today_count", "by_label", "by_importance",
            "response_rate", "avg_response_time_hours",
        }
        assert set(body.keys()) == expected_keys

    def test_inbox_storage_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_inbox_summary()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_messages"] == 0

    def test_inbox_summary_sql_error(self):
        h = TestableHandler(storage=InMemoryStorage())
        h._get_summary_metrics_sql = MagicMock(side_effect=ValueError("oops"))
        result = h._get_inbox_summary()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_messages"] == 0


# ===========================================================================
# Tests: Integration / edge cases
# ===========================================================================


class TestIntegration:
    """Cross-cutting integration tests."""

    def test_all_endpoints_return_200(self, mock_http):
        """Every view endpoint returns 200 with empty handler."""
        h = TestableHandler()
        assert _status(h._get_overview({}, mock_http)) == 200
        assert _status(h._get_dashboard_debates(10, 0, None)) == 200
        assert _status(h._get_dashboard_debate("x")) == 200
        assert _status(h._get_dashboard_stats()) == 200
        assert _status(h._get_stat_cards()) == 200
        assert _status(h._get_team_performance(10, 0)) == 200
        assert _status(h._get_team_performance_detail("x")) == 200
        assert _status(h._get_top_senders(10, 0)) == 200
        assert _status(h._get_labels()) == 200
        assert _status(h._get_activity(20, 0)) == 200
        assert _status(h._get_inbox_summary()) == 200

    def test_all_responses_are_json(self, mock_http):
        """Every response has application/json content type."""
        h = TestableHandler()
        results = [
            h._get_overview({}, mock_http),
            h._get_dashboard_debates(10, 0, None),
            h._get_dashboard_debate("x"),
            h._get_dashboard_stats(),
            h._get_stat_cards(),
            h._get_team_performance(10, 0),
            h._get_team_performance_detail("x"),
            h._get_top_senders(10, 0),
            h._get_labels(),
            h._get_activity(20, 0),
            h._get_inbox_summary(),
        ]
        for r in results:
            assert r.content_type == "application/json"

    def test_empty_storage_table(self, mock_http):
        """Endpoints work with an empty debates table."""
        storage = InMemoryStorage([])
        h = TestableHandler(storage=storage)
        assert _body(h._get_dashboard_debates(10, 0, None))["total"] == 0
        assert _body(h._get_top_senders(10, 0))["total"] == 0
        assert _body(h._get_labels())["labels"] == []
        assert _body(h._get_activity(20, 0))["total"] == 0

    def test_large_dataset(self, mock_http):
        """Endpoints handle many rows without error."""
        rows = [
            (f"d{i}", f"dom{i % 5}", "completed", i % 2, i / 100, f"2026-01-{1 + i % 28:02d}T00:00:00")
            for i in range(200)
        ]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_dashboard_debates(10, 0, None)
        body = _body(result)
        assert body["total"] == 200
        assert len(body["debates"]) == 10

    def test_concurrent_different_args_produce_different_results(self, mock_http):
        """Different query params produce different debate lists."""
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        all_result = h._get_dashboard_debates(50, 0, None)
        completed = h._get_dashboard_debates(50, 0, "completed")
        assert _body(all_result)["total"] != _body(completed)["total"]

    def test_handler_result_iterable(self, handler, mock_http):
        """HandlerResult supports tuple-style unpacking."""
        result = handler._get_overview({}, mock_http)
        body_data, status_code, headers = result
        assert status_code == 200
        assert isinstance(body_data, dict)

    def test_handler_result_dict_access(self, handler, mock_http):
        """HandlerResult supports dict-like access."""
        result = handler._get_overview({}, mock_http)
        assert result["status"] == 200
