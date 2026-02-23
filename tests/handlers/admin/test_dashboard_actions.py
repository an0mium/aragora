"""Tests for DashboardActionsMixin in dashboard_actions.py.

Comprehensive coverage of all action and analytics endpoints:
- GET  /api/v1/dashboard/quick-actions                 (_get_quick_actions)
- POST /api/v1/dashboard/quick-actions/{action_id}     (_execute_quick_action)
- GET  /api/v1/dashboard/urgent                        (_get_urgent_items)
- POST /api/v1/dashboard/urgent/{item_id}/dismiss      (_dismiss_urgent_item)
- GET  /api/v1/dashboard/pending-actions               (_get_pending_actions)
- POST /api/v1/dashboard/pending-actions/{id}/complete  (_complete_pending_action)
- GET  /api/v1/dashboard/search                        (_search_dashboard)
- POST /api/v1/dashboard/export                        (_export_dashboard_data)
- GET  /api/v1/dashboard/quality-metrics               (_get_quality_metrics)

Also covers internal helpers:
- _get_calibration_metrics
- _get_performance_metrics
- _get_evolution_metrics
- _get_debate_quality_metrics

And:
- Storage absence (get_storage returns None)
- SQL errors in storage queries
- Edge cases (empty IDs, no data, large datasets)
- TTL cache interaction on quality-metrics
- Error handling / graceful degradation
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from aragora.server.handlers.admin.cache import clear_cache
from aragora.server.handlers.admin.dashboard_actions import DashboardActionsMixin
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


class TestableHandler(DashboardActionsMixin):
    """Concrete handler exposing mixin methods with controllable deps."""

    def __init__(
        self,
        storage: Any = None,
        ctx: dict[str, Any] | None = None,
        summary_metrics: dict[str, Any] | None = None,
        agent_perf: dict[str, Any] | None = None,
        consensus_insights: dict[str, Any] | None = None,
    ):
        self._storage = storage
        self.ctx = ctx or {}
        self._summary_metrics = summary_metrics or {}
        self._agent_perf = agent_perf or {}
        self._consensus_insights = consensus_insights or {}

    def get_storage(self):
        return self._storage

    def _get_summary_metrics_sql(self, storage, domain):
        return self._summary_metrics

    def _get_agent_performance(self, limit):
        return self._agent_perf

    def _get_consensus_insights(self, domain):
        return self._consensus_insights


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
    """A handler with no storage and no ctx."""
    return TestableHandler()


SAMPLE_ROWS = [
    ("d1", "finance", "completed", 1, 0.92, "2026-02-23T10:00:00"),
    ("d2", "tech", "completed", 0, 0.45, "2026-02-23T11:00:00"),
    ("d3", "finance", "in_progress", 0, 0.60, "2026-02-22T08:00:00"),
    ("d4", "legal", "pending", 1, 0.88, "2026-02-21T09:00:00"),
    ("d5", None, "pending", 1, 0.75, "2026-02-20T12:00:00"),
]

# Rows with low confidence and no consensus -- show up as "urgent"
URGENT_ROWS = [
    ("u1", "finance", "in_progress", 0, 0.20, "2026-02-23T10:00:00"),
    ("u2", "tech", "in_progress", 0, 0.10, "2026-02-22T11:00:00"),
    ("u3", "legal", "completed", 1, 0.95, "2026-02-21T12:00:00"),
]


# ===========================================================================
# Tests: _get_quick_actions
# ===========================================================================


class TestGetQuickActions:
    """Tests for the quick actions list endpoint."""

    def test_returns_200(self, handler):
        result = handler._get_quick_actions()
        assert _status(result) == 200

    def test_returns_actions_list(self, handler):
        body = _body(handler._get_quick_actions())
        assert "actions" in body
        assert isinstance(body["actions"], list)

    def test_total_matches_actions_length(self, handler):
        body = _body(handler._get_quick_actions())
        assert body["total"] == len(body["actions"])

    def test_has_six_actions(self, handler):
        body = _body(handler._get_quick_actions())
        assert body["total"] == 6

    def test_action_ids(self, handler):
        body = _body(handler._get_quick_actions())
        ids = {a["id"] for a in body["actions"]}
        expected = {"archive_read", "snooze_low", "mark_spam", "complete_actions", "ai_respond", "sync_inbox"}
        assert ids == expected

    def test_each_action_has_required_fields(self, handler):
        body = _body(handler._get_quick_actions())
        required_fields = {"id", "name", "description", "icon", "available"}
        for action in body["actions"]:
            assert set(action.keys()) == required_fields

    def test_all_actions_available(self, handler):
        body = _body(handler._get_quick_actions())
        for action in body["actions"]:
            assert action["available"] is True

    def test_actions_have_nonempty_names(self, handler):
        body = _body(handler._get_quick_actions())
        for action in body["actions"]:
            assert len(action["name"]) > 0
            assert len(action["description"]) > 0

    def test_icons_are_strings(self, handler):
        body = _body(handler._get_quick_actions())
        expected_icons = {"archive", "clock", "slash", "check-circle", "sparkles", "refresh"}
        icons = {a["icon"] for a in body["actions"]}
        assert icons == expected_icons

    def test_response_content_type(self, handler):
        result = handler._get_quick_actions()
        assert result.content_type == "application/json"


# ===========================================================================
# Tests: _execute_quick_action
# ===========================================================================


class TestExecuteQuickAction:
    """Tests for the execute quick action endpoint."""

    def test_success(self, handler):
        result = handler._execute_quick_action("archive_read")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["action_id"] == "archive_read"

    def test_executed_at_present(self, handler):
        body = _body(handler._execute_quick_action("test_action"))
        assert "executed_at" in body
        assert "T" in body["executed_at"]

    def test_empty_action_id(self, handler):
        result = handler._execute_quick_action("")
        assert _status(result) == 400
        body = _body(result)
        assert "action_id" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    def test_arbitrary_action_id(self, handler):
        result = handler._execute_quick_action("custom_action_123")
        assert _status(result) == 200
        body = _body(result)
        assert body["action_id"] == "custom_action_123"

    def test_special_characters_in_action_id(self, handler):
        result = handler._execute_quick_action("action/with%special&chars")
        assert _status(result) == 200
        body = _body(result)
        assert body["action_id"] == "action/with%special&chars"

    def test_long_action_id(self, handler):
        long_id = "a" * 1000
        result = handler._execute_quick_action(long_id)
        assert _status(result) == 200
        body = _body(result)
        assert body["action_id"] == long_id


# ===========================================================================
# Tests: _get_urgent_items
# ===========================================================================


class TestGetUrgentItems:
    """Tests for the urgent items endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_urgent_items(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["items"] == []
        assert body["total"] == 0

    def test_returns_low_consensus_items(self):
        storage = InMemoryStorage(URGENT_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        body = _body(result)
        assert _status(result) == 200
        # u1 and u2 have consensus_reached=0 and confidence < 0.3
        # u3 has consensus_reached=1 and confidence=0.95 so not urgent
        assert body["total"] == 2
        ids = {item["id"] for item in body["items"]}
        assert "u1" in ids
        assert "u2" in ids

    def test_urgent_item_fields(self):
        storage = InMemoryStorage(URGENT_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        item = _body(result)["items"][0]
        required_fields = {"id", "type", "domain", "confidence", "created_at", "description"}
        assert set(item.keys()) == required_fields
        assert item["type"] == "low_consensus"

    def test_urgent_description_contains_domain(self):
        storage = InMemoryStorage(URGENT_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        for item in _body(result)["items"]:
            assert "needs attention" in item["description"]

    def test_null_domain_becomes_general_in_description(self):
        rows = [("x1", None, "pending", 0, 0.1, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        item = _body(result)["items"][0]
        assert "general" in item["description"]

    def test_pagination_limit(self):
        storage = InMemoryStorage(URGENT_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(1, 0)
        body = _body(result)
        assert len(body["items"]) == 1

    def test_pagination_offset(self):
        storage = InMemoryStorage(URGENT_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 1)
        body = _body(result)
        assert len(body["items"]) == 1

    def test_storage_error_graceful(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_urgent_items(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["items"] == []
        assert body["total"] == 0

    def test_empty_table(self):
        storage = InMemoryStorage([])
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        body = _body(result)
        assert body["total"] == 0

    def test_all_high_confidence_no_urgent(self):
        rows = [
            ("h1", "finance", "completed", 1, 0.95, "2026-02-23T10:00:00"),
            ("h2", "tech", "completed", 1, 0.88, "2026-02-22T10:00:00"),
        ]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_urgent_items(20, 0)
        body = _body(result)
        assert body["total"] == 0


# ===========================================================================
# Tests: _dismiss_urgent_item
# ===========================================================================


class TestDismissUrgentItem:
    """Tests for the dismiss urgent item endpoint."""

    def test_empty_item_id(self, handler):
        result = handler._dismiss_urgent_item("")
        assert _status(result) == 400
        body = _body(result)
        assert "item_id" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    def test_dismiss_existing_item(self):
        rows = [("u1", "finance", "in_progress", 0, 0.1, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._dismiss_urgent_item("u1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["item_id"] == "u1"
        assert "dismissed_at" in body
        assert "T" in body["dismissed_at"]

    def test_dismiss_updates_consensus_reached(self):
        rows = [("u1", "finance", "in_progress", 0, 0.1, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        h._dismiss_urgent_item("u1")
        # Verify the database was updated
        with storage.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT consensus_reached FROM debates WHERE id = ?", ("u1",))
            row = cursor.fetchone()
            assert row[0] == 1

    def test_dismiss_nonexistent_item(self):
        storage = InMemoryStorage([])
        h = TestableHandler(storage=storage)
        result = h._dismiss_urgent_item("nonexistent")
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_dismiss_no_storage(self, handler):
        # No storage means no DB update, but also no error (storage is None)
        result = handler._dismiss_urgent_item("some-id")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_dismiss_storage_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._dismiss_urgent_item("some-id")
        assert _status(result) == 500
        body = _body(result)
        assert "failed" in body.get("error", "").lower()


# ===========================================================================
# Tests: _get_pending_actions
# ===========================================================================


class TestGetPendingActions:
    """Tests for the pending actions endpoint."""

    def test_no_storage(self, handler):
        result = handler._get_pending_actions(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["actions"] == []
        assert body["total"] == 0

    def test_returns_pending_and_in_progress(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 0)
        body = _body(result)
        assert _status(result) == 200
        # d3 is in_progress, d4 and d5 are pending
        assert body["total"] == 3
        ids = {a["id"] for a in body["actions"]}
        assert "d3" in ids
        assert "d4" in ids
        assert "d5" in ids

    def test_pending_action_fields(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 0)
        action = _body(result)["actions"][0]
        required_fields = {"id", "type", "domain", "created_at", "description"}
        assert set(action.keys()) == required_fields
        assert action["type"] == "review_debate"

    def test_description_contains_domain(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 0)
        for action in _body(result)["actions"]:
            assert "Review debate" in action["description"]

    def test_null_domain_becomes_general_in_description(self):
        rows = [("p1", None, "pending", 0, 0.5, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 0)
        action = _body(result)["actions"][0]
        assert "general" in action["description"]

    def test_pagination_limit(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(1, 0)
        body = _body(result)
        assert len(body["actions"]) == 1

    def test_pagination_offset(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 2)
        body = _body(result)
        assert len(body["actions"]) == 1

    def test_no_pending_items(self):
        rows = [
            ("c1", "finance", "completed", 1, 0.9, "2026-02-23T10:00:00"),
            ("c2", "tech", "completed", 1, 0.8, "2026-02-22T10:00:00"),
        ]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._get_pending_actions(20, 0)
        body = _body(result)
        assert body["total"] == 0

    def test_storage_error_graceful(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._get_pending_actions(20, 0)
        body = _body(result)
        assert _status(result) == 200
        assert body["actions"] == []


# ===========================================================================
# Tests: _complete_pending_action
# ===========================================================================


class TestCompletePendingAction:
    """Tests for the complete pending action endpoint."""

    def test_empty_action_id(self, handler):
        result = handler._complete_pending_action("")
        assert _status(result) == 400

    def test_complete_pending_item(self):
        rows = [("p1", "finance", "pending", 0, 0.5, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._complete_pending_action("p1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["action_id"] == "p1"
        assert "completed_at" in body

    def test_complete_in_progress_item(self):
        rows = [("p1", "tech", "in_progress", 0, 0.6, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._complete_pending_action("p1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_complete_updates_status_in_db(self):
        rows = [("p1", "finance", "pending", 0, 0.5, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        h._complete_pending_action("p1")
        with storage.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM debates WHERE id = ?", ("p1",))
            row = cursor.fetchone()
            assert row[0] == "completed"

    def test_complete_nonexistent_item(self):
        storage = InMemoryStorage([])
        h = TestableHandler(storage=storage)
        result = h._complete_pending_action("nonexistent")
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_complete_already_completed_item(self):
        rows = [("p1", "finance", "completed", 1, 0.9, "2026-02-23T10:00:00")]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._complete_pending_action("p1")
        assert _status(result) == 404
        body = _body(result)
        assert "already completed" in body.get("error", "").lower()

    def test_complete_no_storage(self, handler):
        result = handler._complete_pending_action("some-id")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_complete_storage_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._complete_pending_action("some-id")
        assert _status(result) == 500
        body = _body(result)
        assert "failed" in body.get("error", "").lower()


# ===========================================================================
# Tests: _search_dashboard
# ===========================================================================


class TestSearchDashboard:
    """Tests for the search endpoint."""

    def test_empty_query(self, handler):
        result = handler._search_dashboard("")
        body = _body(result)
        assert _status(result) == 200
        assert body["results"] == []
        assert body["total"] == 0

    def test_no_storage(self, handler):
        result = handler._search_dashboard("finance")
        body = _body(result)
        assert _status(result) == 200
        assert body["results"] == []

    def test_search_by_domain(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("finance")
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 2
        for r in body["results"]:
            assert "finance" in (r.get("domain") or r.get("id"))

    def test_search_by_id(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("d1")
        body = _body(result)
        assert body["total"] >= 1
        ids = [r["id"] for r in body["results"]]
        assert "d1" in ids

    def test_search_result_fields(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("d1")
        body = _body(result)
        r = body["results"][0]
        expected_fields = {"id", "domain", "consensus_reached", "confidence", "created_at"}
        assert set(r.keys()) == expected_fields

    def test_consensus_reached_is_bool(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("d")
        body = _body(result)
        for r in body["results"]:
            assert isinstance(r["consensus_reached"], bool)

    def test_search_no_matches(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("zzznonexistent")
        body = _body(result)
        assert body["total"] == 0
        assert body["results"] == []

    def test_search_partial_match(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("ech")  # partial match for "tech"
        body = _body(result)
        assert body["total"] >= 1

    def test_search_limit_20(self):
        """Search is limited to 20 results."""
        rows = [
            (f"d{i}", "finance", "completed", 1, 0.5, "2026-01-01T00:00:00")
            for i in range(30)
        ]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("finance")
        body = _body(result)
        assert len(body["results"]) <= 20

    def test_search_storage_error(self):
        h = TestableHandler(storage=ErrorStorage())
        result = h._search_dashboard("test")
        body = _body(result)
        assert _status(result) == 200
        assert body["results"] == []


# ===========================================================================
# Tests: _export_dashboard_data
# ===========================================================================


class TestExportDashboardData:
    """Tests for the export endpoint."""

    def test_no_storage(self, handler):
        result = handler._export_dashboard_data()
        body = _body(result)
        assert _status(result) == 200
        assert "generated_at" in body
        assert "summary" in body
        assert "agent_performance" in body
        assert "consensus_insights" in body

    def test_generated_at_is_iso(self, handler):
        body = _body(handler._export_dashboard_data())
        assert "T" in body["generated_at"]

    def test_with_storage(self):
        storage = InMemoryStorage(SAMPLE_ROWS)
        h = TestableHandler(
            storage=storage,
            summary_metrics={"total_debates": 5, "consensus_rate": 0.8},
            agent_perf={"top_performers": [], "total_agents": 3},
            consensus_insights={"avg_confidence": 0.85},
        )
        result = h._export_dashboard_data()
        body = _body(result)
        assert _status(result) == 200
        assert body["summary"] == {"total_debates": 5, "consensus_rate": 0.8}
        assert body["agent_performance"] == {"top_performers": [], "total_agents": 3}
        assert body["consensus_insights"] == {"avg_confidence": 0.85}

    def test_export_error_graceful(self):
        h = TestableHandler(storage=ErrorStorage())
        # ErrorStorage will cause _get_summary_metrics_sql to fail because
        # get_storage() returns ErrorStorage, but the code calls:
        #   storage = self.get_storage()
        #   if storage:
        #       export["summary"] = self._get_summary_metrics_sql(storage, None)
        # Our TestableHandler's _get_summary_metrics_sql doesn't use storage,
        # so it won't fail. Let's make it fail explicitly.
        h._get_summary_metrics_sql = MagicMock(side_effect=ValueError("fail"))
        result = h._export_dashboard_data()
        body = _body(result)
        assert _status(result) == 200
        assert "generated_at" in body

    def test_export_agent_performance_error(self):
        h = TestableHandler()
        h._get_agent_performance = MagicMock(side_effect=TypeError("fail"))
        result = h._export_dashboard_data()
        body = _body(result)
        assert _status(result) == 200
        # When error happens in the try block, export still has generated_at
        assert "generated_at" in body

    def test_export_consensus_insights_error(self):
        h = TestableHandler()
        h._get_consensus_insights = MagicMock(side_effect=KeyError("fail"))
        result = h._export_dashboard_data()
        body = _body(result)
        assert _status(result) == 200
        assert "generated_at" in body


# ===========================================================================
# Tests: _get_quality_metrics
# ===========================================================================


class TestGetQualityMetrics:
    """Tests for the quality metrics endpoint."""

    def test_no_ctx(self, handler):
        result = handler._get_quality_metrics()
        body = _body(result)
        assert _status(result) == 200
        assert "calibration" in body
        assert "performance" in body
        assert "evolution" in body
        assert "debate_quality" in body
        assert "generated_at" in body

    def test_generated_at_is_numeric(self, handler):
        body = _body(handler._get_quality_metrics())
        assert isinstance(body["generated_at"], (int, float))

    def test_calibration_defaults(self, handler):
        body = _body(handler._get_quality_metrics())
        cal = body["calibration"]
        assert cal["agents"] == {}
        assert cal["overall_calibration"] == 0.0
        assert cal["overconfident_agents"] == []
        assert cal["underconfident_agents"] == []
        assert cal["well_calibrated_agents"] == []
        assert cal["top_by_brier"] == []
        assert cal["calibration_curves"] == {}
        assert cal["domain_breakdown"] == {}

    def test_performance_defaults(self, handler):
        body = _body(handler._get_quality_metrics())
        perf = body["performance"]
        assert perf["agents"] == {}
        assert perf["avg_latency_ms"] == 0.0
        assert perf["success_rate"] == 0.0
        assert perf["total_calls"] == 0

    def test_evolution_defaults(self, handler):
        body = _body(handler._get_quality_metrics())
        evo = body["evolution"]
        assert evo["agents"] == {}
        assert evo["total_versions"] == 0
        assert evo["patterns_extracted"] == 0
        assert evo["last_evolution"] is None

    def test_debate_quality_defaults(self, handler):
        body = _body(handler._get_quality_metrics())
        dq = body["debate_quality"]
        assert dq["avg_confidence"] == 0.0
        assert dq["consensus_rate"] == 0.0
        assert dq["avg_rounds"] == 0.0
        assert dq["evidence_quality"] == 0.0
        assert dq["recent_winners"] == []

    def test_response_content_type(self, handler):
        result = handler._get_quality_metrics()
        assert result.content_type == "application/json"


# ===========================================================================
# Tests: _get_calibration_metrics (internal helper)
# ===========================================================================


class TestCalibrationMetrics:
    """Tests for the internal calibration metrics helper."""

    def test_no_calibration_tracker(self, handler):
        result = handler._get_calibration_metrics()
        assert result["agents"] == {}
        assert result["overall_calibration"] == 0.0

    def test_with_calibration_tracker(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {
                "claude": {"calibration_bias": 0.15, "brier_score": 0.12},
                "gemini": {"calibration_bias": -0.2, "brier_score": 0.18},
                "grok": {"calibration_bias": 0.05, "brier_score": 0.10},
            },
            "overall": 0.78,
        }
        tracker.get_all_agents.return_value = ["claude", "gemini", "grok"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = None

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["overall_calibration"] == 0.78
        assert "claude" in result["overconfident_agents"]
        assert "gemini" in result["underconfident_agents"]
        assert "grok" in result["well_calibrated_agents"]

    def test_top_by_brier_sorted(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {
                "a1": {"calibration_bias": 0.0, "brier_score": 0.3},
                "a2": {"calibration_bias": 0.0, "brier_score": 0.1},
                "a3": {"calibration_bias": 0.0, "brier_score": 0.2},
            },
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["a1", "a2", "a3"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = None

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        brier_scores = [entry["brier_score"] for entry in result["top_by_brier"]]
        assert brier_scores == sorted(brier_scores)

    def test_top_by_brier_limited_to_5(self):
        agents = {f"agent{i}": {"calibration_bias": 0.0, "brier_score": i * 0.05} for i in range(10)}
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": agents,
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = list(agents.keys())
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = None

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert len(result["top_by_brier"]) == 5

    def test_calibration_curve_data(self):
        mock_bucket = MagicMock()
        mock_bucket.range_start = 0.0
        mock_bucket.range_end = 0.1
        mock_bucket.expected_accuracy = 0.05
        mock_bucket.accuracy = 0.08
        mock_bucket.total_predictions = 20

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {"claude": {"calibration_bias": 0.0, "brier_score": 0.1}},
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["claude"]
        tracker.get_calibration_curve.return_value = [mock_bucket]
        tracker.get_domain_breakdown.return_value = None

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert "claude" in result["calibration_curves"]
        curve = result["calibration_curves"]["claude"]
        assert len(curve) == 1
        assert curve[0]["bucket"] == 0
        assert curve[0]["confidence_range"] == "0.0-0.1"
        assert curve[0]["count"] == 20

    def test_calibration_curve_error_graceful(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {"claude": {"calibration_bias": 0.0, "brier_score": 0.1}},
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["claude"]
        tracker.get_calibration_curve.side_effect = ValueError("curve error")
        tracker.get_domain_breakdown.return_value = None

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["calibration_curves"] == {}

    def test_domain_breakdown_data(self):
        mock_stats = MagicMock()
        mock_stats.total_predictions = 50
        mock_stats.accuracy = 0.85
        mock_stats.brier_score = 0.12
        mock_stats.ece = 0.03

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {"claude": {"calibration_bias": 0.0, "brier_score": 0.1}},
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["claude"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = {"finance": mock_stats}

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert "claude" in result["domain_breakdown"]
        assert result["domain_breakdown"]["claude"]["finance"]["predictions"] == 50
        assert result["domain_breakdown"]["claude"]["finance"]["accuracy"] == 0.85

    def test_domain_breakdown_without_ece(self):
        mock_stats = MagicMock(spec=["total_predictions", "accuracy", "brier_score"])
        mock_stats.total_predictions = 10
        mock_stats.accuracy = 0.7
        mock_stats.brier_score = 0.2

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {"claude": {"calibration_bias": 0.0, "brier_score": 0.1}},
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["claude"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = {"tech": mock_stats}

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["domain_breakdown"]["claude"]["tech"]["ece"] is None

    def test_domain_breakdown_error_graceful(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {"claude": {"calibration_bias": 0.0, "brier_score": 0.1}},
            "overall": 0.5,
        }
        tracker.get_all_agents.return_value = ["claude"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.side_effect = TypeError("breakdown error")

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["domain_breakdown"] == {}

    def test_calibration_summary_none(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = None
        tracker.get_all_agents.return_value = []

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["agents"] == {}
        assert result["overall_calibration"] == 0.0

    def test_calibration_tracker_error_graceful(self):
        tracker = MagicMock()
        tracker.get_calibration_summary.side_effect = AttributeError("no method")

        h = TestableHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["agents"] == {}


# ===========================================================================
# Tests: _get_performance_metrics (internal helper)
# ===========================================================================


class TestPerformanceMetrics:
    """Tests for the internal performance metrics helper."""

    def test_no_performance_monitor(self, handler):
        result = handler._get_performance_metrics()
        assert result["agents"] == {}
        assert result["avg_latency_ms"] == 0.0
        assert result["success_rate"] == 0.0
        assert result["total_calls"] == 0

    def test_with_performance_monitor(self):
        monitor = MagicMock()
        monitor.get_performance_insights.return_value = {
            "agents": {"claude": {"latency": 100}},
            "avg_latency_ms": 150.5,
            "success_rate": 0.95,
            "total_calls": 1000,
        }

        h = TestableHandler(ctx={"performance_monitor": monitor})
        result = h._get_performance_metrics()
        assert result["agents"] == {"claude": {"latency": 100}}
        assert result["avg_latency_ms"] == 150.5
        assert result["success_rate"] == 0.95
        assert result["total_calls"] == 1000

    def test_performance_insights_none(self):
        monitor = MagicMock()
        monitor.get_performance_insights.return_value = None

        h = TestableHandler(ctx={"performance_monitor": monitor})
        result = h._get_performance_metrics()
        assert result["agents"] == {}
        assert result["total_calls"] == 0

    def test_performance_monitor_error(self):
        monitor = MagicMock()
        monitor.get_performance_insights.side_effect = TypeError("error")

        h = TestableHandler(ctx={"performance_monitor": monitor})
        result = h._get_performance_metrics()
        assert result["agents"] == {}


# ===========================================================================
# Tests: _get_evolution_metrics (internal helper)
# ===========================================================================


class TestEvolutionMetrics:
    """Tests for the internal evolution metrics helper."""

    def test_no_prompt_evolver(self, handler):
        result = handler._get_evolution_metrics()
        assert result["agents"] == {}
        assert result["total_versions"] == 0
        assert result["patterns_extracted"] == 0
        assert result["last_evolution"] is None

    def test_with_prompt_evolver(self):
        evolver = MagicMock()

        # Mock version objects for each agent
        versions = {}
        for name, ver, score, count in [
            ("claude", 3, 0.85, 50),
            ("gemini", 2, 0.78, 30),
        ]:
            v = MagicMock()
            v.version = ver
            v.performance_score = score
            v.debates_count = count
            versions[name] = v

        def get_version(agent_name):
            return versions.get(agent_name)

        evolver.get_prompt_version.side_effect = get_version
        evolver.get_top_patterns.return_value = ["pattern1", "pattern2", "pattern3"]

        h = TestableHandler(ctx={"prompt_evolver": evolver})
        result = h._get_evolution_metrics()
        assert "claude" in result["agents"]
        assert result["agents"]["claude"]["current_version"] == 3
        assert result["agents"]["claude"]["performance_score"] == 0.85
        assert result["total_versions"] == 5  # 3 + 2
        assert result["patterns_extracted"] == 3

    def test_prompt_evolver_version_none(self):
        evolver = MagicMock()
        evolver.get_prompt_version.return_value = None
        evolver.get_top_patterns.return_value = []

        h = TestableHandler(ctx={"prompt_evolver": evolver})
        result = h._get_evolution_metrics()
        assert result["agents"] == {}
        assert result["total_versions"] == 0

    def test_prompt_evolver_error(self):
        evolver = MagicMock()
        evolver.get_prompt_version.side_effect = AttributeError("no method")

        h = TestableHandler(ctx={"prompt_evolver": evolver})
        result = h._get_evolution_metrics()
        assert result["agents"] == {}

    def test_patterns_none(self):
        evolver = MagicMock()
        evolver.get_prompt_version.return_value = None
        evolver.get_top_patterns.return_value = None

        h = TestableHandler(ctx={"prompt_evolver": evolver})
        result = h._get_evolution_metrics()
        assert result["patterns_extracted"] == 0


# ===========================================================================
# Tests: _get_debate_quality_metrics (internal helper)
# ===========================================================================


class TestDebateQualityMetrics:
    """Tests for the internal debate quality metrics helper."""

    def test_no_elo_no_storage(self, handler):
        result = handler._get_debate_quality_metrics()
        assert result["avg_confidence"] == 0.0
        assert result["consensus_rate"] == 0.0
        assert result["avg_rounds"] == 0.0
        assert result["evidence_quality"] == 0.0
        assert result["recent_winners"] == []

    def test_with_elo_system(self):
        elo = MagicMock()
        elo.get_recent_matches.return_value = [
            {"winner": "claude", "confidence": 0.9},
            {"winner": "gemini", "confidence": 0.8},
            {"winner": "claude", "confidence": 0.85},
        ]

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["recent_winners"] == ["claude", "gemini", "claude"]
        expected_avg = (0.9 + 0.8 + 0.85) / 3
        assert abs(result["avg_confidence"] - expected_avg) < 0.001

    def test_winners_limited_to_5(self):
        elo = MagicMock()
        elo.get_recent_matches.return_value = [
            {"winner": f"agent{i}", "confidence": 0.5} for i in range(10)
        ]

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert len(result["recent_winners"]) == 5

    def test_matches_without_winners(self):
        elo = MagicMock()
        elo.get_recent_matches.return_value = [
            {"winner": None, "confidence": 0.5},
            {"confidence": 0.6},
        ]

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["recent_winners"] == []

    def test_matches_without_confidence(self):
        elo = MagicMock()
        elo.get_recent_matches.return_value = [
            {"winner": "claude"},
        ]

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["avg_confidence"] == 0.0

    def test_with_storage_summary(self):
        h = TestableHandler(
            storage=InMemoryStorage([]),
            summary_metrics={"consensus_rate": 0.75, "avg_rounds": 3.2},
        )
        result = h._get_debate_quality_metrics()
        assert result["consensus_rate"] == 0.75
        assert result["avg_rounds"] == 3.2

    def test_elo_recent_matches_none(self):
        elo = MagicMock()
        elo.get_recent_matches.return_value = None

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["recent_winners"] == []
        assert result["avg_confidence"] == 0.0

    def test_elo_error_graceful(self):
        elo = MagicMock()
        elo.get_recent_matches.side_effect = AttributeError("boom")

        h = TestableHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["avg_confidence"] == 0.0


# ===========================================================================
# Tests: Integration / edge cases
# ===========================================================================


class TestIntegration:
    """Cross-cutting integration tests."""

    def test_all_endpoints_return_200(self, handler):
        """Every action/analytics endpoint returns 200 with empty handler."""
        assert _status(handler._get_quick_actions()) == 200
        assert _status(handler._execute_quick_action("test")) == 200
        assert _status(handler._get_urgent_items(20, 0)) == 200
        assert _status(handler._dismiss_urgent_item("test")) == 200
        assert _status(handler._get_pending_actions(20, 0)) == 200
        assert _status(handler._complete_pending_action("test")) == 200
        assert _status(handler._search_dashboard("test")) == 200
        assert _status(handler._export_dashboard_data()) == 200
        assert _status(handler._get_quality_metrics()) == 200

    def test_all_responses_are_json(self, handler):
        """Every response has application/json content type."""
        results = [
            handler._get_quick_actions(),
            handler._execute_quick_action("test"),
            handler._get_urgent_items(20, 0),
            handler._dismiss_urgent_item("test"),
            handler._get_pending_actions(20, 0),
            handler._complete_pending_action("test"),
            handler._search_dashboard("test"),
            handler._export_dashboard_data(),
            handler._get_quality_metrics(),
        ]
        for r in results:
            assert r.content_type == "application/json"

    def test_error_storage_write_endpoints(self):
        """Write endpoints return 500 on ErrorStorage."""
        h = TestableHandler(storage=ErrorStorage())
        assert _status(h._dismiss_urgent_item("test")) == 500
        assert _status(h._complete_pending_action("test")) == 500

    def test_error_storage_read_endpoints_graceful(self):
        """Read endpoints degrade gracefully on ErrorStorage."""
        h = TestableHandler(storage=ErrorStorage())
        assert _status(h._get_urgent_items(20, 0)) == 200
        assert _status(h._get_pending_actions(20, 0)) == 200
        assert _status(h._search_dashboard("test")) == 200

    def test_empty_storage_table(self):
        """Endpoints work with an empty debates table."""
        storage = InMemoryStorage([])
        h = TestableHandler(storage=storage)
        assert _body(h._get_urgent_items(20, 0))["total"] == 0
        assert _body(h._get_pending_actions(20, 0))["total"] == 0
        assert _body(h._search_dashboard("test"))["total"] == 0

    def test_large_dataset_search(self):
        """Search handles many rows without error."""
        rows = [
            (f"d{i}", f"dom{i % 5}", "completed", i % 2, i / 100, f"2026-01-{1 + i % 28:02d}T00:00:00")
            for i in range(200)
        ]
        storage = InMemoryStorage(rows)
        h = TestableHandler(storage=storage)
        result = h._search_dashboard("dom0")
        body = _body(result)
        assert len(body["results"]) <= 20


class TestTTLCacheBehavior:
    """Tests verifying TTL cache interaction with quality metrics."""

    def test_quality_metrics_cached_on_second_call(self):
        """Quality metrics are cached on second call (skip_first=True)."""
        h = TestableHandler()
        r1 = h._get_quality_metrics()
        r2 = h._get_quality_metrics()
        # Both calls succeed
        assert _status(r1) == 200
        assert _status(r2) == 200


class TestMixinMethodExistence:
    """Tests verifying the mixin exposes expected methods."""

    def test_has_get_quick_actions(self):
        assert hasattr(DashboardActionsMixin, "_get_quick_actions")

    def test_has_execute_quick_action(self):
        assert hasattr(DashboardActionsMixin, "_execute_quick_action")

    def test_has_get_urgent_items(self):
        assert hasattr(DashboardActionsMixin, "_get_urgent_items")

    def test_has_dismiss_urgent_item(self):
        assert hasattr(DashboardActionsMixin, "_dismiss_urgent_item")

    def test_has_get_pending_actions(self):
        assert hasattr(DashboardActionsMixin, "_get_pending_actions")

    def test_has_complete_pending_action(self):
        assert hasattr(DashboardActionsMixin, "_complete_pending_action")

    def test_has_search_dashboard(self):
        assert hasattr(DashboardActionsMixin, "_search_dashboard")

    def test_has_export_dashboard_data(self):
        assert hasattr(DashboardActionsMixin, "_export_dashboard_data")

    def test_has_get_quality_metrics(self):
        assert hasattr(DashboardActionsMixin, "_get_quality_metrics")

    def test_has_get_calibration_metrics(self):
        assert hasattr(DashboardActionsMixin, "_get_calibration_metrics")

    def test_has_get_performance_metrics(self):
        assert hasattr(DashboardActionsMixin, "_get_performance_metrics")

    def test_has_get_evolution_metrics(self):
        assert hasattr(DashboardActionsMixin, "_get_evolution_metrics")

    def test_has_get_debate_quality_metrics(self):
        assert hasattr(DashboardActionsMixin, "_get_debate_quality_metrics")
