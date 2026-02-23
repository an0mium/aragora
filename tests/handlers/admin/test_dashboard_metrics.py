"""Tests for dashboard_metrics.py utility functions.

Comprehensive coverage of all 6 public functions:
- get_summary_metrics_sql       (SQL-based summary)
- get_recent_activity_sql       (SQL-based recent activity)
- get_summary_metrics_legacy    (list-based summary)
- get_recent_activity_legacy    (list-based recent activity)
- process_debates_single_pass   (consolidated single-pass)
- get_debate_patterns           (disagreement / early-stopping stats)

Also covers:
- RBAC permission enforcement (local require_permission decorator)
- Rate-limit decorator pass-through
- SQL error handling (OSError, TypeError, etc.)
- Edge cases: empty data, None values, malformed timestamps
- Confidence averaging, consensus rate calculation
- Domain counting and most-active-domain selection
- Disagreement type aggregation
- Early stopping vs full duration counting
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext

# Import the module so we can patch its local helper
import aragora.server.handlers.admin.dashboard_metrics as dm_module
from aragora.server.handlers.admin.dashboard_metrics import (
    get_debate_patterns,
    get_recent_activity_legacy,
    get_recent_activity_sql,
    get_summary_metrics_legacy,
    get_summary_metrics_sql,
    process_debates_single_pass,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _admin_ctx() -> AuthorizationContext:
    """Return an admin AuthorizationContext with wildcard permissions."""
    return AuthorizationContext(
        user_id="test-admin-001",
        user_email="admin@example.com",
        org_id="test-org-001",
        roles={"admin", "owner"},
        permissions={"*"},
    )


def _now_iso() -> str:
    """Return current time as ISO string."""
    return datetime.now().isoformat()


def _past_iso(hours: int = 48) -> str:
    """Return a timestamp from the past."""
    return (datetime.now() - timedelta(hours=hours)).isoformat()


def _make_debate(
    debate_id: str = "d1",
    consensus: bool = False,
    confidence: float | None = 0.8,
    domain: str = "general",
    created_at: str | None = None,
    disagreement_report: dict | None = None,
    early_stopped: bool = False,
) -> dict[str, Any]:
    """Build a debate dict for legacy/single-pass functions."""
    d: dict[str, Any] = {
        "id": debate_id,
        "domain": domain,
        "consensus_reached": consensus,
        "confidence": confidence,
        "created_at": created_at or _now_iso(),
    }
    if disagreement_report is not None:
        d["disagreement_report"] = disagreement_report
    if early_stopped:
        d["early_stopped"] = True
    return d


# ===========================================================================
# In-memory SQLite storage for SQL-level testing
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
            cur.executemany("INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?)", rows)
        self._conn.commit()

    @contextmanager
    def connection(self):
        yield self._conn


class ErrorStorage:
    """Storage whose connection raises on entry."""

    @contextmanager
    def connection(self):
        raise OSError("disk failure")


class CursorErrorStorage:
    """Storage whose cursor.execute raises."""

    @contextmanager
    def connection(self):
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.execute.side_effect = TypeError("bad query")
        yield mock_conn


# ===========================================================================
# Fixtures
# ===========================================================================


_ADMIN_CTX = _admin_ctx()


@pytest.fixture(autouse=True)
def _bypass_local_rbac(request, monkeypatch):
    """Patch the module-local _get_context_from_args_strict to inject admin context.

    The dashboard_metrics module defines its own local require_permission
    decorator that uses _get_context_from_args_strict (NOT the one in
    aragora.rbac.decorators). The conftest auto-auth patches do not reach it.
    We patch it here so the functions can be called without threading a
    ``context`` kwarg that would leak into the underlying function signature.

    Opted out by the @pytest.mark.no_auto_auth marker so that permission
    enforcement tests exercise the real decorator.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    monkeypatch.setattr(
        dm_module,
        "_get_context_from_args_strict",
        lambda args, kwargs, context_param: _ADMIN_CTX,
    )
    yield


@pytest.fixture()
def empty_storage():
    """Storage with no debate rows."""
    return InMemoryStorage()


@pytest.fixture()
def populated_storage():
    """Storage with representative debate rows."""
    now = _now_iso()
    old = _past_iso(72)
    rows = [
        ("d1", "tech", "completed", 1, 0.9, now),
        ("d2", "tech", "completed", 1, 0.85, now),
        ("d3", "finance", "completed", 0, 0.6, now),
        ("d4", "tech", "completed", 0, 0.5, old),
        ("d5", "health", "completed", 1, 0.95, old),
    ]
    return InMemoryStorage(rows)


# ===========================================================================
# get_summary_metrics_sql
# ===========================================================================


class TestGetSummaryMetricsSQL:
    """Tests for get_summary_metrics_sql."""

    def test_empty_database(self, empty_storage):
        result = get_summary_metrics_sql(empty_storage, None)
        assert result["total_debates"] == 0
        assert result["consensus_reached"] == 0
        assert result["consensus_rate"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_populated_database(self, populated_storage):
        result = get_summary_metrics_sql(populated_storage, None)
        assert result["total_debates"] == 5
        assert result["consensus_reached"] == 3
        assert result["consensus_rate"] == 0.6
        assert isinstance(result["avg_confidence"], float)
        assert result["avg_confidence"] > 0

    def test_consensus_rate_calculation(self):
        rows = [
            ("d1", "g", "completed", 1, 0.8, _now_iso()),
            ("d2", "g", "completed", 1, 0.9, _now_iso()),
            ("d3", "g", "completed", 0, 0.7, _now_iso()),
            ("d4", "g", "completed", 1, 0.85, _now_iso()),
        ]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        assert result["total_debates"] == 4
        assert result["consensus_reached"] == 3
        assert result["consensus_rate"] == 0.75

    def test_avg_confidence(self):
        rows = [
            ("d1", "g", "completed", 1, 0.8, _now_iso()),
            ("d2", "g", "completed", 0, 0.6, _now_iso()),
        ]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        assert result["avg_confidence"] == 0.7

    def test_null_confidence(self):
        rows = [
            ("d1", "g", "completed", 1, None, _now_iso()),
        ]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        # avg_confidence stays 0.0 when all NULL
        assert result["avg_confidence"] == 0.0

    def test_domain_param_accepted(self, populated_storage):
        # domain param is reserved but unused; verify it doesn't break
        result = get_summary_metrics_sql(populated_storage, "tech")
        assert result["total_debates"] == 5  # domain filter not applied

    def test_storage_error_returns_defaults(self):
        result = get_summary_metrics_sql(ErrorStorage(), None)
        assert result["total_debates"] == 0
        assert result["consensus_rate"] == 0.0

    def test_cursor_error_returns_defaults(self):
        result = get_summary_metrics_sql(CursorErrorStorage(), None)
        assert result["total_debates"] == 0

    def test_all_consensus(self):
        rows = [("d1", "g", "ok", 1, 1.0, _now_iso())]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        assert result["consensus_rate"] == 1.0
        assert result["consensus_reached"] == 1

    def test_no_consensus(self):
        rows = [
            ("d1", "g", "ok", 0, 0.3, _now_iso()),
            ("d2", "g", "ok", 0, 0.4, _now_iso()),
        ]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        assert result["consensus_rate"] == 0.0
        assert result["consensus_reached"] == 0

    def test_single_row_metrics(self):
        rows = [("d1", "g", "ok", 1, 0.75, _now_iso())]
        storage = InMemoryStorage(rows)
        result = get_summary_metrics_sql(storage, None)
        assert result["total_debates"] == 1
        assert result["consensus_reached"] == 1
        assert result["consensus_rate"] == 1.0
        assert result["avg_confidence"] == 0.75


# ===========================================================================
# get_recent_activity_sql
# ===========================================================================


class TestGetRecentActivitySQL:
    """Tests for get_recent_activity_sql."""

    def test_empty_database(self, empty_storage):
        result = get_recent_activity_sql(empty_storage, 24)
        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0
        assert result["period_hours"] == 24

    def test_all_recent(self):
        now = _now_iso()
        rows = [
            ("d1", "g", "ok", 1, 0.8, now),
            ("d2", "g", "ok", 0, 0.6, now),
        ]
        storage = InMemoryStorage(rows)
        result = get_recent_activity_sql(storage, 24)
        assert result["debates_last_period"] == 2
        assert result["consensus_last_period"] == 1

    def test_mixed_recent_and_old(self):
        now = _now_iso()
        old = _past_iso(72)
        rows = [
            ("d1", "g", "ok", 1, 0.8, now),
            ("d2", "g", "ok", 1, 0.9, old),
        ]
        storage = InMemoryStorage(rows)
        result = get_recent_activity_sql(storage, 24)
        assert result["debates_last_period"] == 1
        assert result["consensus_last_period"] == 1

    def test_custom_hours(self):
        # 100 hours window should include 72-hour-old entries
        now = _now_iso()
        old = _past_iso(72)
        rows = [
            ("d1", "g", "ok", 1, 0.8, now),
            ("d2", "g", "ok", 0, 0.6, old),
        ]
        storage = InMemoryStorage(rows)
        result = get_recent_activity_sql(storage, 100)
        assert result["debates_last_period"] == 2
        assert result["period_hours"] == 100

    def test_storage_error_returns_defaults(self):
        result = get_recent_activity_sql(ErrorStorage(), 24)
        assert result["debates_last_period"] == 0
        assert result["period_hours"] == 24

    def test_zero_hours_window(self):
        rows = [("d1", "g", "ok", 1, 0.8, _now_iso())]
        storage = InMemoryStorage(rows)
        # 0-hour window means cutoff == now, so nothing qualifies
        result = get_recent_activity_sql(storage, 0)
        assert result["period_hours"] == 0

    def test_only_old_entries(self):
        old = _past_iso(72)
        rows = [
            ("d1", "g", "ok", 1, 0.8, old),
            ("d2", "g", "ok", 1, 0.9, old),
        ]
        storage = InMemoryStorage(rows)
        result = get_recent_activity_sql(storage, 24)
        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0


# ===========================================================================
# get_summary_metrics_legacy
# ===========================================================================


class TestGetSummaryMetricsLegacy:
    """Tests for get_summary_metrics_legacy."""

    def test_empty_debates(self):
        result = get_summary_metrics_legacy(None, [])
        assert result["total_debates"] == 0
        assert result["consensus_rate"] == 0.0

    def test_basic_summary(self):
        debates = [
            _make_debate("d1", consensus=True, confidence=0.9),
            _make_debate("d2", consensus=False, confidence=0.6),
            _make_debate("d3", consensus=True, confidence=0.8),
        ]
        result = get_summary_metrics_legacy(None, debates)
        assert result["total_debates"] == 3
        assert result["consensus_reached"] == 2
        assert result["consensus_rate"] == round(2 / 3, 3)

    def test_avg_confidence_excludes_falsy(self):
        debates = [
            _make_debate("d1", confidence=0.8),
            _make_debate("d2", confidence=None),  # no confidence
            _make_debate("d3", confidence=0.6),
        ]
        result = get_summary_metrics_legacy(None, debates)
        # None confidence -> d.get("confidence") is None -> falsy, skipped
        # 0.8 and 0.6 only
        assert result["avg_confidence"] == 0.7

    def test_all_consensus(self):
        debates = [
            _make_debate("d1", consensus=True, confidence=1.0),
            _make_debate("d2", consensus=True, confidence=0.95),
        ]
        result = get_summary_metrics_legacy(None, debates)
        assert result["consensus_rate"] == 1.0

    def test_no_consensus(self):
        debates = [
            _make_debate("d1", consensus=False),
            _make_debate("d2", consensus=False),
        ]
        result = get_summary_metrics_legacy(None, debates)
        assert result["consensus_rate"] == 0.0

    def test_domain_param_accepted(self):
        debates = [_make_debate("d1", consensus=True)]
        result = get_summary_metrics_legacy("tech", debates)
        assert result["total_debates"] == 1

    def test_returns_extra_fields(self):
        result = get_summary_metrics_legacy(None, [])
        assert "avg_rounds" in result
        assert "total_tokens_used" in result

    def test_single_debate_consensus(self):
        debates = [_make_debate("d1", consensus=True, confidence=0.88)]
        result = get_summary_metrics_legacy(None, debates)
        assert result["total_debates"] == 1
        assert result["consensus_reached"] == 1
        assert result["consensus_rate"] == 1.0
        assert result["avg_confidence"] == 0.88

    def test_confidence_with_default(self):
        """Debates with confidence=None use 0.5 default but falsy values skipped."""
        debates = [
            {"id": "d1", "consensus_reached": True, "confidence": 0.5},
        ]
        result = get_summary_metrics_legacy(None, debates)
        assert result["avg_confidence"] == 0.5


# ===========================================================================
# get_recent_activity_legacy
# ===========================================================================


class TestGetRecentActivityLegacy:
    """Tests for get_recent_activity_legacy."""

    def test_empty_debates(self):
        result = get_recent_activity_legacy(None, 24, [])
        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0
        assert result["domains_active"] == []
        assert result["most_active_domain"] is None
        assert result["period_hours"] == 24

    def test_all_recent(self):
        now = _now_iso()
        debates = [
            _make_debate("d1", consensus=True, domain="tech", created_at=now),
            _make_debate("d2", consensus=False, domain="tech", created_at=now),
            _make_debate("d3", consensus=True, domain="finance", created_at=now),
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 3
        assert result["consensus_last_period"] == 2
        assert set(result["domains_active"]) == {"tech", "finance"}
        assert result["most_active_domain"] == "tech"

    def test_filters_old_debates(self):
        now = _now_iso()
        old = _past_iso(48)
        debates = [
            _make_debate("d1", consensus=True, created_at=now),
            _make_debate("d2", consensus=True, created_at=old),
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 1

    def test_most_active_domain(self):
        now = _now_iso()
        debates = [
            _make_debate("d1", domain="finance", created_at=now),
            _make_debate("d2", domain="finance", created_at=now),
            _make_debate("d3", domain="tech", created_at=now),
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["most_active_domain"] == "finance"

    def test_default_domain(self):
        now = _now_iso()
        # debate with no explicit domain -> "general"
        debates = [{"id": "d1", "consensus_reached": False, "created_at": now}]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 1
        assert "general" in result["domains_active"]

    def test_invalid_timestamp_skipped(self):
        debates = [
            _make_debate("d1", created_at="not-a-date"),
            _make_debate("d2", created_at=_now_iso()),
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 1

    def test_no_created_at_skipped(self):
        debates = [{"id": "d1", "consensus_reached": True}]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 0

    def test_timezone_aware_timestamp(self):
        # Timestamps with Z suffix
        now_z = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        debates = [_make_debate("d1", consensus=True, created_at=now_z)]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 1

    def test_domains_limited_to_10(self):
        now = _now_iso()
        debates = [
            _make_debate(f"d{i}", domain=f"domain_{i}", created_at=now)
            for i in range(15)
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        assert len(result["domains_active"]) <= 10

    def test_wide_hours_window(self):
        old = _past_iso(72)
        debates = [
            _make_debate("d1", consensus=True, domain="tech", created_at=old),
        ]
        result = get_recent_activity_legacy(None, 100, debates)
        assert result["debates_last_period"] == 1
        assert result["period_hours"] == 100

    def test_multiple_domains_tie(self):
        now = _now_iso()
        debates = [
            _make_debate("d1", domain="alpha", created_at=now),
            _make_debate("d2", domain="beta", created_at=now),
        ]
        result = get_recent_activity_legacy(None, 24, debates)
        # With equal counts, max picks first alphabetically (CPython dict order)
        assert result["most_active_domain"] in ("alpha", "beta")


# ===========================================================================
# process_debates_single_pass
# ===========================================================================


class TestProcessDebatesSinglePass:
    """Tests for process_debates_single_pass."""

    def test_empty_list(self):
        summary, activity, patterns = process_debates_single_pass([], None, 24)
        assert summary["total_debates"] == 0
        assert activity["debates_last_period"] == 0
        assert patterns["disagreement_stats"]["with_disagreements"] == 0
        assert patterns["early_stopping"]["early_stopped"] == 0

    def test_returns_three_dicts(self):
        result = process_debates_single_pass([], None, 24)
        assert isinstance(result, tuple)
        assert len(result) == 3
        summary, activity, patterns = result
        assert isinstance(summary, dict)
        assert isinstance(activity, dict)
        assert isinstance(patterns, dict)

    def test_summary_metrics(self):
        debates = [
            _make_debate("d1", consensus=True, confidence=0.9),
            _make_debate("d2", consensus=False, confidence=0.7),
            _make_debate("d3", consensus=True, confidence=0.8),
        ]
        summary, _, _ = process_debates_single_pass(debates, None, 24)
        assert summary["total_debates"] == 3
        assert summary["consensus_reached"] == 2
        assert summary["consensus_rate"] == round(2 / 3, 3)
        assert summary["avg_confidence"] == 0.8

    def test_activity_metrics(self):
        now = _now_iso()
        old = _past_iso(48)
        debates = [
            _make_debate("d1", consensus=True, domain="tech", created_at=now),
            _make_debate("d2", consensus=False, domain="finance", created_at=now),
            _make_debate("d3", consensus=True, domain="tech", created_at=old),
        ]
        _, activity, _ = process_debates_single_pass(debates, None, 24)
        assert activity["debates_last_period"] == 2
        assert activity["consensus_last_period"] == 1
        assert set(activity["domains_active"]) == {"tech", "finance"}
        assert activity["period_hours"] == 24

    def test_pattern_metrics_disagreement(self):
        debates = [
            _make_debate(
                "d1",
                disagreement_report={"types": ["factual", "methodological"]},
            ),
            _make_debate(
                "d2",
                disagreement_report={"types": ["factual"]},
            ),
            _make_debate("d3"),  # no disagreement
        ]
        _, _, patterns = process_debates_single_pass(debates, None, 24)
        ds = patterns["disagreement_stats"]
        assert ds["with_disagreements"] == 2
        assert ds["disagreement_types"]["factual"] == 2
        assert ds["disagreement_types"]["methodological"] == 1

    def test_pattern_metrics_early_stopping(self):
        debates = [
            _make_debate("d1", early_stopped=True),
            _make_debate("d2", early_stopped=True),
            _make_debate("d3", early_stopped=False),
        ]
        _, _, patterns = process_debates_single_pass(debates, None, 24)
        es = patterns["early_stopping"]
        assert es["early_stopped"] == 2
        assert es["full_duration"] == 1

    def test_invalid_timestamp_skipped(self):
        debates = [
            _make_debate("d1", created_at="garbage"),
            _make_debate("d2", created_at=_now_iso()),
        ]
        _, activity, _ = process_debates_single_pass(debates, None, 24)
        assert activity["debates_last_period"] == 1

    def test_none_confidence_excluded(self):
        debates = [
            _make_debate("d1", confidence=0.8),
            _make_debate("d2", confidence=None),
        ]
        summary, _, _ = process_debates_single_pass(debates, None, 24)
        assert summary["avg_confidence"] == 0.8

    def test_domain_param_accepted(self):
        debates = [_make_debate("d1")]
        summary, _, _ = process_debates_single_pass(debates, "tech", 24)
        assert summary["total_debates"] == 1

    def test_large_dataset(self):
        now = _now_iso()
        debates = [
            _make_debate(
                f"d{i}",
                consensus=(i % 3 == 0),
                confidence=0.5 + (i % 5) * 0.1,
                domain=f"domain_{i % 4}",
                created_at=now,
                early_stopped=(i % 7 == 0),
            )
            for i in range(100)
        ]
        summary, activity, patterns = process_debates_single_pass(debates, None, 24)
        assert summary["total_debates"] == 100
        assert summary["consensus_reached"] == 34  # 0,3,6,...,99
        assert activity["debates_last_period"] == 100
        assert len(activity["domains_active"]) == 4

    def test_single_debate_all_fields(self):
        now = _now_iso()
        debates = [
            _make_debate(
                "d1",
                consensus=True,
                confidence=0.95,
                domain="tech",
                created_at=now,
                disagreement_report={"types": ["ethical"]},
                early_stopped=True,
            ),
        ]
        summary, activity, patterns = process_debates_single_pass(debates, None, 24)
        assert summary["total_debates"] == 1
        assert summary["consensus_reached"] == 1
        assert summary["consensus_rate"] == 1.0
        assert summary["avg_confidence"] == 0.95
        assert activity["debates_last_period"] == 1
        assert activity["consensus_last_period"] == 1
        assert "tech" in activity["domains_active"]
        assert patterns["disagreement_stats"]["with_disagreements"] == 1
        assert patterns["early_stopping"]["early_stopped"] == 1
        assert patterns["early_stopping"]["full_duration"] == 0

    def test_activity_most_active_domain(self):
        now = _now_iso()
        debates = [
            _make_debate("d1", domain="finance", created_at=now),
            _make_debate("d2", domain="finance", created_at=now),
            _make_debate("d3", domain="tech", created_at=now),
        ]
        _, activity, _ = process_debates_single_pass(debates, None, 24)
        assert activity["most_active_domain"] == "finance"

    def test_no_recent_debates(self):
        old = _past_iso(72)
        debates = [
            _make_debate("d1", created_at=old),
            _make_debate("d2", created_at=old),
        ]
        _, activity, _ = process_debates_single_pass(debates, None, 24)
        assert activity["debates_last_period"] == 0
        assert activity["most_active_domain"] is None
        assert activity["domains_active"] == []


# ===========================================================================
# get_debate_patterns
# ===========================================================================


class TestGetDebatePatterns:
    """Tests for get_debate_patterns."""

    def test_empty_list(self):
        result = get_debate_patterns([])
        assert result["disagreement_stats"]["with_disagreements"] == 0
        assert result["disagreement_stats"]["disagreement_types"] == {}
        assert result["early_stopping"]["early_stopped"] == 0
        assert result["early_stopping"]["full_duration"] == 0

    def test_disagreement_counting(self):
        debates = [
            _make_debate("d1", disagreement_report={"types": ["factual"]}),
            _make_debate("d2", disagreement_report={"types": ["factual", "ethical"]}),
            _make_debate("d3"),
        ]
        result = get_debate_patterns(debates)
        ds = result["disagreement_stats"]
        assert ds["with_disagreements"] == 2
        assert ds["disagreement_types"]["factual"] == 2
        assert ds["disagreement_types"]["ethical"] == 1

    def test_early_stopping_counting(self):
        debates = [
            _make_debate("d1", early_stopped=True),
            _make_debate("d2"),
            _make_debate("d3", early_stopped=True),
            _make_debate("d4"),
        ]
        result = get_debate_patterns(debates)
        es = result["early_stopping"]
        assert es["early_stopped"] == 2
        assert es["full_duration"] == 2

    def test_no_disagreements(self):
        debates = [_make_debate("d1"), _make_debate("d2")]
        result = get_debate_patterns(debates)
        assert result["disagreement_stats"]["with_disagreements"] == 0

    def test_all_early_stopped(self):
        debates = [
            _make_debate("d1", early_stopped=True),
            _make_debate("d2", early_stopped=True),
        ]
        result = get_debate_patterns(debates)
        assert result["early_stopping"]["early_stopped"] == 2
        assert result["early_stopping"]["full_duration"] == 0

    def test_disagreement_report_without_types(self):
        # disagreement_report exists but has no "types" key
        debates = [_make_debate("d1", disagreement_report={"severity": "high"})]
        result = get_debate_patterns(debates)
        assert result["disagreement_stats"]["with_disagreements"] == 1
        assert result["disagreement_stats"]["disagreement_types"] == {}

    def test_disagreement_report_empty_types(self):
        debates = [_make_debate("d1", disagreement_report={"types": []})]
        result = get_debate_patterns(debates)
        assert result["disagreement_stats"]["with_disagreements"] == 1
        assert result["disagreement_stats"]["disagreement_types"] == {}

    def test_many_disagreement_types(self):
        debates = [
            _make_debate(
                "d1",
                disagreement_report={
                    "types": ["factual", "ethical", "methodological", "interpretive"]
                },
            ),
        ]
        result = get_debate_patterns(debates)
        dt = result["disagreement_stats"]["disagreement_types"]
        assert len(dt) == 4
        assert all(v == 1 for v in dt.values())

    def test_mixed_full_and_early_with_disagreements(self):
        debates = [
            _make_debate(
                "d1",
                early_stopped=True,
                disagreement_report={"types": ["factual"]},
            ),
            _make_debate("d2"),
            _make_debate(
                "d3",
                early_stopped=True,
                disagreement_report={"types": ["ethical"]},
            ),
        ]
        result = get_debate_patterns(debates)
        assert result["disagreement_stats"]["with_disagreements"] == 2
        assert result["early_stopping"]["early_stopped"] == 2
        assert result["early_stopping"]["full_duration"] == 1


# ===========================================================================
# RBAC Permission Enforcement
# ===========================================================================


class TestPermissionEnforcement:
    """Tests for the local require_permission decorator.

    These tests use no_auto_auth to disable BOTH the conftest patch and our
    local _bypass_local_rbac fixture, so the real local decorator runs.
    """

    @pytest.mark.no_auto_auth
    def test_no_context_raises(self):
        """Calling without context should raise PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_sql(MagicMock(), None)

    @pytest.mark.no_auto_auth
    def test_denied_permission_raises(self):
        """A context with insufficient permissions should be denied."""
        denied_ctx = AuthorizationContext(
            user_id="user-no-perms",
            roles=set(),
            permissions=set(),
        )
        # Pass context as positional arg via _auth_context on a carrier object
        carrier = MagicMock()
        carrier._auth_context = denied_ctx
        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_sql(carrier, None)

    @pytest.mark.no_auto_auth
    def test_no_context_for_legacy(self):
        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_legacy(None, [])

    @pytest.mark.no_auto_auth
    def test_no_context_for_patterns(self):
        with pytest.raises(PermissionDeniedError):
            get_debate_patterns([])

    @pytest.mark.no_auto_auth
    def test_no_context_for_single_pass(self):
        with pytest.raises(PermissionDeniedError):
            process_debates_single_pass([], None, 24)

    @pytest.mark.no_auto_auth
    def test_no_context_for_recent_sql(self):
        with pytest.raises(PermissionDeniedError):
            get_recent_activity_sql(MagicMock(), 24)

    @pytest.mark.no_auto_auth
    def test_no_context_for_recent_legacy(self):
        with pytest.raises(PermissionDeniedError):
            get_recent_activity_legacy(None, 24, [])


# ===========================================================================
# Error Handling / Edge Cases
# ===========================================================================


class TestErrorHandling:
    """Tests for error-handling paths in each function."""

    def test_sql_summary_attribute_error(self):
        """Storage that returns a non-connection object."""

        class BadStorage:
            @contextmanager
            def connection(self):
                yield "not-a-connection"

        result = get_summary_metrics_sql(BadStorage(), None)
        assert result["total_debates"] == 0

    def test_sql_recent_attribute_error(self):
        class BadStorage:
            @contextmanager
            def connection(self):
                yield "not-a-connection"

        result = get_recent_activity_sql(BadStorage(), 24)
        assert result["debates_last_period"] == 0

    def test_legacy_summary_with_corrupt_debate(self):
        """Debates with non-dict entries (e.g., None) should not crash."""
        debates = [None]  # type: ignore[list-item]
        result = get_summary_metrics_legacy(None, debates)
        # Should return defaults because of TypeError on None.get
        assert result["total_debates"] == 0

    def test_legacy_recent_with_corrupt_debate(self):
        debates = [None]  # type: ignore[list-item]
        result = get_recent_activity_legacy(None, 24, debates)
        assert result["debates_last_period"] == 0

    def test_single_pass_with_corrupt_debate(self):
        debates = [None]  # type: ignore[list-item]
        summary, activity, patterns = process_debates_single_pass(debates, None, 24)
        assert summary["total_debates"] == 0

    def test_patterns_with_corrupt_debate(self):
        debates = [None]  # type: ignore[list-item]
        result = get_debate_patterns(debates)
        assert result["disagreement_stats"]["with_disagreements"] == 0

    def test_zero_confidence_excluded_from_legacy_avg(self):
        """Confidence of 0 is falsy, so it's excluded from average (by design)."""
        debates = [
            _make_debate("d1", confidence=0),
            _make_debate("d2", confidence=0.8),
        ]
        result = get_summary_metrics_legacy(None, debates)
        # confidence 0 -> d.get("confidence", 0.5) returns 0 -> falsy -> skipped
        # Only d2 counts
        assert result["avg_confidence"] == 0.8

    def test_single_pass_zero_confidence_excluded(self):
        """Same zero-confidence behavior in single-pass."""
        debates = [
            _make_debate("d1", confidence=0),
            _make_debate("d2", confidence=0.6),
        ]
        summary, _, _ = process_debates_single_pass(debates, None, 24)
        # confidence 0 -> falsy -> excluded
        assert summary["avg_confidence"] == 0.6

    def test_cursor_error_recent_sql(self):
        result = get_recent_activity_sql(CursorErrorStorage(), 24)
        assert result["debates_last_period"] == 0


# ===========================================================================
# Consistency Between Legacy and Single-Pass
# ===========================================================================


class TestConsistency:
    """Verify that single-pass produces the same results as legacy functions."""

    def test_summary_matches(self):
        debates = [
            _make_debate("d1", consensus=True, confidence=0.9),
            _make_debate("d2", consensus=False, confidence=0.7),
            _make_debate("d3", consensus=True, confidence=0.8),
        ]
        legacy = get_summary_metrics_legacy(None, debates)
        single, _, _ = process_debates_single_pass(debates, None, 24)

        assert legacy["total_debates"] == single["total_debates"]
        assert legacy["consensus_reached"] == single["consensus_reached"]
        assert legacy["consensus_rate"] == single["consensus_rate"]
        assert legacy["avg_confidence"] == single["avg_confidence"]

    def test_activity_matches(self):
        now = _now_iso()
        old = _past_iso(48)
        debates = [
            _make_debate("d1", consensus=True, domain="tech", created_at=now),
            _make_debate("d2", consensus=False, domain="finance", created_at=now),
            _make_debate("d3", consensus=True, domain="tech", created_at=old),
        ]
        legacy_act = get_recent_activity_legacy(None, 24, debates)
        _, single_act, _ = process_debates_single_pass(debates, None, 24)

        assert legacy_act["debates_last_period"] == single_act["debates_last_period"]
        assert legacy_act["consensus_last_period"] == single_act["consensus_last_period"]
        assert legacy_act["period_hours"] == single_act["period_hours"]

    def test_patterns_match(self):
        debates = [
            _make_debate(
                "d1", disagreement_report={"types": ["factual"]}, early_stopped=True
            ),
            _make_debate("d2", disagreement_report={"types": ["ethical"]}),
            _make_debate("d3", early_stopped=True),
        ]
        legacy_pat = get_debate_patterns(debates)
        _, _, single_pat = process_debates_single_pass(debates, None, 24)

        assert (
            legacy_pat["disagreement_stats"]["with_disagreements"]
            == single_pat["disagreement_stats"]["with_disagreements"]
        )
        assert (
            legacy_pat["disagreement_stats"]["disagreement_types"]
            == single_pat["disagreement_stats"]["disagreement_types"]
        )
        assert (
            legacy_pat["early_stopping"]["early_stopped"]
            == single_pat["early_stopping"]["early_stopped"]
        )
        assert (
            legacy_pat["early_stopping"]["full_duration"]
            == single_pat["early_stopping"]["full_duration"]
        )

    def test_empty_consistency(self):
        """All functions agree on empty input."""
        legacy_sum = get_summary_metrics_legacy(None, [])
        legacy_act = get_recent_activity_legacy(None, 24, [])
        legacy_pat = get_debate_patterns([])
        single_sum, single_act, single_pat = process_debates_single_pass(
            [], None, 24
        )

        assert legacy_sum["total_debates"] == single_sum["total_debates"] == 0
        assert legacy_act["debates_last_period"] == single_act["debates_last_period"] == 0
        assert (
            legacy_pat["disagreement_stats"]["with_disagreements"]
            == single_pat["disagreement_stats"]["with_disagreements"]
            == 0
        )
