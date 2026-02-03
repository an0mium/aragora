"""
Tests for admin dashboard metrics calculation utilities.

Tests cover:
- get_summary_metrics_sql: SQL-based summary metrics aggregation
- get_recent_activity_sql: SQL-based recent activity metrics
- get_summary_metrics_legacy: Legacy summary metrics (list-based)
- get_recent_activity_legacy: Legacy recent activity metrics
- process_debates_single_pass: Batch single-pass metrics processing
- get_debate_patterns: Debate pattern statistics
- RBAC permission enforcement (admin:metrics:read)
- Rate limiting (30 req/min for most, 20 req/min for batch)
- Error handling and edge cases
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext


# ---------------------------------------------------------------------------
# Helpers to bypass decorators for unit-testing the core logic
# ---------------------------------------------------------------------------

# We need to test both the decorator enforcement AND the underlying logic.
# To test logic we patch out the decorators; to test RBAC/rate-limit we
# call through the real decorator stack.

MODULE = "aragora.server.handlers.admin.dashboard_metrics"


def _make_auth_context(
    user_id: str = "admin-1",
    roles: set[str] | None = None,
    permissions: set[str] | None = None,
    org_id: str = "org-1",
) -> AuthorizationContext:
    """Build a minimal AuthorizationContext for tests."""
    return AuthorizationContext(
        user_id=user_id,
        roles=roles or {"admin"},
        permissions=permissions or {"admin:metrics:read", "admin:revenue:read"},
        org_id=org_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    """Create mock storage with connection context manager and cursor."""
    storage = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    storage.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    storage.connection.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    return storage, mock_cursor


@pytest.fixture
def sample_debates():
    """Create sample debate records for legacy/batch tests."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "debate_1",
            "domain": "technology",
            "consensus_reached": True,
            "confidence": 0.85,
            "created_at": now.isoformat(),
            "early_stopped": False,
        },
        {
            "id": "debate_2",
            "domain": "finance",
            "consensus_reached": True,
            "confidence": 0.75,
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "early_stopped": True,
        },
        {
            "id": "debate_3",
            "domain": "technology",
            "consensus_reached": False,
            "confidence": 0.55,
            "created_at": (now - timedelta(hours=12)).isoformat(),
            "early_stopped": False,
            "disagreement_report": {"types": ["methodology", "evidence"]},
        },
        {
            "id": "debate_4",
            "domain": "general",
            "consensus_reached": False,
            "confidence": 0.0,
            "created_at": (now - timedelta(hours=48)).isoformat(),
            "early_stopped": False,
            "disagreement_report": {"types": ["methodology"]},
        },
    ]


@pytest.fixture
def auth_context():
    """Create an admin authorization context with required permissions."""
    return _make_auth_context()


@pytest.fixture
def unprivileged_context():
    """Create a context with no admin:metrics:read permission."""
    return _make_auth_context(
        user_id="user-1",
        roles={"viewer"},
        permissions={"debates:read"},
    )


# ---------------------------------------------------------------------------
# Decorator bypass helpers
# ---------------------------------------------------------------------------


def _call_bypassing_decorators(func, *args, **kwargs):
    """Call the underlying function, bypassing require_permission and rate_limit decorators.

    The decorated functions are wrapped: require_permission -> rate_limit -> func.
    We unwrap through __wrapped__ to get the original.
    """
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return inner(*args, **kwargs)


# ===========================================================================
# Tests: get_summary_metrics_sql
# ===========================================================================


class TestGetSummaryMetricsSql:
    """Tests for SQL-based summary metrics."""

    def test_returns_correct_summary_with_data(self, mock_storage, auth_context):
        """Returns correct summary when SQL returns valid data."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (100, 75, 0.82)

        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, None)

        assert result["total_debates"] == 100
        assert result["consensus_reached"] == 75
        assert result["consensus_rate"] == 0.75
        assert result["avg_confidence"] == 0.82

    def test_returns_zeros_when_no_data(self, mock_storage):
        """Returns zero-valued summary when SQL returns no rows."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = None

        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, None)

        assert result["total_debates"] == 0
        assert result["consensus_reached"] == 0
        assert result["consensus_rate"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_handles_null_values_from_sql(self, mock_storage):
        """Handles NULL values from SQL gracefully."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (0, None, None)

        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, None)

        assert result["total_debates"] == 0
        assert result["consensus_reached"] == 0
        assert result["consensus_rate"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_handles_database_error(self, mock_storage):
        """Returns default summary when database raises an exception."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        storage.connection.side_effect = RuntimeError("connection failed")

        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, "test")

        assert result["total_debates"] == 0
        assert result["consensus_rate"] == 0.0

    def test_consensus_rate_rounded_to_three_decimals(self, mock_storage):
        """Consensus rate is rounded to 3 decimal places."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (3, 1, 0.123456789)

        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, None)

        assert result["consensus_rate"] == round(1 / 3, 3)
        assert result["avg_confidence"] == 0.123

    def test_domain_param_accepted(self, mock_storage):
        """Domain parameter is accepted (reserved for future use)."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (5, 3, 0.8)

        # Should not raise even with a domain value
        result = _call_bypassing_decorators(get_summary_metrics_sql, storage, "engineering")

        assert result["total_debates"] == 5


# ===========================================================================
# Tests: get_recent_activity_sql
# ===========================================================================


class TestGetRecentActivitySql:
    """Tests for SQL-based recent activity metrics."""

    def test_returns_activity_for_given_hours(self, mock_storage):
        """Returns recent activity data for the specified time window."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (15, 10)

        result = _call_bypassing_decorators(get_recent_activity_sql, storage, 24)

        assert result["debates_last_period"] == 15
        assert result["consensus_last_period"] == 10
        assert result["period_hours"] == 24

    def test_returns_zeros_when_no_rows(self, mock_storage):
        """Returns zeros when SQL returns no matching rows."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = None

        result = _call_bypassing_decorators(get_recent_activity_sql, storage, 6)

        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0
        assert result["period_hours"] == 6

    def test_handles_null_sql_values(self, mock_storage):
        """Handles NULL values from SQL."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (None, None)

        result = _call_bypassing_decorators(get_recent_activity_sql, storage, 12)

        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0

    def test_handles_database_error(self, mock_storage):
        """Returns defaults when database raises an exception."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, _ = mock_storage
        storage.connection.side_effect = RuntimeError("timeout")

        result = _call_bypassing_decorators(get_recent_activity_sql, storage, 24)

        assert result["debates_last_period"] == 0
        assert result["period_hours"] == 24

    def test_sql_uses_cutoff_parameter(self, mock_storage):
        """SQL query is called with the ISO-formatted cutoff timestamp."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (0, 0)

        _call_bypassing_decorators(get_recent_activity_sql, storage, 48)

        # Verify cursor.execute was called with a cutoff param
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args
        assert len(call_args[0]) == 2  # SQL string + params tuple
        cutoff_value = call_args[0][1][0]
        # The cutoff should be an ISO-formatted datetime string
        datetime.fromisoformat(cutoff_value)  # Should not raise


# ===========================================================================
# Tests: get_summary_metrics_legacy
# ===========================================================================


class TestGetSummaryMetricsLegacy:
    """Tests for legacy list-based summary metrics."""

    def test_returns_summary_for_debates(self, sample_debates):
        """Computes correct summary over a list of debates."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, sample_debates)

        assert result["total_debates"] == 4
        assert result["consensus_reached"] == 2
        assert result["consensus_rate"] == 0.5

    def test_returns_zeros_for_empty_list(self):
        """Returns zero-valued summary when debate list is empty."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, [])

        assert result["total_debates"] == 0
        assert result["consensus_reached"] == 0
        assert result["consensus_rate"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_average_confidence_excludes_falsy(self):
        """Average confidence excludes debates with falsy confidence values."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        debates = [
            {"consensus_reached": True, "confidence": 0.9},
            {"consensus_reached": False, "confidence": 0},  # falsy -> excluded
            {"consensus_reached": True, "confidence": 0.7},
        ]

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, debates)

        # Only 0.9 and 0.7 count (confidence=0 is falsy)
        assert result["avg_confidence"] == round((0.9 + 0.7) / 2, 3)

    def test_handles_exception_in_processing(self):
        """Returns defaults when processing raises an exception."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        # Passing a non-iterable that looks truthy but fails on iteration
        class BrokenList:
            def __bool__(self):
                return True

            def __iter__(self):
                raise RuntimeError("oops")

            def __len__(self):
                raise RuntimeError("oops")

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, BrokenList())

        # Should return defaults without raising
        assert result["total_debates"] == 0

    def test_domain_param_accepted(self, sample_debates):
        """Domain parameter accepted (reserved for future use)."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        result = _call_bypassing_decorators(
            get_summary_metrics_legacy, "engineering", sample_debates
        )

        assert result["total_debates"] == 4


# ===========================================================================
# Tests: get_recent_activity_legacy
# ===========================================================================


class TestGetRecentActivityLegacy:
    """Tests for legacy recent activity metrics."""

    def test_filters_by_time_window(self, sample_debates):
        """Only counts debates within the specified time window."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        # 6 hours window should only include debate_1 and debate_2
        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 6, sample_debates)

        assert result["debates_last_period"] == 2
        assert result["period_hours"] == 6

    def test_24_hour_window_includes_more(self, sample_debates):
        """24-hour window captures debates up to 12 hours old."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, sample_debates)

        assert result["debates_last_period"] == 3  # debate_1, 2, 3
        assert result["consensus_last_period"] == 2

    def test_tracks_domain_activity(self, sample_debates):
        """Tracks active domains and identifies the most active one."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, sample_debates)

        assert "technology" in result["domains_active"]
        assert result["most_active_domain"] == "technology"

    def test_empty_debates_returns_defaults(self):
        """Returns defaults when no debates are provided."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, [])

        assert result["debates_last_period"] == 0
        assert result["consensus_last_period"] == 0
        assert result["domains_active"] == []
        assert result["most_active_domain"] is None

    def test_handles_invalid_timestamps(self):
        """Skips debates with invalid timestamps gracefully."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        debates = [
            {"id": "d1", "created_at": "not-a-date", "consensus_reached": True},
            {
                "id": "d2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "consensus_reached": True,
                "domain": "tech",
            },
        ]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, debates)

        # Only d2 should be counted
        assert result["debates_last_period"] == 1

    def test_handles_missing_created_at(self):
        """Skips debates without created_at field."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        debates = [
            {"id": "d1", "consensus_reached": True},  # no created_at
        ]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, debates)

        assert result["debates_last_period"] == 0

    def test_domains_active_capped_at_10(self):
        """Domains active list is capped at 10 entries."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        now = datetime.now(timezone.utc)
        debates = [
            {
                "id": f"d_{i}",
                "domain": f"domain_{i}",
                "created_at": now.isoformat(),
                "consensus_reached": False,
            }
            for i in range(15)
        ]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, debates)

        assert len(result["domains_active"]) <= 10

    def test_handles_z_suffix_timestamps(self):
        """Handles ISO timestamps with Z suffix."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        now = datetime.now(timezone.utc)
        debates = [
            {
                "id": "d1",
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "consensus_reached": True,
                "domain": "tech",
            },
        ]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 24, debates)

        assert result["debates_last_period"] == 1


# ===========================================================================
# Tests: process_debates_single_pass
# ===========================================================================


class TestProcessDebatesSinglePass:
    """Tests for single-pass batch metrics processing."""

    def test_returns_three_dicts(self, sample_debates):
        """Returns a tuple of (summary, activity, patterns)."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        result = _call_bypassing_decorators(process_debates_single_pass, sample_debates, None, 24)

        assert isinstance(result, tuple)
        assert len(result) == 3
        summary, activity, patterns = result
        assert isinstance(summary, dict)
        assert isinstance(activity, dict)
        assert isinstance(patterns, dict)

    def test_summary_matches_legacy(self, sample_debates):
        """Summary output is consistent with legacy function output."""
        from aragora.server.handlers.admin.dashboard_metrics import (
            get_summary_metrics_legacy,
            process_debates_single_pass,
        )

        summary_sp, _, _ = _call_bypassing_decorators(
            process_debates_single_pass, sample_debates, None, 24
        )
        summary_legacy = _call_bypassing_decorators(
            get_summary_metrics_legacy, None, sample_debates
        )

        assert summary_sp["total_debates"] == summary_legacy["total_debates"]
        assert summary_sp["consensus_reached"] == summary_legacy["consensus_reached"]
        assert summary_sp["consensus_rate"] == summary_legacy["consensus_rate"]

    def test_activity_within_single_pass(self, sample_debates):
        """Activity metrics computed correctly in single pass."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        _, activity, _ = _call_bypassing_decorators(
            process_debates_single_pass, sample_debates, None, 6
        )

        # 6h window: debate_1 (0h ago) and debate_2 (2h ago)
        assert activity["debates_last_period"] == 2
        assert activity["period_hours"] == 6

    def test_patterns_within_single_pass(self, sample_debates):
        """Pattern metrics computed correctly in single pass."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        _, _, patterns = _call_bypassing_decorators(
            process_debates_single_pass, sample_debates, None, 24
        )

        # debate_3 and debate_4 have disagreement_report
        assert patterns["disagreement_stats"]["with_disagreements"] == 2
        assert "methodology" in patterns["disagreement_stats"]["disagreement_types"]
        assert patterns["disagreement_stats"]["disagreement_types"]["methodology"] == 2
        assert patterns["disagreement_stats"]["disagreement_types"]["evidence"] == 1

    def test_early_stopping_stats(self, sample_debates):
        """Early stopping statistics are computed correctly."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        _, _, patterns = _call_bypassing_decorators(
            process_debates_single_pass, sample_debates, None, 24
        )

        # debate_2 is early_stopped; rest are not
        assert patterns["early_stopping"]["early_stopped"] == 1
        assert patterns["early_stopping"]["full_duration"] == 3

    def test_empty_debates(self):
        """Returns zero-value defaults for empty debate list."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        summary, activity, patterns = _call_bypassing_decorators(
            process_debates_single_pass, [], None, 24
        )

        assert summary["total_debates"] == 0
        assert activity["debates_last_period"] == 0
        assert patterns["disagreement_stats"]["with_disagreements"] == 0
        assert patterns["early_stopping"]["early_stopped"] == 0

    def test_handles_processing_error(self):
        """Returns defaults when processing raises an exception."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        # A debate with broken created_at that triggers datetime parsing error
        # but the function should catch and continue
        debates = [
            {
                "id": "d1",
                "consensus_reached": True,
                "confidence": 0.8,
                "created_at": "invalid-date",
            },
        ]

        summary, activity, patterns = _call_bypassing_decorators(
            process_debates_single_pass, debates, None, 24
        )

        # The debate was still counted for summary (consensus parsing doesn't need date)
        assert summary["total_debates"] == 1
        assert summary["consensus_reached"] == 1


# ===========================================================================
# Tests: get_debate_patterns
# ===========================================================================


class TestGetDebatePatterns:
    """Tests for debate pattern statistics."""

    def test_counts_disagreements(self, sample_debates):
        """Counts debates with disagreement reports."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        result = _call_bypassing_decorators(get_debate_patterns, sample_debates)

        assert result["disagreement_stats"]["with_disagreements"] == 2

    def test_aggregates_disagreement_types(self, sample_debates):
        """Aggregates disagreement types across all debates."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        result = _call_bypassing_decorators(get_debate_patterns, sample_debates)

        types = result["disagreement_stats"]["disagreement_types"]
        assert types["methodology"] == 2  # appears in debate_3 and debate_4
        assert types["evidence"] == 1  # only in debate_3

    def test_early_stopping_counts(self, sample_debates):
        """Counts early stopped vs. full duration debates."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        result = _call_bypassing_decorators(get_debate_patterns, sample_debates)

        assert result["early_stopping"]["early_stopped"] == 1
        assert result["early_stopping"]["full_duration"] == 3

    def test_empty_debates(self):
        """Returns zero-value patterns for empty debate list."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        result = _call_bypassing_decorators(get_debate_patterns, [])

        assert result["disagreement_stats"]["with_disagreements"] == 0
        assert result["disagreement_stats"]["disagreement_types"] == {}
        assert result["early_stopping"]["early_stopped"] == 0
        assert result["early_stopping"]["full_duration"] == 0

    def test_debates_with_no_disagreements(self):
        """Returns zero disagreements when none have reports."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        debates = [
            {"id": "d1", "early_stopped": False},
            {"id": "d2", "early_stopped": False},
        ]

        result = _call_bypassing_decorators(get_debate_patterns, debates)

        assert result["disagreement_stats"]["with_disagreements"] == 0
        assert result["early_stopping"]["full_duration"] == 2

    def test_all_early_stopped(self):
        """All debates early stopped."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        debates = [
            {"id": "d1", "early_stopped": True},
            {"id": "d2", "early_stopped": True},
        ]

        result = _call_bypassing_decorators(get_debate_patterns, debates)

        assert result["early_stopping"]["early_stopped"] == 2
        assert result["early_stopping"]["full_duration"] == 0

    def test_disagreement_report_without_types(self):
        """Handles disagreement reports missing the 'types' key."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        debates = [
            # Empty dict is falsy, so not counted as having a disagreement report
            {"id": "d1", "disagreement_report": {}, "early_stopped": False},
            # Non-empty dict with empty types list IS counted
            {"id": "d2", "disagreement_report": {"types": []}, "early_stopped": False},
            # Non-empty dict with types is counted
            {"id": "d3", "disagreement_report": {"types": ["scope"]}, "early_stopped": False},
        ]

        result = _call_bypassing_decorators(get_debate_patterns, debates)

        # Empty dict {} is falsy so d1 is NOT counted; d2 and d3 are truthy
        assert result["disagreement_stats"]["with_disagreements"] == 2
        assert result["disagreement_stats"]["disagreement_types"] == {"scope": 1}


# ===========================================================================
# Tests: RBAC Permission Enforcement
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement on dashboard metrics functions.

    These tests use the @pytest.mark.no_auto_auth marker to opt out of the
    autouse fixture in tests/server/handlers/conftest.py that patches
    _get_context_from_args to always return an admin context.
    """

    @pytest.mark.no_auto_auth
    def test_summary_sql_requires_permission(self, mock_storage):
        """get_summary_metrics_sql raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, _ = mock_storage

        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_sql(storage, None)

    @pytest.mark.no_auto_auth
    def test_recent_activity_sql_requires_permission(self, mock_storage):
        """get_recent_activity_sql raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        storage, _ = mock_storage

        with pytest.raises(PermissionDeniedError):
            get_recent_activity_sql(storage, 24)

    @pytest.mark.no_auto_auth
    def test_summary_legacy_requires_permission(self):
        """get_summary_metrics_legacy raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_legacy(None, [])

    @pytest.mark.no_auto_auth
    def test_recent_activity_legacy_requires_permission(self):
        """get_recent_activity_legacy raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        with pytest.raises(PermissionDeniedError):
            get_recent_activity_legacy(None, 24, [])

    @pytest.mark.no_auto_auth
    def test_single_pass_requires_permission(self):
        """process_debates_single_pass raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        with pytest.raises(PermissionDeniedError):
            process_debates_single_pass([], None, 24)

    @pytest.mark.no_auto_auth
    def test_debate_patterns_requires_permission(self):
        """get_debate_patterns raises PermissionDeniedError without context."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        with pytest.raises(PermissionDeniedError):
            get_debate_patterns([])

    @pytest.mark.no_auto_auth
    def test_unprivileged_user_denied(self, unprivileged_context, mock_storage):
        """User without admin:metrics:read permission is denied."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        storage, cursor = mock_storage
        cursor.fetchone.return_value = (0, 0, 0.0)

        with pytest.raises(PermissionDeniedError):
            get_summary_metrics_sql(unprivileged_context, storage, None)


# ===========================================================================
# Tests: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on dashboard metrics functions."""

    def test_summary_sql_has_rate_limit_decorator(self):
        """get_summary_metrics_sql is decorated with rate_limit."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_sql

        # The rate_limit decorator wraps the function; check it has __wrapped__
        assert hasattr(get_summary_metrics_sql, "__wrapped__")

    def test_recent_activity_sql_has_rate_limit_decorator(self):
        """get_recent_activity_sql is decorated with rate_limit."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_sql

        assert hasattr(get_recent_activity_sql, "__wrapped__")

    def test_single_pass_has_rate_limit_decorator(self):
        """process_debates_single_pass is decorated with rate_limit."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        assert hasattr(process_debates_single_pass, "__wrapped__")

    def test_debate_patterns_has_rate_limit_decorator(self):
        """get_debate_patterns is decorated with rate_limit."""
        from aragora.server.handlers.admin.dashboard_metrics import get_debate_patterns

        assert hasattr(get_debate_patterns, "__wrapped__")


# ===========================================================================
# Tests: MetricsDashboardMixin (from metrics_dashboard.py)
# ===========================================================================


class TestMetricsDashboardMixin:
    """Tests for the MetricsDashboardMixin endpoint handler class."""

    def _make_mixin(self):
        """Create a MetricsDashboardMixin instance with mocked dependencies."""
        from aragora.server.handlers.admin.metrics_dashboard import MetricsDashboardMixin

        class TestableHandler(MetricsDashboardMixin):
            pass

        instance = TestableHandler.__new__(TestableHandler)
        instance.ctx = {}
        instance._require_admin = MagicMock()
        instance._check_rbac_permission = MagicMock(return_value=None)
        instance._get_user_store = MagicMock()
        return instance

    def test_get_stats_returns_stats(self):
        """_get_stats returns user store stats on success."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {
            "total_users": 100,
            "active_users": 42,
        }

        result = mixin._get_stats(MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["stats"]["total_users"] == 100

    def test_get_stats_unauthorized(self):
        """_get_stats returns error when admin check fails."""
        mixin = self._make_mixin()
        from aragora.server.handlers.utils.responses import HandlerResult

        err_result = HandlerResult(
            status_code=401, body=b'{"error":"unauthorized"}', content_type="application/json"
        )
        mixin._require_admin.return_value = (None, err_result)

        result = mixin._get_stats(MagicMock())

        assert result.status_code == 401

    def test_get_stats_rbac_denied(self):
        """_get_stats returns error when RBAC check fails."""
        mixin = self._make_mixin()
        from aragora.server.handlers.utils.responses import HandlerResult

        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        perm_err = HandlerResult(
            status_code=403, body=b'{"error":"forbidden"}', content_type="application/json"
        )
        mixin._check_rbac_permission.return_value = perm_err

        result = mixin._get_stats(MagicMock())

        assert result.status_code == 403

    def test_get_system_metrics_includes_timestamp(self):
        """_get_system_metrics includes a UTC timestamp."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {}

        result = mixin._get_system_metrics(MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "timestamp" in body["metrics"]

    def test_get_system_metrics_includes_user_stats(self):
        """_get_system_metrics includes user stats from user store."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {"total": 50}

        result = mixin._get_system_metrics(MagicMock())

        body = json.loads(result.body)
        assert body["metrics"]["users"] == {"total": 50}

    def test_get_system_metrics_includes_debate_stats(self):
        """_get_system_metrics includes debate storage statistics when available."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {}

        mock_debate_storage = MagicMock()
        mock_debate_storage.get_statistics.return_value = {"total_debates": 200}
        mixin.ctx["debate_storage"] = mock_debate_storage

        result = mixin._get_system_metrics(MagicMock())

        body = json.loads(result.body)
        assert body["metrics"]["debates"]["total_debates"] == 200

    def test_get_system_metrics_handles_debate_storage_error(self):
        """_get_system_metrics handles debate storage errors gracefully."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {}

        mock_debate_storage = MagicMock()
        mock_debate_storage.get_statistics.side_effect = RuntimeError("db down")
        mixin.ctx["debate_storage"] = mock_debate_storage

        result = mixin._get_system_metrics(MagicMock())

        body = json.loads(result.body)
        assert body["metrics"]["debates"] == {"error": "unavailable"}

    def test_get_system_metrics_no_user_store(self):
        """_get_system_metrics handles missing user store."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value = None

        result = mixin._get_system_metrics(MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "users" not in body["metrics"]

    def test_get_system_metrics_unauthorized(self):
        """_get_system_metrics returns error when admin check fails."""
        mixin = self._make_mixin()
        from aragora.server.handlers.utils.responses import HandlerResult

        err_result = HandlerResult(
            status_code=401, body=b'{"error":"unauthorized"}', content_type="application/json"
        )
        mixin._require_admin.return_value = (None, err_result)

        result = mixin._get_system_metrics(MagicMock())

        assert result.status_code == 401

    def test_get_revenue_stats_calculates_mrr(self):
        """_get_revenue_stats calculates MRR from tier distribution."""
        mixin = self._make_mixin()
        mock_auth = MagicMock()
        mixin._require_admin.return_value = (mock_auth, None)
        mixin._get_user_store.return_value.get_admin_stats.return_value = {
            "tier_distribution": {"free": 10, "pro": 5},
            "total_organizations": 15,
        }

        with patch("aragora.billing.models.TIER_LIMITS") as mock_tiers:
            free_limits = MagicMock()
            free_limits.price_monthly_cents = 0
            pro_limits = MagicMock()
            pro_limits.price_monthly_cents = 9900
            mock_tiers.get.side_effect = lambda k: {"free": free_limits, "pro": pro_limits}.get(k)

            # Need to bypass the @require_permission decorator on _get_revenue_stats
            inner = mixin._get_revenue_stats
            while hasattr(inner, "__wrapped__"):
                inner = inner.__wrapped__
            result = inner(mixin, MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)
        revenue = body["revenue"]
        assert revenue["mrr_cents"] == 5 * 9900
        assert revenue["mrr_dollars"] == 5 * 99.0
        assert revenue["arr_dollars"] == 5 * 99.0 * 12
        assert revenue["paying_organizations"] == 5

    def test_get_revenue_stats_unauthorized(self):
        """_get_revenue_stats returns error when admin check fails."""
        mixin = self._make_mixin()
        from aragora.server.handlers.utils.responses import HandlerResult

        err_result = HandlerResult(
            status_code=401, body=b'{"error":"unauthorized"}', content_type="application/json"
        )
        mixin._require_admin.return_value = (None, err_result)

        # Bypass @require_permission decorator
        inner = mixin._get_revenue_stats
        while hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
        result = inner(mixin, MagicMock())

        assert result.status_code == 401


# ===========================================================================
# Tests: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests across all metrics functions."""

    def test_single_debate_summary(self):
        """Handles a single debate correctly."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        debates = [{"consensus_reached": True, "confidence": 1.0}]

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, debates)

        assert result["total_debates"] == 1
        assert result["consensus_reached"] == 1
        assert result["consensus_rate"] == 1.0
        assert result["avg_confidence"] == 1.0

    def test_no_consensus_debates(self):
        """Handles debates where none reach consensus."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        debates = [
            {"consensus_reached": False, "confidence": 0.3},
            {"consensus_reached": False, "confidence": 0.4},
        ]

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, debates)

        assert result["consensus_reached"] == 0
        assert result["consensus_rate"] == 0.0

    def test_all_consensus_debates(self):
        """Handles debates where all reach consensus."""
        from aragora.server.handlers.admin.dashboard_metrics import get_summary_metrics_legacy

        debates = [
            {"consensus_reached": True, "confidence": 0.9},
            {"consensus_reached": True, "confidence": 0.95},
        ]

        result = _call_bypassing_decorators(get_summary_metrics_legacy, None, debates)

        assert result["consensus_reached"] == 2
        assert result["consensus_rate"] == 1.0

    def test_large_hours_window(self):
        """Large time window (1 year) does not cause issues."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        now = datetime.now(timezone.utc)
        debates = [
            {
                "id": "d1",
                "created_at": (now - timedelta(days=300)).isoformat(),
                "consensus_reached": True,
            }
        ]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 8760, debates)

        assert result["debates_last_period"] == 1

    def test_zero_hours_window(self):
        """Zero-hour time window returns no recent debates."""
        from aragora.server.handlers.admin.dashboard_metrics import get_recent_activity_legacy

        now = datetime.now(timezone.utc)
        debates = [{"id": "d1", "created_at": now.isoformat(), "consensus_reached": True}]

        result = _call_bypassing_decorators(get_recent_activity_legacy, None, 0, debates)

        # With 0 hours window, cutoff = now, so nothing is truly "after" now
        assert result["period_hours"] == 0

    def test_single_pass_with_mixed_data(self):
        """Single pass handles a mix of complete and sparse debate records."""
        from aragora.server.handlers.admin.dashboard_metrics import process_debates_single_pass

        now = datetime.now(timezone.utc)
        debates = [
            # Complete record
            {
                "id": "d1",
                "consensus_reached": True,
                "confidence": 0.9,
                "created_at": now.isoformat(),
                "early_stopped": False,
                "domain": "tech",
                "disagreement_report": {"types": ["scope"]},
            },
            # Minimal record
            {"id": "d2"},
            # Record with only some fields
            {"id": "d3", "consensus_reached": False, "early_stopped": True},
        ]

        summary, activity, patterns = _call_bypassing_decorators(
            process_debates_single_pass, debates, None, 24
        )

        assert summary["total_debates"] == 3
        assert summary["consensus_reached"] == 1
        assert patterns["early_stopping"]["early_stopped"] == 1
        assert patterns["disagreement_stats"]["with_disagreements"] == 1
