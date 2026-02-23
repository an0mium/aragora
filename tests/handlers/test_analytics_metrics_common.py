"""Comprehensive tests for _analytics_metrics_common.py.

Tests the shared constants and utility functions used across analytics metrics handlers:

- VALID_GRANULARITIES: set of accepted granularity values
- VALID_TIME_RANGES: set of accepted time range strings
- _parse_time_range: converts time range strings to start datetimes
- _group_by_time: groups items into time buckets by granularity
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers._analytics_metrics_common import (
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _group_by_time,
    _parse_time_range,
)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestValidGranularities:
    """Tests for the VALID_GRANULARITIES constant."""

    def test_contains_daily(self):
        assert "daily" in VALID_GRANULARITIES

    def test_contains_weekly(self):
        assert "weekly" in VALID_GRANULARITIES

    def test_contains_monthly(self):
        assert "monthly" in VALID_GRANULARITIES

    def test_exactly_three_entries(self):
        assert len(VALID_GRANULARITIES) == 3

    def test_is_a_set(self):
        assert isinstance(VALID_GRANULARITIES, set)

    def test_does_not_contain_hourly(self):
        assert "hourly" not in VALID_GRANULARITIES

    def test_does_not_contain_yearly(self):
        assert "yearly" not in VALID_GRANULARITIES


class TestValidTimeRanges:
    """Tests for the VALID_TIME_RANGES constant."""

    def test_contains_7d(self):
        assert "7d" in VALID_TIME_RANGES

    def test_contains_14d(self):
        assert "14d" in VALID_TIME_RANGES

    def test_contains_30d(self):
        assert "30d" in VALID_TIME_RANGES

    def test_contains_90d(self):
        assert "90d" in VALID_TIME_RANGES

    def test_contains_180d(self):
        assert "180d" in VALID_TIME_RANGES

    def test_contains_365d(self):
        assert "365d" in VALID_TIME_RANGES

    def test_contains_all(self):
        assert "all" in VALID_TIME_RANGES

    def test_exactly_seven_entries(self):
        assert len(VALID_TIME_RANGES) == 7

    def test_is_a_set(self):
        assert isinstance(VALID_TIME_RANGES, set)


# ---------------------------------------------------------------------------
# _parse_time_range tests
# ---------------------------------------------------------------------------


class TestParseTimeRange:
    """Tests for _parse_time_range()."""

    def test_all_returns_none(self):
        """'all' should return None (no lower bound)."""
        result = _parse_time_range("all")
        assert result is None

    def test_7d_returns_datetime_about_7_days_ago(self):
        before = datetime.now(timezone.utc) - timedelta(days=7, seconds=2)
        result = _parse_time_range("7d")
        after = datetime.now(timezone.utc) - timedelta(days=7)
        assert before <= result <= after

    def test_30d_returns_datetime_about_30_days_ago(self):
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("30d")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after

    def test_90d_returns_datetime_about_90_days_ago(self):
        before = datetime.now(timezone.utc) - timedelta(days=90, seconds=2)
        result = _parse_time_range("90d")
        after = datetime.now(timezone.utc) - timedelta(days=90)
        assert before <= result <= after

    def test_365d_returns_datetime_about_365_days_ago(self):
        before = datetime.now(timezone.utc) - timedelta(days=365, seconds=2)
        result = _parse_time_range("365d")
        after = datetime.now(timezone.utc) - timedelta(days=365)
        assert before <= result <= after

    def test_14d_returns_datetime_about_14_days_ago(self):
        before = datetime.now(timezone.utc) - timedelta(days=14, seconds=2)
        result = _parse_time_range("14d")
        after = datetime.now(timezone.utc) - timedelta(days=14)
        assert before <= result <= after

    def test_1d_returns_datetime_about_1_day_ago(self):
        """Arbitrary numeric 'd' pattern should work even if not in VALID_TIME_RANGES."""
        before = datetime.now(timezone.utc) - timedelta(days=1, seconds=2)
        result = _parse_time_range("1d")
        after = datetime.now(timezone.utc) - timedelta(days=1)
        assert before <= result <= after

    def test_invalid_string_returns_default_30d(self):
        """Non-matching strings default to 30 days."""
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("invalid")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after

    def test_empty_string_returns_default_30d(self):
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after

    def test_hours_string_returns_default(self):
        """'24h' is not a valid pattern (expects 'd' suffix)."""
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("24h")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after

    def test_result_is_timezone_aware(self):
        result = _parse_time_range("7d")
        assert result.tzinfo is not None

    def test_0d_returns_now(self):
        """0d should return approximately now."""
        before = datetime.now(timezone.utc) - timedelta(seconds=2)
        result = _parse_time_range("0d")
        after = datetime.now(timezone.utc) + timedelta(seconds=2)
        assert before <= result <= after

    def test_negative_pattern_returns_default(self):
        """'-7d' does not match the regex pattern '^(\\d+)d$'."""
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("-7d")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after

    def test_decimal_pattern_returns_default(self):
        """'7.5d' does not match the regex (only digits before 'd')."""
        before = datetime.now(timezone.utc) - timedelta(days=30, seconds=2)
        result = _parse_time_range("7.5d")
        after = datetime.now(timezone.utc) - timedelta(days=30)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# _group_by_time tests
# ---------------------------------------------------------------------------


class TestGroupByTimeDaily:
    """Tests for _group_by_time with daily granularity."""

    def test_groups_items_by_date(self):
        items = [
            {"ts": "2025-03-01T10:00:00+00:00", "val": 1},
            {"ts": "2025-03-01T14:00:00+00:00", "val": 2},
            {"ts": "2025-03-02T09:00:00+00:00", "val": 3},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 2
        assert len(result["2025-03-01"]) == 2
        assert len(result["2025-03-02"]) == 1

    def test_preserves_original_items(self):
        items = [{"ts": "2025-03-01T10:00:00+00:00", "val": "hello"}]
        result = _group_by_time(items, "ts", "daily")
        assert result["2025-03-01"][0]["val"] == "hello"

    def test_empty_list_returns_empty_dict(self):
        result = _group_by_time([], "ts", "daily")
        assert result == {}


class TestGroupByTimeWeekly:
    """Tests for _group_by_time with weekly granularity."""

    def test_groups_same_week_together(self):
        # Monday and Tuesday of same week
        items = [
            {"ts": "2025-03-03T10:00:00+00:00", "val": 1},  # Monday
            {"ts": "2025-03-04T10:00:00+00:00", "val": 2},  # Tuesday
        ]
        result = _group_by_time(items, "ts", "weekly")
        assert len(result) == 1

    def test_separates_different_weeks(self):
        items = [
            {"ts": "2025-03-03T10:00:00+00:00", "val": 1},  # Week 9
            {"ts": "2025-03-10T10:00:00+00:00", "val": 2},  # Week 10
        ]
        result = _group_by_time(items, "ts", "weekly")
        assert len(result) == 2

    def test_weekly_key_format(self):
        items = [{"ts": "2025-03-03T10:00:00+00:00", "val": 1}]
        result = _group_by_time(items, "ts", "weekly")
        keys = list(result.keys())
        assert len(keys) == 1
        # Format is YYYY-WNN
        assert keys[0].startswith("2025-W")


class TestGroupByTimeMonthly:
    """Tests for _group_by_time with monthly granularity."""

    def test_groups_same_month_together(self):
        items = [
            {"ts": "2025-03-01T10:00:00+00:00", "val": 1},
            {"ts": "2025-03-15T10:00:00+00:00", "val": 2},
            {"ts": "2025-03-31T10:00:00+00:00", "val": 3},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert len(result) == 1
        assert len(result["2025-03"]) == 3

    def test_separates_different_months(self):
        items = [
            {"ts": "2025-03-15T10:00:00+00:00", "val": 1},
            {"ts": "2025-04-15T10:00:00+00:00", "val": 2},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert len(result) == 2
        assert "2025-03" in result
        assert "2025-04" in result

    def test_monthly_key_format(self):
        items = [{"ts": "2025-01-15T10:00:00+00:00", "val": 1}]
        result = _group_by_time(items, "ts", "monthly")
        assert "2025-01" in result


class TestGroupByTimeTimestampParsing:
    """Tests for timestamp parsing in _group_by_time."""

    def test_handles_iso_format_with_z_suffix(self):
        items = [{"ts": "2025-03-01T10:00:00Z", "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert "2025-03-01" in result

    def test_handles_iso_format_with_offset(self):
        items = [{"ts": "2025-03-01T10:00:00+05:30", "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_handles_datetime_objects(self):
        dt = datetime(2025, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        items = [{"ts": dt, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert "2025-03-01" in result

    def test_handles_naive_datetime_objects(self):
        dt = datetime(2025, 3, 1, 10, 0, 0)
        items = [{"ts": dt, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert "2025-03-01" in result

    def test_skips_items_with_missing_timestamp(self):
        items = [
            {"ts": "2025-03-01T10:00:00+00:00", "val": 1},
            {"val": 2},  # no ts key
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1
        assert len(result["2025-03-01"]) == 1

    def test_skips_items_with_none_timestamp(self):
        items = [{"ts": None, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_skips_items_with_invalid_string_timestamp(self):
        items = [{"ts": "not-a-date", "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_skips_items_with_numeric_timestamp(self):
        """Numeric timestamps (epoch) are not supported, should be skipped."""
        items = [{"ts": 1709290800, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_skips_items_with_boolean_timestamp(self):
        items = [{"ts": True, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_skips_items_with_empty_string_timestamp(self):
        items = [{"ts": "", "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_custom_timestamp_key(self):
        items = [{"created_at": "2025-03-01T10:00:00+00:00", "val": 1}]
        result = _group_by_time(items, "created_at", "daily")
        assert "2025-03-01" in result


class TestGroupByTimeEdgeCases:
    """Edge cases for _group_by_time."""

    def test_large_number_of_items(self):
        base_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        items = [{"ts": (base_dt + timedelta(days=i)).isoformat(), "val": i} for i in range(100)]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 100  # each day unique

    def test_all_items_same_bucket(self):
        items = [{"ts": "2025-03-01T01:00:00+00:00", "val": i} for i in range(10)]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1
        assert len(result["2025-03-01"]) == 10

    def test_mixed_valid_and_invalid_timestamps(self):
        items = [
            {"ts": "2025-03-01T10:00:00+00:00", "val": 1},
            {"ts": "invalid", "val": 2},
            {"ts": "2025-03-01T12:00:00+00:00", "val": 3},
            {"ts": None, "val": 4},
            {"val": 5},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1
        assert len(result["2025-03-01"]) == 2

    def test_mixed_datetime_and_string_timestamps(self):
        dt = datetime(2025, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        items = [
            {"ts": dt, "val": 1},
            {"ts": "2025-03-01T14:00:00+00:00", "val": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1
        assert len(result["2025-03-01"]) == 2

    def test_returns_regular_dict_not_defaultdict(self):
        """Return value should be a plain dict, not a defaultdict."""
        items = [{"ts": "2025-03-01T10:00:00+00:00", "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert type(result) is dict

    def test_unknown_granularity_defaults_to_monthly(self):
        """Any granularity not 'daily' or 'weekly' falls through to monthly."""
        items = [{"ts": "2025-03-15T10:00:00+00:00", "val": 1}]
        result = _group_by_time(items, "ts", "quarterly")
        # Falls through to else branch which is monthly
        assert "2025-03" in result

    def test_year_boundary(self):
        items = [
            {"ts": "2024-12-31T23:00:00+00:00", "val": 1},
            {"ts": "2025-01-01T01:00:00+00:00", "val": 2},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert len(result) == 2
        assert "2024-12" in result
        assert "2025-01" in result
