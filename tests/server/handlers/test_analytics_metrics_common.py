"""
Tests for aragora.server.handlers._analytics_metrics_common.

Tests cover:
1. _parse_time_range: Parsing time range strings to datetime boundaries
2. _group_by_time: Grouping items by daily/weekly/monthly buckets
3. VALID_GRANULARITIES and VALID_TIME_RANGES constants
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.server.handlers._analytics_metrics_common import (
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _group_by_time,
    _parse_time_range,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test that exported constants have expected values."""

    def test_valid_granularities_contains_daily(self):
        assert "daily" in VALID_GRANULARITIES

    def test_valid_granularities_contains_weekly(self):
        assert "weekly" in VALID_GRANULARITIES

    def test_valid_granularities_contains_monthly(self):
        assert "monthly" in VALID_GRANULARITIES

    def test_valid_granularities_is_set(self):
        assert isinstance(VALID_GRANULARITIES, set)

    def test_valid_time_ranges_contains_expected(self):
        expected = {"7d", "14d", "30d", "90d", "180d", "365d", "all"}
        assert expected == VALID_TIME_RANGES

    def test_valid_time_ranges_is_set(self):
        assert isinstance(VALID_TIME_RANGES, set)


# =============================================================================
# _parse_time_range Tests
# =============================================================================


class TestParseTimeRange:
    """Test _parse_time_range function."""

    def test_all_returns_none(self):
        """'all' time range should return None (no start boundary)."""
        result = _parse_time_range("all")
        assert result is None

    def test_7d_returns_datetime_about_7_days_ago(self):
        """7d should return a datetime approximately 7 days ago."""
        now = datetime.now(timezone.utc)
        result = _parse_time_range("7d")
        assert result is not None
        delta = now - result
        # Allow 2 seconds tolerance for test execution
        assert 6.99 < delta.total_seconds() / 86400 < 7.01

    def test_30d_returns_datetime_about_30_days_ago(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("30d")
        assert result is not None
        delta = now - result
        assert 29.99 < delta.total_seconds() / 86400 < 30.01

    def test_90d_returns_datetime_about_90_days_ago(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("90d")
        assert result is not None
        delta = now - result
        assert 89.99 < delta.total_seconds() / 86400 < 90.01

    def test_365d_returns_datetime_about_365_days_ago(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("365d")
        assert result is not None
        delta = now - result
        assert 364.99 < delta.total_seconds() / 86400 < 365.01

    def test_14d_returns_datetime_about_14_days_ago(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("14d")
        assert result is not None
        delta = now - result
        assert 13.99 < delta.total_seconds() / 86400 < 14.01

    def test_180d_returns_datetime_about_180_days_ago(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("180d")
        assert result is not None
        delta = now - result
        assert 179.99 < delta.total_seconds() / 86400 < 180.01

    def test_invalid_string_defaults_to_30d(self):
        """Invalid time range string should default to ~30 days ago."""
        now = datetime.now(timezone.utc)
        result = _parse_time_range("invalid")
        assert result is not None
        delta = now - result
        assert 29.99 < delta.total_seconds() / 86400 < 30.01

    def test_empty_string_defaults_to_30d(self):
        now = datetime.now(timezone.utc)
        result = _parse_time_range("")
        assert result is not None
        delta = now - result
        assert 29.99 < delta.total_seconds() / 86400 < 30.01

    def test_result_is_timezone_aware(self):
        result = _parse_time_range("7d")
        assert result is not None
        assert result.tzinfo is not None

    def test_1d_returns_datetime_about_1_day_ago(self):
        """Even non-standard day counts should work with the regex."""
        now = datetime.now(timezone.utc)
        result = _parse_time_range("1d")
        assert result is not None
        delta = now - result
        assert 0.99 < delta.total_seconds() / 86400 < 1.01


# =============================================================================
# _group_by_time Tests
# =============================================================================


class TestGroupByTime:
    """Test _group_by_time function."""

    def _make_item(self, ts: datetime, label: str = "x") -> dict:
        return {"ts": ts, "label": label}

    def test_empty_items(self):
        result = _group_by_time([], "ts", "daily")
        assert result == {}

    def test_daily_grouping(self):
        """Items on the same day should be grouped together."""
        dt1 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        dt3 = datetime(2026, 1, 16, 9, 0, tzinfo=timezone.utc)

        items = [
            self._make_item(dt1, "a"),
            self._make_item(dt2, "b"),
            self._make_item(dt3, "c"),
        ]
        result = _group_by_time(items, "ts", "daily")

        assert "2026-01-15" in result
        assert "2026-01-16" in result
        assert len(result["2026-01-15"]) == 2
        assert len(result["2026-01-16"]) == 1

    def test_monthly_grouping(self):
        """Items in the same month should be grouped together."""
        dt1 = datetime(2026, 1, 5, tzinfo=timezone.utc)
        dt2 = datetime(2026, 1, 25, tzinfo=timezone.utc)
        dt3 = datetime(2026, 2, 3, tzinfo=timezone.utc)

        items = [self._make_item(dt1), self._make_item(dt2), self._make_item(dt3)]
        result = _group_by_time(items, "ts", "monthly")

        assert "2026-01" in result
        assert "2026-02" in result
        assert len(result["2026-01"]) == 2
        assert len(result["2026-02"]) == 1

    def test_weekly_grouping(self):
        """Items in the same week should be grouped together."""
        # 2026-01-12 is a Monday (week 02)
        dt1 = datetime(2026, 1, 12, tzinfo=timezone.utc)
        dt2 = datetime(2026, 1, 14, tzinfo=timezone.utc)
        # 2026-01-19 is next Monday (week 03)
        dt3 = datetime(2026, 1, 19, tzinfo=timezone.utc)

        items = [self._make_item(dt1), self._make_item(dt2), self._make_item(dt3)]
        result = _group_by_time(items, "ts", "weekly")

        # Should have 2 groups
        assert len(result) == 2

    def test_missing_timestamp_key_skips_item(self):
        """Items without the timestamp key should be skipped."""
        items = [{"other": "data"}, {"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)}]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_none_timestamp_skips_item(self):
        """Items with None timestamp should be skipped."""
        items = [{"ts": None}, {"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)}]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_string_iso_timestamp(self):
        """String ISO timestamps should be parsed correctly."""
        items = [
            {"ts": "2026-01-15T10:00:00+00:00"},
            {"ts": "2026-01-15T14:00:00Z"},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2026-01-15" in result
        assert len(result["2026-01-15"]) == 2

    def test_invalid_string_timestamp_skipped(self):
        """Invalid string timestamps should be skipped."""
        items = [
            {"ts": "not-a-date"},
            {"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_non_datetime_non_string_skipped(self):
        """Non-datetime, non-string timestamps should be skipped."""
        items = [
            {"ts": 12345},
            {"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_custom_timestamp_key(self):
        """Should use the specified timestamp key."""
        items = [
            {"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "ts": None},
        ]
        result = _group_by_time(items, "created_at", "daily")
        assert "2026-01-01" in result
        assert len(result["2026-01-01"]) == 1

    def test_preserves_full_item_in_group(self):
        """Grouped items should retain all their original fields."""
        item = {"ts": datetime(2026, 1, 1, tzinfo=timezone.utc), "extra": "data", "count": 42}
        result = _group_by_time([item], "ts", "daily")
        grouped = result["2026-01-01"][0]
        assert grouped["extra"] == "data"
        assert grouped["count"] == 42
