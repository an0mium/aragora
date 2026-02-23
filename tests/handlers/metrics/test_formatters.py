"""Comprehensive tests for metrics formatting utilities.

Covers all public functions, branches, edge cases, and boundary conditions
for the format_uptime and format_size helper functions.

Test classes:
  TestFormatUptimeDays       - Uptime with days component
  TestFormatUptimeHours      - Uptime with hours (no days)
  TestFormatUptimeMinutes    - Uptime with minutes (no hours)
  TestFormatUptimeSeconds    - Uptime with seconds only
  TestFormatUptimeBoundaries - Boundary values between tiers
  TestFormatUptimeEdgeCases  - Zero, fractional, large values
  TestFormatSizeBytes        - Sizes in byte range
  TestFormatSizeKB           - Sizes in kilobyte range
  TestFormatSizeMB           - Sizes in megabyte range
  TestFormatSizeGB           - Sizes in gigabyte range
  TestFormatSizeTB           - Sizes in terabyte range
  TestFormatSizeEdgeCases    - Zero, exact boundaries, large values
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.metrics.formatters import format_size, format_uptime


# ---------------------------------------------------------------------------
# format_uptime – days branch
# ---------------------------------------------------------------------------


class TestFormatUptimeDays:
    """Tests where days > 0."""

    def test_one_day_exact(self):
        result = format_uptime(86400)
        assert result == "1d 0h 0m"

    def test_one_day_with_hours_and_minutes(self):
        # 1 day, 5 hours, 30 minutes
        seconds = 86400 + 5 * 3600 + 30 * 60
        result = format_uptime(seconds)
        assert result == "1d 5h 30m"

    def test_multiple_days(self):
        # 7 days, 12 hours, 45 minutes
        seconds = 7 * 86400 + 12 * 3600 + 45 * 60
        result = format_uptime(seconds)
        assert result == "7d 12h 45m"

    def test_days_branch_omits_seconds(self):
        # 2 days, 3 hours, 10 minutes, 55 seconds — seconds are not shown
        seconds = 2 * 86400 + 3 * 3600 + 10 * 60 + 55
        result = format_uptime(seconds)
        assert result == "2d 3h 10m"


# ---------------------------------------------------------------------------
# format_uptime – hours branch
# ---------------------------------------------------------------------------


class TestFormatUptimeHours:
    """Tests where hours > 0 but days == 0."""

    def test_one_hour_exact(self):
        result = format_uptime(3600)
        assert result == "1h 0m 0s"

    def test_hours_minutes_seconds(self):
        seconds = 5 * 3600 + 30 * 60 + 12
        result = format_uptime(seconds)
        assert result == "5h 30m 12s"

    def test_max_hours_before_days(self):
        # 23 hours, 59 minutes, 59 seconds — still in hours branch
        seconds = 23 * 3600 + 59 * 60 + 59
        result = format_uptime(seconds)
        assert result == "23h 59m 59s"


# ---------------------------------------------------------------------------
# format_uptime – minutes branch
# ---------------------------------------------------------------------------


class TestFormatUptimeMinutes:
    """Tests where minutes > 0 but hours and days == 0."""

    def test_one_minute_exact(self):
        result = format_uptime(60)
        assert result == "1m 0s"

    def test_minutes_and_seconds(self):
        result = format_uptime(45 * 60 + 12)
        assert result == "45m 12s"

    def test_max_minutes_before_hours(self):
        # 59 minutes, 59 seconds — still in minutes branch
        result = format_uptime(59 * 60 + 59)
        assert result == "59m 59s"


# ---------------------------------------------------------------------------
# format_uptime – seconds branch
# ---------------------------------------------------------------------------


class TestFormatUptimeSeconds:
    """Tests where only seconds remain (< 60)."""

    def test_one_second(self):
        result = format_uptime(1)
        assert result == "1s"

    def test_zero_seconds(self):
        result = format_uptime(0)
        assert result == "0s"

    def test_59_seconds(self):
        result = format_uptime(59)
        assert result == "59s"

    def test_thirty_seconds(self):
        result = format_uptime(30)
        assert result == "30s"


# ---------------------------------------------------------------------------
# format_uptime – boundary values
# ---------------------------------------------------------------------------


class TestFormatUptimeBoundaries:
    """Tests at exact tier boundaries."""

    def test_just_below_one_minute(self):
        result = format_uptime(59.9)
        assert result == "59s"

    def test_exactly_one_minute(self):
        result = format_uptime(60.0)
        assert result == "1m 0s"

    def test_just_below_one_hour(self):
        result = format_uptime(3599)
        assert result == "59m 59s"

    def test_exactly_one_hour(self):
        result = format_uptime(3600.0)
        assert result == "1h 0m 0s"

    def test_just_below_one_day(self):
        result = format_uptime(86399)
        assert result == "23h 59m 59s"

    def test_exactly_one_day(self):
        result = format_uptime(86400.0)
        assert result == "1d 0h 0m"


# ---------------------------------------------------------------------------
# format_uptime – edge cases
# ---------------------------------------------------------------------------


class TestFormatUptimeEdgeCases:
    """Edge cases for format_uptime."""

    def test_fractional_seconds_truncated(self):
        # 0.5 seconds → int(0.5 % 60) == 0
        result = format_uptime(0.5)
        assert result == "0s"

    def test_fractional_with_minutes(self):
        # 90.7 seconds → 1 minute, 30 seconds (int truncation)
        result = format_uptime(90.7)
        assert result == "1m 30s"

    def test_large_uptime(self):
        # 365 days
        seconds = 365 * 86400
        result = format_uptime(seconds)
        assert result == "365d 0h 0m"

    def test_very_large_uptime(self):
        # 1000 days, 23 hours, 59 minutes
        seconds = 1000 * 86400 + 23 * 3600 + 59 * 60
        result = format_uptime(seconds)
        assert result == "1000d 23h 59m"


# ---------------------------------------------------------------------------
# format_size – bytes range
# ---------------------------------------------------------------------------


class TestFormatSizeBytes:
    """Tests for sizes that stay in the byte range (< 1024)."""

    def test_zero_bytes(self):
        result = format_size(0)
        assert result == "0.0 B"

    def test_one_byte(self):
        result = format_size(1)
        assert result == "1.0 B"

    def test_512_bytes(self):
        result = format_size(512)
        assert result == "512.0 B"

    def test_1023_bytes(self):
        result = format_size(1023)
        assert result == "1023.0 B"


# ---------------------------------------------------------------------------
# format_size – KB range
# ---------------------------------------------------------------------------


class TestFormatSizeKB:
    """Tests for sizes in the kilobyte range."""

    def test_exactly_1_kb(self):
        result = format_size(1024)
        assert result == "1.0 KB"

    def test_1_5_kb(self):
        result = format_size(1536)
        assert result == "1.5 KB"

    def test_256_kb(self):
        result = format_size(256 * 1024)
        assert result == "256.0 KB"

    def test_just_below_1_mb(self):
        result = format_size(1024 * 1024 - 1)
        # 1048575 / 1024 = 1023.999... which is < 1024, stays in KB
        # but .1f rounds 1023.999 to 1024.0
        assert result == "1024.0 KB"


# ---------------------------------------------------------------------------
# format_size – MB range
# ---------------------------------------------------------------------------


class TestFormatSizeMB:
    """Tests for sizes in the megabyte range."""

    def test_exactly_1_mb(self):
        result = format_size(1024 * 1024)
        assert result == "1.0 MB"

    def test_1_5_mb(self):
        result = format_size(int(1.5 * 1024 * 1024))
        assert result == "1.5 MB"

    def test_500_mb(self):
        result = format_size(500 * 1024 * 1024)
        assert result == "500.0 MB"


# ---------------------------------------------------------------------------
# format_size – GB range
# ---------------------------------------------------------------------------


class TestFormatSizeGB:
    """Tests for sizes in the gigabyte range."""

    def test_exactly_1_gb(self):
        result = format_size(1024**3)
        assert result == "1.0 GB"

    def test_2_5_gb(self):
        result = format_size(int(2.5 * 1024**3))
        assert result == "2.5 GB"

    def test_512_gb(self):
        result = format_size(512 * 1024**3)
        assert result == "512.0 GB"


# ---------------------------------------------------------------------------
# format_size – TB range (fallback)
# ---------------------------------------------------------------------------


class TestFormatSizeTB:
    """Tests for sizes that exceed GB and fall into TB."""

    def test_exactly_1_tb(self):
        result = format_size(1024**4)
        assert result == "1.0 TB"

    def test_2_tb(self):
        result = format_size(2 * 1024**4)
        assert result == "2.0 TB"

    def test_large_tb_value(self):
        # 10 TB
        result = format_size(10 * 1024**4)
        assert result == "10.0 TB"

    def test_very_large_value(self):
        # 1 PB = 1024 TB, format_size doesn't have PB so it shows 1024.0 TB
        result = format_size(1024**5)
        assert result == "1024.0 TB"


# ---------------------------------------------------------------------------
# format_size – edge cases
# ---------------------------------------------------------------------------


class TestFormatSizeEdgeCases:
    """Edge and corner cases for format_size."""

    def test_exact_boundary_kb(self):
        # Exactly 1024 bytes should display as KB
        assert format_size(1024) == "1.0 KB"

    def test_exact_boundary_mb(self):
        assert format_size(1024**2) == "1.0 MB"

    def test_exact_boundary_gb(self):
        assert format_size(1024**3) == "1.0 GB"

    def test_exact_boundary_tb(self):
        assert format_size(1024**4) == "1.0 TB"

    def test_negative_is_handled_as_bytes(self):
        # Negative values: -1 < 1024 so stays in B loop
        result = format_size(-1)
        assert result == "-1.0 B"

    def test_decimal_precision(self):
        # 1536 bytes = 1.5 KB, verify single decimal
        result = format_size(1536)
        assert ".5" in result
        assert result.endswith(" KB")

    def test_rounding_behavior(self):
        # 1536 + 51 = 1587 bytes = 1.5498046875 KB ≈ 1.5 KB
        result = format_size(1587)
        assert result == "1.5 KB"
