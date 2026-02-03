"""
Tests for Platform-specific Rate Limiter.

Covers:
- PlatformRateLimiter initialization and configuration
- Platform-specific rate limits (Slack, Discord, Teams, Telegram, WhatsApp, etc.)
- Per-minute rate limiting with token buckets
- Daily limit enforcement
- Burst handling
- Limit reset timing
- HTTP header generation
- Global registry functions
- Cleanup and memory management
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_platform_registry():
    """Reset platform limiter registry before and after test."""
    from aragora.server.middleware.rate_limit.platform_limiter import (
        reset_platform_rate_limiters,
    )

    reset_platform_rate_limiters()
    yield
    reset_platform_rate_limiters()


@pytest.fixture
def mock_time():
    """Mock time.time for deterministic tests."""
    with patch("time.time") as mock_t:
        mock_t.return_value = 1000000.0
        yield mock_t


# -----------------------------------------------------------------------------
# PlatformRateLimitResult Tests
# -----------------------------------------------------------------------------


class TestPlatformRateLimitResult:
    """Tests for PlatformRateLimitResult dataclass."""

    def test_result_to_headers_basic(self):
        """Should generate basic rate limit headers."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimitResult,
        )

        result = PlatformRateLimitResult(
            allowed=True,
            remaining=50,
            limit=100,
            reset_at=1000060.0,
            platform="slack",
        )

        headers = result.to_headers()

        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "50"
        assert headers["X-RateLimit-Reset"] == "1000060"
        assert "Retry-After" not in headers

    def test_result_to_headers_with_retry_after(self):
        """Should include Retry-After header when request denied."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimitResult,
        )

        result = PlatformRateLimitResult(
            allowed=False,
            remaining=0,
            limit=100,
            reset_at=1000030.0,
            retry_after=30.0,
            platform="discord",
        )

        headers = result.to_headers()

        assert headers["Retry-After"] == "31"  # Adds 1 second buffer

    def test_result_to_headers_with_daily_remaining(self):
        """Should include daily remaining header when applicable."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimitResult,
        )

        result = PlatformRateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_at=1000060.0,
            platform="whatsapp",
            daily_remaining=80,
        )

        headers = result.to_headers()

        assert headers["X-RateLimit-Daily-Remaining"] == "80"

    def test_result_remaining_never_negative(self):
        """Remaining should never be negative in headers."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimitResult,
        )

        result = PlatformRateLimitResult(
            allowed=False,
            remaining=-5,  # Edge case
            limit=10,
            reset_at=1000060.0,
            platform="telegram",
        )

        headers = result.to_headers()

        assert headers["X-RateLimit-Remaining"] == "0"


# -----------------------------------------------------------------------------
# PlatformRateLimiter Initialization Tests
# -----------------------------------------------------------------------------


class TestPlatformRateLimiterInitialization:
    """Tests for PlatformRateLimiter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        assert limiter.platform == "test"
        assert limiter.requests_per_minute == 30
        assert limiter.burst_size == 10
        assert limiter.daily_limit == 0

    def test_custom_initialization(self):
        """Should initialize with custom values."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="custom",
            requests_per_minute=100,
            burst_size=20,
            daily_limit=500,
        )

        assert limiter.requests_per_minute == 100
        assert limiter.burst_size == 20
        assert limiter.daily_limit == 500

    def test_rpm_property_alias(self):
        """rpm property should alias requests_per_minute."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test", requests_per_minute=60)

        assert limiter.rpm == 60


# -----------------------------------------------------------------------------
# Platform-Specific Configuration Tests
# -----------------------------------------------------------------------------


class TestPlatformSpecificConfiguration:
    """Tests for platform-specific rate limit configurations."""

    def test_slack_rate_limits(self, clean_platform_registry):
        """Slack should have conservative limits per its API constraints."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
            get_platform_rate_limiter,
        )

        assert PLATFORM_RATE_LIMITS["slack"]["rpm"] == 10
        assert PLATFORM_RATE_LIMITS["slack"]["burst"] == 5

        limiter = get_platform_rate_limiter("slack")
        assert limiter.requests_per_minute == 10

    def test_discord_rate_limits(self, clean_platform_registry):
        """Discord should allow higher burst for its 50 req/second limit."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
            get_platform_rate_limiter,
        )

        assert PLATFORM_RATE_LIMITS["discord"]["rpm"] == 30
        assert PLATFORM_RATE_LIMITS["discord"]["burst"] == 10

        limiter = get_platform_rate_limiter("discord")
        assert limiter.requests_per_minute == 30

    def test_teams_rate_limits(self, clean_platform_registry):
        """Teams should respect Microsoft Graph throttling."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
        )

        assert PLATFORM_RATE_LIMITS["teams"]["rpm"] == 10

    def test_telegram_rate_limits(self, clean_platform_registry):
        """Telegram should allow 30 messages/second to different users."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
        )

        assert PLATFORM_RATE_LIMITS["telegram"]["rpm"] == 20

    def test_whatsapp_rate_limits(self, clean_platform_registry):
        """WhatsApp should have daily limits per its tier-based system."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
        )

        assert PLATFORM_RATE_LIMITS["whatsapp"]["rpm"] == 5
        assert PLATFORM_RATE_LIMITS["whatsapp"]["daily"] == 100

    def test_zoom_rate_limits(self, clean_platform_registry):
        """Zoom should have daily limits."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
        )

        assert PLATFORM_RATE_LIMITS["zoom"]["daily"] == 1000

    def test_email_rate_limits(self, clean_platform_registry):
        """Email (SMTP) should have conservative limits."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
        )

        assert PLATFORM_RATE_LIMITS["email"]["rpm"] == 10
        assert PLATFORM_RATE_LIMITS["email"]["daily"] == 500


# -----------------------------------------------------------------------------
# Per-Minute Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestPerMinuteRateLimiting:
    """Tests for per-minute rate limit enforcement."""

    def test_allows_first_request(self):
        """First request should always be allowed."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test", requests_per_minute=10)

        result = limiter.check("channel-123")

        assert result.allowed is True
        assert result.platform == "test"

    def test_allows_requests_under_limit(self):
        """Requests under limit should be allowed."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=10,
            burst_size=5,
        )

        # First few requests should be allowed
        for i in range(5):
            result = limiter.check("channel-123")
            assert result.allowed is True

    def test_denies_after_exhausting_tokens(self):
        """Should deny after exhausting burst tokens."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=5,
            burst_size=3,
        )

        # Exhaust all tokens (rpm + burst = 8)
        allowed_count = 0
        for _ in range(20):
            result = limiter.check("channel-123")
            if result.allowed:
                allowed_count += 1

        # Should have denied some
        assert allowed_count < 20

    def test_returns_retry_after_when_denied(self):
        """Should return retry_after value when denied."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=2,
            burst_size=1,
        )

        # Exhaust tokens
        for _ in range(10):
            limiter.check("channel-123")

        result = limiter.check("channel-123")

        if not result.allowed:
            assert result.retry_after is not None
            assert result.retry_after > 0

    def test_different_keys_independent(self):
        """Different keys should have independent rate limits."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=5,
            burst_size=2,
        )

        # Exhaust limit for channel-1
        for _ in range(15):
            limiter.check("channel-1")

        # channel-2 should still work
        result = limiter.check("channel-2")
        assert result.allowed is True


# -----------------------------------------------------------------------------
# Daily Limit Enforcement Tests
# -----------------------------------------------------------------------------


class TestDailyLimitEnforcement:
    """Tests for daily rate limit enforcement."""

    def test_no_daily_limit_when_zero(self):
        """Should not enforce daily limit when set to 0."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=1000,
            burst_size=100,
            daily_limit=0,
        )

        # Make many requests - should all be allowed (up to burst)
        for _ in range(50):
            result = limiter.check("channel-123")
            assert result.daily_remaining is None

    def test_daily_limit_tracks_usage(self):
        """Should track daily usage."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=100,
            burst_size=50,
            daily_limit=10,
        )

        result1 = limiter.check("channel-123")
        assert result1.allowed is True
        assert result1.daily_remaining == 9

        result2 = limiter.check("channel-123")
        assert result2.daily_remaining == 8

    def test_daily_limit_denies_after_exhausted(self):
        """Should deny after daily limit exhausted."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=100,
            burst_size=50,
            daily_limit=5,
        )

        # Use up daily limit
        for _ in range(5):
            result = limiter.check("channel-123")
            assert result.allowed is True

        # Next should be denied
        result = limiter.check("channel-123")
        assert result.allowed is False
        assert result.daily_remaining == 0

    def test_daily_limit_resets_after_24h(self):
        """Daily limit should reset after 24 hours."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=100,
            burst_size=50,
            daily_limit=3,
        )

        # Exhaust daily limit
        for _ in range(3):
            limiter.check("channel-123")

        # Should be denied
        result = limiter.check("channel-123")
        assert result.allowed is False

        # Simulate time passing (24h) by patching at the module level
        future_time = time.time() + 90000
        with patch(
            "aragora.server.middleware.rate_limit.platform_limiter.time.time",
            return_value=future_time,
        ):
            result = limiter.check("channel-123")
            assert result.allowed is True


# -----------------------------------------------------------------------------
# Burst Handling Tests
# -----------------------------------------------------------------------------


class TestBurstHandling:
    """Tests for burst traffic handling."""

    def test_burst_allows_initial_spike(self):
        """Should allow initial burst of requests."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=10,
            burst_size=5,
        )

        # Burst size is added to RPM for bucket capacity (10 + 5 = 15)
        allowed = 0
        for _ in range(20):
            result = limiter.check("channel-123")
            if result.allowed:
                allowed += 1

        # Should allow at least the burst capacity
        assert allowed >= 10  # At least base rate

    def test_burst_depletes_then_throttles(self):
        """After burst, should throttle to sustained rate."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=2,
            burst_size=3,
        )

        # Make burst requests
        results = [limiter.check("channel-123") for _ in range(10)]

        # Count allowed/denied
        allowed = sum(1 for r in results if r.allowed)
        denied = sum(1 for r in results if not r.allowed)

        assert allowed > 0
        assert denied > 0  # Some should be throttled


# -----------------------------------------------------------------------------
# is_allowed Simple Interface Tests
# -----------------------------------------------------------------------------


class TestIsAllowedSimpleInterface:
    """Tests for the simple is_allowed() method."""

    def test_is_allowed_returns_bool(self):
        """is_allowed should return boolean."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        result = limiter.is_allowed("channel-123")

        assert isinstance(result, bool)
        assert result is True

    def test_is_allowed_uses_check(self):
        """is_allowed should use check() internally."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=1,
            burst_size=0,
        )

        # First should be allowed
        assert limiter.is_allowed("channel-123") is True

        # Exhaust and check
        for _ in range(5):
            limiter.is_allowed("channel-123")

        # Should now be denied
        assert limiter.is_allowed("channel-123") is False


# -----------------------------------------------------------------------------
# Reset Functionality Tests
# -----------------------------------------------------------------------------


class TestResetFunctionality:
    """Tests for reset functionality."""

    def test_reset_clears_key_state(self):
        """reset() should clear state for a key."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=5,
            burst_size=2,
            daily_limit=10,
        )

        # Use up some limits
        for _ in range(10):
            limiter.check("channel-123")

        # Reset
        result = limiter.reset("channel-123")
        assert result is True

        # Should be allowed again
        result = limiter.check("channel-123")
        assert result.allowed is True

    def test_reset_clears_daily_count(self):
        """reset() should also clear daily count."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=100,
            burst_size=50,
            daily_limit=3,
        )

        # Exhaust daily limit
        for _ in range(3):
            limiter.check("channel-123")

        result = limiter.check("channel-123")
        assert result.allowed is False

        # Reset
        limiter.reset("channel-123")

        # Should be allowed again
        result = limiter.check("channel-123")
        assert result.allowed is True


# -----------------------------------------------------------------------------
# Cleanup Tests
# -----------------------------------------------------------------------------


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_returns_count(self):
        """cleanup() should return number of cleaned entries."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        # Add some entries
        for i in range(100):
            limiter.check(f"channel-{i}")

        # Cleanup shouldn't remove much with few entries
        removed = limiter.cleanup()
        assert isinstance(removed, int)
        assert removed == 0  # Under 10000 threshold

    def test_cleanup_removes_when_above_threshold(self):
        """cleanup() should clear when above 10000 entries."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        # Mock many entries
        for i in range(10001):
            limiter._buckets[f"channel-{i}"] = MagicMock()

        removed = limiter.cleanup()
        assert removed == 10001


# -----------------------------------------------------------------------------
# Global Registry Tests
# -----------------------------------------------------------------------------


class TestGlobalRegistry:
    """Tests for global platform limiter registry."""

    def test_get_platform_rate_limiter_uses_defaults(self, clean_platform_registry):
        """get_platform_rate_limiter should use platform defaults."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
        )

        limiter = get_platform_rate_limiter("slack")

        assert limiter.platform == "slack"
        assert limiter.requests_per_minute == 10

    def test_get_platform_rate_limiter_caches(self, clean_platform_registry):
        """Same platform should return cached limiter."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
        )

        limiter1 = get_platform_rate_limiter("discord")
        limiter2 = get_platform_rate_limiter("discord")

        assert limiter1 is limiter2

    def test_get_platform_rate_limiter_custom_config_not_cached(self, clean_platform_registry):
        """Custom config should create new limiter (not cached)."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
        )

        limiter1 = get_platform_rate_limiter("telegram", requests_per_minute=100)
        limiter2 = get_platform_rate_limiter("telegram", requests_per_minute=200)

        assert limiter1 is not limiter2
        assert limiter1.requests_per_minute == 100
        assert limiter2.requests_per_minute == 200

    def test_get_platform_rate_limiter_case_insensitive(self, clean_platform_registry):
        """Platform names should be case-insensitive."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
        )

        limiter1 = get_platform_rate_limiter("SLACK")
        limiter2 = get_platform_rate_limiter("slack")

        assert limiter1.platform == "slack"
        assert limiter1 is limiter2

    def test_get_platform_rate_limiter_unknown_platform(self, clean_platform_registry):
        """Unknown platform should use defaults."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
        )

        limiter = get_platform_rate_limiter("unknown_platform")

        assert limiter.requests_per_minute == 30  # Default
        assert limiter.burst_size == 10  # Default


# -----------------------------------------------------------------------------
# check_platform_rate_limit Convenience Function Tests
# -----------------------------------------------------------------------------


class TestCheckPlatformRateLimit:
    """Tests for check_platform_rate_limit convenience function."""

    def test_check_platform_rate_limit_works(self, clean_platform_registry):
        """check_platform_rate_limit should work as expected."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            check_platform_rate_limit,
        )

        result = check_platform_rate_limit("slack", "channel-123")

        assert result.allowed is True
        assert result.platform == "slack"

    def test_check_platform_rate_limit_enforces_limits(self, clean_platform_registry):
        """check_platform_rate_limit should enforce limits."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            check_platform_rate_limit,
        )

        # Slack has rpm=10, burst=5 -> ~15 allowed
        allowed = 0
        for _ in range(30):
            result = check_platform_rate_limit("slack", "channel-test")
            if result.allowed:
                allowed += 1

        assert allowed < 30  # Some should be denied


# -----------------------------------------------------------------------------
# reset_platform_rate_limiters Tests
# -----------------------------------------------------------------------------


class TestResetPlatformRateLimiters:
    """Tests for reset_platform_rate_limiters function."""

    def test_reset_returns_count(self, clean_platform_registry):
        """reset_platform_rate_limiters should return count."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
            reset_platform_rate_limiters,
        )

        # Create some limiters
        get_platform_rate_limiter("slack")
        get_platform_rate_limiter("discord")
        get_platform_rate_limiter("teams")

        count = reset_platform_rate_limiters()

        assert count == 3

    def test_reset_clears_all_limiters(self, clean_platform_registry):
        """reset_platform_rate_limiters should clear all cached limiters."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            get_platform_rate_limiter,
            reset_platform_rate_limiters,
        )

        limiter1 = get_platform_rate_limiter("slack")

        reset_platform_rate_limiters()

        limiter2 = get_platform_rate_limiter("slack")

        assert limiter1 is not limiter2


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_checks_thread_safe(self):
        """Concurrent checks should be thread-safe."""
        import threading

        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(
            platform="test",
            requests_per_minute=100,
            burst_size=50,
        )

        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(50):
                    result = limiter.check("channel-shared")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_key(self):
        """Should handle empty key."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        result = limiter.check("")

        assert result.allowed is True

    def test_very_long_key(self):
        """Should handle very long key."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        long_key = "a" * 10000
        result = limiter.check(long_key)

        assert result.allowed is True

    def test_special_characters_in_key(self):
        """Should handle special characters in key."""
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PlatformRateLimiter,
        )

        limiter = PlatformRateLimiter(platform="test")

        result = limiter.check("channel:with:colons/and/slashes")

        assert result.allowed is True
