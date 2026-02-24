"""
Tests for the per-endpoint rate limiter.

Covers tier classification, token bucket mechanics, overflow,
cleanup, path overrides, and response header generation.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.server.security.endpoint_rate_limiter import (
    AUTH_PATH_PREFIXES,
    DEFAULT_TIER_RATES,
    EndpointRateLimitConfig,
    EndpointRateLimiter,
    RateLimitCheckResult,
    RateTier,
    _TokenBucket,
)


# ============================================================================
# Tier classification
# ============================================================================


class TestTierClassification:
    """Verify that paths and methods map to the correct tier."""

    def test_auth_login(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/auth/login", "POST") == RateTier.AUTH

    def test_auth_register(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/register", "POST") == RateTier.AUTH

    def test_auth_oauth(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/oauth/token", "POST") == RateTier.AUTH

    def test_auth_mfa(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/mfa/verify", "POST") == RateTier.AUTH

    def test_write_post(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/debates", "POST") == RateTier.WRITE

    def test_write_put(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/debates/123", "PUT") == RateTier.WRITE

    def test_write_patch(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/debates/123", "PATCH") == RateTier.WRITE

    def test_write_delete(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/debates/123", "DELETE") == RateTier.WRITE

    def test_read_get(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/debates", "GET") == RateTier.READ

    def test_read_head(self) -> None:
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/health", "HEAD") == RateTier.READ

    def test_auth_takes_priority_over_write(self) -> None:
        """Auth paths should classify as AUTH even for POST."""
        limiter = EndpointRateLimiter()
        assert limiter.classify("/api/v1/auth/login", "POST") == RateTier.AUTH


# ============================================================================
# Token bucket internals
# ============================================================================


class TestTokenBucket:
    """Low-level token bucket behaviour."""

    def test_initial_burst(self) -> None:
        bucket = _TokenBucket(rate_per_minute=10, burst_size=10)
        for _ in range(10):
            assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill_over_time(self) -> None:
        bucket = _TokenBucket(rate_per_minute=60, burst_size=1)
        assert bucket.consume() is True
        assert bucket.consume() is False

        # Simulate 1 second passing (rate is 1 per second)
        future = time.monotonic() + 1.1
        assert bucket.consume(now=future) is True

    def test_burst_size_caps_tokens(self) -> None:
        bucket = _TokenBucket(rate_per_minute=600, burst_size=5)
        # Even after a long time, tokens should not exceed burst_size
        far_future = time.monotonic() + 3600
        bucket.consume(now=far_future)
        assert bucket.tokens <= 5

    def test_retry_after_positive_when_empty(self) -> None:
        bucket = _TokenBucket(rate_per_minute=60, burst_size=1)
        bucket.consume()  # exhaust
        assert bucket.retry_after() > 0

    def test_retry_after_zero_when_available(self) -> None:
        bucket = _TokenBucket(rate_per_minute=60, burst_size=10)
        assert bucket.retry_after() == 0.0


# ============================================================================
# EndpointRateLimiter core
# ============================================================================


class TestEndpointRateLimiterCore:
    """Integration tests for the limiter."""

    def test_allows_within_limit(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.READ: 10, RateTier.WRITE: 5, RateTier.AUTH: 3},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)

        for _ in range(10):
            result = limiter.check("client-1", "/api/v1/data", "GET")
            assert result.allowed is True

    def test_blocks_after_limit(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.AUTH: 3, RateTier.WRITE: 5, RateTier.READ: 10},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)

        for _ in range(3):
            result = limiter.check("client-1", "/api/v1/auth/login", "POST")
            assert result.allowed is True

        result = limiter.check("client-1", "/api/v1/auth/login", "POST")
        assert result.allowed is False

    def test_different_clients_independent(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.AUTH: 2, RateTier.WRITE: 5, RateTier.READ: 10},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)

        # Exhaust client-1
        for _ in range(2):
            limiter.check("client-1", "/api/v1/auth/login", "POST")
        assert limiter.check("client-1", "/api/v1/auth/login", "POST").allowed is False

        # client-2 should still be allowed
        assert limiter.check("client-2", "/api/v1/auth/login", "POST").allowed is True

    def test_different_tiers_independent(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.AUTH: 2, RateTier.WRITE: 5, RateTier.READ: 10},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)

        # Exhaust auth
        for _ in range(2):
            limiter.check("client-1", "/api/v1/auth/login", "POST")
        assert limiter.check("client-1", "/api/v1/auth/login", "POST").allowed is False

        # Write tier for same client should still work
        assert limiter.check("client-1", "/api/v1/debates", "POST").allowed is True

    def test_result_has_correct_tier(self) -> None:
        limiter = EndpointRateLimiter()
        result = limiter.check("c1", "/api/v1/auth/login", "POST")
        assert result.tier == RateTier.AUTH

    def test_result_has_retry_after_on_block(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.AUTH: 1, RateTier.WRITE: 5, RateTier.READ: 10},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)
        limiter.check("c1", "/api/v1/auth/login", "POST")  # exhaust
        result = limiter.check("c1", "/api/v1/auth/login", "POST")
        assert result.allowed is False
        assert result.retry_after > 0


# ============================================================================
# Path overrides
# ============================================================================


class TestPathOverrides:
    """Test per-path rate limit overrides."""

    def test_path_override_applied(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.READ: 100, RateTier.WRITE: 50, RateTier.AUTH: 5},
            burst_multiplier=1.0,
            path_overrides={"/api/v1/expensive": 2},
        )
        limiter = EndpointRateLimiter(config)

        # First 2 allowed
        assert limiter.check("c1", "/api/v1/expensive", "GET").allowed is True
        assert limiter.check("c1", "/api/v1/expensive", "GET").allowed is True
        # Third blocked
        assert limiter.check("c1", "/api/v1/expensive", "GET").allowed is False


# ============================================================================
# Response headers
# ============================================================================


class TestRateLimitResponseHeaders:
    """Test the headers generated for rate limit results."""

    def test_allowed_result_headers(self) -> None:
        result = RateLimitCheckResult(allowed=True, tier=RateTier.READ, limit=120, remaining=119)
        headers = result.headers()
        assert headers["X-RateLimit-Limit"] == "120"
        assert headers["X-RateLimit-Remaining"] == "119"
        assert "Retry-After" not in headers

    def test_blocked_result_includes_retry_after(self) -> None:
        result = RateLimitCheckResult(
            allowed=False, tier=RateTier.AUTH, retry_after=5.3, limit=5, remaining=0
        )
        headers = result.headers()
        assert headers["X-RateLimit-Limit"] == "5"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in headers
        assert int(headers["Retry-After"]) >= 6  # ceil(5.3) + 1

    def test_remaining_never_negative(self) -> None:
        result = RateLimitCheckResult(allowed=False, tier=RateTier.WRITE, remaining=-1, limit=30)
        headers = result.headers()
        assert headers["X-RateLimit-Remaining"] == "0"


# ============================================================================
# Reset and cleanup
# ============================================================================


class TestResetAndCleanup:
    """Test bucket reset and stale-bucket cleanup."""

    def test_reset_all(self) -> None:
        limiter = EndpointRateLimiter()
        limiter.check("c1", "/api/v1/data", "GET")
        limiter.check("c2", "/api/v1/data", "GET")
        assert limiter.bucket_count > 0

        limiter.reset()
        assert limiter.bucket_count == 0

    def test_reset_single_client(self) -> None:
        config = EndpointRateLimitConfig(
            tier_rates={RateTier.AUTH: 1, RateTier.WRITE: 1, RateTier.READ: 1},
            burst_multiplier=1.0,
        )
        limiter = EndpointRateLimiter(config)

        limiter.check("c1", "/api/v1/auth/login", "POST")
        limiter.check("c2", "/api/v1/auth/login", "POST")

        # c1 exhausted
        assert limiter.check("c1", "/api/v1/auth/login", "POST").allowed is False

        limiter.reset("c1")
        # c1 should be allowed again (bucket recreated)
        assert limiter.check("c1", "/api/v1/auth/login", "POST").allowed is True
        # c2 should still be blocked
        assert limiter.check("c2", "/api/v1/auth/login", "POST").allowed is False

    def test_stale_cleanup_runs(self) -> None:
        config = EndpointRateLimitConfig(
            cleanup_interval=0,  # always eligible
            stale_threshold=0,  # everything is stale
        )
        limiter = EndpointRateLimiter(config)

        limiter.check("c1", "/api/v1/data", "GET")
        assert limiter.bucket_count >= 1

        # Next check triggers cleanup (interval=0 means always eligible)
        # Set last_refill in the past to make them stale
        for b in limiter._buckets.values():
            b.last_refill = time.monotonic() - 1

        limiter.check("c2", "/api/v1/data", "GET")
        # c1's bucket should have been cleaned up
        # (c2's was just created so it won't be stale)
        assert "c1:read" not in limiter._buckets


# ============================================================================
# Default tier rates
# ============================================================================


class TestDefaultTierRates:
    """Sanity checks for the default tier configuration."""

    def test_auth_rate(self) -> None:
        assert DEFAULT_TIER_RATES[RateTier.AUTH] == 5

    def test_write_rate(self) -> None:
        assert DEFAULT_TIER_RATES[RateTier.WRITE] == 30

    def test_read_rate(self) -> None:
        assert DEFAULT_TIER_RATES[RateTier.READ] == 120
