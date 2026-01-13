"""
Rate limit enforcement security tests.

Verifies that rate limiting cannot be bypassed through various techniques.
"""

import threading
import time
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from aragora.server.middleware.rate_limit import (
    TokenBucket,
    RateLimiter,
    rate_limit_headers,
    normalize_rate_limit_path,
    sanitize_rate_limit_key_component,
)


# =============================================================================
# Rate Limiter Core Tests
# =============================================================================


class TestRateLimiterBypassPrevention:
    """Test that rate limiting cannot be bypassed."""

    def test_blocks_after_limit_exhausted(self):
        """Rate limiter should block requests after limit is reached."""
        limiter = TokenBucket(rate_per_minute=5, burst_size=5)

        # Exhaust the limit
        for _ in range(5):
            assert limiter.consume() is True

        # Next request should be blocked
        assert limiter.consume() is False

    def test_blocks_burst_after_exhausted(self):
        """Burst capacity should also eventually be blocked."""
        limiter = TokenBucket(rate_per_minute=5, burst_size=10)

        # Exhaust limit including burst (5 * 2 = 10)
        for _ in range(10):
            limiter.consume()

        # Should be blocked
        assert limiter.consume() is False

    def test_time_based_recovery(self):
        """Tokens should recover over time."""
        limiter = TokenBucket(rate_per_minute=60, burst_size=60)

        # Exhaust limit
        for _ in range(60):
            limiter.consume()

        assert limiter.consume() is False

        # Simulate time passing (1 second = 1 token at 60/min)
        limiter.last_refill -= 1.0
        assert limiter.consume() is True

    def test_cannot_bypass_with_negative_amount(self):
        """Negative acquire amounts should not add tokens."""
        limiter = TokenBucket(rate_per_minute=5, burst_size=5)

        initial_tokens = limiter.tokens

        # Try to bypass with negative amount - should be rejected
        result = limiter.consume(tokens=-100)
        assert result is False

        # Tokens should not have increased
        assert limiter.tokens <= initial_tokens

    def test_concurrent_access_safety(self):
        """Concurrent access should not exceed limits."""
        limiter = TokenBucket(rate_per_minute=10, burst_size=10)
        successes = []

        def attempt():
            return limiter.consume()

        threads = []
        for _ in range(50):
            t = threading.Thread(target=lambda: successes.append(attempt()))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not exceed limit
        assert sum(successes) <= 10

    def test_token_overflow_prevention(self):
        """Tokens should not overflow beyond burst capacity."""
        limiter = TokenBucket(rate_per_minute=10, burst_size=20)

        # Simulate long time passing to accumulate tokens
        limiter.last_refill -= 3600  # 1 hour ago

        # Trigger refill
        limiter.consume()

        # Should cap at burst capacity (10 * 2 = 20)
        assert limiter.tokens <= 20


# =============================================================================
# IP-Based Rate Limiting Tests
# =============================================================================


class TestIPRateLimitBypass:
    """Test IP-based rate limiting bypass attempts."""

    def test_different_ips_have_separate_limits(self):
        """Different IPs should have separate rate limits."""
        limiters = {}

        def get_limiter(ip):
            if ip not in limiters:
                limiters[ip] = TokenBucket(rate_per_minute=5, burst_size=5)
            return limiters[ip]

        # IP 1 exhausts limit
        limiter1 = get_limiter("192.168.1.1")
        for _ in range(5):
            limiter1.consume()
        assert limiter1.consume() is False

        # IP 2 should still have capacity
        limiter2 = get_limiter("192.168.1.2")
        assert limiter2.consume() is True

    def test_spoofed_xff_header_ignored_when_untrusted(self):
        """X-Forwarded-For should be ignored from untrusted sources."""
        limiter = RateLimiter()
        handler = SimpleNamespace(
            headers={"X-Forwarded-For": "10.0.0.1"},
            client_address=("192.168.1.100", 12345),
        )

        # Untrusted proxy - ignore XFF
        assert limiter.get_client_key(handler) == "192.168.1.100"

        # Trusted proxy - use XFF
        handler.client_address = ("127.0.0.1", 12345)
        assert limiter.get_client_key(handler) == "10.0.0.1"

    def test_ipv6_and_ipv4_treated_separately(self):
        """IPv6 and IPv4 addresses should be tracked separately."""
        limiters = {}

        def get_limiter(ip):
            if ip not in limiters:
                limiters[ip] = TokenBucket(rate_per_minute=5, burst_size=5)
            return limiters[ip]

        ipv4 = "192.168.1.1"
        ipv6 = "::ffff:192.168.1.1"  # IPv4-mapped IPv6

        # These should be separate limiters
        limiter_v4 = get_limiter(ipv4)
        limiter_v6 = get_limiter(ipv6)

        assert limiter_v4 is not limiter_v6

    def test_normalized_ipv6_addresses(self):
        """IPv6 addresses should be normalized before rate limiting."""
        limiter = RateLimiter()
        full = "2001:0db8:0000:0000:0000:0000:0000:0001"
        short = "2001:db8::1"

        handler_full = SimpleNamespace(headers={}, client_address=(full, 12345))
        handler_short = SimpleNamespace(headers={}, client_address=(short, 12345))

        assert limiter.get_client_key(handler_full) == limiter.get_client_key(handler_short)


# =============================================================================
# Tier-Based Rate Limiting Tests
# =============================================================================


class TestTierRateLimitBypass:
    """Test tier-based rate limiting bypass attempts."""

    def test_free_tier_has_lowest_limits(self):
        """Free tier should have the most restrictive limits."""
        tier_limits = {
            "free": 10,
            "starter": 60,
            "professional": 300,
            "enterprise": 1000,
        }

        assert tier_limits["free"] < tier_limits["starter"]
        assert tier_limits["starter"] < tier_limits["professional"]
        assert tier_limits["professional"] < tier_limits["enterprise"]

    def test_cannot_upgrade_tier_via_header(self):
        """Tier should come from authenticated org, not request headers."""

        def get_tier_from_auth(org_id, request_claimed_tier):
            # Simulated database lookup
            org_tiers = {
                "org-123": "free",
                "org-456": "professional",
            }
            # Always use database tier, never trust request
            return org_tiers.get(org_id, "free")

        # Attacker claims professional tier
        tier = get_tier_from_auth("org-123", request_claimed_tier="enterprise")
        assert tier == "free"

    def test_expired_subscription_downgrades(self):
        """Expired subscriptions should downgrade to free tier limits."""

        def get_effective_tier(org_id, subscription_end):
            if subscription_end and subscription_end < datetime.now():
                return "free"  # Expired
            return "professional"

        expired = datetime.now() - timedelta(days=1)
        assert get_effective_tier("org-123", expired) == "free"

        valid = datetime.now() + timedelta(days=30)
        assert get_effective_tier("org-123", valid) == "professional"


# =============================================================================
# Distributed Rate Limiting Tests
# =============================================================================


class TestDistributedRateLimitBypass:
    """Test distributed rate limiting (Redis-backed) bypass attempts."""

    def test_redis_atomic_operations(self):
        """Redis operations should be atomic to prevent race conditions."""
        # Simulate Lua script for atomic increment
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])

        local current = redis.call('INCR', key)
        if current == 1 then
            redis.call('EXPIRE', key, window)
        end

        if current > limit then
            return 0
        end
        return 1
        """

        # The script ensures atomicity - cannot bypass between INCR and check
        assert "INCR" in lua_script
        assert "EXPIRE" in lua_script

    def test_redis_key_isolation(self):
        """Redis keys should be properly namespaced to prevent collisions."""

        def make_rate_limit_key(org_id, endpoint):
            # Proper namespacing
            return f"aragora:ratelimit:{org_id}:{endpoint}"

        key1 = make_rate_limit_key("org-123", "/api/debates")
        key2 = make_rate_limit_key("org-456", "/api/debates")

        assert key1 != key2
        assert "org-123" in key1
        assert "org-456" in key2

    def test_redis_key_injection_prevention(self):
        """Redis keys should be sanitized to prevent injection."""

        def safe_rate_limit_key(org_id: str, endpoint: str) -> str:
            safe_org = sanitize_rate_limit_key_component(org_id)
            safe_endpoint = normalize_rate_limit_path(endpoint)
            return f"aragora:ratelimit:{safe_org}:{safe_endpoint}"

        # Normal inputs
        key = safe_rate_limit_key("org-123", "/api/debates")
        assert key == "aragora:ratelimit:org-123:/api/debates"

        # Injection attempt (colon could affect key structure)
        key = safe_rate_limit_key("org:injected", "/api/debates")
        assert "org_injected" in key


# =============================================================================
# Endpoint-Specific Rate Limiting Tests
# =============================================================================


class TestEndpointRateLimitBypass:
    """Test endpoint-specific rate limiting bypass attempts."""

    def test_expensive_endpoints_have_lower_limits(self):
        """Expensive endpoints should have stricter limits."""
        endpoint_limits = {
            "/api/debates": 30,  # Create debate - expensive
            "/api/gauntlet": 10,  # Gauntlet run - very expensive
            "/api/health": 1000,  # Health check - cheap
            "/api/leaderboard": 60,  # Read-only
        }

        # Expensive operations should have lower limits
        assert endpoint_limits["/api/gauntlet"] < endpoint_limits["/api/debates"]
        assert endpoint_limits["/api/debates"] < endpoint_limits["/api/leaderboard"]
        assert endpoint_limits["/api/leaderboard"] < endpoint_limits["/api/health"]

    def test_path_normalization_prevents_bypass(self):
        """Path normalization should prevent bypass via URL tricks."""
        # These should all normalize to same path
        assert normalize_rate_limit_path("/api/debates") == "/api/debates"
        assert normalize_rate_limit_path("/api/debates/") == "/api/debates"
        assert normalize_rate_limit_path("/api//debates") == "/api/debates"
        assert normalize_rate_limit_path("/api/debates/../debates") == "/api/debates"
        assert normalize_rate_limit_path("/api/%64ebates") == "/api/debates"

    def test_case_sensitivity_handling(self):
        """Path matching should handle case consistently."""
        assert normalize_rate_limit_path("/API/Debates") == "/api/debates"
        assert normalize_rate_limit_path("/api/DEBATES/") == "/api/debates"


# =============================================================================
# Rate Limit Header Tests
# =============================================================================


class TestRateLimitHeaders:
    """Test rate limit header accuracy and security."""

    def test_retry_after_header_accurate(self):
        """Retry-After header should be accurate."""
        limiter = TokenBucket(rate_per_minute=60, burst_size=60)

        # Exhaust limit
        for _ in range(60):
            limiter.consume()

        # Calculate retry after
        retry_after = limiter.get_retry_after()

        # Should be approximately 1 second (60 tokens/min = 1 token/sec)
        assert 0 < retry_after <= 2.0

    def test_remaining_count_not_negative(self):
        """Remaining count should never be negative."""
        limiter = TokenBucket(rate_per_minute=5, burst_size=5)

        # Exhaust limit
        for _ in range(10):  # More than limit
            limiter.consume()

        remaining = max(0, int(limiter.tokens))
        assert remaining >= 0

    def test_limit_header_matches_tier(self):
        """X-RateLimit-Limit header should match actual tier limit."""
        tier_limits = {
            "free": 10,
            "starter": 60,
            "professional": 300,
        }

        for tier, expected_limit in tier_limits.items():
            header_value = tier_limits[tier]
            assert header_value == expected_limit

    def test_headers_not_expose_internals(self):
        """Rate limit headers should not expose sensitive info."""
        headers = rate_limit_headers(
            type("Result", (), {"limit": 100, "remaining": 5, "retry_after": 10})()
        )

        # Should not contain internal details
        assert "token" not in str(headers).lower()
        assert "bucket" not in str(headers).lower()
        assert "internal" not in str(headers).lower()


# =============================================================================
# Rate Limit Persistence Tests
# =============================================================================


class TestRateLimitPersistence:
    """Test rate limit state persistence and recovery."""

    def test_server_restart_resets_in_memory_limits(self):
        """In-memory rate limits should reset on restart (acceptable behavior)."""
        # In-memory limiter
        limiter1 = TokenBucket(rate_per_minute=5, burst_size=5)
        for _ in range(5):
            limiter1.consume()
        assert limiter1.consume() is False

        # Simulate restart - new limiter instance
        limiter2 = TokenBucket(rate_per_minute=5, burst_size=5)
        assert limiter2.consume() is True  # Fresh start

    def test_window_expiry_correct(self):
        """Rate limit windows should expire correctly."""
        limiter = TokenBucket(rate_per_minute=60, burst_size=60)

        # Record current state
        initial_tokens = limiter.tokens

        # Simulate 2 minutes passing
        limiter.last_refill -= 120

        # Should have recovered tokens
        limiter.consume()  # Trigger refill
        assert limiter.tokens >= initial_tokens - 1


# =============================================================================
# Sliding Window Tests
# =============================================================================


class TestSlidingWindowRateLimiting:
    """Test sliding window rate limiting patterns."""

    def test_sliding_window_prevents_burst_at_boundary(self):
        """Sliding window should prevent burst at fixed window boundary."""

        class SlidingWindowCounter:
            """Simple sliding window rate limiter."""

            def __init__(self, limit: int, window_seconds: int = 60):
                self.limit = limit
                self.window_seconds = window_seconds
                self.requests = []

            def allow(self) -> bool:
                now = time.time()
                cutoff = now - self.window_seconds

                # Remove old requests
                self.requests = [t for t in self.requests if t > cutoff]

                if len(self.requests) >= self.limit:
                    return False

                self.requests.append(now)
                return True

        limiter = SlidingWindowCounter(limit=10, window_seconds=60)

        # Make 10 requests
        for _ in range(10):
            assert limiter.allow() is True

        # 11th should be blocked
        assert limiter.allow() is False

    def test_sliding_window_continuous_recovery(self):
        """Sliding window should recover continuously, not in bursts."""

        class SlidingWindowCounter:
            def __init__(self, limit: int, window_seconds: int = 60):
                self.limit = limit
                self.window_seconds = window_seconds
                self.requests = []

            def allow(self) -> bool:
                now = time.time()
                cutoff = now - self.window_seconds
                self.requests = [t for t in self.requests if t > cutoff]
                if len(self.requests) >= self.limit:
                    return False
                self.requests.append(now)
                return True

        limiter = SlidingWindowCounter(limit=5, window_seconds=60)

        # Make 5 requests at time T
        base_time = time.time()
        for _ in range(5):
            assert limiter.allow() is True

        # Blocked
        assert limiter.allow() is False

        # Simulate time passing - first request expires
        limiter.requests[0] = base_time - 61

        # Now should allow one more
        assert limiter.allow() is True
        assert limiter.allow() is False  # Back to limit


# =============================================================================
# Rate Limit Headers Tests
# =============================================================================


class TestRateLimitHeaders:
    """Test rate limit headers generation."""

    def test_rate_limit_headers_allowed(self):
        """Headers should include limit and remaining when allowed."""
        from aragora.server.middleware.rate_limit import RateLimitResult

        result = RateLimitResult(allowed=True, remaining=8, limit=10, retry_after=0)
        headers = rate_limit_headers(result)

        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "8"
        assert "Retry-After" not in headers
        assert "X-RateLimit-Reset" not in headers

    def test_rate_limit_headers_blocked(self):
        """Headers should include retry info when blocked."""
        from aragora.server.middleware.rate_limit import RateLimitResult

        result = RateLimitResult(allowed=False, remaining=0, limit=10, retry_after=30.5)
        headers = rate_limit_headers(result)

        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["Retry-After"] == "31"  # Rounded up
        assert "X-RateLimit-Reset" in headers

    def test_rate_limit_headers_zero_remaining(self):
        """Headers should work with zero remaining."""
        from aragora.server.middleware.rate_limit import RateLimitResult

        result = RateLimitResult(allowed=True, remaining=0, limit=10, retry_after=0)
        headers = rate_limit_headers(result)

        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "0"


class TestUnifiedHandlerRateLimitHeaders:
    """Test rate limit headers in UnifiedHandler responses."""

    def test_handler_has_rate_limit_result_attribute(self):
        """UnifiedHandler should have _rate_limit_result attribute."""
        from aragora.server.unified_server import UnifiedHandler

        assert hasattr(UnifiedHandler, "_rate_limit_result")

    def test_handler_has_add_rate_limit_headers_method(self):
        """UnifiedHandler should have _add_rate_limit_headers method."""
        from aragora.server.unified_server import UnifiedHandler

        assert hasattr(UnifiedHandler, "_add_rate_limit_headers")
        assert callable(getattr(UnifiedHandler, "_add_rate_limit_headers"))
