"""
Tests for server middleware package.

Tests authentication, rate limiting, and caching middleware decorators.
"""

import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


class TestAuthMiddleware:
    """Tests for authentication middleware."""

    def test_auth_context_defaults(self):
        """AuthContext should have sensible defaults."""
        from aragora.server.middleware import AuthContext

        ctx = AuthContext()
        assert ctx.authenticated is False
        assert ctx.token is None
        assert ctx.client_ip is None
        assert ctx.user_id is None
        assert ctx.is_authenticated is False

    def test_auth_context_authenticated(self):
        """AuthContext should reflect authenticated state."""
        from aragora.server.middleware import AuthContext

        ctx = AuthContext(
            authenticated=True,
            token="test-token-123",
            client_ip="192.168.1.1",
            user_id="user-456",
        )
        assert ctx.authenticated is True
        assert ctx.is_authenticated is True
        assert ctx.token == "test-token-123"
        assert ctx.client_ip == "192.168.1.1"
        assert ctx.user_id == "user-456"

    def test_extract_token_bearer(self):
        """extract_token should extract Bearer token from headers."""
        from aragora.server.middleware import extract_token

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer my-secret-token"}

        token = extract_token(handler)
        assert token == "my-secret-token"

    def test_extract_token_no_bearer_prefix(self):
        """extract_token should return None if not Bearer token."""
        from aragora.server.middleware import extract_token

        handler = MagicMock()
        handler.headers = {"Authorization": "Basic abc123"}

        token = extract_token(handler)
        assert token is None

    def test_extract_token_no_header(self):
        """extract_token should return None if no Authorization header."""
        from aragora.server.middleware import extract_token

        handler = MagicMock()
        handler.headers = {}

        token = extract_token(handler)
        assert token is None

    def test_extract_token_none_handler(self):
        """extract_token should handle None handler."""
        from aragora.server.middleware import extract_token

        token = extract_token(None)
        assert token is None

    def test_extract_client_ip_direct(self):
        """extract_client_ip should get IP from client_address."""
        from aragora.server.middleware import extract_client_ip

        handler = MagicMock()
        handler.headers = {}
        handler.client_address = ("192.168.1.100", 12345)

        ip = extract_client_ip(handler)
        assert ip == "192.168.1.100"

    def test_extract_client_ip_forwarded(self):
        """extract_client_ip should prefer X-Forwarded-For."""
        from aragora.server.middleware import extract_client_ip

        handler = MagicMock()
        handler.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        handler.client_address = ("127.0.0.1", 12345)

        ip = extract_client_ip(handler)
        assert ip == "10.0.0.1"  # First IP in chain

    def test_extract_client_ip_none_handler(self):
        """extract_client_ip should handle None handler."""
        from aragora.server.middleware import extract_client_ip

        ip = extract_client_ip(None)
        assert ip is None

    def test_require_auth_no_handler(self):
        """require_auth should deny access if no handler provided."""
        from aragora.server.middleware import require_auth

        @require_auth
        def protected_func():
            return "success"

        result = protected_func()
        assert result.status_code == 401

    def test_require_auth_no_token_configured(self):
        """require_auth should deny if no API token is configured."""
        from aragora.server.middleware import require_auth

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer test"}

        @require_auth
        def protected_func(handler=None):
            return "success"

        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.api_token = None  # No token configured

            result = protected_func(handler=handler)
            assert result.status_code == 401

    def test_require_auth_invalid_token(self):
        """require_auth should deny if token is invalid."""
        from aragora.server.middleware import require_auth

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer wrong-token"}

        @require_auth
        def protected_func(handler=None):
            return "success"

        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.api_token = "correct-token"
            mock_config.validate_token.return_value = False

            result = protected_func(handler=handler)
            assert result.status_code == 401

    def test_require_auth_valid_token(self):
        """require_auth should allow access with valid token."""
        from aragora.server.middleware import require_auth
        from aragora.server.handlers.base import HandlerResult

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer correct-token"}

        @require_auth
        def protected_func(handler=None):
            return HandlerResult(status_code=200, content_type="text/plain", body=b"success")

        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.api_token = "correct-token"
            mock_config.validate_token.return_value = True

            result = protected_func(handler=handler)
            assert result.status_code == 200
            assert result.body == b"success"

    def test_optional_auth_unauthenticated(self):
        """optional_auth should provide unauthenticated context."""
        from aragora.server.middleware import optional_auth, AuthContext

        handler = MagicMock()
        handler.headers = {}
        handler.client_address = ("192.168.1.1", 12345)

        @optional_auth
        def public_func(handler=None, auth_context: AuthContext = None):
            return auth_context

        result = public_func(handler=handler)
        assert result.authenticated is False
        assert result.token is None
        assert result.client_ip == "192.168.1.1"

    def test_optional_auth_authenticated(self):
        """optional_auth should provide authenticated context with valid token."""
        from aragora.server.middleware import optional_auth, AuthContext

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer valid-token"}
        handler.client_address = ("192.168.1.1", 12345)

        @optional_auth
        def public_func(handler=None, auth_context: AuthContext = None):
            return auth_context

        with patch("aragora.server.middleware.auth.validate_token") as mock_validate:
            mock_validate.return_value = True

            result = public_func(handler=handler)
            assert result.authenticated is True
            assert result.token == "valid-token"

    def test_require_auth_or_localhost_allows_localhost(self):
        """require_auth_or_localhost should allow localhost access."""
        from aragora.server.middleware import require_auth_or_localhost
        from aragora.server.handlers.base import HandlerResult

        handler = MagicMock()
        handler.headers = {}  # No auth header
        handler.client_address = ("127.0.0.1", 12345)

        @require_auth_or_localhost
        def dev_func(handler=None):
            return HandlerResult(status_code=200, content_type="text/plain", body=b"ok")

        result = dev_func(handler=handler)
        assert result.status_code == 200

    def test_require_auth_or_localhost_ipv6(self):
        """require_auth_or_localhost should allow IPv6 localhost."""
        from aragora.server.middleware import require_auth_or_localhost
        from aragora.server.handlers.base import HandlerResult

        handler = MagicMock()
        handler.headers = {}
        handler.client_address = ("::1", 12345)

        @require_auth_or_localhost
        def dev_func(handler=None):
            return HandlerResult(status_code=200, content_type="text/plain", body=b"ok")

        result = dev_func(handler=handler)
        assert result.status_code == 200


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    def test_token_bucket_allows_initial_requests(self):
        """TokenBucket should allow initial burst of requests."""
        from aragora.server.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        # Should allow burst of 10 requests
        for i in range(10):
            assert bucket.consume(1) is True, f"Request {i+1} should be allowed"

        # 11th should be denied
        assert bucket.consume(1) is False

    def test_token_bucket_refills_over_time(self):
        """TokenBucket should refill tokens over time."""
        from aragora.server.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate_per_minute=600, burst_size=10)  # 10 per second

        # Consume all tokens
        for _ in range(10):
            bucket.consume(1)

        # Should be denied immediately
        assert bucket.consume(1) is False

        # Wait a bit for refill (100ms for ~1 token at 10/s)
        time.sleep(0.15)

        # Should have refilled at least 1 token
        assert bucket.consume(1) is True

    def test_token_bucket_remaining(self):
        """TokenBucket.remaining should reflect available tokens."""
        from aragora.server.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate_per_minute=60, burst_size=5)

        assert bucket.remaining == 5
        bucket.consume(2)
        assert bucket.remaining == 3

    def test_rate_limiter_config(self):
        """RateLimiter should support endpoint configuration."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter()
        limiter.configure_endpoint("/api/expensive", 10, key_type="ip")

        config = limiter.get_config("/api/expensive")
        assert config.requests_per_minute == 10
        assert config.key_type == "ip"

    def test_rate_limiter_wildcard_config(self):
        """RateLimiter should match wildcard endpoints."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter()
        limiter.configure_endpoint("/api/debates/*", 30, key_type="combined")

        config = limiter.get_config("/api/debates/123/fork")
        assert config.requests_per_minute == 30
        assert config.key_type == "combined"

    def test_rate_limiter_allow_default(self):
        """RateLimiter should use default limit for unconfigured endpoints."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter(default_limit=100)

        result = limiter.allow("192.168.1.1", endpoint="/api/unknown")
        assert result.allowed is True
        assert result.limit == 100

    def test_rate_limiter_blocks_after_limit(self):
        """RateLimiter should block after limit exceeded."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter(ip_limit=3)

        # First 3 requests should be allowed
        for _ in range(3):
            result = limiter.allow("192.168.1.1")
            assert result.allowed is True

        # 4th should be blocked (3 * burst_multiplier = 6, but we consumed 3)
        # Actually, default burst is 2x, so should allow 6
        # Let's consume more to hit the limit
        for _ in range(3):
            limiter.allow("192.168.1.1")

        # Now should be blocked
        result = limiter.allow("192.168.1.1")
        assert result.allowed is False
        assert result.retry_after > 0

    def test_rate_limiter_per_ip_isolation(self):
        """RateLimiter should track limits per IP separately."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter(ip_limit=2)

        # Exhaust limit for IP 1
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")

        # IP 2 should still be allowed
        result = limiter.allow("192.168.1.2")
        assert result.allowed is True

    def test_rate_limiter_stats(self):
        """RateLimiter should provide stats."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter()
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.2")

        stats = limiter.get_stats()
        assert "ip_buckets" in stats
        assert stats["ip_buckets"] == 2
        assert "default_limit" in stats

    def test_rate_limiter_cleanup(self):
        """RateLimiter.cleanup should remove stale entries."""
        from aragora.server.middleware import RateLimiter

        limiter = RateLimiter()
        limiter.allow("192.168.1.1")

        # With max_age=0, all entries are stale
        removed = limiter.cleanup(max_age_seconds=0)
        assert removed >= 1

        stats = limiter.get_stats()
        assert stats["ip_buckets"] == 0

    def test_rate_limit_decorator_allows_under_limit(self):
        """rate_limit decorator should allow requests under limit."""
        from aragora.server.middleware import rate_limit
        from aragora.server.middleware.rate_limit import reset_rate_limiters
        from aragora.server.handlers.base import HandlerResult

        reset_rate_limiters()

        handler = MagicMock()
        handler.headers = {}
        handler.client_address = ("192.168.1.1", 12345)

        @rate_limit(requests_per_minute=60)
        def my_endpoint(self, path, query_params, handler=None):
            return HandlerResult(status_code=200, content_type="application/json", body=b"{}")

        result = my_endpoint(None, "/api/test", {}, handler=handler)
        assert result.status_code == 200

    def test_rate_limit_decorator_blocks_over_limit(self):
        """rate_limit decorator should block requests over limit."""
        from aragora.server.middleware import rate_limit
        from aragora.server.middleware.rate_limit import reset_rate_limiters

        reset_rate_limiters()

        handler = MagicMock()
        handler.headers = {}
        handler.client_address = ("10.0.0.99", 12345)  # Unique IP

        # Very low limit to trigger blocking quickly
        @rate_limit(requests_per_minute=1, burst=1, limiter_name="test_block_unique")
        def limited_endpoint(self, path, query_params, handler=None):
            return MagicMock(status_code=200, headers={})

        # Exhaust all tokens (initial burst capacity)
        for _ in range(3):
            limited_endpoint(None, "/api/test", {}, handler=handler)

        # Now should be blocked
        result = limited_endpoint(None, "/api/test", {}, handler=handler)
        assert result.status_code == 429

    def test_get_rate_limiter_creates_default(self):
        """get_rate_limiter should create default limiter."""
        from aragora.server.middleware import get_rate_limiter
        from aragora.server.middleware.rate_limit import reset_rate_limiters

        reset_rate_limiters()

        limiter = get_rate_limiter()
        assert limiter is not None

        # Calling again returns same instance
        limiter2 = get_rate_limiter()
        assert limiter is limiter2

    def test_cleanup_rate_limiters(self):
        """cleanup_rate_limiters should clean all limiters."""
        from aragora.server.middleware import get_rate_limiter, cleanup_rate_limiters
        from aragora.server.middleware.rate_limit import reset_rate_limiters

        reset_rate_limiters()

        limiter = get_rate_limiter()
        limiter.allow("192.168.1.1")

        # Cleanup with max_age=0 removes all
        removed = cleanup_rate_limiters(max_age_seconds=0)
        assert removed >= 1


class TestCacheMiddleware:
    """Tests for caching middleware."""

    def test_bounded_ttl_cache_set_get(self):
        """BoundedTTLCache should store and retrieve values."""
        from aragora.server.middleware.cache import get_bounded_ttl_cache_class

        BoundedTTLCache = get_bounded_ttl_cache_class()
        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value1"

    def test_bounded_ttl_cache_expiry(self):
        """BoundedTTLCache should expire old entries."""
        from aragora.server.middleware.cache import get_bounded_ttl_cache_class

        BoundedTTLCache = get_bounded_ttl_cache_class()
        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        # Wait for TTL to expire
        time.sleep(0.1)

        # Should miss with 0.05s TTL
        hit, value = cache.get("key1", ttl_seconds=0.05)
        assert hit is False
        assert value is None

    def test_bounded_ttl_cache_lru_eviction(self):
        """BoundedTTLCache should evict old entries when full."""
        from aragora.server.middleware.cache import get_bounded_ttl_cache_class

        BoundedTTLCache = get_bounded_ttl_cache_class()
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Adding 4th should trigger eviction
        cache.set("key4", "value4")

        # key1 should be evicted (oldest)
        assert "key1" not in cache
        # key4 should exist
        assert "key4" in cache

    def test_bounded_ttl_cache_stats(self):
        """BoundedTTLCache should track stats."""
        from aragora.server.middleware.cache import get_bounded_ttl_cache_class

        BoundedTTLCache = get_bounded_ttl_cache_class()
        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        cache.get("key1", 60)  # Hit
        cache.get("key2", 60)  # Miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["hit_rate"] == 0.5

    def test_bounded_ttl_cache_clear(self):
        """BoundedTTLCache.clear should remove entries."""
        from aragora.server.middleware.cache import get_bounded_ttl_cache_class

        BoundedTTLCache = get_bounded_ttl_cache_class()
        cache = BoundedTTLCache()
        cache.set("prefix1:key1", "value1")
        cache.set("prefix1:key2", "value2")
        cache.set("prefix2:key3", "value3")

        # Clear only prefix1
        cleared = cache.clear("prefix1")
        assert cleared == 2
        assert len(cache) == 1
        assert "prefix2:key3" in cache

    def test_ttl_cache_decorator(self):
        """ttl_cache decorator should cache function results."""
        from aragora.server.middleware import ttl_cache
        from aragora.server.middleware.cache import reset_cache

        reset_cache()
        call_count = 0

        @ttl_cache(ttl_seconds=60, key_prefix="test", skip_first=False)
        def expensive_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call - cache miss
        result1 = expensive_func(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call - cache hit
        result2 = expensive_func(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented

        # Different args - cache miss
        result3 = expensive_func(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_cache_alias(self):
        """cache should be an alias for ttl_cache."""
        from aragora.server.middleware import cache, ttl_cache

        assert cache is ttl_cache

    def test_clear_cache_function(self):
        """clear_cache should clear global cache."""
        from aragora.server.middleware import clear_cache, ttl_cache
        from aragora.server.middleware.cache import reset_cache, get_cache

        reset_cache()

        @ttl_cache(ttl_seconds=60, key_prefix="test", skip_first=False)
        def cached_func():
            return "result"

        cached_func()  # Populate cache
        assert len(get_cache()) > 0

        cleared = clear_cache()
        assert cleared > 0
        assert len(get_cache()) == 0

    def test_get_cache_stats_function(self):
        """get_cache_stats should return cache statistics."""
        from aragora.server.middleware import get_cache_stats, ttl_cache
        from aragora.server.middleware.cache import reset_cache

        reset_cache()

        @ttl_cache(ttl_seconds=60, key_prefix="stats_test", skip_first=False)
        def cached_func():
            return "result"

        cached_func()  # Miss
        cached_func()  # Hit

        stats = get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_invalidate_cache_by_source(self):
        """invalidate_cache should clear related cache entries."""
        from aragora.server.middleware import invalidate_cache
        from aragora.server.middleware.cache import (
            reset_cache,
            get_cache,
            get_cache_invalidation_map,
        )

        reset_cache()
        cache = get_cache()

        # Add some cache entries matching ELO prefixes
        # Note: The CACHE_INVALIDATION_MAP uses event names like "elo_updated" not "elo"
        cache_map = get_cache_invalidation_map()
        elo_prefixes = cache_map.get("elo_updated", [])[:3]
        for prefix in elo_prefixes:
            cache.set(f"{prefix}:test", "value")

        # Add unrelated entry
        cache.set("other:test", "value")

        initial_count = len(cache)
        assert (
            initial_count >= 4
        ), f"Expected at least 4 entries, got {initial_count}. Prefixes: {elo_prefixes}"

        # Invalidate ELO caches using the data source name (not event name)
        cleared = invalidate_cache("elo")
        assert cleared >= 3

        # Unrelated entry should remain
        hit, _ = cache.get("other:test", 60)
        assert hit is True

    def test_cache_config_dataclass(self):
        """CacheConfig should have expected fields."""
        from aragora.server.middleware import CacheConfig

        # Valid config with defaults
        config = CacheConfig()
        assert config.ttl_seconds == 60.0
        assert config.key_prefix == ""
        assert config.max_entries == 1000
        assert config.enabled is True

        # Custom config
        config = CacheConfig(
            ttl_seconds=300.0,
            key_prefix="test",
            max_entries=500,
            enabled=False,
        )
        assert config.ttl_seconds == 300.0
        assert config.key_prefix == "test"
        assert config.max_entries == 500
        assert config.enabled is False


class TestMiddlewareComposition:
    """Tests for composing multiple middleware decorators."""

    def test_stacked_decorators(self):
        """Multiple middleware decorators should work together."""
        from aragora.server.middleware import require_auth, rate_limit, ttl_cache
        from aragora.server.middleware.rate_limit import reset_rate_limiters
        from aragora.server.middleware.cache import reset_cache
        from aragora.server.handlers.base import HandlerResult

        reset_rate_limiters()
        reset_cache()

        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer valid-token"}
        handler.client_address = ("192.168.1.1", 12345)

        call_count = 0

        @require_auth
        @rate_limit(requests_per_minute=60, limiter_name="composed_test")
        @ttl_cache(ttl_seconds=60, key_prefix="composed")
        def protected_endpoint(self, path, query_params, handler=None):
            nonlocal call_count
            call_count += 1
            return HandlerResult(
                status_code=200, content_type="application/json", body=b'{"ok":true}'
            )

        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.api_token = "valid-token"
            mock_config.validate_token.return_value = True

            # First call - should execute function
            result = protected_endpoint(None, "/api/test", {}, handler=handler)
            assert result.status_code == 200
            assert call_count == 1

            # Second call - cached, so function not called again
            result = protected_endpoint(None, "/api/test", {}, handler=handler)
            assert result.status_code == 200
            assert call_count == 1  # Still 1 due to cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
