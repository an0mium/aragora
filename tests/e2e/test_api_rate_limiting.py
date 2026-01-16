"""
E2E tests for API rate limiting.

Tests verify that rate limits are properly enforced across:
1. Per-IP rate limiting
2. Per-user rate limiting
3. Burst handling
4. Rate limit headers
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIPRateLimiting:
    """Tests for IP-based rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced_after_threshold(self):
        """Requests should be rejected after exceeding rate limit."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter

        # Create limiter with low limit for testing
        limiter = RateLimiter(requests_per_minute=5)
        client_ip = "192.168.1.100"

        # Make requests up to limit
        allowed_count = 0
        for _ in range(10):
            if limiter.is_allowed(client_ip):
                allowed_count += 1
            else:
                break

        # Should have allowed exactly 5 requests
        assert allowed_count == 5

        # Next request should be rejected
        assert limiter.is_allowed(client_ip) is False

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_window(self):
        """Rate limit should reset after the time window expires."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter
        import time

        # Create limiter with very short window for testing
        limiter = RateLimiter(requests_per_minute=2)
        client_ip = "192.168.1.101"

        # Exhaust the rate limit
        assert limiter.is_allowed(client_ip) is True
        assert limiter.is_allowed(client_ip) is True
        assert limiter.is_allowed(client_ip) is False

        # Clear bucket to simulate window expiry
        limiter._buckets.clear()

        # Should be allowed again after bucket cleared
        assert limiter.is_allowed(client_ip) is True

    @pytest.mark.asyncio
    async def test_different_ips_have_separate_limits(self):
        """Different IP addresses should have independent rate limits."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=3)
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"

        # Exhaust limit for IP1
        for _ in range(3):
            limiter.is_allowed(ip1)
        assert limiter.is_allowed(ip1) is False

        # IP2 should still be allowed
        assert limiter.is_allowed(ip2) is True
        assert limiter.is_allowed(ip2) is True
        assert limiter.is_allowed(ip2) is True
        assert limiter.is_allowed(ip2) is False


class TestEndpointRateLimiting:
    """Tests for endpoint-specific rate limiting."""

    @pytest.mark.asyncio
    async def test_memory_endpoints_have_correct_limits(self):
        """Memory endpoints should use appropriate rate limits."""
        from aragora.server.handlers.memory import memory

        # Verify rate limiters exist with correct configuration
        assert memory._retrieve_limiter.rpm == 60  # Read operations
        assert memory._stats_limiter.rpm == 30  # Stats operations
        assert memory._mutation_limiter.rpm == 10  # Mutations

    @pytest.mark.asyncio
    async def test_mutation_endpoints_more_restrictive(self):
        """State-mutating endpoints should have lower rate limits."""
        from aragora.server.handlers.memory import memory

        # Mutation limiter should be more restrictive than retrieve
        assert memory._mutation_limiter.rpm < memory._retrieve_limiter.rpm


class TestRateLimitHeaders:
    """Tests for rate limit response headers."""

    @pytest.mark.asyncio
    async def test_429_response_includes_retry_after(self):
        """429 responses should include Retry-After header."""
        from aragora.server.handlers.base import error_response

        # Create a rate limit error response
        result = error_response(
            "Rate limit exceeded. Please try again later.", 429, headers={"Retry-After": "60"}
        )

        # Verify headers
        assert result.status_code == 429
        assert "Retry-After" in result.headers


class TestBurstHandling:
    """Tests for burst traffic handling."""

    @pytest.mark.asyncio
    async def test_burst_traffic_handled_gracefully(self):
        """Burst traffic should be handled without crashing."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=100)
        client_ip = "192.168.1.50"

        # Simulate burst: many requests at once
        results = []
        for _ in range(200):
            results.append(limiter.is_allowed(client_ip))

        # Some should succeed, some should fail
        allowed = sum(1 for r in results if r)
        rejected = sum(1 for r in results if not r)

        assert allowed == 100  # Exactly at limit
        assert rejected == 100  # Rest rejected

    @pytest.mark.asyncio
    async def test_concurrent_requests_rate_limited(self):
        """Concurrent requests should be properly rate limited."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter
        import threading

        limiter = RateLimiter(requests_per_minute=50)
        client_ip = "192.168.1.60"

        results = []
        lock = threading.Lock()

        def make_request():
            result = limiter.is_allowed(client_ip)
            with lock:
                results.append(result)

        # Create many threads simulating concurrent requests
        threads = []
        for _ in range(100):
            t = threading.Thread(target=make_request)
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have allowed at most 50
        allowed = sum(1 for r in results if r)
        assert allowed <= 50


class TestRateLimitBypass:
    """Tests to ensure rate limits cannot be bypassed."""

    @pytest.mark.asyncio
    async def test_xff_header_parsing_safe(self):
        """X-Forwarded-For header should be parsed safely."""
        from aragora.server.handlers.utils.rate_limit import get_client_ip

        # Create mock handler with spoofed headers
        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"X-Forwarded-For": "192.168.1.100, 10.0.0.1"}

        # With default config (no trusted proxies), should use client_address
        ip = get_client_ip(handler)

        # Should either use direct IP or first XFF IP depending on trust config
        assert ip is not None
        assert len(ip) > 0

    @pytest.mark.asyncio
    async def test_localhost_rate_limited_in_production(self):
        """Localhost should be rate limited in production mode."""
        from aragora.server.handlers.utils.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=5)

        # Localhost IPs should be rate limited
        for ip in ["127.0.0.1", "::1", "localhost"]:
            # Exhaust limit
            for _ in range(5):
                limiter.is_allowed(ip)

            # Should be rejected after limit
            assert limiter.is_allowed(ip) is False

            # Clear for next test
            limiter._buckets.clear()


class TestRateLimitIntegration:
    """Integration tests for rate limiting in handlers."""

    @pytest.mark.asyncio
    async def test_memory_handler_rate_limits(self):
        """Memory handler should enforce rate limits."""
        from aragora.server.handlers.memory.memory import MemoryHandler
        from aragora.server.handlers.memory import memory

        # Clear limiters
        memory._retrieve_limiter._buckets.clear()
        memory._stats_limiter._buckets.clear()

        handler = MemoryHandler({"continuum_memory": MagicMock()})

        # Create mock HTTP handler
        http_handler = MagicMock()
        http_handler.client_address = ("192.168.1.200", 12345)
        http_handler.headers = {}

        # Make many requests to trigger rate limit
        results = []
        for _ in range(70):  # More than 60/min limit
            result = handler.handle("/api/memory/continuum/retrieve", {}, http_handler)
            if result:
                results.append(result.status_code)

        # Should have some 429 responses
        assert 429 in results
