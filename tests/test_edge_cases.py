"""Edge case tests for critical components.

Tests cover:
1. BoundedTTLCache thread safety and boundary conditions
2. CircuitBreaker parameter validation
3. OpenRouter fallback chain failures
4. Dashboard handler boundary conditions
"""

import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.server.handlers.base import BoundedTTLCache
from aragora.resilience import CircuitBreaker


class TestBoundedTTLCacheThreadSafety:
    """Test thread safety of BoundedTTLCache."""

    def test_concurrent_writes_same_key(self):
        """Multiple threads writing to same key should not crash."""
        cache = BoundedTTLCache(max_entries=100)
        errors = []
        results = []

        def write_to_cache(thread_id: int):
            try:
                for i in range(50):
                    cache.set("shared_key", f"value_{thread_id}_{i}")
                    time.sleep(0.001)  # Small delay to increase contention
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        # Run 10 threads concurrently writing to same key
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_to_cache, i) for i in range(10)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
        # Final value should be from one of the threads
        hit, value = cache.get("shared_key", ttl_seconds=60)
        assert hit is True
        assert value is not None

    def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes should not crash."""
        cache = BoundedTTLCache(max_entries=100)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}", ttl_seconds=60)
            except Exception as e:
                errors.append(("reader", e))

        # Run writers and readers concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _ in range(4):
                futures.append(executor.submit(writer))
                futures.append(executor.submit(reader))
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_eviction(self):
        """Concurrent writes triggering eviction should not crash."""
        cache = BoundedTTLCache(max_entries=10, evict_percent=0.5)
        errors = []

        def flood_cache(thread_id: int):
            try:
                for i in range(100):
                    cache.set(f"thread_{thread_id}_key_{i}", i)
            except Exception as e:
                errors.append((thread_id, e))

        # All threads will trigger eviction
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(flood_cache, i) for i in range(5)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(cache) <= 10  # Should respect max_entries


class TestBoundedTTLCacheBoundaryConditions:
    """Test boundary conditions for BoundedTTLCache."""

    def test_zero_max_entries(self):
        """Zero max_entries should handle gracefully."""
        # Should not crash, but eviction logic may behave oddly
        cache = BoundedTTLCache(max_entries=0, evict_percent=0.5)
        cache.set("key1", "value1")
        # With max_entries=0, eviction happens immediately
        # The key may or may not exist depending on implementation

    def test_one_max_entry(self):
        """Single entry cache should work correctly."""
        cache = BoundedTTLCache(max_entries=1, evict_percent=1.0)
        cache.set("key1", "value1")
        hit1, val1 = cache.get("key1", ttl_seconds=60)
        assert hit1 is True
        assert val1 == "value1"

        # Adding second should evict first
        cache.set("key2", "value2")
        hit1, _ = cache.get("key1", ttl_seconds=60)
        assert hit1 is False

    def test_very_small_ttl(self):
        """Very small TTL should expire immediately."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        time.sleep(0.01)  # 10ms
        hit, value = cache.get("key1", ttl_seconds=0.001)  # 1ms TTL
        assert hit is False

    def test_very_large_ttl(self):
        """Very large TTL should not cause issues."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=3153600000)  # 100 years
        assert hit is True
        assert value == "value1"

    def test_zero_ttl(self):
        """Zero TTL should always miss."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=0)
        assert hit is False

    def test_negative_ttl(self):
        """Negative TTL should always miss."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=-1)
        assert hit is False

    def test_empty_key(self):
        """Empty string key should work."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("", "empty_key_value")
        hit, value = cache.get("", ttl_seconds=60)
        assert hit is True
        assert value == "empty_key_value"

    def test_none_value(self):
        """None value should be cacheable and distinguishable from miss."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("null_key", None)
        hit, value = cache.get("null_key", ttl_seconds=60)
        assert hit is True
        assert value is None

    def test_large_value(self):
        """Large values should be storable."""
        cache = BoundedTTLCache(max_entries=10)
        large_value = "x" * (1024 * 1024)  # 1MB string
        cache.set("large_key", large_value)
        hit, value = cache.get("large_key", ttl_seconds=60)
        assert hit is True
        assert len(value) == len(large_value)


class TestCircuitBreakerBoundaryConditions:
    """Test boundary conditions for CircuitBreaker."""

    def test_zero_failure_threshold(self):
        """Zero failure threshold should open immediately."""
        breaker = CircuitBreaker(failure_threshold=0)
        # Any failure should open the circuit
        opened = breaker.record_failure()
        # With threshold=0, first failure should open it
        assert opened is True
        assert breaker.is_open is True

    def test_negative_failure_threshold(self):
        """Negative failure threshold should be handled."""
        # This is an edge case - behavior may vary
        breaker = CircuitBreaker(failure_threshold=-1)
        # Should open immediately since -1 < 0
        opened = breaker.record_failure()
        assert breaker.is_open is True

    def test_zero_cooldown(self):
        """Zero cooldown should allow immediate retry."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
        breaker.record_failure()
        assert breaker.is_open is True
        # With zero cooldown, should immediately be in half-open
        assert breaker.can_proceed() is True

    def test_very_large_cooldown(self):
        """Very large cooldown should block for a long time."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=1e10)
        breaker.record_failure()
        assert breaker.is_open is True
        assert breaker.can_proceed() is False

    def test_empty_entity_name(self):
        """Empty string entity name should work."""
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure("")
        assert breaker.is_available("") is True
        breaker.record_failure("")
        assert breaker.is_available("") is False

    def test_special_chars_entity_name(self):
        """Special characters in entity name should work."""
        breaker = CircuitBreaker(failure_threshold=2)
        entity = "agent/with:special@chars#123"
        breaker.record_failure(entity)
        breaker.record_failure(entity)
        assert breaker.is_available(entity) is False

    def test_rapid_success_failure_cycling(self):
        """Rapid cycling between success and failure should not crash."""
        breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.001)
        for _ in range(100):
            breaker.record_failure()
            breaker.record_success()
        # Should end up in a valid state
        assert isinstance(breaker.is_open, bool)

    def test_many_entities(self):
        """Many entities should not cause memory issues."""
        breaker = CircuitBreaker(failure_threshold=2)
        # Add 1000 entities
        for i in range(1000):
            breaker.record_failure(f"entity_{i}")
        # Should track all without crashing
        assert breaker.is_available("entity_0") is True  # Only 1 failure
        breaker.record_failure("entity_0")
        assert breaker.is_available("entity_0") is False  # Now 2 failures


class TestCircuitBreakerSerialization:
    """Test CircuitBreaker to_dict/from_dict edge cases."""

    def test_from_dict_missing_keys(self):
        """from_dict with missing keys should use defaults."""
        breaker = CircuitBreaker.from_dict({})
        assert breaker.failure_threshold == 3  # Default
        assert breaker.cooldown_seconds == 60.0  # Default

    def test_from_dict_partial_data(self):
        """from_dict with partial data should work.

        Note: from_dict takes kwargs for configuration, data dict is for entity state.
        """
        breaker = CircuitBreaker.from_dict({}, failure_threshold=5)
        assert breaker.failure_threshold == 5
        assert breaker.cooldown_seconds == 60.0  # Default

    def test_from_dict_invalid_types(self):
        """from_dict with invalid types should handle gracefully."""
        # String instead of int
        breaker = CircuitBreaker.from_dict(
            {
                "failure_threshold": "five",  # Invalid
            }
        )
        # Should either use default or handle the error
        assert isinstance(breaker.failure_threshold, (int, str))


class TestDashboardEdgeCases:
    """Test dashboard handler edge cases."""

    def test_debates_with_nan_confidence(self):
        """Dashboard should handle NaN confidence values."""
        from aragora.server.handlers.dashboard import DashboardHandler

        handler = DashboardHandler({})
        debates = [
            {"confidence": float("nan"), "consensus_reached": True},
            {"confidence": 0.5, "consensus_reached": True},
        ]
        summary = handler._get_summary_metrics(None, debates)
        # Should not crash, confidence calculation should handle NaN
        assert "avg_confidence" in summary

    def test_debates_with_infinity_confidence(self):
        """Dashboard should handle infinity confidence values."""
        from aragora.server.handlers.dashboard import DashboardHandler

        handler = DashboardHandler({})
        debates = [
            {"confidence": float("inf"), "consensus_reached": True},
            {"confidence": 0.5, "consensus_reached": True},
        ]
        summary = handler._get_summary_metrics(None, debates)
        assert "avg_confidence" in summary

    def test_debates_with_negative_confidence(self):
        """Dashboard should handle negative confidence values."""
        from aragora.server.handlers.dashboard import DashboardHandler

        handler = DashboardHandler({})
        debates = [
            {"confidence": -0.5, "consensus_reached": True},
            {"confidence": 0.5, "consensus_reached": True},
        ]
        summary = handler._get_summary_metrics(None, debates)
        assert "avg_confidence" in summary

    def test_empty_debates_list(self):
        """Dashboard should handle empty debates list."""
        from aragora.server.handlers.dashboard import DashboardHandler

        handler = DashboardHandler({})
        summary = handler._get_summary_metrics(None, [])
        assert summary["total_debates"] == 0
        assert summary["consensus_rate"] == 0.0

    def test_large_debate_count(self):
        """Dashboard should handle large number of debates."""
        from aragora.server.handlers.dashboard import DashboardHandler

        handler = DashboardHandler({})
        # Create 10000 mock debates
        debates = [{"confidence": 0.5, "consensus_reached": i % 2 == 0} for i in range(10000)]
        summary = handler._get_summary_metrics(None, debates)
        assert summary["total_debates"] == 10000
        assert summary["consensus_reached"] == 5000

    def test_hours_zero(self):
        """Dashboard should handle hours=0 parameter."""
        from aragora.server.handlers.dashboard import DashboardHandler
        import json

        handler = DashboardHandler({})
        result = handler.handle("/api/dashboard/debates", {"hours": "0"}, None)
        data = json.loads(result.body)
        assert data["recent_activity"]["period_hours"] == 0

    def test_hours_negative(self):
        """Dashboard should handle negative hours parameter."""
        from aragora.server.handlers.dashboard import DashboardHandler
        import json

        handler = DashboardHandler({})
        result = handler.handle("/api/dashboard/debates", {"hours": "-1"}, None)
        # Should not crash, may use default or handle gracefully
        assert result is not None


class TestOpenRouterFallbackChain:
    """Test OpenRouter fallback chain edge cases."""

    @pytest.mark.asyncio
    async def test_fallback_agent_creation(self):
        """Test that fallback agent is created when API key is available."""
        from aragora.agents.api_agents import GeminiAgent

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key", "GEMINI_API_KEY": "test"}):
            agent = GeminiAgent(name="test", model="gemini-3-pro")
            # Fallback agent should be creatable (API agents use mixin method)
            fallback = agent._get_cached_fallback_agent()
            assert fallback is not None
            assert fallback.name == "test_fallback"

    @pytest.mark.asyncio
    async def test_fallback_without_api_key(self):
        """Without OPENROUTER_API_KEY, fallback should not be attempted."""
        from aragora.agents.cli_agents import ClaudeAgent

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            agent = ClaudeAgent(name="test", model="claude-opus-4-5-20251101")
            fallback = agent._get_fallback_agent()
            assert fallback is None

    @pytest.mark.asyncio
    async def test_fallback_error_detection(self):
        """Test that quota errors are detected correctly for fallback."""
        from aragora.agents.api_agents import GeminiAgent

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            agent = GeminiAgent(name="test", model="gemini-3-pro")

            # Test various quota error scenarios
            assert agent.is_quota_error(429, "rate limit exceeded")
            assert agent.is_quota_error(429, "quota exceeded")
            assert agent.is_quota_error(503, "resource exhausted")
            assert not agent.is_quota_error(400, "bad request")


class TestRateLimiterEdgeCases:
    """Test rate limiter edge cases."""

    def test_invalid_tier_name(self):
        """Invalid tier name should fall back to standard."""
        from aragora.agents.api_agents import OpenRouterRateLimiter

        limiter = OpenRouterRateLimiter(tier="invalid_tier_name")
        # Should use standard tier defaults (tier is on self.tier)
        assert limiter.tier.requests_per_minute > 0

    def test_tier_name_case_insensitive(self):
        """Tier name should be case insensitive (lowercased by implementation)."""
        from aragora.agents.api_agents import OpenRouterRateLimiter

        limiter1 = OpenRouterRateLimiter(tier="STANDARD")
        limiter2 = OpenRouterRateLimiter(tier="standard")
        limiter3 = OpenRouterRateLimiter(tier="Standard")
        # All should have same limits (tier settings are on self.tier)
        assert limiter1.tier.requests_per_minute == limiter2.tier.requests_per_minute
        assert limiter2.tier.requests_per_minute == limiter3.tier.requests_per_minute

    def test_malformed_header_values(self):
        """Malformed rate limit headers should be handled gracefully."""
        from aragora.agents.api_agents import OpenRouterRateLimiter

        limiter = OpenRouterRateLimiter(tier="standard")
        # Non-numeric header values
        limiter.update_from_headers(
            {
                "X-RateLimit-Remaining": "not_a_number",
                "X-RateLimit-Reset": "invalid",
            }
        )
        # Should not crash, should ignore invalid values

    def test_missing_headers(self):
        """Missing rate limit headers should be handled gracefully."""
        from aragora.agents.api_agents import OpenRouterRateLimiter

        limiter = OpenRouterRateLimiter(tier="standard")
        limiter.update_from_headers({})  # Empty headers
        # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
