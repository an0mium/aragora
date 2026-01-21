"""
Tests for event rate limiter.
"""

import time
from unittest.mock import patch

import pytest


class TestEventRateLimiter:
    """Tests for EventRateLimiter class."""

    def test_allows_events_up_to_burst(self):
        """Should allow events up to burst capacity."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=10, burst_capacity=5)

        # First 5 should be allowed
        results = [limiter.is_allowed("test") for _ in range(5)]
        assert all(results)

    def test_rejects_after_burst_exhausted(self):
        """Should reject events after burst is exhausted."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=10, burst_capacity=5)

        # Exhaust burst
        for _ in range(5):
            limiter.is_allowed("test")

        # Next should be rejected
        assert limiter.is_allowed("test") is False

    def test_tokens_refill_over_time(self):
        """Should refill tokens over time."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=100, burst_capacity=5)

        # Exhaust burst
        for _ in range(5):
            limiter.is_allowed("test")

        # Should be rejected immediately
        assert limiter.is_allowed("test") is False

        # Wait for tokens to refill (100/sec = 1 token per 10ms)
        time.sleep(0.05)  # 50ms = ~5 tokens

        # Should be allowed again
        assert limiter.is_allowed("test") is True

    def test_separate_buckets_per_event_type(self):
        """Should maintain separate buckets per event type."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=10, burst_capacity=3)

        # Exhaust "type_a" bucket
        for _ in range(3):
            limiter.is_allowed("type_a")
        assert limiter.is_allowed("type_a") is False

        # "type_b" should still have tokens
        assert limiter.is_allowed("type_b") is True
        assert limiter.is_allowed("type_b") is True
        assert limiter.is_allowed("type_b") is True

    def test_get_stats(self):
        """Should track statistics."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=10, burst_capacity=2)

        limiter.is_allowed("test")  # accepted
        limiter.is_allowed("test")  # accepted
        limiter.is_allowed("test")  # rejected

        stats = limiter.get_stats()

        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["rate_per_second"] == 10
        assert stats["burst_capacity"] == 2
        assert stats["active_buckets"] == 1

    def test_reset_stats(self):
        """Should reset statistics."""
        from aragora.events.dispatcher import EventRateLimiter

        limiter = EventRateLimiter(rate_per_second=10, burst_capacity=2)

        limiter.is_allowed("test")
        limiter.is_allowed("test")
        limiter.is_allowed("test")

        limiter.reset_stats()
        stats = limiter.get_stats()

        assert stats["accepted"] == 0
        assert stats["rejected"] == 0


class TestGetEventRateLimiter:
    """Tests for global rate limiter functions."""

    def test_returns_none_when_disabled(self):
        """Should return None when rate limiting is disabled."""
        from aragora.events import dispatcher

        with patch.object(dispatcher, "EVENT_RATE_LIMIT_ENABLED", False):
            # Reset global state
            dispatcher._event_rate_limiter = None

            result = dispatcher.get_event_rate_limiter()
            assert result is None

    def test_returns_singleton_when_enabled(self):
        """Should return singleton when enabled."""
        from aragora.events import dispatcher

        with patch.object(dispatcher, "EVENT_RATE_LIMIT_ENABLED", True):
            # Reset global state
            dispatcher._event_rate_limiter = None

            rl1 = dispatcher.get_event_rate_limiter()
            rl2 = dispatcher.get_event_rate_limiter()

            assert rl1 is not None
            assert rl1 is rl2


class TestDispatcherRateLimiting:
    """Tests for rate limiting in WebhookDispatcher."""

    def test_dispatcher_stats_include_rate_limited(self):
        """Dispatcher stats should include rate_limited count."""
        from aragora.events.dispatcher import WebhookDispatcher

        dispatcher = WebhookDispatcher()
        stats = dispatcher.get_stats()

        assert "rate_limited" in stats
        assert stats["rate_limited"] == 0

        dispatcher.shutdown(wait=False)

    def test_dispatcher_stats_include_rate_limiter_stats(self):
        """Dispatcher stats should include rate limiter stats when enabled."""
        from aragora.events import dispatcher as disp_module
        from aragora.events.dispatcher import WebhookDispatcher

        with patch.object(disp_module, "EVENT_RATE_LIMIT_ENABLED", True):
            # Reset global state
            disp_module._event_rate_limiter = None

            dispatcher = WebhookDispatcher()
            stats = dispatcher.get_stats()

            assert "rate_limiter" in stats
            assert "rate_per_second" in stats["rate_limiter"]

            dispatcher.shutdown(wait=False)
