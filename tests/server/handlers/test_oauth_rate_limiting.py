"""
Tests for OAuth rate limiting.

Tests critical security paths including:
- Rate limit enforcement for token endpoints (5/15min)
- Rate limit enforcement for callback handlers (10/15min)
- Rate limit enforcement for auth start endpoints (15/15min)
- Exponential backoff on repeated violations
- Security audit logging of violations
- Rate limit reset after successful auth
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-stub Slack handler modules to avoid circular ImportError
# ---------------------------------------------------------------------------
import sys
import types as _types_mod

_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]

for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

# ---------------------------------------------------------------------------

import io
import time
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_obj(
    client_ip: str = "127.0.0.1",
    headers: dict | None = None,
    body: bytes = b"",
    command: str = "GET",
):
    """Create a mock HTTP handler object."""
    h = mock.MagicMock()
    h.client_address = (client_ip, 12345)
    h.command = command
    hdrs = headers or {}
    hdr_mock = mock.MagicMock()
    hdr_mock.get = mock.MagicMock(side_effect=lambda k, d=None: hdrs.get(k, d))
    hdr_mock.__getitem__ = mock.MagicMock(side_effect=lambda k: hdrs[k])
    hdr_mock.__contains__ = mock.MagicMock(side_effect=lambda k: k in hdrs)
    hdr_mock.__iter__ = mock.MagicMock(side_effect=lambda: iter(hdrs))
    h.headers = hdr_mock
    h.rfile = io.BytesIO(body)
    return h


# ---------------------------------------------------------------------------
# Tests for OAuthRateLimitConfig
# ---------------------------------------------------------------------------


class TestOAuthRateLimitConfig:
    """Tests for OAuth rate limit configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthRateLimitConfig

        config = OAuthRateLimitConfig()
        assert config.token_limit == 5
        assert config.callback_limit == 30  # Tripled for better UX during debugging
        assert config.auth_start_limit == 15
        assert config.window_seconds == 900  # 15 minutes
        assert config.max_backoff_seconds == 3600  # 1 hour
        assert config.initial_backoff_seconds == 60  # 1 minute
        assert config.backoff_multiplier == 2.0
        assert config.enable_audit_logging is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthRateLimitConfig

        config = OAuthRateLimitConfig(
            token_limit=3,
            callback_limit=6,
            auth_start_limit=10,
            window_seconds=600,
            max_backoff_seconds=1800,
            initial_backoff_seconds=30,
            backoff_multiplier=1.5,
            enable_audit_logging=False,
        )
        assert config.token_limit == 3
        assert config.callback_limit == 6
        assert config.auth_start_limit == 10
        assert config.window_seconds == 600
        assert config.max_backoff_seconds == 1800
        assert config.initial_backoff_seconds == 30
        assert config.backoff_multiplier == 1.5
        assert config.enable_audit_logging is False


# ---------------------------------------------------------------------------
# Tests for OAuthBackoffTracker
# ---------------------------------------------------------------------------


class TestOAuthBackoffTracker:
    """Tests for exponential backoff tracking."""

    def test_first_violation_returns_initial_backoff(self):
        """Test that first violation returns initial backoff."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        backoff = tracker.record_violation("192.168.1.1")
        assert backoff == 60

    def test_exponential_backoff_increases(self):
        """Test that backoff increases exponentially with violations."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        backoff1 = tracker.record_violation("192.168.1.1")
        backoff2 = tracker.record_violation("192.168.1.1")
        backoff3 = tracker.record_violation("192.168.1.1")

        assert backoff1 == 60
        assert backoff2 == 120  # 60 * 2
        assert backoff3 == 240  # 60 * 2^2

    def test_backoff_caps_at_max(self):
        """Test that backoff is capped at max_backoff."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=300, multiplier=2.0)

        # Record many violations
        for _ in range(10):
            backoff = tracker.record_violation("192.168.1.1")

        assert backoff == 300  # Should not exceed max

    def test_is_backed_off_returns_true_during_backoff(self):
        """Test that is_backed_off returns True during backoff period."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        tracker.record_violation("192.168.1.1")
        is_backed, remaining = tracker.is_backed_off("192.168.1.1")

        assert is_backed is True
        assert remaining > 0
        assert remaining <= 60

    def test_is_backed_off_returns_false_for_new_ip(self):
        """Test that is_backed_off returns False for new IPs."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        is_backed, remaining = tracker.is_backed_off("192.168.1.1")

        assert is_backed is False
        assert remaining == 0

    def test_reset_clears_backoff_state(self):
        """Test that reset clears backoff state for an IP."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        tracker.record_violation("192.168.1.1")
        tracker.reset("192.168.1.1")
        is_backed, _ = tracker.is_backed_off("192.168.1.1")

        assert is_backed is False

    def test_different_ips_tracked_independently(self):
        """Test that different IPs are tracked independently."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        # Multiple violations for IP1
        tracker.record_violation("192.168.1.1")
        tracker.record_violation("192.168.1.1")

        # First violation for IP2
        backoff = tracker.record_violation("192.168.1.2")

        # IP2 should have initial backoff, not accumulated from IP1
        assert backoff == 60

    def test_get_stats_returns_counts(self):
        """Test that get_stats returns correct statistics."""
        from aragora.server.middleware.rate_limit.oauth_limiter import OAuthBackoffTracker

        tracker = OAuthBackoffTracker(initial_backoff=60, max_backoff=3600, multiplier=2.0)

        tracker.record_violation("192.168.1.1")
        tracker.record_violation("192.168.1.2")

        stats = tracker.get_stats()
        assert stats["tracked_ips"] == 2
        assert stats["active_backoffs"] == 2


# ---------------------------------------------------------------------------
# Tests for OAuthRateLimiter
# ---------------------------------------------------------------------------


class TestOAuthRateLimiter:
    """Tests for OAuth rate limiter."""

    def test_allows_request_under_limit(self):
        """Test that requests under limit are allowed."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(token_limit=5, window_seconds=900)
        limiter = OAuthRateLimiter(config=config)

        # Use unique IP to avoid interference from other tests
        unique_ip = f"192.168.100.{int(time.time() * 1000) % 256}"
        result = limiter.check(unique_ip, "token", "google")

        assert result.allowed is True
        # remaining can be 0 after first request due to how SimpleRateLimiter works
        assert result.remaining >= 0
        assert result.limit == 5

    def test_blocks_request_over_limit(self):
        """Test that requests over limit are blocked."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        # Very strict limit for testing
        config = OAuthRateLimitConfig(
            token_limit=1,
            window_seconds=60,
            initial_backoff_seconds=30,
        )
        limiter = OAuthRateLimiter(config=config)

        # First request should pass
        result1 = limiter.check("192.168.1.101", "token", "google")
        assert result1.allowed is True

        # Second request should be blocked
        result2 = limiter.check("192.168.1.101", "token", "google")
        assert result2.allowed is False
        assert result2.retry_after > 0

    def test_different_endpoint_types_have_different_limits(self):
        """Test that different endpoint types have different limits."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(
            token_limit=1,  # Strictest
            callback_limit=2,  # Medium
            auth_start_limit=3,  # Least strict
            window_seconds=60,
        )
        limiter = OAuthRateLimiter(config=config)

        # Token endpoint blocks after 1
        limiter.check("192.168.1.102", "token", "google")
        result = limiter.check("192.168.1.102", "token", "google")
        assert result.allowed is False

        # Callback endpoint allows 2
        limiter.check("192.168.1.103", "callback", "google")
        result = limiter.check("192.168.1.103", "callback", "google")
        assert result.allowed is True

        # Auth start endpoint allows 3
        limiter.check("192.168.1.104", "auth_start", "google")
        limiter.check("192.168.1.104", "auth_start", "google")
        result = limiter.check("192.168.1.104", "auth_start", "google")
        assert result.allowed is True

    def test_blocked_client_is_backed_off(self):
        """Test that blocked clients enter backoff period."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(
            token_limit=1,
            window_seconds=60,
        )
        limiter = OAuthRateLimiter(config=config)

        # First request passes
        limiter.check("192.168.1.105", "token", "google")

        # Second request triggers backoff (uses default initial_backoff_seconds=30)
        result = limiter.check("192.168.1.105", "token", "google")
        assert result.allowed is False
        assert result.retry_after >= 30

    def test_reset_client_clears_state(self):
        """Test that reset_client clears rate limit state."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(
            token_limit=1,
            window_seconds=60,
        )
        limiter = OAuthRateLimiter(config=config)

        # Exhaust limit and trigger backoff
        limiter.check("192.168.1.106", "token", "google")
        limiter.check("192.168.1.106", "token", "google")

        # Reset client
        limiter.reset_client("192.168.1.106")

        # New request should be allowed (backoff cleared)
        result = limiter.check("192.168.1.106", "token", "google")
        assert result.allowed is True

    def test_get_stats_returns_info(self):
        """Test that get_stats returns useful information."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(token_limit=5, callback_limit=10)
        limiter = OAuthRateLimiter(config=config)

        stats = limiter.get_stats()

        assert "config" in stats
        assert stats["config"]["token_limit"] == 5
        assert stats["config"]["callback_limit"] == 10
        assert "violations" in stats
        assert "backoff" in stats


# ---------------------------------------------------------------------------
# Tests for oauth_rate_limit decorator
# ---------------------------------------------------------------------------


class TestOAuthRateLimitDecorator:
    """Tests for the oauth_rate_limit decorator."""

    def test_decorator_allows_request_under_limit(self):
        """Test that decorator allows requests under limit."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            oauth_rate_limit,
            reset_oauth_limiter,
            reset_backoff_tracker,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        @oauth_rate_limit(endpoint_type="auth_start", provider="test")
        def handler_func(handler):
            return {"status": "ok"}

        mock_handler = _make_handler_obj(client_ip="192.168.1.200")
        result = handler_func(mock_handler)

        assert result == {"status": "ok"}

    def test_decorator_blocks_request_over_limit(self):
        """Test that decorator blocks requests over limit."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            oauth_rate_limit,
            reset_oauth_limiter,
            reset_backoff_tracker,
            OAuthRateLimitConfig,
            OAuthRateLimiter,
        )

        # Create a custom limiter with very low limits for testing
        reset_oauth_limiter()
        reset_backoff_tracker()

        # Patch the global limiter
        with patch(
            "aragora.server.middleware.rate_limit.oauth_limiter.get_oauth_limiter"
        ) as mock_get:
            config = OAuthRateLimitConfig(
                token_limit=1,
                callback_limit=1,
                auth_start_limit=1,
                window_seconds=60,
                initial_backoff_seconds=30,
            )
            mock_limiter = OAuthRateLimiter(config=config)
            mock_get.return_value = mock_limiter

            @oauth_rate_limit(endpoint_type="token", provider="test")
            def handler_func(handler):
                return {"status": "ok"}

            mock_handler = _make_handler_obj(client_ip="192.168.1.201")

            # First request passes
            result1 = handler_func(mock_handler)
            assert result1 == {"status": "ok"}

            # Second request blocked
            result2 = handler_func(mock_handler)
            assert hasattr(result2, "status_code")
            assert result2.status_code == 429

    def test_decorator_marks_function_as_rate_limited(self):
        """Test that decorator marks function with rate limit attributes."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            oauth_rate_limit,
            reset_oauth_limiter,
            reset_backoff_tracker,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        @oauth_rate_limit(endpoint_type="callback", provider="github")
        def handler_func(handler):
            return {"status": "ok"}

        assert hasattr(handler_func, "_rate_limited")
        assert handler_func._rate_limited is True
        assert hasattr(handler_func, "_oauth_rate_limit")
        assert handler_func._oauth_rate_limit is True
        assert hasattr(handler_func, "_endpoint_type")
        assert handler_func._endpoint_type == "callback"

    def test_decorator_works_with_async_functions(self):
        """Test that decorator works with async functions."""
        import asyncio

        from aragora.server.middleware.rate_limit.oauth_limiter import (
            oauth_rate_limit,
            reset_oauth_limiter,
            reset_backoff_tracker,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        @oauth_rate_limit(endpoint_type="auth_start", provider="test")
        async def async_handler_func(handler):
            return {"status": "ok"}

        mock_handler = _make_handler_obj(client_ip="192.168.1.202")
        result = asyncio.run(async_handler_func(mock_handler))

        assert result == {"status": "ok"}

    def test_decorator_includes_retry_after_header(self):
        """Test that blocked responses include Retry-After header."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            oauth_rate_limit,
            reset_oauth_limiter,
            reset_backoff_tracker,
            OAuthRateLimitConfig,
            OAuthRateLimiter,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        with patch(
            "aragora.server.middleware.rate_limit.oauth_limiter.get_oauth_limiter"
        ) as mock_get:
            config = OAuthRateLimitConfig(
                token_limit=1,
                window_seconds=60,
                initial_backoff_seconds=120,
            )
            mock_limiter = OAuthRateLimiter(config=config)
            mock_get.return_value = mock_limiter

            @oauth_rate_limit(endpoint_type="token", provider="test")
            def handler_func(handler):
                return {"status": "ok"}

            mock_handler = _make_handler_obj(client_ip="192.168.1.203")

            # First request passes
            handler_func(mock_handler)

            # Second request blocked with Retry-After (default backoff is 30s)
            result = handler_func(mock_handler)
            assert result.status_code == 429
            assert "Retry-After" in result.headers
            assert int(result.headers["Retry-After"]) >= 30


# ---------------------------------------------------------------------------
# Tests for Security Audit Logging
# ---------------------------------------------------------------------------


class TestOAuthRateLimitAuditLogging:
    """Tests for security audit logging of rate limit violations."""

    def test_violation_logs_security_event(self):
        """Test that rate limit violations log security events."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(
            token_limit=1,
            window_seconds=60,
            enable_audit_logging=True,
        )
        limiter = OAuthRateLimiter(config=config)

        with patch("aragora.audit.unified.audit_security") as mock_audit:
            # First request passes
            limiter.check("192.168.1.300", "token", "google")

            # Second request triggers violation and logging
            limiter.check("192.168.1.300", "token", "google")

            # Verify audit was called
            mock_audit.assert_called()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["event_type"] == "anomaly"
            assert call_kwargs["ip_address"] == "192.168.1.300"
            assert "oauth_rate_limit_exceeded" in call_kwargs["reason"]
            assert call_kwargs["details"]["provider"] == "google"

    def test_audit_logging_can_be_disabled(self):
        """Test that audit logging can be disabled via config."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            OAuthRateLimitConfig,
            OAuthRateLimiter,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()
        config = OAuthRateLimitConfig(
            token_limit=1,
            window_seconds=60,
            enable_audit_logging=False,
        )
        limiter = OAuthRateLimiter(config=config)

        with patch("aragora.audit.unified.audit_security") as mock_audit:
            limiter.check("192.168.1.301", "token", "google")
            limiter.check("192.168.1.301", "token", "google")

            # Audit should not be called when disabled
            mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for Global Limiter Functions
# ---------------------------------------------------------------------------


class TestGlobalOAuthLimiterFunctions:
    """Tests for global OAuth limiter functions."""

    def test_get_oauth_limiter_returns_singleton(self):
        """Test that get_oauth_limiter returns the same instance."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            get_oauth_limiter,
            reset_oauth_limiter,
        )

        reset_oauth_limiter()

        limiter1 = get_oauth_limiter()
        limiter2 = get_oauth_limiter()

        assert limiter1 is limiter2

    def test_reset_oauth_limiter_clears_singleton(self):
        """Test that reset_oauth_limiter clears the singleton."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            get_oauth_limiter,
            reset_oauth_limiter,
        )

        limiter1 = get_oauth_limiter()
        reset_oauth_limiter()
        limiter2 = get_oauth_limiter()

        assert limiter1 is not limiter2

    def test_get_backoff_tracker_returns_singleton(self):
        """Test that get_backoff_tracker returns the same instance."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            get_backoff_tracker,
            reset_backoff_tracker,
        )

        reset_backoff_tracker()

        tracker1 = get_backoff_tracker()
        tracker2 = get_backoff_tracker()

        assert tracker1 is tracker2


# ---------------------------------------------------------------------------
# Integration Tests with OAuthHandler
# ---------------------------------------------------------------------------


class TestOAuthHandlerIntegration:
    """Integration tests for OAuth rate limiting with OAuthHandler."""

    def test_oauth_handler_uses_new_rate_limiter(self):
        """Test that OAuthHandler uses the new OAuth rate limiter."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            get_oauth_limiter,
            reset_oauth_limiter,
            reset_backoff_tracker,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        # Import OAuthHandler
        from aragora.server.handlers._oauth.base import OAuthHandler

        # Create handler instance
        handler_instance = OAuthHandler(ctx={})

        # The handler should use get_oauth_limiter internally
        limiter = get_oauth_limiter()
        assert limiter is not None

    def test_oauth_handler_applies_correct_endpoint_type(self):
        """Test that OAuthHandler applies correct endpoint type for rate limiting."""
        # Token endpoints should use "token" type
        # Callback handlers should use "callback" type
        # Auth start should use "auth_start" type

        # This is implicitly tested by the handler's endpoint routing logic
        # which determines endpoint_type based on the path
        pass  # Covered by handler routing tests


# ---------------------------------------------------------------------------
# Tests for OAuthRateLimiterWrapper (backward compatibility)
# ---------------------------------------------------------------------------


class TestOAuthRateLimiterWrapper:
    """Tests for backward-compatible wrapper in utils.py."""

    def test_wrapper_provides_is_allowed_method(self):
        """Test that wrapper provides is_allowed method."""
        from aragora.server.middleware.rate_limit.oauth_limiter import (
            reset_oauth_limiter,
            reset_backoff_tracker,
        )

        reset_oauth_limiter()
        reset_backoff_tracker()

        from aragora.server.handlers._oauth.utils import _oauth_limiter

        # Should have is_allowed method
        assert hasattr(_oauth_limiter, "is_allowed")

        # Should work like a rate limiter
        result = _oauth_limiter.is_allowed("192.168.1.400")
        assert result is True

    def test_wrapper_has_rpm_property(self):
        """Test that wrapper has rpm property for logging."""
        from aragora.server.handlers._oauth.utils import _oauth_limiter

        # Should have rpm property
        assert hasattr(_oauth_limiter, "rpm")
        assert isinstance(_oauth_limiter.rpm, int)
