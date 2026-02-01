"""
Tests for Default Rate Limiter Middleware.

Tests the automatic rate limiting applied to handlers without explicit
@rate_limit decorators, including tier-based limits (public/authenticated/admin).
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from aragora.server.middleware.rate_limit.default_limiter import (
    DEFAULT_RATE_LIMITS,
    DefaultRateLimiter,
    get_default_rate_limiter,
    reset_default_rate_limiter,
    determine_auth_tier,
    check_default_rate_limit,
    handler_has_rate_limit_decorator,
    should_apply_default_rate_limit,
    get_handler_default_rpm,
)
from aragora.server.middleware.rate_limit import rate_limit


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset the default rate limiter before and after each test."""
    reset_default_rate_limiter()
    yield
    reset_default_rate_limiter()


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.client_address = ("192.168.1.100", 12345)
    handler.headers = {
        "Authorization": "",
        "X-Forwarded-For": "",
        "X-Real-IP": "",
    }
    return handler


@pytest.fixture
def mock_authenticated_handler():
    """Create a mock authenticated HTTP handler."""
    handler = Mock()
    handler.client_address = ("192.168.1.100", 12345)
    handler.headers = {
        "Authorization": "Bearer test_token",
        "X-Forwarded-For": "",
        "X-Real-IP": "",
    }
    return handler


# ============================================================================
# DefaultRateLimiter Class Tests
# ============================================================================


class TestDefaultRateLimiter:
    """Tests for DefaultRateLimiter class."""

    def test_default_tier_limits(self):
        """Test default tier limits are correctly configured."""
        assert "public" in DEFAULT_RATE_LIMITS
        assert "authenticated" in DEFAULT_RATE_LIMITS
        assert "admin" in DEFAULT_RATE_LIMITS

        # Check default values
        assert DEFAULT_RATE_LIMITS["public"][0] == 30  # 30 req/min
        assert DEFAULT_RATE_LIMITS["authenticated"][0] == 120  # 120 req/min
        assert DEFAULT_RATE_LIMITS["admin"][0] == 300  # 300 req/min

    def test_get_tier_limits(self):
        """Test getting limits for different tiers."""
        limiter = DefaultRateLimiter()

        assert limiter.get_tier_limits("public") == (30, 60)
        assert limiter.get_tier_limits("authenticated") == (120, 240)
        assert limiter.get_tier_limits("admin") == (300, 600)

    def test_get_tier_limits_defaults_to_public(self):
        """Test unknown tiers default to public limits."""
        limiter = DefaultRateLimiter()

        assert limiter.get_tier_limits("unknown") == (30, 60)
        assert limiter.get_tier_limits("invalid") == (30, 60)

    def test_allow_request_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = DefaultRateLimiter()

        result = limiter.allow("test_client", "public")
        assert result.allowed is True
        assert result.remaining > 0

    def test_allow_tracks_by_tier(self):
        """Test rate limits are tracked separately by tier."""
        limiter = DefaultRateLimiter()

        # Same client key, different tiers - should be separate buckets
        # Public tier has burst of 60, so exhaust that
        for _ in range(65):
            limiter.allow("test_client", "public")

        # Public tier should be exhausted (burst=60 consumed)
        result_public = limiter.allow("test_client", "public")
        assert result_public.allowed is False

        # Authenticated tier should still work (separate bucket)
        result_auth = limiter.allow("test_client", "authenticated")
        assert result_auth.allowed is True

    def test_allow_rate_limit_exceeded(self):
        """Test requests exceeding limit are denied."""
        limiter = DefaultRateLimiter()

        # Exhaust the public tier (30 + burst of 60 = need to exceed 60)
        # Actually, burst is the initial tokens, so we exhaust 60 requests
        for _ in range(65):
            limiter.allow("test_client", "public")

        # Next request should be denied
        result = limiter.allow("test_client", "public")
        assert result.allowed is False
        assert result.retry_after > 0

    def test_get_stats(self):
        """Test getting limiter statistics."""
        limiter = DefaultRateLimiter()

        limiter.allow("client1", "public")
        limiter.allow("client2", "authenticated")
        limiter.allow("client3", "admin")

        stats = limiter.get_stats()
        assert "tier_buckets" in stats
        assert "tier_limits" in stats
        assert stats["tier_buckets"]["public"] == 1
        assert stats["tier_buckets"]["authenticated"] == 1
        assert stats["tier_buckets"]["admin"] == 1

    def test_reset(self):
        """Test resetting all rate limiter state."""
        limiter = DefaultRateLimiter()

        limiter.allow("client1", "public")
        limiter.allow("client2", "authenticated")

        limiter.reset()

        stats = limiter.get_stats()
        assert stats["tier_buckets"]["public"] == 0
        assert stats["tier_buckets"]["authenticated"] == 0


# ============================================================================
# Global Functions Tests
# ============================================================================


class TestGlobalFunctions:
    """Tests for global rate limiter functions."""

    def test_get_default_rate_limiter_singleton(self):
        """Test get_default_rate_limiter returns singleton."""
        limiter1 = get_default_rate_limiter()
        limiter2 = get_default_rate_limiter()

        assert limiter1 is limiter2

    def test_reset_default_rate_limiter(self):
        """Test reset_default_rate_limiter clears state."""
        limiter1 = get_default_rate_limiter()
        limiter1.allow("test", "public")

        reset_default_rate_limiter()

        limiter2 = get_default_rate_limiter()
        assert limiter1 is not limiter2


# ============================================================================
# Auth Tier Detection Tests
# ============================================================================


class TestDetermineAuthTier:
    """Tests for determine_auth_tier function."""

    def test_unauthenticated_request(self, mock_handler):
        """Test unauthenticated requests get public tier."""
        tier, client_key = determine_auth_tier(mock_handler)

        assert tier == "public"
        assert "192.168.1.100" in client_key or client_key == "192.168.1.100"

    def test_authenticated_user_tier(self, mock_authenticated_handler):
        """Test authenticated users get authenticated tier."""
        # Mock the auth extraction
        mock_auth_ctx = Mock()
        mock_auth_ctx.is_authenticated = True
        mock_auth_ctx.user_id = "user123"
        mock_auth_ctx.role = "member"
        mock_auth_ctx.roles = []
        mock_auth_ctx.permissions = []
        mock_auth_ctx.is_admin = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth_ctx,
        ):
            tier, client_key = determine_auth_tier(mock_authenticated_handler)

            assert tier == "authenticated"
            assert client_key == "user123"

    def test_admin_user_tier(self, mock_authenticated_handler):
        """Test admin users get admin tier."""
        mock_auth_ctx = Mock()
        mock_auth_ctx.is_authenticated = True
        mock_auth_ctx.user_id = "admin123"
        mock_auth_ctx.role = "admin"
        mock_auth_ctx.roles = ["admin"]
        mock_auth_ctx.permissions = []
        mock_auth_ctx.is_admin = True

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth_ctx,
        ):
            tier, client_key = determine_auth_tier(mock_authenticated_handler)

            assert tier == "admin"
            assert client_key == "admin123"

    def test_admin_by_permission(self, mock_authenticated_handler):
        """Test users with admin permission get admin tier."""
        mock_auth_ctx = Mock()
        mock_auth_ctx.is_authenticated = True
        mock_auth_ctx.user_id = "user456"
        mock_auth_ctx.role = "member"
        mock_auth_ctx.roles = []
        mock_auth_ctx.permissions = ["admin"]
        mock_auth_ctx.is_admin = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth_ctx,
        ):
            tier, client_key = determine_auth_tier(mock_authenticated_handler)

            assert tier == "admin"

    def test_auth_extraction_failure_falls_back_to_public(self, mock_handler):
        """Test auth extraction failure falls back to public tier."""
        # When extract_user_from_request returns None or non-authenticated context
        mock_auth_ctx = Mock()
        mock_auth_ctx.is_authenticated = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth_ctx,
        ):
            tier, client_key = determine_auth_tier(mock_handler)

            assert tier == "public"


# ============================================================================
# Handler Rate Limit Detection Tests
# ============================================================================


class TestHandlerRateLimitDetection:
    """Tests for detecting explicit rate limit decorators on handlers."""

    def test_detect_rate_limit_decorator(self):
        """Test detection of @rate_limit decorated methods."""

        class MockHandler:
            @rate_limit(requests_per_minute=30)
            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()
        assert handler_has_rate_limit_decorator(handler, "handle") is True

    def test_no_rate_limit_decorator(self):
        """Test detection returns False when no decorator."""

        class MockHandler:
            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()
        assert handler_has_rate_limit_decorator(handler, "handle") is False

    def test_missing_method(self):
        """Test detection returns False for missing methods."""

        class MockHandler:
            pass

        handler = MockHandler()
        assert handler_has_rate_limit_decorator(handler, "handle_post") is False


class TestShouldApplyDefaultRateLimit:
    """Tests for should_apply_default_rate_limit function."""

    def test_applies_to_normal_handler(self):
        """Test default rate limit applies to handlers without decorator."""

        class MockHandler:
            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()
        assert should_apply_default_rate_limit(handler, "handle") is True

    def test_does_not_apply_when_exempt(self):
        """Test default rate limit skipped when RATE_LIMIT_EXEMPT is True."""

        class MockHandler:
            RATE_LIMIT_EXEMPT = True

            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()
        assert should_apply_default_rate_limit(handler, "handle") is False

    def test_does_not_apply_when_has_decorator(self):
        """Test default rate limit skipped when handler has @rate_limit."""

        class MockHandler:
            @rate_limit(requests_per_minute=30)
            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()
        assert should_apply_default_rate_limit(handler, "handle") is False

    def test_does_not_apply_when_globally_disabled(self):
        """Test default rate limit skipped when globally disabled."""

        class MockHandler:
            def handle(self, path, query_params, handler):
                pass

        handler = MockHandler()

        with patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": "1"}):
            assert should_apply_default_rate_limit(handler, "handle") is False


class TestGetHandlerDefaultRpm:
    """Tests for get_handler_default_rpm function."""

    def test_returns_custom_rpm(self):
        """Test returns custom RPM when set."""

        class MockHandler:
            DEFAULT_RATE_LIMIT_RPM = 60

        handler = MockHandler()
        assert get_handler_default_rpm(handler) == 60

    def test_returns_none_when_not_set(self):
        """Test returns None when no custom RPM."""

        class MockHandler:
            pass

        handler = MockHandler()
        assert get_handler_default_rpm(handler) is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestCheckDefaultRateLimit:
    """Integration tests for check_default_rate_limit."""

    def test_public_rate_limit(self, mock_handler):
        """Test public tier rate limiting."""
        # Make requests up to the burst limit
        for _ in range(60):
            result = check_default_rate_limit(mock_handler)
            assert result.allowed is True

        # Next requests should be denied
        for _ in range(5):
            check_default_rate_limit(mock_handler)

        result = check_default_rate_limit(mock_handler)
        assert result.allowed is False

    def test_authenticated_higher_limit(self):
        """Test authenticated users have higher limits."""
        mock_handler = Mock()
        mock_handler.client_address = ("192.168.1.100", 12345)
        mock_handler.headers = {
            "Authorization": "Bearer test_token",
            "X-Forwarded-For": "",
            "X-Real-IP": "",
        }

        mock_auth_ctx = Mock()
        mock_auth_ctx.is_authenticated = True
        mock_auth_ctx.user_id = "user123"
        mock_auth_ctx.role = "member"
        mock_auth_ctx.roles = []
        mock_auth_ctx.permissions = []
        mock_auth_ctx.is_admin = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth_ctx,
        ):
            # Authenticated users can make more requests (120/min + 240 burst)
            for _ in range(240):
                result = check_default_rate_limit(mock_handler)
                assert result.allowed is True

    def test_rate_limit_result_contains_metadata(self, mock_handler):
        """Test rate limit result contains proper metadata."""
        result = check_default_rate_limit(mock_handler)

        assert hasattr(result, "allowed")
        assert hasattr(result, "remaining")
        assert hasattr(result, "limit")
        assert hasattr(result, "retry_after")
        assert hasattr(result, "key")

        assert result.limit == 30  # Public tier limit


# ============================================================================
# Environment Variable Override Tests
# ============================================================================


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_public_rate_limit_env_override(self):
        """Test ARAGORA_PUBLIC_RATE_LIMIT env var override."""
        # Note: This test would need to reload the module to test env vars
        # For now, we test that the DEFAULT_RATE_LIMITS can be customized
        custom_limits = {
            "public": (50, 100),
            "authenticated": (200, 400),
            "admin": (500, 1000),
        }

        limiter = DefaultRateLimiter(tier_limits=custom_limits)
        assert limiter.get_tier_limits("public") == (50, 100)
        assert limiter.get_tier_limits("authenticated") == (200, 400)
        assert limiter.get_tier_limits("admin") == (500, 1000)
