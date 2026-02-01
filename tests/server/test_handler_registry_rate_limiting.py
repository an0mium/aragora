"""
Integration tests for default rate limiting in handler registry.

Tests that the handler registry correctly applies default rate limits
to handlers without explicit @rate_limit decorators.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from aragora.server.middleware.rate_limit.default_limiter import (
    reset_default_rate_limiter,
)


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
def mock_handler_class():
    """Create a mock handler class without rate limit decorator."""

    class MockHandler:
        RATE_LIMIT_EXEMPT = False

        def __init__(self, ctx):
            self.ctx = ctx

        @staticmethod
        def can_handle(path):
            return path.startswith("/api/test")

        def handle(self, path, query_params, handler):
            from aragora.server.handlers.base import json_response

            return json_response({"status": "ok"})

    return MockHandler


@pytest.fixture
def mock_exempt_handler_class():
    """Create a mock handler class that is exempt from rate limiting."""

    class MockExemptHandler:
        RATE_LIMIT_EXEMPT = True

        def __init__(self, ctx):
            self.ctx = ctx

        @staticmethod
        def can_handle(path):
            return path.startswith("/api/exempt")

        def handle(self, path, query_params, handler):
            from aragora.server.handlers.base import json_response

            return json_response({"status": "ok"})

    return MockExemptHandler


# ============================================================================
# Test Cases
# ============================================================================


class TestShouldApplyDefaultRateLimit:
    """Tests for rate limit application logic."""

    def test_should_apply_to_normal_handler(self, mock_handler_class):
        """Test that default rate limiting applies to normal handlers."""
        from aragora.server.middleware.rate_limit import should_apply_default_rate_limit

        handler = mock_handler_class({})
        assert should_apply_default_rate_limit(handler, "handle") is True

    def test_should_not_apply_to_exempt_handler(self, mock_exempt_handler_class):
        """Test that exempt handlers are skipped."""
        from aragora.server.middleware.rate_limit import should_apply_default_rate_limit

        handler = mock_exempt_handler_class({})
        assert should_apply_default_rate_limit(handler, "handle") is False

    def test_should_not_apply_with_global_disable(self, mock_handler_class):
        """Test that global disable flag is respected."""
        from aragora.server.middleware.rate_limit import should_apply_default_rate_limit

        handler = mock_handler_class({})

        with patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": "1"}):
            assert should_apply_default_rate_limit(handler, "handle") is False


class TestDefaultRateLimitIntegration:
    """Integration tests for default rate limiting."""

    def test_rate_limit_applied_to_unauthenticated_request(self):
        """Test that unauthenticated requests are rate limited at 30/min."""
        from aragora.server.middleware.rate_limit import (
            check_default_rate_limit,
            get_default_rate_limiter,
        )

        # Create a mock request handler
        handler = Mock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {
            "Authorization": "",
            "X-Forwarded-For": "",
            "X-Real-IP": "",
        }

        # Make requests up to the burst limit (60 for public tier)
        for i in range(60):
            result = check_default_rate_limit(handler)
            assert result.allowed is True, f"Request {i + 1} should be allowed"

        # Next request should be denied
        result = check_default_rate_limit(handler)
        assert result.allowed is False
        assert result.retry_after > 0

    def test_rate_limit_result_includes_metadata(self):
        """Test that rate limit result includes proper metadata."""
        from aragora.server.middleware.rate_limit import check_default_rate_limit

        handler = Mock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {}

        result = check_default_rate_limit(handler)

        assert result.allowed is True
        assert result.limit == 30  # Public tier
        assert result.remaining >= 0
        assert "default:public:" in result.key


class TestRateLimitTierDetection:
    """Tests for authentication tier detection."""

    def test_unauthenticated_gets_public_tier(self):
        """Test unauthenticated requests get public tier."""
        from aragora.server.middleware.rate_limit import determine_auth_tier

        handler = Mock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {}

        tier, _ = determine_auth_tier(handler)
        assert tier == "public"

    def test_authenticated_gets_auth_tier(self):
        """Test authenticated users get authenticated tier."""
        from aragora.server.middleware.rate_limit import determine_auth_tier

        handler = Mock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {"Authorization": "Bearer test"}

        mock_auth = Mock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "user123"
        mock_auth.role = "member"
        mock_auth.roles = []
        mock_auth.permissions = []
        mock_auth.is_admin = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth,
        ):
            tier, client_key = determine_auth_tier(handler)

        assert tier == "authenticated"
        assert client_key == "user123"

    def test_admin_gets_admin_tier(self):
        """Test admin users get admin tier."""
        from aragora.server.middleware.rate_limit import determine_auth_tier

        handler = Mock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {"Authorization": "Bearer admin_token"}

        mock_auth = Mock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "admin123"
        mock_auth.role = "admin"
        mock_auth.roles = ["admin"]
        mock_auth.permissions = []
        mock_auth.is_admin = True

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_auth,
        ):
            tier, client_key = determine_auth_tier(handler)

        assert tier == "admin"
        assert client_key == "admin123"


class TestHandlerRateLimitDecoratorDetection:
    """Tests for detecting explicit rate limit decorators on handlers."""

    def test_detects_rate_limited_method(self):
        """Test detection of @rate_limit decorated methods."""
        from aragora.server.middleware.rate_limit import (
            rate_limit,
            handler_has_rate_limit_decorator,
        )

        class TestHandler:
            @rate_limit(requests_per_minute=30)
            def handle(self, path, query_params, handler):
                pass

        handler = TestHandler()
        assert handler_has_rate_limit_decorator(handler, "handle") is True

    def test_detects_non_rate_limited_method(self):
        """Test detection returns False for non-decorated methods."""
        from aragora.server.middleware.rate_limit import handler_has_rate_limit_decorator

        class TestHandler:
            def handle(self, path, query_params, handler):
                pass

        handler = TestHandler()
        assert handler_has_rate_limit_decorator(handler, "handle") is False
