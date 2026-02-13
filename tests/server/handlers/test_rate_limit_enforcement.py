"""
Tests for rate limit enforcement on protected endpoints.

Covers:
1. Knowledge Mound handler - 100 req/min per user
2. Integration stats endpoint - 30 req/min per user
3. WhatsApp webhook handler - 1000 req/min per IP

Run with:
    python -m pytest tests/server/handlers/test_rate_limit_enforcement.py -v --timeout=30
"""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
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
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reenable_rate_limiting():
    """Re-enable rate limiting for enforcement tests."""
    import sys

    rl_mod = sys.modules.get("aragora.server.handlers.utils.rate_limit")
    if rl_mod and hasattr(rl_mod, "RATE_LIMITING_DISABLED"):
        old_val = rl_mod.RATE_LIMITING_DISABLED
        rl_mod.RATE_LIMITING_DISABLED = False
        yield
        rl_mod.RATE_LIMITING_DISABLED = old_val
    else:
        yield


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    ctx = MagicMock()
    ctx.config = {}
    return ctx


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.client_address = ("192.168.1.100", 12345)
    handler.headers = MagicMock()
    handler.headers.get = lambda k, d=None: None
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_admin_handler():
    """Create mock handler with admin auth context."""
    handler = MagicMock()
    handler.client_address = ("192.168.1.101", 12345)
    handler.headers = MagicMock()
    handler.headers.get = lambda k, d=None: None
    handler.command = "GET"

    # Set up admin auth context
    auth_ctx = MagicMock()
    auth_ctx.user_id = "admin-user-001"
    auth_ctx.permissions = {"*", "admin", "knowledge:read", "knowledge:write"}
    auth_ctx.roles = {"admin"}
    handler._auth_context = auth_ctx

    return handler


@pytest.fixture
def mock_user_handler():
    """Create mock handler with regular user auth context."""
    handler = MagicMock()
    handler.client_address = ("192.168.1.102", 12345)
    handler.headers = MagicMock()
    handler.headers.get = lambda k, d=None: None
    handler.command = "GET"

    # Set up regular user auth context
    auth_ctx = MagicMock()
    auth_ctx.user_id = "user-123"
    auth_ctx.sub = "user-123"
    auth_ctx.permissions = {"knowledge:read"}
    auth_ctx.roles = {"viewer"}
    handler._auth_context = auth_ctx

    return handler


# =============================================================================
# Knowledge Mound Rate Limit Tests
# =============================================================================


class TestKnowledgeMoundRateLimits:
    """Tests for Knowledge Mound handler rate limiting (100 req/min per user)."""

    @pytest.fixture(autouse=True)
    def reset_limiter(self):
        """Reset the rate limiter before each test."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        _knowledge_limiter.clear()
        yield
        _knowledge_limiter.clear()

    def test_rate_limit_is_100_per_minute(self):
        """Verify rate limiter is configured for 100 requests per minute."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        assert _knowledge_limiter.rpm == 100

    def test_allows_requests_under_limit(self):
        """Should allow requests under the 100/min limit."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        for i in range(100):
            assert _knowledge_limiter.is_allowed("user-123") is True

    def test_blocks_requests_over_limit(self):
        """Should block requests over the 100/min limit."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        # Exhaust the limit
        for _ in range(100):
            _knowledge_limiter.is_allowed("user-123")

        # Next request should be blocked
        assert _knowledge_limiter.is_allowed("user-123") is False

    def test_separate_limits_per_user(self):
        """Should maintain separate limits per user."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        # Exhaust limit for user-1
        for _ in range(100):
            _knowledge_limiter.is_allowed("user-1")

        # user-1 is rate limited
        assert _knowledge_limiter.is_allowed("user-1") is False

        # user-2 is not rate limited
        assert _knowledge_limiter.is_allowed("user-2") is True

    def test_rate_limit_headers_in_response(self, mock_server_context, mock_user_handler):
        """Should return rate limit headers when limit exceeded."""
        from aragora.server.handlers.knowledge_base.mound.handler import (
            KnowledgeMoundHandler,
            _knowledge_limiter,
        )

        handler = KnowledgeMoundHandler(mock_server_context)

        # Set up auth context on handler for rate key extraction
        mock_user_handler._auth_context = MagicMock()
        mock_user_handler._auth_context.user_id = "rate-test-user"
        mock_user_handler._auth_context.sub = "rate-test-user"

        # Mock require_auth_or_error to return user
        mock_user = MagicMock()
        mock_user.user_id = "rate-test-user"
        mock_user.permissions = {"knowledge:read"}
        handler.require_auth_or_error = MagicMock(return_value=(mock_user, None))

        # Exhaust the limit for this specific user
        for _ in range(100):
            _knowledge_limiter.is_allowed("rate-test-user")

        # Make request after limit is exhausted
        result = handler.handle(
            "/api/v1/knowledge/mound/stats",
            {},
            mock_user_handler,
        )

        # Should return 429 with rate limit headers
        assert result is not None
        assert result.status_code == 429
        assert result.headers is not None
        assert "X-RateLimit-Limit" in result.headers
        assert result.headers["X-RateLimit-Limit"] == "100"
        assert "Retry-After" in result.headers

    def test_uses_user_id_for_rate_key(self, mock_server_context, mock_user_handler):
        """Should use user ID from auth context as rate limit key."""
        from aragora.server.handlers.knowledge_base.mound.handler import (
            KnowledgeMoundHandler,
            _knowledge_limiter,
        )

        handler = KnowledgeMoundHandler(mock_server_context)

        # Mock methods
        mock_user = MagicMock()
        mock_user.user_id = "auth-user-id"
        mock_user.permissions = {"knowledge:read"}
        handler.require_auth_or_error = MagicMock(return_value=(mock_user, None))

        # Make a request - this should use auth_user_id as the rate key
        with patch.object(handler, "_get_mound", return_value=MagicMock()):
            with patch.object(
                handler, "_handle_mound_stats", return_value=MagicMock(status_code=200)
            ):
                handler.handle(
                    "/api/v1/knowledge/mound/stats",
                    {},
                    mock_user_handler,
                )

        # Check that the user ID was used for rate limiting
        # The user_id from auth context should have been used
        remaining = _knowledge_limiter.get_remaining(
            "user-123"
        )  # from mock_user_handler._auth_context
        assert remaining == 99  # One request was made


# =============================================================================
# Integration Stats Rate Limit Tests
# =============================================================================


class TestIntegrationStatsRateLimits:
    """Tests for Integration stats endpoint rate limiting (30 req/min per user)."""

    @pytest.fixture(autouse=True)
    def reset_limiter(self):
        """Reset the rate limiter before each test."""
        from aragora.server.handlers.integration_management import _stats_limiter

        _stats_limiter.clear()
        yield
        _stats_limiter.clear()

    def test_rate_limit_is_30_per_minute(self):
        """Verify stats rate limiter is configured for 30 requests per minute."""
        from aragora.server.handlers.integration_management import _stats_limiter

        assert _stats_limiter.rpm == 30

    def test_allows_requests_under_limit(self):
        """Should allow requests under the 30/min limit."""
        from aragora.server.handlers.integration_management import _stats_limiter

        for i in range(30):
            assert _stats_limiter.is_allowed("tenant-001") is True

    def test_blocks_requests_over_limit(self):
        """Should block requests over the 30/min limit."""
        from aragora.server.handlers.integration_management import _stats_limiter

        # Exhaust the limit
        for _ in range(30):
            _stats_limiter.is_allowed("tenant-001")

        # Next request should be blocked
        assert _stats_limiter.is_allowed("tenant-001") is False

    def test_separate_limits_per_tenant(self):
        """Should maintain separate limits per tenant."""
        from aragora.server.handlers.integration_management import _stats_limiter

        # Exhaust limit for tenant-1
        for _ in range(30):
            _stats_limiter.is_allowed("tenant-1")

        # tenant-1 is rate limited
        assert _stats_limiter.is_allowed("tenant-1") is False

        # tenant-2 is not rate limited
        assert _stats_limiter.is_allowed("tenant-2") is True

    @pytest.mark.asyncio
    async def test_stats_endpoint_returns_429_when_rate_limited(self, mock_server_context):
        """Should return 429 with rate limit headers when limit exceeded."""
        from aragora.server.handlers.integration_management import (
            IntegrationsHandler,
            _stats_limiter,
        )

        handler = IntegrationsHandler(mock_server_context)

        # Mock stores
        mock_slack_store = MagicMock()
        mock_slack_store.get_stats.return_value = {"active_workspaces": 5}
        mock_teams_store = MagicMock()
        mock_teams_store.get_stats.return_value = {"active_workspaces": 3}

        handler._slack_store = mock_slack_store
        handler._teams_store = mock_teams_store

        # Exhaust the limit for tenant-id
        tenant_id = "rate-test-tenant"
        for _ in range(30):
            _stats_limiter.is_allowed(tenant_id)

        # Mock handler for IP fallback
        mock_req_handler = MagicMock()
        mock_req_handler.client_address = ("192.168.1.200", 12345)
        mock_req_handler.headers = MagicMock()
        mock_req_handler.headers.get = lambda k, d=None: None

        # Make request after limit is exhausted
        result = await handler._get_stats(tenant_id, mock_req_handler)

        # Should return 429 with rate limit headers
        assert result is not None
        assert result.status_code == 429
        assert result.headers is not None
        assert "X-RateLimit-Limit" in result.headers
        assert result.headers["X-RateLimit-Limit"] == "30"
        assert "Retry-After" in result.headers


# =============================================================================
# WhatsApp Webhook Rate Limit Tests
# =============================================================================


class TestWhatsAppWebhookRateLimits:
    """Tests for WhatsApp webhook handler rate limiting (1000 req/min per IP).

    Note: The WhatsApp handler's rate limit decorator is evaluated at module import
    time, which means the limiter is created when the module is first loaded.
    To test the correct RPM value, we need to ensure the module is reloaded
    after clearing the limiters registry.
    """

    @pytest.fixture(autouse=True)
    def reset_limiters_and_reload_whatsapp(self):
        """Reset rate limiters and reload WhatsApp module before each test.

        This is necessary because the rate_limit decorator creates the limiter
        at module import time, so we need to:
        1. Clear the limiters registry
        2. Remove the WhatsApp module from sys.modules
        3. Reimport to get fresh decorator evaluation
        """
        import sys
        import importlib

        # Import the rate_limit module to access _limiters
        rate_limit_mod = importlib.import_module("aragora.server.handlers.utils.rate_limit")

        # Clear all limiters
        rate_limit_mod.clear_all_limiters()
        with rate_limit_mod._limiters_lock:
            rate_limit_mod._limiters.clear()

        # Remove WhatsApp module from cache to force reimport
        whatsapp_modules = [k for k in sys.modules.keys() if "whatsapp" in k.lower()]
        for mod_name in whatsapp_modules:
            del sys.modules[mod_name]

        yield

        # Cleanup
        rate_limit_mod.clear_all_limiters()

    def test_whatsapp_handler_has_high_rate_limit(self, mock_server_context):
        """Verify WhatsApp handler uses the whatsapp_webhook limiter with 1000 rpm."""
        import importlib
        import aragora.server.handlers.bots.whatsapp as whatsapp_mod

        whatsapp_mod = importlib.reload(whatsapp_mod)

        handler = whatsapp_mod.WhatsAppHandler(mock_server_context)

        # Check that handle_post method has rate limit decorator
        assert hasattr(handler.handle_post, "_rate_limited")
        assert handler.handle_post._rate_limited is True

        # The limiter should be the whatsapp_webhook limiter with 1000 rpm
        limiter = handler.handle_post._rate_limiter
        assert limiter.rpm == 1000

    def test_allows_high_volume_webhook_requests(self, mock_server_context):
        """Should allow high volume of webhook requests (up to 1000/min)."""
        import importlib
        import aragora.server.handlers.bots.whatsapp as whatsapp_mod

        whatsapp_mod = importlib.reload(whatsapp_mod)

        handler = whatsapp_mod.WhatsAppHandler(mock_server_context)

        # Get the limiter from the decorated method
        limiter = handler.handle_post._rate_limiter

        # Should allow 1000 requests
        allowed_count = 0
        for _ in range(1000):
            if limiter.is_allowed("webhook-source-ip"):
                allowed_count += 1

        assert allowed_count == 1000

    def test_blocks_after_1000_requests(self, mock_server_context):
        """Should block requests after 1000/min limit exceeded."""
        import importlib
        import aragora.server.handlers.bots.whatsapp as whatsapp_mod

        whatsapp_mod = importlib.reload(whatsapp_mod)

        handler = whatsapp_mod.WhatsAppHandler(mock_server_context)

        # Get the limiter from the decorated method
        limiter = handler.handle_post._rate_limiter

        # Exhaust the limit
        for _ in range(1000):
            limiter.is_allowed("webhook-source-ip-2")

        # Next request should be blocked
        assert limiter.is_allowed("webhook-source-ip-2") is False

    def test_separate_limits_per_ip(self, mock_server_context):
        """Should maintain separate limits per IP for webhooks."""
        import importlib
        import aragora.server.handlers.bots.whatsapp as whatsapp_mod

        whatsapp_mod = importlib.reload(whatsapp_mod)

        handler = whatsapp_mod.WhatsAppHandler(mock_server_context)
        limiter = handler.handle_post._rate_limiter

        # Exhaust limit for one IP
        for _ in range(1000):
            limiter.is_allowed("ip-1")

        # ip-1 is rate limited
        assert limiter.is_allowed("ip-1") is False

        # ip-2 is not rate limited
        assert limiter.is_allowed("ip-2") is True


# =============================================================================
# Rate Limit Headers Tests
# =============================================================================


class TestRateLimitHeaders:
    """Tests for rate limit headers in responses."""

    @pytest.fixture(autouse=True)
    def reset_limiters(self):
        """Reset rate limiters before each test."""
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
        yield
        clear_all_limiters()

    def test_429_response_includes_retry_after(self):
        """Should include Retry-After header in 429 response."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter
        from aragora.server.handlers.base import error_response

        # Exhaust limit
        for _ in range(100):
            _knowledge_limiter.is_allowed("test-user")

        # Get remaining to simulate what handler does
        remaining = _knowledge_limiter.get_remaining("test-user")

        # Create error response with headers
        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": str(remaining),
            "Retry-After": "60",
        }
        result = error_response("Rate limit exceeded", 429, headers=headers)

        assert result.status_code == 429
        assert "Retry-After" in result.headers
        assert result.headers["Retry-After"] == "60"

    def test_rate_limit_remaining_header_is_accurate(self):
        """Should accurately report remaining requests in header."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        # Make 50 requests
        for _ in range(50):
            _knowledge_limiter.is_allowed("remaining-test-user")

        # Check remaining
        remaining = _knowledge_limiter.get_remaining("remaining-test-user")
        assert remaining == 50  # 100 - 50 = 50


# =============================================================================
# Admin Bypass Tests (if applicable)
# =============================================================================


class TestAdminRateLimitBehavior:
    """Tests for admin user rate limit behavior."""

    @pytest.fixture(autouse=True)
    def reset_limiters(self):
        """Reset rate limiters before each test."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        _knowledge_limiter.clear()
        yield
        _knowledge_limiter.clear()

    def test_admin_users_are_still_rate_limited(self):
        """Admin users should still be rate limited (no bypass)."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        # Admin users are subject to the same rate limits
        # (If admin bypass is desired, this test documents current behavior)
        for _ in range(100):
            _knowledge_limiter.is_allowed("admin-user")

        # Admin should still be rate limited
        assert _knowledge_limiter.is_allowed("admin-user") is False

    def test_different_users_have_independent_limits(self):
        """Each user should have their own independent rate limit quota."""
        from aragora.server.handlers.knowledge_base.mound.handler import _knowledge_limiter

        # Exhaust one user's limit
        for _ in range(100):
            _knowledge_limiter.is_allowed("user-a")

        # Other users should be unaffected
        assert _knowledge_limiter.is_allowed("user-b") is True
        assert _knowledge_limiter.is_allowed("user-c") is True
        assert _knowledge_limiter.is_allowed("admin") is True

        # First user should be blocked
        assert _knowledge_limiter.is_allowed("user-a") is False
