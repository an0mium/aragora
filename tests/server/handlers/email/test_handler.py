"""
Tests for EmailHandler routing class.

Covers:
- ROUTES and ROUTE_PREFIXES definitions
- can_handle routing logic
- _get_user_id from auth context
- _get_auth_context from handler ctx
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.email.handler import EmailHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an EmailHandler with minimal ctx."""
    return EmailHandler(ctx={})


@pytest.fixture
def handler_with_auth():
    """Create an EmailHandler with auth context in ctx."""
    auth = MagicMock()
    auth.user_id = "test-user-42"
    return EmailHandler(ctx={"auth_context": auth})


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_has_expected_routes(self, handler):
        assert "/api/v1/email/prioritize" in handler.ROUTES
        assert "/api/v1/email/rank-inbox" in handler.ROUTES
        assert "/api/v1/email/feedback" in handler.ROUTES
        assert "/api/v1/email/feedback/batch" in handler.ROUTES
        assert "/api/v1/email/inbox" in handler.ROUTES
        assert "/api/v1/email/config" in handler.ROUTES
        assert "/api/v1/email/vip" in handler.ROUTES
        assert "/api/v1/email/categorize" in handler.ROUTES
        assert "/api/v1/email/categorize/batch" in handler.ROUTES
        assert "/api/v1/email/categorize/apply-label" in handler.ROUTES
        assert "/api/v1/email/gmail/oauth/url" in handler.ROUTES
        assert "/api/v1/email/gmail/oauth/callback" in handler.ROUTES
        assert "/api/v1/email/gmail/status" in handler.ROUTES
        assert "/api/v1/email/context/boost" in handler.ROUTES

    def test_route_count(self, handler):
        assert len(handler.ROUTES) == 14


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_exact_route_match(self, handler):
        assert handler.can_handle("/api/v1/email/prioritize") is True
        assert handler.can_handle("/api/v1/email/inbox") is True
        assert handler.can_handle("/api/v1/email/vip") is True

    def test_prefix_route_match(self, handler):
        """Dynamic routes like /api/v1/email/context/:email_address."""
        assert handler.can_handle("/api/v1/email/context/user@test.com") is True
        assert handler.can_handle("/api/v1/email/context/another@email.org") is True

    def test_prefix_root_with_trailing_slash_matches(self, handler):
        """Prefix path with trailing slash matches (startswith check)."""
        # "/api/v1/email/context/" startswith the prefix and != prefix.rstrip("/")
        assert handler.can_handle("/api/v1/email/context/") is True

    def test_prefix_root_without_trailing_slash_not_matched(self, handler):
        """Bare prefix path without trailing slash is not in ROUTES."""
        assert handler.can_handle("/api/v1/email/context") is False

    def test_unrelated_path_not_matched(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/email/nonexistent") is False
        assert handler.can_handle("/api/health") is False


# ---------------------------------------------------------------------------
# _get_user_id
# ---------------------------------------------------------------------------


class TestGetUserId:
    def test_returns_user_id_from_auth(self, handler_with_auth):
        assert handler_with_auth._get_user_id() == "test-user-42"

    def test_returns_default_without_auth(self, handler):
        assert handler._get_user_id() == "default"

    def test_returns_default_when_no_user_id_attr(self):
        """Auth context present but no user_id attribute."""
        auth = MagicMock(spec=[])  # no attributes
        h = EmailHandler(ctx={"auth_context": auth})
        assert h._get_user_id() == "default"


# ---------------------------------------------------------------------------
# _get_auth_context
# ---------------------------------------------------------------------------


class TestGetAuthContext:
    def test_returns_auth_context(self, handler_with_auth):
        ctx = handler_with_auth._get_auth_context()
        assert ctx is not None
        assert ctx.user_id == "test-user-42"

    def test_returns_none_without_auth(self, handler):
        assert handler._get_auth_context() is None
