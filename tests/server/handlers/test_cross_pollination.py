"""Tests for the CrossPollination handlers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.cross_pollination import (
    CrossPollinationStatsHandler,
    CrossPollinationSubscribersHandler,
)


class TestCrossPollinationStatsHandler:
    """Tests for CrossPollinationStatsHandler."""

    def _make_handler(self, ctx: dict | None = None) -> CrossPollinationStatsHandler:
        return CrossPollinationStatsHandler(ctx=ctx)

    # -------------------------------------------------------------------------
    # ROUTES tests
    # -------------------------------------------------------------------------

    def test_routes_defined(self):
        handler = self._make_handler()
        assert "/api/v1/cross-pollination/stats" in handler.ROUTES

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_init_with_context(self):
        ctx = {"key": "value"}
        handler = CrossPollinationStatsHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_init_without_context(self):
        handler = CrossPollinationStatsHandler()
        assert handler.ctx == {}

    # -------------------------------------------------------------------------
    # GET tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_returns_stats(self):
        handler = self._make_handler()

        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "sub1": {"events_processed": 10, "events_failed": 1, "enabled": True},
            "sub2": {"events_processed": 20, "events_failed": 0, "enabled": True},
        }

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await handler.get.__wrapped__(handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_calculates_summary(self):
        handler = self._make_handler()

        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "sub1": {"events_processed": 10, "events_failed": 1, "enabled": True},
            "sub2": {"events_processed": 20, "events_failed": 2, "enabled": False},
        }

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await handler.get.__wrapped__(handler)

        # Check that summary is calculated
        import json

        body = json.loads(result.body)
        assert body["summary"]["total_subscribers"] == 2
        assert body["summary"]["enabled_subscribers"] == 1
        assert body["summary"]["total_events_processed"] == 30
        assert body["summary"]["total_events_failed"] == 3

    @pytest.mark.asyncio
    async def test_get_handles_import_error(self):
        handler = self._make_handler()

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=ImportError("Module not found"),
        ):
            result = await handler.get.__wrapped__(handler)

        assert result is not None
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_handles_generic_error(self):
        handler = self._make_handler()

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await handler.get.__wrapped__(handler)

        assert result is not None
        assert result.status_code == 500


class TestCrossPollinationSubscribersHandler:
    """Tests for CrossPollinationSubscribersHandler."""

    def _make_handler(self) -> CrossPollinationSubscribersHandler:
        return CrossPollinationSubscribersHandler(server_context={})

    # -------------------------------------------------------------------------
    # ROUTES tests
    # -------------------------------------------------------------------------

    def test_routes_defined(self):
        handler = self._make_handler()
        assert "/api/v1/cross-pollination/subscribers" in handler.ROUTES

    # -------------------------------------------------------------------------
    # GET tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_returns_subscribers(self):
        handler = self._make_handler()

        mock_event_type = MagicMock()
        mock_event_type.value = "debate_complete"

        mock_handler = MagicMock(__name__="on_debate_complete")

        mock_manager = MagicMock()
        mock_manager._subscribers = {
            mock_event_type: [("subscriber1", mock_handler)],
        }

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await handler.get.__wrapped__(handler)

        assert result is not None
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["subscribers"][0]["name"] == "subscriber1"
        assert body["subscribers"][0]["event_type"] == "debate_complete"

    @pytest.mark.asyncio
    async def test_get_handles_handler_without_name(self):
        handler = self._make_handler()

        mock_event_type = MagicMock()
        mock_event_type.value = "event_type"

        # Handler without __name__ attribute
        def mock_handler(x):
            return x

        mock_manager = MagicMock()
        mock_manager._subscribers = {
            mock_event_type: [("anon_sub", mock_handler)],
        }

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await handler.get.__wrapped__(handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_empty_subscribers(self):
        handler = self._make_handler()

        mock_manager = MagicMock()
        mock_manager._subscribers = {}

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await handler.get.__wrapped__(handler)

        import json

        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["subscribers"] == []


class TestCrossPollinationRateLimiter:
    """Tests for cross-pollination rate limiting."""

    def test_rate_limiter_exists(self):
        from aragora.server.handlers.cross_pollination import _cross_pollination_limiter

        assert _cross_pollination_limiter is not None
        assert hasattr(_cross_pollination_limiter, "is_allowed")

    def test_rate_limiter_config(self):
        from aragora.server.handlers.cross_pollination import _cross_pollination_limiter

        assert _cross_pollination_limiter.rpm == 60
