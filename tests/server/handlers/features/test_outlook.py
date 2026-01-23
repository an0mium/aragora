"""
Tests for Outlook/Microsoft 365 Handler.

Tests cover basic handler creation and route definitions.
"""

import pytest

from aragora.server.handlers.features.outlook import OutlookHandler


class TestOutlookHandler:
    """Tests for OutlookHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = OutlookHandler(ctx={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(OutlookHandler, "ROUTES")
        routes = OutlookHandler.ROUTES

        # OAuth routes
        assert "/api/v1/outlook/oauth/url" in routes
        assert "/api/v1/outlook/oauth/callback" in routes

        # Folder and message routes
        assert "/api/v1/outlook/folders" in routes
        assert "/api/v1/outlook/messages" in routes
        assert "/api/v1/outlook/send" in routes
        assert "/api/v1/outlook/reply" in routes
        assert "/api/v1/outlook/search" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = OutlookHandler(ctx={})

        # OAuth routes
        assert handler.can_handle("/api/v1/outlook/oauth/url") is True
        assert handler.can_handle("/api/v1/outlook/oauth/callback") is True

        # Data routes
        assert handler.can_handle("/api/v1/outlook/folders") is True
        assert handler.can_handle("/api/v1/outlook/messages") is True
        assert handler.can_handle("/api/v1/outlook/send") is True

        # Dynamic routes
        assert handler.can_handle("/api/v1/outlook/messages/msg123") is True
        assert handler.can_handle("/api/v1/outlook/conversations/conv456") is True
        assert handler.can_handle("/api/v1/outlook/messages/msg123/read") is True
        assert handler.can_handle("/api/v1/outlook/messages/msg123/move") is True

        # Invalid routes
        assert handler.can_handle("/api/v1/gmail/messages") is False
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_handler_has_handle_method(self):
        """Test that handler has handle method."""
        handler = OutlookHandler(ctx={})
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_handler_route_prefixes(self):
        """Test that handler has route prefix definitions."""
        assert hasattr(OutlookHandler, "ROUTE_PREFIXES")
        prefixes = OutlookHandler.ROUTE_PREFIXES
        assert "/api/v1/outlook/messages/" in prefixes
        assert "/api/v1/outlook/conversations/" in prefixes
