"""
Tests for Gmail Labels Handler.

Tests cover basic handler creation and route definitions.
"""

import pytest

from aragora.server.handlers.features.gmail_labels import GmailLabelsHandler


class TestGmailLabelsHandler:
    """Tests for GmailLabelsHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = GmailLabelsHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(GmailLabelsHandler, "ROUTES")
        routes = GmailLabelsHandler.ROUTES
        assert "/api/v1/gmail/labels" in routes
        assert "/api/v1/gmail/filters" in routes

    def test_handler_route_prefixes(self):
        """Test that handler has route prefix definitions."""
        assert hasattr(GmailLabelsHandler, "ROUTE_PREFIXES")
        prefixes = GmailLabelsHandler.ROUTE_PREFIXES
        assert "/api/v1/gmail/labels/" in prefixes
        assert "/api/v1/gmail/messages/" in prefixes
        assert "/api/v1/gmail/filters/" in prefixes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = GmailLabelsHandler(server_context={})

        # Base routes
        assert handler.can_handle("/api/v1/gmail/labels") is True
        assert handler.can_handle("/api/v1/gmail/filters") is True

        # Prefixed routes
        assert handler.can_handle("/api/v1/gmail/labels/abc123") is True
        assert handler.can_handle("/api/v1/gmail/messages/msg123/labels") is True
        assert handler.can_handle("/api/v1/gmail/messages/msg123/read") is True
        assert handler.can_handle("/api/v1/gmail/messages/msg123/star") is True
        assert handler.can_handle("/api/v1/gmail/messages/msg123/archive") is True
        assert handler.can_handle("/api/v1/gmail/messages/msg123/trash") is True
        assert handler.can_handle("/api/v1/gmail/filters/filter123") is True

        # Invalid routes
        assert handler.can_handle("/api/v1/invalid/route") is False
        assert handler.can_handle("/api/v1/gmail/invalid") is False

    def test_handler_has_post_method(self):
        """Test that handler has handle_post method."""
        handler = GmailLabelsHandler(server_context={})
        assert hasattr(handler, "handle_post")
        assert callable(handler.handle_post)

    def test_handler_has_patch_method(self):
        """Test that handler has handle_patch method."""
        handler = GmailLabelsHandler(server_context={})
        assert hasattr(handler, "handle_patch")
        assert callable(handler.handle_patch)

    def test_handler_has_delete_method(self):
        """Test that handler has handle_delete method."""
        handler = GmailLabelsHandler(server_context={})
        assert hasattr(handler, "handle_delete")
        assert callable(handler.handle_delete)
