"""
Tests for RLM (Recursive Language Model) Handler.

Tests cover basic handler creation, route definitions, and path parsing.
"""

import pytest

from aragora.server.handlers.features.rlm import RLMHandler


class TestRLMHandler:
    """Tests for RLMHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = RLMHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(RLMHandler, "ROUTES")
        routes = RLMHandler.ROUTES
        assert "/api/v1/knowledge/query-rlm" in routes
        assert "/api/v1/rlm/status" in routes
        assert "/api/v1/metrics/rlm" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = RLMHandler(server_context={})

        # Static routes
        assert handler.can_handle("/api/v1/knowledge/query-rlm") is True
        assert handler.can_handle("/api/v1/rlm/status") is True
        assert handler.can_handle("/api/v1/metrics/rlm") is True

        # Dynamic debate routes
        assert handler.can_handle("/api/v1/debates/debate123/query-rlm") is True
        assert handler.can_handle("/api/v1/debates/debate123/compress") is True
        assert handler.can_handle("/api/v1/debates/debate123/context/SUMMARY") is True
        assert handler.can_handle("/api/v1/debates/debate123/refinement-status") is True

        # Invalid routes
        assert handler.can_handle("/api/v1/invalid/route") is False
        assert handler.can_handle("/api/v1/debates/debate123/invalid") is False

    def test_extract_debate_id(self):
        """Test debate ID extraction from path."""
        handler = RLMHandler(server_context={})

        # Valid paths
        assert handler._extract_debate_id("/api/v1/debates/abc123/query-rlm") == "abc123"
        assert handler._extract_debate_id("/api/v1/debates/debate_xyz/compress") == "debate_xyz"
        assert handler._extract_debate_id("/api/v1/debates/test-id/context/SUMMARY") == "test-id"

        # Invalid paths
        assert handler._extract_debate_id("/api/v1/invalid/path") is None
        assert handler._extract_debate_id("/api/v1/debates") is None

    def test_extract_level(self):
        """Test abstraction level extraction from path."""
        handler = RLMHandler(server_context={})

        # Valid paths
        assert handler._extract_level("/api/v1/debates/abc123/context/summary") == "SUMMARY"
        assert handler._extract_level("/api/v1/debates/abc123/context/abstract") == "ABSTRACT"
        assert handler._extract_level("/api/v1/debates/abc123/context/detailed") == "DETAILED"

        # Invalid paths (not enough parts)
        assert handler._extract_level("/api/v1/debates/abc123/context") is None
        assert handler._extract_level("/api/v1/debates/abc123/query-rlm") is None

    def test_handler_has_post_method(self):
        """Test that handler has handle_post method."""
        handler = RLMHandler(server_context={})
        assert hasattr(handler, "handle_post")
        assert callable(handler.handle_post)

    def test_handler_has_handle_method(self):
        """Test that handler has handle method for GET requests."""
        handler = RLMHandler(server_context={})
        assert hasattr(handler, "handle")
        assert callable(handler.handle)
