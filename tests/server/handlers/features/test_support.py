"""
Tests for Support/Helpdesk Handler.

Tests cover basic platform configuration and handler creation.
"""

import pytest

from aragora.server.handlers.features.support import (
    SupportHandler,
    SUPPORTED_PLATFORMS,
)


class TestSupportedPlatforms:
    """Tests for support platform configuration."""

    def test_platforms_defined(self):
        """Test that support platforms are configured."""
        assert len(SUPPORTED_PLATFORMS) > 0

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config
            assert "features" in config


class TestSupportHandler:
    """Tests for SupportHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = SupportHandler(server_context={})
        assert handler is not None

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = SupportHandler(server_context={})
        assert hasattr(handler, "handle_get") or hasattr(handler, "handle_request")
