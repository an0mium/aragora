"""
Tests for E-commerce Platform Handler.

Tests cover basic platform configuration and handler creation.
"""

import pytest

from aragora.server.handlers.features.ecommerce import (
    EcommerceHandler,
    SUPPORTED_PLATFORMS,
)


class TestSupportedPlatforms:
    """Tests for e-commerce platform configuration."""

    def test_all_platforms_defined(self):
        """Test that e-commerce platforms are configured."""
        # At least shopify should be defined
        assert "shopify" in SUPPORTED_PLATFORMS
        assert len(SUPPORTED_PLATFORMS) >= 1

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config
            assert "description" in config
            assert "features" in config


class TestEcommerceHandler:
    """Tests for EcommerceHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = EcommerceHandler(server_context={})
        assert handler is not None

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = EcommerceHandler(server_context={})
        assert hasattr(handler, "handle_get") or hasattr(handler, "handle_request")
