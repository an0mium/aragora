"""
Tests for Audio and Podcast Handler.

Tests cover basic handler creation, route definitions, and constants.
"""

import pytest

from aragora.server.handlers.features.audio import (
    AudioHandler,
    MAX_PODCAST_EPISODES,
    PODCAST_AVAILABLE,
)


class TestAudioConstants:
    """Tests for audio module constants."""

    def test_max_podcast_episodes(self):
        """Test max podcast episodes limit."""
        assert MAX_PODCAST_EPISODES == 200

    def test_podcast_available_flag(self):
        """Test that PODCAST_AVAILABLE is a boolean."""
        assert isinstance(PODCAST_AVAILABLE, bool)


class TestAudioHandler:
    """Tests for AudioHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = AudioHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(AudioHandler, "ROUTES")
        routes = AudioHandler.ROUTES
        assert "/audio/*" in routes
        assert "/api/v1/podcast/feed.xml" in routes
        assert "/api/v1/podcast/episodes" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = AudioHandler(server_context={})

        # Audio files
        assert handler.can_handle("/audio/debate123.mp3") is True
        assert handler.can_handle("/audio/test-id.mp3") is True

        # Podcast endpoints
        assert handler.can_handle("/api/v1/podcast/feed.xml") is True
        assert handler.can_handle("/api/v1/podcast/episodes") is True

        # Invalid routes
        assert handler.can_handle("/audio/file.wav") is False
        assert handler.can_handle("/api/v1/invalid/route") is False
        assert handler.can_handle("/api/v1/podcast/invalid") is False

    def test_handler_has_handle_method(self):
        """Test that handler has handle method for GET requests."""
        handler = AudioHandler(server_context={})
        assert hasattr(handler, "handle")
        assert callable(handler.handle)
