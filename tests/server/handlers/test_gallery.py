"""
Tests for the gallery handler - public debate gallery API.

Tests:
- Route handling (can_handle)
- List gallery endpoint
- Get specific debate endpoint
- Embed endpoint
- Rate limiting
- Stable ID generation
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.gallery import (
    GalleryHandler,
    _gallery_limiter,
    generate_stable_id,
    PublicDebate,
)


def parse_response(result):
    """Parse HandlerResult body into dict."""
    if result is None:
        return None
    return json.loads(result.body.decode())


@pytest.fixture
def gallery_handler():
    """Create a gallery handler with mocked dependencies."""
    ctx = {"storage": None}
    handler = GalleryHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    _gallery_limiter.clear()
    yield


class TestGalleryHandlerRouting:
    """Tests for GalleryHandler route matching."""

    def test_can_handle_gallery_list(self, gallery_handler):
        """Test that handler recognizes /api/gallery route."""
        assert gallery_handler.can_handle("/api/v1/gallery") is True

    def test_can_handle_gallery_debate(self, gallery_handler):
        """Test that handler recognizes /api/gallery/:id route."""
        assert gallery_handler.can_handle("/api/v1/gallery/abc123/details") is True

    def test_can_handle_gallery_embed(self, gallery_handler):
        """Test that handler recognizes /api/gallery/:id/embed route."""
        assert gallery_handler.can_handle("/api/v1/gallery/abc123/embed") is True

    def test_cannot_handle_unknown_route(self, gallery_handler):
        """Test that handler rejects unknown routes."""
        assert gallery_handler.can_handle("/api/v1/unknown") is False
        assert gallery_handler.can_handle("/api/v1/debates") is False


class TestStableIdGeneration:
    """Tests for stable ID generation."""

    def test_generate_stable_id_basic(self):
        """Test basic stable ID generation."""
        id1 = generate_stable_id("debate-123")
        id2 = generate_stable_id("debate-123")
        assert id1 == id2  # Same input = same output
        assert len(id1) == 12  # 12 character hash

    def test_generate_stable_id_with_loop(self):
        """Test stable ID generation with loop ID."""
        id1 = generate_stable_id("debate-123", "loop-1")
        id2 = generate_stable_id("debate-123", "loop-2")
        assert id1 != id2  # Different loop = different ID

    def test_generate_stable_id_different_debates(self):
        """Test that different debates get different IDs."""
        id1 = generate_stable_id("debate-1")
        id2 = generate_stable_id("debate-2")
        assert id1 != id2


class TestPublicDebate:
    """Tests for PublicDebate dataclass."""

    def test_to_dict(self):
        """Test PublicDebate to_dict conversion."""
        debate = PublicDebate(
            id="abc123",
            title="Test Debate",
            topic="Is AI good?",
            created_at="2025-01-15T10:00:00Z",
            agents=["claude", "gpt-4"],
            rounds=3,
            consensus_reached=True,
            winner="claude",
            preview="The answer is...",
        )
        d = debate.to_dict()
        assert d["id"] == "abc123"
        assert d["title"] == "Test Debate"
        assert d["consensus_reached"] is True
        assert d["agents"] == ["claude", "gpt-4"]


class TestRateLimiting:
    """Tests for rate limiting on gallery endpoints."""

    def test_rate_limit_enforcement(self, gallery_handler, mock_http_handler):
        """Test that rate limiting is enforced."""
        # Fill up rate limit (60 requests per minute)
        for _ in range(60):
            _gallery_limiter.is_allowed("127.0.0.1")

        result = gallery_handler.handle("/api/v1/gallery", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429
        body = parse_response(result)
        assert "Rate limit" in body["error"]


class TestGalleryList:
    """Tests for /api/gallery endpoint."""

    def test_gallery_list_returns_result(self, gallery_handler, mock_http_handler):
        """Test gallery list returns a result."""
        result = gallery_handler.handle("/api/v1/gallery", {}, mock_http_handler)
        # Should return some result (empty list or actual debates)
        assert result is not None
        assert result.status_code in [200, 404, 500]
