"""Tests for gallery handler endpoints.

Tests the public gallery API endpoints including:
- GET /api/gallery - List public debates
- GET /api/gallery/:debate_id - Get specific debate
- GET /api/gallery/:debate_id/embed - Get embeddable summary
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockHandler:
    """Mock HTTP handler."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


@pytest.fixture
def mock_handler():
    """Create mock handler."""
    return MockHandler()


@pytest.fixture
def temp_nomic_dir():
    """Create temporary nomic directory with replays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir(parents=True)

        # Create sample debate replays
        for i in range(3):
            debate_dir = replays_dir / f"debate_{i}"
            debate_dir.mkdir()

            meta = {
                "debate_id": f"debate_{i}",
                "loop_id": "loop_001",
                "title": f"Test Debate {i}",
                "topic": f"Should we use approach {i}?",
                "agents": ["claude", "gpt-4"],
                "rounds": 3,
                "consensus_reached": i % 2 == 0,
                "winner": "claude" if i % 2 == 0 else None,
                "final_answer": f"The conclusion for debate {i} is that...",
                "created_at": datetime.now().isoformat(),
            }

            with open(debate_dir / "meta.json", "w") as f:
                json.dump(meta, f)

            # Create events file
            events = [
                {"type": "debate_start", "timestamp": datetime.now().isoformat()},
                {"type": "agent_message", "agent": "claude", "content": "Initial position"},
                {"type": "debate_end", "timestamp": datetime.now().isoformat()},
            ]
            with open(debate_dir / "events.jsonl", "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

        yield nomic_dir


@pytest.fixture
def gallery_handler(temp_nomic_dir):
    """Create GalleryHandler for testing."""
    from aragora.server.handlers.gallery import GalleryHandler

    ctx = {"nomic_dir": temp_nomic_dir}
    handler = GalleryHandler(ctx)
    return handler


class TestGalleryHandlerRouting:
    """Test routing logic for gallery handler."""

    def test_can_handle_list(self):
        """Test can_handle for /api/gallery."""
        from aragora.server.handlers.gallery import GalleryHandler

        handler = GalleryHandler({})
        assert handler.can_handle("/api/v1/gallery") is True

    def test_can_handle_debate(self):
        """Test can_handle for /api/gallery/:id."""
        from aragora.server.handlers.gallery import GalleryHandler

        handler = GalleryHandler({})
        assert handler.can_handle("/api/v1/gallery/abc123") is True
        assert handler.can_handle("/api/v1/gallery/debate_001") is True

    def test_can_handle_embed(self):
        """Test can_handle for /api/gallery/:id/embed."""
        from aragora.server.handlers.gallery import GalleryHandler

        handler = GalleryHandler({})
        assert handler.can_handle("/api/v1/gallery/abc123/embed") is True

    def test_cannot_handle_invalid(self):
        """Test can_handle rejects invalid paths."""
        from aragora.server.handlers.gallery import GalleryHandler

        handler = GalleryHandler({})
        assert handler.can_handle("/api/v1/other") is False
        assert handler.can_handle("/api/v1/debates") is False


class TestGalleryHandlerList:
    """Test /api/gallery endpoint."""

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_list_debates_success(self, mock_limiter, gallery_handler, mock_handler):
        """Test successful debate listing."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle("/api/gallery", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "debates" in body
        assert body["total"] == 3
        assert body["limit"] == 20
        assert body["offset"] == 0

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_list_debates_with_pagination(self, mock_limiter, gallery_handler, mock_handler):
        """Test debate listing with pagination."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle(
            "/api/gallery",
            {"limit": ["2"], "offset": ["1"]},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["limit"] == 2
        assert body["offset"] == 1

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_list_debates_limit_capped(self, mock_limiter, gallery_handler, mock_handler):
        """Test debate listing caps limit at 100."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle(
            "/api/gallery",
            {"limit": ["500"]},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["limit"] == 100  # Capped at max

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_list_debates_agent_filter(self, mock_limiter, gallery_handler, mock_handler):
        """Test debate listing with agent filter."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle(
            "/api/gallery",
            {"agent": ["claude"]},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        # All test debates have claude as agent
        assert body["total"] == 3


class TestGalleryHandlerGetDebate:
    """Test /api/gallery/:id endpoint."""

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_get_debate_not_found(self, mock_limiter, gallery_handler, mock_handler):
        """Test get debate with invalid ID returns 404."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle(
            "/api/gallery/nonexistent123",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 404
        body = parse_body(result)
        assert "not found" in body["error"].lower()

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_get_debate_success(self, mock_limiter, gallery_handler, mock_handler, temp_nomic_dir):
        """Test successful debate retrieval by stable ID."""
        mock_limiter.is_allowed.return_value = True

        # Generate the stable ID for debate_0
        from aragora.server.handlers.gallery import generate_stable_id

        stable_id = generate_stable_id("debate_0", "loop_001")

        result = gallery_handler.handle(
            f"/api/gallery/{stable_id}",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["id"] == stable_id
        assert body["debate_id"] == "debate_0"
        assert "events" in body
        assert len(body["events"]) == 3


class TestGalleryHandlerEmbed:
    """Test /api/gallery/:id/embed endpoint."""

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_embed_not_found(self, mock_limiter, gallery_handler, mock_handler):
        """Test embed with invalid ID returns 404."""
        mock_limiter.is_allowed.return_value = True

        result = gallery_handler.handle(
            "/api/gallery/nonexistent123/embed",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 404

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_embed_success(self, mock_limiter, gallery_handler, mock_handler, temp_nomic_dir):
        """Test successful embed retrieval."""
        mock_limiter.is_allowed.return_value = True

        from aragora.server.handlers.gallery import generate_stable_id

        stable_id = generate_stable_id("debate_0", "loop_001")

        result = gallery_handler.handle(
            f"/api/gallery/{stable_id}/embed",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["id"] == stable_id
        assert "title" in body
        assert "preview" in body
        assert "embed_url" in body
        assert "full_url" in body
        assert len(body["preview"]) <= 300  # Preview is truncated


class TestGalleryHandlerRateLimiting:
    """Test rate limiting for gallery endpoints."""

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_rate_limit_exceeded(self, mock_limiter, gallery_handler, mock_handler):
        """Test rate limit exceeded response."""
        mock_limiter.is_allowed.return_value = False

        result = gallery_handler.handle("/api/gallery", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429
        body = parse_body(result)
        assert "rate limit" in body["error"].lower()


class TestGalleryHandlerNoReplays:
    """Test behavior when no replays exist."""

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_empty_gallery(self, mock_limiter, mock_handler):
        """Test empty gallery when no replays exist."""
        from aragora.server.handlers.gallery import GalleryHandler

        mock_limiter.is_allowed.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            # Don't create replays directory

            handler = GalleryHandler({"nomic_dir": nomic_dir})

            result = handler.handle("/api/gallery", {}, mock_handler)

            assert result is not None
            assert result.status_code == 200
            body = parse_body(result)
            assert body["debates"] == []
            assert body["total"] == 0

    @patch("aragora.server.handlers.gallery._gallery_limiter")
    def test_no_nomic_dir(self, mock_limiter, mock_handler):
        """Test behavior when nomic_dir is not configured."""
        from aragora.server.handlers.gallery import GalleryHandler

        mock_limiter.is_allowed.return_value = True

        handler = GalleryHandler({"nomic_dir": None})

        result = handler.handle("/api/gallery", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["debates"] == []


class TestGenerateStableId:
    """Test stable ID generation."""

    def test_stable_id_consistency(self):
        """Test that same input produces same stable ID."""
        from aragora.server.handlers.gallery import generate_stable_id

        id1 = generate_stable_id("debate_1", "loop_001")
        id2 = generate_stable_id("debate_1", "loop_001")
        assert id1 == id2

    def test_stable_id_uniqueness(self):
        """Test that different inputs produce different IDs."""
        from aragora.server.handlers.gallery import generate_stable_id

        id1 = generate_stable_id("debate_1", "loop_001")
        id2 = generate_stable_id("debate_2", "loop_001")
        id3 = generate_stable_id("debate_1", "loop_002")
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_stable_id_without_loop(self):
        """Test stable ID generation without loop_id."""
        from aragora.server.handlers.gallery import generate_stable_id

        id1 = generate_stable_id("debate_1", None)
        id2 = generate_stable_id("debate_1", None)
        assert id1 == id2
        assert len(id1) == 12  # SHA256 truncated to 12 chars
