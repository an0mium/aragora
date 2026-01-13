"""
Tests for GalleryHandler - public debate archive endpoints.

Tests cover:
- GET /api/gallery - List public debates with pagination
- GET /api/gallery/:debate_id - Get specific debate details
- GET /api/gallery/:debate_id/embed - Get embeddable debate summary
- Stable ID generation
- Replay file parsing
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.gallery import (
    GalleryHandler,
    PublicDebate,
    generate_stable_id,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_nomic_dir():
    """Create a temporary nomic directory with replay files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create replays directory
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir(parents=True)

        # Create sample debate replay 1
        debate1_dir = replays_dir / "debate-001"
        debate1_dir.mkdir()
        (debate1_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "debate-001",
                    "loop_id": "loop-abc",
                    "title": "AI Ethics Discussion",
                    "topic": "Should AI systems be regulated?",
                    "agents": ["claude", "gpt4"],
                    "rounds": 3,
                    "consensus_reached": True,
                    "winner": "claude",
                    "final_answer": "AI systems should have reasonable regulations...",
                    "created_at": "2026-01-10T10:00:00",
                }
            )
        )
        (debate1_dir / "events.jsonl").write_text(
            '{"type": "round_start", "round": 1}\n'
            '{"type": "agent_message", "agent": "claude", "content": "I argue..."}\n'
        )

        # Create sample debate replay 2
        debate2_dir = replays_dir / "debate-002"
        debate2_dir.mkdir()
        (debate2_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "debate-002",
                    "loop_id": "loop-def",
                    "title": "Open Source Benefits",
                    "topic": "Should AI be open-sourced?",
                    "agents": ["claude", "gemini"],
                    "rounds": 2,
                    "consensus_reached": False,
                    "winner": None,
                    "final_answer": "",
                    "created_at": "2026-01-09T10:00:00",
                }
            )
        )

        yield nomic_dir


@pytest.fixture
def gallery_handler(temp_nomic_dir):
    """Create GalleryHandler with test context."""
    ctx = {"nomic_dir": temp_nomic_dir}
    return GalleryHandler(ctx)


@pytest.fixture
def empty_gallery_handler():
    """Create GalleryHandler with no nomic directory."""
    return GalleryHandler({})


# ============================================================================
# PublicDebate Tests
# ============================================================================


class TestPublicDebate:
    """Tests for PublicDebate dataclass."""

    def test_public_debate_creation(self):
        """Test creating a PublicDebate instance."""
        debate = PublicDebate(
            id="abc123",
            title="Test Debate",
            topic="Test topic",
            created_at="2026-01-10T10:00:00",
            agents=["claude", "gpt4"],
            rounds=3,
            consensus_reached=True,
            winner="claude",
            preview="This is a preview...",
        )

        assert debate.id == "abc123"
        assert debate.title == "Test Debate"
        assert debate.consensus_reached is True
        assert debate.winner == "claude"

    def test_public_debate_to_dict(self):
        """Test PublicDebate serialization."""
        debate = PublicDebate(
            id="abc123",
            title="Test Debate",
            topic="Test topic",
            created_at="2026-01-10T10:00:00",
            agents=["claude", "gpt4"],
            rounds=3,
            consensus_reached=True,
            winner="claude",
            preview="Preview text",
        )

        result = debate.to_dict()

        assert result["id"] == "abc123"
        assert result["title"] == "Test Debate"
        assert result["agents"] == ["claude", "gpt4"]
        assert result["consensus_reached"] is True


# ============================================================================
# Stable ID Generation Tests
# ============================================================================


class TestStableIdGeneration:
    """Tests for stable ID generation."""

    def test_generate_stable_id_basic(self):
        """Test basic stable ID generation."""
        stable_id = generate_stable_id("debate-123")

        assert stable_id is not None
        assert len(stable_id) == 12
        assert stable_id.isalnum()

    def test_generate_stable_id_with_loop_id(self):
        """Test stable ID with loop_id."""
        stable_id = generate_stable_id("debate-123", "loop-abc")

        assert stable_id is not None
        assert len(stable_id) == 12

    def test_stable_id_consistency(self):
        """Test that same inputs produce same ID."""
        id1 = generate_stable_id("debate-123", "loop-abc")
        id2 = generate_stable_id("debate-123", "loop-abc")

        assert id1 == id2

    def test_stable_id_uniqueness(self):
        """Test that different inputs produce different IDs."""
        id1 = generate_stable_id("debate-123", "loop-abc")
        id2 = generate_stable_id("debate-456", "loop-abc")
        id3 = generate_stable_id("debate-123", "loop-def")

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3


# ============================================================================
# GalleryHandler Route Tests
# ============================================================================


class TestGalleryHandlerRoutes:
    """Tests for GalleryHandler routing."""

    def test_can_handle_gallery_list(self, gallery_handler):
        """Test handler recognizes gallery list route."""
        assert gallery_handler.can_handle("/api/gallery")

    def test_can_handle_specific_debate(self, gallery_handler):
        """Test handler recognizes specific debate route."""
        assert gallery_handler.can_handle("/api/gallery/abc123def")

    def test_can_handle_embed_route(self, gallery_handler):
        """Test handler recognizes embed route."""
        assert gallery_handler.can_handle("/api/gallery/abc123def/embed")

    def test_cannot_handle_unknown_route(self, gallery_handler):
        """Test handler rejects unknown routes."""
        assert not gallery_handler.can_handle("/api/gallery/")  # trailing slash only
        assert not gallery_handler.can_handle("/api/debates")
        assert not gallery_handler.can_handle("/api/public")


# ============================================================================
# List Debates Tests
# ============================================================================


class TestListDebates:
    """Tests for listing public debates."""

    def test_list_debates_returns_debates(self, gallery_handler):
        """Test listing debates returns results."""
        result = gallery_handler.handle("/api/gallery", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "debates" in data
        assert "total" in data
        assert len(data["debates"]) == 2

    def test_list_debates_includes_metadata(self, gallery_handler):
        """Test debate listings include required metadata."""
        result = gallery_handler.handle("/api/gallery", {}, None)
        data = json.loads(result.body)

        debate = data["debates"][0]
        assert "id" in debate
        assert "title" in debate
        assert "topic" in debate
        assert "agents" in debate
        assert "consensus_reached" in debate

    def test_list_debates_with_limit(self, gallery_handler):
        """Test pagination with limit parameter."""
        result = gallery_handler.handle("/api/gallery", {"limit": ["1"]}, None)
        data = json.loads(result.body)

        assert len(data["debates"]) == 1
        assert data["limit"] == 1

    def test_list_debates_with_offset(self, gallery_handler):
        """Test pagination with offset parameter."""
        result = gallery_handler.handle("/api/gallery", {"offset": ["1"]}, None)
        data = json.loads(result.body)

        assert data["offset"] == 1
        assert len(data["debates"]) == 1

    def test_list_debates_with_agent_filter(self, gallery_handler):
        """Test filtering by agent name."""
        result = gallery_handler.handle("/api/gallery", {"agent": ["gemini"]}, None)
        data = json.loads(result.body)

        # Only debate-002 has gemini
        assert len(data["debates"]) == 1
        assert "gemini" in data["debates"][0]["agents"]

    def test_list_debates_empty_directory(self, empty_gallery_handler):
        """Test listing when no debates exist."""
        result = empty_gallery_handler.handle("/api/gallery", {}, None)
        data = json.loads(result.body)

        assert data["debates"] == []
        assert data["total"] == 0

    def test_list_debates_max_limit_enforced(self, gallery_handler):
        """Test that max limit of 100 is enforced."""
        result = gallery_handler.handle("/api/gallery", {"limit": ["1000"]}, None)
        data = json.loads(result.body)

        # Should cap at 100
        assert data["limit"] == 100


# ============================================================================
# Get Specific Debate Tests
# ============================================================================


class TestGetDebate:
    """Tests for getting specific debate details."""

    def test_get_debate_success(self, gallery_handler, temp_nomic_dir):
        """Test retrieving a specific debate."""
        # First get the stable ID
        stable_id = generate_stable_id("debate-001", "loop-abc")

        result = gallery_handler.handle(f"/api/gallery/{stable_id}", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["id"] == stable_id
        assert data["title"] == "AI Ethics Discussion"
        assert data["topic"] == "Should AI systems be regulated?"
        assert "events" in data

    def test_get_debate_includes_events(self, gallery_handler):
        """Test that debate includes event history."""
        stable_id = generate_stable_id("debate-001", "loop-abc")

        result = gallery_handler.handle(f"/api/gallery/{stable_id}", {}, None)
        data = json.loads(result.body)

        assert "events" in data
        assert len(data["events"]) == 2  # We created 2 events

    def test_get_debate_not_found(self, gallery_handler):
        """Test 404 for non-existent debate."""
        result = gallery_handler.handle("/api/gallery/nonexistent123", {}, None)

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()

    def test_get_debate_no_nomic_dir(self, empty_gallery_handler):
        """Test when nomic directory not configured."""
        result = empty_gallery_handler.handle("/api/gallery/abc123", {}, None)

        assert result.status_code == 404


# ============================================================================
# Get Embed Tests
# ============================================================================


class TestGetEmbed:
    """Tests for getting embeddable debate summary."""

    def test_get_embed_success(self, gallery_handler):
        """Test getting embeddable summary."""
        stable_id = generate_stable_id("debate-001", "loop-abc")

        result = gallery_handler.handle(f"/api/gallery/{stable_id}/embed", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["id"] == stable_id
        assert "title" in data
        assert "preview" in data
        assert "embed_url" in data
        assert "full_url" in data

    def test_embed_has_truncated_fields(self, gallery_handler):
        """Test that embed has truncated fields for sharing."""
        stable_id = generate_stable_id("debate-001", "loop-abc")

        result = gallery_handler.handle(f"/api/gallery/{stable_id}/embed", {}, None)
        data = json.loads(result.body)

        # Topic should be truncated to 200 chars
        assert len(data["topic"]) <= 200
        # Preview should be truncated to 300 chars
        assert len(data["preview"]) <= 300

    def test_embed_includes_urls(self, gallery_handler):
        """Test that embed includes share URLs."""
        stable_id = generate_stable_id("debate-001", "loop-abc")

        result = gallery_handler.handle(f"/api/gallery/{stable_id}/embed", {}, None)
        data = json.loads(result.body)

        assert data["embed_url"] == f"/api/gallery/{stable_id}/embed"
        assert data["full_url"] == f"/api/gallery/{stable_id}"

    def test_get_embed_not_found(self, gallery_handler):
        """Test 404 for non-existent debate embed."""
        result = gallery_handler.handle("/api/gallery/nonexistent/embed", {}, None)

        assert result.status_code == 404


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestGalleryErrorHandling:
    """Tests for error handling in gallery handler."""

    def test_malformed_meta_json_skipped(self, temp_nomic_dir, gallery_handler):
        """Test that malformed meta.json files are skipped."""
        # Create a malformed replay
        bad_dir = temp_nomic_dir / "replays" / "bad-debate"
        bad_dir.mkdir()
        (bad_dir / "meta.json").write_text("{ invalid json }")

        # Should still work, just skip the bad one
        result = gallery_handler.handle("/api/gallery", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should have 2 valid debates, skip the bad one
        assert len(data["debates"]) == 2

    def test_missing_meta_fields_handled(self, temp_nomic_dir, gallery_handler):
        """Test handling of meta.json with missing fields."""
        # Create minimal replay
        minimal_dir = temp_nomic_dir / "replays" / "minimal-debate"
        minimal_dir.mkdir()
        (minimal_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "minimal",
                    # Missing most fields
                }
            )
        )

        result = gallery_handler.handle("/api/gallery", {}, None)

        assert result.status_code == 200
        # Should include minimal debate with defaults
        data = json.loads(result.body)
        assert len(data["debates"]) >= 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestGalleryIntegration:
    """Integration tests for gallery handler."""

    def test_roundtrip_list_then_get(self, gallery_handler):
        """Test listing debates then getting specific one."""
        # List debates
        list_result = gallery_handler.handle("/api/gallery", {}, None)
        list_data = json.loads(list_result.body)

        assert len(list_data["debates"]) > 0

        # Get first debate
        first_id = list_data["debates"][0]["id"]
        get_result = gallery_handler.handle(f"/api/gallery/{first_id}", {}, None)

        assert get_result.status_code == 200
        get_data = json.loads(get_result.body)
        assert get_data["id"] == first_id

    def test_embed_matches_full_debate(self, gallery_handler):
        """Test that embed data matches full debate."""
        stable_id = generate_stable_id("debate-001", "loop-abc")

        full_result = gallery_handler.handle(f"/api/gallery/{stable_id}", {}, None)
        embed_result = gallery_handler.handle(f"/api/gallery/{stable_id}/embed", {}, None)

        full_data = json.loads(full_result.body)
        embed_data = json.loads(embed_result.body)

        # IDs should match
        assert full_data["id"] == embed_data["id"]
        # Title should match
        assert full_data["title"] == embed_data["title"]
