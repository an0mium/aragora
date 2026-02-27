"""Tests for the Obsidian bidirectional sync service."""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from aragora.connectors.obsidian.sync import (
    ObsidianSyncService,
    SyncConfig,
    SyncResult,
)


@pytest.fixture
def vault(tmp_path):
    """Create a temporary vault with sample notes."""
    # Note with frontmatter
    (tmp_path / "decision.md").write_text(
        "---\n"
        "title: Launch Decision\n"
        "tags: [decision, product]\n"
        "aragora-id: abc-123\n"
        "---\n"
        "\n"
        "# Should we launch?\n"
        "\n"
        "Pros and cons analysis.\n"
    )

    # Note without frontmatter
    (tmp_path / "brainstorm.md").write_text(
        "# Brainstorm\n\n- Feature A\n- Feature B\n- Feature C\n"
    )

    # Empty note
    (tmp_path / "empty.md").write_text("")

    # .obsidian folder (should be ignored)
    (tmp_path / ".obsidian").mkdir()

    return tmp_path


@pytest.fixture
def config(vault):
    return SyncConfig(
        vault_path=str(vault),
        workspace_id="test-workspace",
    )


@pytest.fixture
def service(config):
    return ObsidianSyncService(config)


class TestSyncConfig:
    def test_defaults(self, vault):
        config = SyncConfig(vault_path=str(vault))
        assert config.workspace_id == "default"
        assert config.results_folder == "aragora-results"
        assert config.poll_interval == 5.0
        assert config.max_notes_per_sync == 100


class TestSyncResult:
    def test_defaults(self):
        result = SyncResult(direction="forward")
        assert result.synced == 0
        assert result.failed == 0
        assert result.errors == []


class TestNoteParser:
    """Test the internal note parsing logic."""

    def test_parse_with_frontmatter(self, service, vault):
        content, fm = service._parse_note(str(vault / "decision.md"))
        assert "Should we launch?" in content
        assert fm["title"] == "Launch Decision"
        assert "decision" in fm["tags"]
        assert fm["aragora-id"] == "abc-123"

    def test_parse_without_frontmatter(self, service, vault):
        content, fm = service._parse_note(str(vault / "brainstorm.md"))
        assert "Feature A" in content
        assert fm == {}

    def test_parse_empty_note(self, service, vault):
        content, fm = service._parse_note(str(vault / "empty.md"))
        assert content == ""
        assert fm == {}

    def test_parse_boolean_values(self, tmp_path):
        (tmp_path / "test.md").write_text("---\nconsensus: true\nvetoed: false\n---\nContent")
        svc = ObsidianSyncService(SyncConfig(vault_path=str(tmp_path)))
        _, fm = svc._parse_note(str(tmp_path / "test.md"))
        assert fm["consensus"] is True
        assert fm["vetoed"] is False


class TestForwardSync:
    """Test Obsidian → KnowledgeMound sync."""

    @pytest.mark.asyncio
    async def test_forward_sync_ingests_notes(self, service, vault):
        """Forward sync should call KM ingest for each note."""
        mock_mound = AsyncMock()

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await service.sync_forward()

        # decision.md and brainstorm.md should sync, empty.md should be skipped
        assert result.direction == "forward"
        assert result.synced >= 1
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_forward_sync_skips_empty_notes(self, service, vault):
        """Empty notes should be skipped."""
        mock_mound = AsyncMock()

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await service.sync_forward()

        assert result.skipped >= 1  # empty.md

    @pytest.mark.asyncio
    async def test_forward_sync_handles_km_unavailable(self, service):
        """Should handle KM import failure gracefully."""
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            side_effect=ImportError("No KM"),
        ):
            result = await service.sync_forward()

        # All notes should fail but not crash
        assert result.direction == "forward"

    @pytest.mark.asyncio
    async def test_forward_sync_tag_filter(self, vault):
        """Should filter notes by tags when watch_tags is set."""
        config = SyncConfig(
            vault_path=str(vault),
            watch_tags=["decision"],
        )
        svc = ObsidianSyncService(config)

        mock_mound = AsyncMock()
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await svc.sync_forward()

        # Only decision.md has the "decision" tag
        # brainstorm.md should be skipped (no matching tag)
        assert result.skipped >= 1


class TestReverseSync:
    """Test KnowledgeMound → Obsidian sync."""

    @pytest.mark.asyncio
    async def test_reverse_sync_writes_notes(self, service, vault):
        """Reverse sync should write result notes."""
        mock_mound = AsyncMock()
        mock_result = MagicMock()
        mock_result.id = "result-001"
        mock_result.content = "Pipeline result: all stages passed."
        mock_result.relevance_score = 0.85
        mock_result.metadata = {"title": "Test Result", "tags": ["pipeline"]}
        mock_mound.query.return_value = [mock_result]

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await service.sync_reverse()

        assert result.direction == "reverse"
        assert result.synced == 1

        # Check the written note
        results_dir = vault / "aragora-results"
        assert results_dir.exists()
        notes = list(results_dir.glob("*.md"))
        assert len(notes) == 1

        content = notes[0].read_text()
        assert "aragora-id: result-001" in content
        assert "Pipeline result" in content

    @pytest.mark.asyncio
    async def test_reverse_sync_skips_existing(self, service, vault):
        """Should not overwrite existing result notes."""
        # Pre-create the results dir with an existing note
        # ID "result-001" sanitizes to "result-001" (hyphens are \w-safe)
        results_dir = vault / "aragora-results"
        results_dir.mkdir()
        (results_dir / "result-001.md").write_text("# Existing")

        mock_mound = AsyncMock()
        mock_result = MagicMock()
        mock_result.id = "result-001"
        mock_result.content = "New content"
        mock_result.relevance_score = 0.5
        mock_result.metadata = {}
        mock_mound.query.return_value = [mock_result]

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await service.sync_reverse()

        # Should skip since file already exists
        assert result.skipped >= 1

    @pytest.mark.asyncio
    async def test_reverse_sync_handles_km_unavailable(self, service):
        """Should handle KM unavailability gracefully."""
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            side_effect=ImportError("No KM"),
        ):
            result = await service.sync_reverse()

        assert result.direction == "reverse"
        assert result.synced == 0

    @pytest.mark.asyncio
    async def test_reverse_sync_creates_results_dir(self, service, vault):
        """Should create results directory if it doesn't exist."""
        results_dir = vault / "aragora-results"
        assert not results_dir.exists()

        mock_mound = AsyncMock()
        mock_mound.query.return_value = []

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            await service.sync_reverse()

        assert results_dir.exists()
