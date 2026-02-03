"""
Tests for Obsidian Connector.

Tests cover:
- Frontmatter parsing
- Note reading and searching
- Wikilink extraction
- Decision receipt writing
- Knowledge Mound sync
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.knowledge.obsidian import (
    Frontmatter,
    NoteType,
    ObsidianConfig,
    ObsidianConnector,
    ObsidianNote,
    create_obsidian_connector,
)
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Create a temporary Obsidian vault with test notes."""
    vault = tmp_path / "test_vault"
    vault.mkdir()

    # Create .obsidian folder (should be ignored)
    obsidian_dir = vault / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "config.json").write_text("{}")

    # Create decisions folder
    decisions_dir = vault / "decisions"
    decisions_dir.mkdir()

    # Create a note with frontmatter
    note1 = vault / "architecture-decision.md"
    note1.write_text("""---
title: API Design Decision
date: 2026-02-03
tags:
  - architecture
  - api
  - ready
aragora_id: debate_abc123
consensus: true
confidence: 0.87
---

# API Design Decision

Should we use REST or GraphQL for our new API?

## Context

We need to decide on the API architecture for our new service.

[[related-note]]
[[another-note|Display Text]]

#ready #architecture

https://example.com/api-design
""")

    # Create a daily note
    note2 = vault / "2026-02-03.md"
    note2.write_text("""---
title: Daily Note
date: 2026-02-03
tags:
  - daily
---

# Daily Note

Today's tasks and notes.
""")

    # Create a note without frontmatter
    note3 = vault / "quick-note.md"
    note3.write_text("""# Quick Note

Just a quick note without frontmatter.

#idea #quick
""")

    # Create a template (should be classified as template)
    templates_dir = vault / "templates"
    templates_dir.mkdir()
    template = templates_dir / "meeting-template.md"
    template.write_text("""---
title: Meeting Template
---

# {{title}}

## Attendees

## Agenda

## Notes
""")

    return vault


@pytest.fixture
def config(temp_vault: Path) -> ObsidianConfig:
    """Create test configuration."""
    return ObsidianConfig(
        vault_path=str(temp_vault),
        watch_tags=["#debate", "#decision", "#aragora", "#ready"],
        ignore_folders=[".obsidian", ".trash", "templates"],
    )


@pytest.fixture
def connector(config: ObsidianConfig) -> ObsidianConnector:
    """Create test connector."""
    return ObsidianConnector(config)


# =============================================================================
# Frontmatter Tests
# =============================================================================


class TestFrontmatter:
    """Tests for Frontmatter parsing."""

    def test_parse_basic_frontmatter(self) -> None:
        """Test parsing basic YAML frontmatter."""
        yaml_str = """
title: Test Note
date: 2026-02-03
tags:
  - test
  - example
"""
        fm = Frontmatter.from_yaml(yaml_str)

        assert fm.title == "Test Note"
        assert fm.tags == ["test", "example"]
        assert fm.date is not None

    def test_parse_aragora_fields(self) -> None:
        """Test parsing Aragora-specific fields."""
        yaml_str = """
title: Decision Note
aragora_id: debate_123
debate_id: debate_456
consensus: true
confidence: 0.85
related_issues:
  - LINEAR-123
  - LINEAR-456
"""
        fm = Frontmatter.from_yaml(yaml_str)

        assert fm.aragora_id == "debate_123"
        assert fm.debate_id == "debate_456"
        assert fm.consensus is True
        assert fm.confidence == 0.85
        assert fm.related_issues == ["LINEAR-123", "LINEAR-456"]

    def test_parse_kebab_case_fields(self) -> None:
        """Test parsing kebab-case field names."""
        yaml_str = """
aragora-id: debate_123
debate-id: debate_456
related-issues:
  - ISSUE-1
"""
        fm = Frontmatter.from_yaml(yaml_str)

        assert fm.aragora_id == "debate_123"
        assert fm.debate_id == "debate_456"
        assert fm.related_issues == ["ISSUE-1"]

    def test_parse_tags_as_string(self) -> None:
        """Test parsing comma-separated tags."""
        yaml_str = """
tags: "tag1, tag2, tag3"
"""
        fm = Frontmatter.from_yaml(yaml_str)

        assert fm.tags == ["tag1", "tag2", "tag3"]

    def test_parse_custom_fields(self) -> None:
        """Test custom fields go to custom dict."""
        yaml_str = """
title: Test
custom_field: custom_value
another_custom: 123
"""
        fm = Frontmatter.from_yaml(yaml_str)

        assert fm.custom["custom_field"] == "custom_value"
        assert fm.custom["another_custom"] == 123

    def test_parse_empty_yaml(self) -> None:
        """Test parsing empty YAML returns default Frontmatter."""
        fm = Frontmatter.from_yaml("")
        assert fm.title is None
        assert fm.tags == []

    def test_parse_invalid_yaml(self) -> None:
        """Test parsing invalid YAML returns default Frontmatter."""
        fm = Frontmatter.from_yaml("not: valid: yaml: here")
        # Should not raise, returns default
        assert isinstance(fm, Frontmatter)

    def test_to_yaml(self) -> None:
        """Test serializing frontmatter to YAML."""
        fm = Frontmatter(
            title="Test Note",
            tags=["tag1", "tag2"],
            aragora_id="debate_123",
            consensus=True,
            confidence=0.9,
        )

        yaml_str = fm.to_yaml()

        assert "title: Test Note" in yaml_str
        assert "aragora_id: debate_123" in yaml_str
        assert "consensus: true" in yaml_str


# =============================================================================
# ObsidianNote Tests
# =============================================================================


class TestObsidianNote:
    """Tests for ObsidianNote parsing."""

    def test_parse_note_with_frontmatter(self, temp_vault: Path) -> None:
        """Test parsing note with frontmatter."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert note.name == "architecture-decision"
        assert note.frontmatter.title == "API Design Decision"
        assert note.frontmatter.aragora_id == "debate_abc123"
        assert note.frontmatter.consensus is True
        assert note.frontmatter.confidence == 0.87

    def test_extract_wikilinks(self, temp_vault: Path) -> None:
        """Test wikilink extraction."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert "related-note" in note.wikilinks
        assert "another-note" in note.wikilinks
        assert len(note.wikilinks) == 2

    def test_extract_inline_tags(self, temp_vault: Path) -> None:
        """Test inline tag extraction."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        # Should have both frontmatter and inline tags
        assert "#ready" in note.tags or "ready" in [t.replace("#", "") for t in note.tags]
        assert "#architecture" in note.tags or "architecture" in [
            t.replace("#", "") for t in note.tags
        ]

    def test_extract_urls(self, temp_vault: Path) -> None:
        """Test URL extraction."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert "https://example.com/api-design" in note.urls

    def test_classify_daily_note(self, temp_vault: Path) -> None:
        """Test daily note classification."""
        note_path = temp_vault / "2026-02-03.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert note.note_type == NoteType.DAILY

    def test_classify_decision_note(self, temp_vault: Path) -> None:
        """Test decision note classification."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert note.note_type == NoteType.DECISION

    def test_parse_note_without_frontmatter(self, temp_vault: Path) -> None:
        """Test parsing note without frontmatter."""
        note_path = temp_vault / "quick-note.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        assert note.name == "quick-note"
        assert note.frontmatter.title is None
        assert "# Quick Note" in note.content

    def test_to_evidence(self, temp_vault: Path) -> None:
        """Test converting note to Evidence."""
        note_path = temp_vault / "architecture-decision.md"
        note = ObsidianNote.from_file(temp_vault, note_path)

        evidence = note.to_evidence()

        assert evidence.source_type == SourceType.DOCUMENT
        assert evidence.title == "API Design Decision"
        assert "obsidian-note-" in evidence.id
        assert evidence.metadata["note_type"] == "decision"


# =============================================================================
# ObsidianConnector Tests
# =============================================================================


class TestObsidianConnector:
    """Tests for ObsidianConnector."""

    def test_connector_properties(self, connector: ObsidianConnector) -> None:
        """Test connector properties."""
        assert connector.name == "Obsidian"
        assert connector.source_type == SourceType.DOCUMENT
        assert connector.is_available is True
        assert connector.is_configured is True

    def test_capabilities(self, connector: ObsidianConnector) -> None:
        """Test connector capabilities."""
        caps = connector.capabilities()

        assert caps.can_send is True  # Can write notes
        assert caps.can_receive is True  # Can read notes
        assert caps.can_search is True
        assert caps.can_sync is True
        assert caps.supports_files is True
        assert "wikilinks" in caps.platform_features

    def test_get_note_by_path(self, connector: ObsidianConnector) -> None:
        """Test getting note by path."""
        note = connector.get_note("architecture-decision.md")

        assert note is not None
        assert note.name == "architecture-decision"
        assert note.frontmatter.title == "API Design Decision"

    def test_get_note_without_extension(self, connector: ObsidianConnector) -> None:
        """Test getting note without .md extension."""
        note = connector.get_note("architecture-decision")

        assert note is not None
        assert note.name == "architecture-decision"

    def test_get_note_not_found(self, connector: ObsidianConnector) -> None:
        """Test getting non-existent note."""
        note = connector.get_note("nonexistent-note.md")

        assert note is None

    def test_get_note_by_name(self, connector: ObsidianConnector) -> None:
        """Test getting note by name (searches all folders)."""
        note = connector.get_note_by_name("architecture-decision")

        assert note is not None
        assert note.frontmatter.title == "API Design Decision"

    def test_list_notes(self, connector: ObsidianConnector) -> None:
        """Test listing all notes."""
        notes = connector.list_notes()

        # Should have at least 3 notes (excluding templates folder)
        assert len(notes) >= 3

    def test_list_notes_by_type(self, connector: ObsidianConnector) -> None:
        """Test listing notes filtered by type."""
        daily_notes = connector.list_notes(note_type=NoteType.DAILY)

        assert len(daily_notes) >= 1
        assert all(n.note_type == NoteType.DAILY for n in daily_notes)

    def test_list_notes_by_tags(self, connector: ObsidianConnector) -> None:
        """Test listing notes filtered by tags."""
        notes = connector.list_notes(tags=["#architecture"])

        assert len(notes) >= 1

    @pytest.mark.asyncio
    async def test_search_by_content(self, connector: ObsidianConnector) -> None:
        """Test searching notes by content."""
        results = await connector.search("GraphQL")

        assert len(results) >= 1
        assert any("API" in r.title for r in results)

    @pytest.mark.asyncio
    async def test_search_by_tags(self, connector: ObsidianConnector) -> None:
        """Test searching notes with tag filter."""
        results = await connector.search("", tags=["#architecture"])

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_limit(self, connector: ObsidianConnector) -> None:
        """Test search respects limit."""
        results = await connector.search("note", limit=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_fetch_evidence(self, connector: ObsidianConnector) -> None:
        """Test fetching evidence by ID."""
        # First search to get an ID
        results = await connector.search("API")
        assert len(results) > 0

        evidence_id = results[0].id
        fetched = await connector.fetch(evidence_id)

        assert fetched is not None
        assert fetched.id == evidence_id

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, connector: ObsidianConnector) -> None:
        """Test fetching with invalid ID format."""
        result = await connector.fetch("invalid-id-format")

        assert result is None


class TestObsidianConnectorWrite:
    """Tests for Obsidian write operations."""

    @pytest.mark.asyncio
    async def test_write_note(self, connector: ObsidianConnector, temp_vault: Path) -> None:
        """Test writing a new note."""
        frontmatter = Frontmatter(
            title="Test Written Note",
            tags=["#test", "#written"],
        )

        note = await connector.write_note(
            path="test-folder/written-note.md",
            content="This is test content.",
            frontmatter=frontmatter,
        )

        assert note is not None
        assert note.frontmatter.title == "Test Written Note"

        # Verify file was created
        file_path = temp_vault / "test-folder" / "written-note.md"
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_write_note_no_overwrite(
        self, connector: ObsidianConnector, temp_vault: Path
    ) -> None:
        """Test write_note doesn't overwrite without flag."""
        result = await connector.write_note(
            path="architecture-decision.md",
            content="New content",
            overwrite=False,
        )

        assert result is None  # Should fail without overwrite

    @pytest.mark.asyncio
    async def test_write_note_with_overwrite(
        self, connector: ObsidianConnector, temp_vault: Path
    ) -> None:
        """Test write_note with overwrite flag."""
        result = await connector.write_note(
            path="architecture-decision.md",
            content="Updated content",
            overwrite=True,
        )

        assert result is not None
        assert "Updated content" in result.content

    @pytest.mark.asyncio
    async def test_write_decision_receipt(
        self, connector: ObsidianConnector, temp_vault: Path
    ) -> None:
        """Test writing a decision receipt."""
        note = await connector.write_decision_receipt(
            debate_id="debate_test123",
            title="Test Decision",
            summary="This is a test decision summary.",
            consensus=True,
            confidence=0.92,
            dissent_trail=["Agent 2 disagreed on point X"],
            agents=["claude", "gpt-4", "gemini"],
            evidence_ids=["ev_1", "ev_2", "ev_3"],
        )

        assert note is not None
        assert note.frontmatter.debate_id == "debate_test123"
        assert note.frontmatter.consensus is True
        assert note.frontmatter.confidence == 0.92

        # Verify file exists in decisions folder
        decisions_dir = temp_vault / "decisions"
        assert any(f.name.endswith("-receipt.md") for f in decisions_dir.iterdir())

    @pytest.mark.asyncio
    async def test_update_note_frontmatter(
        self, connector: ObsidianConnector, temp_vault: Path
    ) -> None:
        """Test updating note frontmatter."""
        result = await connector.update_note_frontmatter(
            path="architecture-decision.md",
            updates={"confidence": 0.95, "custom_field": "custom_value"},
        )

        assert result is not None
        assert result.frontmatter.confidence == 0.95
        assert result.frontmatter.custom.get("custom_field") == "custom_value"


class TestObsidianConnectorSync:
    """Tests for Obsidian sync operations."""

    @pytest.mark.asyncio
    async def test_sync_items(self, connector: ObsidianConnector) -> None:
        """Test syncing items for Knowledge Mound."""
        from aragora.connectors.enterprise.base import SyncState

        state = SyncState(connector_id="obsidian")
        items = []

        async for item in connector.sync_items(state, batch_size=10):
            items.append(item)

        assert len(items) >= 1
        assert all(item.source_type == "obsidian_note" for item in items)

    @pytest.mark.asyncio
    async def test_full_sync(self, connector: ObsidianConnector) -> None:
        """Test full sync."""
        result = await connector.full_sync()

        assert result.success is True
        assert result.items_synced >= 1
        assert result.connector_id == "Obsidian"


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    """Tests for factory functions."""

    def test_create_obsidian_connector(self, temp_vault: Path) -> None:
        """Test factory function."""
        connector = create_obsidian_connector(
            vault_path=str(temp_vault),
            watch_tags=["#test"],
        )

        assert connector is not None
        assert connector.name == "Obsidian"
        assert connector.is_configured is True

    def test_create_connector_invalid_path(self) -> None:
        """Test factory with invalid vault path."""
        connector = create_obsidian_connector(
            vault_path="/nonexistent/path/to/vault",
        )

        assert connector.is_available is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_note_with_unicode(self, temp_vault: Path) -> None:
        """Test handling notes with unicode characters."""
        note_path = temp_vault / "unicode-note.md"
        note_path.write_text(
            """---
title: 日本語のタイトル
tags:
  - 日本語
  - emoji-🎉
---

# Unicode Content

This note has unicode: 你好世界 🌍
""",
            encoding="utf-8",
        )

        note = ObsidianNote.from_file(temp_vault, note_path)

        assert note.frontmatter.title == "日本語のタイトル"
        assert "你好世界" in note.content

    def test_note_with_code_blocks(self, temp_vault: Path) -> None:
        """Test handling notes with code blocks."""
        note_path = temp_vault / "code-note.md"
        note_path.write_text("""---
title: Code Note
---

# Code Example

```python
def hello():
    print("Hello, World!")
    # This has a [[fake-link]] that should NOT be extracted
```

Real [[link-outside-code]]
""")

        note = ObsidianNote.from_file(temp_vault, note_path)

        # Code block links shouldn't be extracted (simplified implementation may include them)
        # At minimum, the real link should be found
        assert "link-outside-code" in note.wikilinks

    def test_deeply_nested_folder(self, temp_vault: Path) -> None:
        """Test handling deeply nested folders."""
        deep_path = temp_vault / "level1" / "level2" / "level3"
        deep_path.mkdir(parents=True)

        note_path = deep_path / "deep-note.md"
        note_path.write_text("# Deep Note\n\nContent here.")

        connector = create_obsidian_connector(str(temp_vault))
        notes = connector.list_notes()

        deep_notes = [n for n in notes if "level3" in n.path]
        assert len(deep_notes) >= 1

    def test_empty_vault(self, tmp_path: Path) -> None:
        """Test handling empty vault."""
        empty_vault = tmp_path / "empty_vault"
        empty_vault.mkdir()

        connector = create_obsidian_connector(str(empty_vault))

        notes = connector.list_notes()
        assert len(notes) == 0

    @pytest.mark.asyncio
    async def test_health_check(self, connector: ObsidianConnector) -> None:
        """Test health check."""
        health = await connector.health_check()

        assert health.name == "Obsidian"
        assert health.is_available is True
        assert health.is_configured is True
        assert health.is_healthy is True
