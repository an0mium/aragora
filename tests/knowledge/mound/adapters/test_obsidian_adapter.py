"""
Tests for ObsidianAdapter - Bidirectional Obsidian <-> Knowledge Mound sync.

Tests cover:
- Forward sync (Obsidian -> KM): sync_to_km
- Reverse sync (KM -> Obsidian): sync_from_km with frontmatter writeback
- Conflict detection: user edits vs KM updates
- Filesystem watcher: debouncing, tag filtering
- Factory registration: reverse_method set to sync_from_km
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.knowledge.obsidian import (
    ConflictRecord,
    ConflictStrategy,
    Frontmatter,
    NoteType,
    ObsidianConfig,
    ObsidianConnector,
    ObsidianNote,
)
from aragora.knowledge.mound.adapters.obsidian_adapter import (
    ObsidianAdapter,
    ObsidianSyncConfig,
)
from aragora.knowledge.mound.adapters._types import SyncResult, ValidationSyncResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Create a temporary Obsidian vault with test notes."""
    vault = tmp_path / "test_vault"
    vault.mkdir()

    # Create .obsidian folder (should be ignored)
    (vault / ".obsidian").mkdir()

    # Create a decision note with existing frontmatter
    note1 = vault / "api-decision.md"
    note1.write_text("""---
title: API Design Decision
date: 2026-02-03
tags:
  - decision
  - aragora
aragora_id: debate_abc123
debate_id: debate_abc123
consensus: true
confidence: 0.87
---

# API Design Decision

Should we use REST or GraphQL?

#decision #aragora
""")

    # Create a note without aragora fields
    note2 = vault / "research-note.md"
    note2.write_text("""---
title: Research on Microservices
tags:
  - research
  - aragora
---

# Microservices Research

Notes on microservices architecture.

#research #aragora
""")

    # Create a note with existing KM fields (simulates previous reverse sync)
    note3 = vault / "previously-synced.md"
    note3.write_text("""---
title: Previously Synced Note
tags:
  - decision
  - aragora
aragora_id: debate_prev
km_confidence: 0.75
km_validated_at: "2026-02-01T00:00:00+00:00"
km_validation_result: validated
---

# Previously Synced

This note was synced before.

#decision #aragora
""")

    return vault


@pytest.fixture
def config(temp_vault: Path) -> ObsidianConfig:
    """Create test configuration."""
    return ObsidianConfig(
        vault_path=str(temp_vault),
        watch_tags=["#decision", "#aragora"],
        ignore_folders=[".obsidian", ".trash", "templates"],
        enable_reverse_sync=True,
    )


@pytest.fixture
def connector(config: ObsidianConfig) -> ObsidianConnector:
    """Create test connector."""
    return ObsidianConnector(config)


@pytest.fixture
def adapter(connector: ObsidianConnector) -> ObsidianAdapter:
    """Create test adapter with connector."""
    return ObsidianAdapter(
        connector=connector,
        workspace_id="test-workspace",
    )


@pytest.fixture
def mock_mound() -> MagicMock:
    """Create a mock Knowledge Mound."""
    mound = MagicMock()

    async def mock_query(term: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return mock KM validation results."""
        if "api-decision" in term or "debate_abc123" in term:
            return [
                {
                    "confidence": 0.92,
                    "metadata": {
                        "validation_result": "validated",
                        "cross_debate_utility": 0.78,
                    },
                }
            ]
        if "research-note" in term:
            return [
                {
                    "confidence": 0.65,
                    "metadata": {
                        "validation_result": "needs_review",
                    },
                }
            ]
        if "previously-synced" in term or "debate_prev" in term:
            return [
                {
                    "confidence": 0.88,
                    "metadata": {
                        "validation_result": "validated",
                        "cross_debate_utility": 0.90,
                    },
                }
            ]
        return []

    mound.query = mock_query
    return mound


@pytest.fixture
def mock_mound_for_ingest() -> MagicMock:
    """Create a mock Knowledge Mound that supports ingest."""
    mound = MagicMock()
    mound.ingest = AsyncMock()
    return mound


# =============================================================================
# Reverse Sync Tests: sync_from_km
# =============================================================================


class TestSyncFromKM:
    """Tests for reverse sync (KM -> Obsidian frontmatter)."""

    @pytest.mark.asyncio
    async def test_sync_from_km_writes_frontmatter(
        self,
        adapter: ObsidianAdapter,
        mock_mound: MagicMock,
        temp_vault: Path,
    ) -> None:
        """Test that sync_from_km writes KM validation fields to frontmatter."""
        result = await adapter.sync_from_km(knowledge_mound=mock_mound)

        assert isinstance(result, ValidationSyncResult)
        assert result.records_analyzed >= 1
        assert result.records_updated >= 1
        assert len(result.errors) == 0

        # Verify frontmatter was updated on the decision note
        updated_content = (temp_vault / "api-decision.md").read_text()
        assert "km_confidence" in updated_content
        assert "km_validated_at" in updated_content
        assert "km_validation_result" in updated_content

    @pytest.mark.asyncio
    async def test_sync_from_km_writes_cross_debate_utility(
        self,
        adapter: ObsidianAdapter,
        mock_mound: MagicMock,
        temp_vault: Path,
    ) -> None:
        """Test that cross_debate_utility is written when available."""
        result = await adapter.sync_from_km(knowledge_mound=mock_mound)

        updated_content = (temp_vault / "api-decision.md").read_text()
        assert "cross_debate_utility" in updated_content

    @pytest.mark.asyncio
    async def test_sync_from_km_min_confidence_filter(
        self,
        adapter: ObsidianAdapter,
        mock_mound: MagicMock,
    ) -> None:
        """Test that notes below min_confidence are skipped."""
        result = await adapter.sync_from_km(
            knowledge_mound=mock_mound,
            min_confidence=0.99,  # Very high threshold
        )

        # Everything should be skipped since no note has confidence >= 0.99
        assert result.records_updated == 0
        assert result.records_skipped >= 1

    @pytest.mark.asyncio
    async def test_sync_from_km_limit(
        self,
        adapter: ObsidianAdapter,
        mock_mound: MagicMock,
    ) -> None:
        """Test that limit caps the number of notes analyzed."""
        result = await adapter.sync_from_km(
            knowledge_mound=mock_mound,
            limit=1,
        )

        assert result.records_analyzed <= 1

    @pytest.mark.asyncio
    async def test_sync_from_km_no_connector(self) -> None:
        """Test sync_from_km with no connector configured."""
        adapter = ObsidianAdapter(connector=None, workspace_id="test")
        result = await adapter.sync_from_km()

        assert result.records_analyzed == 0
        assert len(result.errors) > 0
        assert "not configured" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_sync_from_km_no_mound(
        self,
        adapter: ObsidianAdapter,
    ) -> None:
        """Test sync_from_km when Knowledge Mound is not available."""
        with patch.object(adapter, "_get_mound", return_value=None):
            result = await adapter.sync_from_km()

        assert result.records_analyzed == 0
        assert len(result.errors) > 0
        assert "not available" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_sync_from_km_emits_event(
        self,
        connector: ObsidianConnector,
        mock_mound: MagicMock,
    ) -> None:
        """Test that sync_from_km emits completion event."""
        events: list[tuple[str, dict]] = []

        def capture_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        adapter = ObsidianAdapter(
            connector=connector,
            workspace_id="test",
            event_callback=capture_event,
        )

        await adapter.sync_from_km(knowledge_mound=mock_mound)

        assert any(event_type == "obsidian_reverse_sync_complete" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_sync_from_km_reverse_sync_disabled(
        self,
        temp_vault: Path,
        mock_mound: MagicMock,
    ) -> None:
        """Test that notes are skipped when reverse sync is disabled in config."""
        config = ObsidianConfig(
            vault_path=str(temp_vault),
            watch_tags=["#decision", "#aragora"],
            enable_reverse_sync=False,
        )
        connector = ObsidianConnector(config)
        adapter = ObsidianAdapter(connector=connector, workspace_id="test")

        result = await adapter.sync_from_km(knowledge_mound=mock_mound)

        # Notes analyzed but none updated (write_km_validation returns None)
        assert result.records_updated == 0


# =============================================================================
# Conflict Detection Tests
# =============================================================================


class TestConflictDetection:
    """Tests for conflict detection between Obsidian and KM."""

    @pytest.mark.asyncio
    async def test_conflict_prefers_user_edits_for_content(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test that user content edits are preserved over KM values."""
        # Simulate a previous KM write
        connector._km_write_timestamps["api-decision.md"] = datetime(
            2026, 2, 1, tzinfo=timezone.utc
        )

        # The note was modified AFTER the KM write (file on disk is newer)
        note = connector.get_note("api-decision.md")
        assert note is not None

        # Write KM validation that conflicts with user-edited title
        result = await connector.write_km_validation(
            path="api-decision.md",
            km_confidence=0.95,
            km_validation_result="validated",
        )

        assert result is not None

        # The title should still be the user's original value
        assert result.frontmatter.title == "API Design Decision"

    @pytest.mark.asyncio
    async def test_conflict_log_records_conflicts(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test that conflicts are logged."""
        # Write km_confidence that differs from an existing value
        note3 = connector.get_note("previously-synced.md")
        assert note3 is not None
        assert note3.frontmatter.custom.get("km_confidence") == 0.75

        connector.clear_conflict_log()

        result = await connector.write_km_validation(
            path="previously-synced.md",
            km_confidence=0.88,
            km_validation_result="validated",
        )

        assert result is not None

        log = connector.get_conflict_log()
        # Should have at least one conflict (km_confidence changed from 0.75 to 0.88)
        km_conf_conflicts = [c for c in log if c.field_name == "km_confidence"]
        assert len(km_conf_conflicts) >= 1
        conflict = km_conf_conflicts[0]
        assert conflict.obsidian_value == 0.75
        assert conflict.km_value == 0.88

    @pytest.mark.asyncio
    async def test_conflict_strategy_km_wins(
        self,
        temp_vault: Path,
    ) -> None:
        """Test KM_WINS strategy always takes KM value."""
        config = ObsidianConfig(
            vault_path=str(temp_vault),
            conflict_strategy=ConflictStrategy.KM_WINS,
        )
        connector = ObsidianConnector(config)

        result = await connector.write_km_validation(
            path="previously-synced.md",
            km_confidence=0.99,
            km_validation_result="validated",
        )

        assert result is not None
        assert result.frontmatter.custom.get("km_confidence") == 0.99

    @pytest.mark.asyncio
    async def test_conflict_strategy_user_wins(
        self,
        temp_vault: Path,
    ) -> None:
        """Test USER_WINS strategy always keeps existing value."""
        config = ObsidianConfig(
            vault_path=str(temp_vault),
            conflict_strategy=ConflictStrategy.USER_WINS,
        )
        connector = ObsidianConnector(config)

        result = await connector.write_km_validation(
            path="previously-synced.md",
            km_confidence=0.99,
            km_validation_result="validated",
        )

        assert result is not None
        # User wins: the original 0.75 should be preserved
        assert result.frontmatter.custom.get("km_confidence") == 0.75

    def test_clear_conflict_log(self, connector: ObsidianConnector) -> None:
        """Test clearing the conflict log."""
        connector._conflict_log.append(
            ConflictRecord(
                note_path="test.md",
                field_name="km_confidence",
                obsidian_value=0.5,
                km_value=0.9,
                resolved_value=0.9,
                strategy=ConflictStrategy.KM_WINS,
            )
        )
        assert len(connector.get_conflict_log()) == 1

        connector.clear_conflict_log()
        assert len(connector.get_conflict_log()) == 0

    @pytest.mark.asyncio
    async def test_no_conflict_for_new_fields(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test that writing new fields (not existing) creates no conflicts."""
        connector.clear_conflict_log()

        # research-note.md has no km_ fields, so no conflicts expected
        result = await connector.write_km_validation(
            path="research-note.md",
            km_confidence=0.65,
            km_validation_result="needs_review",
        )

        assert result is not None
        assert len(connector.get_conflict_log()) == 0


# =============================================================================
# write_km_validation Tests
# =============================================================================


class TestWriteKMValidation:
    """Tests for the connector's write_km_validation method."""

    @pytest.mark.asyncio
    async def test_write_km_validation_basic(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test basic write_km_validation writes correct fields."""
        result = await connector.write_km_validation(
            path="research-note.md",
            km_confidence=0.82,
            km_validation_result="validated",
            cross_debate_utility=0.71,
        )

        assert result is not None
        assert result.frontmatter.custom.get("km_confidence") == 0.82
        assert result.frontmatter.custom.get("km_validation_result") == "validated"
        assert result.frontmatter.custom.get("cross_debate_utility") == 0.71
        assert "km_validated_at" in result.frontmatter.custom

    @pytest.mark.asyncio
    async def test_write_km_validation_disabled(
        self,
        temp_vault: Path,
    ) -> None:
        """Test write_km_validation returns None when reverse sync disabled."""
        config = ObsidianConfig(
            vault_path=str(temp_vault),
            enable_reverse_sync=False,
        )
        connector = ObsidianConnector(config)

        result = await connector.write_km_validation(
            path="research-note.md",
            km_confidence=0.82,
            km_validation_result="validated",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_write_km_validation_nonexistent_note(
        self,
        connector: ObsidianConnector,
    ) -> None:
        """Test write_km_validation returns None for missing note."""
        result = await connector.write_km_validation(
            path="nonexistent-note.md",
            km_confidence=0.5,
            km_validation_result="validated",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_write_km_validation_extra_fields(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test write_km_validation supports extra_fields."""
        result = await connector.write_km_validation(
            path="research-note.md",
            km_confidence=0.82,
            km_validation_result="validated",
            extra_fields={"km_debate_count": 3, "km_source_quality": "high"},
        )

        assert result is not None
        assert result.frontmatter.custom.get("km_debate_count") == 3
        assert result.frontmatter.custom.get("km_source_quality") == "high"

    @pytest.mark.asyncio
    async def test_write_km_validation_tracks_timestamp(
        self,
        connector: ObsidianConnector,
        temp_vault: Path,
    ) -> None:
        """Test that write_km_validation tracks the write timestamp."""
        assert "research-note.md" not in connector._km_write_timestamps

        await connector.write_km_validation(
            path="research-note.md",
            km_confidence=0.82,
            km_validation_result="validated",
        )

        assert "research-note.md" in connector._km_write_timestamps
        assert isinstance(connector._km_write_timestamps["research-note.md"], datetime)


# =============================================================================
# Filesystem Watcher Tests
# =============================================================================


class TestFilesystemWatcher:
    """Tests for the filesystem watcher."""

    def test_start_watcher_without_watchdog(self, connector: ObsidianConnector) -> None:
        """Test watcher gracefully degrades when watchdog not installed."""
        with patch("aragora.connectors.knowledge.obsidian._WATCHDOG_AVAILABLE", False):
            result = connector.start_watcher()

        assert result is False
        assert not connector.watcher_running

    def test_start_watcher_unavailable_vault(self, tmp_path: Path) -> None:
        """Test watcher refuses to start when vault is not available."""
        config = ObsidianConfig(vault_path=str(tmp_path / "nonexistent"))
        connector = ObsidianConnector(config)

        # Even with watchdog available, should fail
        with patch("aragora.connectors.knowledge.obsidian._WATCHDOG_AVAILABLE", True):
            result = connector.start_watcher()

        assert result is False

    def test_stop_watcher_when_not_running(self, connector: ObsidianConnector) -> None:
        """Test stop_watcher is safe when watcher is not running."""
        connector.stop_watcher()  # Should not raise
        assert not connector.watcher_running

    def test_watcher_running_property(self, connector: ObsidianConnector) -> None:
        """Test watcher_running reflects state."""
        assert not connector.watcher_running

    def test_on_file_change_calls_callback(self, connector: ObsidianConnector) -> None:
        """Test _on_file_change calls the registered callback."""
        changes: list[str] = []
        connector._watcher_callback = lambda p: changes.append(p)

        connector._on_file_change("test-note.md")

        assert changes == ["test-note.md"]

    def test_on_file_change_handles_callback_error(self, connector: ObsidianConnector) -> None:
        """Test _on_file_change handles callback errors gracefully."""

        def bad_callback(path: str) -> None:
            raise RuntimeError("callback failed")

        connector._watcher_callback = bad_callback

        # Should not raise
        connector._on_file_change("test-note.md")


class TestFileHandlerDebounce:
    """Tests for the _ObsidianFileHandler debounce logic."""

    def test_debounce_fires_after_delay(self) -> None:
        """Test that debounced events fire after the debounce window."""
        from aragora.connectors.knowledge.obsidian import _ObsidianFileHandler

        fired: list[str] = []

        handler = _ObsidianFileHandler(
            vault_path=Path("/tmp/test"),
            watch_tags=["#test"],
            ignore_folders=[".obsidian"],
            debounce_seconds=0.1,
            callback=lambda p: fired.append(p),
        )

        handler._debounce("note.md")

        # Should not have fired yet
        assert len(fired) == 0

        # Wait for debounce
        time.sleep(0.25)

        assert fired == ["note.md"]

    def test_debounce_coalesces_rapid_events(self) -> None:
        """Test that rapid events are coalesced into a single callback."""
        from aragora.connectors.knowledge.obsidian import _ObsidianFileHandler

        fired: list[str] = []

        handler = _ObsidianFileHandler(
            vault_path=Path("/tmp/test"),
            watch_tags=["#test"],
            ignore_folders=[".obsidian"],
            debounce_seconds=0.2,
            callback=lambda p: fired.append(p),
        )

        # Fire multiple rapid events for the same file
        handler._debounce("note.md")
        time.sleep(0.05)
        handler._debounce("note.md")
        time.sleep(0.05)
        handler._debounce("note.md")

        # Should not have fired yet (each reset the timer)
        assert len(fired) == 0

        # Wait for final debounce
        time.sleep(0.4)

        # Should have fired exactly once
        assert fired == ["note.md"]

    def test_debounce_different_files_independent(self) -> None:
        """Test that different files debounce independently."""
        from aragora.connectors.knowledge.obsidian import _ObsidianFileHandler

        fired: list[str] = []

        handler = _ObsidianFileHandler(
            vault_path=Path("/tmp/test"),
            watch_tags=["#test"],
            ignore_folders=[".obsidian"],
            debounce_seconds=0.1,
            callback=lambda p: fired.append(p),
        )

        handler._debounce("note-a.md")
        handler._debounce("note-b.md")

        time.sleep(0.25)

        assert "note-a.md" in fired
        assert "note-b.md" in fired


# =============================================================================
# Forward Sync Tests: sync_to_km
# =============================================================================


class TestSyncToKM:
    """Tests for forward sync (Obsidian -> KM) -- ensure it still works."""

    @pytest.mark.asyncio
    async def test_sync_to_km_basic(
        self,
        adapter: ObsidianAdapter,
        mock_mound_for_ingest: MagicMock,
    ) -> None:
        """Test basic forward sync."""
        result = await adapter.sync_to_km(
            knowledge_mound=mock_mound_for_ingest,
            tags=["#decision", "#aragora"],
        )

        assert isinstance(result, SyncResult)
        assert result.records_synced >= 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_sync_to_km_no_connector(self) -> None:
        """Test forward sync with no connector."""
        adapter = ObsidianAdapter(connector=None, workspace_id="test")
        result = await adapter.sync_to_km()

        assert result.records_synced == 0
        assert len(result.errors) > 0


# =============================================================================
# Factory Registration Tests
# =============================================================================


class TestFactoryRegistration:
    """Tests for adapter factory registration."""

    def test_obsidian_spec_has_reverse_method(self) -> None:
        """Test that the obsidian adapter spec has sync_from_km reverse method."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        spec = ADAPTER_SPECS.get("obsidian")
        assert spec is not None
        assert spec.reverse_method == "sync_from_km"
        assert spec.forward_method == "sync_to_km"

    def test_obsidian_spec_config_key(self) -> None:
        """Test that the obsidian adapter spec has correct config key."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        spec = ADAPTER_SPECS.get("obsidian")
        assert spec is not None
        assert spec.config_key == "km_obsidian_adapter"

    def test_obsidian_adapter_has_sync_from_km(self) -> None:
        """Test that ObsidianAdapter actually has sync_from_km method."""
        assert hasattr(ObsidianAdapter, "sync_from_km")
        assert callable(getattr(ObsidianAdapter, "sync_from_km"))


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestAdapterInit:
    """Tests for ObsidianAdapter initialization."""

    def test_init_with_connector(self, connector: ObsidianConnector) -> None:
        """Test initialization with a pre-configured connector."""
        adapter = ObsidianAdapter(connector=connector, workspace_id="test")

        assert adapter.connector is connector
        assert adapter.adapter_name == "obsidian"

    def test_init_with_vault_path(self, temp_vault: Path) -> None:
        """Test initialization with just a vault path."""
        adapter = ObsidianAdapter(vault_path=str(temp_vault), workspace_id="test")

        assert adapter.connector is not None
        assert adapter.connector.is_configured

    def test_init_with_config(self, config: ObsidianConfig) -> None:
        """Test initialization with ObsidianConfig."""
        adapter = ObsidianAdapter(config=config, workspace_id="test")

        assert adapter.connector is not None

    def test_reverse_flow_stats(self, adapter: ObsidianAdapter) -> None:
        """Test reverse flow stats tracking from base class."""
        stats = adapter.get_reverse_flow_stats()

        assert "validations_applied" in stats
        assert "adjustments_made" in stats
        assert stats["validations_applied"] == 0


# =============================================================================
# Config Tests
# =============================================================================


class TestObsidianConfigExtensions:
    """Tests for the new config fields."""

    def test_default_reverse_sync_enabled(self) -> None:
        """Test reverse sync is enabled by default."""
        config = ObsidianConfig(vault_path="/tmp/test")
        assert config.enable_reverse_sync is True

    def test_default_conflict_strategy(self) -> None:
        """Test default conflict strategy is user_content_km_frontmatter."""
        config = ObsidianConfig(vault_path="/tmp/test")
        assert config.conflict_strategy == ConflictStrategy.USER_CONTENT_KM_FRONTMATTER

    def test_default_watcher_debounce(self) -> None:
        """Test default debounce is 500ms."""
        config = ObsidianConfig(vault_path="/tmp/test")
        assert config.watcher_debounce_ms == 500

    def test_from_env_reverse_sync(self) -> None:
        """Test from_env parses reverse sync setting."""
        env = {
            "ARAGORA_OBSIDIAN_VAULT_PATH": "/tmp/test",
            "ARAGORA_OBSIDIAN_ENABLE_REVERSE_SYNC": "false",
        }
        config = ObsidianConfig.from_env(env)
        assert config is not None
        assert config.enable_reverse_sync is False

    def test_from_env_debounce(self) -> None:
        """Test from_env parses debounce setting."""
        env = {
            "ARAGORA_OBSIDIAN_VAULT_PATH": "/tmp/test",
            "ARAGORA_OBSIDIAN_WATCHER_DEBOUNCE_MS": "1000",
        }
        config = ObsidianConfig.from_env(env)
        assert config is not None
        assert config.watcher_debounce_ms == 1000
