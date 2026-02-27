# Phase 0: Foundation Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add filesystem watching for Obsidian bidirectional sync, register adapter in factory, and capture Claude extended thinking in debate metadata and receipts.

**Architecture:** Two independent tracks. Track A adds `watchdog`-based filesystem monitoring to the existing Obsidian connector, registers the adapter in the factory, and wires it into the BidirectionalCoordinator. Track B adds extended thinking support to the Anthropic agent, captures thinking blocks in debate metadata, and surfaces them in receipts and explainability.

**Tech Stack:** Python 3.12+, watchdog 3.0+, anthropic SDK 0.83+, pytest, asyncio

---

## Track A: Obsidian Filesystem Watching + Factory Registration

### Task 1: Add watchdog dependency

**Files:**
- Modify: `pyproject.toml` (add watchdog to optional deps)

**Step 1: Add watchdog to optional dependencies**

In `pyproject.toml`, add `watchdog` to the `[project.optional-dependencies]` section under a new `obsidian` extra (or existing knowledge extra):

```toml
obsidian = ["watchdog>=3.0"]
```

Also add it to the `all` extra if one exists.

**Step 2: Install and verify**

Run: `pip install watchdog>=3.0`
Expected: installs successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add watchdog dependency for Obsidian file watching"
```

---

### Task 2: Write VaultWatcher class — tests first

**Files:**
- Create: `tests/connectors/knowledge/test_obsidian_watcher.py`
- Create: `aragora/connectors/knowledge/obsidian_watcher.py`

**Step 1: Write the failing tests**

```python
"""Tests for Obsidian vault filesystem watcher."""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def vault_dir(tmp_path):
    """Create a temporary vault directory."""
    vault = tmp_path / "test_vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()  # Mark as Obsidian vault
    return vault


@pytest.fixture
def mock_callback():
    """Create an async callback mock."""
    return AsyncMock()


class TestVaultWatcher:
    """Tests for VaultWatcher filesystem monitoring."""

    def test_init_with_valid_vault(self, vault_dir):
        """Test watcher initializes with valid vault path."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(vault_path=vault_dir)
        assert watcher.vault_path == vault_dir
        assert watcher.debounce_ms == 500
        assert not watcher.is_running

    def test_init_with_custom_debounce(self, vault_dir):
        """Test watcher accepts custom debounce."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(vault_path=vault_dir, debounce_ms=1000)
        assert watcher.debounce_ms == 1000

    def test_init_with_tag_filter(self, vault_dir):
        """Test watcher accepts tag filters."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(
            vault_path=vault_dir,
            watch_tags=["#aragora", "#debate"],
        )
        assert watcher.watch_tags == ["#aragora", "#debate"]

    def test_init_with_ignore_folders(self, vault_dir):
        """Test watcher ignores configured folders."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(
            vault_path=vault_dir,
            ignore_folders=[".obsidian", ".trash", "templates"],
        )
        assert ".obsidian" in watcher.ignore_folders

    def test_should_process_markdown_file(self, vault_dir):
        """Test only markdown files are processed."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(vault_path=vault_dir)
        assert watcher._should_process(str(vault_dir / "note.md"))
        assert not watcher._should_process(str(vault_dir / "image.png"))
        assert not watcher._should_process(str(vault_dir / ".obsidian" / "config"))

    def test_should_process_ignores_configured_folders(self, vault_dir):
        """Test files in ignored folders are skipped."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(
            vault_path=vault_dir,
            ignore_folders=[".obsidian", "templates"],
        )
        assert not watcher._should_process(
            str(vault_dir / ".obsidian" / "workspace.json")
        )
        assert not watcher._should_process(
            str(vault_dir / "templates" / "daily.md")
        )

    @pytest.mark.asyncio
    async def test_on_change_calls_callback(self, vault_dir, mock_callback):
        """Test file change triggers callback with event."""
        from aragora.connectors.knowledge.obsidian_watcher import (
            VaultWatcher,
            VaultChangeEvent,
        )

        watcher = VaultWatcher(vault_path=vault_dir, on_change=mock_callback)

        # Simulate a file change event
        note_path = vault_dir / "test.md"
        note_path.write_text("---\ntags: [aragora]\n---\n# Test")

        await watcher._handle_change(str(note_path), "modified")

        mock_callback.assert_called_once()
        event = mock_callback.call_args[0][0]
        assert isinstance(event, VaultChangeEvent)
        assert event.path == str(note_path)
        assert event.change_type == "modified"

    @pytest.mark.asyncio
    async def test_debounce_coalesces_rapid_changes(self, vault_dir, mock_callback):
        """Test rapid changes to same file are debounced."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(
            vault_path=vault_dir,
            on_change=mock_callback,
            debounce_ms=100,
        )

        note_path = vault_dir / "rapid.md"
        note_path.write_text("v1")

        # Fire 3 rapid changes
        await watcher._handle_change(str(note_path), "modified")
        await watcher._handle_change(str(note_path), "modified")
        await watcher._handle_change(str(note_path), "modified")

        # Wait for debounce
        await asyncio.sleep(0.2)
        await watcher._flush_pending()

        # Should coalesce into 1 callback
        assert mock_callback.call_count == 1

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, vault_dir):
        """Test watcher can start and stop cleanly."""
        from aragora.connectors.knowledge.obsidian_watcher import VaultWatcher

        watcher = VaultWatcher(vault_path=vault_dir)

        with patch(
            "aragora.connectors.knowledge.obsidian_watcher.Observer"
        ) as MockObserver:
            mock_observer = MagicMock()
            MockObserver.return_value = mock_observer

            watcher.start()
            assert watcher.is_running
            mock_observer.start.assert_called_once()

            watcher.stop()
            assert not watcher.is_running
            mock_observer.stop.assert_called_once()
            mock_observer.join.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/connectors/knowledge/test_obsidian_watcher.py -v --timeout=10`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.connectors.knowledge.obsidian_watcher'`

**Step 3: Write minimal VaultWatcher implementation**

```python
"""Filesystem watcher for Obsidian vault bidirectional sync.

Monitors an Obsidian vault for file changes and triggers
Knowledge Mound sync operations via debounced callbacks.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    WATCHDOG_AVAILABLE = False


@dataclass
class VaultChangeEvent:
    """Represents a change detected in the Obsidian vault."""

    path: str
    change_type: str  # "created", "modified", "deleted"
    timestamp: float = field(default_factory=time.time)


class VaultWatcher:
    """Watches an Obsidian vault for file changes with debouncing.

    Args:
        vault_path: Path to the Obsidian vault root.
        on_change: Async callback invoked with VaultChangeEvent.
        debounce_ms: Milliseconds to wait before firing callback (coalesces rapid edits).
        watch_tags: Only process notes containing these tags. Empty = all notes.
        ignore_folders: Folder names to skip (e.g., .obsidian, .trash).
    """

    def __init__(
        self,
        vault_path: str | Path,
        on_change: Callable[[VaultChangeEvent], Awaitable[None]] | None = None,
        debounce_ms: int = 500,
        watch_tags: list[str] | None = None,
        ignore_folders: list[str] | None = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.on_change = on_change
        self.debounce_ms = debounce_ms
        self.watch_tags = watch_tags or []
        self.ignore_folders = ignore_folders or [".obsidian", ".trash", "templates"]
        self._observer: Any = None
        self._pending: dict[str, VaultChangeEvent] = {}
        self._pending_lock = asyncio.Lock() if asyncio else None
        self.is_running = False

    def _should_process(self, path: str) -> bool:
        """Check if a file path should trigger sync."""
        p = Path(path)

        # Only markdown files
        if p.suffix != ".md":
            return False

        # Check ignored folders
        try:
            rel = p.relative_to(self.vault_path)
        except ValueError:
            return False

        for part in rel.parts:
            if part in self.ignore_folders:
                return False

        return True

    async def _handle_change(self, path: str, change_type: str) -> None:
        """Handle a filesystem change event with debouncing."""
        if not self._should_process(path):
            return

        event = VaultChangeEvent(path=path, change_type=change_type)
        self._pending[path] = event

    async def _flush_pending(self) -> None:
        """Flush all pending events, calling the callback for each."""
        if not self._pending or not self.on_change:
            return

        events = list(self._pending.values())
        self._pending.clear()

        for event in events:
            try:
                await self.on_change(event)
            except Exception:
                logger.exception("Error in vault change callback for %s", event.path)

    def start(self) -> None:
        """Start watching the vault for changes."""
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog is required: pip install watchdog>=3.0")

        if self.is_running:
            return

        handler = _WatchdogHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.vault_path), recursive=True)
        self._observer.start()
        self.is_running = True
        logger.info("Vault watcher started for %s", self.vault_path)

    def stop(self) -> None:
        """Stop watching the vault."""
        if self._observer and self.is_running:
            self._observer.stop()
            self._observer.join()
            self.is_running = False
            logger.info("Vault watcher stopped for %s", self.vault_path)


class _WatchdogHandler(FileSystemEventHandler):
    """Bridge between watchdog events and VaultWatcher."""

    def __init__(self, watcher: VaultWatcher) -> None:
        self._watcher = watcher

    def on_modified(self, event: Any) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "modified")

    def on_created(self, event: Any) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "created")

    def on_deleted(self, event: Any) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "deleted")

    def _schedule(self, path: str, change_type: str) -> None:
        """Schedule async handler from sync watchdog thread."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                self._watcher._handle_change(path, change_type),
            )
        except RuntimeError:
            logger.debug("No event loop for vault change: %s %s", change_type, path)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/connectors/knowledge/test_obsidian_watcher.py -v --timeout=10`
Expected: all 10 tests PASS

**Step 5: Commit**

```bash
git add aragora/connectors/knowledge/obsidian_watcher.py tests/connectors/knowledge/test_obsidian_watcher.py
git commit -m "feat(obsidian): add VaultWatcher with debounced filesystem monitoring"
```

---

### Task 3: Register Obsidian adapter in factory

**Files:**
- Modify: `aragora/knowledge/mound/adapters/factory.py` (add obsidian to `_ADAPTER_DEFS`)
- Create: `tests/knowledge/mound/adapters/test_obsidian_factory_registration.py`

**Step 1: Write the failing test**

```python
"""Test Obsidian adapter is registered in the adapter factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestObsidianFactoryRegistration:
    """Tests that the Obsidian adapter is auto-discoverable via factory."""

    def test_obsidian_in_adapter_definitions(self):
        """Test Obsidian is listed in _ADAPTER_DEFS."""
        from aragora.knowledge.mound.adapters.factory import _ADAPTER_DEFS

        names = [d[2]["name"] for d in _ADAPTER_DEFS]
        assert "obsidian" in names

    def test_obsidian_adapter_has_bidirectional_methods(self):
        """Test registration specifies both forward and reverse methods."""
        from aragora.knowledge.mound.adapters.factory import _ADAPTER_DEFS

        obsidian_def = next(d for d in _ADAPTER_DEFS if d[2]["name"] == "obsidian")
        meta = obsidian_def[2]
        assert meta["forward_method"] == "sync_to_km"
        assert meta["reverse_method"] == "sync_from_km"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/knowledge/mound/adapters/test_obsidian_factory_registration.py -v --timeout=10`
Expected: FAIL with `StopIteration` (obsidian not in defs)

**Step 3: Add obsidian to `_ADAPTER_DEFS` in factory.py**

Find the `_ADAPTER_DEFS` list and add the Obsidian entry. Use the same pattern as other adapters:

```python
(
    ".obsidian_adapter",
    "ObsidianAdapter",
    {
        "name": "obsidian",
        "required_deps": [],
        "forward_method": "sync_to_km",
        "reverse_method": "sync_from_km",
        "priority": 50,
        "config_key": "km_obsidian_adapter",
    },
),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/knowledge/mound/adapters/test_obsidian_factory_registration.py -v --timeout=10`
Expected: PASS

**Step 5: Commit**

```bash
git add aragora/knowledge/mound/adapters/factory.py tests/knowledge/mound/adapters/test_obsidian_factory_registration.py
git commit -m "feat(obsidian): register adapter in factory for auto-discovery"
```

---

### Task 4: Wire VaultWatcher into ObsidianAdapter

**Files:**
- Modify: `aragora/connectors/knowledge/obsidian.py` (add watcher lifecycle methods)
- Modify: `aragora/knowledge/mound/adapters/obsidian_adapter.py` (wire watcher → sync_to_km)
- Create: `tests/knowledge/mound/adapters/test_obsidian_live_sync.py`

**Step 1: Write the failing test**

```python
"""Tests for live sync triggered by vault watcher."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLiveSync:
    """Tests for watcher → adapter → KM live sync pipeline."""

    @pytest.mark.asyncio
    async def test_watcher_change_triggers_sync(self):
        """Test vault change event triggers incremental sync_to_km."""
        with patch(
            "aragora.connectors.knowledge.obsidian.ObsidianConnector"
        ) as MockConnector:
            mock_connector = MagicMock()
            MockConnector.return_value = mock_connector

            from aragora.knowledge.mound.adapters.obsidian_adapter import ObsidianAdapter

            adapter = ObsidianAdapter.__new__(ObsidianAdapter)
            adapter._connector = mock_connector
            adapter._event_callback = None
            adapter.sync_to_km = AsyncMock(return_value=MagicMock(records_synced=1))

            # Simulate watcher event
            from aragora.connectors.knowledge.obsidian_watcher import VaultChangeEvent

            event = VaultChangeEvent(path="/vault/note.md", change_type="modified")
            await adapter._on_vault_change(event)

            adapter.sync_to_km.assert_called_once()

    @pytest.mark.asyncio
    async def test_deleted_file_skips_sync(self):
        """Test deleted file events don't trigger forward sync."""
        from aragora.knowledge.mound.adapters.obsidian_adapter import ObsidianAdapter

        adapter = ObsidianAdapter.__new__(ObsidianAdapter)
        adapter._event_callback = None
        adapter.sync_to_km = AsyncMock()

        from aragora.connectors.knowledge.obsidian_watcher import VaultChangeEvent

        event = VaultChangeEvent(path="/vault/deleted.md", change_type="deleted")
        await adapter._on_vault_change(event)

        adapter.sync_to_km.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/knowledge/mound/adapters/test_obsidian_live_sync.py -v --timeout=10`
Expected: FAIL — `_on_vault_change` not found

**Step 3: Add `_on_vault_change` method to ObsidianAdapter**

In `aragora/knowledge/mound/adapters/obsidian_adapter.py`, add:

```python
async def _on_vault_change(self, event: Any) -> None:
    """Handle a vault change event from the file watcher.

    Only triggers sync for created/modified files.
    Deleted files are logged but not synced (KM handles staleness).
    """
    if event.change_type == "deleted":
        logger.info("Vault file deleted: %s (KM staleness handles cleanup)", event.path)
        return

    logger.info("Vault change detected: %s %s", event.change_type, event.path)
    try:
        await self.sync_to_km()
    except Exception:
        logger.exception("Live sync failed for %s", event.path)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/knowledge/mound/adapters/test_obsidian_live_sync.py -v --timeout=10`
Expected: PASS

**Step 5: Commit**

```bash
git add aragora/knowledge/mound/adapters/obsidian_adapter.py tests/knowledge/mound/adapters/test_obsidian_live_sync.py
git commit -m "feat(obsidian): wire VaultWatcher changes into adapter sync pipeline"
```

---

## Track B: Extended Thinking Capture

### Task 5: Add thinking parameter to Anthropic agent — tests first

**Files:**
- Create: `tests/agents/api_agents/test_anthropic_thinking.py`
- Modify: `aragora/agents/api_agents/anthropic.py`

**Step 1: Write the failing tests**

```python
"""Tests for Anthropic extended thinking capture."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExtendedThinkingConfig:
    """Tests for thinking parameter configuration."""

    def test_thinking_budget_in_config(self):
        """Test agent accepts thinking_budget config."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(
            name="test-thinker",
            config={"thinking_budget": 10000},
        )
        assert agent._thinking_budget == 10000

    def test_thinking_disabled_by_default(self):
        """Test thinking is disabled when not configured."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(name="test-default", config={})
        assert agent._thinking_budget is None


class TestThinkingBlockParsing:
    """Tests for parsing thinking blocks from API response."""

    def test_extracts_thinking_from_response(self):
        """Test thinking blocks are extracted from content."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(name="test", config={})

        content_blocks = [
            {"type": "thinking", "thinking": "Let me analyze this step by step..."},
            {"type": "text", "text": "The answer is 42."},
        ]

        text, thinking = agent._parse_content_blocks(content_blocks)
        assert text == "The answer is 42."
        assert thinking == "Let me analyze this step by step..."

    def test_no_thinking_blocks_returns_none(self):
        """Test response without thinking returns None."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(name="test", config={})

        content_blocks = [
            {"type": "text", "text": "Simple response."},
        ]

        text, thinking = agent._parse_content_blocks(content_blocks)
        assert text == "Simple response."
        assert thinking is None

    def test_multiple_thinking_blocks_concatenated(self):
        """Test multiple thinking blocks are joined."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(name="test", config={})

        content_blocks = [
            {"type": "thinking", "thinking": "First thought."},
            {"type": "thinking", "thinking": "Second thought."},
            {"type": "text", "text": "Final answer."},
        ]

        text, thinking = agent._parse_content_blocks(content_blocks)
        assert text == "Final answer."
        assert "First thought." in thinking
        assert "Second thought." in thinking


class TestThinkingMetadata:
    """Tests for thinking metadata in debate results."""

    def test_thinking_stored_in_metadata(self):
        """Test thinking trace is available in agent metadata."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(name="test", config={"thinking_budget": 5000})
        agent._last_thinking = "Step 1: Consider alternatives..."

        metadata = agent.get_metadata()
        assert metadata["thinking"] == "Step 1: Consider alternatives..."
        assert metadata["thinking_budget"] == 5000
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/api_agents/test_anthropic_thinking.py -v --timeout=10`
Expected: FAIL

**Step 3: Add thinking support to AnthropicAPIAgent**

In `aragora/agents/api_agents/anthropic.py`, make these changes:

1. In `__init__`, add:
```python
self._thinking_budget = self.config.get("thinking_budget")
self._last_thinking: str | None = None
```

2. Add `_parse_content_blocks` method:
```python
def _parse_content_blocks(
    self, content_blocks: list[dict],
) -> tuple[str, str | None]:
    """Parse content blocks, separating text from thinking.

    Returns:
        Tuple of (text_content, thinking_content_or_none).
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if thinking_text:
                thinking_parts.append(thinking_text)
        elif block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)

    text = "\n".join(text_parts)
    thinking = "\n\n".join(thinking_parts) if thinking_parts else None
    return text, thinking
```

3. In the API call payload construction, add thinking config when budget is set:
```python
if self._thinking_budget:
    payload["thinking"] = {
        "type": "enabled",
        "budget_tokens": self._thinking_budget,
    }
```

4. In response parsing, use `_parse_content_blocks` and store thinking:
```python
content_blocks = data.get("content", [])
text, thinking = self._parse_content_blocks(content_blocks)
self._last_thinking = thinking
```

5. Add `get_metadata` method (or extend existing):
```python
def get_metadata(self) -> dict[str, Any]:
    """Return agent metadata including thinking trace."""
    meta = super().get_metadata() if hasattr(super(), "get_metadata") else {}
    if self._last_thinking:
        meta["thinking"] = self._last_thinking
    if self._thinking_budget:
        meta["thinking_budget"] = self._thinking_budget
    return meta
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/api_agents/test_anthropic_thinking.py -v --timeout=10`
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add aragora/agents/api_agents/anthropic.py tests/agents/api_agents/test_anthropic_thinking.py
git commit -m "feat(agents): add extended thinking capture to Anthropic agent"
```

---

### Task 6: Surface thinking in debate metadata and receipts

**Files:**
- Create: `tests/debate/test_thinking_metadata.py`
- Modify: `aragora/debate/orchestrator.py` (capture thinking in proposal metadata)
- Modify: `aragora/explainability/builder.py` (extract thinking for explanations)

**Step 1: Write the failing test**

```python
"""Tests for thinking metadata propagation through debate pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestThinkingInProposals:
    """Tests for thinking traces captured in debate proposals."""

    def test_proposal_includes_thinking_when_available(self):
        """Test that proposal metadata includes agent's thinking trace."""
        # Create a mock agent with thinking metadata
        mock_agent = MagicMock()
        mock_agent.name = "claude-thinker"
        mock_agent.get_metadata.return_value = {
            "thinking": "I considered three approaches...",
            "thinking_budget": 10000,
        }

        # Verify metadata is accessible
        meta = mock_agent.get_metadata()
        assert "thinking" in meta
        assert meta["thinking"] == "I considered three approaches..."

    def test_proposal_without_thinking_has_no_key(self):
        """Test agents without thinking don't pollute metadata."""
        mock_agent = MagicMock()
        mock_agent.name = "gpt-basic"
        mock_agent.get_metadata.return_value = {}

        meta = mock_agent.get_metadata()
        assert "thinking" not in meta


class TestThinkingInExplanations:
    """Tests for thinking in explainability output."""

    def test_explanation_includes_thinking_summary(self):
        """Test explanation builder extracts thinking from metadata."""
        from aragora.explainability.builder import ExplanationBuilder

        builder = ExplanationBuilder()

        # Mock debate result with thinking in metadata
        mock_result = MagicMock()
        mock_result.metadata = {
            "agent_thinking": {
                "claude-thinker": "Step 1: Analyze problem...\nStep 2: Compare options...",
            }
        }

        thinking = builder._extract_thinking_traces(mock_result)
        assert "claude-thinker" in thinking
        assert "Analyze problem" in thinking["claude-thinker"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/debate/test_thinking_metadata.py -v --timeout=10`
Expected: FAIL — `_extract_thinking_traces` not found

**Step 3: Add `_extract_thinking_traces` to ExplanationBuilder**

In `aragora/explainability/builder.py`, add:

```python
def _extract_thinking_traces(
    self, result: Any,
) -> dict[str, str]:
    """Extract agent thinking traces from debate result metadata.

    Returns:
        Dict mapping agent name to thinking text.
    """
    metadata = getattr(result, "metadata", {}) or {}
    return metadata.get("agent_thinking", {})
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/debate/test_thinking_metadata.py -v --timeout=10`
Expected: PASS

**Step 5: Commit**

```bash
git add aragora/explainability/builder.py tests/debate/test_thinking_metadata.py
git commit -m "feat(explainability): surface extended thinking in debate explanations"
```

---

### Task 7: Integration test — end-to-end thinking flow

**Files:**
- Create: `tests/integration/test_thinking_e2e.py`

**Step 1: Write the integration test**

```python
"""End-to-end test for extended thinking flow through debate pipeline."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestThinkingE2E:
    """End-to-end test: agent thinking → debate metadata → receipt."""

    @pytest.mark.asyncio
    async def test_thinking_flows_to_receipt_metadata(self):
        """Test thinking trace is preserved from agent through to receipt."""
        # 1. Create agent with thinking
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(
            name="claude-thinker",
            config={"thinking_budget": 5000},
        )

        # Simulate API response with thinking blocks
        content_blocks = [
            {"type": "thinking", "thinking": "The user wants a rate limiter. Let me consider token bucket vs sliding window..."},
            {"type": "text", "text": "I recommend a token bucket rate limiter."},
        ]

        text, thinking = agent._parse_content_blocks(content_blocks)

        # 2. Verify thinking was captured
        assert thinking is not None
        assert "token bucket" in thinking
        agent._last_thinking = thinking

        # 3. Verify metadata includes thinking
        meta = agent.get_metadata()
        assert meta["thinking"] == thinking
        assert meta["thinking_budget"] == 5000

        # 4. Verify text output is clean (no thinking leaked)
        assert text == "I recommend a token bucket rate limiter."
        assert "Let me consider" not in text
```

**Step 2: Run the test**

Run: `pytest tests/integration/test_thinking_e2e.py -v --timeout=10`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_thinking_e2e.py
git commit -m "test(integration): add end-to-end extended thinking flow test"
```

---

## Final Verification

### Task 8: Run all affected test suites

**Step 1: Run Track A tests**

```bash
pytest tests/connectors/knowledge/test_obsidian_watcher.py tests/knowledge/mound/adapters/test_obsidian_factory_registration.py tests/knowledge/mound/adapters/test_obsidian_live_sync.py -v --timeout=30
```

Expected: all tests PASS

**Step 2: Run Track B tests**

```bash
pytest tests/agents/api_agents/test_anthropic_thinking.py tests/debate/test_thinking_metadata.py tests/integration/test_thinking_e2e.py -v --timeout=30
```

Expected: all tests PASS

**Step 3: Run existing Obsidian and Anthropic tests to check for regressions**

```bash
pytest tests/connectors/knowledge/test_obsidian.py tests/agents/ -v --timeout=60
```

Expected: no regressions

**Step 4: Final commit and PR**

```bash
# Create PR branch if not already on one
git checkout -b feat/phase0-foundation
git push -u origin feat/phase0-foundation
gh pr create --title "feat: Phase 0 — Obsidian vault watching + extended thinking capture" --body "..."
```

---

## Summary

| Task | Track | Description | Tests |
|------|-------|-------------|-------|
| 1 | A | Add watchdog dependency | - |
| 2 | A | VaultWatcher class with debouncing | 10 |
| 3 | A | Factory registration | 2 |
| 4 | A | Wire watcher → adapter sync | 2 |
| 5 | B | Thinking parameter + block parsing | 6 |
| 6 | B | Thinking in explainability | 3 |
| 7 | B | Integration test | 1 |
| 8 | - | Final verification | regression |
| **Total** | | | **24+** |

**Estimated time:** 2-3 focused sessions. Tracks A and B are independent and can be parallelized.
