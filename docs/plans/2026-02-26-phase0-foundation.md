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

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add watchdog dependency for Obsidian file watching"
```

---

### Task 2: Write VaultWatcher class (TDD)

**Files:**
- Create: `tests/connectors/knowledge/test_obsidian_watcher.py`
- Create: `aragora/connectors/knowledge/obsidian_watcher.py`

**Tests cover:**
- Init with valid vault, custom debounce, tag filter, ignore folders
- `_should_process()` filters: markdown only, ignores configured folders
- Change callback fires with `VaultChangeEvent`
- Debounce coalesces rapid changes to same file
- Start/stop lifecycle

**Implementation:**
- `VaultWatcher` class wrapping watchdog `Observer`
- `VaultChangeEvent` dataclass (path, change_type, timestamp)
- `_WatchdogHandler` bridges sync watchdog events to async callbacks
- Debounce via pending dict + flush timer

---

### Task 3: Register Obsidian adapter in factory

**Files:**
- Modify: `aragora/knowledge/mound/adapters/factory.py` (add to `_ADAPTER_DEFS`)
- Create: `tests/knowledge/mound/adapters/test_obsidian_factory_registration.py`

**Registration entry:**
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

---

### Task 4: Wire VaultWatcher into ObsidianAdapter

**Files:**
- Modify: `aragora/knowledge/mound/adapters/obsidian_adapter.py`
- Create: `tests/knowledge/mound/adapters/test_obsidian_live_sync.py`

**Add `_on_vault_change` method:**
- Created/modified files → trigger `sync_to_km()`
- Deleted files → log only (KM staleness handles cleanup)

---

## Track B: Extended Thinking Capture

### Task 5: Add thinking parameter to Anthropic agent (TDD)

**Files:**
- Create: `tests/agents/api_agents/test_anthropic_thinking.py`
- Modify: `aragora/agents/api_agents/anthropic.py`

**Tests cover:**
- `thinking_budget` config acceptance
- Thinking disabled by default
- `_parse_content_blocks()` separates text from thinking blocks
- Multiple thinking blocks concatenated
- Thinking stored in agent metadata via `get_metadata()`

**Implementation:**
- Add `_thinking_budget` and `_last_thinking` to `__init__`
- Add `_parse_content_blocks()` method
- Add thinking config to API payload when budget is set
- Add `get_metadata()` returning thinking trace

---

### Task 6: Surface thinking in explainability

**Files:**
- Create: `tests/debate/test_thinking_metadata.py`
- Modify: `aragora/explainability/builder.py`

**Add `_extract_thinking_traces()` to ExplanationBuilder:**
- Extracts `agent_thinking` from debate result metadata
- Returns dict mapping agent name to thinking text

---

### Task 7: Integration test

**Files:**
- Create: `tests/integration/test_thinking_e2e.py`

**End-to-end test:** agent thinking → parse → metadata → accessible in receipt chain.

---

### Task 8: Final verification

Run all affected test suites (Obsidian, Anthropic, existing regressions).

---

## Summary

| Task | Track | Description | Tests |
|------|-------|-------------|-------|
| 1 | A | Add watchdog dependency | - |
| 2 | A | VaultWatcher class with debouncing | ~10 |
| 3 | A | Factory registration | 2 |
| 4 | A | Wire watcher into adapter sync | 2 |
| 5 | B | Thinking parameter + block parsing | 6 |
| 6 | B | Thinking in explainability | 3 |
| 7 | B | Integration test | 1 |
| 8 | - | Final verification | regression |
| **Total** | | | **24+** |

Tracks A and B are independent and can be parallelized.
