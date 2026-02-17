"""Tests for IdeaCanvasStore SQLite persistence."""

from __future__ import annotations

import os
import tempfile

import pytest

from aragora.canvas.idea_store import IdeaCanvasStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Create a temporary IdeaCanvasStore."""
    monkeypatch.setattr(
        "aragora.config.resolve_db_path",
        lambda p: str(tmp_path / p),
    )
    db_path = tmp_path / "test_idea_canvas.db"
    return IdeaCanvasStore(str(db_path))


class TestIdeaCanvasStoreCRUD:
    """Basic CRUD operations."""

    def test_save_and_load(self, store):
        store.save_canvas("ic-1", "My Ideas", owner_id="user-1")
        result = store.load_canvas("ic-1")
        assert result is not None
        assert result["name"] == "My Ideas"
        assert result["owner_id"] == "user-1"

    def test_load_nonexistent(self, store):
        assert store.load_canvas("nonexistent") is None

    def test_save_updates_existing(self, store):
        store.save_canvas("ic-1", "Original", owner_id="u1")
        store.save_canvas("ic-1", "Updated", owner_id="u1")
        result = store.load_canvas("ic-1")
        assert result["name"] == "Updated"

    def test_delete(self, store):
        store.save_canvas("ic-1", "Test")
        assert store.delete_canvas("ic-1") is True
        assert store.load_canvas("ic-1") is None

    def test_delete_nonexistent(self, store):
        assert store.delete_canvas("nonexistent") is False

    def test_update_canvas_name(self, store):
        store.save_canvas("ic-1", "Original")
        result = store.update_canvas("ic-1", name="Renamed")
        assert result is not None
        assert result["name"] == "Renamed"

    def test_update_canvas_description(self, store):
        store.save_canvas("ic-1", "Test")
        result = store.update_canvas("ic-1", description="A test canvas")
        assert result["description"] == "A test canvas"

    def test_update_canvas_metadata(self, store):
        store.save_canvas("ic-1", "Test")
        result = store.update_canvas("ic-1", metadata={"stage": "ideas", "custom": True})
        assert result["metadata"]["custom"] is True

    def test_update_nonexistent(self, store):
        assert store.update_canvas("nonexistent", name="X") is None

    def test_update_no_changes(self, store):
        store.save_canvas("ic-1", "Test")
        result = store.update_canvas("ic-1")
        assert result["name"] == "Test"


class TestIdeaCanvasStoreList:
    """Listing and filtering."""

    def test_list_empty(self, store):
        assert store.list_canvases() == []

    def test_list_all(self, store):
        store.save_canvas("ic-1", "A")
        store.save_canvas("ic-2", "B")
        result = store.list_canvases()
        assert len(result) == 2

    def test_filter_by_workspace(self, store):
        store.save_canvas("ic-1", "A", workspace_id="ws-1")
        store.save_canvas("ic-2", "B", workspace_id="ws-2")
        result = store.list_canvases(workspace_id="ws-1")
        assert len(result) == 1
        assert result[0]["id"] == "ic-1"

    def test_filter_by_owner(self, store):
        store.save_canvas("ic-1", "A", owner_id="u1")
        store.save_canvas("ic-2", "B", owner_id="u2")
        result = store.list_canvases(owner_id="u1")
        assert len(result) == 1
        assert result[0]["id"] == "ic-1"

    def test_limit_and_offset(self, store):
        for i in range(5):
            store.save_canvas(f"ic-{i}", f"Canvas {i}")
        result = store.list_canvases(limit=2, offset=0)
        assert len(result) == 2
        result2 = store.list_canvases(limit=2, offset=2)
        assert len(result2) == 2

    def test_ordered_by_updated(self, store):
        import time

        store.save_canvas("ic-1", "Old")
        time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second precision
        store.save_canvas("ic-2", "New")
        result = store.list_canvases()
        # Most recently updated first
        assert result[0]["id"] == "ic-2"


class TestIdeaCanvasStoreMetadata:
    """Metadata JSON handling."""

    def test_metadata_round_trip(self, store):
        meta = {"stage": "ideas", "tags": ["test"], "count": 42}
        store.save_canvas("ic-1", "Test", metadata=meta)
        result = store.load_canvas("ic-1")
        assert result["metadata"] == meta

    def test_empty_metadata(self, store):
        store.save_canvas("ic-1", "Test")
        result = store.load_canvas("ic-1")
        assert result["metadata"] == {}

    def test_description_preserved(self, store):
        store.save_canvas("ic-1", "Test", description="My description")
        result = store.load_canvas("ic-1")
        assert result["description"] == "My description"
