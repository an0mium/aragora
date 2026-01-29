"""Tests for aragora.sync.models."""

from datetime import datetime

import pytest

from aragora.sync.models import (
    FileChange,
    FileChangeType,
    SyncConfig,
    SyncResult,
    SyncState,
    SyncStatus,
)


# --- Import verification ---


class TestImports:
    """Verify all public model symbols are importable."""

    def test_file_change_importable(self):
        assert FileChange is not None

    def test_file_change_type_importable(self):
        assert FileChangeType is not None

    def test_sync_config_importable(self):
        assert SyncConfig is not None

    def test_sync_state_importable(self):
        assert SyncState is not None

    def test_sync_result_importable(self):
        assert SyncResult is not None

    def test_sync_status_importable(self):
        assert SyncStatus is not None


# --- FileChangeType enum ---


class TestFileChangeType:
    """Tests for FileChangeType enum values and behaviour."""

    def test_added_value(self):
        assert FileChangeType.ADDED.value == "added"

    def test_modified_value(self):
        assert FileChangeType.MODIFIED.value == "modified"

    def test_deleted_value(self):
        assert FileChangeType.DELETED.value == "deleted"

    def test_member_count(self):
        assert len(FileChangeType) == 3

    def test_is_str_enum(self):
        assert isinstance(FileChangeType.ADDED, str)


# --- SyncStatus enum ---


class TestSyncStatus:
    """Tests for SyncStatus enum values."""

    @pytest.mark.parametrize(
        "member,value",
        [
            (SyncStatus.IDLE, "idle"),
            (SyncStatus.SCANNING, "scanning"),
            (SyncStatus.SYNCING, "syncing"),
            (SyncStatus.COMPLETED, "completed"),
            (SyncStatus.FAILED, "failed"),
            (SyncStatus.PAUSED, "paused"),
        ],
    )
    def test_values(self, member, value):
        assert member.value == value

    def test_member_count(self):
        assert len(SyncStatus) == 6

    def test_is_str_enum(self):
        assert isinstance(SyncStatus.IDLE, str)


# --- FileChange dataclass ---


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_required_fields(self):
        fc = FileChange(
            path="readme.md",
            absolute_path="/tmp/readme.md",
            change_type=FileChangeType.ADDED,
        )
        assert fc.path == "readme.md"
        assert fc.absolute_path == "/tmp/readme.md"
        assert fc.change_type is FileChangeType.ADDED

    def test_default_detected_at_is_datetime(self):
        fc = FileChange(
            path="a.txt",
            absolute_path="/a.txt",
            change_type=FileChangeType.MODIFIED,
        )
        assert isinstance(fc.detected_at, datetime)

    def test_optional_defaults_are_none(self):
        fc = FileChange(
            path="a.txt",
            absolute_path="/a.txt",
            change_type=FileChangeType.DELETED,
        )
        assert fc.size_bytes is None
        assert fc.mime_type is None
        assert fc.extension is None
        assert fc.content_hash is None
        assert fc.document_id is None
        assert fc.error is None

    def test_processed_defaults_false(self):
        fc = FileChange(
            path="a.txt",
            absolute_path="/a.txt",
            change_type=FileChangeType.ADDED,
        )
        assert fc.processed is False

    def test_to_dict_keys(self):
        fc = FileChange(
            path="a.txt",
            absolute_path="/a.txt",
            change_type=FileChangeType.ADDED,
        )
        d = fc.to_dict()
        expected_keys = {
            "path",
            "absolute_path",
            "change_type",
            "detected_at",
            "size_bytes",
            "mime_type",
            "extension",
            "content_hash",
            "processed",
            "document_id",
            "error",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_change_type_is_string(self):
        fc = FileChange(
            path="a.txt",
            absolute_path="/a.txt",
            change_type=FileChangeType.MODIFIED,
        )
        assert fc.to_dict()["change_type"] == "modified"


# --- SyncConfig dataclass ---


class TestSyncConfig:
    """Tests for SyncConfig default values."""

    def test_defaults(self):
        cfg = SyncConfig()
        assert cfg.debounce_ms == 1000
        assert cfg.poll_interval_ms == 500
        assert cfg.max_file_size_mb == 50.0
        assert cfg.delete_missing is False
        assert cfg.update_existing is True
        assert cfg.hash_check is True
        assert cfg.batch_size == 10
        assert cfg.max_concurrent == 4

    def test_exclude_patterns_default(self):
        cfg = SyncConfig()
        assert isinstance(cfg.exclude_patterns, list)
        assert len(cfg.exclude_patterns) == 8
        assert "**/.git/**" in cfg.exclude_patterns

    def test_include_patterns_default_none(self):
        cfg = SyncConfig()
        assert cfg.include_patterns is None

    def test_custom_values(self):
        cfg = SyncConfig(debounce_ms=500, batch_size=20, delete_missing=True)
        assert cfg.debounce_ms == 500
        assert cfg.batch_size == 20
        assert cfg.delete_missing is True


# --- SyncState dataclass ---


class TestSyncState:
    """Tests for SyncState dataclass."""

    def test_required_fields(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        assert state.workspace_id == "ws_1"
        assert state.root_path == "/data"

    def test_status_default(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        assert state.status is SyncStatus.IDLE

    def test_numeric_defaults(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        assert state.total_files == 0
        assert state.sync_count == 0
        assert state.error_count == 0

    def test_dict_defaults_are_empty(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        assert state.known_files == {}
        assert state.document_map == {}

    def test_optional_defaults_none(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        assert state.last_sync_at is None
        assert state.last_change_at is None
        assert state.last_error is None

    def test_to_dict_keys(self):
        state = SyncState(workspace_id="ws_1", root_path="/data")
        d = state.to_dict()
        assert d["workspace_id"] == "ws_1"
        assert d["status"] == "idle"
        assert d["last_sync_at"] is None


# --- SyncResult dataclass ---


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_required_fields(self):
        res = SyncResult(workspace_id="ws_1", root_path="/data", success=True)
        assert res.workspace_id == "ws_1"
        assert res.success is True

    def test_counter_defaults(self):
        res = SyncResult(workspace_id="ws_1", root_path="/data", success=True)
        assert res.files_added == 0
        assert res.files_modified == 0
        assert res.files_deleted == 0
        assert res.files_unchanged == 0
        assert res.files_failed == 0

    def test_list_defaults_empty(self):
        res = SyncResult(workspace_id="ws_1", root_path="/data", success=True)
        assert res.changes == []
        assert res.errors == []

    def test_total_processed_property(self):
        res = SyncResult(
            workspace_id="ws_1",
            root_path="/data",
            success=True,
            files_added=2,
            files_modified=3,
            files_deleted=1,
        )
        assert res.total_processed == 6

    def test_duration_seconds_none_when_incomplete(self):
        res = SyncResult(workspace_id="ws_1", root_path="/data", success=True)
        assert res.duration_seconds is None

    def test_duration_seconds_computed(self):
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 0, 0, 5)
        res = SyncResult(
            workspace_id="ws_1",
            root_path="/data",
            success=True,
            started_at=start,
            completed_at=end,
        )
        assert res.duration_seconds == 5.0

    def test_to_dict_includes_total_processed(self):
        res = SyncResult(
            workspace_id="ws_1",
            root_path="/data",
            success=True,
            files_added=1,
        )
        d = res.to_dict()
        assert d["total_processed"] == 1
        assert d["success"] is True
