"""Tests for aragora.nomic.session_manifest."""

from __future__ import annotations

import json
import os

import pytest

from aragora.nomic.session_manifest import (
    SessionEntry,
    SessionManifest,
)


@pytest.fixture
def manifest_path(tmp_path):
    """Create a temporary manifest path."""
    return tmp_path / ".aragora_sessions.yaml"


@pytest.fixture
def manifest(tmp_path, manifest_path):
    """Create a SessionManifest with a temp directory."""
    return SessionManifest(manifest_path=manifest_path, repo_root=tmp_path)


class TestSessionEntry:
    """Tests for SessionEntry dataclass."""

    def test_defaults(self):
        entry = SessionEntry(track="core")
        assert entry.track == "core"
        assert entry.agent == "claude"
        assert entry.status == "active"
        assert entry.files_claimed == []

    def test_custom_fields(self):
        entry = SessionEntry(
            track="qa",
            agent="codex",
            current_goal="Fix flaky tests",
            files_claimed=["tests/test_a.py"],
        )
        assert entry.agent == "codex"
        assert entry.current_goal == "Fix flaky tests"


class TestSessionManifestRegister:
    """Tests for registering sessions."""

    def test_register_creates_entry(self, manifest, manifest_path):
        entry = manifest.register("core", goal="Improve orchestrator")
        assert entry.track == "core"
        assert entry.current_goal == "Improve orchestrator"
        assert entry.pid == os.getpid()

    def test_register_overwrites_existing(self, manifest):
        manifest.register("core", goal="Goal 1")
        manifest.register("core", goal="Goal 2")
        active = manifest.list_active()
        assert len(active) == 1
        assert active[0].current_goal == "Goal 2"

    def test_register_multiple_tracks(self, manifest):
        manifest.register("core", goal="Core goal")
        manifest.register("qa", goal="QA goal")
        active = manifest.list_active()
        assert len(active) == 2
        tracks = {s.track for s in active}
        assert tracks == {"core", "qa"}


class TestSessionManifestDeregister:
    """Tests for deregistering sessions."""

    def test_deregister_removes_entry(self, manifest):
        manifest.register("core")
        result = manifest.deregister("core")
        assert result is True
        assert len(manifest.list_active()) == 0

    def test_deregister_nonexistent_returns_false(self, manifest):
        result = manifest.deregister("nonexistent")
        assert result is False


class TestSessionManifestUpdateGoal:
    """Tests for updating session goals."""

    def test_update_goal(self, manifest):
        manifest.register("core", goal="Original")
        result = manifest.update_goal("core", "Updated goal")
        assert result is True
        active = manifest.list_active()
        assert active[0].current_goal == "Updated goal"

    def test_update_nonexistent_returns_false(self, manifest):
        result = manifest.update_goal("nonexistent", "goal")
        assert result is False


class TestSessionManifestClaimFiles:
    """Tests for file claiming."""

    def test_claim_files_no_conflict(self, manifest):
        manifest.register("core")
        conflicts = manifest.claim_files("core", ["a.py", "b.py"])
        assert conflicts == []

    def test_claim_files_with_conflict(self, manifest):
        manifest.register("core")
        manifest.register("qa")
        manifest.claim_files("core", ["shared.py"])
        conflicts = manifest.claim_files("qa", ["shared.py"])
        assert len(conflicts) == 1
        assert "shared.py" in conflicts[0]
        assert "core" in conflicts[0]

    def test_claim_files_cumulative(self, manifest):
        manifest.register("core")
        manifest.claim_files("core", ["a.py"])
        manifest.claim_files("core", ["b.py"])
        active = manifest.list_active()
        assert set(active[0].files_claimed) == {"a.py", "b.py"}


class TestSessionManifestOverlaps:
    """Tests for overlap detection."""

    def test_no_overlaps(self, manifest):
        manifest.register("core")
        manifest.register("qa")
        overlaps = manifest.detect_overlapping_goals()
        assert overlaps == []

    def test_file_overlap_detected(self, manifest):
        manifest.register("core")
        manifest.register("qa")
        manifest.claim_files("core", ["shared.py"])
        manifest.claim_files("qa", ["shared.py"])
        overlaps = manifest.detect_overlapping_goals()
        assert len(overlaps) == 1
        assert overlaps[0][0] in ("core", "qa")
        assert overlaps[0][1] in ("core", "qa")


class TestSessionManifestCleanup:
    """Tests for stale session cleanup."""

    def test_cleanup_removes_old_sessions(self, manifest):
        manifest.register("old-track", goal="Old goal")
        # Manually set started_at to 48 hours ago
        data = manifest._load()
        from datetime import datetime, timedelta, timezone
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        data["sessions"][0]["started_at"] = old_time
        manifest._save(data)

        cleaned = manifest.cleanup_stale(max_age_hours=24)
        assert cleaned == 1
        assert len(manifest.list_active()) == 0

    def test_cleanup_keeps_recent_sessions(self, manifest):
        manifest.register("core", goal="Fresh goal")
        cleaned = manifest.cleanup_stale(max_age_hours=24)
        assert cleaned == 0
        assert len(manifest.list_active()) == 1


class TestSessionManifestFallback:
    """Tests for JSON fallback when PyYAML is unavailable."""

    def test_json_fallback(self, tmp_path):
        """Test that manifest works with JSON when YAML unavailable."""
        json_path = tmp_path / ".aragora_sessions.json"
        # Create a manifest with JSON data
        json_path.write_text(json.dumps({
            "sessions": [
                {
                    "track": "core",
                    "worktree": "/tmp/core",
                    "agent": "claude",
                    "current_goal": "Test",
                    "started_at": "2026-02-15T00:00:00+00:00",
                    "files_claimed": [],
                    "status": "active",
                    "pid": 0,
                }
            ]
        }))

        manifest = SessionManifest(
            manifest_path=tmp_path / ".aragora_sessions.yaml",
            repo_root=tmp_path,
        )
        # The fallback should try .json if .yaml doesn't exist
        # Since we're testing the _load_fallback path
        data = manifest._load_fallback()
        assert len(data["sessions"]) == 1
