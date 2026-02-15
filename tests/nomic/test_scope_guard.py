"""Tests for aragora.nomic.scope_guard."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from aragora.nomic.scope_guard import (
    DEFAULT_TRACK_SCOPES,
    PROTECTED_FILES,
    ScopeGuard,
    ScopeViolation,
    TrackScope,
)


class TestTrackScope:
    """Tests for TrackScope dataclass."""

    def test_default_scopes_have_all_tracks(self):
        expected = {"sme-track", "developer-track", "qa-track",
                    "security-track", "core-track", "infra-track"}
        assert set(DEFAULT_TRACK_SCOPES.keys()) == expected

    def test_each_scope_has_name(self):
        for name, scope in DEFAULT_TRACK_SCOPES.items():
            assert scope.name == name

    def test_each_scope_has_allowed_paths(self):
        for scope in DEFAULT_TRACK_SCOPES.values():
            assert isinstance(scope.allowed_paths, list)

    def test_each_scope_has_description(self):
        for scope in DEFAULT_TRACK_SCOPES.values():
            assert scope.description


class TestScopeGuardCheckFiles:
    """Tests for ScopeGuard.check_files."""

    def setup_method(self):
        self.guard = ScopeGuard(mode="warn")

    def test_file_in_scope_no_violation(self):
        violations = self.guard.check_files(
            ["aragora/debate/consensus.py"], "core-track"
        )
        assert not violations

    def test_file_out_of_scope(self):
        violations = self.guard.check_files(
            ["aragora/live/src/app.tsx"], "core-track"
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "outside_scope"
        assert violations[0].track == "core-track"

    def test_file_in_denied_path(self):
        violations = self.guard.check_files(
            ["aragora/debate/consensus.py"], "sme-track"
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "outside_scope"

    def test_protected_file_always_blocked(self):
        violations = self.guard.check_files(
            ["CLAUDE.md"], "core-track"
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "protected_file"
        assert violations[0].severity == "block"

    def test_multiple_files_mixed_scope(self):
        violations = self.guard.check_files(
            [
                "aragora/memory/coordinator.py",  # in scope for core-track
                "aragora/live/src/page.tsx",       # out of scope (denied)
            ],
            "core-track",
        )
        # Only the live/ file should be a violation
        assert len(violations) == 1
        assert "live" in violations[0].file_path

    def test_unknown_track_returns_empty(self):
        violations = self.guard.check_files(
            ["aragora/debate/orchestrator.py"], "nonexistent-track"
        )
        assert violations == []

    def test_qa_track_can_modify_tests(self):
        violations = self.guard.check_files(
            ["tests/nomic/test_scope_guard.py"], "qa-track"
        )
        assert not violations

    def test_qa_track_cannot_modify_source(self):
        violations = self.guard.check_files(
            ["aragora/debate/orchestrator.py"], "qa-track"
        )
        assert len(violations) >= 1

    def test_infra_track_can_modify_deploy(self):
        violations = self.guard.check_files(
            ["deploy/kubernetes/deployment.yaml"], "infra-track"
        )
        assert not violations

    def test_infra_track_cannot_modify_frontend(self):
        violations = self.guard.check_files(
            ["aragora/live/src/app.tsx"], "infra-track"
        )
        assert len(violations) >= 1

    def test_developer_track_can_modify_sdk(self):
        violations = self.guard.check_files(
            ["sdk/python/aragora_sdk/client.py"], "developer-track"
        )
        assert not violations

    def test_security_track_can_modify_rbac(self):
        violations = self.guard.check_files(
            ["aragora/rbac/checker.py"], "security-track"
        )
        assert not violations

    def test_block_mode_sets_severity(self):
        guard = ScopeGuard(mode="block")
        violations = guard.check_files(
            ["aragora/live/src/app.tsx"], "core-track"
        )
        assert violations[0].severity == "block"

    def test_warn_mode_sets_severity(self):
        guard = ScopeGuard(mode="warn")
        violations = guard.check_files(
            ["aragora/live/src/app.tsx"], "core-track"
        )
        assert violations[0].severity == "warn"


class TestScopeGuardCrossTrack:
    """Tests for cross-track overlap detection."""

    def setup_method(self):
        self.guard = ScopeGuard()

    def test_no_overlap(self):
        worktree_files = {
            "core-track": ["aragora/debate/orchestrator.py"],
            "sme-track": ["aragora/live/src/app.tsx"],
        }
        violations = self.guard.check_cross_track_overlaps(worktree_files)
        assert not violations

    def test_overlap_detected(self):
        worktree_files = {
            "core-track": ["aragora/debate/orchestrator.py", "shared.py"],
            "sme-track": ["aragora/live/src/app.tsx", "shared.py"],
        }
        violations = self.guard.check_cross_track_overlaps(worktree_files)
        assert len(violations) == 1
        assert violations[0].file_path == "shared.py"
        assert violations[0].violation_type == "cross_track"

    def test_triple_overlap_high_severity(self):
        worktree_files = {
            "core-track": ["shared.py"],
            "sme-track": ["shared.py"],
            "qa-track": ["shared.py"],
        }
        violations = self.guard.check_cross_track_overlaps(worktree_files)
        assert len(violations) == 1
        assert violations[0].severity == "block"


class TestScopeGuardBranchDetection:
    """Tests for track detection from branch names."""

    def setup_method(self):
        self.guard = ScopeGuard()

    def test_detect_dev_prefix(self):
        track = self.guard.detect_track_from_branch("dev/core-track")
        assert track == "core-track"

    def test_detect_work_prefix(self):
        track = self.guard.detect_track_from_branch("work/security-20260215")
        assert track == "security-track"

    def test_detect_sprint_prefix(self):
        track = self.guard.detect_track_from_branch("sprint/qa-track-fix-flaky")
        assert track == "qa-track"

    def test_detect_main_returns_none(self):
        track = self.guard.detect_track_from_branch("main")
        assert track is None

    def test_detect_unknown_branch_returns_none(self):
        track = self.guard.detect_track_from_branch("feature/random-thing")
        assert track is None


class TestProtectedFiles:
    """Tests for protected file list."""

    def test_claude_md_protected(self):
        assert "CLAUDE.md" in PROTECTED_FILES

    def test_env_protected(self):
        assert ".env" in PROTECTED_FILES

    def test_nomic_loop_protected(self):
        assert "scripts/nomic_loop.py" in PROTECTED_FILES
