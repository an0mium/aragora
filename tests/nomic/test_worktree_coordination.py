"""Integration tests for worktree coordination infrastructure.

Tests for worktree_sync.py and ci_gate.py script modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Import the sync module from scripts
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


class TestWorktreeSync:
    """Tests for scripts/worktree_sync.py functionality."""

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        """Add scripts directory to path for imports."""
        sys.path.insert(0, str(SCRIPTS_DIR))
        yield
        sys.path.remove(str(SCRIPTS_DIR))

    def test_import_worktree_sync(self):
        import worktree_sync
        assert hasattr(worktree_sync, "build_report")
        assert hasattr(worktree_sync, "SyncReport")

    def test_parse_worktree_list_empty(self):
        import worktree_sync
        with patch.object(worktree_sync, "run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="")
            result = worktree_sync.parse_worktree_list()
            assert result == []

    def test_parse_worktree_list_with_worktrees(self):
        import worktree_sync
        porcelain_output = (
            "worktree /repo\n"
            "HEAD abc123\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /repo/.worktrees/core-track\n"
            "HEAD def456\n"
            "branch refs/heads/dev/core-track\n"
            "\n"
        )
        with patch.object(worktree_sync, "run_git") as mock_git:
            mock_git.return_value = MagicMock(
                returncode=0, stdout=porcelain_output
            )
            result = worktree_sync.parse_worktree_list()
            # Should skip main, include dev/core-track
            assert len(result) == 1
            assert result[0][1] == "dev/core-track"

    def test_detect_file_overlaps_none(self):
        import worktree_sync
        wt1 = worktree_sync.WorktreeStatus(
            branch="dev/core", path="/core", track="core",
            changed_files=["a.py"],
        )
        wt2 = worktree_sync.WorktreeStatus(
            branch="dev/qa", path="/qa", track="qa",
            changed_files=["b.py"],
        )
        overlaps = worktree_sync.detect_file_overlaps([wt1, wt2])
        assert overlaps == []

    def test_detect_file_overlaps_found(self):
        import worktree_sync
        wt1 = worktree_sync.WorktreeStatus(
            branch="dev/core", path="/core", track="core",
            changed_files=["shared.py", "a.py"],
        )
        wt2 = worktree_sync.WorktreeStatus(
            branch="dev/qa", path="/qa", track="qa",
            changed_files=["shared.py", "b.py"],
        )
        overlaps = worktree_sync.detect_file_overlaps([wt1, wt2])
        assert len(overlaps) == 1
        assert overlaps[0].file_path == "shared.py"

    def test_generate_recommendations_behind(self):
        import worktree_sync
        wt = worktree_sync.WorktreeStatus(
            branch="dev/core", path="/core", track="core",
            behind=10, ahead=3,
        )
        recs = worktree_sync.generate_recommendations([wt], [])
        assert any("stale" in r.lower() or "rebase" in r.lower() for r in recs)

    def test_sync_report_structure(self):
        import worktree_sync
        report = worktree_sync.SyncReport(
            timestamp="2026-02-15T00:00:00Z",
            base_branch="main",
        )
        assert report.worktrees == []
        assert report.overlaps == []
        assert report.errors == []


class TestCIGate:
    """Tests for scripts/ci_gate.py functionality."""

    @pytest.fixture(autouse=True)
    def _setup_path(self):
        """Add scripts directory to path for imports."""
        sys.path.insert(0, str(SCRIPTS_DIR))
        yield
        sys.path.remove(str(SCRIPTS_DIR))

    def test_import_ci_gate(self):
        import ci_gate
        assert hasattr(ci_gate, "CIStatus")
        assert hasattr(ci_gate, "get_ci_status")

    def test_ci_status_dataclass(self):
        import ci_gate
        status = ci_gate.CIStatus(
            is_running=True,
            run_id=12345,
            workflow_name="Tests",
            status="in_progress",
        )
        assert status.is_running is True
        assert status.run_id == 12345

    def test_check_gh_unavailable(self):
        import ci_gate
        with patch.object(ci_gate, "run_gh") as mock_gh:
            mock_gh.return_value = MagicMock(returncode=1)
            assert ci_gate.check_gh_available() is False

    def test_get_ci_status_no_runs(self):
        import ci_gate
        with patch.object(ci_gate, "run_gh") as mock_gh:
            mock_gh.return_value = MagicMock(
                returncode=0, stdout="[]"
            )
            statuses = ci_gate.get_ci_status("main")
            assert statuses == []

    def test_get_ci_status_with_runs(self):
        import ci_gate
        runs = [
            {
                "databaseId": 123,
                "name": "Tests",
                "status": "in_progress",
                "conclusion": "",
                "headBranch": "main",
                "url": "https://github.com/runs/123",
                "createdAt": "2026-02-15T00:00:00Z",
            },
            {
                "databaseId": 124,
                "name": "Smoke Tests",
                "status": "completed",
                "conclusion": "success",
                "headBranch": "main",
                "url": "https://github.com/runs/124",
                "createdAt": "2026-02-15T00:00:00Z",
            },
        ]
        with patch.object(ci_gate, "run_gh") as mock_gh:
            mock_gh.return_value = MagicMock(
                returncode=0, stdout=json.dumps(runs)
            )
            statuses = ci_gate.get_ci_status("main")
            assert len(statuses) == 2
            assert statuses[0].is_running is True
            assert statuses[1].is_running is False

    def test_any_ci_running(self):
        import ci_gate
        with patch.object(ci_gate, "get_ci_status") as mock_status:
            mock_status.return_value = [
                ci_gate.CIStatus(is_running=True, workflow_name="Tests", status="in_progress"),
            ]
            running, items = ci_gate.any_ci_running("main")
            assert running is True
            assert len(items) == 1

    def test_any_ci_not_running(self):
        import ci_gate
        with patch.object(ci_gate, "get_ci_status") as mock_status:
            mock_status.return_value = [
                ci_gate.CIStatus(is_running=False, status="completed", conclusion="success"),
            ]
            running, items = ci_gate.any_ci_running("main")
            assert running is False
            assert len(items) == 0
