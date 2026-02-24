"""Tests for WorktreeAuditor - worktree validation and health checks."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.worktree_auditor import (
    AuditFinding,
    AuditReport,
    AuditorConfig,
    WorktreeAuditor,
    WorktreeStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repo_path(tmp_path):
    """Create a mock git repository structure."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    return tmp_path


@pytest.fixture
def auditor(repo_path):
    """Create an auditor with default config."""
    return WorktreeAuditor(repo_path=repo_path)


@pytest.fixture
def auditor_with_worktrees(repo_path):
    """Create an auditor with mock worktree directories."""
    wt_dir = repo_path / ".worktrees"
    wt_dir.mkdir()

    # Create two mock worktrees
    wt1 = wt_dir / "dev-sme-1"
    wt1.mkdir()
    (wt1 / ".git").write_text(f"gitdir: {repo_path}/.git/worktrees/dev-sme-1\n")

    wt2 = wt_dir / "dev-qa-1"
    wt2.mkdir()
    (wt2 / ".git").write_text(f"gitdir: {repo_path}/.git/worktrees/dev-qa-1\n")

    return WorktreeAuditor(repo_path=repo_path)


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestPathValidation:
    """Tests for worktree path checks."""

    def test_missing_worktree_dir_is_info(self, auditor):
        """Missing worktree dir should be info, not error (auto-created on use)."""
        findings = auditor._check_base_directory()

        info_findings = [f for f in findings if f.severity == "info"]
        assert len(info_findings) >= 1
        assert any("does not exist" in f.message for f in info_findings)

    def test_existing_worktree_dir(self, repo_path):
        """Existing worktree dir should pass with info."""
        wt_dir = repo_path / ".worktrees"
        wt_dir.mkdir()

        auditor = WorktreeAuditor(repo_path=repo_path)
        findings = auditor._check_base_directory()

        assert all(f.severity in ("info",) for f in findings)

    def test_worktree_path_is_file_not_dir(self, repo_path):
        """File where directory expected should be critical."""
        wt_path = repo_path / ".worktrees"
        wt_path.write_text("not a directory")

        auditor = WorktreeAuditor(repo_path=repo_path)
        findings = auditor._check_base_directory()

        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 1
        assert "not a directory" in critical[0].message


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermissionChecks:
    """Tests for directory permission validation."""

    def test_normal_permissions_pass(self, repo_path):
        """Standard 755 permissions should pass."""
        wt_dir = repo_path / ".worktrees"
        wt_dir.mkdir(mode=0o755)

        auditor = WorktreeAuditor(repo_path=repo_path)
        findings = auditor._check_base_directory()

        errors = [f for f in findings if f.severity in ("error", "critical")]
        assert len(errors) == 0

    def test_restrictive_permissions_flagged(self, repo_path):
        """Overly restrictive permissions should be flagged."""
        wt_dir = repo_path / ".worktrees"
        wt_dir.mkdir(mode=0o000)

        auditor = WorktreeAuditor(repo_path=repo_path)
        findings = auditor._check_base_directory()

        # Restore permissions for cleanup
        wt_dir.chmod(0o755)

        # Should have an error about restrictive permissions
        perm_findings = [f for f in findings if f.category == "permission"]
        assert len(perm_findings) >= 1


# =============================================================================
# Disk Space Tests
# =============================================================================


class TestDiskSpaceChecks:
    """Tests for disk space validation."""

    def test_sufficient_disk_space(self, auditor):
        """Should report info when disk space is sufficient."""
        findings = auditor._check_disk_space()

        # On test machines, disk space should be sufficient
        info = [f for f in findings if f.severity == "info"]
        assert len(info) >= 1
        assert any("Disk space" in f.message for f in info)

    def test_insufficient_disk_space(self, repo_path):
        """Should report error when disk space is below threshold."""
        config = AuditorConfig(
            min_disk_space_bytes=999 * 1024 * 1024 * 1024 * 1024  # 999 TB (unreachable)
        )
        auditor = WorktreeAuditor(repo_path=repo_path, config=config)

        findings = auditor._check_disk_space()

        errors = [f for f in findings if f.severity == "error"]
        assert len(errors) >= 1
        assert any("Insufficient" in f.message for f in errors)


# =============================================================================
# Git Configuration Tests
# =============================================================================


class TestGitConfig:
    """Tests for git configuration checks."""

    def test_not_a_git_repo(self, tmp_path):
        """Should flag non-git directory as critical."""
        auditor = WorktreeAuditor(repo_path=tmp_path)

        # Mock git rev-parse to fail
        with patch.object(auditor, "_run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=128, stdout="", stderr="")
            findings = auditor._check_git_config()

        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) >= 1
        assert any("Not a git repository" in f.message for f in critical)

    def test_valid_git_repo(self, repo_path):
        """Should pass for a valid git repo."""
        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(auditor, "_run_git") as mock_git, \
             patch.object(auditor, "_list_git_worktrees", return_value=[]):
            mock_git.side_effect = [
                MagicMock(returncode=0, stdout=".git\n"),  # rev-parse --git-dir
                MagicMock(returncode=0, stdout="git version 2.40.0\n"),  # --version
                MagicMock(returncode=0, stdout=""),  # worktree list --porcelain
            ]
            findings = auditor._check_git_config()

        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 0


# =============================================================================
# Isolation Validation Tests
# =============================================================================


class TestIsolationValidation:
    """Tests for worktree isolation checks."""

    def test_no_worktrees_no_findings(self, auditor):
        """No isolation findings when fewer than 2 worktrees."""
        with patch.object(auditor, "_list_git_worktrees", return_value=[]):
            findings = auditor._check_isolation()

        assert len(findings) == 0

    def test_symlink_between_worktrees_flagged(self, repo_path):
        """Symlink from one worktree to another should be flagged."""
        wt_dir = repo_path / ".worktrees"
        wt_dir.mkdir()

        wt1 = wt_dir / "dev-sme-1"
        wt1.mkdir()
        wt2 = wt_dir / "dev-qa-1"
        wt2.mkdir()

        # Create a real .env in wt2 and symlink from wt1
        (wt2 / ".env").write_text("SECRET=123")
        (wt1 / ".env").symlink_to(wt2 / ".env")

        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(
            auditor,
            "_list_git_worktrees",
            return_value=[
                {"path": str(repo_path), "branch": "main"},
                {"path": str(wt1), "branch": "dev/sme-1"},
                {"path": str(wt2), "branch": "dev/qa-1"},
            ],
        ):
            findings = auditor._check_isolation()

        isolation_errors = [
            f for f in findings if f.category == "isolation" and f.severity == "error"
        ]
        assert len(isolation_errors) >= 1
        assert any("Isolation violation" in f.message for f in isolation_errors)

    def test_validate_isolation_returns_bool(self, auditor):
        """validate_isolation() should return True when no violations."""
        with patch.object(auditor, "_check_isolation", return_value=[]):
            assert auditor.validate_isolation() is True

    def test_validate_isolation_returns_false_on_violation(self, auditor):
        """validate_isolation() should return False when violations exist."""
        violation = AuditFinding(
            severity="error",
            category="isolation",
            message="Shared state detected",
        )
        with patch.object(auditor, "_check_isolation", return_value=[violation]):
            assert auditor.validate_isolation() is False


# =============================================================================
# Full Audit Tests
# =============================================================================


class TestFullAudit:
    """Tests for the complete audit() method."""

    def test_audit_returns_report(self, repo_path):
        """audit() should return an AuditReport."""
        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(auditor, "_list_git_worktrees", return_value=[]):
            report = auditor.audit()

        assert isinstance(report, AuditReport)
        assert report.repo_path == str(repo_path)
        assert report.timestamp  # ISO format string
        assert isinstance(report.findings, list)
        assert isinstance(report.healthy, bool)

    def test_audit_report_healthy_when_no_errors(self, repo_path):
        """Report should be healthy when no errors or critical findings."""
        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(auditor, "_check_base_directory", return_value=[]), \
             patch.object(auditor, "_check_disk_space", return_value=[]), \
             patch.object(auditor, "_check_git_config", return_value=[]), \
             patch.object(auditor, "_check_isolation", return_value=[]), \
             patch.object(auditor, "_list_git_worktrees", return_value=[]):
            report = auditor.audit()

        assert report.healthy is True
        assert report.error_count == 0

    def test_audit_report_unhealthy_with_errors(self, repo_path):
        """Report should be unhealthy when errors exist."""
        auditor = WorktreeAuditor(repo_path=repo_path)

        error_finding = AuditFinding(
            severity="error",
            category="disk",
            message="Insufficient disk space",
        )

        with patch.object(auditor, "_check_base_directory", return_value=[]), \
             patch.object(auditor, "_check_disk_space", return_value=[error_finding]), \
             patch.object(auditor, "_check_git_config", return_value=[]), \
             patch.object(auditor, "_check_isolation", return_value=[]), \
             patch.object(auditor, "_list_git_worktrees", return_value=[]):
            report = auditor.audit()

        assert report.healthy is False
        assert report.error_count == 1

    def test_audit_summary_content(self, repo_path):
        """Audit summary should contain worktree count and status."""
        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(auditor, "_list_git_worktrees", return_value=[
            {"path": str(repo_path), "branch": "main"},
        ]):
            report = auditor.audit()

        assert "0 worktrees" in report.summary


# =============================================================================
# Status Reporting Tests
# =============================================================================


class TestStatusReporting:
    """Tests for worktree status reporting."""

    def test_get_status_empty(self, auditor):
        """Should return empty list when no worktrees."""
        with patch.object(auditor, "_list_git_worktrees", return_value=[
            {"path": str(auditor.repo_path), "branch": "main"},
        ]):
            statuses = auditor.get_status()

        assert statuses == []

    def test_get_status_missing_worktree(self, auditor):
        """Should report missing status for non-existent worktree dir."""
        with patch.object(auditor, "_list_git_worktrees", return_value=[
            {"path": str(auditor.repo_path), "branch": "main"},
            {"path": "/nonexistent/path", "branch": "dev/sme-1"},
        ]):
            statuses = auditor.get_status()

        assert len(statuses) == 1
        assert statuses[0].status == "missing"
        assert statuses[0].branch_name == "dev/sme-1"

    def test_get_status_healthy_worktree(self, repo_path):
        """Should report healthy status for well-configured worktree."""
        wt_dir = repo_path / ".worktrees"
        wt_dir.mkdir()
        wt = wt_dir / "dev-sme-1"
        wt.mkdir()

        # Create .git file pointing to main repo
        git_wt_dir = repo_path / ".git" / "worktrees" / "dev-sme-1"
        git_wt_dir.mkdir(parents=True)
        (wt / ".git").write_text(f"gitdir: {git_wt_dir}\n")

        auditor = WorktreeAuditor(repo_path=repo_path)

        with patch.object(auditor, "_list_git_worktrees", return_value=[
            {"path": str(repo_path), "branch": "main"},
            {"path": str(wt), "branch": "dev/sme-1"},
        ]), patch.object(auditor, "_run_git") as mock_git:
            # git status --porcelain returns empty (no changes)
            mock_git.return_value = MagicMock(
                returncode=0,
                stdout="",
            )
            statuses = auditor.get_status()

        assert len(statuses) == 1
        assert statuses[0].status == "healthy"
        assert statuses[0].branch_name == "dev/sme-1"
        assert statuses[0].has_uncommitted_changes is False
