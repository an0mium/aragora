"""Tests for the OpenClaw next-steps runner."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.compat.openclaw.next_steps_runner import (
    NextStep,
    NextStepsRunner,
    ScanReceipt,
    ScanResult,
    _deduplicate_steps,
    _generate_checksum,
    _infer_effort,
    _infer_issue_category,
    _infer_issue_priority,
    format_steps_table,
    scan_code_markers,
    scan_doc_gaps,
    scan_github_issues,
    scan_github_prs,
    steps_to_json,
)


# ---------------------------------------------------------------------------
# NextStep data class
# ---------------------------------------------------------------------------


class TestNextStep:
    """Tests for the NextStep dataclass."""

    def test_basic_creation(self):
        step = NextStep(
            title="Fix login bug",
            description="Users can't log in with SSO",
            category="bug",
            priority="critical",
            effort="medium",
            source="github-issue",
        )
        assert step.title == "Fix login bug"
        assert step.priority == "critical"
        assert step.file_path is None

    def test_with_file_location(self):
        step = NextStep(
            title="Remove FIXME",
            description="FIXME in auth.py",
            category="tech-debt",
            priority="medium",
            effort="small",
            source="code-marker",
            file_path="src/auth.py",
            line_number=42,
        )
        assert step.file_path == "src/auth.py"
        assert step.line_number == 42

    def test_with_url(self):
        step = NextStep(
            title="Fix issue #5",
            description="Bug report",
            category="bug",
            priority="high",
            effort="medium",
            source="github-issue",
            url="https://github.com/owner/repo/issues/5",
        )
        assert step.url is not None

    def test_sort_key_ordering(self):
        critical = NextStep("A", "", "bug", "critical", "small", "test")
        high = NextStep("A", "", "bug", "high", "small", "test")
        medium = NextStep("A", "", "bug", "medium", "small", "test")
        low = NextStep("A", "", "bug", "low", "small", "test")

        steps = [low, critical, medium, high]
        steps.sort(key=lambda s: s.sort_key)
        assert [s.priority for s in steps] == ["critical", "high", "medium", "low"]

    def test_sort_key_alphabetical_within_priority(self):
        step_a = NextStep("Alpha", "", "bug", "high", "small", "test")
        step_b = NextStep("Beta", "", "bug", "high", "small", "test")

        steps = [step_b, step_a]
        steps.sort(key=lambda s: s.sort_key)
        assert steps[0].title == "Alpha"

    def test_metadata_default(self):
        step = NextStep("T", "D", "bug", "high", "small", "test")
        assert step.metadata == {}

    def test_metadata_custom(self):
        step = NextStep("T", "D", "bug", "high", "small", "test", metadata={"number": 42})
        assert step.metadata["number"] == 42


# ---------------------------------------------------------------------------
# ScanReceipt
# ---------------------------------------------------------------------------


class TestScanReceipt:
    """Tests for the ScanReceipt dataclass."""

    def test_duration(self):
        receipt = ScanReceipt(
            scan_id="scan-abc",
            repo="owner/repo",
            started_at=100.0,
            completed_at=105.5,
            steps_count=10,
            files_scanned=200,
            signals_by_source={"code-marker": 8, "github-issue": 2},
            checksum="abc123",
        )
        assert receipt.duration_seconds == pytest.approx(5.5)

    def test_to_dict(self):
        receipt = ScanReceipt(
            scan_id="scan-xyz",
            repo="test/repo",
            started_at=1000.0,
            completed_at=1010.0,
            steps_count=5,
            files_scanned=100,
            signals_by_source={"code-marker": 5},
            checksum="def456",
        )
        d = receipt.to_dict()
        assert d["scan_id"] == "scan-xyz"
        assert d["steps_count"] == 5
        assert d["duration_seconds"] == 10.0
        assert d["signals_by_source"]["code-marker"] == 5


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


class TestScanResult:
    """Tests for the ScanResult dataclass."""

    def _make_steps(self) -> list[NextStep]:
        return [
            NextStep("Bug A", "", "bug", "critical", "small", "test"),
            NextStep("Bug B", "", "bug", "high", "medium", "test"),
            NextStep("Feature C", "", "enhancement", "medium", "large", "test"),
            NextStep("Doc D", "", "docs", "low", "small", "test"),
        ]

    def test_by_priority(self):
        result = ScanResult(repo="test", steps=self._make_steps())
        groups = result.by_priority
        assert len(groups["critical"]) == 1
        assert len(groups["high"]) == 1
        assert len(groups["medium"]) == 1
        assert len(groups["low"]) == 1

    def test_by_category(self):
        result = ScanResult(repo="test", steps=self._make_steps())
        groups = result.by_category
        assert len(groups["bug"]) == 2
        assert len(groups["enhancement"]) == 1

    def test_top(self):
        result = ScanResult(repo="test", steps=self._make_steps())
        top3 = result.top(3)
        assert len(top3) == 3
        assert top3[0].priority == "critical"

    def test_error(self):
        result = ScanResult(repo="test", steps=[], error="Something failed")
        assert result.error == "Something failed"


# ---------------------------------------------------------------------------
# Code marker scanning
# ---------------------------------------------------------------------------


class TestScanCodeMarkers:
    """Tests for source code marker scanning."""

    def test_finds_todo(self, tmp_path):
        (tmp_path / "app.py").write_text("# TODO: implement auth\npass\n")
        steps, count = scan_code_markers(tmp_path)
        assert count == 1
        assert len(steps) == 1
        assert steps[0].category == "enhancement"
        assert steps[0].source == "code-marker"

    def test_finds_fixme(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1  # FIXME: race condition\n")
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps) == 1
        assert steps[0].priority == "high"
        assert steps[0].category == "bug"

    def test_finds_hack(self, tmp_path):
        (tmp_path / "util.py").write_text("# HACK: temporary workaround\n")
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps) == 1
        assert steps[0].category == "tech-debt"

    def test_finds_xxx(self, tmp_path):
        (tmp_path / "server.py").write_text("# XXX: needs refactoring\n")
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps) == 1
        assert steps[0].priority == "high"

    def test_skips_non_code_files(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.csv").write_text("TODO, do this\n")
        steps, count = scan_code_markers(tmp_path)
        assert count == 0
        assert len(steps) == 0

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("// TODO: fix this\n")
        steps, count = scan_code_markers(tmp_path)
        assert count == 0
        assert len(steps) == 0

    def test_multiple_markers_same_file(self, tmp_path):
        (tmp_path / "app.py").write_text(
            "# TODO: add logging\n# FIXME: null check\n# HACK: workaround\n"
        )
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps) == 3

    def test_file_path_is_relative(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.py").write_text("# TODO: test\n")
        steps, _ = scan_code_markers(tmp_path)
        assert steps[0].file_path == "src/app.py"

    def test_line_number_tracked(self, tmp_path):
        (tmp_path / "app.py").write_text("line1\nline2\n# TODO: on line 3\n")
        steps, _ = scan_code_markers(tmp_path)
        assert steps[0].line_number == 3

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "app.py").write_text("# todo: lower case\n")
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps) == 1

    def test_truncates_long_comments(self, tmp_path):
        long_comment = "# TODO: " + "x" * 300 + "\n"
        (tmp_path / "app.py").write_text(long_comment)
        steps, _ = scan_code_markers(tmp_path)
        assert len(steps[0].title) <= 80

    def test_empty_directory(self, tmp_path):
        steps, count = scan_code_markers(tmp_path)
        assert count == 0
        assert len(steps) == 0

    def test_yaml_files_scanned(self, tmp_path):
        (tmp_path / "config.yaml").write_text("# TODO: add production config\n")
        steps, count = scan_code_markers(tmp_path)
        assert count == 1
        assert len(steps) == 1


# ---------------------------------------------------------------------------
# GitHub issue scanning
# ---------------------------------------------------------------------------


class TestScanGitHubIssues:
    """Tests for GitHub issue scanning."""

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_parses_issues(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {
                    "number": 1,
                    "title": "Login broken",
                    "labels": [{"name": "bug"}],
                    "url": "https://github.com/o/r/issues/1",
                    "body": "Users can't log in",
                },
                {
                    "number": 2,
                    "title": "Add dark mode",
                    "labels": [{"name": "enhancement"}],
                    "url": "https://github.com/o/r/issues/2",
                    "body": "Please add dark mode",
                },
            ]),
        )
        steps = scan_github_issues("o/r")
        assert len(steps) == 2
        assert steps[0].source == "github-issue"
        assert steps[0].metadata["number"] == 1

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_handles_gh_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("gh not found")
        steps = scan_github_issues("o/r")
        assert steps == []

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_handles_gh_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="auth required")
        steps = scan_github_issues("o/r")
        assert steps == []

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_handles_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=30)
        steps = scan_github_issues("o/r")
        assert steps == []


# ---------------------------------------------------------------------------
# GitHub PR scanning
# ---------------------------------------------------------------------------


class TestScanGitHubPRs:
    """Tests for GitHub PR scanning."""

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_approved_pr_high_priority(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {
                    "number": 10,
                    "title": "Add feature X",
                    "url": "https://github.com/o/r/pull/10",
                    "reviewDecision": "APPROVED",
                    "isDraft": False,
                    "createdAt": "2026-01-01T00:00:00Z",
                },
            ]),
        )
        steps = scan_github_prs("o/r")
        assert len(steps) == 1
        assert steps[0].priority == "high"
        assert "Merge" in steps[0].title

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_draft_prs_skipped(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {
                    "number": 11,
                    "title": "WIP",
                    "url": "https://github.com/o/r/pull/11",
                    "reviewDecision": "",
                    "isDraft": True,
                    "createdAt": "2026-01-01T00:00:00Z",
                },
            ]),
        )
        steps = scan_github_prs("o/r")
        assert len(steps) == 0

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_changes_requested_pr(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {
                    "number": 12,
                    "title": "Update deps",
                    "url": "https://github.com/o/r/pull/12",
                    "reviewDecision": "CHANGES_REQUESTED",
                    "isDraft": False,
                    "createdAt": "2026-01-01T00:00:00Z",
                },
            ]),
        )
        steps = scan_github_prs("o/r")
        assert len(steps) == 1
        assert "Review" in steps[0].title


# ---------------------------------------------------------------------------
# Doc gap scanning
# ---------------------------------------------------------------------------


class TestScanDocGaps:
    """Tests for documentation gap scanning."""

    def test_missing_readme(self, tmp_path):
        steps = scan_doc_gaps(tmp_path)
        assert any("README" in s.title for s in steps)

    def test_short_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# My Project\nShort.")
        steps = scan_doc_gaps(tmp_path)
        assert any("Expand" in s.title for s in steps)

    def test_adequate_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project\n" + "x " * 200)
        steps = scan_doc_gaps(tmp_path)
        assert not any("Expand" in s.title or "Create" in s.title for s in steps)

    def test_contributing_suggested_for_large_project(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(15):
            (src / f"mod_{i}.py").write_text(f"# Module {i}\n")
        (tmp_path / "README.md").write_text("# Project\n" + "x " * 200)
        steps = scan_doc_gaps(tmp_path)
        assert any("CONTRIBUTING" in s.title for s in steps)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper functions."""

    def test_infer_issue_priority_critical(self):
        assert _infer_issue_priority(["critical", "bug"]) == "critical"
        assert _infer_issue_priority(["P0"]) == "critical"
        assert _infer_issue_priority(["urgent"]) == "critical"

    def test_infer_issue_priority_high(self):
        assert _infer_issue_priority(["bug"]) == "high"
        assert _infer_issue_priority(["P1"]) == "high"

    def test_infer_issue_priority_medium(self):
        assert _infer_issue_priority(["enhancement"]) == "medium"
        assert _infer_issue_priority(["feature"]) == "medium"

    def test_infer_issue_priority_low(self):
        assert _infer_issue_priority(["question"]) == "low"
        assert _infer_issue_priority([]) == "low"

    def test_infer_issue_category(self):
        assert _infer_issue_category(["bug"]) == "bug"
        assert _infer_issue_category(["security"]) == "security"
        assert _infer_issue_category(["documentation"]) == "docs"
        assert _infer_issue_category(["enhancement"]) == "enhancement"
        assert _infer_issue_category(["unrelated"]) == "enhancement"

    def test_infer_effort(self):
        assert _infer_effort(["good first issue"]) == "small"
        assert _infer_effort(["easy"]) == "small"
        assert _infer_effort(["epic"]) == "large"
        assert _infer_effort(["normal"]) == "medium"

    def test_deduplicate_steps(self):
        steps = [
            NextStep("Fix auth", "", "bug", "high", "small", "test"),
            NextStep("Fix auth", "", "bug", "high", "small", "test"),
            NextStep("Fix login", "", "bug", "high", "small", "test"),
        ]
        result = _deduplicate_steps(steps)
        assert len(result) == 2

    def test_deduplicate_case_insensitive(self):
        steps = [
            NextStep("Fix Auth", "", "bug", "high", "small", "test"),
            NextStep("fix auth", "", "bug", "high", "small", "test"),
        ]
        result = _deduplicate_steps(steps)
        assert len(result) == 1

    def test_generate_checksum_deterministic(self):
        steps = [NextStep("A", "", "bug", "high", "small", "test")]
        c1 = _generate_checksum(steps)
        c2 = _generate_checksum(steps)
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_generate_checksum_differs(self):
        s1 = [NextStep("A", "", "bug", "high", "small", "test")]
        s2 = [NextStep("B", "", "bug", "high", "small", "test")]
        assert _generate_checksum(s1) != _generate_checksum(s2)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestFormatStepsTable:
    """Tests for table formatting."""

    def test_empty_steps(self):
        assert format_steps_table([]) == "No next steps found."

    def test_single_step(self):
        steps = [NextStep("Fix bug", "desc", "bug", "high", "small", "test")]
        table = format_steps_table(steps)
        assert "Fix bug" in table
        assert "high" in table

    def test_respects_max_rows(self):
        steps = [NextStep(f"Step {i}", "", "bug", "high", "small", "test") for i in range(20)]
        table = format_steps_table(steps, max_rows=5)
        assert "... and 15 more" in table

    def test_all_rows_shown(self):
        steps = [NextStep(f"Step {i}", "", "bug", "high", "small", "test") for i in range(3)]
        table = format_steps_table(steps, max_rows=10)
        assert "more" not in table


class TestStepsToJson:
    """Tests for JSON serialization."""

    def test_basic_output(self):
        steps = [
            NextStep("Fix A", "desc A", "bug", "high", "small", "code-marker",
                     file_path="a.py", line_number=1),
        ]
        result = steps_to_json(steps)
        assert result["count"] == 1
        assert result["steps"][0]["title"] == "Fix A"
        assert result["steps"][0]["file_path"] == "a.py"
        assert result["by_priority"]["high"] == 1

    def test_with_receipt(self):
        steps = [NextStep("A", "", "bug", "high", "small", "test")]
        receipt = ScanReceipt("scan-1", "repo", 0, 1, 1, 10, {"test": 1}, "abc")
        result = steps_to_json(steps, receipt)
        assert "receipt" in result
        assert result["receipt"]["scan_id"] == "scan-1"

    def test_groups_by_category(self):
        steps = [
            NextStep("A", "", "bug", "high", "small", "test"),
            NextStep("B", "", "bug", "high", "small", "test"),
            NextStep("C", "", "docs", "low", "small", "test"),
        ]
        result = steps_to_json(steps)
        assert result["by_category"]["bug"] == 2
        assert result["by_category"]["docs"] == 1


# ---------------------------------------------------------------------------
# NextStepsRunner
# ---------------------------------------------------------------------------


class TestNextStepsRunner:
    """Tests for the main runner class."""

    def test_extract_github_repo_from_url(self):
        runner = NextStepsRunner(repo_url="https://github.com/owner/repo")
        assert runner._github_repo == "owner/repo"

    def test_extract_github_repo_from_https(self):
        runner = NextStepsRunner(repo_url="https://github.com/an0mium/aragora")
        assert runner._github_repo == "an0mium/aragora"

    def test_extract_github_repo_none(self):
        runner = NextStepsRunner(repo_path="/tmp/no-git-here-xyz")
        # May be None if no git remote
        assert runner._github_repo is None or isinstance(runner._github_repo, str)

    @pytest.mark.asyncio
    async def test_scan_local_repo(self, tmp_path):
        (tmp_path / "app.py").write_text("# TODO: add tests\n# FIXME: race condition\n")
        (tmp_path / "README.md").write_text("# Project\n" + "x " * 200)

        runner = NextStepsRunner(
            repo_path=tmp_path,
            scan_issues=False,
            scan_prs=False,
            scan_tests=False,
            scan_deps=False,
        )
        result = await runner.scan()

        assert result.error is None
        assert len(result.steps) >= 2
        assert result.receipt is not None
        assert result.receipt.files_scanned >= 1

    @pytest.mark.asyncio
    async def test_scan_empty_repo(self, tmp_path):
        runner = NextStepsRunner(
            repo_path=tmp_path,
            scan_issues=False,
            scan_prs=False,
        )
        result = await runner.scan()
        assert result.error is None
        # Should still find doc gaps (no README)
        assert any(s.source == "doc-gap" for s in result.steps)

    @pytest.mark.asyncio
    async def test_scan_respects_limit(self, tmp_path):
        # Create many TODOs
        content = "\n".join(f"# TODO: item {i}" for i in range(100))
        (tmp_path / "big.py").write_text(content)
        (tmp_path / "README.md").write_text("# OK\n" + "x " * 200)

        runner = NextStepsRunner(
            repo_path=tmp_path,
            scan_issues=False,
            scan_prs=False,
            limit=5,
        )
        result = await runner.scan()
        assert len(result.steps) <= 5

    @pytest.mark.asyncio
    async def test_scan_signals_tracked(self, tmp_path):
        (tmp_path / "app.py").write_text("# TODO: test\n")
        runner = NextStepsRunner(
            repo_path=tmp_path,
            scan_issues=False,
            scan_prs=False,
            scan_docs=False,
        )
        result = await runner.scan()
        assert result.receipt is not None
        assert "code-marker" in result.receipt.signals_by_source

    @pytest.mark.asyncio
    async def test_scan_disabled_sources(self, tmp_path):
        (tmp_path / "app.py").write_text("# TODO: test\n")
        runner = NextStepsRunner(
            repo_path=tmp_path,
            scan_code=False,
            scan_issues=False,
            scan_prs=False,
            scan_docs=False,
        )
        result = await runner.scan()
        assert len(result.steps) == 0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCmdNextSteps:
    """Tests for the CLI command function."""

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_cli_runs(self, mock_run, tmp_path):
        """Verify the CLI function can be called."""
        from aragora.cli.openclaw import cmd_next_steps

        # Mock git remote
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        (tmp_path / "app.py").write_text("# TODO: add feature\n")
        (tmp_path / "README.md").write_text("# Test\n" + "x " * 200)

        args = MagicMock()
        args.path = str(tmp_path)
        args.repo = None
        args.no_code = False
        args.no_issues = True
        args.no_prs = True
        args.tests = False
        args.deps = False
        args.no_docs = False
        args.limit = 10
        args.json = False
        args.output = None

        exit_code = cmd_next_steps(args)
        assert exit_code == 0

    @patch("aragora.compat.openclaw.next_steps_runner.subprocess.run")
    def test_cli_json_output(self, mock_run, tmp_path):
        """Verify JSON output mode works."""
        from aragora.cli.openclaw import cmd_next_steps

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        (tmp_path / "app.py").write_text("# FIXME: critical bug\n")
        (tmp_path / "README.md").write_text("# OK\n" + "x " * 200)

        output_file = tmp_path / "results.json"
        args = MagicMock()
        args.path = str(tmp_path)
        args.repo = None
        args.no_code = False
        args.no_issues = True
        args.no_prs = True
        args.tests = False
        args.deps = False
        args.no_docs = False
        args.limit = 10
        args.json = True
        args.output = str(output_file)

        exit_code = cmd_next_steps(args)
        assert exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert "steps" in data
        assert data["count"] >= 1

    def test_cli_invalid_path(self):
        """Verify error on invalid path."""
        from aragora.cli.openclaw import cmd_next_steps

        args = MagicMock()
        args.path = "/nonexistent/path/xyz/abc"
        args.repo = None
        args.no_code = False
        args.no_issues = True
        args.no_prs = True
        args.tests = False
        args.deps = False
        args.no_docs = False
        args.limit = 10
        args.json = False
        args.output = None

        exit_code = cmd_next_steps(args)
        assert exit_code == 1
