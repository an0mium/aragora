"""Tests for the autonomous issue generation pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.swarm.issue_scanner import (
    BossIssueCandidate,
    CATEGORY_PRIORITY,
    scan_all,
    scan_bare_except_handlers,
    scan_silent_exception_swallowing,
    scan_untested_modules,
    scan_actionable_todos,
)


# -- BossIssueCandidate --


class TestBossIssueCandidate:
    def test_fingerprint_auto_generated(self):
        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests for foo",
            description="Create tests",
            file_scope=["aragora/foo.py"],
        )
        assert len(c.fingerprint) == 16
        assert c.fingerprint.isalnum()

    def test_fingerprint_stable(self):
        """Same inputs produce same fingerprint."""
        c1 = BossIssueCandidate(
            category="test_coverage",
            title="Add tests",
            description="desc",
            file_scope=["aragora/foo.py", "aragora/bar.py"],
        )
        c2 = BossIssueCandidate(
            category="test_coverage",
            title="Different title",
            description="different desc",
            file_scope=["aragora/bar.py", "aragora/foo.py"],  # different order
        )
        assert c1.fingerprint == c2.fingerprint

    def test_fingerprint_differs_by_category(self):
        c1 = BossIssueCandidate(
            category="test_coverage",
            title="t",
            description="d",
            file_scope=["aragora/foo.py"],
        )
        c2 = BossIssueCandidate(
            category="silent_exception",
            title="t",
            description="d",
            file_scope=["aragora/foo.py"],
        )
        assert c1.fingerprint != c2.fingerprint


# -- format_boss_ready_body --


class TestFormatBossReadyBody:
    def test_includes_task_section(self):
        from scripts.generate_boss_issues import format_boss_ready_body

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests for foo.py",
            description="Create comprehensive unit tests for foo module.",
            file_scope=["aragora/foo.py"],
            new_files=["tests/test_foo.py"],
            validation_command="pytest tests/test_foo.py -v",
            acceptance_criteria=["All tests pass"],
        )
        body = format_boss_ready_body(c)
        assert "## Task" in body
        assert "Create comprehensive unit tests" in body

    def test_includes_file_scope(self):
        from scripts.generate_boss_issues import format_boss_ready_body

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests",
            description="Create tests.",
            file_scope=["aragora/foo.py"],
            new_files=["tests/test_foo.py"],
        )
        body = format_boss_ready_body(c)
        assert "`aragora/foo.py`" in body
        assert "`tests/test_foo.py` (create)" in body

    def test_includes_fingerprint(self):
        from scripts.generate_boss_issues import format_boss_ready_body

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests",
            description="Create tests.",
            file_scope=["aragora/foo.py"],
        )
        body = format_boss_ready_body(c)
        assert f"<!-- fingerprint:{c.fingerprint} -->" in body

    def test_passes_sanitation(self):
        from scripts.generate_boss_issues import format_boss_ready_body

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add unit tests for aragora/swarm/config.py module",
            description=(
                "Add comprehensive unit tests for `aragora/swarm/config.py`.\n\n"
                "### Requirements\n"
                "1. Read the module and identify all public functions\n"
                "2. Create test file with comprehensive coverage"
            ),
            file_scope=["aragora/swarm/config.py"],
            new_files=["tests/swarm/test_config.py"],
            validation_command="pytest tests/swarm/test_config.py -v",
            acceptance_criteria=["All tests pass", "At least 8 test functions"],
        )
        body = format_boss_ready_body(c)

        from aragora.swarm.boss_validation import assess_issue_body_sanitation

        ok, reason = assess_issue_body_sanitation(body)
        assert ok, f"Sanitation failed: {reason}"


# -- Deduplication --


class TestDeduplication:
    def test_fingerprint_match(self):
        from scripts.generate_boss_issues import is_duplicate

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests for foo",
            description="desc",
            file_scope=["aragora/foo.py"],
        )
        existing = [
            {"title": "Something else", "body": f"stuff <!-- fingerprint:{c.fingerprint} -->"}
        ]
        assert is_duplicate(c, existing)

    def test_title_similarity(self):
        from scripts.generate_boss_issues import is_duplicate

        c = BossIssueCandidate(
            category="broad_exception",
            title="Narrow broad except Exception in campaign.py",
            description="desc",
            file_scope=["aragora/swarm/campaign.py"],
        )
        existing = [{"title": "Narrow broad except Exception in campaign.py", "body": ""}]
        assert is_duplicate(c, existing)

    def test_file_scope_overlap(self):
        from scripts.generate_boss_issues import is_duplicate

        c = BossIssueCandidate(
            category="test_coverage",
            title="Completely different title here",
            description="desc",
            file_scope=["aragora/swarm/campaign.py"],
        )
        existing = [{"title": "Other issue", "body": "work on `aragora/swarm/campaign.py`"}]
        assert is_duplicate(c, existing)

    def test_no_duplicate(self):
        from scripts.generate_boss_issues import is_duplicate

        c = BossIssueCandidate(
            category="test_coverage",
            title="Add tests for new_module.py",
            description="desc",
            file_scope=["aragora/brand_new.py"],
        )
        existing = [{"title": "Fix bug in old_module.py", "body": "stuff about old_module.py"}]
        assert not is_duplicate(c, existing)


# -- Scanners on real repo --


class TestScannersOnRealRepo:
    """Integration tests running scanners against the actual repo."""

    @pytest.fixture
    def repo_root(self):
        return Path(__file__).resolve().parent.parent.parent

    def test_scan_all_returns_candidates(self, repo_root):
        candidates = scan_all(repo_root)
        assert len(candidates) > 0
        assert all(isinstance(c, BossIssueCandidate) for c in candidates)

    def test_candidates_have_required_fields(self, repo_root):
        candidates = scan_all(repo_root)
        for c in candidates[:10]:
            assert len(c.title) > 20, f"Title too short: {c.title}"
            assert len(c.description) > 40, f"Description too short for {c.title}"
            assert len(c.file_scope) > 0, f"Empty file scope for {c.title}"
            assert c.validation_command, f"Missing validation for {c.title}"
            assert c.fingerprint, f"Missing fingerprint for {c.title}"
            assert 0 < c.expected_success_rate <= 1.0

    def test_scan_all_sorted_by_category_priority(self, repo_root):
        candidates = scan_all(repo_root)
        priorities = [CATEGORY_PRIORITY.get(c.category, 99) for c in candidates]
        assert priorities == sorted(priorities)

        grouped_rates: dict[str, list[float]] = {}
        for candidate in candidates:
            grouped_rates.setdefault(candidate.category, []).append(candidate.expected_success_rate)
        for rates in grouped_rates.values():
            assert rates == sorted(rates, reverse=True)

    def test_scan_all_prioritizes_roadmap_aligned_categories(self, monkeypatch):
        import aragora.swarm.issue_scanner as issue_scanner

        def candidate(category: str, rate: float) -> BossIssueCandidate:
            return BossIssueCandidate(
                category=category,
                title=f"{category} candidate",
                description=f"{category} description",
                file_scope=[f"aragora/{category}.py"],
                expected_success_rate=rate,
            )

        monkeypatch.setattr(
            issue_scanner,
            "scan_bare_except_handlers",
            lambda repo_root: [candidate("broad_exception", 0.9)],
        )
        monkeypatch.setattr(
            issue_scanner,
            "scan_silent_exception_swallowing",
            lambda repo_root: [candidate("silent_exception", 0.8)],
        )
        monkeypatch.setattr(
            issue_scanner,
            "scan_untested_modules",
            lambda repo_root: [candidate("test_coverage", 0.7)],
        )
        monkeypatch.setattr(
            issue_scanner,
            "scan_handler_validation_gaps",
            lambda repo_root: [candidate("handler_validation", 0.5)],
        )
        monkeypatch.setattr(issue_scanner, "scan_actionable_todos", lambda repo_root: [])
        monkeypatch.setattr(issue_scanner, "scan_type_annotation_gaps", lambda repo_root: [])

        candidates = issue_scanner.scan_all(Path("/tmp/repo"))

        assert [candidate.category for candidate in candidates] == [
            "test_coverage",
            "handler_validation",
            "silent_exception",
            "broad_exception",
        ]


class TestGitHubDiscovery:
    def test_fetch_existing_boss_issues_raises_on_gh_failure(self):
        import scripts.generate_boss_issues as generate_boss_issues

        failure = SimpleNamespace(
            returncode=1, stdout="", stderr="error connecting to api.github.com"
        )
        with patch.object(generate_boss_issues.subprocess, "run", return_value=failure):
            with pytest.raises(
                generate_boss_issues.GitHubDiscoveryError, match="gh issue list failed"
            ):
                generate_boss_issues.fetch_existing_boss_issues("synaptent/aragora")

    def test_main_aborts_when_github_discovery_fails(self, monkeypatch, capsys):
        import scripts.generate_boss_issues as generate_boss_issues

        monkeypatch.setattr(
            generate_boss_issues,
            "scan_all",
            lambda repo_root, categories=None: [
                BossIssueCandidate(
                    category="test_coverage",
                    title="Add tests for example.py",
                    description="Create tests for example.py",
                    file_scope=["aragora/example.py"],
                    new_files=["tests/test_example.py"],
                    validation_command="pytest tests/test_example.py -v",
                )
            ],
        )
        monkeypatch.setattr(
            generate_boss_issues,
            "fetch_existing_boss_issues",
            lambda repo: (_ for _ in ()).throw(
                generate_boss_issues.GitHubDiscoveryError("gh issue list failed: error connecting")
            ),
        )
        monkeypatch.setattr(
            generate_boss_issues.sys,
            "argv",
            ["generate_boss_issues.py", "--dry-run", "--max-issues", "1"],
        )

        assert generate_boss_issues.main() == 1

        captured = capsys.readouterr()
        assert "GitHub discovery failed: gh issue list failed: error connecting" in captured.err


class TestAdditionalRealRepoScanners:
    @pytest.fixture
    def repo_root(self):
        return Path(__file__).resolve().parent.parent.parent

    def test_untested_modules_finds_some(self, repo_root):
        results = scan_untested_modules(repo_root, limit=5)
        assert len(results) > 0
        for c in results:
            assert c.category == "test_coverage"
            assert c.new_files  # Should have a test file to create

    def test_silent_exception_scanner(self, repo_root):
        results = scan_silent_exception_swallowing(repo_root, limit=5)
        # May or may not find results, but shouldn't crash
        for c in results:
            assert c.category == "silent_exception"
            assert "pass" in c.description.lower() or "silent" in c.description.lower()

    def test_bare_except_scanner(self, repo_root):
        results = scan_bare_except_handlers(repo_root, limit=5)
        for c in results:
            assert c.category == "broad_exception"
            assert "except Exception" in c.description

    def test_todo_scanner(self, repo_root):
        results = scan_actionable_todos(repo_root, limit=5)
        for c in results:
            assert c.category == "actionable_todo"
            assert "TODO" in c.description or "FIXME" in c.description
