"""Tests for the autonomous issue generation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.swarm.issue_scanner import (
    BossIssueCandidate,
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

    def test_scan_all_sorted_by_success_rate(self, repo_root):
        candidates = scan_all(repo_root)
        rates = [c.expected_success_rate for c in candidates]
        # Should be roughly descending (within same rate, category order matters)
        for i in range(len(rates) - 1):
            if rates[i] < rates[i + 1]:
                # Only ok if categories differ
                assert candidates[i].category != candidates[i + 1].category

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

    def test_bare_except_scanner_ignores_docstring_examples(self, tmp_path):
        aragora_dir = tmp_path / "aragora"
        aragora_dir.mkdir()
        (aragora_dir / "__init__.py").write_text("", encoding="utf-8")
        (aragora_dir / "docstring_only.py").write_text(
            '"""Example:\n'
            "try:\n"
            "    run()\n"
            "except Exception:\n"
            "    fallback()\n"
            '"""\n'
            "\n"
            "def demo() -> None:\n"
            "    return None\n",
            encoding="utf-8",
        )
        (aragora_dir / "real_handler.py").write_text(
            "def demo() -> None:\n"
            "    try:\n"
            "        run()\n"
            "    except Exception:\n"
            "        return None\n",
            encoding="utf-8",
        )

        results = scan_bare_except_handlers(tmp_path, limit=10)
        scopes = {candidate.file_scope[0] for candidate in results}

        assert "aragora/docstring_only.py" not in scopes
        assert "aragora/real_handler.py" in scopes

    def test_todo_scanner(self, repo_root):
        results = scan_actionable_todos(repo_root, limit=5)
        for c in results:
            assert c.category == "actionable_todo"
            assert "TODO" in c.description or "FIXME" in c.description
