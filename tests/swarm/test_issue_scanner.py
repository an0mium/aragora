"""Tests for the autonomous issue generation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.swarm.issue_scanner import (
    BossIssueCandidate,
    historical_success_rates,
    infer_issue_category_from_title,
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
        candidates = scan_all(repo_root, metrics_path=repo_root / ".missing-boss-metrics.jsonl")
        assert len(candidates) > 0
        assert all(isinstance(c, BossIssueCandidate) for c in candidates)

    def test_candidates_have_required_fields(self, repo_root):
        candidates = scan_all(repo_root, metrics_path=repo_root / ".missing-boss-metrics.jsonl")
        for c in candidates[:10]:
            assert len(c.title) > 20, f"Title too short: {c.title}"
            assert len(c.description) > 40, f"Description too short for {c.title}"
            assert len(c.file_scope) > 0, f"Empty file scope for {c.title}"
            assert c.validation_command, f"Missing validation for {c.title}"
            assert c.fingerprint, f"Missing fingerprint for {c.title}"
            assert 0 < c.expected_success_rate <= 1.0

    def test_scan_all_sorted_by_success_rate(self, repo_root):
        candidates = scan_all(repo_root, metrics_path=repo_root / ".missing-boss-metrics.jsonl")
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

    def test_todo_scanner(self, repo_root):
        results = scan_actionable_todos(repo_root, limit=5)
        for c in results:
            assert c.category == "actionable_todo"
            assert "TODO" in c.description or "FIXME" in c.description


class TestHistoricalSuccessRates:
    def test_infer_issue_category_from_title(self) -> None:
        assert infer_issue_category_from_title("Narrow broad except Exception in foo.py")
        assert infer_issue_category_from_title("Replace silent exception swallowing in foo.py")
        assert infer_issue_category_from_title("Add unit tests for swarm/foo.py")
        assert infer_issue_category_from_title("Add request body validation to foo.py handlers")
        assert infer_issue_category_from_title("Add return type annotations to foo.py")
        assert infer_issue_category_from_title("Address TODO/FIXME items in foo.py")
        assert infer_issue_category_from_title("Unrelated issue title") is None

    def test_historical_success_rates_uses_inline_issue_titles(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "boss_metrics.jsonl"
        rows = [
            {
                "issue_number": 101,
                "issue_title": "Narrow broad except Exception in foo.py",
                "prompt_chars": 1000,
                "worker_status": "completed",
                "worker_outcome": "pr_adopted",
                "publish_action": "pr_created",
                "elapsed_seconds": 120.0,
                "files_changed": 1,
                "has_deliverable": True,
            },
            {
                "issue_number": 102,
                "issue_title": "Narrow broad except Exception in bar.py",
                "prompt_chars": 1200,
                "worker_status": "failed",
                "worker_outcome": "worker_crash",
                "publish_action": "",
                "elapsed_seconds": 240.0,
                "files_changed": 0,
                "has_deliverable": False,
            },
            {
                "issue_number": 201,
                "issue_title": "Add request body validation to baz.py handlers",
                "prompt_chars": 1500,
                "worker_status": "needs_human",
                "worker_outcome": "blocked",
                "publish_action": "",
                "elapsed_seconds": 90.0,
                "files_changed": 0,
                "has_deliverable": False,
            },
        ]
        metrics_path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")

        rates = historical_success_rates(metrics_path)

        assert rates["broad_exception"] == pytest.approx(0.5)
        assert rates["handler_validation"] == pytest.approx(0.0)

    def test_scan_all_overrides_rates_and_filters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        candidate = BossIssueCandidate(
            category="handler_validation",
            title="Add request body validation to foo.py handlers",
            description="desc",
            file_scope=["aragora/server/handlers/foo.py"],
            expected_success_rate=0.5,
        )

        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_handler_validation_gaps",
            lambda repo_root, limit=15: [candidate],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_bare_except_handlers",
            lambda repo_root, limit=20: [],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_silent_exception_swallowing",
            lambda repo_root, limit=20: [],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_untested_modules",
            lambda repo_root, min_loc=50, max_loc=300, limit=30: [],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_actionable_todos",
            lambda repo_root, min_length=25, limit=15: [],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.scan_type_annotation_gaps",
            lambda repo_root, limit=10: [],
        )
        monkeypatch.setattr(
            "aragora.swarm.issue_scanner.historical_success_rates",
            lambda metrics_path: {"handler_validation": 0.2},
        )

        filtered = scan_all(
            tmp_path,
            categories=["handler_validation"],
            metrics_path=tmp_path / "boss_metrics.jsonl",
            min_success_rate=0.3,
        )
        assert filtered == []

        unfiltered = scan_all(
            tmp_path,
            categories=["handler_validation"],
            metrics_path=tmp_path / "boss_metrics.jsonl",
            min_success_rate=0.0,
        )
        assert len(unfiltered) == 1
        assert unfiltered[0].expected_success_rate == pytest.approx(0.2)
