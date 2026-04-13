"""Tests for task_sanitizer admission rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.task_sanitizer import SanitizationOutcome, TaskSanitizer


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "tests" / "swarm").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "aragora" / "swarm" / "boss_validation.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "aragora" / "swarm" / "supervisor.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "aragora" / "swarm" / "worker_launcher.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "aragora" / "swarm" / "spec.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "tests" / "swarm" / "test_boss_loop.py").write_text(
        "def test_ok():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "swarm" / "test_supervisor.py").write_text(
        "def test_ok():\n    pass\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def sanitizer(repo_root: Path) -> TaskSanitizer:
    return TaskSanitizer(repo_root=repo_root)


class TestOutcomes:
    def test_accepted_issue_passes_through_unchanged(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/boss_loop.py` (modify)\n\n"
            "## Validation\n"
            "- python3 -m ruff check aragora/swarm/boss_loop.py\n"
        )

        result = sanitizer.sanitize("Narrow boss loop guard", body)

        assert result.outcome is SanitizationOutcome.ACCEPTED
        assert result.sanitized_text == result.original_text
        assert result.checks_failed == []

    def test_rewritten_when_validation_missing(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/boss_validation.py` (modify)\n\n"
            "Tighten the pre-dispatch rule so it rejects incomplete tasks before worker launch."
        )

        result = sanitizer.sanitize("Add a missing validation contract", body)

        assert result.outcome is SanitizationOutcome.REWRITTEN
        assert "## Validation" in result.sanitized_text
        assert "python3 -m ruff check aragora/swarm/boss_validation.py" in result.sanitized_text
        assert "missing_validation" in result.checks_failed

    def test_dropped_when_description_too_short(self, sanitizer: TaskSanitizer) -> None:
        result = sanitizer.sanitize("Too short", "Tiny task only.")
        assert result.outcome is SanitizationOutcome.DROPPED
        assert result.reason == "task description is too short to dispatch safely"
        assert "description_length" in result.checks_failed

    def test_quarantined_when_scope_is_too_broad(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## Allowed Write Set\n"
            "- `aragora/swarm/a.py` (modify)\n"
            "- `aragora/swarm/b.py` (modify)\n"
            "- `aragora/swarm/c.py` (modify)\n"
            "- `aragora/swarm/d.py` (modify)\n"
            "- `aragora/swarm/e.py` (modify)\n"
            "- `aragora/swarm/f.py` (modify)\n"
        )

        result = sanitizer.sanitize("Over-broad lane", body)

        assert result.outcome is SanitizationOutcome.QUARANTINED
        assert "scope_too_broad" in result.checks_failed
        assert "- `aragora/swarm/a.py`" in result.sanitized_text
        assert "- `aragora/swarm/f.py`" not in result.sanitized_text


class TestImpossibleValidation:
    def test_impossible_validation_quarantines_missing_target(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/boss_validation.py` (modify)\n\n"
            "## Validation\n"
            "- python3 -m pytest tests/swarm/test_missing.py -q\n"
        )

        finding = sanitizer._check_impossible_validation(body, sanitizer.repo_root)

        assert finding is not None
        assert finding.outcome is SanitizationOutcome.QUARANTINED
        assert "tests/swarm/test_missing.py" in finding.reason

    def test_impossible_validation_allows_existing_target(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/boss_loop.py` (modify)\n\n"
            "## Validation\n"
            "- python3 -m pytest tests/swarm/test_boss_loop.py -q\n"
        )

        assert sanitizer._check_impossible_validation(body, sanitizer.repo_root) is None

    def test_impossible_validation_allows_declared_created_file(
        self, sanitizer: TaskSanitizer
    ) -> None:
        body = (
            "## File Scope\n"
            "- `tests/swarm/test_new_lane.py` (create)\n\n"
            "## Validation\n"
            "- python3 -m pytest tests/swarm/test_new_lane.py -q\n"
        )

        assert sanitizer._check_impossible_validation(body, sanitizer.repo_root) is None

    def test_impossible_validation_allows_declared_created_target_with_new_file_marker(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## Files\n"
            "- `tests/swarm/test_generated_lane.py` (new file)\n\n"
            "## Validation\n"
            "- pytest tests/swarm/test_generated_lane.py -q\n"
        )

        assert sanitizer._check_impossible_validation(body, sanitizer.repo_root) is None


class TestScopeTooBroad:
    def test_scope_too_broad_ignores_five_files(self, sanitizer: TaskSanitizer) -> None:
        body = "\n".join(
            [
                "## File Scope",
                "- `aragora/swarm/a.py` (modify)",
                "- `aragora/swarm/b.py` (modify)",
                "- `aragora/swarm/c.py` (modify)",
                "- `aragora/swarm/d.py` (modify)",
                "- `aragora/swarm/e.py` (modify)",
            ]
        )
        assert sanitizer._check_scope_too_broad(body) is None

    def test_scope_too_broad_rejects_six_files(self, sanitizer: TaskSanitizer) -> None:
        body = "\n".join(
            [
                "## File Scope",
                "- `aragora/swarm/a.py` (modify)",
                "- `aragora/swarm/b.py` (modify)",
                "- `aragora/swarm/c.py` (modify)",
                "- `aragora/swarm/d.py` (modify)",
                "- `aragora/swarm/e.py` (modify)",
                "- `aragora/swarm/f.py` (modify)",
            ]
        )
        finding = sanitizer._check_scope_too_broad(body)
        assert finding is not None
        assert finding.outcome is SanitizationOutcome.QUARANTINED

    def test_scope_too_broad_deduplicates_duplicate_paths(self, sanitizer: TaskSanitizer) -> None:
        body = "\n".join(
            [
                "## Allowed Write Set",
                "- `aragora/swarm/a.py` (modify)",
                "- `aragora/swarm/a.py` (modify)",
                "- `aragora/swarm/b.py` (modify)",
                "- `aragora/swarm/c.py` (modify)",
                "- `aragora/swarm/d.py` (modify)",
                "- `aragora/swarm/e.py` (modify)",
            ]
        )
        assert sanitizer._check_scope_too_broad(body) is None

    def test_scope_too_broad_uses_scope_section_not_validation_targets(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/boss_loop.py` (modify)\n\n"
            "## Validation\n"
            "- pytest tests/swarm/test_boss_loop.py -q\n"
            "- pytest tests/swarm/test_supervisor.py -q\n"
            "- pytest tests/swarm/test_generated.py -q\n"
            "- pytest tests/swarm/test_more.py -q\n"
            "- pytest tests/swarm/test_extra.py -q\n"
            "- pytest tests/swarm/test_overflow.py -q\n"
        )
        assert sanitizer._check_scope_too_broad(body) is None


class TestContradictoryScope:
    def test_contradictory_scope_detects_create_and_modify_same_path(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (create)\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n"
        )
        finding = sanitizer._check_contradictory_scope(body)
        assert finding is not None
        assert finding.outcome is SanitizationOutcome.QUARANTINED

    def test_contradictory_scope_detects_create_and_update_same_path(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## Files\n"
            "- create `tests/swarm/test_task_sanitizer.py`\n"
            "- update `tests/swarm/test_task_sanitizer.py`\n"
        )
        finding = sanitizer._check_contradictory_scope(body)
        assert finding is not None
        assert "contradicts itself" in finding.reason

    def test_contradictory_scope_ignores_different_paths(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (create)\n"
            "- `tests/swarm/test_task_sanitizer.py` (modify)\n"
        )
        assert sanitizer._check_contradictory_scope(body) is None

    def test_contradictory_scope_ignores_single_create_marker(
        self, sanitizer: TaskSanitizer
    ) -> None:
        body = "## Allowed Write Set\n- `tests/swarm/test_task_sanitizer.py` (create)\n"
        assert sanitizer._check_contradictory_scope(body) is None

    def test_contradictory_scope_ignores_modify_with_add_in_description(
        self, sanitizer: TaskSanitizer
    ) -> None:
        """Modify annotations should not false-positive when descriptive text contains 'add'."""
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify) — add a helper for scope dedup\n"
        )
        assert sanitizer._check_contradictory_scope(body) is None


class TestRewriteMissingValidation:
    def test_rewrite_missing_validation_appends_section(self, sanitizer: TaskSanitizer) -> None:
        body = "## File Scope\n- `aragora/swarm/task_sanitizer.py` (modify)\n"
        rewritten = sanitizer._rewrite_missing_validation(body)
        assert "## Validation" in rewritten
        assert "python3 -m ruff check aragora/swarm/task_sanitizer.py" in rewritten

    def test_rewrite_missing_validation_uses_most_specific_path(
        self, sanitizer: TaskSanitizer
    ) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n"
            "- `aragora/swarm/worker_launcher.py` (modify)\n"
        )
        rewritten = sanitizer._rewrite_missing_validation(body)
        assert "aragora/swarm/worker_launcher.py" not in rewritten.split("## Validation", 1)[1]
        assert "aragora/swarm/task_sanitizer.py" in rewritten

    def test_rewrite_missing_validation_falls_back_to_aragora_root(
        self, sanitizer: TaskSanitizer
    ) -> None:
        rewritten = sanitizer._rewrite_missing_validation(
            "Investigate crash-heavy overnight tasks."
        )
        assert "python3 -m ruff check aragora/" in rewritten

    def test_rewrite_missing_validation_keeps_existing_text(self, sanitizer: TaskSanitizer) -> None:
        body = "Investigate why rescue_worker_crash dominates the overnight metrics."
        rewritten = sanitizer._rewrite_missing_validation(body)
        assert rewritten.startswith(body)


class TestDuplicateMerged:
    def test_duplicate_of_merged_drops_issue(
        self, sanitizer: TaskSanitizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sanitizer, "_is_pr_merged", lambda pr_number: pr_number == 1234)
        finding = sanitizer._check_duplicate_of_merged("Follow-up to PR #1234")
        assert finding is not None
        assert finding.outcome is SanitizationOutcome.DROPPED

    def test_duplicate_of_open_pr_is_ignored(
        self, sanitizer: TaskSanitizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sanitizer, "_is_pr_merged", lambda _pr_number: False)
        assert sanitizer._check_duplicate_of_merged("Follow-up to PR #1234") is None


class TestComplexityEstimate:
    def test_high_complexity_without_work_orders_is_quarantined(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "Estimated complexity: high\n\n"
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n"
            "- `aragora/swarm/boss_validation.py` (modify)\n"
            "- `tests/swarm/test_task_sanitizer.py` (modify)\n"
        )
        finding = sanitizer._check_complexity_estimate("Refactor the admission pipeline", body)
        assert finding is not None
        assert finding.outcome is SanitizationOutcome.QUARANTINED

    def test_high_complexity_with_explicit_work_orders_is_allowed(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "Estimated complexity: high\n\n"
            "## Task Breakdown\n"
            "1. Build the sanitizer helper\n"
            "2. Add regression tests\n"
            "3. Validate the new admission gate\n"
        )
        assert sanitizer._check_complexity_estimate("Break down the fix", body) is None


class TestSanitizePipeline:
    def test_pipeline_drops_truncated_overnight_issue(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## Task\n"
            "Investigate why the overnight crash-heavy lane keeps dispatching impossible tasks and \\\n"
        )
        result = sanitizer.sanitize("Overnight crash lane", body)
        assert result.outcome is SanitizationOutcome.DROPPED
        assert "truncation" in result.checks_failed

    def test_pipeline_quarantines_impossible_validation_issue(
        self, sanitizer: TaskSanitizer
    ) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n\n"
            "## Validation\n"
            "- pytest tests/swarm/test_nonexistent_retry.py -q\n"
        )
        result = sanitizer.sanitize("Fix rescue_worker_crash lane", body)
        assert result.outcome is SanitizationOutcome.QUARANTINED
        assert "impossible_validation" in result.checks_failed

    def test_pipeline_rewrites_missing_validation_from_metrics_style_issue(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "## Allowed Write Set\n"
            "- `aragora/swarm/task_sanitizer.py` (create)\n"
            "- `tests/swarm/test_task_sanitizer.py` (create)\n\n"
            "The overnight metrics show 153/205 rescue_worker_crash outcomes. Add a small pre-dispatch sanitizer."
        )
        result = sanitizer.sanitize("BC-04 Add sanitizer outcomes", body)
        assert result.outcome is SanitizationOutcome.REWRITTEN
        assert "## Validation" in result.sanitized_text

    def test_pipeline_quarantines_high_complexity_without_work_orders(
        self,
        sanitizer: TaskSanitizer,
    ) -> None:
        body = (
            "Estimated complexity: large\n\n"
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n"
            "- `aragora/swarm/boss_validation.py` (modify)\n"
            "- `aragora/swarm/spec.py` (modify)\n"
            "- `aragora/swarm/issue_scanner.py` (modify)\n"
            "- `tests/swarm/test_task_sanitizer.py` (modify)\n"
            "- `tests/swarm/test_boss_validation.py` (modify)\n"
        )
        result = sanitizer.sanitize("Overnight crash reduction pass", body)
        assert result.outcome is SanitizationOutcome.QUARANTINED
        assert "complexity_estimate" in result.checks_failed

    def test_pipeline_prefers_dropped_over_rewrite(self, sanitizer: TaskSanitizer) -> None:
        body = "Tiny task only."
        result = sanitizer.sanitize("Short and missing validation", body)
        assert result.outcome is SanitizationOutcome.DROPPED
        assert "description_length" in result.checks_failed


class TestAcceptedUnchanged:
    def test_accepted_issue_is_returned_unchanged(self, sanitizer: TaskSanitizer) -> None:
        body = (
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n\n"
            "## Validation\n"
            "- python3 -m ruff check aragora/swarm/task_sanitizer.py\n"
        )
        result = sanitizer.sanitize("Tighten sanitizer matchers", body)
        assert result.outcome is SanitizationOutcome.ACCEPTED
        assert result.original_text == result.sanitized_text
