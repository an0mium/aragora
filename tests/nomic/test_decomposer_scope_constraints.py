"""Tests for TaskDecomposer file-scope constraint enforcement.

Covers the improvement where file_scope_hints are passed directly to
analyze() as a first-class parameter, and the decomposer validates/constrains
subtask file_scope values against those hints:
- Empty scopes → backfilled from hints
- Non-overlapping scopes → overridden with hints
- Overlapping scopes → preserved (decomposer correctly narrowed)
"""

from __future__ import annotations

import pytest

from aragora.nomic.task_decomposer import SubTask, TaskDecomposer


class TestScopeOverlapsHints:
    """Unit tests for _scope_overlaps_hints static method."""

    def test_exact_match(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints(["aragora/live"], ["aragora/live"]) is True

    def test_scope_under_hint(self) -> None:
        assert (
            TaskDecomposer._scope_overlaps_hints(["aragora/live/package.json"], ["aragora/live"])
            is True
        )

    def test_hint_under_scope(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints(["aragora"], ["aragora/live"]) is True

    def test_no_overlap(self) -> None:
        assert (
            TaskDecomposer._scope_overlaps_hints(
                ["aragora/audit/", "aragora/compliance/"], ["aragora/live"]
            )
            is False
        )

    def test_empty_scope(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints([], ["aragora/live"]) is False

    def test_empty_hints(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints(["aragora/live"], []) is False

    def test_leading_dot_slash_stripped(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints(["./aragora/live"], ["aragora/live"]) is True

    def test_trailing_slash_stripped(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints(["aragora/live/"], ["aragora/live"]) is True

    def test_both_empty(self) -> None:
        assert TaskDecomposer._scope_overlaps_hints([], []) is False


class TestConstrainScopesToHints:
    """Unit tests for _constrain_scopes_to_hints method."""

    def setup_method(self) -> None:
        self.decomposer = TaskDecomposer()

    def test_empty_scope_backfilled(self) -> None:
        subtasks = [SubTask(id="s1", title="t", description="d", file_scope=[])]
        result = self.decomposer._constrain_scopes_to_hints(subtasks, ["aragora/live"])
        assert result[0].file_scope == ["aragora/live"]

    def test_overlapping_scope_preserved(self) -> None:
        subtasks = [
            SubTask(
                id="s1",
                title="t",
                description="d",
                file_scope=["aragora/live/package.json"],
            )
        ]
        result = self.decomposer._constrain_scopes_to_hints(subtasks, ["aragora/live"])
        assert result[0].file_scope == ["aragora/live/package.json"]

    def test_non_overlapping_scope_overridden(self) -> None:
        subtasks = [
            SubTask(
                id="s1",
                title="t",
                description="d",
                file_scope=["aragora/audit/", "aragora/compliance/"],
            )
        ]
        result = self.decomposer._constrain_scopes_to_hints(subtasks, ["aragora/live"])
        assert result[0].file_scope == ["aragora/live"]

    def test_mixed_subtasks(self) -> None:
        """Overlapping subtask preserved, empty backfilled, non-overlapping overridden."""
        subtasks = [
            SubTask(
                id="s1",
                title="correct",
                description="d",
                file_scope=["aragora/live/package.json"],
            ),
            SubTask(id="s2", title="empty", description="d", file_scope=[]),
            SubTask(
                id="s3",
                title="wrong",
                description="d",
                file_scope=["aragora/audit/"],
            ),
        ]
        result = self.decomposer._constrain_scopes_to_hints(subtasks, ["aragora/live"])
        assert result[0].file_scope == ["aragora/live/package.json"]
        assert result[1].file_scope == ["aragora/live"]
        assert result[2].file_scope == ["aragora/live"]

    def test_wider_scope_preserved(self) -> None:
        """Scope wider than hint still overlaps → preserved."""
        subtasks = [SubTask(id="s1", title="t", description="d", file_scope=["aragora"])]
        result = self.decomposer._constrain_scopes_to_hints(subtasks, ["aragora/live"])
        assert result[0].file_scope == ["aragora"]


class TestAnalyzeWithFileeScopeHints:
    """Integration tests: analyze() with file_scope_hints constrains subtask scopes."""

    def test_heuristic_subtasks_constrained_to_hints(self) -> None:
        """When the heuristic decomposer generates subtasks with wrong scopes,
        the file_scope_hints parameter causes them to be overridden."""
        decomposer = TaskDecomposer()
        # This task mentions "security" and "api" concepts, which the heuristic
        # decomposer maps to aragora/auth/, aragora/server/ etc.
        result = decomposer.analyze(
            "Refactor security and api handling in the codebase",
            file_scope_hints=["aragora/live"],
        )
        if result.should_decompose:
            for subtask in result.subtasks:
                # Every subtask's file_scope must overlap with the hints
                if subtask.file_scope:
                    has_overlap = TaskDecomposer._scope_overlaps_hints(
                        subtask.file_scope, ["aragora/live"]
                    )
                    assert has_overlap or subtask.file_scope == ["aragora/live"], (
                        f"subtask {subtask.id} scope {subtask.file_scope} "
                        f"should be constrained to aragora/live"
                    )

    def test_no_hints_leaves_scopes_unchanged(self) -> None:
        """Without hints, the decomposer produces whatever scopes it wants."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Refactor security and api handling in the codebase",
        )
        # No assertion on specific scopes — just verify it doesn't crash
        assert result.original_task

    def test_forensic_873_decomposer_constrained(self) -> None:
        """Forensic #873 shape: simple dependency bump goal with file_scope_hints.
        The decomposer should produce subtasks scoped to the hints."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Bump @eslint/eslintrc from 3.2.0 to 3.3.0 in /aragora/live",
            file_scope_hints=["aragora/live", "@eslint/eslintrc"],
        )
        # Whether or not it decomposes, any subtasks should respect hints
        for subtask in result.subtasks:
            if subtask.file_scope:
                has_overlap = TaskDecomposer._scope_overlaps_hints(
                    subtask.file_scope, ["aragora/live", "@eslint/eslintrc"]
                )
                assert has_overlap or subtask.file_scope == [
                    "aragora/live",
                    "@eslint/eslintrc",
                ], f"subtask {subtask.id} scope {subtask.file_scope} violates hints"
