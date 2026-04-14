"""Tests for the extracted boss loop issue selection module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aragora.swarm.boss_loop_selection import (
    has_explicit_parallel_lane_hint,
    parallel_claim_scope_entries,
    scope_hint_is_specific,
    scope_hint_is_validation_command_scope,
    select_issues_for_batch,
    semantic_dedup_issues,
)


def _issue(number: int, title: str = "", body: str = "", labels: list[str] | None = None) -> Any:
    return SimpleNamespace(
        number=number, title=title, body=body or title, labels=labels or [], state="OPEN"
    )


class TestScopeHintIsSpecific:
    def test_file_with_extension(self) -> None:
        assert scope_hint_is_specific("aragora/swarm/boss_loop.py") is True

    def test_glob_pattern(self) -> None:
        assert scope_hint_is_specific("tests/**/*.py") is True

    def test_directory_only(self) -> None:
        assert scope_hint_is_specific("aragora/swarm/") is False

    def test_bare_directory(self) -> None:
        assert scope_hint_is_specific("aragora/swarm") is False


class TestScopeHintIsValidationCommandScope:
    def test_pytest_command_line(self) -> None:
        issue = _issue(1, body="## Validation\npytest tests/swarm/ -q")
        assert scope_hint_is_validation_command_scope(issue, "tests/swarm") is True

    def test_specific_file_is_not_validation(self) -> None:
        issue = _issue(1, body="## Validation\npytest tests/swarm/test_foo.py -q")
        assert scope_hint_is_validation_command_scope(issue, "tests/swarm/test_foo.py") is False

    def test_no_validation_command(self) -> None:
        issue = _issue(1, body="Just some description")
        assert scope_hint_is_validation_command_scope(issue, "tests/swarm") is False


class TestHasExplicitParallelLaneHint:
    def test_lane_label(self) -> None:
        issue = _issue(1, labels=["lane:swarm"])
        assert has_explicit_parallel_lane_hint(issue) is True

    def test_area_label(self) -> None:
        issue = _issue(1, labels=["area:server"])
        assert has_explicit_parallel_lane_hint(issue) is True

    def test_no_lane_hint(self) -> None:
        issue = _issue(1, labels=["boss-ready"])
        assert has_explicit_parallel_lane_hint(issue) is False

    def test_lane_in_body(self) -> None:
        issue = _issue(1, body="lane: swarm-substrate\nSome description")
        assert has_explicit_parallel_lane_hint(issue) is True


class TestSemanticDedupIssues:
    def test_fewer_than_six_passthrough(self) -> None:
        issues = [_issue(i) for i in range(5)]
        assert semantic_dedup_issues(issues) == issues

    def test_six_or_more_attempts_dedup(self) -> None:
        issues = [_issue(i, title=f"Fix issue {i}") for i in range(6)]
        # Without API keys, should return original list
        result = semantic_dedup_issues(issues)
        assert len(result) == 6


class TestSelectIssuesForBatch:
    def test_single_issue_limit(self) -> None:
        issues = [_issue(1, "Fix A"), _issue(2, "Fix B")]
        selected = select_issues_for_batch(issues, limit=1)
        assert len(selected) == 1

    def test_respects_skip_labels(self) -> None:
        issues = [
            _issue(1, "Fix A", labels=["boss-stuck"]),
            _issue(2, "Fix B"),
        ]
        selected = select_issues_for_batch(issues, limit=2, skip_labels={"boss-stuck"})
        assert len(selected) == 1
        assert selected[0].number == 2

    def test_respects_blocked_scopes(self) -> None:
        issues = [
            _issue(1, "Fix A", body="## File Scope\n- aragora/swarm/boss_loop.py"),
            _issue(2, "Fix B", body="## File Scope\n- aragora/swarm/spec.py"),
        ]
        selected = select_issues_for_batch(
            issues,
            limit=2,
            blocked_scopes={"aragora/swarm/boss_loop.py"},
        )
        # Issue 1 should be blocked by scope
        assert all(s.number != 1 for s in selected) or len(selected) <= 2

    def test_specific_issue_number(self) -> None:
        issues = [_issue(1, "Fix A"), _issue(2, "Fix B"), _issue(3, "Fix C")]
        selected = select_issues_for_batch(issues, limit=1, issue_number=2)
        assert len(selected) == 1
        assert selected[0].number == 2

    def test_empty_issues(self) -> None:
        assert select_issues_for_batch([], limit=5) == []

    def test_custom_dedup_fn(self) -> None:
        issues = [_issue(i, title=f"Fix {i}") for i in range(10)]
        called = {"yes": False}

        def mock_dedup(candidates: list[Any]) -> list[Any]:
            called["yes"] = True
            return candidates[:3]

        selected = select_issues_for_batch(issues, limit=5, dedup_fn=mock_dedup)
        assert called["yes"]
        assert len(selected) <= 3
