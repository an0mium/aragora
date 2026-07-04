"""Regression tests for Stage-Gate Conductor log issue targeting."""

from __future__ import annotations

import pytest

from aragora.ops.stage_gate_conductor_log import (
    AUTOMATION_LOG_LABEL,
    CANONICAL_STAGE_GATE_LOG_ISSUE,
    CANONICAL_STAGE_GATE_LOG_LABEL,
    STAGE_GATE_CONDUCTOR_LOG_TITLE,
    STAGE_GATE_ISSUE_LIST_LIMIT,
    StageGateLogResolutionError,
    build_gh_issue_comment_args,
    build_gh_issue_list_args,
    resolve_stage_gate_conductor_log_issue,
)


def _issue(
    number: object,
    *,
    title: str = STAGE_GATE_CONDUCTOR_LOG_TITLE,
    labels: tuple[str, ...] = (AUTOMATION_LOG_LABEL,),
    state: str = "OPEN",
) -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "labels": [{"name": label} for label in labels],
        "state": state,
    }


def test_pinned_canonical_log_issue_is_8671() -> None:
    assert CANONICAL_STAGE_GATE_LOG_ISSUE == 8671


def test_labeled_canonical_issue_wins_over_duplicate_history() -> None:
    issues = [
        _issue(7162),
        _issue(8671, labels=("automation-log", CANONICAL_STAGE_GATE_LOG_LABEL)),
        _issue(8402),
    ]

    assert resolve_stage_gate_conductor_log_issue(issues) == 8671


def test_canonical_label_does_not_require_automation_log_label() -> None:
    issues = [
        _issue(7162),
        _issue(8671, labels=(CANONICAL_STAGE_GATE_LOG_LABEL,)),
        _issue(8402),
    ]

    assert resolve_stage_gate_conductor_log_issue(issues) == 8671


def test_pinned_issue_wins_when_duplicates_have_no_canonical_label() -> None:
    issues = [
        _issue(6432),
        _issue(6763),
        _issue(7162),
        _issue(8402),
        _issue(8671),
    ]

    assert resolve_stage_gate_conductor_log_issue(issues) == 8671


def test_pinned_issue_wins_even_when_label_metadata_has_drifted() -> None:
    issues = [
        _issue(8402),
        _issue(8671, labels=()),
    ]

    assert resolve_stage_gate_conductor_log_issue(issues) == 8671


def test_ambiguous_duplicate_logs_fail_closed_when_no_canonical_exists() -> None:
    issues = [
        _issue(6432),
        _issue(6763),
        _issue(7162),
        _issue(8402),
    ]

    with pytest.raises(StageGateLogResolutionError, match="multiple Stage-Gate Conductor Log"):
        resolve_stage_gate_conductor_log_issue(issues)


def test_unlabeled_duplicate_prevents_singleton_drift() -> None:
    issues = [
        _issue(8402),
        _issue(9001, labels=()),
    ]

    with pytest.raises(StageGateLogResolutionError, match="multiple Stage-Gate Conductor Log"):
        resolve_stage_gate_conductor_log_issue(issues)


@pytest.mark.parametrize("state", ["CLOSED", "", "unknown"])
def test_non_open_state_fails_closed(state: str) -> None:
    with pytest.raises(StageGateLogResolutionError, match="no open Stage-Gate"):
        resolve_stage_gate_conductor_log_issue([_issue(8671, state=state)])


def test_missing_state_fails_closed() -> None:
    issue = _issue(8671)
    issue.pop("state")

    with pytest.raises(StageGateLogResolutionError, match="no open Stage-Gate"):
        resolve_stage_gate_conductor_log_issue([issue])


def test_missing_numeric_issue_number_raises_instead_of_returning_zero() -> None:
    issue = _issue("not-a-number", labels=(CANONICAL_STAGE_GATE_LOG_LABEL,))

    with pytest.raises(StageGateLogResolutionError, match="numeric issue number"):
        resolve_stage_gate_conductor_log_issue([issue])


def test_comment_args_target_resolved_issue_number_not_title_search() -> None:
    args = build_gh_issue_comment_args([_issue(7162), _issue(8671)], body="run body")

    assert args == ["issue", "comment", "8671", "--body", "run body"]
    assert STAGE_GATE_CONDUCTOR_LOG_TITLE not in args


def test_issue_list_query_uses_high_limit_to_reduce_partial_result_drift() -> None:
    args = build_gh_issue_list_args()

    assert args[args.index("--limit") + 1] == str(STAGE_GATE_ISSUE_LIST_LIMIT)
    assert STAGE_GATE_ISSUE_LIST_LIMIT == 1000
