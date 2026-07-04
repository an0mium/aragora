"""Canonical Stage-Gate Conductor log issue selection.

The Stage-Gate Conductor has had multiple open issues with the exact same log
title. Automation must not pick among them by comment count, age, or GitHub
search ordering; those heuristics caused the log target to drift.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

STAGE_GATE_CONDUCTOR_LOG_TITLE = "[automation] Stage-Gate Conductor Log"
CANONICAL_STAGE_GATE_LOG_ISSUE = 8671
CANONICAL_STAGE_GATE_LOG_LABEL = "stage-gate-log-canonical"
AUTOMATION_LOG_LABEL = "automation-log"
STAGE_GATE_ISSUE_LIST_LIMIT = 1000


class StageGateLogResolutionError(RuntimeError):
    """Raised when the conductor log target cannot be resolved safely."""


def _label_names(issue: Mapping[str, Any]) -> set[str]:
    labels = issue.get("labels") or ()
    names: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            names.add(label.lower())
        elif isinstance(label, Mapping):
            name = label.get("name")
            if isinstance(name, str):
                names.add(name.lower())
    return names


def _issue_number(issue: Mapping[str, Any]) -> int | None:
    value = issue.get("number")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_open(issue: Mapping[str, Any]) -> bool:
    state = issue.get("state")
    return isinstance(state, str) and state.upper() == "OPEN"


def _has_stage_gate_log_title(issue: Mapping[str, Any]) -> bool:
    return _is_open(issue) and issue.get("title") == STAGE_GATE_CONDUCTOR_LOG_TITLE


def _require_issue_number(issue: Mapping[str, Any]) -> int:
    number = _issue_number(issue)
    if number is None:
        raise StageGateLogResolutionError(
            "matching Stage-Gate Conductor Log issue is missing a numeric issue number"
        )
    return number


def resolve_stage_gate_conductor_log_issue(
    issues: Iterable[Mapping[str, Any]],
    *,
    pinned_issue: int = CANONICAL_STAGE_GATE_LOG_ISSUE,
    canonical_label: str = CANONICAL_STAGE_GATE_LOG_LABEL,
) -> int:
    """Return the single issue number the conductor must comment on.

    Resolution order is intentionally fail-closed:
    1. exactly one open exact-title log issue with
       ``stage-gate-log-canonical`` wins;
    2. if no unique canonical label is present, the open exact-title pinned
       issue ``#8671`` wins even if duplicate label metadata has drifted;
    3. if there is exactly one open exact-title log issue and it still carries
       ``automation-log``, use it;
    4. otherwise raise instead of creating or choosing another duplicate.
    """

    log_issues = [dict(issue) for issue in issues if _has_stage_gate_log_title(issue)]
    if not log_issues:
        raise StageGateLogResolutionError(
            "no open Stage-Gate Conductor Log issue found; do not create a duplicate"
        )

    canonical_label = canonical_label.lower()
    labeled = [issue for issue in log_issues if canonical_label in _label_names(issue)]
    if len(labeled) == 1:
        return _require_issue_number(labeled[0])
    if len(labeled) > 1:
        pinned_labeled = [issue for issue in labeled if _issue_number(issue) == pinned_issue]
        if len(pinned_labeled) == 1:
            return pinned_issue
        numbers = sorted(
            number for issue in labeled if (number := _issue_number(issue)) is not None
        )
        raise StageGateLogResolutionError(
            "multiple Stage-Gate Conductor Log issues carry "
            f"{CANONICAL_STAGE_GATE_LOG_LABEL}: {numbers}"
        )

    pinned_matches = [issue for issue in log_issues if _issue_number(issue) == pinned_issue]
    if len(pinned_matches) == 1:
        return pinned_issue

    strict_candidates = [
        issue for issue in log_issues if AUTOMATION_LOG_LABEL in _label_names(issue)
    ]
    if len(log_issues) == 1 and len(strict_candidates) == 1:
        return _require_issue_number(strict_candidates[0])

    numbers = sorted(number for issue in log_issues if (number := _issue_number(issue)) is not None)
    raise StageGateLogResolutionError(
        "multiple Stage-Gate Conductor Log issues found without a canonical target: "
        f"{numbers}; label one {CANONICAL_STAGE_GATE_LOG_LABEL} or include #{pinned_issue}"
    )


def build_gh_issue_comment_args(
    issues: Iterable[Mapping[str, Any]],
    *,
    body: str,
    repo: str | None = None,
) -> list[str]:
    """Build ``gh issue comment`` args for the resolved canonical log issue."""

    issue_number = resolve_stage_gate_conductor_log_issue(issues)
    args = ["issue", "comment", str(issue_number), "--body", body]
    if repo:
        args.extend(["--repo", repo])
    return args


def build_gh_issue_list_args(
    *,
    repo: str = "synaptent/aragora",
    limit: int = STAGE_GATE_ISSUE_LIST_LIMIT,
) -> list[str]:
    """Build the bounded ``gh issue list`` query used before resolution."""

    return [
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--search",
        f'"{STAGE_GATE_CONDUCTOR_LOG_TITLE}" in:title repo:{repo}',
        "--limit",
        str(limit),
        "--json",
        "number,title,labels,state,url,createdAt,updatedAt",
    ]


__all__ = [
    "AUTOMATION_LOG_LABEL",
    "CANONICAL_STAGE_GATE_LOG_ISSUE",
    "CANONICAL_STAGE_GATE_LOG_LABEL",
    "STAGE_GATE_CONDUCTOR_LOG_TITLE",
    "STAGE_GATE_ISSUE_LIST_LIMIT",
    "StageGateLogResolutionError",
    "build_gh_issue_comment_args",
    "build_gh_issue_list_args",
    "resolve_stage_gate_conductor_log_issue",
]
