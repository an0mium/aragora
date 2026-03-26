from __future__ import annotations

from scripts.merge_codex_automation_prs import (
    MergeDecision,
    PullRequestSnapshot,
    select_mergeable_prs,
)


def _pr(
    number: int,
    *,
    head_ref: str = "codex/safe-fix",
    is_draft: bool = False,
    mergeable: str = "MERGEABLE",
    body: str = "## Validation\n- pytest -q",
    changed_files: list[str] | None = None,
    status_rollup: list[dict[str, str]] | None = None,
) -> PullRequestSnapshot:
    if changed_files is None:
        changed_files = ["aragora/live/src/app/page.tsx"]
    if status_rollup is None:
        status_rollup = [{"status": "COMPLETED", "conclusion": "SUCCESS", "name": "tests"}]
    return PullRequestSnapshot(
        number=number,
        title=f"PR {number}",
        head_ref=head_ref,
        is_draft=is_draft,
        mergeable=mergeable,
        body=body,
        url=f"https://example.com/pr/{number}",
        changed_files=changed_files,
        status_rollup=status_rollup,
    )


def _decision(decisions: list[MergeDecision], number: int) -> MergeDecision:
    for decision in decisions:
        if decision.number == number:
            return decision
    raise AssertionError(f"missing decision for PR {number}")


def test_select_mergeable_prs_marks_safe_codex_pr_eligible() -> None:
    decisions = select_mergeable_prs([_pr(1)])

    decision = _decision(decisions, 1)
    assert decision.eligible is True
    assert decision.reason == "eligible"


def test_select_mergeable_prs_skips_non_codex_draft_and_missing_validation() -> None:
    decisions = select_mergeable_prs(
        [
            _pr(1, head_ref="feature/not-codex"),
            _pr(2, is_draft=True),
            _pr(3, body="No evidence here"),
        ]
    )

    assert _decision(decisions, 1).reason == "not_codex_branch"
    assert _decision(decisions, 2).reason == "draft"
    assert _decision(decisions, 3).reason == "missing_validation"


def test_select_mergeable_prs_skips_pending_and_failed_checks() -> None:
    decisions = select_mergeable_prs(
        [
            _pr(1, status_rollup=[{"status": "IN_PROGRESS", "conclusion": "", "name": "tests"}]),
            _pr(
                2, status_rollup=[{"status": "COMPLETED", "conclusion": "FAILURE", "name": "tests"}]
            ),
        ]
    )

    assert _decision(decisions, 1).reason == "checks_pending"
    assert _decision(decisions, 2).reason == "checks_failed"


def test_select_mergeable_prs_skips_sensitive_and_large_changes() -> None:
    decisions = select_mergeable_prs(
        [
            _pr(1, changed_files=["aragora/billing/auth/config.py"]),
            _pr(2, changed_files=[f"aragora/file_{idx}.py" for idx in range(7)]),
        ]
    )

    assert _decision(decisions, 1).reason == "sensitive_paths"
    assert _decision(decisions, 2).reason == "too_many_files"


def test_select_mergeable_prs_skips_non_mergeable_or_unchecked_prs() -> None:
    decisions = select_mergeable_prs(
        [
            _pr(1, mergeable="UNKNOWN"),
            _pr(2, status_rollup=[]),
        ]
    )

    assert _decision(decisions, 1).reason == "not_mergeable"
    assert _decision(decisions, 2).reason == "no_status_checks"
