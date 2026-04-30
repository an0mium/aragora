"""Dispatch evidence aggregation for the H1-01 promotion gate.

The boss-loop emits two complementary signals when an issue is worked:

1. A row in ``.aragora/overnight/boss_metrics.jsonl`` with the issue
   number and a ``terminal_class``.
2. A pull request opened against ``main`` from the deterministic
   branch pattern ``aragora/boss-harvest/issue-N-*`` (where ``N`` is
   the issue number).

The metrics ledger is local-only and can drift: rows that pre-date the
ledger or that were lost in a rotation simply do not exist there. A
**merged** PR is, by contrast, a strong, GitHub-canonical signal that
the issue was actually dispatched and produced shippable work.

This module defines a single pure predicate :func:`is_issue_dispatched_via_pr`
plus a helper :func:`extract_issue_number_from_branch` so the renderer
and any future promotion-gate consumers can ask "does GitHub state
agree this issue was dispatched?" without each call site rolling its
own branch-name parsing or PR-state filtering.

This module makes **no GitHub calls itself**. The caller passes in the
already-fetched PR records (e.g. via ``gh pr list --json
number,state,headRefName`` or the GitHub REST API). That keeps the
predicate fast, deterministic, and safe to unit-test offline.
"""

from __future__ import annotations

import re
from typing import Final, Iterable

# Branch pattern emitted by ``aragora.swarm.boss_loop`` when it
# publishes a deliverable PR for an issue. Anchored at the front so we
# never accept arbitrary branches that happen to mention an issue
# number in their suffix.
_BOSS_HARVEST_BRANCH_RE: Final[re.Pattern[str]] = re.compile(
    r"^aragora/boss-harvest/issue-(\d+)(?:[-/].*)?$"
)

# PR states accepted as dispatch evidence. ``MERGED`` is the strongest
# signal; ``CLOSED`` (without merge) is *not* accepted because the work
# was abandoned. Open PRs are accepted as in-flight evidence so the
# operator can see that work has been kicked off, but they bucket
# separately from merged PRs.
DISPATCH_EVIDENCE_STATES: Final[frozenset[str]] = frozenset({"MERGED", "OPEN"})


def extract_issue_number_from_branch(branch_name: str | None) -> int | None:
    """Return the issue number encoded by a boss-harvest branch.

    Returns ``None`` if ``branch_name`` is empty, ``None``, or does not
    match the boss-harvest naming convention. Only positive integers
    are returned; a branch named ``aragora/boss-harvest/issue-0`` is
    rejected.
    """
    if not branch_name:
        return None
    match = _BOSS_HARVEST_BRANCH_RE.match(branch_name)
    if match is None:
        return None
    try:
        n = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def is_issue_dispatched_via_pr(
    issue_number: int,
    *,
    pr_records: Iterable[dict[str, object]],
    accept_open: bool = True,
) -> dict[str, object]:
    """Decide whether GitHub PR state implies the issue was dispatched.

    Args:
        issue_number: The corpus-side issue number to check.
        pr_records: Iterable of dicts with at least ``number``,
            ``state``, and ``headRefName`` keys (the shape returned by
            ``gh pr list --json number,state,headRefName``).
        accept_open: If True (default), an open PR on the
            boss-harvest branch counts as in-flight dispatch evidence.

    Returns:
        A dict with keys:

        - ``dispatched`` (bool): True iff at least one matching PR was
          found whose state is in :data:`DISPATCH_EVIDENCE_STATES`
          (and ``OPEN`` is allowed when ``accept_open`` is True).
        - ``best_state`` (str or None): The strongest state seen
          (``"MERGED"`` > ``"OPEN"``).
        - ``pr_numbers_merged`` (list[int]): Sorted list of merged PRs
          that target this issue.
        - ``pr_numbers_open`` (list[int]): Sorted list of open PRs
          that target this issue.
    """
    if not isinstance(issue_number, int) or issue_number <= 0:
        return {
            "dispatched": False,
            "best_state": None,
            "pr_numbers_merged": [],
            "pr_numbers_open": [],
        }

    merged: list[int] = []
    open_: list[int] = []

    for record in pr_records or []:
        if not isinstance(record, dict):
            continue
        branch = record.get("headRefName")
        if not isinstance(branch, str):
            continue
        candidate_issue = extract_issue_number_from_branch(branch)
        if candidate_issue != issue_number:
            continue
        state = str(record.get("state") or "").strip().upper()
        raw_number = record.get("number")
        if not isinstance(raw_number, int):
            continue
        pr_number = raw_number
        if pr_number <= 0:
            continue
        if state == "MERGED":
            merged.append(pr_number)
        elif state == "OPEN" and accept_open:
            open_.append(pr_number)
        # CLOSED (without merge) is intentionally ignored.

    merged.sort()
    open_.sort()

    if merged:
        best_state = "MERGED"
        dispatched = True
    elif open_ and accept_open:
        best_state = "OPEN"
        dispatched = True
    else:
        best_state = None
        dispatched = False

    return {
        "dispatched": dispatched,
        "best_state": best_state,
        "pr_numbers_merged": merged,
        "pr_numbers_open": open_,
    }


def issues_dispatched_via_pr(
    issue_numbers: Iterable[int],
    *,
    pr_records: Iterable[dict[str, object]],
    accept_open: bool = True,
) -> dict[int, dict[str, object]]:
    """Batch variant of :func:`is_issue_dispatched_via_pr`.

    Returns a mapping from each issue number in the input to its
    per-issue verdict dict. The PR records are scanned exactly once,
    so this is O(R + N) where R is the PR count and N is the issue
    count.
    """
    targets: set[int] = set()
    for n in issue_numbers or []:
        if isinstance(n, int) and n > 0:
            targets.add(n)
    if not targets:
        return {}

    per_issue_merged: dict[int, list[int]] = {n: [] for n in targets}
    per_issue_open: dict[int, list[int]] = {n: [] for n in targets}

    for record in pr_records or []:
        if not isinstance(record, dict):
            continue
        branch = record.get("headRefName")
        if not isinstance(branch, str):
            continue
        candidate_issue = extract_issue_number_from_branch(branch)
        if candidate_issue not in targets:
            continue
        state = str(record.get("state") or "").strip().upper()
        raw_number = record.get("number")
        if not isinstance(raw_number, int):
            continue
        pr_number = raw_number
        if pr_number <= 0:
            continue
        if state == "MERGED":
            per_issue_merged[candidate_issue].append(pr_number)
        elif state == "OPEN" and accept_open:
            per_issue_open[candidate_issue].append(pr_number)

    out: dict[int, dict[str, object]] = {}
    for n in sorted(targets):
        merged = sorted(per_issue_merged[n])
        open_ = sorted(per_issue_open[n])
        if merged:
            out[n] = {
                "dispatched": True,
                "best_state": "MERGED",
                "pr_numbers_merged": merged,
                "pr_numbers_open": open_,
            }
        elif open_ and accept_open:
            out[n] = {
                "dispatched": True,
                "best_state": "OPEN",
                "pr_numbers_merged": merged,
                "pr_numbers_open": open_,
            }
        else:
            out[n] = {
                "dispatched": False,
                "best_state": None,
                "pr_numbers_merged": merged,
                "pr_numbers_open": open_,
            }
    return out


__all__ = [
    "DISPATCH_EVIDENCE_STATES",
    "extract_issue_number_from_branch",
    "is_issue_dispatched_via_pr",
    "issues_dispatched_via_pr",
]
