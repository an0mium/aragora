from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.publish_codex_automation_branches import (
    BranchSnapshot,
    WorktreeSnapshot,
    select_publishable_branches,
)

UTC = timezone.utc


def _branch(
    name: str,
    *,
    hours_ago: int = 1,
    unique_commit_count: int = 1,
    upstream: str | None = None,
) -> BranchSnapshot:
    return BranchSnapshot(
        branch=name,
        upstream=upstream,
        head_sha="abc1234",
        committed_at=datetime.now(UTC) - timedelta(hours=hours_ago),
        subject=f"subject for {name}",
        unique_commit_count=unique_commit_count,
    )


def _worktree(
    path: str,
    *,
    branch: str | None,
    dirty: bool = False,
    active_session: bool = False,
) -> WorktreeSnapshot:
    return WorktreeSnapshot(
        path=path,
        branch=branch,
        detached=False,
        dirty=dirty,
        active_session=active_session,
    )


def test_select_publishable_branches_marks_recent_clean_branch_eligible() -> None:
    decisions = select_publishable_branches(
        [_branch("codex/recent-fix")],
        [],
        set(),
        cutoff=datetime.now(UTC) - timedelta(hours=24),
        base="main",
        is_merged={"codex/recent-fix": False},
    )

    assert len(decisions) == 1
    assert decisions[0].eligible is True
    assert decisions[0].reason == "eligible"


def test_select_publishable_branches_skips_open_pr_and_old_or_merged_branches() -> None:
    decisions = select_publishable_branches(
        [
            _branch("codex/already-open"),
            _branch("codex/old", hours_ago=200),
            _branch("codex/merged"),
            _branch("codex/no-unique", unique_commit_count=0),
        ],
        [],
        {"codex/already-open"},
        cutoff=datetime.now(UTC) - timedelta(hours=24),
        base="main",
        is_merged={
            "codex/already-open": False,
            "codex/old": False,
            "codex/merged": True,
            "codex/no-unique": False,
        },
        historical_pr_branches=set(),
    )

    by_branch = {decision.branch: decision for decision in decisions}
    assert by_branch["codex/already-open"].reason == "open_pr_exists"
    assert by_branch["codex/old"].reason == "older_than_cutoff"
    assert by_branch["codex/merged"].reason == "already_merged"
    assert by_branch["codex/no-unique"].reason == "no_unique_commits"


def test_select_publishable_branches_skips_dirty_and_active_worktrees() -> None:
    decisions = select_publishable_branches(
        [
            _branch("codex/dirty"),
            _branch("codex/active"),
        ],
        [
            _worktree("/tmp/dirty", branch="codex/dirty", dirty=True),
            _worktree("/tmp/active", branch="codex/active", active_session=True),
        ],
        set(),
        cutoff=datetime.now(UTC) - timedelta(hours=24),
        base="main",
        is_merged={"codex/dirty": False, "codex/active": False},
        historical_pr_branches=set(),
    )

    by_branch = {decision.branch: decision for decision in decisions}
    assert by_branch["codex/dirty"].reason == "dirty_worktree"
    assert by_branch["codex/active"].reason == "active_session"


def test_select_publishable_branches_skips_branches_with_historical_prs() -> None:
    decisions = select_publishable_branches(
        [_branch("codex/already-reviewed")],
        [],
        set(),
        cutoff=datetime.now(UTC) - timedelta(hours=24),
        base="main",
        is_merged={"codex/already-reviewed": False},
        historical_pr_branches={"codex/already-reviewed"},
    )

    assert decisions[0].reason == "historical_pr_exists"
