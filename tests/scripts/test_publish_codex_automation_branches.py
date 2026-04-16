from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import scripts.publish_codex_automation_branches as mod
from scripts.publish_codex_automation_branches import (
    BranchSnapshot,
    WorktreeSnapshot,
    _worktree_is_dirty,
    _build_parser,
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


def test_parser_defaults_to_single_branch_publish_budget() -> None:
    args = _build_parser().parse_args([])

    assert args.limit == 1
    assert args.max_open_prs == 1
    assert args.scan_limit == mod.DEFAULT_SCAN_LIMIT
    assert args.skip_preflight is False
    assert args.preflight_script == mod.DEFAULT_PREFLIGHT_SCRIPT


def test_select_publishable_branches_skips_open_pr_and_old_or_merged_branches() -> None:
    decisions = select_publishable_branches(
        [
            _branch("codex/already-open"),
            _branch("codex/old", hours_ago=200),
            _branch("codex/merged"),
            _branch("codex/cherry-picked"),
            _branch("codex/related-resolved"),
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
            "codex/cherry-picked": False,
            "codex/related-resolved": False,
            "codex/no-unique": False,
        },
        is_patch_equivalent={"codex/cherry-picked": True},
        historical_pr_branches=set(),
        resolved_related_branches={"codex/related-resolved"},
    )

    by_branch = {decision.branch: decision for decision in decisions}
    assert by_branch["codex/already-open"].reason == "open_pr_exists"
    assert by_branch["codex/old"].reason == "older_than_cutoff"
    assert by_branch["codex/merged"].reason == "already_merged"
    assert by_branch["codex/cherry-picked"].reason == "patch_equivalent_to_base"
    assert by_branch["codex/related-resolved"].reason == "related_resolved_work_exists"
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


def test_related_subjects_match_resolved_github_items() -> None:
    assert mod._looks_related_subject(
        "fix(autonomy): parse multiline github app keys",
        "Support multiline GitHub App keys in automation env",
    )
    assert not mod._looks_related_subject(
        "fix(playground): preserve mock debate receipts",
        "Support multiline GitHub App keys in automation env",
    )


def test_open_pr_heads_counts_only_codex_branches(monkeypatch: Any, tmp_path: Path) -> None:
    payload = """
    [
      {"headRefName": "codex/fix-one"},
      {"headRefName": "dependabot/npm_and_yarn/picomatch-4.0.4"},
      {"headRefName": "feature/manual-branch"},
      {"headRefName": "codex/fix-two"}
    ]
    """.strip()

    monkeypatch.setattr(
        mod,
        "_run",
        lambda args, cwd, check=False: subprocess.CompletedProcess(
            args=args, returncode=0, stdout=payload, stderr=""
        ),
    )

    assert mod._open_pr_heads(tmp_path, "synaptent/aragora") == {
        "codex/fix-one",
        "codex/fix-two",
    }


def test_worktree_is_dirty_ignores_untracked_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    (repo / "untracked.txt").write_text("scratch\n", encoding="utf-8")
    assert _worktree_is_dirty(repo) is False

    tracked.write_text("changed\n", encoding="utf-8")
    assert _worktree_is_dirty(repo) is True


def test_list_worktrees_filters_before_dirty_checks(monkeypatch: Any, tmp_path: Path) -> None:
    payload = """
worktree /tmp/codex-a
HEAD abc123
branch refs/heads/codex/a

worktree /tmp/codex-b
HEAD def456
branch refs/heads/codex/b
""".strip()
    dirty_checked: list[str] = []

    monkeypatch.setattr(
        mod,
        "_run",
        lambda args, cwd, check=False: subprocess.CompletedProcess(
            args=args, returncode=0, stdout=payload, stderr=""
        ),
    )
    monkeypatch.setattr(mod, "_has_active_session", lambda path: False)

    def fake_dirty(path: Path) -> bool:
        dirty_checked.append(str(path))
        return False

    monkeypatch.setattr(mod, "_worktree_is_dirty", fake_dirty)

    snapshots = mod._list_worktrees(tmp_path, branch_filter={"codex/b"})

    assert [snapshot.branch for snapshot in snapshots] == ["codex/b"]
    assert dirty_checked == [str(Path("/tmp/codex-b").resolve())]


def test_publish_decisions_respects_open_pr_cap(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(mod, "_ensure_gh_auth", lambda repo_root: None)
    monkeypatch.setattr(
        mod, "_push_branch", lambda repo_root, branch, upstream: calls.append(f"push:{branch}")
    )
    monkeypatch.setattr(mod, "_existing_pr_number", lambda repo_root, repo, branch, base: None)
    monkeypatch.setattr(mod, "_create_pr", lambda repo_root, repo, branch, base: 1234)
    monkeypatch.setattr(
        mod, "_add_labels", lambda repo_root, repo, number, labels: calls.append(f"label:{number}")
    )

    results = mod._publish_decisions(
        tmp_path,
        "synaptent/aragora",
        "main",
        [
            mod.PublishDecision(
                branch="codex/ready-1",
                eligible=True,
                reason="eligible",
                subject="fix one",
                head_sha="abc1234",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            )
        ],
        limit=5,
        open_pr_count=3,
        max_open_prs=3,
        labels=["codex"],
    )

    assert results == []
    assert calls == []


def test_publish_decisions_skips_branch_when_preflight_fails(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(mod, "_ensure_gh_auth", lambda repo_root: None)
    monkeypatch.setattr(
        mod,
        "_run_publish_preflight",
        lambda repo_root, base, branch, preflight_script: (False, "session artifact found"),
    )
    monkeypatch.setattr(
        mod, "_push_branch", lambda repo_root, branch, upstream: calls.append(f"push:{branch}")
    )

    results = mod._publish_decisions(
        tmp_path,
        "synaptent/aragora",
        "main",
        [
            mod.PublishDecision(
                branch="codex/artifact-branch",
                eligible=True,
                reason="eligible",
                subject="fix one",
                head_sha="abc1234",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            )
        ],
        limit=5,
        open_pr_count=0,
        max_open_prs=3,
        labels=["codex"],
        preflight_script=mod.DEFAULT_PREFLIGHT_SCRIPT,
    )

    assert results == [
        {
            "branch": "codex/artifact-branch",
            "status": "preflight_failed",
            "subject": "fix one",
            "head_sha": "abc1234",
            "reason": "session artifact found",
        }
    ]
    assert calls == []


def test_publish_decisions_uses_remaining_open_pr_capacity(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[str] = []
    created_numbers = iter([2001, 2002])

    monkeypatch.setattr(mod, "_ensure_gh_auth", lambda repo_root: None)
    monkeypatch.setattr(
        mod, "_push_branch", lambda repo_root, branch, upstream: calls.append(f"push:{branch}")
    )
    monkeypatch.setattr(mod, "_existing_pr_number", lambda repo_root, repo, branch, base: None)
    monkeypatch.setattr(
        mod, "_create_pr", lambda repo_root, repo, branch, base: next(created_numbers)
    )
    monkeypatch.setattr(
        mod, "_add_labels", lambda repo_root, repo, number, labels: calls.append(f"label:{number}")
    )

    results = mod._publish_decisions(
        tmp_path,
        "synaptent/aragora",
        "main",
        [
            mod.PublishDecision(
                branch="codex/ready-1",
                eligible=True,
                reason="eligible",
                subject="fix one",
                head_sha="abc1234",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            ),
            mod.PublishDecision(
                branch="codex/ready-2",
                eligible=True,
                reason="eligible",
                subject="fix two",
                head_sha="def5678",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            ),
        ],
        limit=5,
        open_pr_count=2,
        max_open_prs=3,
        labels=["codex"],
    )

    assert [item["branch"] for item in results] == ["codex/ready-1"]
    assert calls == ["push:codex/ready-1", "label:2001"]
