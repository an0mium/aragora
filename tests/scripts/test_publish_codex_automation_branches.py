from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import scripts.publish_codex_automation_branches as mod
from scripts.github_cli_health import GitHubCLIHealth
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
    head_sha: str = "abc1234",
) -> BranchSnapshot:
    return BranchSnapshot(
        branch=name,
        upstream=upstream,
        head_sha=head_sha,
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


def test_parser_defaults_match_publisher_budget_constants() -> None:
    args = _build_parser().parse_args([])

    assert args.limit == mod.DEFAULT_PUBLISH_LIMIT
    assert args.max_open_prs == mod.DEFAULT_MAX_OPEN_PRS
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
            _branch("codex/empty-diff", unique_commit_count=2),
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
            "codex/empty-diff": False,
        },
        is_patch_equivalent={"codex/cherry-picked": True},
        has_pr_diff={"codex/empty-diff": False},
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
    assert by_branch["codex/empty-diff"].reason == "empty_pr_diff"


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


def test_select_publishable_branches_skips_remote_branch_conflicts() -> None:
    decisions = select_publishable_branches(
        [_branch("codex/conflict", head_sha="abc1234")],
        [],
        set(),
        cutoff=datetime.now(UTC) - timedelta(hours=24),
        base="main",
        is_merged={"codex/conflict": False},
        remote_head_lookup={"codex/conflict": "def5678"},
    )

    assert decisions[0].reason == "remote_branch_conflict"


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
    assert mod._looks_related_subject(
        "[codex] Restore prompt engine SDK contract",
        "fix(sdk): restore prompt engine contracts",
    )


def test_related_search_queries_include_stable_nouns() -> None:
    assert mod._related_search_queries("fix(autonomy): parse multiline github app keys") == [
        "fix(autonomy): parse multiline github app keys",
        "multiline github app keys",
    ]


def test_github_base_ref_strips_remote_tracking_prefix() -> None:
    assert mod._github_base_ref("origin/main") == "main"
    assert mod._github_base_ref("main") == "main"


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


def test_run_uses_env_overrides_for_git_timeout(monkeypatch: Any, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setenv("ARAGORA_AUTOMATION_GIT_TIMEOUT_SECONDS", "90")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = mod._run(["git", "status"], cwd=tmp_path)

    assert result.returncode == 0
    assert recorded["timeout"] == 90


def test_push_branch_disables_local_pre_push_hooks_by_default(
    monkeypatch: Any, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(
        args: list[str],
        *,
        cwd: Path,
        check: bool = False,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["env_overrides"] = env_overrides
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setenv("SKIP", "gitleaks")
    monkeypatch.setenv("ARAGORA_AUTOMATION_PRE_PUSH_SKIP", "mypy-baseline")
    monkeypatch.setattr(mod, "_run", fake_run)

    mod._push_branch(tmp_path, "codex/test-branch", "origin/main")

    assert recorded["args"] == ["git", "push", "--no-verify", "origin", "codex/test-branch"]
    assert recorded["env_overrides"] == {"SKIP": "gitleaks,mypy-baseline"}


def test_push_branch_can_opt_into_git_pre_push_hooks(monkeypatch: Any, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(
        args: list[str],
        *,
        cwd: Path,
        check: bool = False,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["env_overrides"] = env_overrides
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setenv("ARAGORA_AUTOMATION_GIT_PUSH_VERIFY", "true")
    monkeypatch.setattr(mod, "_run", fake_run)

    mod._push_branch(tmp_path, "codex/test-branch", None)

    assert recorded["args"] == ["git", "push", "-u", "origin", "codex/test-branch"]


def test_run_uses_user_auth_for_gh_write_ops(monkeypatch: Any, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    def fake_gh_run(
        args: list[str],
        *,
        timeout: float,
        prefer_app: bool,
        write_op: bool,
        env: dict[str, str],
        max_retries: int,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["prefer_app"] = prefer_app
        recorded["write_op"] = write_op
        recorded["env"] = env
        recorded["max_retries"] = max_retries
        return subprocess.CompletedProcess(args=["gh", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "gh_subprocess_run", fake_gh_run)

    result = mod._run(["gh", "pr", "create", "--base", "main"], cwd=tmp_path)

    assert result.returncode == 0
    assert recorded["args"][:2] == ["pr", "create"]
    assert recorded["prefer_app"] is True
    assert recorded["write_op"] is True
    assert recorded["max_retries"] == 0


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


def test_main_reports_github_health_when_unavailable(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "_local_codex_branches", lambda repo_root: [])
    monkeypatch.setattr(mod, "_list_worktrees", lambda repo_root, branch_filter=None: [])
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=True,
            api_ok=False,
            mode="connectivity_failed",
            error="error connecting to api.github.com",
            repo=str(tmp_path),
        ),
    )

    exit_code = mod.main(["--repo", str(tmp_path), "--json"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert '"mode": "connectivity_failed"' in out
    assert '"decisions": []' in out


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


def test_publish_decisions_records_publish_failures_and_continues(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_push(repo_root: Path, branch: str, upstream: str | None) -> None:
        calls.append(f"push:{branch}")
        if branch == "codex/bad-branch":
            raise RuntimeError("non-fast-forward")

    monkeypatch.setattr(mod, "_ensure_gh_auth", lambda repo_root: None)
    monkeypatch.setattr(mod, "_push_branch", fake_push)
    monkeypatch.setattr(mod, "_existing_pr_number", lambda repo_root, repo, branch, base: None)
    monkeypatch.setattr(mod, "_create_pr", lambda repo_root, repo, branch, base: 2001)
    monkeypatch.setattr(
        mod, "_add_labels", lambda repo_root, repo, number, labels: calls.append(f"label:{number}")
    )

    results = mod._publish_decisions(
        tmp_path,
        "synaptent/aragora",
        "main",
        [
            mod.PublishDecision(
                branch="codex/bad-branch",
                eligible=True,
                reason="eligible",
                subject="bad branch",
                head_sha="abc1234",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            ),
            mod.PublishDecision(
                branch="codex/good-branch",
                eligible=True,
                reason="eligible",
                subject="good branch",
                head_sha="def5678",
                unique_commit_count=1,
                upstream=None,
                committed_at=datetime.now(UTC).isoformat(),
                worktree_paths=[],
            ),
        ],
        limit=5,
        open_pr_count=0,
        max_open_prs=3,
        labels=["codex"],
    )

    assert results == [
        {
            "branch": "codex/bad-branch",
            "status": "publish_failed",
            "subject": "bad branch",
            "head_sha": "abc1234",
            "reason": "non-fast-forward",
        },
        {
            "branch": "codex/good-branch",
            "status": "published",
            "pr_number": 2001,
            "subject": "good branch",
            "head_sha": "def5678",
        },
    ]
    assert calls == ["push:codex/bad-branch", "push:codex/good-branch", "label:2001"]


def test_main_pauses_apply_when_open_codex_queue_is_unhealthy(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "_local_codex_branches", lambda repo_root: [])
    monkeypatch.setattr(mod, "_list_worktrees", lambda repo_root, branch_filter=None: [])
    monkeypatch.setattr(mod, "_branches_with_pr_history", lambda repo_root, repo, branches: set())
    monkeypatch.setattr(
        mod,
        "_branches_with_resolved_related_work",
        lambda repo_root, repo, branches: set(),
    )
    monkeypatch.setattr(
        mod,
        "_open_codex_prs",
        lambda repo_root, repo: [
            {"headRefName": "codex/a", "mergeStateStatus": "BLOCKED"},
            {"headRefName": "codex/b", "mergeStateStatus": "DIRTY"},
        ],
    )
    monkeypatch.setattr(mod, "_branch_is_merged", lambda repo_root, base, branch: False)
    monkeypatch.setattr(
        mod, "_branch_patch_equivalent_to_base", lambda repo_root, base, branch: False
    )
    monkeypatch.setattr(mod, "_branch_remote_head", lambda repo_root, branch: None)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(tmp_path),
        ),
    )
    publish_called = False

    def fake_publish(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal publish_called
        publish_called = True
        return []

    monkeypatch.setattr(mod, "_publish_decisions", fake_publish)

    exit_code = mod.main(["--repo", str(tmp_path), "--apply", "--json"])

    assert exit_code == 0
    assert publish_called is False
    out = capsys.readouterr().out
    assert '"publish_paused_reason": "open_pr_queue_unhealthy"' in out


def test_main_does_not_pause_for_green_review_required_codex_pr(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "_local_codex_branches", lambda repo_root: [])
    monkeypatch.setattr(mod, "_list_worktrees", lambda repo_root, branch_filter=None: [])
    monkeypatch.setattr(mod, "_branches_with_pr_history", lambda repo_root, repo, branches: set())
    monkeypatch.setattr(
        mod,
        "_branches_with_resolved_related_work",
        lambda repo_root, repo, branches: set(),
    )
    monkeypatch.setattr(
        mod,
        "_open_codex_prs",
        lambda repo_root, repo: [
            {
                "headRefName": "codex/review-only",
                "isDraft": False,
                "mergeStateStatus": "BLOCKED",
                "reviewDecision": "REVIEW_REQUIRED",
                "statusCheckRollup": [
                    {"conclusion": "SUCCESS"},
                    {"conclusion": "SKIPPED"},
                ],
            },
        ],
    )
    monkeypatch.setattr(mod, "_branch_is_merged", lambda repo_root, base, branch: False)
    monkeypatch.setattr(
        mod, "_branch_patch_equivalent_to_base", lambda repo_root, base, branch: False
    )
    monkeypatch.setattr(mod, "_branch_remote_head", lambda repo_root, branch: None)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(tmp_path),
        ),
    )
    publish_called = False

    def fake_publish(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal publish_called
        publish_called = True
        return []

    monkeypatch.setattr(mod, "_publish_decisions", fake_publish)

    exit_code = mod.main(["--repo", str(tmp_path), "--apply", "--json"])

    assert exit_code == 0
    assert publish_called is True
    out = capsys.readouterr().out
    assert '"publish_paused_reason"' not in out
    assert '"unhealthy_open_pr_count": 0' in out
