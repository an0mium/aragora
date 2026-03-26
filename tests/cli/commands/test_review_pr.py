from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from aragora.cli.commands import review_pr
from aragora.cli.parser import build_parser


@pytest.fixture
def sample_target() -> review_pr.PullRequestTarget:
    return review_pr.PullRequestTarget(
        number=1137,
        repo="synaptent/aragora",
        url="https://github.com/synaptent/aragora/pull/1137",
        title="Surface unified pipeline live state",
        base_ref="main",
        head_ref="codex/swarm-f6852e63-pipeline-dag-live-status-slice",
        head_sha="abc123",
        files=["aragora/server/handlers/canvas_pipeline.py"],
        mergeable="MERGEABLE",
    )


def test_review_pr_parser_accepts_fix_loop_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "review-pr",
            "1137",
            "--reviewer",
            "claude",
            "--fixer",
            "codex",
            "--auto-rerun",
            "--json",
        ]
    )
    assert args.command == "review-pr"
    assert args.reviewer == "claude"
    assert args.fixer == "codex"
    assert args.auto_rerun is True
    assert args.json_output is True


@pytest.mark.asyncio
async def test_run_review_pr_loop_review_only_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: sample_target)
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", lambda *_: "diff --git a/foo b/foo\n+ok\n")

    async def _fake_review(**_: object) -> review_pr.ReviewPass:
        return review_pr.ReviewPass(
            reviewer="claude",
            reviewed_at="2026-03-21T10:00:00+00:00",
            status="passed",
            summary="Looks good",
            findings=[],
            candidate={"label": "claude:max-01"},
            attempts=[],
            raw_response='{"status":"passed","summary":"Looks good","findings":[]}',
        )

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)
    published: dict[str, object] = {}

    async def _fake_publish(**kwargs: object) -> dict[str, object]:
        published.update(kwargs)
        return {
            "posted": True,
            "event": "APPROVE",
            "url": "https://github.com/review/1",
            "error": None,
        }

    monkeypatch.setattr(review_pr, "_publish_review_outcome", _fake_publish)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        artifact_root=tmp_path / "artifacts",
    )

    assert result["final_status"] == "passed"
    assert result["fix_run"] is None
    assert len(result["review_runs"]) == 1
    assert result["github_review"]["posted"] is True
    assert result["github_review"]["event"] == "APPROVE"
    assert published["final_status"] == "passed"

    run_path = Path(result["artifact_dir"]) / "run.json"
    assert run_path.exists()
    persisted = json.loads(run_path.read_text())
    assert persisted["final_status"] == "passed"
    assert persisted["pr"]["number"] == 1137
    assert persisted["github_review"]["posted"] is True


@pytest.mark.asyncio
async def test_run_review_pr_loop_auto_reruns_after_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    fetched_targets = [
        sample_target,
        review_pr.PullRequestTarget(
            **{
                **asdict(sample_target),
                "head_sha": "def456",
            }
        ),
    ]
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: fetched_targets.pop(0))
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", lambda *_: "diff --git a/foo b/foo\n+ok\n")

    review_calls = 0

    async def _fake_review(**_: object) -> review_pr.ReviewPass:
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return review_pr.ReviewPass(
                reviewer="claude",
                reviewed_at="2026-03-21T10:00:00+00:00",
                status="changes_requested",
                summary="Fix the crash",
                findings=[{"title": "Crash", "body": "Fix the closure bug", "priority": "P1"}],
                candidate={"label": "claude:max-01"},
                attempts=[],
                raw_response="{}",
            )
        return review_pr.ReviewPass(
            reviewer="claude",
            reviewed_at="2026-03-21T10:10:00+00:00",
            status="passed",
            summary="Clean now",
            findings=[],
            candidate={"label": "claude:max-01"},
            attempts=[],
            raw_response="{}",
        )

    async def _fake_fix(**_: object) -> review_pr.FixPass:
        return review_pr.FixPass(
            fixer="codex",
            started_at="2026-03-21T10:02:00+00:00",
            completed_at="2026-03-21T10:05:00+00:00",
            status="applied",
            worktree_path=str(tmp_path / "wt"),
            pushed=True,
            head_sha="def456",
            commit_shas=["deadbeef"],
            changed_paths=["aragora/server/handlers/canvas_pipeline.py"],
        )

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)
    monkeypatch.setattr(review_pr, "_run_fix_pass", _fake_fix)
    published: dict[str, object] = {}

    async def _fake_publish(**kwargs: object) -> dict[str, object]:
        published.update(kwargs)
        return {"posted": True, "event": "APPROVE", "url": None, "error": None}

    monkeypatch.setattr(review_pr, "_publish_review_outcome", _fake_publish)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        fixer="codex",
        auto_rerun=True,
        artifact_root=tmp_path / "artifacts",
    )

    assert result["final_status"] == "passed"
    assert len(result["review_runs"]) == 2
    assert result["fix_run"]["status"] == "applied"
    assert result["pr"]["head_sha"] == "def456"
    assert result["github_review"]["posted"] is True
    assert published["final_status"] == "passed"
    assert published["fix_run"]["status"] == "applied"


def test_build_github_review_body_includes_fix_and_findings(
    sample_target: review_pr.PullRequestTarget,
) -> None:
    body = review_pr._build_github_review_body(
        target=sample_target,
        latest_review={
            "reviewer": "claude",
            "reviewed_at": "2026-03-21T10:00:00+00:00",
            "summary": "Fix the crash before merge.",
            "findings": [
                {
                    "title": "Crash",
                    "body": "Guard the empty branch path.",
                    "file": "aragora/cli/commands/review_pr.py",
                    "priority": "P1",
                }
            ],
            "candidate": {"label": "claude:max-01"},
        },
        fix_run={
            "fixer": "codex",
            "status": "applied",
            "pushed": True,
            "head_sha": "def456",
        },
        final_status="changes_requested",
        review_run_count=2,
    )

    assert "## Aragora review-pr: changes requested" in body
    assert "- Final status: `changes_requested`" in body
    assert "- Review route: `claude:max-01`" in body
    assert "### Fix Loop" in body
    assert "- [P1] Crash (aragora/cli/commands/review_pr.py): Guard the empty branch path." in body


def test_cleanup_worktree_uses_safe_cleanup_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "safe_worktree_cleanup.py").write_text("# stub\n")
    worktree_path = tmp_path / "scratch" / "wt"
    worktree_path.parent.mkdir(parents=True)
    worktree_path.parent.joinpath("keep").write_text("x")

    calls: list[list[str]] = []

    def _fake_run(*args, **kwargs):
        calls.append(list(args[0]))
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout='{"status":"removed"}', stderr=""
        )

    monkeypatch.setattr(review_pr.subprocess, "run", _fake_run)

    review_pr._cleanup_worktree(repo_root, worktree_path)

    assert calls == [
        [
            review_pr.sys.executable,
            str(repo_root / "scripts" / "safe_worktree_cleanup.py"),
            "--repo",
            str(repo_root),
            "remove",
            str(worktree_path),
            "--purge-path",
            "--json",
        ]
    ]
