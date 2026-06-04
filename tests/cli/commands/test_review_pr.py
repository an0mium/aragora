from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


def test_review_pr_parser_accepts_no_publish_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "review-pr",
            "1137",
            "--no-publish-review",
        ]
    )
    assert args.command == "review-pr"
    assert args.publish_review is False


def test_review_local_parser_accepts_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "review-local",
            "--diff",
            "x.diff",
            "--reviewer",
            "claude",
            "--worker-model",
            "codex",
            "--json",
        ]
    )
    assert args.command == "review-local"
    assert args.diff == "x.diff"
    assert args.reviewer == "claude"
    assert args.worker_model == "codex"
    assert args.json_output is True


@pytest.mark.asyncio
async def test_run_review_local_writes_receipt_without_github(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_github(*_: object, **__: object) -> None:
        raise AssertionError("review-local must not touch GitHub")

    monkeypatch.setattr(review_pr, "_fetch_pr_target", _no_github)
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", _no_github)

    async def _fake_generate(*_: object, **__: object) -> dict[str, object]:
        return {
            "candidate": {"provider": "claude", "label": "claude:max-02"},
            "response": '{"status":"passed","summary":"LGTM","findings":[]}',
            "attempts": [{"candidate": "claude:max-02", "stage": "generate", "detail": "ok"}],
        }

    monkeypatch.setattr(review_pr, "generate_review_response", _fake_generate)

    result = await review_pr.run_review_local(
        diff_text="diff --git a/foo b/foo\n+ok\n",
        repo_root=tmp_path,
        reviewer="claude",
        worker_model="codex",
    )

    assert result["final_status"] == "passed"
    assert result["review"]["candidate"] == {"provider": "claude", "label": "claude:max-02"}
    run_dir = Path(result["artifact_dir"])
    assert run_dir.is_relative_to(tmp_path / ".aragora" / "review-local")
    assert (run_dir / "input.diff").read_text().startswith("diff --git")
    persisted = json.loads((run_dir / "review.json").read_text())
    assert persisted["kind"] == "review_local"
    assert persisted["final_status"] == "passed"
    assert persisted["worker_model"] == "codex"


@pytest.mark.asyncio
async def test_run_review_local_records_routing_failure_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _routing_failure(*_: object, **__: object) -> dict[str, object]:
        raise review_pr.ReviewRoutingError(
            [
                {
                    "candidate": "claude:max-01",
                    "stage": "preflight",
                    "kind": "claude_unauthenticated",
                    "detail": "expired",
                }
            ],
            category="claude_pool_unauthenticated",
            public_message=(
                "No authenticated Claude Max profiles. "
                "Run scripts/claude_profiles_bootstrap.sh login."
            ),
        )

    monkeypatch.setattr(review_pr, "generate_review_response", _routing_failure)

    result = await review_pr.run_review_local(
        diff_text="diff --git a/foo b/foo\n+ok\n",
        repo_root=tmp_path,
        reviewer="claude",
        worker_model="codex",
    )

    assert result["final_status"] == "blocked_nonreviewable"
    review = result["review"]
    assert review["summary"].startswith("No authenticated Claude Max profiles")
    assert review["findings"][0]["category"] == "claude_pool_unauthenticated"
    assert review["findings"][0]["priority"] == "P1"
    assert (Path(result["artifact_dir"]) / "review.json").exists()


def test_cmd_review_local_missing_diff_file_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(["review-local", "--diff", str(tmp_path / "nope.diff")])
    rc = review_pr.cmd_review_local(args)
    assert rc == 1
    assert "cannot read diff" in capsys.readouterr().err


def test_cmd_review_local_truncates_oversized_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = tmp_path / "big.diff"
    big.write_text("diff --git a/x b/x\n" + ("+x\n" * 40000), encoding="utf-8")
    assert big.stat().st_size > review_pr.MAX_DIFF_CHARS

    captured: dict[str, str] = {}

    async def _fake_run(**kwargs: object) -> dict[str, object]:
        captured["diff_text"] = str(kwargs["diff_text"])
        return {"final_status": "passed", "review": {}, "artifact_dir": str(tmp_path)}

    monkeypatch.setattr(review_pr, "run_review_local", _fake_run)
    args = build_parser().parse_args(["review-local", "--diff", str(big), "--json"])
    rc = review_pr.cmd_review_local(args)
    assert rc == 0
    assert len(captured["diff_text"]) <= review_pr.MAX_DIFF_CHARS + 64
    assert "[truncated at" in captured["diff_text"]


def test_cmd_review_local_truncates_oversized_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diff = tmp_path / "small.diff"
    diff.write_text("diff --git a/x b/x\n+ok\n", encoding="utf-8")
    spec = tmp_path / "big-spec.md"
    spec.write_text("x" * (review_pr.MAX_SPEC_CHARS + 1000), encoding="utf-8")

    captured: dict[str, str] = {}

    async def _fake_run(**kwargs: object) -> dict[str, object]:
        captured["spec_text"] = str(kwargs["spec_text"])
        return {"final_status": "passed", "review": {}, "artifact_dir": str(tmp_path)}

    monkeypatch.setattr(review_pr, "run_review_local", _fake_run)
    args = build_parser().parse_args(
        ["review-local", "--diff", str(diff), "--spec", str(spec), "--json"]
    )
    rc = review_pr.cmd_review_local(args)
    assert rc == 0
    assert len(captured["spec_text"]) <= review_pr.MAX_SPEC_CHARS + 64
    assert "[truncated at" in captured["spec_text"]


def test_cmd_review_local_rejects_worker_family_reviewer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diff = tmp_path / "small.diff"
    diff.write_text("diff --git a/x b/x\n+ok\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "review-local",
            "--diff",
            str(diff),
            "--reviewer",
            "openai",
            "--worker-model",
            "codex",
        ]
    )
    rc = review_pr.cmd_review_local(args)
    assert rc == 1
    assert "reviewer must be a non-worker model family" in capsys.readouterr().err


def test_normalize_optional_agent_rejects_placeholder_none() -> None:
    assert review_pr._normalize_optional_agent(None) is None
    assert review_pr._normalize_optional_agent("") is None
    assert review_pr._normalize_optional_agent("None") is None
    assert review_pr._normalize_optional_agent("null") is None
    assert review_pr._normalize_optional_agent(" codex ") == "codex"


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_reviewer", ["grok", "claude", "gemini"])
async def test_run_review_pass_blocks_codex_fallback_for_requested_noncodex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
    requested_reviewer: str,
) -> None:
    async def _fake_generate_review_response(
        *_: object,
        candidate_blocker: object,
        **__: object,
    ) -> dict[str, object]:
        candidate = {"provider": "codex", "label": "codex"}
        assert callable(candidate_blocker)
        blocked = candidate_blocker(candidate)
        assert blocked
        return {
            "candidate": candidate,
            "response": "",
            "attempts": [
                {
                    "candidate": "codex",
                    "stage": "route_guard",
                    "kind": "blocked_nonreviewable",
                    "detail": "Requested reviewer routed to Codex",
                }
            ],
            "blocked": blocked,
        }

    monkeypatch.setattr(review_pr, "generate_review_response", _fake_generate_review_response)

    result = await review_pr._run_review_pass(
        target=sample_target,
        diff_text="diff --git a/foo b/foo\n+ok\n",
        reviewer=requested_reviewer,
        worker_model="codex",
        repo_root=tmp_path,
    )

    assert result.status == "blocked_nonreviewable"
    assert result.candidate == {"provider": "codex", "label": "codex"}
    assert result.findings == [
        {
            "title": "Requested reviewer routed to Codex",
            "body": (
                f"Requested reviewer `{requested_reviewer}` requires non-Codex evidence, but review-pr "
                "selected Codex candidate `codex`. Re-run with a real non-Codex reviewer "
                "or do not count this result as non-Codex quorum evidence."
            ),
            "priority": "P1",
        }
    ]
    assert result.raw_response == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_reviewer", ["grok", "claude", "gemini"])
async def test_run_review_pr_loop_does_not_publish_codex_routed_noncodex_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
    requested_reviewer: str,
) -> None:
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: sample_target)
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", lambda *_: "diff --git a/foo b/foo\n+ok\n")

    async def _fake_review(**_: object) -> review_pr.ReviewPass:
        return review_pr.ReviewPass(
            reviewer=requested_reviewer,
            reviewed_at="2026-03-21T10:00:00+00:00",
            status="blocked_nonreviewable",
            summary="Requested non-Codex reviewer routed to Codex.",
            findings=[
                {
                    "title": "Requested reviewer routed to Codex",
                    "body": "Requested non-Codex reviewer routed to Codex.",
                    "priority": "P1",
                }
            ],
            candidate={"provider": "codex", "label": "codex"},
            attempts=[
                {
                    "candidate": "codex",
                    "stage": "route_guard",
                    "kind": "blocked_nonreviewable",
                    "detail": "Requested reviewer routed to Codex",
                }
            ],
            raw_response="",
        )

    async def _should_not_publish(**_: object) -> dict[str, object]:
        raise AssertionError("_publish_review_outcome should not publish Codex-routed evidence")

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)
    monkeypatch.setattr(review_pr, "_publish_review_outcome", _should_not_publish)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer=requested_reviewer,
        artifact_root=tmp_path / "artifacts",
        publish_review=True,
    )

    assert result["final_status"] == "blocked_nonreviewable"
    assert result["github_review"] == {
        "posted": False,
        "event": None,
        "mode": "advisory",
        "url": None,
        "error": "Requested non-Codex reviewer routed to Codex before review generation.",
    }
    assert result["review_runs"][0]["candidate"] == {"provider": "codex", "label": "codex"}


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
    published: dict[str, Any] = {}

    async def _fake_publish(**kwargs: object) -> dict[str, object]:
        published.update(kwargs)
        return {
            "posted": True,
            "event": "COMMENT",
            "mode": "advisory",
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
    assert result["github_review"]["event"] == "COMMENT"
    assert result["github_review"]["mode"] == "advisory"
    assert published["final_status"] == "passed"
    assert published["advisory_only"] is True

    run_path = Path(result["artifact_dir"]) / "run.json"
    assert run_path.exists()
    persisted = json.loads(run_path.read_text())
    assert persisted["final_status"] == "passed"
    assert persisted["pr"]["number"] == 1137
    assert persisted["github_review"]["posted"] is True
    assert persisted["github_review_mode"] == "advisory"
    assert persisted["publish_review"] is True


@pytest.mark.asyncio
async def test_run_review_pr_loop_skips_github_review_when_publish_disabled(
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
            raw_response="{}",
        )

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)

    async def _should_not_publish(**_: object) -> dict[str, object]:
        raise AssertionError("_publish_review_outcome should not be called")

    monkeypatch.setattr(review_pr, "_publish_review_outcome", _should_not_publish)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        artifact_root=tmp_path / "artifacts",
        publish_review=False,
    )

    assert result["final_status"] == "passed"
    assert result["github_review"] == {
        "posted": False,
        "event": None,
        "mode": "advisory",
        "url": None,
        "error": None,
    }
    persisted = json.loads((Path(result["artifact_dir"]) / "run.json").read_text())
    assert persisted["publish_review"] is False


@pytest.mark.asyncio
async def test_run_review_pr_loop_records_routing_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: sample_target)
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", lambda *_: "diff --git a/foo b/foo\n+ok\n")

    async def _routing_failure(*_: object, **__: object) -> dict[str, object]:
        raise review_pr.ReviewRoutingError(
            [
                {
                    "candidate": "claude:max-01",
                    "stage": "generate",
                    "kind": "auth_or_billing",
                    "detail": "Credit balance is too low",
                }
            ],
            category="billing_exhausted",
            public_message="Reviewer capacity is exhausted.",
        )

    monkeypatch.setattr(review_pr, "generate_review_response", _routing_failure)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        artifact_root=tmp_path / "artifacts",
        publish_review=False,
    )

    assert result["final_status"] == "blocked_nonreviewable"
    assert result["github_review"] == {
        "posted": False,
        "event": None,
        "mode": "advisory",
        "url": None,
        "error": None,
    }
    review_run = result["review_runs"][0]
    assert review_run["summary"] == "Reviewer capacity is exhausted."
    assert review_run["findings"] == [
        {
            "title": "Review routing failed",
            "body": "Reviewer capacity is exhausted.",
            "priority": "P1",
            "category": "billing_exhausted",
        }
    ]
    assert review_run["attempts"] == [
        {
            "candidate": "claude:max-01",
            "stage": "generate",
            "kind": "auth_or_billing",
            "detail": "Credit balance is too low",
        }
    ]
    persisted = json.loads((Path(result["artifact_dir"]) / "run.json").read_text())
    assert persisted["final_status"] == "blocked_nonreviewable"


@pytest.mark.asyncio
async def test_review_pr_loop_changes_requested_without_fixer_does_not_run_fix_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: sample_target)
    monkeypatch.setattr(review_pr, "_fetch_pr_diff", lambda *_: "diff --git a/foo b/foo\n+bad\n")

    async def _fake_review(**_: object) -> review_pr.ReviewPass:
        return review_pr.ReviewPass(
            reviewer="claude",
            reviewed_at="2026-03-21T10:00:00+00:00",
            status="changes_requested",
            summary="Fix the crash",
            findings=[{"title": "Crash", "body": "Fix it", "priority": "P1"}],
            candidate={"label": "claude:max-01"},
            attempts=[],
            raw_response="{}",
        )

    async def _should_not_fix(**_: object) -> review_pr.FixPass:
        raise AssertionError("_run_fix_pass should not be called without a real fixer")

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)
    monkeypatch.setattr(review_pr, "_run_fix_pass", _should_not_fix)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        fixer="None",
        artifact_root=tmp_path / "artifacts",
        publish_review=False,
    )

    assert result["final_status"] == "changes_requested"
    assert result["fixer"] is None
    assert result["fixer_requested"] is False
    assert result["fix_run"] is None
    assert not (Path(result["artifact_dir"]) / "fix.json").exists()


@pytest.mark.asyncio
async def test_review_pr_loop_detects_head_sha_drift_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    refreshed_target = review_pr.PullRequestTarget(
        **{
            **asdict(sample_target),
            "head_sha": "def456",
        }
    )
    targets = [sample_target, refreshed_target]
    monkeypatch.setattr(review_pr, "_fetch_pr_target", lambda *_, **__: targets.pop(0))
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
            raw_response="{}",
        )

    async def _should_not_publish(**_: object) -> dict[str, object]:
        raise AssertionError("_publish_review_outcome should not be called for a stale review")

    monkeypatch.setattr(review_pr, "_run_review_pass", _fake_review)
    monkeypatch.setattr(review_pr, "_publish_review_outcome", _should_not_publish)

    result = await review_pr.run_review_pr_loop(
        pr_ref="1137",
        repo_root=tmp_path,
        reviewer="claude",
        artifact_root=tmp_path / "artifacts",
        publish_review=True,
    )

    assert result["final_status"] == "blocked_nonreviewable"
    assert result["head_sha_stale"] is True
    assert result["observed_head_sha_before"] == "abc123"
    assert result["observed_head_sha_after"] == "def456"
    assert result["github_review"]["posted"] is False
    assert "changed during review" in result["github_review"]["error"]


@pytest.mark.asyncio
async def test_run_review_pr_loop_auto_reruns_after_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_target: review_pr.PullRequestTarget,
) -> None:
    fetched_targets = [
        sample_target,
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
    published: dict[str, Any] = {}

    async def _fake_publish(**kwargs: object) -> dict[str, object]:
        published.update(kwargs)
        return {"posted": True, "event": "COMMENT", "mode": "advisory", "url": None, "error": None}

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
    assert result["github_review"]["event"] == "COMMENT"
    assert published["final_status"] == "passed"
    assert published["fix_run"]["status"] == "applied"
    assert published["advisory_only"] is True


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
        advisory_only=True,
    )

    assert "## Aragora review-pr: advisory findings" in body
    assert "- Review mode: `advisory comment only`" in body
    assert "machine review is advisory only" in body
    assert "- Final status: `changes_requested`" in body
    assert "- Review route: `claude:max-01`" in body
    assert "### Fix Loop" in body
    assert "- [P1] Crash (aragora/cli/commands/review_pr.py): Guard the empty branch path." in body


def test_github_review_event_defaults_to_comment_in_advisory_mode() -> None:
    assert review_pr._github_review_event("passed", advisory_only=True) == "COMMENT"
    assert review_pr._github_review_event("changes_requested", advisory_only=True) == "COMMENT"
    assert review_pr._github_review_event("blocked_nonreviewable", advisory_only=True) == "COMMENT"


def test_github_review_event_preserves_status_reviews_when_not_advisory() -> None:
    assert review_pr._github_review_event("passed", advisory_only=False) == "APPROVE"
    assert (
        review_pr._github_review_event("changes_requested", advisory_only=False)
        == "REQUEST_CHANGES"
    )
    assert review_pr._github_review_event("blocked_nonreviewable", advisory_only=False) == "COMMENT"


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


def test_cleanup_worktree_logs_parent_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "safe_worktree_cleanup.py").write_text("# stub\n")
    worktree_path = tmp_path / "scratch" / "wt"
    worktree_path.parent.mkdir(parents=True)

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout='{"status":"removed"}', stderr=""
        )

    monkeypatch.setattr(review_pr.subprocess, "run", _fake_run)

    with patch.object(Path, "rmdir", autospec=True, side_effect=OSError("directory busy")):
        with patch.object(review_pr.logger, "debug") as debug:
            review_pr._cleanup_worktree(repo_root, worktree_path)

    debug.assert_called_once()
    assert "review-pr parent cleanup skipped for" in debug.call_args.args[0]
    assert debug.call_args.args[1] == worktree_path.parent


def test_is_generated_diff_path_flags_generated_and_lock_files() -> None:
    assert review_pr._is_generated_diff_path(".mypy-baseline")
    assert review_pr._is_generated_diff_path("sdk/python/aragora/generated_types.py")
    assert review_pr._is_generated_diff_path("frontend/package-lock.json")
    assert review_pr._is_generated_diff_path("uv.lock")
    assert review_pr._is_generated_diff_path("tests/__snapshots__/foo.snap")
    # Real source must NOT be flagged.
    assert not review_pr._is_generated_diff_path("aragora/cli/commands/review_pr.py")
    assert not review_pr._is_generated_diff_path("scripts/run_typecheck_gate.py")


def test_strip_generated_file_diffs_drops_only_generated_sections() -> None:
    diff = (
        "diff --git a/.mypy-baseline b/.mypy-baseline\n"
        "--- a/.mypy-baseline\n+++ b/.mypy-baseline\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/aragora/foo.py b/aragora/foo.py\n"
        "--- a/aragora/foo.py\n+++ b/aragora/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )
    filtered, dropped = review_pr._strip_generated_file_diffs(diff)
    assert dropped == [".mypy-baseline"]
    assert "aragora/foo.py" in filtered
    assert "x = 2" in filtered
    assert ".mypy-baseline" not in filtered


def _fake_gh_diff(raw: str):
    return lambda *a, **k: subprocess.CompletedProcess(
        args=["gh"], returncode=0, stdout=raw, stderr=""
    )


def test_fetch_pr_diff_preserves_code_when_generated_file_is_huge(
    monkeypatch: pytest.MonkeyPatch, sample_target: review_pr.PullRequestTarget
) -> None:
    # Regression: a generated file larger than MAX_DIFF_CHARS must not crowd out
    # the human-authored change (previously caused blocked_nonreviewable).
    huge_baseline = "x" * (review_pr.MAX_DIFF_CHARS * 2)
    raw = (
        "diff --git a/.mypy-baseline b/.mypy-baseline\n"
        "--- a/.mypy-baseline\n+++ b/.mypy-baseline\n"
        f"@@ -1 +1 @@\n-{huge_baseline}\n+{huge_baseline}\n"
        "diff --git a/aragora/foo.py b/aragora/foo.py\n"
        "--- a/aragora/foo.py\n+++ b/aragora/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )
    monkeypatch.setattr(review_pr, "_run_command", _fake_gh_diff(raw))
    result = review_pr._fetch_pr_diff(sample_target)
    assert "aragora/foo.py" in result
    assert "x = 2" in result
    assert "omitted 1 generated/lock file" in result
    assert huge_baseline not in result  # the giant generated content is gone


def test_fetch_pr_diff_all_generated_returns_informative_note(
    monkeypatch: pytest.MonkeyPatch, sample_target: review_pr.PullRequestTarget
) -> None:
    raw = (
        "diff --git a/.mypy-baseline b/.mypy-baseline\n"
        "--- a/.mypy-baseline\n+++ b/.mypy-baseline\n@@ -1 +1 @@\n-a\n+b\n"
    )
    monkeypatch.setattr(review_pr, "_run_command", _fake_gh_diff(raw))
    result = review_pr._fetch_pr_diff(sample_target)
    assert "no human-authored source changes to review" in result


def test_fetch_pr_diff_raises_on_truly_empty(
    monkeypatch: pytest.MonkeyPatch, sample_target: review_pr.PullRequestTarget
) -> None:
    monkeypatch.setattr(review_pr, "_run_command", _fake_gh_diff("   \n"))
    with pytest.raises(RuntimeError, match="no diff to review"):
        review_pr._fetch_pr_diff(sample_target)
