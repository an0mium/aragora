"""Tests for ``scripts/stale_pr_janitor.py`` (stale-tail classifier, NON-DESTRUCTIVE).

All boundaries (PR listing, comment fetching, comment posting) are injected;
no test touches the network or spawns a subprocess. Style mirrors
``tests/scripts/test_boss_pr_janitor.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


janitor = _load_module("stale_pr_janitor.py")

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)

OLD = "2026-06-01T00:00:00Z"  # 11 days before NOW: stale
RECENT = "2026-06-11T12:00:00Z"  # 1 day before NOW: active


def _pr(
    number: int,
    *,
    head: str = "codex/feature-branch",
    draft: bool = True,
    mergeable: str = "MERGEABLE",
    created: str = OLD,
    updated: str = OLD,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "headRefName": head,
        "isDraft": draft,
        "title": f"codex: change {number}",
        "mergeable": mergeable,
        "createdAt": created,
        "updatedAt": updated,
        "statusCheckRollup": checks if checks is not None else [],
    }


def _failing_check(name: str = "pytest") -> dict[str, Any]:
    return {"name": name, "state": "FAILURE", "conclusion": "FAILURE"}


class CommentRecorder:
    def __init__(self, results: dict[int, tuple[bool, str]] | None = None) -> None:
        self.calls: list[tuple[int, str]] = []
        self.results = results or {}

    def __call__(self, pr: int, body: str) -> tuple[bool, str]:
        self.calls.append((pr, body))
        return self.results.get(pr, (True, ""))


def _run(
    prs: list[dict[str, Any]],
    *,
    queue_file: Path,
    apply: bool = False,
    existing_comments: dict[int, list[str]] | None = None,
    post: CommentRecorder | None = None,
    max_comments: int = 10,
    stale_days: int = 4,
    inactive_days: int = 2,
    breaker_threshold: int = 3,
    branch_prefix: str = "codex/",
) -> tuple[dict[str, Any], CommentRecorder, list[int]]:
    post = post or CommentRecorder()
    comments = existing_comments or {}
    fetch_calls: list[int] = []

    def fetch_comments(pr: int) -> list[str]:
        fetch_calls.append(pr)
        return comments.get(pr, [])

    summary = janitor.run_janitor(
        list_prs=lambda: prs,
        fetch_comments=fetch_comments,
        post_comment=post,
        apply=apply,
        queue_file=str(queue_file),
        branch_prefix=branch_prefix,
        stale_days=stale_days,
        inactive_days=inactive_days,
        max_comments=max_comments,
        breaker_threshold=breaker_threshold,
        now=NOW,
        log=lambda line: None,
    )
    return summary, post, fetch_calls


def _read_queue(queue_file: Path) -> list[dict[str, Any]]:
    return json.loads(queue_file.read_text(encoding="utf-8"))


def _classifications(summary: dict[str, Any]) -> dict[int, str]:
    return {entry["pr"]: entry["classification"] for entry in summary["classified"]}


# ---------------------------------------------------------------------------
# Selection: prefix + staleness
# ---------------------------------------------------------------------------


def test_young_pr_not_selected(tmp_path: Path) -> None:
    young = _pr(1, created="2026-06-10T00:00:00Z")  # 2 days old < stale_days=4
    summary, _, _ = _run([young], queue_file=tmp_path / "q.json")
    assert summary["classified"] == []


def test_non_codex_prefix_not_selected(tmp_path: Path) -> None:
    summary, _, _ = _run(
        [_pr(1, head="feature/manual"), _pr(2, head="codex/auto", mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
    )
    assert list(_classifications(summary)) == [2]


def test_ready_and_draft_prs_both_selected(tmp_path: Path) -> None:
    summary, _, _ = _run(
        [
            _pr(1, draft=True, mergeable="CONFLICTING"),
            _pr(2, draft=False, mergeable="CONFLICTING"),
        ],
        queue_file=tmp_path / "q.json",
    )
    assert sorted(_classifications(summary)) == [1, 2]


def test_unparseable_created_at_skipped_fail_safe(tmp_path: Path) -> None:
    summary, _, _ = _run(
        [_pr(1, created="not-a-date", mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
    )
    assert summary["classified"] == []


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_recent_activity_classified_active_and_skipped(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    summary, post, _ = _run(
        [_pr(1, updated=RECENT, mergeable="CONFLICTING")],
        queue_file=qf,
        apply=True,
    )
    assert _classifications(summary) == {1: "active"}
    assert summary["queued"] == []
    assert post.calls == []
    assert not qf.exists(), "active PRs must not touch the queue"


def test_conflicting_classified_restack(tmp_path: Path) -> None:
    summary, _, _ = _run(
        [_pr(1, mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
    )
    assert _classifications(summary) == {1: "restack"}
    assert summary["queued"] == [1]


def test_failing_checks_classified_blocked_with_check_names(tmp_path: Path) -> None:
    summary, _, _ = _run(
        [_pr(1, checks=[_failing_check("pytest"), _failing_check("ruff")])],
        queue_file=tmp_path / "q.json",
    )
    entry = summary["classified"][0]
    assert entry["classification"] == "blocked"
    assert entry["failing_checks"] == ["pytest", "ruff"]
    assert "pytest" in entry["reason"] and "ruff" in entry["reason"]
    assert summary["queued"] == [1]


def test_green_mergeable_classified_promotable_recommendation_only(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    summary, post, _ = _run(
        [_pr(1, checks=[{"name": "ci", "state": "SUCCESS", "conclusion": "SUCCESS"}])],
        queue_file=qf,
        apply=True,
    )
    assert _classifications(summary) == {1: "promotable"}
    assert summary["recommendations"] == [
        {"pr": 1, "recommendation": "run scripts/pr_ready_triage.py (no action taken here)"}
    ]
    assert summary["queued"] == []
    assert post.calls == []
    assert not qf.exists(), "promotable PRs must not be queued"


def test_pending_checks_classified_indeterminate_and_skipped(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    summary, post, _ = _run(
        [_pr(1, checks=[{"name": "ci", "status": "IN_PROGRESS"}])],
        queue_file=qf,
        apply=True,
    )
    assert _classifications(summary) == {1: "indeterminate"}
    assert summary["queued"] == []
    assert post.calls == []
    assert not qf.exists()


# ---------------------------------------------------------------------------
# Restack queue: write, merge, dedupe, never-drop, corruption
# ---------------------------------------------------------------------------


def test_queue_entries_have_required_fields(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    _run([_pr(1, head="codex/conflicted", mergeable="CONFLICTING")], queue_file=qf)
    entries = _read_queue(qf)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["pr"] == 1
    assert entry["head_ref"] == "codex/conflicted"
    assert entry["classification"] == "restack"
    assert entry["reason"]
    assert entry["detected_at"] == "2026-06-12T12:00:00Z"


def test_queue_merge_preserves_entries_not_classified_this_run(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    pre_existing = [
        {
            "pr": 99,
            "head_ref": "codex/old-thing",
            "classification": "restack",
            "reason": "previously detected",
            "detected_at": "2026-06-01T00:00:00Z",
        }
    ]
    qf.write_text(json.dumps(pre_existing), encoding="utf-8")

    _run([_pr(1, mergeable="CONFLICTING")], queue_file=qf)

    entries = _read_queue(qf)
    assert {e["pr"] for e in entries} == {1, 99}
    kept = next(e for e in entries if e["pr"] == 99)
    assert kept == pre_existing[0], "entries from earlier runs must never be dropped or altered"


def test_queue_dedupes_by_pr_number_current_run_wins(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    qf.write_text(
        json.dumps(
            [
                {
                    "pr": 1,
                    "head_ref": "codex/feature-branch",
                    "classification": "blocked",
                    "reason": "old reason",
                    "detected_at": "2026-06-01T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    _run([_pr(1, mergeable="CONFLICTING")], queue_file=qf)

    entries = _read_queue(qf)
    assert len(entries) == 1, "dedupe by PR number"
    assert entries[0]["classification"] == "restack"
    assert entries[0]["detected_at"] == "2026-06-12T12:00:00Z"


def test_opaque_existing_entries_preserved(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    qf.write_text(json.dumps([{"note": "hand-written marker, no pr key"}]), encoding="utf-8")

    _run([_pr(1, mergeable="CONFLICTING")], queue_file=qf)

    entries = _read_queue(qf)
    assert {"note": "hand-written marker, no pr key"} in entries
    assert any(e.get("pr") == 1 for e in entries)


def test_corrupt_queue_file_fails_closed_and_is_never_overwritten(tmp_path: Path) -> None:
    qf = tmp_path / "q.json"
    qf.write_text("{not json", encoding="utf-8")

    summary, post, _ = _run([_pr(1, mergeable="CONFLICTING")], queue_file=qf, apply=True)

    assert summary["exit_code"] == 1
    assert qf.read_text(encoding="utf-8") == "{not json", "corrupt queue must not be overwritten"
    assert post.calls == [], "no comments after a queue failure"


# ---------------------------------------------------------------------------
# Dry-run vs apply: comments
# ---------------------------------------------------------------------------


def test_dry_run_posts_no_comments_and_exits_zero(tmp_path: Path) -> None:
    summary, post, _ = _run(
        [_pr(1, mergeable="CONFLICTING"), _pr(2, checks=[_failing_check()])],
        queue_file=tmp_path / "q.json",
        apply=False,
    )
    assert post.calls == []
    assert summary["comments_posted"] == []
    assert summary["exit_code"] == 0
    assert summary["mode"] == "dry-run"


def test_apply_posts_one_marked_comment_per_queued_pr(tmp_path: Path) -> None:
    summary, post, _ = _run(
        [_pr(1, mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
        apply=True,
    )
    assert summary["comments_posted"] == [1]
    assert len(post.calls) == 1
    pr, body = post.calls[0]
    assert pr == 1
    assert janitor.COMMENT_MARKER in body
    assert "never closes PRs" in body


def test_existing_marker_comment_skips_posting_idempotent(tmp_path: Path) -> None:
    summary, post, _ = _run(
        [_pr(1, mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
        apply=True,
        existing_comments={1: [f"earlier note\n{janitor.COMMENT_MARKER}\nclassified"]},
    )
    assert post.calls == []
    assert summary["comments_skipped_existing"] == [1]
    assert summary["comments_posted"] == []
    assert summary["exit_code"] == 0


def test_unrelated_existing_comments_do_not_block_posting(tmp_path: Path) -> None:
    summary, post, _ = _run(
        [_pr(1, mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
        apply=True,
        existing_comments={1: ["LGTM", "please rebase"]},
    )
    assert summary["comments_posted"] == [1]
    assert len(post.calls) == 1


def test_max_comments_caps_posting(tmp_path: Path) -> None:
    prs = [_pr(n, mergeable="CONFLICTING") for n in range(1, 6)]
    summary, post, _ = _run(
        prs,
        queue_file=tmp_path / "q.json",
        apply=True,
        max_comments=2,
    )
    assert [pr for pr, _ in post.calls] == [1, 2]
    assert summary["comments_posted"] == [1, 2]
    # The queue still records ALL classified PRs, beyond the comment cap.
    assert summary["queued"] == [1, 2, 3, 4, 5]


def test_failed_comment_fails_closed_exit_one(tmp_path: Path) -> None:
    post = CommentRecorder(results={1: (False, "boom")})
    summary, _, _ = _run(
        [_pr(1, mergeable="CONFLICTING"), _pr(2, mergeable="CONFLICTING")],
        queue_file=tmp_path / "q.json",
        apply=True,
        post=post,
    )
    assert summary["failed"] == [1]
    assert summary["comments_posted"] == [2]
    assert summary["exit_code"] == 1


def test_breaker_trips_on_three_identical_comment_errors(tmp_path: Path) -> None:
    err = (False, "gh: HTTP 401 bad credentials")
    post = CommentRecorder(results={1: err, 2: err, 3: err, 4: err})
    summary, recorder, _ = _run(
        [_pr(n, mergeable="CONFLICTING") for n in (1, 2, 3, 4)],
        queue_file=tmp_path / "q.json",
        apply=True,
        post=post,
    )
    assert [pr for pr, _ in recorder.calls] == [1, 2, 3], "PR 4 untouched after the breaker"
    assert summary["breaker_tripped"] is True
    assert summary["exit_code"] == 2


def test_distinct_comment_errors_do_not_trip_breaker(tmp_path: Path) -> None:
    post = CommentRecorder(
        results={1: (False, "error A"), 2: (False, "error B"), 3: (False, "error A")}
    )
    summary, recorder, _ = _run(
        [_pr(n, mergeable="CONFLICTING") for n in (1, 2, 3)],
        queue_file=tmp_path / "q.json",
        apply=True,
        post=post,
    )
    assert len(recorder.calls) == 3
    assert summary["breaker_tripped"] is False
    assert summary["exit_code"] == 1


def test_comment_fetch_failure_fails_closed_without_posting(tmp_path: Path) -> None:
    post = CommentRecorder()

    def failing_fetch(pr: int) -> list[str]:
        raise RuntimeError("gh pr view failed")

    summary = janitor.run_janitor(
        list_prs=lambda: [_pr(1, mergeable="CONFLICTING")],
        fetch_comments=failing_fetch,
        post_comment=post,
        apply=True,
        queue_file=str(tmp_path / "q.json"),
        now=NOW,
        log=lambda line: None,
    )
    assert post.calls == [], "never post when idempotency cannot be verified"
    assert summary["failed"] == [1]
    assert summary["exit_code"] == 1


# ---------------------------------------------------------------------------
# Non-destructive guarantee: no close / delete / push, ever
# ---------------------------------------------------------------------------


def test_janitor_never_emits_close_delete_or_push(monkeypatch: Any, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))

        class _Result:
            returncode = 0
            stdout = "{}" if command[:3] == ["gh", "pr", "view"] else ""
            stderr = ""

        return _Result()

    adversarial_mix = [
        _pr(1, mergeable="CONFLICTING"),
        _pr(2, checks=[_failing_check()]),
        _pr(3, checks=[{"name": "ci", "state": "SUCCESS", "conclusion": "SUCCESS"}]),
        _pr(4, updated=RECENT),
        _pr(5, head="feature/other"),
    ]
    monkeypatch.setattr(janitor, "default_list_prs", lambda repo: adversarial_mix)
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    exit_code = janitor.main(
        ["--repo", "synaptent/aragora", "--apply", "--queue-file", str(tmp_path / "q.json")]
    )

    assert exit_code == 0
    forbidden = {"close", "delete", "push", "--delete-branch", "merge"}
    for command in commands:
        assert not (set(command) & forbidden), f"destructive command emitted: {command}"
    # Only read (view) and comment commands are permitted.
    for command in commands:
        assert command[:2] == ["gh", "pr"]
        assert command[2] in {"view", "comment"}, command


def test_plan_actions_limited_to_safe_vocabulary(tmp_path: Path) -> None:
    lines: list[str] = []
    janitor.run_janitor(
        list_prs=lambda: [
            _pr(1, mergeable="CONFLICTING"),
            _pr(2, checks=[_failing_check()]),
            _pr(3, checks=[{"name": "ci", "state": "SUCCESS", "conclusion": "SUCCESS"}]),
        ],
        fetch_comments=lambda pr: [],
        post_comment=CommentRecorder(),
        apply=True,
        queue_file=str(tmp_path / "q.json"),
        now=NOW,
        log=lines.append,
    )
    allowed = {
        "classify",
        "recommend_promote",
        "queue_written",
        "queue_error",
        "comment",
        "comment_skipped",
        "comment_failed",
        "breaker_tripped",
        "summary",
    }
    actions = {json.loads(line).get("action") for line in lines}
    assert actions <= allowed, actions


def test_main_dry_run_with_mocked_listing_only_reads(monkeypatch: Any, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))

        class _Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        return _Result()

    monkeypatch.setattr(janitor, "default_list_prs", lambda repo: [_pr(1, mergeable="CONFLICTING")])
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    exit_code = janitor.main(
        ["--repo", "synaptent/aragora", "--queue-file", str(tmp_path / "q.json")]
    )

    assert exit_code == 0
    assert all(c[:3] == ["gh", "pr", "view"] for c in commands), commands


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
