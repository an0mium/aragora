from __future__ import annotations

from scripts.benchmark_publication_run_followup import (
    _branch_heads_for_runs,
    classify_publication_runs,
)


def test_classify_publication_runs_flags_stale_pending_main_run() -> None:
    runs = [
        {
            "databaseId": 26855039734,
            "status": "pending",
            "event": "workflow_dispatch",
            "headBranch": "main",
            "headSha": "old-main",
            "createdAt": "2026-06-02T23:53:53Z",
            "updatedAt": "2026-06-02T23:53:55Z",
            "url": "https://github.com/synaptent/aragora/actions/runs/26855039734",
        }
    ]

    actions = classify_publication_runs(
        runs,
        branch_heads={"main": "new-main"},
        now="2026-06-03T00:54:00Z",
        stale_after_minutes=30,
    )

    assert actions == [
        {
            "run_id": 26855039734,
            "action": "cancel",
            "reason": "stale-branch-sha",
            "event": "workflow_dispatch",
            "status": "pending",
            "head_branch": "main",
            "head_sha": "old-main",
            "current_branch_sha": "new-main",
            "age_minutes": 60.12,
            "updated_age_minutes": 60.08,
            "url": "https://github.com/synaptent/aragora/actions/runs/26855039734",
        }
    ]


def test_classify_publication_runs_keeps_recent_or_current_runs() -> None:
    runs = [
        {
            "id": 1,
            "status": "pending",
            "event": "workflow_dispatch",
            "head_branch": "main",
            "head_sha": "old-main",
            "created_at": "2026-06-03T00:45:00Z",
            "updated_at": "2026-06-03T00:45:00Z",
            "html_url": "https://example.test/recent",
        },
        {
            "id": 2,
            "status": "queued",
            "event": "schedule",
            "head_branch": "main",
            "head_sha": "new-main",
            "created_at": "2026-06-02T23:00:00Z",
            "updated_at": "2026-06-02T23:00:00Z",
            "html_url": "https://example.test/current",
        },
        {
            "id": 3,
            "status": "in_progress",
            "event": "pull_request",
            "head_branch": "feature",
            "head_sha": "old-feature",
            "created_at": "2026-06-02T23:00:00Z",
            "updated_at": "2026-06-02T23:00:00Z",
            "html_url": "https://example.test/pr",
        },
    ]

    actions = classify_publication_runs(
        runs,
        branch_heads={"main": "new-main", "feature": "new-feature"},
        now="2026-06-03T00:54:00Z",
        stale_after_minutes=30,
    )

    assert actions == []


def test_classify_publication_runs_reports_unknown_branch_after_stale_window() -> None:
    runs = [
        {
            "id": 42,
            "status": "queued",
            "event": "schedule",
            "head_branch": "main",
            "head_sha": "orphaned",
            "created_at": "2026-06-02T20:00:00Z",
            "updated_at": "2026-06-02T20:01:00Z",
        }
    ]

    actions = classify_publication_runs(
        runs,
        branch_heads={},
        now="2026-06-03T00:54:00Z",
        stale_after_minutes=30,
    )

    assert actions[0]["action"] == "report"
    assert actions[0]["reason"] == "unknown-branch-head"
    assert actions[0]["run_id"] == 42


def test_classify_publication_runs_can_explicitly_cancel_unknown_branch() -> None:
    runs = [
        {
            "id": 42,
            "status": "queued",
            "event": "schedule",
            "head_branch": "retired-branch",
            "head_sha": "orphaned",
            "created_at": "2026-06-02T20:00:00Z",
            "updated_at": "2026-06-02T20:01:00Z",
        }
    ]

    actions = classify_publication_runs(
        runs,
        branch_heads={},
        now="2026-06-03T00:54:00Z",
        stale_after_minutes=30,
        allow_unknown_branch_cancel=True,
    )

    assert actions[0]["action"] == "cancel"
    assert actions[0]["reason"] == "unknown-branch-head"
    assert actions[0]["run_id"] == 42


def test_branch_heads_for_runs_url_encodes_branch_names() -> None:
    class FakeClient:
        repo = "synaptent/aragora"

        def __init__(self) -> None:
            self.paths: list[str] = []

        def get(self, path: str) -> dict[str, object]:
            self.paths.append(path)
            return {"commit": {"sha": "head-sha"}}

    client = FakeClient()

    heads = _branch_heads_for_runs(
        client,  # type: ignore[arg-type]
        [{"head_branch": "codex/example/slash"}],
    )

    assert heads == {"codex/example/slash": "head-sha"}
    assert client.paths == ["/repos/synaptent/aragora/branches/codex%2Fexample%2Fslash"]
