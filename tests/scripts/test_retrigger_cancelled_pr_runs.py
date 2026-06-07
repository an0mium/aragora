from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scripts.retrigger_cancelled_pr_runs import (
    compute_retriggerable_runs,
    prune_marker,
)

NOW = datetime(2026, 6, 6, 21, 0, 0, tzinfo=timezone.utc)
RECENT = "2026-06-06T20:59:00Z"  # 1 min before NOW
OLD = "2026-06-06T19:00:00Z"  # 2 h before NOW
PR_EVENTS = {"pull_request", "pull_request_target"}


def make_run(**over: Any) -> dict[str, Any]:
    run = {
        "id": 1,
        "event": "pull_request",
        "conclusion": "cancelled",
        "status": "completed",
        "head_branch": "feat/x",
        "head_sha": "sha-x",
        "run_attempt": 1,
        "run_number": 1,
        "workflow_id": 100,
        "name": "Portability Lint",
        "created_at": RECENT,
    }
    run.update(over)
    return run


def _compute(runs: list[dict[str, Any]], active_heads: dict[str, str], **kw: Any):
    params: dict[str, Any] = {
        "active_heads": active_heads,
        "cancel_events": PR_EVENTS,
        "now": NOW,
        "ttl_minutes": 60,
    }
    params.update(kw)
    return compute_retriggerable_runs(runs, **params)


def test_genuine_cancelled_non_superseded_is_selected() -> None:
    runs = [make_run(id=1, head_branch="feat/a", head_sha="sha-a")]
    eligible, reasons, candidates = _compute(runs, {"feat/a": "sha-a"})
    assert candidates == 1
    assert reasons == {}
    assert [e["run_id"] for e in eligible] == [1]
    assert eligible[0]["rerun_command"] == "gh run rerun 1"


def test_superseded_sha_is_skipped() -> None:
    runs = [make_run(id=2, head_branch="feat/b", head_sha="old-sha")]
    eligible, reasons, candidates = _compute(runs, {"feat/b": "new-sha"})
    assert eligible == []
    assert reasons == {"superseded-sha": 1}
    assert candidates == 1


def test_draft_or_closed_branch_is_skipped() -> None:
    # Draft PRs are excluded from active_heads upstream, so the branch is absent.
    runs = [make_run(id=3, head_branch="feat/c", head_sha="sha-c")]
    eligible, reasons, _ = _compute(runs, {})
    assert eligible == []
    assert reasons == {"draft-or-closed": 1}


def test_ttl_expired_is_skipped() -> None:
    runs = [make_run(id=4, head_branch="feat/d", head_sha="sha-d", created_at=OLD)]
    eligible, reasons, _ = _compute(runs, {"feat/d": "sha-d"})
    assert eligible == []
    assert reasons == {"ttl-expired": 1}


def test_loop_guard_marker_is_honored() -> None:
    runs = [make_run(id=5, head_branch="feat/e", head_sha="sha-e")]
    eligible, reasons, _ = _compute(runs, {"feat/e": "sha-e"}, already_retriggered={5})
    assert eligible == []
    assert reasons == {"already-retriggered": 1}


def test_superseded_by_newer_run_is_skipped() -> None:
    runs = [
        make_run(
            id=6,
            head_branch="feat/f",
            head_sha="sha-f",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=7,
            head_branch="feat/f",
            head_sha="sha-f",
            run_number=2,
            conclusion="success",
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/f": "sha-f"})
    assert eligible == []
    assert reasons == {"superseded-by-newer-run": 1}
    # only the cancelled run is a candidate; the success sibling is not counted
    assert candidates == 1


def test_newer_non_pr_sibling_does_not_supersede() -> None:
    # A newer push/workflow_dispatch run on the same branch+workflow+SHA must not
    # suppress re-running a still-current cancelled PR run.
    runs = [
        make_run(
            id=20,
            head_branch="feat/p",
            head_sha="sha-p",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=21,
            event="push",
            conclusion="success",
            head_branch="feat/p",
            head_sha="sha-p",
            run_number=2,
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/p": "sha-p"})
    assert [e["run_id"] for e in eligible] == [20]
    assert reasons == {}
    assert candidates == 1


def test_newer_different_sha_sibling_does_not_supersede() -> None:
    # A newer run for a *different* head SHA on the same branch+workflow must not
    # suppress the cancelled run that still matches the current PR head.
    runs = [
        make_run(
            id=22,
            head_branch="feat/q",
            head_sha="sha-q",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=23,
            head_branch="feat/q",
            head_sha="other-sha",
            conclusion="success",
            run_number=2,
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/q": "sha-q"})
    assert [e["run_id"] for e in eligible] == [22]
    assert reasons == {}
    assert candidates == 1


def test_max_attempts_guard_is_honored() -> None:
    runs = [make_run(id=8, head_branch="feat/g", head_sha="sha-g", run_attempt=2)]
    eligible, reasons, _ = _compute(runs, {"feat/g": "sha-g"}, max_attempts=2)
    assert eligible == []
    assert reasons == {"max-attempts": 1}


def test_non_pr_and_non_cancelled_runs_are_not_candidates() -> None:
    runs = [
        make_run(id=9, event="push", head_branch="feat/h", head_sha="sha-h"),
        make_run(id=10, conclusion="success", head_branch="feat/h", head_sha="sha-h"),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/h": "sha-h"})
    assert eligible == []
    assert reasons == {}
    assert candidates == 0


def test_prune_marker_drops_old_entries() -> None:
    data = {"1": RECENT, "2": "2026-06-05T10:00:00Z"}
    pruned = prune_marker(data, now=NOW, retention_hours=24)
    assert pruned == {"1": RECENT}
