"""Selection logic for scripts/retrigger_cancelled_pr_runs.py (M1 guardian).

Pins the five guard conditions from PR_RUN_CANCELLATION_DIAGNOSIS.md M1: exact
current head, no newer run, non-draft open PR, TTL, and the run_attempt==1
once-per-run marker that bounds rerun loops.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "retrigger_cancelled_pr_runs.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("retrigger_cancelled_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _run(
    run_id: int = 1,
    *,
    workflow_id: int = 77,
    name: str = "Portability Lint",
    branch: str = "feat/x",
    sha: str = "a" * 40,
    conclusion: str = "cancelled",
    event: str = "pull_request",
    attempt: int = 1,
    age_hours: float = 1.0,
) -> dict:
    return {
        "id": run_id,
        "workflow_id": workflow_id,
        "name": name,
        "head_branch": branch,
        "head_sha": sha,
        "conclusion": conclusion,
        "event": event,
        "run_attempt": attempt,
        "created_at": (NOW - timedelta(hours=age_hours)).isoformat().replace("+00:00", "Z"),
    }


def _heads(branch: str = "feat/x", sha: str = "a" * 40) -> dict[str, str]:
    return {branch: sha}


def test_cancelled_current_head_run_is_selected(mod) -> None:
    reruns = mod.compute_reruns([_run()], active_heads=_heads(), now=NOW, ttl_hours=6.0)
    assert [r["run_id"] for r in reruns] == [1]
    assert reruns[0]["workflow"] == "Portability Lint"


def test_superseded_head_is_skipped(mod) -> None:
    reruns = mod.compute_reruns(
        [_run(sha="b" * 40)], active_heads=_heads(sha="a" * 40), now=NOW, ttl_hours=6.0
    )
    assert reruns == []


def test_draft_or_closed_pr_branch_is_skipped(mod) -> None:
    # compute_active_head_map excludes drafts, so an absent branch == draft/closed.
    reruns = mod.compute_reruns(
        [_run(branch="gone")], active_heads=_heads(), now=NOW, ttl_hours=6.0
    )
    assert reruns == []


def test_newer_run_of_same_workflow_supersedes(mod) -> None:
    cancelled = _run(run_id=1, age_hours=2.0)
    newer = _run(run_id=2, conclusion="success", age_hours=0.5)
    reruns = mod.compute_reruns([cancelled, newer], active_heads=_heads(), now=NOW, ttl_hours=6.0)
    assert reruns == []


def test_second_attempt_is_never_rerun_again(mod) -> None:
    # run_attempt > 1 means a rerun already happened: the stateless once-marker.
    reruns = mod.compute_reruns([_run(attempt=2)], active_heads=_heads(), now=NOW, ttl_hours=6.0)
    assert reruns == []


def test_old_cancellation_outside_ttl_is_skipped(mod) -> None:
    reruns = mod.compute_reruns(
        [_run(age_hours=7.0)], active_heads=_heads(), now=NOW, ttl_hours=6.0
    )
    assert reruns == []


def test_non_pr_events_and_non_cancelled_conclusions_are_skipped(mod) -> None:
    runs = [
        _run(run_id=1, event="push"),
        _run(run_id=2, conclusion="failure"),
        _run(run_id=3, conclusion="success"),
    ]
    assert mod.compute_reruns(runs, active_heads=_heads(), now=NOW, ttl_hours=6.0) == []


def test_active_head_map_excludes_drafts(mod) -> None:
    pulls = [
        {"draft": False, "head": {"ref": "feat/x", "sha": "a" * 40}},
        {"draft": True, "head": {"ref": "feat/draft", "sha": "b" * 40}},
    ]
    assert mod.compute_active_head_map(pulls) == {"feat/x": "a" * 40}
