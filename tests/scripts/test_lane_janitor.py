"""Tests for ``scripts/lane_janitor.py`` — dead-lane detection and bounded cleanup.

Failure class A (2026-06-10/11): coordinator-spawned lanes died at setup,
leaving in_progress ledgers and empty branches on origin that nobody noticed.
All git interaction goes through an injected FakeGit — no network in tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


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


janitor = _load_module("lane_janitor.py")

NOW = janitor.parse_iso("2026-06-11T12:00:00Z")


class FakeGit:
    """Injected git boundary: remote heads, ahead counts, tip dates, deletion."""

    def __init__(
        self,
        heads: dict[str, str],
        ahead: dict[str, int | None],
        dates: dict[str, str] | None = None,
    ) -> None:
        self.heads = heads
        self.ahead = ahead
        self.dates = {sha: janitor.parse_iso(d) for sha, d in (dates or {}).items()}
        self.deleted: list[str] = []

    def remote_heads(self) -> dict[str, str]:
        return dict(self.heads)

    def ahead_count(self, sha: str) -> int | None:
        return self.ahead.get(sha)

    def commit_date(self, sha: str) -> Any:
        return self.dates.get(sha)

    def delete_remote_branch(self, branch: str) -> None:
        self.deleted.append(branch)
        self.heads.pop(branch, None)


def _write_ledger(
    lanes_dir: Path,
    lane: str,
    *,
    branch: str,
    launched_at: str,
    status: str = "in_progress",
    brief: str = "do the thing",
    **extra: Any,
) -> Path:
    lanes_dir.mkdir(parents=True, exist_ok=True)
    path = lanes_dir / f"{lane}.json"
    entry = {
        "lane": lane,
        "agent_id": "a0000000000000000",
        "branch": branch,
        "brief": brief,
        "launched_at": launched_at,
        "status": status,
    }
    entry.update(extra)
    path.write_text(json.dumps(entry))
    return path


def _run(tmp_path: Path, git: FakeGit, *, apply: bool = False, **kw: Any) -> dict[str, Any]:
    return janitor.build_plan(
        str(tmp_path / "run-*" / "lanes"),
        git=git,
        now=NOW,
        lane_max_age_hours=kw.pop("lane_max_age_hours", 3.0),
        branch_ttl_hours=kw.pop("branch_ttl_hours", 24.0),
        apply=apply,
    )


def _lanes_dir(tmp_path: Path, run: str = "run-20260610") -> Path:
    return tmp_path / run / "lanes"


# ---------------------------------------------------------------------------
# dead-lane detection + marking
# ---------------------------------------------------------------------------


def test_dry_run_plans_dead_lane_without_writing(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    path = _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-11T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-11T02:00:00Z"})
    plan = _run(tmp_path, git)
    assert [d["lane"] for d in plan["mark_dead"]] == ["c06"]
    assert plan["applied"] is False
    # Dry run: ledger untouched, no queue file, no deletions.
    assert json.loads(path.read_text())["status"] == "in_progress"
    assert not (lanes / "RELAUNCH_QUEUE.md").exists()
    assert git.deleted == []


def test_apply_marks_ledger_dead_with_detected_at(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    path = _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-11T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0})
    plan = _run(tmp_path, git, apply=True)
    assert plan["applied"] is True
    entry = json.loads(path.read_text())
    assert entry["status"] == "dead"
    assert entry["detected_at"] == "2026-06-11T12:00:00Z"
    # Original fields preserved.
    assert entry["brief"] == "do the thing"
    assert entry["agent_id"] == "a0000000000000000"


def test_fresh_lane_not_marked_dead(tmp_path: Path) -> None:
    _write_ledger(
        _lanes_dir(tmp_path),
        "q2",
        branch="elves/run-x-q2",
        launched_at="2026-06-11T11:00:00Z",  # 1h old
    )
    git = FakeGit({"elves/run-x-q2": "aaa"}, {"aaa": 0})
    plan = _run(tmp_path, git)
    assert plan["mark_dead"] == []


def test_stale_lane_with_commits_not_marked_dead(tmp_path: Path) -> None:
    _write_ledger(
        _lanes_dir(tmp_path),
        "q2",
        branch="elves/run-x-q2",
        launched_at="2026-06-11T02:00:00Z",
    )
    git = FakeGit({"elves/run-x-q2": "aaa"}, {"aaa": 3})
    plan = _run(tmp_path, git)
    assert plan["mark_dead"] == []


def test_stale_lane_branch_absent_marked_dead(tmp_path: Path) -> None:
    _write_ledger(
        _lanes_dir(tmp_path),
        "c07",
        branch="elves/run-x-c07",
        launched_at="2026-06-11T02:00:00Z",
    )
    git = FakeGit({}, {})
    plan = _run(tmp_path, git)
    assert [d["lane"] for d in plan["mark_dead"]] == ["c07"]


def test_unresolvable_ahead_count_never_dead(tmp_path: Path) -> None:
    _write_ledger(
        _lanes_dir(tmp_path),
        "q2",
        branch="elves/run-x-q2",
        launched_at="2026-06-11T02:00:00Z",
    )
    git = FakeGit({"elves/run-x-q2": "aaa"}, {"aaa": None})
    plan = _run(tmp_path, git)
    assert plan["mark_dead"] == []


# ---------------------------------------------------------------------------
# relaunch queue
# ---------------------------------------------------------------------------


def test_apply_writes_relaunch_queue_with_briefs(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    _write_ledger(
        lanes,
        "c06",
        branch="elves/run-x-c06",
        launched_at="2026-06-11T02:00:00Z",
        brief="fix #8101 three sub-bugs; Tier2 merge-on-green",
    )
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0})
    _run(tmp_path, git, apply=True)
    queue = (lanes / "RELAUNCH_QUEUE.md").read_text()
    assert "c06" in queue
    assert "fix #8101 three sub-bugs; Tier2 merge-on-green" in queue
    assert "elves/run-x-c06" in queue


def test_relaunch_queue_includes_previously_dead_lanes(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    _write_ledger(
        lanes,
        "old-dead",
        branch="elves/run-x-old",
        launched_at="2026-06-10T02:00:00Z",
        status="dead",
        detected_at="2026-06-10T08:00:00Z",
        brief="previously detected",
    )
    _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-11T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0})
    _run(tmp_path, git, apply=True)
    queue = (lanes / "RELAUNCH_QUEUE.md").read_text()
    assert "old-dead" in queue
    assert "c06" in queue


def test_no_dead_lanes_no_queue_file(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    _write_ledger(lanes, "q2", branch="elves/run-x-q2", launched_at="2026-06-11T11:30:00Z")
    git = FakeGit({"elves/run-x-q2": "aaa"}, {"aaa": 0})
    _run(tmp_path, git, apply=True)
    assert not (lanes / "RELAUNCH_QUEUE.md").exists()


# ---------------------------------------------------------------------------
# branch deletion — bounded, with the hard "never delete real work" guarantee
# ---------------------------------------------------------------------------


def test_never_deletes_branch_with_unique_commits(tmp_path: Path) -> None:
    """HARD GUARANTEE: a branch with any unique commit is never deleted,
    even when its ledger is dead and the branch is older than the TTL."""
    lanes = _lanes_dir(tmp_path)
    _write_ledger(
        lanes,
        "c06",
        branch="elves/run-x-c06",
        launched_at="2026-06-09T02:00:00Z",
        status="dead",
        detected_at="2026-06-10T02:00:00Z",
    )
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 1}, {"aaa": "2026-06-01T00:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []


def test_never_deletes_branch_with_unresolvable_ahead(tmp_path: Path) -> None:
    git = FakeGit({"elves/run-x-mystery": "aaa"}, {"aaa": None}, {"aaa": "2026-06-01T00:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []


def test_deletes_ledger_dead_zero_ahead_old_branch_on_apply(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    _write_ledger(
        lanes,
        "c06",
        branch="elves/run-x-c06",
        launched_at="2026-06-10T02:00:00Z",
        status="dead",
        detected_at="2026-06-10T08:00:00Z",
    )
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-09T02:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert [d["branch"] for d in plan["delete_branches"]] == ["elves/run-x-c06"]
    assert git.deleted == ["elves/run-x-c06"]


def test_newly_detected_dead_lane_branch_deletable_same_run(tmp_path: Path) -> None:
    """A lane marked dead in this run is immediately eligible (if old + empty)."""
    lanes = _lanes_dir(tmp_path)
    _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-09T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-09T02:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert [d["lane"] for d in plan["mark_dead"]] == ["c06"]
    assert git.deleted == ["elves/run-x-c06"]


def test_dry_run_plans_deletion_without_deleting(tmp_path: Path) -> None:
    git = FakeGit(
        {"elves/run-20260610-c06-dead": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-09T02:00:00Z"}
    )
    plan = _run(tmp_path, git)
    assert [d["branch"] for d in plan["delete_branches"]] == ["elves/run-20260610-c06-dead"]
    assert git.deleted == []


def test_deletes_ledgerless_orphan_pattern_branch(tmp_path: Path) -> None:
    """The real morning-after state: empty branch, no ledger anywhere."""
    git = FakeGit(
        {"aragora/boss-harvest/issue-1-boss-x": "bbb"},
        {"bbb": 0},
        {"bbb": "2026-06-08T00:00:00Z"},
    )
    plan = _run(tmp_path, git, apply=True)
    assert git.deleted == ["aragora/boss-harvest/issue-1-boss-x"]
    assert plan["delete_branches"][0]["reason"] == "ledger-less orphan"


def test_never_deletes_non_pattern_ledgerless_branch(tmp_path: Path) -> None:
    git = FakeGit({"feature/manual-work": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-01T00:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []


def test_never_deletes_branch_younger_than_ttl(tmp_path: Path) -> None:
    git = FakeGit(
        {"elves/run-x-young": "aaa"},
        {"aaa": 0},
        {"aaa": "2026-06-11T02:00:00Z"},  # 10h
    )
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []


def test_never_deletes_branch_of_live_in_progress_lane(tmp_path: Path) -> None:
    """A fresh lane whose branch tip is an old main commit (stale base) must
    be protected by its live ledger even though the branch looks old+empty."""
    lanes = _lanes_dir(tmp_path)
    _write_ledger(
        lanes,
        "q2",
        branch="elves/run-x-q2",
        launched_at="2026-06-11T11:30:00Z",  # 30 min old — alive
    )
    git = FakeGit({"elves/run-x-q2": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-08T00:00:00Z"})
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []
    assert any("live ledger" in s["reason"] for s in plan["skipped"])


def test_unknown_commit_date_never_deleted(tmp_path: Path) -> None:
    git = FakeGit({"elves/run-x-undated": "aaa"}, {"aaa": 0}, {})
    plan = _run(tmp_path, git, apply=True)
    assert plan["delete_branches"] == []
    assert git.deleted == []


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_dry_run_json_exits_zero(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    lanes = _lanes_dir(tmp_path)
    _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-11T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0})
    monkeypatch.setattr(janitor, "GitBoundary", lambda repo: git)
    code = janitor.main(
        [
            "--json",
            "--runs-glob",
            str(tmp_path / "run-*" / "lanes"),
            "--now",
            "2026-06-11T12:00:00Z",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["applied"] is False
    assert [d["lane"] for d in plan["mark_dead"]] == ["c06"]
    assert git.deleted == []


def test_main_apply_executes_plan(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    lanes = _lanes_dir(tmp_path)
    path = _write_ledger(lanes, "c06", branch="elves/run-x-c06", launched_at="2026-06-09T02:00:00Z")
    git = FakeGit({"elves/run-x-c06": "aaa"}, {"aaa": 0}, {"aaa": "2026-06-09T02:00:00Z"})
    monkeypatch.setattr(janitor, "GitBoundary", lambda repo: git)
    code = janitor.main(
        [
            "--apply",
            "--json",
            "--runs-glob",
            str(tmp_path / "run-*" / "lanes"),
            "--now",
            "2026-06-11T12:00:00Z",
        ]
    )
    assert code == 0
    assert json.loads(path.read_text())["status"] == "dead"
    assert (lanes / "RELAUNCH_QUEUE.md").exists()
    assert git.deleted == ["elves/run-x-c06"]


def test_relaunch_queue_md_not_parsed_as_ledger(tmp_path: Path) -> None:
    lanes = _lanes_dir(tmp_path)
    lanes.mkdir(parents=True)
    (lanes / "RELAUNCH_QUEUE.md").write_text("# queue\n")
    git = FakeGit({}, {})
    plan = _run(tmp_path, git)
    assert plan["mark_dead"] == []
    assert plan["errors"] == []
