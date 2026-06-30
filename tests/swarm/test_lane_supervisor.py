"""Tests for ``aragora.swarm.lane_supervisor`` -- the work-order drainer.

The file-state machine (pending -> in_progress -> done/failed) and its atomic
claim are exercised against a tmp dispatch root with an injected fake launcher,
so no worker is ever spawned and no worktree provisioned.
"""

from __future__ import annotations

import json
import errno
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm import lane_supervisor as ls


def _write_pending(root: Path, work_order_id: str, **extra: Any) -> Path:
    pending = root / ls.DISPATCH_ROOT / ls.PENDING
    pending.mkdir(parents=True, exist_ok=True)
    order = {"work_order_id": work_order_id, "pr": extra.get("pr", 0), **extra}
    path = pending / f"{work_order_id}.json"
    path.write_text(json.dumps(order), encoding="utf-8")
    return path


def _names(root: Path, state: str) -> set[str]:
    d = root / ls.DISPATCH_ROOT / state
    return {p.name for p in d.glob("*.json")} if d.is_dir() else set()


# ---------------------------------------------------------------------------
# load_pending
# ---------------------------------------------------------------------------


def test_load_pending_parses_valid_skips_invalid(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    # Invalid: missing work_order_id.
    bad = tmp_path / ls.DISPATCH_ROOT / ls.PENDING / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    empty = tmp_path / ls.DISPATCH_ROOT / ls.PENDING / "empty.json"
    empty.write_text(json.dumps({"pr": 5}), encoding="utf-8")

    orders = ls.load_pending(tmp_path)
    assert [o["work_order_id"] for _p, o in orders] == ["lane-1-a"]


def test_load_pending_uses_created_at_before_filename_order(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-10-later", pr=10, created_at="2026-06-15T00:00:02Z")
    _write_pending(tmp_path, "lane-2-earlier", pr=2, created_at="2026-06-15T00:00:01Z")

    orders = ls.load_pending(tmp_path)

    assert [o["work_order_id"] for _p, o in orders] == ["lane-2-earlier", "lane-10-later"]


def test_load_pending_empty_when_no_dir(tmp_path: Path) -> None:
    assert ls.load_pending(tmp_path) == []


# ---------------------------------------------------------------------------
# claim_order: the double-spawn guard
# ---------------------------------------------------------------------------


def test_claim_moves_pending_to_in_progress(tmp_path: Path) -> None:
    path = _write_pending(tmp_path, "lane-7-x")
    claimed = ls.claim_order(path, tmp_path)
    assert claimed is not None
    assert claimed.parent == tmp_path / ls.DISPATCH_ROOT / ls.IN_PROGRESS
    assert not path.exists()  # source moved


def test_second_claim_returns_none(tmp_path: Path) -> None:
    path = _write_pending(tmp_path, "lane-7-x")
    first = ls.claim_order(path, tmp_path)
    assert first is not None
    # The file is gone from pending; a racing drainer's claim must fail cleanly.
    second = ls.claim_order(path, tmp_path)
    assert second is None


def test_claim_does_not_overwrite_existing_in_progress(tmp_path: Path) -> None:
    path = _write_pending(tmp_path, "lane-7-x", pr=7)
    in_progress = tmp_path / ls.DISPATCH_ROOT / ls.IN_PROGRESS
    in_progress.mkdir(parents=True, exist_ok=True)
    existing = in_progress / path.name
    existing.write_text(json.dumps({"work_order_id": "lane-7-x", "pr": 999}), encoding="utf-8")

    claimed = ls.claim_order(path, tmp_path)

    assert claimed is None
    assert path.exists()
    assert json.loads(existing.read_text(encoding="utf-8"))["pr"] == 999


def test_claim_unexpected_oserror_is_loud(tmp_path: Path, monkeypatch: Any) -> None:
    path = _write_pending(tmp_path, "lane-7-x", pr=7)

    def fail_link(src: Path, dest: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(ls.os, "link", fail_link)

    try:
        ls.claim_order(path, tmp_path)
    except ls.ClaimOrderError as exc:
        assert "different filesystems" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected ClaimOrderError")
    assert path.exists()


# ---------------------------------------------------------------------------
# drain_once
# ---------------------------------------------------------------------------


def test_drain_launches_and_moves_to_done(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    _write_pending(tmp_path, "lane-2-b")
    launched: list[str] = []

    result = ls.drain_once(
        root=tmp_path,
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        max_launches=5,
    )
    assert sorted(launched) == ["lane-1-a", "lane-2-b"]
    assert sorted(result.launched) == ["lane-1-a", "lane-2-b"]
    assert _names(tmp_path, ls.DONE) == {"lane-1-a.json", "lane-2-b.json"}
    assert _names(tmp_path, ls.PENDING) == set()
    assert _names(tmp_path, ls.IN_PROGRESS) == set()


def test_drain_respects_max_launches(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    _write_pending(tmp_path, "lane-2-b")
    _write_pending(tmp_path, "lane-3-c")
    launched: list[str] = []

    result = ls.drain_once(
        root=tmp_path,
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        max_launches=2,
    )
    assert len(result.launched) == 2
    assert len(result.deferred) == 1
    # The deferred order stays pending for the next pass.
    assert len(_names(tmp_path, ls.PENDING)) == 1


def test_drain_can_filter_to_newly_dispatched_work_orders(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-old")
    _write_pending(tmp_path, "lane-new")
    launched: list[str] = []

    result = ls.drain_once(
        root=tmp_path,
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        max_launches=5,
        work_order_ids={"lane-new"},
    )

    assert launched == ["lane-new"]
    assert result.launched == ["lane-new"]
    assert _names(tmp_path, ls.DONE) == {"lane-new.json"}
    # Existing pending work stays pending; a combined conductor/supervisor pass
    # must not unexpectedly launch stale backlog.
    assert _names(tmp_path, ls.PENDING) == {"lane-old.json"}


def test_drain_records_failure_and_continues(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    _write_pending(tmp_path, "lane-2-boom")
    _write_pending(tmp_path, "lane-3-c")

    def launch(wo: dict[str, Any]) -> None:
        if "boom" in wo["work_order_id"]:
            raise RuntimeError("spawn exploded")

    result = ls.drain_once(root=tmp_path, launch_fn=launch, max_launches=5)
    assert sorted(result.launched) == ["lane-1-a", "lane-3-c"]
    assert [f["work_order_id"] for f in result.failed] == ["lane-2-boom"]
    assert "spawn exploded" in result.failed[0]["error"]
    # Failed order is preserved in failed/ with the error recorded.
    assert _names(tmp_path, ls.FAILED) == {"lane-2-boom.json"}
    failed_doc = json.loads(
        (tmp_path / ls.DISPATCH_ROOT / ls.FAILED / "lane-2-boom.json").read_text()
    )
    assert "spawn exploded" in failed_doc["_launch_error"]
    assert _names(tmp_path, ls.DONE) == {"lane-1-a.json", "lane-3-c.json"}


def test_drain_failures_consume_max_launches_cap(tmp_path: Path) -> None:
    # Regression: max_launches must bound launch *attempts*, not just successes.
    # Otherwise a queue of failing orders never trips the cap and a single pass
    # would claim+fail the entire pending set, ignoring --max-launches.
    for i in range(5):
        _write_pending(tmp_path, f"lane-{i}-boom")

    def always_fail(wo: dict[str, Any]) -> None:
        raise RuntimeError("spawn exploded")

    result = ls.drain_once(root=tmp_path, launch_fn=always_fail, max_launches=2)

    assert len(result.failed) == 2  # only 2 attempts, not all 5
    assert len(result.deferred) == 3
    # The 3 deferred orders are left pending for the next pass (not failed/).
    assert len(_names(tmp_path, ls.PENDING)) == 3
    assert len(_names(tmp_path, ls.FAILED)) == 2


def test_drain_records_claim_failure_without_launching(tmp_path: Path, monkeypatch: Any) -> None:
    _write_pending(tmp_path, "lane-1-cross-fs", pr=1)
    launched: list[str] = []

    def fail_link(src: Path, dest: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(ls.os, "link", fail_link)

    result = ls.drain_once(
        root=tmp_path,
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        max_launches=5,
    )

    assert launched == []
    assert result.failed == [
        {
            "work_order_id": "lane-1-cross-fs",
            "error": "cannot atomically claim lane-1-cross-fs.json: pending and in_progress are on different filesystems",
        }
    ]
    assert _names(tmp_path, ls.PENDING) == {"lane-1-cross-fs.json"}


def test_claim_rolls_back_link_when_pending_unlink_fails(tmp_path: Path, monkeypatch: Any) -> None:
    # Regression: if os.link succeeds but the pending unlink fails, the link must
    # be rolled back -- otherwise the order wedges (a duplicate sits in
    # in_progress/ while the orphaned pending makes every future claim a no-op).
    src = _write_pending(tmp_path, "lane-stuck")
    real_unlink = Path.unlink

    def flaky_unlink(self: Path, *a: Any, **k: Any) -> Any:
        if self.name == "lane-stuck.json" and self.parent.name == ls.PENDING:
            raise OSError(errno.EACCES, "pending is read-only")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(ls.ClaimOrderError, match="could not remove pending source"):
        ls.claim_order(src, tmp_path)

    # Rolled back cleanly: no in_progress duplicate, pending source preserved.
    assert _names(tmp_path, ls.IN_PROGRESS) == set()
    assert _names(tmp_path, ls.PENDING) == {"lane-stuck.json"}


def test_drain_is_idempotent_after_done(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    launched: list[str] = []
    fn = lambda wo: launched.append(wo["work_order_id"])  # noqa: E731
    ls.drain_once(root=tmp_path, launch_fn=fn, max_launches=5)
    # A second drain finds nothing pending -> no re-launch.
    second = ls.drain_once(root=tmp_path, launch_fn=fn, max_launches=5)
    assert launched == ["lane-1-a"]
    assert second.launched == []


# ---------------------------------------------------------------------------
# plan_drain (dry-run preview)
# ---------------------------------------------------------------------------


def test_plan_drain_previews_without_moving(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-1-a")
    _write_pending(tmp_path, "lane-2-b")
    _write_pending(tmp_path, "lane-3-c")
    result = ls.plan_drain(tmp_path, max_launches=2)
    assert len(result.launched) == 2
    assert len(result.deferred) == 1
    assert "dry-run" in result.reason
    # Nothing moved: all still pending.
    assert len(_names(tmp_path, ls.PENDING)) == 3
    assert _names(tmp_path, ls.IN_PROGRESS) == set()
