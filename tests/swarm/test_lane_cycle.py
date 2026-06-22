"""Tests for the conductor -> supervisor lane launch cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aragora.swarm import lane_conductor as lc
from aragora.swarm import lane_cycle
from aragora.swarm import lane_supervisor as ls


def _cand(number: int, branch: str = "") -> dict[str, Any]:
    return {"number": number, "branch": branch or f"codex/pr{number}"}


def _seq_session(pr: int) -> str:
    return f"sess-{pr}"


def _fixed_now() -> str:
    return "2026-06-15T00:00:00Z"


def _write_pending(root: Path, work_order_id: str) -> None:
    pending = root / ls.DISPATCH_ROOT / ls.PENDING
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"{work_order_id}.json").write_text(
        json.dumps({"work_order_id": work_order_id, "branch": "codex/old"}),
        encoding="utf-8",
    )


def _pending_names(root: Path) -> set[str]:
    pending = root / ls.DISPATCH_ROOT / ls.PENDING
    return {p.name for p in pending.glob("*.json")} if pending.is_dir() else set()


def test_cycle_dry_run_previews_without_claim_dispatch_or_launch(tmp_path: Path) -> None:
    claimed: list[str] = []
    dispatched: list[str] = []
    launched: list[str] = []

    result = lane_cycle.run_cycle(
        repo="synaptent/aragora",
        root=tmp_path,
        fetch_candidates=lambda repo: [_cand(8428, "claude/swarm-lane-orchestrator")],
        fetch_live_claims=lambda repo, cands: {},
        execute=False,
        claim_fn=lambda wo: claimed.append(wo.work_order_id) or True,
        dispatch_fn=lambda wo: dispatched.append(wo.work_order_id) or "unused",
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        session_id_for=_seq_session,
        now=_fixed_now,
    )

    assert [wo.pr for wo in result.conductor.work_orders] == [8428]
    assert result.supervisor.launched == ["lane-8428-sess-8428"]
    assert "dry-run" in result.reason
    assert claimed == []
    assert dispatched == []
    assert launched == []
    assert _pending_names(tmp_path) == set()


def test_cycle_execute_dispatches_then_launches_only_new_work_orders(tmp_path: Path) -> None:
    _write_pending(tmp_path, "lane-old")
    launched: list[str] = []

    result = lane_cycle.run_cycle(
        repo="synaptent/aragora",
        root=tmp_path,
        fetch_candidates=lambda repo: [_cand(8428, "claude/swarm-lane-orchestrator")],
        fetch_live_claims=lambda repo, cands: {},
        execute=True,
        claim_fn=lambda wo: True,
        dispatch_fn=lambda wo: lc.default_dispatch(wo, repo_root=tmp_path),
        launch_fn=lambda wo: launched.append(wo["work_order_id"]),
        session_id_for=_seq_session,
        now=_fixed_now,
    )

    assert result.conductor.executed is True
    assert result.conductor.dispatched
    assert launched == ["lane-8428-sess-8428"]
    assert result.supervisor.launched == ["lane-8428-sess-8428"]
    assert _pending_names(tmp_path) == {"lane-old.json"}
