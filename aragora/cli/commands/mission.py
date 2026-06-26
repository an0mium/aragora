"""Native Mission CLI commands."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from aragora.missions import (
    Feature,
    Handoff,
    MissionOrchestrator,
    MissionState,
    ReconcileMode,
    Status,
    WorkArtifact,
)
from aragora.missions.dispatch import BossLoopDispatch
from aragora.missions.live_gate import LiveBossLoopGate

logger = logging.getLogger(__name__)

_ACTIONS = {"seed", "status", "run", "resume", "reconcile"}


def cmd_mission(args: argparse.Namespace) -> int:
    """Execute the ``mission`` command."""
    action, goal_words = _normalize_action(args)
    try:
        if action == "seed":
            return _cmd_seed(args, goal_words)
        if action == "status":
            return _cmd_status(args)
        if action == "run":
            return _cmd_run(args, resume=False)
        if action == "resume":
            return _cmd_run(args, resume=True)
        if action == "reconcile":
            return _cmd_reconcile(args)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("mission command failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Error: unknown mission action {action!r}", file=sys.stderr)
    return 2


def _normalize_action(args: argparse.Namespace) -> tuple[str, list[str]]:
    raw = getattr(args, "mission_action", None)
    goal_words = list(getattr(args, "goal", []) or [])
    if raw in _ACTIONS:
        return raw, goal_words
    if raw:
        return "seed", [raw, *goal_words]
    return "status", goal_words


def _cmd_seed(args: argparse.Namespace, goal_words: list[str]) -> int:
    goal = " ".join(goal_words).strip()
    if not goal:
        raise ValueError("mission seed requires a goal")
    if args.budget is not None and args.budget < 0:
        raise ValueError("budget_usd must be non-negative")
    mission_id = f"mission-{uuid.uuid4().hex[:12]}"
    state_path = _state_path(args, mission_id=mission_id)
    tracks = _tracks(args)
    metadata: dict[str, Any] = {
        "budget_usd": args.budget,
        "max_hours": args.max_hours,
        "relay": args.relay,
        "auto_settle_max_tier": args.auto_settle_max_tier,
        "tracks": tracks,
        "autonomy": args.autonomy,
    }
    state = MissionState(
        mission_id=mission_id,
        goal=goal,
        milestones=["mission"],
        features=[
            Feature(
                id="mission-intake",
                description=goal,
                milestone="mission",
                metadata=metadata,
            )
        ],
    )
    state.save(state_path)
    print(f"Seeded mission {mission_id}: {goal}")
    print(f"State: {state_path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    state = MissionState.load(_state_path(args))
    done, total = state.progress()
    blocked = sum(1 for feature in state.features if feature.status == Status.BLOCKED)
    in_progress = sum(1 for feature in state.features if feature.status == Status.IN_PROGRESS)
    print(f"Mission {state.mission_id}: {state.goal}")
    print(f"Progress: {done}/{total} completed, {blocked} blocked, {in_progress} in progress")
    for feature in state.features:
        print(f"  - {feature.id}: {feature.status} ({feature.milestone}) {feature.description}")
    return 0


def _cmd_run(args: argparse.Namespace, *, resume: bool) -> int:
    state_path = _state_path(args)
    _assert_not_paused(state_path)
    if resume:
        state = MissionState.load(state_path)
        reclaimed = state.reclaim_in_progress()
        if reclaimed:
            state.save(state_path)
            print(f"Reclaimed in-progress features: {', '.join(reclaimed)}")
    dispatch = _dispatch_for(args)
    done, total = MissionOrchestrator(state_path).run(dispatch, max_ticks=args.max_ticks)
    print(f"Mission run: {done}/{total} completed")
    return 0


def _assert_not_paused(state_path: Path) -> None:
    pause_files = [
        state_path.with_name("PAUSED"),
        Path(".aragora") / "missions" / "PAUSED",
    ]
    for pause_file in pause_files:
        if pause_file.exists():
            reason = pause_file.read_text(encoding="utf-8").strip()
            detail = f": {reason}" if reason else ""
            raise RuntimeError(f"mission is paused by {pause_file}{detail}")


def _cmd_reconcile(args: argparse.Namespace) -> int:
    mode = ReconcileMode(args.autonomy)
    artifacts = _load_artifacts(args)
    report = mode.run(artifacts)
    if args.json:
        print(report.to_json())
    else:
        print(f"Reconcile mode: {report.mode}")
        print(f"Artifacts: {len(report.items)}")
        print(f"Authorized cleanup: {len(report.authorized_cleanup)}")
        print(f"Authorized auto-drain: {len(report.authorized_auto_drain)}")
        print(f"Parked: {len(report.parked)}")
        for item in report.parked:
            print(f"  - {item.artifact_id}: {item.category.value} ({item.reason})")
    return 0


def _dispatch_for(args: argparse.Namespace):
    if args.autonomy == "auto-drain":
        gate = LiveBossLoopGate(repo_root=Path.cwd())
        return BossLoopDispatch(gate, operator_tier=args.operator_tier)

    def report_dispatch(feature: Feature) -> Handoff:
        return Handoff(
            success=True,
            discovered=[f"{args.autonomy} mission dispatch recorded without live head mutation"],
        )

    return report_dispatch


def _state_path(args: argparse.Namespace, *, mission_id: str | None = None) -> Path:
    if getattr(args, "state", None):
        return Path(args.state)
    if mission_id is None:
        raise ValueError("--state is required for this mission action")
    return Path(".aragora") / "missions" / mission_id / "state.json"


def _tracks(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "tracks", None)
    if not raw:
        return []
    return [track.strip() for track in raw.split(",") if track.strip()]


def _load_artifacts(args: argparse.Namespace) -> list[WorkArtifact]:
    if getattr(args, "artifact_fixture", None):
        payload = json.loads(Path(args.artifact_fixture).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--artifact-fixture must contain a JSON list")
        return [WorkArtifact.from_dict(item) for item in payload]
    return _load_live_inventory_artifacts(limit=args.limit)


def _load_live_inventory_artifacts(*, limit: int) -> list[WorkArtifact]:
    cmd = [
        sys.executable,
        "scripts/codex_worktree_value_inventory.py",
        "--repo",
        ".",
        "--limit",
        str(limit),
        "--size-mode",
        "none",
        "--skip-gh",
        "--dry-run",
        "--json",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or "codex_worktree_value_inventory.py failed").strip()
        )
    payload = json.loads(proc.stdout)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return [
        _artifact_from_inventory(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _artifact_from_inventory(candidate: dict[str, Any]) -> WorkArtifact:
    raw_git = candidate.get("git")
    raw_links = candidate.get("links")
    raw_safety = candidate.get("cleanup_safety")
    raw_proof = candidate.get("proof")
    git: dict[str, Any] = raw_git if isinstance(raw_git, dict) else {}
    links: dict[str, Any] = raw_links if isinstance(raw_links, dict) else {}
    safety: dict[str, Any] = raw_safety if isinstance(raw_safety, dict) else {}
    classification = str(candidate.get("classification") or "")
    open_prs = links.get("open_prs") if isinstance(links.get("open_prs"), list) else []
    proof: list[Any] = raw_proof if isinstance(raw_proof, list) else []
    clean = False if git.get("dirty") else True
    already_merged = bool(safety.get("safe_to_delete")) and classification in {
        "patch_equivalent_or_merged",
        "no_git_cache_residue",
        "unregistered_git_residue",
    }
    return WorkArtifact(
        artifact_id=str(candidate.get("candidate_id") or candidate.get("path") or "unknown"),
        kind="worktree",
        clean=clean,
        already_merged=already_merged,
        open_pr=classification == "open_pr_or_outbox" or bool(open_prs),
        owner_active=bool(candidate.get("active_session") or candidate.get("lock_files")),
        unique_commits=classification == "unique_unharvested",
        represented_elsewhere=classification in {"patch_equivalent_or_merged", "receipt_protected"},
        superseded=classification == "superseded",
        evidence=[str(item) for item in proof],
    )
