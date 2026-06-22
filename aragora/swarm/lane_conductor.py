"""Lane conductor: turn the live PR queue into claimed, dispatched work orders.

This is the autonomous front of the merge-advance loop, sitting on top of
:mod:`aragora.swarm.lane_dispatcher` (the assignment decision) and feeding the
existing worker machinery. One *pass* does:

1. resolve merge-blocked PR candidates (priority order),
2. resolve which of them already have a LIVE owner (via ``identify_lane_owner``),
3. assign each free worker exactly one unclaimed PR (lane_dispatcher),
4. for each assignment, build a self-describing work order carrying the short,
   constant claim-first worker prompt,
5. (only under ``execute``) write the atomic lane claim and drop the work order
   for the supervisor / ``worker_launcher`` to spawn.

Safety:

* **Dry-run by default.** ``run_pass`` plans and returns/prints; it writes
  claims and dispatches work orders only when ``execute=True``.
* **Never merges or settles.** The conductor only *assigns and launches* gated
  workers; every merge-authority decision still happens in the merge-gate
  tooling the workers call, never here.
* **Bounded.** ``max_workers`` caps fan-out per pass (backpressure), and a live
  owner is never displaced.

Pure by construction: ``plan_pass`` / ``build_work_orders`` make no I/O. The
fetch / claim / dispatch steps are injected callables (the CLI supplies real
``gh``/``identify_lane_owner``/file-drop implementations; tests supply fakes),
so every decision is exercised without a network, a worktree, or a spawned
process.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aragora.swarm.lane_dispatcher import (
    DEFAULT_MAX_WORKERS,
    LaneAssignment,
    build_worker_prompt,
    default_session_id,
    select_assignments,
)
from aragora.swarm.lane_supervisor import DISPATCH_ROOT, PENDING

DEFAULT_TARGET_AGENT = "codex"
# Durable work-order drop the supervisor / worker_launcher drains. A file-drop
# (not a direct async spawn) keeps the conductor decoupled from worktree
# provisioning and lets every dispatch be inspected/replayed.
DISPATCH_PENDING_DIR = DISPATCH_ROOT / PENDING


@dataclass
class WorkOrderSpec:
    """A self-describing unit of work for one assigned lane.

    Carries the launch-relevant fields: the target agent, the claimed owner
    session, the PR/branch, and the exact claim-first ``prompt``. The dict keys
    (``target_agent``, ``owner_session_id``, ``prompt``, ``branch``) match what
    ``worker_launcher.WorkerLauncher.launch`` consumes -- ``launch`` reads
    ``target_agent``/``owner_session_id`` and builds the worker prompt from
    ``prompt`` -- so the work order is handed straight through as ``launch``'s
    ``work_order`` argument.

    It deliberately does NOT carry a ``worktree``: provisioning an isolated
    worktree for the branch is operator-machine work (git fetch + checkout), so
    the supervisor's launch seam (``scripts/lane_supervisor.py``) provisions one
    and passes it as ``launch(worktree_path=...)``. Keeping it out of the spec is
    what lets the conductor stay pure (no git, no network, no spawned process).
    """

    work_order_id: str
    pr: int
    branch: str
    target_agent: str
    owner_session: str
    repo: str
    prompt: str
    created_at: str = ""
    # Opt in to verbatim prompt rendering in WorkerLauncher._build_prompt: the
    # claim-first prompt IS the directive and must reach the worker unmodified.
    prompt_verbatim: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Mirror the launcher's expected key so a thin adapter needs no mapping.
        data["owner_session_id"] = self.owner_session
        return data


@dataclass
class ConductorPass:
    work_orders: list[WorkOrderSpec] = field(default_factory=list)
    owned: dict[int, str] = field(default_factory=dict)
    deferred: list[int] = field(default_factory=list)
    dispatched: list[str] = field(default_factory=list)
    claim_failed: list[int] = field(default_factory=list)
    executed: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_orders": [wo.to_dict() for wo in self.work_orders],
            "owned": {str(pr): owner for pr, owner in self.owned.items()},
            "deferred": list(self.deferred),
            "dispatched": list(self.dispatched),
            "claim_failed": list(self.claim_failed),
            "executed": self.executed,
            "reason": self.reason,
        }


def _work_order_id(pr: int, owner_session: str) -> str:
    return f"lane-{pr}-{owner_session}"


def build_work_orders(
    assignments: Sequence[LaneAssignment],
    *,
    repo: str,
    target_agent: str = DEFAULT_TARGET_AGENT,
    now: Callable[[], str] | None = None,
) -> list[WorkOrderSpec]:
    """Build one self-describing work order per assignment (no I/O)."""
    stamp = now or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    orders: list[WorkOrderSpec] = []
    for assignment in assignments:
        prompt = build_worker_prompt(
            pr=assignment.pr,
            branch=assignment.branch,
            session_id=assignment.owner_session,
            repo=repo,
            target_agent=target_agent,
        )
        orders.append(
            WorkOrderSpec(
                work_order_id=_work_order_id(assignment.pr, assignment.owner_session),
                pr=assignment.pr,
                branch=assignment.branch,
                target_agent=target_agent,
                owner_session=assignment.owner_session,
                repo=repo,
                prompt=prompt,
                created_at=stamp(),
            )
        )
    return orders


def plan_pass(
    *,
    candidates: Sequence[dict[str, Any]],
    live_claims_by_pr: dict[int, str],
    repo: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    target_agent: str = DEFAULT_TARGET_AGENT,
    session_id_for: Callable[[int], str] = default_session_id,
    now: Callable[[], str] | None = None,
) -> ConductorPass:
    """Plan one conductor pass from live state -- pure, no I/O."""
    plan = select_assignments(
        candidates=candidates,
        live_claims_by_pr=live_claims_by_pr,
        max_workers=max_workers,
        session_id_for=session_id_for,
    )
    work_orders = build_work_orders(plan.assignments, repo=repo, target_agent=target_agent, now=now)
    return ConductorPass(
        work_orders=work_orders,
        owned=plan.owned,
        deferred=plan.deferred,
        reason=plan.reason,
    )


# ---------------------------------------------------------------------------
# Execute-mode side effects (only ever called when execute=True). Each is a
# small, replaceable callable so run_pass stays testable with fakes.
# ---------------------------------------------------------------------------


# Owner-liveness assessments (from identify_lane_owner) that mean the lane is
# NOT actively held, so a stale row holding it may be released and reclaimed.
_RECLAIMABLE_ASSESSMENTS = {"stale", "terminal", "absent", "reclaimable"}


def _run_lane_claim(work_order: WorkOrderSpec, root: Path) -> subprocess.CompletedProcess | None:
    """Run one PLAIN lane claim (no --force). None on spawn/timeout failure."""
    try:
        return subprocess.run(
            [
                "python3",
                str(root / "scripts" / "claim_active_agent_lane.py"),
                "--lane-id",
                work_order.work_order_id,
                "--owner-session",
                work_order.owner_session,
                "--pr-number",
                str(work_order.pr),
                "--branch",
                work_order.branch,
                "--source",
                work_order.target_agent,
                "--status",
                "active",
                "--next-action",
                f"advance #{work_order.pr}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _release_stale_conflict(pr: int, root: Path) -> bool:
    """Release PR #``pr``'s current lane owner IFF it is assessed reclaimable.

    claim_lane's conflict check is status-based, not heartbeat-based, so a
    stale-but-still-``active`` row blocks a plain claim. We clear it only when
    identify_lane_owner assesses the owner stale/terminal/absent -- NEVER a live
    owner -- so a competing live worker is never displaced (the blind---force
    clobber race is avoided). Returns True if a stale owner was released.
    """
    try:
        probe = subprocess.run(
            [
                "python3",
                str(root / "scripts" / "identify_lane_owner.py"),
                "--pr",
                str(pr),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if probe.returncode != 0 or not probe.stdout.strip():
        return False
    try:
        data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return False
    owner = str(data.get("owner_session") or "").strip()
    liveness = data.get("owner_liveness")
    assessed = (
        str((liveness.get("assessed") if isinstance(liveness, dict) else "") or "").strip().lower()
    )
    if not owner or assessed not in _RECLAIMABLE_ASSESSMENTS:
        return False  # no owner, or owner is LIVE -> never release / never displace
    try:
        rel = subprocess.run(
            [
                "python3",
                str(root / "scripts" / "claim_active_agent_lane.py"),
                "--release-stale",
                "--owner-session",
                owner,
                "--pr-number",
                str(pr),
                "--ttl-minutes",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return rel.returncode == 0


def default_claim(work_order: WorkOrderSpec, *, repo_root: Path | None = None) -> bool:
    """Claim the lane via scripts/claim_active_agent_lane.py.

    Plain claim first (no --force): claim_lane's identity-conflict check then
    provides mutual exclusion, so a competing LIVE claim makes this fail and we
    never double-dispatch. If it fails because a STALE row holds the resource,
    release that stale row (only when its owner is assessed reclaimable) and retry
    once. This reclaims stale lanes without the blind-force clobber race that
    could overwrite a newly-live owner.
    """
    root = repo_root or Path.cwd()
    proc = _run_lane_claim(work_order, root)
    if proc is None:
        return False
    if proc.returncode == 0:
        return True
    # Claim refused -- if a stale row is holding this PR, clear it and retry once.
    if _release_stale_conflict(work_order.pr, root):
        retry = _run_lane_claim(work_order, root)
        return retry is not None and retry.returncode == 0
    return False


def default_dispatch(work_order: WorkOrderSpec, *, repo_root: Path | None = None) -> str:
    """Drop the work order as JSON for the supervisor/worker_launcher to drain.

    Returns the path written. Atomic (tmp + rename) so a draining supervisor
    never reads a half-written order.
    """
    root = repo_root or Path.cwd()
    pending = root / DISPATCH_PENDING_DIR
    pending.mkdir(parents=True, exist_ok=True)
    target = pending / f"{work_order.work_order_id}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(work_order.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(target)
    return str(target)


def default_release(work_order: WorkOrderSpec, *, repo_root: Path | None = None) -> None:
    """Best-effort release of a claimed lane (used when dispatch fails post-claim).

    Scoped to the order's unique owner_session (ttl 0), so it clears only this
    lane. Failures are swallowed -- the caller already records the lane as failed.
    """
    root = repo_root or Path.cwd()
    try:
        subprocess.run(
            [
                "python3",
                str(root / "scripts" / "claim_active_agent_lane.py"),
                "--release-stale",
                "--lane-id",
                work_order.work_order_id,
                "--owner-session",
                work_order.owner_session,
                "--pr-number",
                str(work_order.pr),
                "--ttl-minutes",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_pass(
    *,
    repo: str,
    fetch_candidates: Callable[[str], Sequence[dict[str, Any]]],
    fetch_live_claims: Callable[[str, Sequence[dict[str, Any]]], dict[int, str]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    target_agent: str = DEFAULT_TARGET_AGENT,
    execute: bool = False,
    claim_fn: Callable[[WorkOrderSpec], bool | None] | None = None,
    dispatch_fn: Callable[[WorkOrderSpec], str] | None = None,
    release_fn: Callable[[WorkOrderSpec], None] | None = None,
    session_id_for: Callable[[int], str] = default_session_id,
    now: Callable[[], str] | None = None,
) -> ConductorPass:
    """Run one conductor pass. Plans always; claims+dispatches only if ``execute``.

    ``fetch_candidates``/``fetch_live_claims`` are injected so the I/O lives in
    the caller (the CLI wires the real ``gh``/``identify_lane_owner``
    implementations). With ``execute=False`` (default) nothing is claimed or
    dispatched -- the returned plan is a preview. In execute mode ``claim_fn``
    must return exactly ``True`` before dispatch happens; any falsey value,
    timeout wrapper, or non-boolean failure sentinel blocks dispatch for that
    lane.
    """
    candidates = list(fetch_candidates(repo))
    live_claims = fetch_live_claims(repo, candidates)
    result = plan_pass(
        candidates=candidates,
        live_claims_by_pr=live_claims,
        repo=repo,
        max_workers=max_workers,
        target_agent=target_agent,
        session_id_for=session_id_for,
        now=now,
    )
    if not execute:
        result.reason += " [dry-run: no claims written, no work orders dispatched]"
        return result

    claim = claim_fn or default_claim
    dispatch = dispatch_fn or default_dispatch
    release = release_fn or default_release
    for work_order in result.work_orders:
        if claim(work_order) is not True:
            result.claim_failed.append(work_order.pr)
            continue
        try:
            result.dispatched.append(dispatch(work_order))
        except Exception:  # noqa: BLE001 - any dispatch failure must not wedge the lane
            # Claimed but dispatch failed: release the claim so the lane isn't
            # left claimed-with-no-pending-order (wedged until TTL/manual cleanup).
            release(work_order)
            result.claim_failed.append(work_order.pr)
    result.executed = True
    return result
