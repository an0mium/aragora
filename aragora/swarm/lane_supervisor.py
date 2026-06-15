"""Lane supervisor: drain dispatched work orders into worker launches.

The lane conductor (:mod:`aragora.swarm.lane_conductor`) drops self-describing
work orders as JSON into ``.aragora/lane_dispatch/pending/``. This module is the
*back* of that handoff: it drains pending orders and hands each to a worker
launcher, with a file-state machine that is the load-bearing safety primitive:

    pending/  --claim (atomic rename)-->  in_progress/  --+--> done/
                                                          +--> failed/

The atomic ``pending -> in_progress`` rename is how two concurrent supervisors
(or a retry) never double-spawn the same work order: exactly one rename wins;
the loser sees the source gone and skips. Each order ends in ``done/`` (launched)
or ``failed/`` (launch raised, error recorded), so the queue is always
inspectable and replayable.

Pure/testable by construction: the actual spawn is an injected ``launch_fn``
(the CLI wires the real :class:`aragora.swarm.worker_launcher.WorkerLauncher`;
tests inject a fake), so the whole state machine is exercised without spawning a
process or provisioning a worktree. ``plan_drain`` is a read-only preview (the
CLI's dry-run default); ``drain_once`` performs the claims + launches.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DISPATCH_ROOT = Path(".aragora") / "lane_dispatch"
PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
FAILED = "failed"
DEFAULT_MAX_LAUNCHES = 3

# A launch callable spawns one worker for a work order; it raises on failure.
LaunchFn = Callable[[dict[str, Any]], None]


@dataclass
class DrainResult:
    launched: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "launched": list(self.launched),
            "failed": list(self.failed),
            "skipped": list(self.skipped),
            "deferred": list(self.deferred),
            "reason": self.reason,
        }


def _state_dir(root: Path, name: str) -> Path:
    return root / DISPATCH_ROOT / name


def _read_order(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not str(data.get("work_order_id") or "").strip():
        return None
    return data


def load_pending(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Parsed, valid pending work orders, oldest file first (stable order)."""
    pending = _state_dir(root, PENDING)
    if not pending.is_dir():
        return []
    orders: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(pending.glob("*.json")):
        order = _read_order(path)
        if order is not None:
            orders.append((path, order))
    return orders


def claim_order(path: Path, root: Path) -> Path | None:
    """Atomically move one order pending -> in_progress; the claim.

    Returns the new in_progress path, or ``None`` if another drainer already
    claimed it (the rename source vanished) -- the double-spawn guard.
    """
    dest_dir = _state_dir(root, IN_PROGRESS)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    try:
        path.rename(dest)
    except (FileNotFoundError, OSError):
        return None
    return dest


def _settle(in_progress_path: Path, root: Path, *, ok: bool) -> Path:
    dest_dir = _state_dir(root, DONE if ok else FAILED)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / in_progress_path.name
    in_progress_path.rename(dest)
    return dest


def plan_drain(root: Path, *, max_launches: int = DEFAULT_MAX_LAUNCHES) -> DrainResult:
    """Read-only preview: which pending orders the next drain would launch."""
    pending = load_pending(root)
    cap = max(0, int(max_launches))
    result = DrainResult()
    for _path, order in pending:
        wo_id = str(order.get("work_order_id"))
        if len(result.launched) < cap:
            result.launched.append(wo_id)
        else:
            result.deferred.append(wo_id)
    result.reason = (
        f"[dry-run] {len(result.launched)} of {len(pending)} pending order(s) would launch; "
        f"{len(result.deferred)} deferred (max_launches={cap})"
    )
    return result


def drain_once(
    *,
    root: Path,
    launch_fn: LaunchFn,
    max_launches: int = DEFAULT_MAX_LAUNCHES,
    now: Callable[[], str] | None = None,
) -> DrainResult:
    """Claim and launch up to ``max_launches`` pending orders.

    Each order is atomically claimed (pending -> in_progress) before launch, so a
    concurrent drainer never double-spawns it. On a successful launch the order
    moves to done/; if ``launch_fn`` raises, it moves to failed/ with the error
    recorded, and the drain continues.
    """
    stamp = now or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    cap = max(0, int(max_launches))
    result = DrainResult()
    for path, order in load_pending(root):
        wo_id = str(order.get("work_order_id"))
        if len(result.launched) >= cap:
            result.deferred.append(wo_id)
            continue
        claimed = claim_order(path, root)
        if claimed is None:
            # Lost the claim race to another drainer; leave it to them.
            result.skipped.append(wo_id)
            continue
        try:
            launch_fn(order)
        except Exception as exc:  # noqa: BLE001 - one bad launch must not abort the drain
            order["_launch_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            order["_failed_at"] = stamp()
            claimed.write_text(json.dumps(order, indent=2), encoding="utf-8")
            _settle(claimed, root, ok=False)
            result.failed.append({"work_order_id": wo_id, "error": order["_launch_error"]})
            continue
        _settle(claimed, root, ok=True)
        result.launched.append(wo_id)
    result.reason = (
        f"launched {len(result.launched)}, failed {len(result.failed)}, "
        f"skipped {len(result.skipped)} (claim race), deferred {len(result.deferred)} "
        f"(max_launches={cap})"
    )
    return result
