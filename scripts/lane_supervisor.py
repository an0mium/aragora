#!/usr/bin/env python3
"""CLI shell for the lane supervisor (see aragora.swarm.lane_supervisor).

Drains conductor-dropped work orders from ``.aragora/lane_dispatch/pending/``
into worker launches. Pure state-machine logic lives in the module; this shell
only supplies the real launch implementation and does I/O.

DRY-RUN BY DEFAULT: prints which orders the next drain would launch and moves
nothing. ``--execute`` claims (atomic pending -> in_progress) and launches.

Launch seam (``--execute``): each work order is handed to
``aragora.swarm.worker_launcher.WorkerLauncher.launch``. That call is async and
needs a provisioned worktree -- which is operator-machine-specific -- so the
work order must carry a ``worktree`` (the conductor/operator provisions it). If
it does not, the launch fails for that order (recorded in failed/) and the drain
continues. Validate this path on your machine before relying on it; the drainer
state machine itself is fully unit-tested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.lane_supervisor import (  # noqa: E402
    DEFAULT_MAX_LAUNCHES,
    drain_once,
    plan_drain,
)


def _worker_launcher_launch(work_order: dict[str, Any]) -> None:
    """Default launch: hand the work order to WorkerLauncher (async).

    Requires a ``worktree`` on the work order (operator/conductor provisions it).
    Raises if absent or if the launch fails -- the drainer records it in failed/.
    """
    worktree = str(work_order.get("worktree") or "").strip()
    if not worktree:
        raise RuntimeError(
            f"work order {work_order.get('work_order_id')} has no 'worktree'; "
            "provision an isolated worktree for the branch and set it before launch"
        )
    from aragora.swarm.worker_launcher import WorkerLauncher

    launcher = WorkerLauncher()
    branch = str(work_order.get("branch") or "main")
    asyncio.run(launcher.launch(work_order, worktree_path=worktree, branch=branch))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repo root holding .aragora/lane_dispatch/")
    parser.add_argument("--max-launches", type=int, default=DEFAULT_MAX_LAUNCHES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Claim + launch pending orders (default: dry-run preview).",
    )
    parser.add_argument("--json", dest="json_output", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.execute:
        result = drain_once(
            root=root, launch_fn=_worker_launcher_launch, max_launches=args.max_launches
        )
    else:
        result = plan_drain(root, max_launches=args.max_launches)

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.reason)
        for wo_id in result.launched:
            print(f"  launched: {wo_id}")
        for failure in result.failed:
            print(f"  FAILED: {failure['work_order_id']} -- {failure['error']}")
        for wo_id in result.skipped:
            print(f"  skipped (claim race): {wo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
