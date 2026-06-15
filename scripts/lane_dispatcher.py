#!/usr/bin/env python3
"""CLI shell for claim-first lane dispatch (see aragora.swarm.lane_dispatcher).

Pure decision logic lives in :mod:`aragora.swarm.lane_dispatcher`; this shell
only does argument parsing + I/O so the decision stays unit-testable. Candidates
and live claims are injected as JSON (inline or ``@file``) -- the live wiring
(resolve merge-blocked PRs via ``merge_quorum_io``, resolve live owners via
``scripts/identify_lane_owner.py``, spawn workers via
``aragora.swarm.worker_launcher``) is the operator/conductor's job and is kept
out of the decision so it can run anywhere.

Examples
--------
::

    # Emit a dispatch plan from explicit inputs.
    python3 scripts/lane_dispatcher.py --json \\
        --candidates-json '[{"number":8405,"branch":"codex/a"}]' \\
        --live-claims-json '{"8406":"sess-x"}' --max-workers 3

    # Print the short claim-first worker prompt for one assigned lane.
    python3 scripts/lane_dispatcher.py --print-prompt --pr 8405 --branch codex/a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.lane_dispatcher import (  # noqa: E402
    DEFAULT_MAX_WORKERS,
    build_worker_prompt,
    default_session_id,
    live_claims_from_arg,
    select_assignments,
)


def _load_json_arg(raw: str | None, *, default: Any) -> Any:
    if not raw:
        return default
    text = raw.strip()
    if text.startswith("@"):
        with open(text[1:], encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument(
        "--candidates-json",
        help="JSON array of {number,branch,head} (merge-blocked PRs, priority order); "
        "prefix with @ to read from a file.",
    )
    parser.add_argument(
        "--live-claims-json",
        help='JSON of live owners: {"<pr>":"owner"} or [{pr,owner_session}]; @file ok.',
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--json", dest="json_output", action="store_true")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the worker prompt for --pr/--branch instead of a dispatch plan.",
    )
    parser.add_argument("--pr", type=int, default=0)
    parser.add_argument("--branch", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args(argv)

    if args.print_prompt:
        if not args.pr:
            print("error: --print-prompt requires --pr", file=sys.stderr)
            return 1
        session = args.session_id or default_session_id(args.pr)
        print(
            build_worker_prompt(pr=args.pr, branch=args.branch, session_id=session, repo=args.repo)
        )
        return 0

    candidates = _load_json_arg(args.candidates_json, default=[])
    live_claims = live_claims_from_arg(_load_json_arg(args.live_claims_json, default={}))
    if not isinstance(candidates, list):
        print("error: --candidates-json must be a JSON array", file=sys.stderr)
        return 1

    plan = select_assignments(
        candidates=candidates,
        live_claims_by_pr=live_claims,
        max_workers=args.max_workers,
    )
    if args.json_output:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.reason)
        for assignment in plan.assignments:
            print(f"  -> PR #{assignment.pr} ({assignment.branch}) :: {assignment.owner_session}")
        for pr, owner in plan.owned.items():
            print(f"  .. PR #{pr} already live-owned by {owner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
