#!/usr/bin/env python3
"""CLI shell for the lane conductor (see aragora.swarm.lane_conductor).

Resolves the live merge-blocked PR queue and which PRs already have a live
owner, then plans (and, only with ``--execute``, claims + dispatches) one
conductor pass. Pure decision logic lives in the module; this shell only does
I/O so the decision stays unit-testable.

Reads route through the GitHub App installation token (separate API budget) so
the conductor and its workers do not starve the operator's per-user PAT quota.

DRY-RUN BY DEFAULT. It never merges or settles -- it only assigns, claims, and
drops work orders for the supervisor / worker_launcher to spawn (each worker is
itself gated by the merge-gate tooling).

Examples
--------
::

    # Preview the next pass (no claims, no dispatch).
    python3 scripts/lane_conductor.py --json --max-workers 3

    # Actually claim + drop work orders for up to 3 lanes.
    python3 scripts/lane_conductor.py --execute --max-workers 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm import github_app_auth  # noqa: E402
from aragora.swarm.lane_conductor import (  # noqa: E402
    DEFAULT_TARGET_AGENT,
    default_claim,
    default_dispatch,
    run_pass,
)
from aragora.swarm.lane_dispatcher import DEFAULT_MAX_WORKERS  # noqa: E402

# mergeStateStatus values that mean "open and waiting on checks/quorum/settlement"
# -- i.e. a candidate the swarm can usefully advance. CLEAN/DIRTY/DRAFT excluded.
_BLOCKED_STATES = {"BLOCKED", "UNSTABLE"}
# Owner-liveness assessments that mean the lane is NOT actively held, so it is
# reassignable. Anything else (including unknown) is treated as live -- the
# fail-safe direction is to avoid double-dispatching a possibly-live lane.
_RECLAIMABLE_ASSESSMENTS = {"stale", "terminal", "absent", "reclaimable"}
_UNKNOWN_OWNER = "owner-liveness-unavailable"
_OWNER_PROBE_TIMEOUT_SECONDS = 60


def _read_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")
    return github_app_auth.github_cli_env(env)


def _gh_json(args: list[str]) -> Any:
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False, env=_read_env(), timeout=60
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def fetch_candidates(repo: str) -> list[dict[str, Any]]:
    """Open, non-draft, merge-blocked PRs in oldest-first priority order."""
    rows = (
        _gh_json(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,headRefName,isDraft,mergeStateStatus",
            ]
        )
        or []
    )
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("isDraft"):
            continue
        if str(row.get("mergeStateStatus") or "").upper() not in _BLOCKED_STATES:
            continue
        number = row.get("number")
        if isinstance(number, int):
            candidates.append({"number": number, "branch": str(row.get("headRefName") or "")})
    # Oldest-first: lower PR numbers are usually closest to done.
    candidates.sort(key=lambda c: c["number"])
    return candidates


def fetch_live_claims(repo: str, candidates: list[dict[str, Any]]) -> dict[int, str]:
    """Map each candidate PR with a LIVE owner to that owner_session.

    Liveness comes from the canonical scripts/identify_lane_owner.py. A lane is
    treated as live (so the dispatcher will not reassign it) unless its owner is
    explicitly stale/terminal/absent -- the fail-safe direction avoids
    double-dispatching a possibly-live lane.
    """
    claims: dict[int, str] = {}

    def unknown_owner(reason: str) -> str:
        clean = " ".join(reason.strip().split())
        if not clean:
            return _UNKNOWN_OWNER
        return f"{_UNKNOWN_OWNER}: {clean[:240]}"

    for cand in candidates:
        pr = cand["number"]
        try:
            proc = subprocess.run(
                [
                    "python3",
                    str(REPO_ROOT / "scripts" / "identify_lane_owner.py"),
                    "--pr",
                    str(pr),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=_OWNER_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            claims[pr] = unknown_owner(
                f"identify_lane_owner.py timed out after {_OWNER_PROBE_TIMEOUT_SECONDS}s"
            )
            continue
        except OSError as exc:
            claims[pr] = unknown_owner(
                f"identify_lane_owner.py failed: {type(exc).__name__}: {exc}"
            )
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            stderr = proc.stderr or ""
            if "no lane matched" not in stderr.lower():
                reason = stderr or proc.stdout or f"identify_lane_owner.py exited {proc.returncode}"
                claims[pr] = unknown_owner(reason)
            continue
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            claims[pr] = unknown_owner("identify_lane_owner.py returned invalid JSON")
            continue
        owner = str(data.get("owner_session") or "").strip()
        if not owner:
            continue
        assessment = str((data.get("owner_liveness") or {}).get("assessed") or "").strip().lower()
        if assessment in _RECLAIMABLE_ASSESSMENTS:
            continue
        claims[pr] = owner
    return claims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument(
        "--root",
        default=".",
        help="repo root holding .aragora/lane_dispatch/ and lane claim scripts",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--target-agent", default=DEFAULT_TARGET_AGENT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write lane claims and dispatch work orders (default: dry-run preview).",
    )
    parser.add_argument("--json", dest="json_output", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    result = run_pass(
        repo=args.repo,
        fetch_candidates=fetch_candidates,
        fetch_live_claims=fetch_live_claims,
        max_workers=args.max_workers,
        target_agent=args.target_agent,
        execute=args.execute,
        claim_fn=lambda wo: default_claim(wo, repo_root=root),
        dispatch_fn=lambda wo: default_dispatch(wo, repo_root=root),
    )

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.reason)
        for wo in result.work_orders:
            print(f"  -> PR #{wo.pr} ({wo.branch}) :: {wo.owner_session} [{wo.target_agent}]")
        for pr, owner in result.owned.items():
            print(f"  .. PR #{pr} already live-owned by {owner}")
        for path in result.dispatched:
            print(f"  dispatched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
