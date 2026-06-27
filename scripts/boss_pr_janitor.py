#!/usr/bin/env python3
"""Boss PR dedupe janitor: enforce one draft PR per boss-loop issue.

The boss loop opens draft PRs on branches matching
``aragora/boss-harvest/issue-<N>-boss-<hash>`` and
``aragora/boss/issue-<N>-...``. Duplicate drafts for the same issue burn
merge-quorum review capacity, one of the stall classes described in
``docs/governance/BOSS_LOOP_MERGE_GATE_RESILIENCE.md``. This janitor
net-closes those duplicates per Sprint 3 goal 3(ii) (anti-goal clause (b):
prefer net-closing existing queue items over creating new work).

Behavior:

* Reads open PRs via ``gh pr list`` (read-only).
* Groups PRs by issue number extracted from the head branch name; only the
  two boss prefixes above are ever considered.
* For each issue with two or more *draft* PRs, selects one keeper — the most
  recently created draft whose checks are passing or pending, falling back to
  the most recently created draft — and plans ``gh pr close`` for the rest
  with a comment naming the keeper. Branches are preserved (no
  ``--delete-branch``).
* Ready (non-draft) PRs are NEVER closed, even when duplicated.
* Dry-run is the default: without ``--apply`` the plan is printed as JSON
  lines and no mutating ``gh`` command is invoked. Exit code 0.
* With ``--apply``, any failed mutation fails closed: exit code 1.
* ``--max-closes`` (default 10) caps the number of planned closes per run.

Stdlib-only by design so it can run anywhere ``gh`` is authenticated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Sequence

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_MAX_CLOSES = 10
GH_TIMEOUT_SECONDS = 60

# Only these boss-loop branch prefixes are ever considered.
_BOSS_BRANCH_RE = re.compile(r"^aragora/boss(?:-harvest)?/issue-(\d+)(?:-|$)")

# Check states/conclusions that disqualify a PR from preferred-keeper status.
_FAILING_STATES = {
    "FAILURE",
    "ERROR",
    "CANCELLED",
    "TIMED_OUT",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
}


def extract_issue_number(head_ref: str) -> int | None:
    """Return the issue number for a boss-loop branch, else None.

    Only ``aragora/boss-harvest/issue-<N>-...`` and
    ``aragora/boss/issue-<N>-...`` qualify; anything else is ignored.
    """
    match = _BOSS_BRANCH_RE.match(head_ref or "")
    if match is None:
        return None
    return int(match.group(1))


def _has_failing_checks(pr: dict[str, Any]) -> bool:
    rollup = pr.get("statusCheckRollup") or []
    if not isinstance(rollup, list):
        return False
    for check in rollup:
        if not isinstance(check, dict):
            continue
        state = str(check.get("state") or check.get("conclusion") or "").upper()
        conclusion = str(check.get("conclusion") or "").upper()
        if state in _FAILING_STATES or conclusion in _FAILING_STATES:
            return True
    return False


def _created_at(pr: dict[str, Any]) -> str:
    return str(pr.get("createdAt") or "")


def select_keeper(drafts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the keeper among duplicate draft PRs.

    Most recently created among those with passing/pending checks; if every
    draft has failing checks, fall back to the most recently created draft.
    """
    healthy = [pr for pr in drafts if not _has_failing_checks(pr)]
    pool = healthy if healthy else list(drafts)
    return max(pool, key=lambda pr: (_created_at(pr), int(pr.get("number") or 0)))


def _close_comment(keeper_number: int) -> str:
    return (
        f"Superseded by #{keeper_number} (boss PR janitor: one draft per issue; branch preserved)."
    )


def build_plan(
    prs: Sequence[dict[str, Any]], max_closes: int = DEFAULT_MAX_CLOSES
) -> list[dict[str, Any]]:
    """Build the close plan. Pure function; never touches the network.

    Only draft PRs on boss-loop branches are ever planned for closure.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for pr in prs:
        issue = extract_issue_number(str(pr.get("headRefName") or ""))
        if issue is None:
            continue
        groups.setdefault(issue, []).append(pr)

    plan: list[dict[str, Any]] = []
    for issue in sorted(groups):
        drafts = [pr for pr in groups[issue] if pr.get("isDraft") is True]
        if len(drafts) < 2:
            continue
        keeper = select_keeper(drafts)
        keeper_number = int(keeper["number"])
        losers = sorted(
            (pr for pr in drafts if int(pr["number"]) != keeper_number),
            key=lambda pr: int(pr["number"]),
        )
        for pr in losers:
            plan.append(
                {
                    "action": "close",
                    "issue": issue,
                    "pr": int(pr["number"]),
                    "keeper": keeper_number,
                    "head_ref": pr.get("headRefName"),
                    "title": pr.get("title"),
                    "comment": _close_comment(keeper_number),
                }
            )

    cap = max(0, int(max_closes))
    return plan[:cap]


def fetch_open_prs(repo: str) -> list[dict[str, Any]]:
    """Read open PRs via gh (read-only).

    Deliberately omits ``statusCheckRollup`` here: requesting it for 200 PRs
    reliably 504s GitHub's GraphQL API. Rollups are fetched lazily per-PR by
    :func:`enrich_duplicate_drafts` for duplicate draft groups only.
    """
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        "number,headRefName,isDraft,createdAt,title",
        "--limit",
        "200",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed (exit {result.returncode}): {result.stderr.strip()}")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list returned unexpected payload (expected a list)")
    return payload


def fetch_status_rollup(repo: str, number: int) -> list[dict[str, Any]]:
    """Read a single PR's check rollup via gh (read-only)."""
    command = [
        "gh",
        "pr",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "statusCheckRollup",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh pr view {number} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout or "{}")
    rollup = payload.get("statusCheckRollup") if isinstance(payload, dict) else None
    return rollup if isinstance(rollup, list) else []


def enrich_duplicate_drafts(repo: str, prs: list[dict[str, Any]]) -> None:
    """Attach ``statusCheckRollup`` to drafts in duplicate groups (in place).

    Read-only. PRs that already carry a ``statusCheckRollup`` key are left
    untouched. A failed per-PR fetch degrades to an empty rollup (treated as
    pending), which only influences keeper choice among drafts — it can never
    cause a ready PR to be closed.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for pr in prs:
        issue = extract_issue_number(str(pr.get("headRefName") or ""))
        if issue is None:
            continue
        groups.setdefault(issue, []).append(pr)

    for members in groups.values():
        drafts = [pr for pr in members if pr.get("isDraft") is True]
        if len(drafts) < 2:
            continue
        for pr in drafts:
            if "statusCheckRollup" in pr:
                continue
            try:
                pr["statusCheckRollup"] = fetch_status_rollup(repo, int(pr["number"]))
            except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError):
                pr["statusCheckRollup"] = []


def _close_command(repo: str, action: dict[str, Any]) -> list[str]:
    return [
        "gh",
        "pr",
        "close",
        str(action["pr"]),
        "--repo",
        repo,
        "--comment",
        str(action["comment"]),
    ]


def apply_plan(repo: str, plan: Sequence[dict[str, Any]]) -> int:
    """Apply close actions. Returns the number of failures (fail closed)."""
    failures = 0
    for action in plan:
        command = _close_command(repo, action)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=GH_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures += 1
            print(json.dumps({**action, "applied": False, "error": f"{type(exc).__name__}"}))
            continue
        ok = result.returncode == 0
        if not ok:
            failures += 1
        print(
            json.dumps(
                {
                    **action,
                    "applied": ok,
                    **({"error": result.stderr.strip()[:300]} if not ok else {}),
                }
            )
        )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Boss PR dedupe janitor: keep one draft PR per boss-loop issue, "
            "close the rest. Dry-run by default."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually close duplicate draft PRs (default: dry-run, read-only)",
    )
    parser.add_argument(
        "--max-closes",
        type=int,
        default=DEFAULT_MAX_CLOSES,
        help=f"Maximum PRs to close per run (default {DEFAULT_MAX_CLOSES})",
    )
    args = parser.parse_args(argv)

    try:
        prs = fetch_open_prs(args.repo)
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return 1

    enrich_duplicate_drafts(args.repo, prs)
    plan = build_plan(prs, max_closes=args.max_closes)

    if not args.apply:
        for action in plan:
            print(json.dumps({**action, "dry_run": True}))
        print(
            json.dumps(
                {
                    "action": "summary",
                    "dry_run": True,
                    "open_prs": len(prs),
                    "planned_closes": len(plan),
                    "max_closes": args.max_closes,
                }
            )
        )
        return 0

    failures = apply_plan(args.repo, plan)
    print(
        json.dumps(
            {
                "action": "summary",
                "dry_run": False,
                "planned_closes": len(plan),
                "failures": failures,
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
