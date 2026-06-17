#!/usr/bin/env python3
"""Run one bounded drain pass over the open-PR queue. Dry-run by default.

Wires the tested, safe-by-construction drain core
(:mod:`aragora.swarm.boss_drain`) to real ``gh`` I/O so the boss loop (or an
operator) can drain the backlog when it is over the cap, instead of idling or
generating new work.

Safety:
- ``--dry-run`` (DEFAULT) executes NOTHING — it prints the plan (what it would
  MERGE / CLOSE / REPAIR / LEAVE). ``--apply`` is required to act.
- MERGE re-confirms authorization with ``settle_one_pr.py`` at apply time (the
  classification proxy is only used to *propose*); a PR is merged only if settle
  reports ``packet_authorized_dry_run`` with no blockers — the gate stays the
  sole authority. Tier-4 / over-tier never auto-merge (settle blocks them).
- CLOSE only fires on genuinely empty PRs (0 changed files). A red-but-useful PR
  is REPAIR, never closed.
- off-limits branch prefixes (Factory ``structex/``, ``claude/fusion-``) and
  pinned PR numbers are LEFT untouched — no cross-fleet collision.
- REPAIR is surfaced (labelled ``drain-repair``) — NOT auto-dispatched in this
  version — and is bounded by the drain-pass caps regardless.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from aragora.swarm.boss_drain import (
    DEFAULT_OFF_LIMITS_PREFIXES,
    DrainContext,
    make_repair_order,
    run_boss_drain,
)
from aragora.swarm.drain_pass import DrainPassPolicy
from aragora.swarm.drain_policy import DrainAction, DrainPolicy

_REQUIRED = {"lint", "typecheck", "sdk-parity", "Generate & Validate", "TypeScript SDK Type Check"}


def _gh_json(args: list[str], timeout: int = 40) -> Any:
    try:
        out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return json.loads(out.stdout)
    except Exception:  # noqa: BLE001 - gh hiccup must not crash the pass
        return None


def list_open_prs(repo: str, limit: int) -> list[dict[str, Any]]:
    data = _gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            "number,headRefName",
        ]
    )
    return data if isinstance(data, list) else []


def view_pr(repo: str, number: int) -> dict[str, Any] | None:
    return _gh_json(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,headRefName,changedFiles,files,mergeable,headRefOid,statusCheckRollup",
        ]
    )


def _proxy_authorized(view: dict[str, Any]) -> tuple[bool, int]:
    """Cheap MERGE proxy from a gh view: 6 required green + mergeable + quorum.

    Only a *proposal*; ``--apply`` re-confirms with settle_one_pr before merging.
    """
    if view.get("mergeable") != "MERGEABLE":
        return (False, 0)
    rollup = view.get("statusCheckRollup") or []
    states = {c.get("name"): (c.get("conclusion") or c.get("status")) for c in rollup}
    required_ok = all(states.get(name) == "SUCCESS" for name in _REQUIRED)
    quorum_ok = states.get("aragora-merge-quorum") == "SUCCESS"
    return (required_ok and quorum_ok, 0)


def _settle_authorized(repo: str, number: int) -> bool:
    try:
        out = subprocess.run(
            ["python3", "scripts/settle_one_pr.py", "--pr", str(number), "--repo", repo, "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        rep = json.loads(out.stdout) if out.stdout.strip() else {}
        return rep.get("status") == "packet_authorized_dry_run" and not rep.get("blockers")
    except Exception:  # noqa: BLE001
        return False


def dispatch_repair(repo: str, pr: int, *, dry_run: bool, enable_repair: bool, agent: str) -> bool:
    """Repair a red-but-useful PR. Spawns a worker ONLY with --apply AND --enable-repair-dispatch.

    Otherwise (dry-run, or --apply without the explicit enable flag) it just prints the
    bounded plan and spawns nothing — so autonomous repair workers can never start by
    accident; turning them on is a deliberate, separate decision.
    """
    branch = str((view_pr(repo, pr) or {}).get("headRefName", ""))
    order = make_repair_order(pr, branch, agent=agent)
    if dry_run or not enable_repair:
        gate = (
            "dry-run" if dry_run else "repair-dispatch NOT enabled (need --enable-repair-dispatch)"
        )
        print(f"  [repair-plan/{gate}] #{pr} branch={branch} agent={agent}")
        return True
    # ENABLED apply path (bounded; isolated worktree on the PR branch).
    import shutil
    import tempfile

    wt = tempfile.mkdtemp(prefix=f"drain-repair-{pr}-")
    try:
        if (
            subprocess.run(
                ["git", "worktree", "add", "--force", wt, branch],
                capture_output=True,
                text=True,
                timeout=120,
            ).returncode
            != 0
        ):
            return False
        if agent == "codex":
            run = subprocess.run(
                ["codex", "exec", "--full-auto", "-"],
                input=order.prompt,
                cwd=wt,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        else:
            run = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--strict-mcp-config",
                    "--mcp-config",
                    '{"mcpServers":{}}',
                    order.prompt,
                ],
                cwd=wt,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        # Only propagate the worker's commits if the run itself succeeded — a
        # broken / scope-violating / timed-out agent run must NOT push whatever
        # it left in the worktree to the PR branch. Report success only if the
        # push also landed.
        if run.returncode != 0:
            return False
        push = subprocess.run(
            ["git", "-C", wt, "push", "origin", branch], capture_output=True, timeout=120
        )
        return push.returncode == 0
    except Exception:  # noqa: BLE001 - one repair failure never aborts the pass
        return False
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", wt], capture_output=True, timeout=60
        )
        shutil.rmtree(wt, ignore_errors=True)  # backstop if 'worktree add' never registered wt


def make_execute_fn(
    repo: str, *, dry_run: bool, enable_repair: bool = False, repair_agent: str = "codex"
):
    def execute(pr: int, action: DrainAction) -> bool:
        if action is DrainAction.REPAIR:
            return dispatch_repair(
                repo, pr, dry_run=dry_run, enable_repair=enable_repair, agent=repair_agent
            )
        if dry_run:
            return True  # plan only
        if action is DrainAction.MERGE:
            if not _settle_authorized(repo, pr):  # re-confirm authority at apply time
                return False
            head = (view_pr(repo, pr) or {}).get("headRefOid", "")
            cmd = ["gh", "pr", "merge", str(pr), "--repo", repo, "--squash"]
            if head:
                cmd += ["--match-head-commit", head]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120).returncode == 0
        if action is DrainAction.CLOSE_SUPERSEDED:
            return (
                subprocess.run(
                    [
                        "gh",
                        "pr",
                        "close",
                        str(pr),
                        "--repo",
                        repo,
                        "--comment",
                        "drain: truly superseded (empty/no changes) — closing",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                ).returncode
                == 0
            )
        return True

    return execute


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="synaptent/aragora")
    p.add_argument("--apply", action="store_true", help="execute (default: dry-run, plan only)")
    p.add_argument("--max-classify", type=int, default=60)
    p.add_argument("--list-limit", type=int, default=300)
    p.add_argument("--max-merges", type=int, default=5)
    p.add_argument("--max-closes", type=int, default=15)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--auto-settle-max-tier", type=int, default=2)
    p.add_argument("--off-limits-pr", type=int, action="append", default=[])
    p.add_argument(
        "--off-limits-prefix",
        action="append",
        default=None,
        help=f"branch prefixes never touched (default: {list(DEFAULT_OFF_LIMITS_PREFIXES)})",
    )
    p.add_argument(
        "--enable-repair-dispatch",
        action="store_true",
        help="SPAWN bounded repair workers for REPAIR PRs (requires --apply too); "
        "OFF by default so autonomous repair never starts accidentally",
    )
    p.add_argument("--repair-agent", default="codex", choices=["codex", "claude"])
    args = p.parse_args(argv)
    dry_run = not args.apply

    prefixes = (
        tuple(args.off_limits_prefix) if args.off_limits_prefix else DEFAULT_OFF_LIMITS_PREFIXES
    )
    ctx = DrainContext(
        off_limits_prefixes=prefixes,
        off_limits_prs=frozenset(args.off_limits_pr),
    )
    policy = DrainPassPolicy(
        drain=DrainPolicy(auto_settle_max_tier=args.auto_settle_max_tier),
        max_merges_per_pass=args.max_merges,
        max_closes_per_pass=args.max_closes,
        max_repairs_per_pass=args.max_repairs,
    )
    result = run_boss_drain(
        ctx,
        policy,
        list_open_prs_fn=lambda: list_open_prs(args.repo, args.list_limit),
        view_pr_fn=lambda n: view_pr(args.repo, n),
        merge_authorized_fn=lambda n: _proxy_authorized(view_pr(args.repo, n) or {}),
        execute_fn=make_execute_fn(
            args.repo,
            dry_run=dry_run,
            enable_repair=args.enable_repair_dispatch,
            repair_agent=args.repair_agent,
        ),
        max_classify=args.max_classify,
    )
    mode = "DRY-RUN (nothing executed)" if dry_run else "APPLIED"
    print(f"=== boss drain pass — {mode} ===")
    print(json.dumps(result.to_dict()["counts"], indent=2))
    for label, items in (("PLAN", result.planned), ("DEFERRED(next pass)", result.deferred)):
        if items:
            print(f"\n{label}:")
            for it in items:
                print(f"  #{it.pr:5d} {it.action.value:18s} {it.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
