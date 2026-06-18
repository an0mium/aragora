#!/usr/bin/env python3
"""Operator-driven PR settlement: collect -> verify quorum -> post -> tier-aware settle.

COMPOSES the existing tools instead of reimplementing the merge gate:
  - scripts/collect_quorum_evidence.py   model-quorum evidence (the authority)
  - scripts/auto_merge_quorum_green.py   Tier 0-2 unattended merge on green quorum
  - scripts/settle_tier4_pr.py           Tier 3-4 head-bound human settlement

The routing/gating decision is the pure, unit-tested
``aragora.swarm.settle_plan.plan_settlement`` -- this CLI only does I/O (gh /
subprocess) and surfaces the plan. Dry-run by default; ``--apply`` mutates.

It is AUTH-AGNOSTIC: it assumes the shell is already authenticated however the
operator does it (env keys / Secrets Manager / Vault). It needs ``gh`` plus the
reviewer keys present in the environment; it never reads, stores, or prints raw
secrets.

Tier-aware behavior under ``--apply``:
  - Tier 0-2: posts evidence (via collect --apply) then runs the existing
    auto-merge-on-green tool. Fully unattended.
  - Tier 3-4: posts the supportive evidence, then SURFACES the exact
    ``settle_tier4_pr.py`` commands for the operator to run. The Tier-4 human
    risk settlement is a deliberate operator act and is never automated here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aragora.swarm.settle_plan import (  # noqa: E402
    ROUTE_AUTO_MERGE,
    ROUTE_OPERATOR_TIER4,
    plan_settlement,
    summarize_collect,
    tier4_settle_commands,
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _resolve_repo(repo: str | None) -> str | None:
    if repo:
        return repo
    out = _run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    return out.stdout.strip() or None


def _collect(repo: str, pr: int, reviewers: list[str], *, apply: bool) -> dict[str, Any]:
    """Run collect_quorum_evidence and return its JSON payload.

    collect exits non-zero when the quorum is not met but still prints the full
    JSON, so we parse stdout regardless of the return code. A genuinely empty /
    unparseable stdout becomes an error envelope.
    """
    script = _REPO_ROOT / "scripts" / "collect_quorum_evidence.py"
    cmd = [
        sys.executable,
        str(script),
        "--repo",
        repo,
        "--pr",
        str(pr),
        "--reviewers",
        *reviewers,
        "--json",
    ]
    if apply:
        cmd.append("--apply")
    out = _run(cmd)
    try:
        payload = json.loads(out.stdout)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, ValueError):
        pass
    detail = (out.stderr or out.stdout or "collect produced no JSON").strip()
    return {"mode": "collect_evidence", "error": detail[:500]}


def _auto_merge(repo: str, pr: int) -> subprocess.CompletedProcess[str]:
    script = _REPO_ROOT / "scripts" / "auto_merge_quorum_green.py"
    return _run([sys.executable, str(script), "--repo", repo, "--pr", str(pr), "--apply"])


def _render_human(summary: dict[str, Any], plan: Any) -> None:
    print(f"\nPR settlement plan  (tier={summary['tier']}  route={plan.route})")
    print(f"  head:                  {summary.get('head_sha')}")
    print(f"  supportive_families:   {summary['supportive_families']}")
    print(f"  dissenting_families:   {summary['dissenting_families']}")
    print(f"  has_supportive_quorum: {summary['quorum_satisfied']}")
    if summary.get("failures"):
        print(f"  reviewer failures:     {summary['failures']}")
    for it in summary["items"]:
        print(
            f"  - {str(it['family']):8} verdict={str(it['verdict']):18} "
            f"would_count={it['would_count']} problems={it['problems']}"
        )
    print(f"  ready_to_mutate:       {plan.ready_to_mutate}")
    for b in plan.blockers:
        print(f"    blocker: {b}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect quorum -> verify -> tier-aware settle a PR.")
    ap.add_argument("--pr", type=int, required=True, help="PR number")
    ap.add_argument("--repo", help="owner/name (default: current gh repo)")
    ap.add_argument(
        "--reviewers",
        nargs="+",
        default=["claude", "grok"],
        help="reviewer model families (default: claude grok)",
    )
    ap.add_argument(
        "--operator-login",
        help="trusted operator GitHub login (required to surface the Tier 3-4 settle path)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="post evidence + settle (default: dry-run)"
    )
    ap.add_argument(
        "--no-app-token",
        action="store_true",
        help="surface Tier-4 merge-apply with ARAGORA_DISABLE_GITHUB_APP_TOKEN=1 "
        "(branch-protection preflight workaround)",
    )
    ap.add_argument("--json", action="store_true", help="emit a JSON summary instead of text")
    args = ap.parse_args(argv)

    repo = _resolve_repo(args.repo)
    if not repo:
        print("error: could not resolve repo (pass --repo owner/name)", file=sys.stderr)
        return 2

    # ONE collect. For Tier 0-2 + --apply, collect itself posts evidence AND runs
    # the quorum reconciler (the authority's own posting path -- we never re-post).
    # For Tier 3-4, collect treats --apply as prepare-only and refuses to post; we
    # honor that invariant and NEVER post Tier 3-4 evidence ourselves -- the human
    # settlement is surfaced, not automated.
    payload = _collect(repo, args.pr, args.reviewers, apply=args.apply)
    summary = summarize_collect(payload)
    # Authority cross-check: under --apply, collect posts (action="post") only for
    # the tiers it deems auto-settleable; "prepare" means it refused to post (Tier
    # >=3, including a recheck tier-promotion the top-level `tier` did not reflect).
    # plan_settlement re-routes such a PR to the operator path so it is never an
    # auto-merge-withheld dead-end.
    collect_prepared_only = str(summary.get("action") or "") == "prepare"

    plan = plan_settlement(
        tier=summary["tier"],
        quorum_satisfied=summary["quorum_satisfied"],
        supportive_families=summary["supportive_families"],
        head_sha=summary.get("head_sha"),
        unresolved_dissent=bool(summary["dissenting_families"]),
        operator_login_provided=bool(args.operator_login),
        authority_prepare_only=collect_prepared_only,
    )

    next_steps: list[str] = []
    actions: list[str] = []
    mutate_ok = True  # set False only if an attempted auto-merge actually fails

    if args.apply and plan.ready_to_mutate and plan.route == ROUTE_AUTO_MERGE:
        # collect --apply already posted + reconciled the Tier 0-2 evidence; merge.
        am = _auto_merge(repo, args.pr)
        mutate_ok = am.returncode == 0
        actions.append(
            f"auto_merge_quorum_green (rc={am.returncode}): "
            + (am.stdout.strip() or am.stderr.strip())[:300]
        )

    if plan.route == ROUTE_OPERATOR_TIER4 and not plan.blockers:
        # Tier 3-4: surface the operator settle commands; never post/settle here.
        next_steps = tier4_settle_commands(
            repo=repo,
            pr=args.pr,
            head=str(summary.get("head_sha") or "<head>"),
            operator_login=args.operator_login or "<gh-login>",
            no_app_token=args.no_app_token,
        )
        if collect_prepared_only:
            actions.append(
                "routed to operator settlement: collect classified prepare-only "
                f"(action_reason={summary.get('action_reason')!r}) -- the evidence "
                "authority puts this above the auto-merge tier despite the packet tier."
            )
        if args.apply:
            actions.append(
                "Tier 3-4: evidence prepared (collect refuses to auto-post it); "
                "run the surfaced commands to settle -- never automated here."
            )

    if args.json:
        print(
            json.dumps(
                {
                    "repo": repo,
                    "pr": args.pr,
                    "route": plan.route,
                    "tier": plan.tier,
                    "quorum_satisfied": plan.quorum_satisfied,
                    "ready_to_mutate": plan.ready_to_mutate,
                    "applied": bool(args.apply),
                    "mutate_ok": mutate_ok,
                    "blockers": list(plan.blockers),
                    "actions": actions,
                    "next_steps": next_steps,
                    "summary": summary,
                },
                indent=2,
            )
        )
    else:
        _render_human(summary, plan)
        if not args.apply:
            print("\n  DRY RUN — nothing mutated. Re-run with --apply to act.")
        for a in actions:
            print(f"  action: {a}")
        if next_steps:
            print("\n  Tier 3-4 operator settle path (run these yourself):")
            for s in next_steps:
                print(f"    {s}")

    # Exit codes (so an automation wrapper can tell settlement from a no-op):
    #   3  --apply on a Tier 3-4 route: the settle commands were SURFACED, not run.
    #      Operator action is still required -- this is NOT a completed settlement,
    #      so a wrapper must not advance past the human settle_tier4_pr steps.
    #   1  blocked, or an attempted Tier 0-2 auto-merge actually failed.
    #   0  actionable: a dry-run with a ready plan, or a Tier 0-2 auto-merge that
    #      succeeded.
    if args.apply and plan.route == ROUTE_OPERATOR_TIER4 and plan.ready_to_mutate:
        return 3
    return 0 if (plan.ready_to_mutate and mutate_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
