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


def _post_supportive_evidence(repo: str, pr: int, payload: dict[str, Any]) -> list[str]:
    """Post each countable item's evidence body as a PR comment. Returns the URLs."""
    posted: list[str] = []
    for item in payload.get("items") or []:
        if not item.get("would_count"):
            continue
        body = item.get("body") or ""
        if not body.strip():
            continue
        out = _run(["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body])
        if out.returncode == 0 and out.stdout.strip():
            posted.append(out.stdout.strip())
    return posted


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

    payload = _collect(repo, args.pr, args.reviewers, apply=False)
    summary = summarize_collect(payload)
    plan = plan_settlement(
        tier=summary["tier"],
        quorum_satisfied=summary["quorum_satisfied"],
        supportive_families=summary["supportive_families"],
        unresolved_dissent=bool(summary["dissenting_families"]),
        operator_login_provided=bool(args.operator_login),
    )

    next_steps: list[str] = []
    actions: list[str] = []

    if args.apply and plan.ready_to_mutate:
        if plan.route == ROUTE_AUTO_MERGE:
            # Re-collect with --apply so Tier 0-2 evidence is posted, then auto-merge.
            _collect(repo, args.pr, args.reviewers, apply=True)
            am = _auto_merge(repo, args.pr)
            actions.append(
                "auto_merge_quorum_green: " + (am.stdout.strip() or am.stderr.strip())[:300]
            )
        elif plan.route == ROUTE_OPERATOR_TIER4:
            # Prepare + post supportive evidence; surface (never auto-run) the settle path.
            prepared = _collect(repo, args.pr, args.reviewers, apply=True)
            posted = _post_supportive_evidence(repo, args.pr, prepared)
            actions.append(f"posted {len(posted)} supportive evidence comment(s)")
            head = (
                summarize_collect(prepared).get("head_sha") or summary.get("head_sha") or "<head>"
            )
            next_steps = tier4_settle_commands(
                repo=repo,
                pr=args.pr,
                head=str(head),
                operator_login=args.operator_login or "<gh-login>",
                no_app_token=args.no_app_token,
            )
    elif plan.route == ROUTE_OPERATOR_TIER4 and not plan.blockers:
        # Dry-run informational: show the operator path even without --apply.
        next_steps = tier4_settle_commands(
            repo=repo,
            pr=args.pr,
            head=str(summary.get("head_sha") or "<head>"),
            operator_login=args.operator_login or "<gh-login>",
            no_app_token=args.no_app_token,
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

    # Exit 0 if ready (or already applied), 1 if blocked.
    return 0 if plan.ready_to_mutate else 1


if __name__ == "__main__":
    raise SystemExit(main())
