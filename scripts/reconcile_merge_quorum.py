#!/usr/bin/env python3
"""A1 — merge-quorum rerun reconciler.

Self-heals the recurring "evidence is complete but the PR is stuck" stall: the
``aragora-merge-quorum`` check only re-evaluates on ``pull_request`` synchronize
events, never on ``issue_comment``, so a check that ran before the evidence was
posted stays FAILURE forever even once the quorum is satisfied. This reconciler
re-runs that **read-only** evaluation once a strictly newer *countable* evidence
comment exists for the current head.

Defaults to ``--dry-run`` (prints the plan). With ``--apply`` it executes
``gh run rerun`` for the safe cases only. It never pushes, comments, merges, or
records settlement. Re-running a read-only evaluation cannot pass a genuinely
failing PR — it only lets the gate re-read public evidence.

Examples
--------
::

    python3 scripts/reconcile_merge_quorum.py --repo synaptent/aragora
    python3 scripts/reconcile_merge_quorum.py --repo synaptent/aragora --pr 7720 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.merge_quorum_io import (  # noqa: E402
    fetch_evidence_comments,
    fetch_latest_quorum_run,
    fetch_merge_packet_classification,
    fetch_pr_context,
    fetch_quorum_run_packet_classification,
    list_open_prs,
    run,
)
from aragora.swarm.merge_quorum_reconcile import (  # noqa: E402
    QuorumRun,
    RerunDecision,
    guard_rerun_classification_divergence,
    parse_iso8601,
    plan_rerun,
)

DEFAULT_STATE_FILE = Path.home() / ".aragora" / "merge_quorum_reconcile_state.json"


_MAX_STATE_ENTRIES = 500


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _prune_state(state: dict[str, Any]) -> dict[str, Any]:
    if len(state) <= _MAX_STATE_ENTRIES:
        return state
    items = sorted(
        state.items(),
        key=lambda kv: parse_iso8601((kv[1] or {}).get("last_rerun_at")) or _EPOCH,
        reverse=True,
    )
    return dict(items[:_MAX_STATE_ENTRIES])


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_prune_state(state), indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fh:
            # Capture the path before writing so a write failure still cleans up.
            tmp = Path(fh.name)
            fh.write(payload)
        os.replace(tmp, path)
        tmp = None  # consumed by os.replace
    except OSError as exc:
        print(f"warning: could not persist state to {path}: {exc}", file=sys.stderr)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def evaluate_pr(
    repo: str,
    pr: int,
    *,
    now: datetime,
    state: dict[str, Any],
    cooldown_seconds: int,
    max_reruns: int,
) -> tuple[RerunDecision, QuorumRun | None]:
    ctx = fetch_pr_context(repo, pr)
    head_sha = ctx["head_sha"]
    quorum_run = fetch_latest_quorum_run(repo, head_sha)
    comments = fetch_evidence_comments(repo, pr, head_sha, ctx["head_committed_at"])
    head_state = state.get(head_sha, {})
    decision = plan_rerun(
        pr_number=pr,
        run=quorum_run,
        comments=comments,
        current_head_sha=head_sha,
        now=now,
        last_rerun_at=parse_iso8601(head_state.get("last_rerun_at")),
        reruns_this_head=int(head_state.get("count", 0)),
        cooldown_seconds=cooldown_seconds,
        max_reruns_per_head=max_reruns,
        has_real_required_failure=ctx["has_real_required_failure"],
    )
    if decision.should_rerun and quorum_run is not None:
        ci_packet = fetch_quorum_run_packet_classification(
            repo, run_id=quorum_run.run_id, pr=pr, head_sha=head_sha
        )
        local_packet = fetch_merge_packet_classification(repo, pr)
        decision = guard_rerun_classification_divergence(
            decision,
            ci_packet=ci_packet,
            local_packet=local_packet,
            head_sha=head_sha,
        )
    return decision, quorum_run


def execute_rerun(repo: str, run_id: int) -> bool:
    try:
        proc = run(["gh", "run", "rerun", str(run_id), "--repo", repo])
    except subprocess.TimeoutExpired:
        print(f"warning: rerun timed out for run {run_id}", file=sys.stderr)
        return False
    if proc.returncode != 0:
        print(f"warning: rerun failed for run {run_id}: {proc.stderr.strip()}", file=sys.stderr)
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="merge-quorum rerun reconciler (A1)")
    parser.add_argument("--repo", required=True, help="GitHub repo slug (owner/name)")
    parser.add_argument("--pr", type=int, help="Single PR to evaluate (default: all open)")
    parser.add_argument("--author", help="Only consider PRs by this login")
    parser.add_argument("--limit", type=int, default=200, help="Max open PRs to walk")
    parser.add_argument("--cooldown-minutes", type=int, default=10)
    parser.add_argument("--max-reruns", type=int, default=3, help="Max reruns per head SHA")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--apply", action="store_true", help="Execute reruns (default: dry-run)")
    parser.add_argument("--json", action="store_true", help="Emit the plan as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    state = _load_state(args.state_file)
    prs = [args.pr] if args.pr else list_open_prs(args.repo, limit=args.limit, author=args.author)

    plan: list[dict[str, Any]] = []
    for pr in prs:
        try:
            decision, quorum_run = evaluate_pr(
                args.repo,
                pr,
                now=now,
                state=state,
                cooldown_seconds=args.cooldown_minutes * 60,
                max_reruns=args.max_reruns,
            )
        except RuntimeError as exc:
            plan.append({"pr": pr, "error": str(exc)})
            continue
        record: dict[str, Any] = {
            "pr": pr,
            "should_rerun": decision.should_rerun,
            "reason": decision.reason,
            "run_id": decision.run_id,
            "applied": False,
        }
        if decision.next_prompt:
            record["next_prompt"] = decision.next_prompt
        if decision.should_rerun and args.apply and quorum_run is not None:
            if execute_rerun(args.repo, quorum_run.run_id):
                record["applied"] = True
                head_state = state.setdefault(
                    quorum_run.head_sha, {"count": 0, "last_rerun_at": None}
                )
                head_state["count"] = int(head_state.get("count", 0)) + 1
                head_state["last_rerun_at"] = now.isoformat()
        plan.append(record)

    if args.apply:
        _save_state(args.state_file, state)

    if args.json:
        print(json.dumps({"plan": plan}, indent=2))
    else:
        for record in plan:
            if "error" in record:
                print(f"PR #{record['pr']}: ERROR {record['error']}")
                continue
            flag = "RERUN" if record["should_rerun"] else "skip"
            applied = " (applied)" if record["applied"] else ""
            print(f"PR #{record['pr']}: {flag}{applied} — {record['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
