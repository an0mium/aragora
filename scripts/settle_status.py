#!/usr/bin/env python3
"""A2 — settlement-status visibility CLI.

Read-only view of where a PR's merge settlement actually stands, computed
directly from linted evidence comments rather than from ``merge-packet`` — so it
stays accurate even when the gate check is failing and ``merge-packet``
short-circuits its quorum detail to an empty object.

Reports the tier, the distinct counted model-reviewer families on the current
head, whether adversarial-dogfood evidence is present, whether the operator's
``aragora/human-settlement`` status is recorded, the quorum check conclusion, and
the single next action. Performs no mutations.

Examples
--------
::

    python3 scripts/settle_status.py --repo synaptent/aragora --pr 7720
    python3 scripts/settle_status.py --repo synaptent/aragora --pr 7720 --tier 4 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.merge_quorum_io import (  # noqa: E402
    fetch_evidence_comments,
    fetch_human_settlement_present,
    fetch_pr_context,
    fetch_pr_tier,
)
from aragora.swarm.merge_quorum_reconcile import summarize_settlement  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="settlement-status visibility (A2)")
    parser.add_argument("--repo", required=True, help="GitHub repo slug (owner/name)")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument(
        "--tier",
        type=int,
        choices=range(0, 5),
        help="Override tier (default: best-effort via merge-packet)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = fetch_pr_context(args.repo, args.pr)
    head_sha = ctx["head_sha"]
    tier = args.tier if args.tier is not None else fetch_pr_tier(args.repo, args.pr)
    comments = fetch_evidence_comments(args.repo, args.pr, head_sha, ctx["head_committed_at"])
    human_settled = fetch_human_settlement_present(args.repo, head_sha)

    status = summarize_settlement(
        pr_number=args.pr,
        head_sha=head_sha,
        tier=tier,
        comments=comments,
        human_settlement_present=human_settled,
        quorum_conclusion=ctx["quorum_conclusion"],
    )

    if args.json:
        print(
            json.dumps(
                {
                    "pr": status.pr_number,
                    "head_sha": status.head_sha,
                    "tier": status.tier,
                    "counted_reviewer_ids": status.counted_reviewer_ids,
                    "signal_count": status.signal_count,
                    "has_dogfood": status.has_dogfood,
                    "human_settlement_present": status.human_settlement_present,
                    "quorum_conclusion": status.quorum_conclusion,
                    "next_action": status.next_action,
                },
                indent=2,
            )
        )
    else:
        tier_label = status.tier if status.tier is not None else "unknown"
        print(f"PR #{status.pr_number} @ {status.head_sha[:12]} | Tier {tier_label}")
        print(
            f"  counted reviewers ({status.signal_count}): "
            f"{', '.join(status.counted_reviewer_ids) or '(none)'}"
        )
        print(f"  dogfood evidence: {'yes' if status.has_dogfood else 'no'}")
        print(f"  human settlement: {'recorded' if status.human_settlement_present else 'absent'}")
        print(f"  quorum check: {status.quorum_conclusion or '(none)'}")
        print(f"  next action: {status.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
