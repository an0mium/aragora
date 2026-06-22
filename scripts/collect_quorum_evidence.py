"""B3 — collect genuine model-review evidence for the merge-quorum gate.

Runs >=2 genuine, heterogeneous model reviewers against a PR's *exact current
head*, composes each reviewer's output into an evidence comment whose heading the
canonical quorum parsers recognize, and validates every comment with the same
``review-queue evidence-lint`` parser the gate uses — before anything is posted.

Two safety invariants (enforced in :mod:`aragora.swarm.quorum_evidence`):

* **Never fabricate** — a comment is only composed from a reviewer that actually
  returned output.
* **Tier-gated posting** — only Tier 0-2 PRs may be auto-posted (with
  ``--apply``); Tier 3-4 (and unknown tier) always prepare evidence for an
  operator and never post.

Defaults to a dry run (prepares + lints, prints, posts nothing).

Examples
--------
::

    python3 scripts/collect_quorum_evidence.py --repo synaptent/aragora --pr 7720
    python3 scripts/collect_quorum_evidence.py --repo synaptent/aragora --pr 7720 \\
        --reviewers claude grok --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.quorum_evidence import DEFAULT_FAMILIES, run_collect_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name of the target repo")
    parser.add_argument("--pr", required=True, type=int, help="PR number to collect evidence for")
    parser.add_argument(
        "--reviewers",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        help=f"reviewer model families to run (default: {' '.join(DEFAULT_FAMILIES)})",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="GitHub login to simulate for evidence-lint (default: gh authenticated user)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Post evidence for Tier 0-2 PRs (Tier 3-4 always prepare-only).",
    )
    parser.add_argument(
        "--prepared-json",
        type=Path,
        default=None,
        help=(
            "Use a previously prepared collect-evidence JSON artifact instead of "
            "re-running reviewers."
        ),
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    return run_collect_cli(
        repo=args.repo,
        pr=args.pr,
        families=args.reviewers,
        author=args.author,
        apply=args.apply,
        json_output=args.json_output,
        prepared_json=args.prepared_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
