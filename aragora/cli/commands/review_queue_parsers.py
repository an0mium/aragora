"""Lightweight parser registration helpers for review-queue subcommands."""

from __future__ import annotations

import argparse


def add_record_settlement_parser(sub: argparse._SubParsersAction) -> None:
    record_p = sub.add_parser(
        "record-settlement",
        help="Record an already-authorized PR settlement without mutating GitHub",
        description=(
            "Write a local review-queue settlement receipt after an external\n"
            "operator decision, such as an exact-head admin squash merge. This\n"
            "is local-only: it verifies the live PR head/state, writes under\n"
            ".aragora/review-queue/receipts (or --review-queue-root), and does\n"
            "not approve, comment, merge, or otherwise mutate GitHub."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    record_p.add_argument("pr", help="PR number or URL")
    record_p.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )
    record_p.add_argument(
        "--head-sha",
        required=True,
        help="Exact PR head SHA that was externally settled.",
    )
    record_p.add_argument(
        "--action",
        required=True,
        choices=("approve", "request_changes", "comment", "admin_squash_merge"),
        help="Externally observed settlement action to record.",
    )
    record_p.add_argument(
        "--reason",
        required=True,
        help="One-line operator reason or authorization reference.",
    )
    record_p.add_argument(
        "--review-queue-root",
        default=None,
        help=(
            "Override the review-queue store root used for settlement "
            "receipts. Defaults to <repo>/.aragora/review-queue."
        ),
    )
    record_p.add_argument(
        "--apply-post-merge-lane-audit",
        action="store_true",
        help=(
            "For admin_squash_merge records, apply merged-PR lane supersession "
            "after verifying GitHub reports MERGED and using the live merge "
            "commit as the exact guard. Default is dry-run/report only."
        ),
    )
    record_p.add_argument(
        "--post-github-status",
        action="store_true",
        help=(
            "After the local receipt is durably written, atomically POST the "
            "'aragora/human-settlement'=success commit status for the exact head "
            "SHA. This is the ONLY GitHub mutation this command performs, and it "
            "is the safe replacement for running 'gh api ... statuses' as a "
            "separate command: the status can never be set unless the receipt "
            "write succeeded first (receipt => status, never status => no "
            "receipt). If the receipt write fails, the status is never posted."
        ),
    )
    record_p.add_argument(
        "--github-status-context",
        default="aragora/human-settlement",
        help="Commit-status context to post with --post-github-status.",
    )
    record_p.add_argument("--json", action="store_true", help="Output local receipt as JSON")


def add_observe_outcomes_parser(sub: argparse._SubParsersAction) -> None:
    """Register observe-outcomes without importing its heavy implementation."""

    p = sub.add_parser(
        "observe-outcomes",
        help=(
            "Observe post-settlement invalidation signals from GitHub "
            "timeline events and (optionally) write them back into v2 "
            "outcome fields on settlement receipts. Dry-run by default."
        ),
        description=(
            "Round 30g phase A. Iterates settled receipts in a bounded\n"
            "window, fetches GitHub timeline events for each PR with\n"
            "bounded fanout, and computes the five canonical v2 outcome\n"
            "signals via aragora.review.settlement_outcome.observe_outcome.\n\n"
            "Default mode is read-only: nothing is written. Pass --write\n"
            "to mutate receipt JSON files in place. The CLI never invokes\n"
            "git or gh write operations and never edits docs/THESIS.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Observation window in days (default: 14).",
    )
    p.add_argument(
        "--max-receipts",
        type=int,
        default=20,
        help="Maximum receipts to inspect in this run (default: 20). Bounds the GitHub fanout.",
    )
    p.add_argument(
        "--per-receipt-event-cap",
        type=int,
        default=100,
        help="Maximum timeline events fetched per receipt (default: 100).",
    )
    p.add_argument(
        "--review-queue-root",
        default=None,
        help=(
            "Override the review-queue store root used for settlement "
            "receipts. Defaults to <repo>/.aragora/review-queue."
        ),
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "OPT-IN: actually write v2 outcome fields back into receipt "
            "JSON files. Default is dry-run preview only."
        ),
    )
    p.add_argument("--json", action="store_true", help="Output the run summary as JSON.")
