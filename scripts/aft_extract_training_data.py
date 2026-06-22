#!/usr/bin/env python3
"""Extract PR-triage training data from this repo's decision history.

Produces a JSONL corpus for the Advocate Feasibility Test (AFT). Each row is one
operator decision on one PR:

    {
      "schema": "aft-pr-triage/0.1",
      "pr_number": 7423,
      "title": "...",
      "head_branch": "claude/...",
      "files_changed_count": 4,
      "additions": 385,
      "deletions": 2,
      "labels": [],
      "author_login": "an0mium",
      "tier_hint": "tier_4",         # heuristic from path/branch tokens
      "decision": "merged_fast",     # one of: merged_fast | closed_no_merge | open_aged
      "decided_at_utc": "2026-05-21T23:53:32Z",
      "rationale_seeds": [           # lightweight hints, NOT a full diff
        "branch starts with 'claude/'",
        "title contains 'governance'",
        "merge commit subject pattern: ..."
      ]
    }

This is intentionally low-fidelity. It captures the OBSERVABLE PR metadata, not
the full diff content (privacy + size). The Advocate Feasibility Test treats
this as the SHARPEST possible advocate-domain ground truth: every PR with a
terminal state is a real operator decision.

Privacy posture:
- Reads PR data via `gh` (already authenticated to the repo)
- Does NOT include full diff content in output
- Does NOT include comment bodies (only counts)
- Output is safe to commit if the repo is private; redact author emails if open-sourcing

Usage:
    python3 scripts/aft_extract_training_data.py \\
        --repo synaptent/aragora \\
        --since 2026-03-01 \\
        --max-prs 500 \\
        --output data/aft/pr_triage_corpus.jsonl

    # Stratify into train / holdout for the harness:
    python3 scripts/aft_extract_training_data.py --split \\
        --input data/aft/pr_triage_corpus.jsonl \\
        --train data/aft/pr_triage_train.jsonl \\
        --holdout data/aft/pr_triage_holdout.jsonl \\
        --holdout-size 50 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "aft-pr-triage/0.1"

# Heuristic Tier hints (per docs/REVIEW_AUTHORITY_PRINCIPLES.md surface mapping).
# Used as a feature, not as authoritative classification.
TIER_3_4_PATH_HINTS = (
    "aragora/markets/",
    "aragora/reputation/",
    "aragora/policy/",
    "aragora/security/",
    "aragora/auth/",
    "aragora/server/handlers/",
    "aragora/cli/commands/review_queue.py",
    ".github/workflows/",
    "docs/governance/",
)

TIER_0_PATH_HINTS = (
    "docs/",
    "README",
    "CHANGELOG",
)


@dataclass(frozen=True)
class PRDecision:
    pr_number: int
    title: str
    head_branch: str
    head_sha: str
    files_changed_count: int
    additions: int
    deletions: int
    labels: tuple[str, ...]
    author_login: str
    state: str
    created_at: str
    closed_at: str | None
    merged_at: str | None
    is_merged: bool
    files_changed: tuple[str, ...]  # paths only, not contents
    comment_count: int
    review_count: int


def run_gh(args: list[str], max_attempts: int = 4, base_delay: float = 2.0) -> str:
    """Run `gh` and return stdout; retry on transient HTTP 5xx and rate-limit errors.

    Retries with exponential backoff (base_delay * 2**attempt + small jitter).
    Raises RuntimeError after `max_attempts` failures or for non-transient errors.
    """
    import random as _random
    import time as _time

    last_err = ""
    for attempt in range(max_attempts):
        result = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        last_err = result.stderr or ""
        # Treat 5xx, rate-limit, and timeout as transient
        transient = any(
            token in last_err
            for token in (
                "502",
                "503",
                "504",
                "rate limit",
                "timed out",
                "timeout",
                "Bad Gateway",
                "stream error",
                "CANCEL",
                "connection reset",
                "EOF",
                "i/o timeout",
            )
        )
        if not transient or attempt == max_attempts - 1:
            raise RuntimeError(f"gh {args[0]} failed: {last_err[:300]}")
        delay = base_delay * (2**attempt) + _random.uniform(0, 1)
        print(
            f"gh transient failure (attempt {attempt + 1}/{max_attempts}): "
            f"{last_err.strip()[:120]}; retrying in {delay:.1f}s",
            file=sys.stderr,
        )
        _time.sleep(delay)
    raise RuntimeError(f"gh {args[0]} failed after {max_attempts} attempts: {last_err[:300]}")


def fetch_prs(
    repo: str,
    since: str,
    max_prs: int,
    states: tuple[str, ...] = ("merged", "closed", "open"),
) -> list[PRDecision]:
    """Fetch PRs from `gh` and return PRDecision objects."""
    prs: list[PRDecision] = []
    for state in states:
        # gh pr list supports --state {open,closed,merged,all}
        raw = run_gh(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                state,
                "--limit",
                str(max_prs),
                "--json",
                (
                    "number,title,headRefName,headRefOid,files,additions,deletions,"
                    "labels,author,state,createdAt,closedAt,mergedAt,comments,reviews"
                ),
            ]
        )
        for item in json.loads(raw):
            created = item.get("createdAt", "")
            if since and created and created < since:
                continue
            files_changed = tuple(f.get("path", "") for f in (item.get("files") or []))
            prs.append(
                PRDecision(
                    pr_number=item["number"],
                    title=item.get("title", "")[:200],
                    head_branch=item.get("headRefName", ""),
                    head_sha=item.get("headRefOid", ""),
                    files_changed_count=len(files_changed),
                    additions=item.get("additions", 0) or 0,
                    deletions=item.get("deletions", 0) or 0,
                    labels=tuple(
                        (lbl.get("name") or "")
                        for lbl in (item.get("labels") or [])
                        if lbl.get("name")
                    ),
                    author_login=((item.get("author") or {}).get("login") or ""),
                    state=item.get("state", "").lower(),
                    created_at=created,
                    closed_at=item.get("closedAt"),
                    merged_at=item.get("mergedAt"),
                    is_merged=bool(item.get("mergedAt")),
                    files_changed=files_changed,
                    comment_count=len(item.get("comments") or []),
                    review_count=len(item.get("reviews") or []),
                )
            )
    # Deduplicate (gh sometimes returns same PR across states for edge cases)
    seen = set()
    out: list[PRDecision] = []
    for pr in sorted(prs, key=lambda p: p.pr_number, reverse=True):
        if pr.pr_number in seen:
            continue
        seen.add(pr.pr_number)
        out.append(pr)
    return out


def classify_decision(pr: PRDecision, now_utc: datetime) -> str:
    """Map PR state + timing to a 3-class triage label.

    - merged_fast: merged within 14 days of creation
    - closed_no_merge: closed without merge
    - open_aged: still open AND >14 days old (operator is deferring)

    Anything else (open <14d, merged >14d) is skipped — the signal is noisy.
    """
    if pr.is_merged and pr.merged_at and pr.created_at:
        created = datetime.fromisoformat(pr.created_at.replace("Z", "+00:00"))
        merged = datetime.fromisoformat(pr.merged_at.replace("Z", "+00:00"))
        if (merged - created).days <= 14:
            return "merged_fast"
        return ""  # merged but slowly — exclude as noisy
    if pr.state == "closed" and not pr.is_merged:
        return "closed_no_merge"
    if pr.state == "open" and pr.created_at:
        created = datetime.fromisoformat(pr.created_at.replace("Z", "+00:00"))
        if (now_utc - created).days >= 14:
            return "open_aged"
    return ""


def tier_hint(pr: PRDecision) -> str:
    """Heuristic Tier classification from paths + branch tokens."""
    paths = pr.files_changed
    for p in paths:
        if any(p.startswith(h) for h in TIER_3_4_PATH_HINTS):
            return "tier_3_or_4"
    # Branch tokens for governance/security/secrets:
    br_low = pr.head_branch.lower()
    if any(tok in br_low for tok in ("adc", "governance", "security", "secret", "auth")):
        return "tier_3_or_4"
    if paths and all(any(p.startswith(h) for h in TIER_0_PATH_HINTS) for p in paths):
        return "tier_0"
    return "tier_1_or_2"


def rationale_seeds(pr: PRDecision) -> list[str]:
    """Surface lightweight hints WITHOUT including diff content.

    These are intentionally low-information observable cues — the AFT measures
    whether an advocate can outperform a frontier model given the same observable
    features. Including full diffs would defeat the test.
    """
    seeds: list[str] = []
    br = pr.head_branch
    if "/" in br:
        seeds.append(f"branch_namespace={br.split('/', 1)[0]}")
    title_low = pr.title.lower()
    for tok in ("governance", "security", "deps", "fix", "feat", "docs", "ci", "test"):
        if tok in title_low:
            seeds.append(f"title_token={tok}")
            break
    if pr.labels:
        seeds.append(f"label_count={len(pr.labels)}")
    if pr.review_count > 0:
        seeds.append(f"has_reviews={pr.review_count}")
    if pr.comment_count > 0:
        seeds.append(f"comment_count={pr.comment_count}")
    seeds.append(f"diff_size={pr.additions + pr.deletions}")
    seeds.append(f"file_count={pr.files_changed_count}")
    return seeds


def to_corpus_row(pr: PRDecision, decision: str) -> dict[str, Any]:
    decided_at = pr.merged_at or pr.closed_at or ""
    return {
        "schema": SCHEMA_VERSION,
        "pr_number": pr.pr_number,
        "title": pr.title,
        "head_branch": pr.head_branch,
        "head_sha_short": pr.head_sha[:12] if pr.head_sha else "",
        "files_changed_count": pr.files_changed_count,
        "additions": pr.additions,
        "deletions": pr.deletions,
        "labels": list(pr.labels),
        "author_login": pr.author_login,
        "tier_hint": tier_hint(pr),
        "decision": decision,
        "decided_at_utc": decided_at,
        "created_at_utc": pr.created_at,
        "comment_count": pr.comment_count,
        "review_count": pr.review_count,
        "rationale_seeds": rationale_seeds(pr),
    }


def extract(repo: str, since: str, max_prs: int, output: Path) -> int:
    now_utc = datetime.now(timezone.utc)
    prs = fetch_prs(repo, since, max_prs)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for pr in prs:
        decision = classify_decision(pr, now_utc)
        if not decision:
            skipped += 1
            continue
        rows.append(to_corpus_row(pr, decision))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
    # Class balance report
    from collections import Counter

    cls = Counter(r["decision"] for r in rows)
    print(
        f"wrote {len(rows)} rows to {output} "
        f"(skipped {skipped} noisy/unclassifiable); class balance: {dict(cls)}",
        file=sys.stderr,
    )
    return len(rows)


def stratified_split(
    input_path: Path,
    train_path: Path,
    holdout_path: Path,
    holdout_size: int,
    seed: int,
) -> tuple[int, int]:
    """Split corpus into train + holdout, stratified by decision class."""
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    from collections import defaultdict

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["decision"]].append(row)

    rng = random.Random(seed)
    holdout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    classes = sorted(by_class.keys())
    per_class_holdout = max(1, holdout_size // max(1, len(classes)))
    for cls in classes:
        bucket = by_class[cls][:]
        rng.shuffle(bucket)
        n = min(per_class_holdout, len(bucket) // 2)  # never take more than half
        holdout.extend(bucket[:n])
        train.extend(bucket[n:])
    rng.shuffle(holdout)
    rng.shuffle(train)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as fh:
        for row in train:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
    with holdout_path.open("w", encoding="utf-8") as fh:
        for row in holdout:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
    print(
        f"split: train={len(train)} -> {train_path}, holdout={len(holdout)} -> {holdout_path}",
        file=sys.stderr,
    )
    return len(train), len(holdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    extract_parser = sub.add_parser(
        "extract", help="Extract PR-triage corpus from a repo (default)."
    )
    extract_parser.add_argument("--repo", default="synaptent/aragora")
    extract_parser.add_argument(
        "--since", default=(datetime.now(timezone.utc) - timedelta(days=120)).date().isoformat()
    )
    extract_parser.add_argument("--max-prs", type=int, default=500)
    extract_parser.add_argument(
        "--output", type=Path, default=Path("data/aft/pr_triage_corpus.jsonl")
    )

    split_parser = sub.add_parser("split", help="Stratified split into train/holdout.")
    split_parser.add_argument("--input", type=Path, default=Path("data/aft/pr_triage_corpus.jsonl"))
    split_parser.add_argument("--train", type=Path, default=Path("data/aft/pr_triage_train.jsonl"))
    split_parser.add_argument(
        "--holdout", type=Path, default=Path("data/aft/pr_triage_holdout.jsonl")
    )
    split_parser.add_argument("--holdout-size", type=int, default=50)
    split_parser.add_argument("--seed", type=int, default=42)

    # If no subcommand given, default to extract for ergonomics.
    args, _ = parser.parse_known_args()
    if not args.cmd:
        args = extract_parser.parse_args()
        args.cmd = "extract"

    if args.cmd == "extract":
        extract(args.repo, args.since, args.max_prs, args.output)
    elif args.cmd == "split":
        stratified_split(args.input, args.train, args.holdout, args.holdout_size, args.seed)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
