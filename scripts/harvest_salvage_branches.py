#!/usr/bin/env python3
"""Phase 4 of cleanup plan: salvage value extraction from old branches.

Reads the Phase 2 classification JSON, finds every branch categorised as
salvage_*_unique, and for each one:

  - Computes diff stat (LOC added/removed, files touched) and commit log
  - Classifies trivial/no-op branches (whitespace-only, .lock-only, etc.)
  - Classifies candidates that match a known archetype (single-purpose,
    bounded LOC, includes tests, clean commit msg)
  - Routes everything else to .aragora/cleanup-state/salvage-review-queue.jsonl
    for human triage

Conservative by default: this script never opens PRs or deletes branches.
It writes decision artifacts for operator-reviewed follow-up.

Usage:
    python3 scripts/harvest_salvage_branches.py \\
        --classification .aragora/cleanup-state/branch-classification.json \\
        [--review-queue .aragora/cleanup-state/salvage-review-queue.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_codex_branch_backlog import run_git  # noqa: E402

UTC = timezone.utc

# Auto-discard heuristics: branch is treated as no-value if any apply.
TRIVIAL_FILE_PATTERNS = [
    re.compile(r"\.lock$"),
    re.compile(r"\.pyc$"),
    re.compile(r"^__pycache__/"),
    re.compile(r"\.DS_Store$"),
    re.compile(r"\.gitignore$"),
]

TRIVIAL_BRANCH_NAME_PATTERNS = [
    re.compile(r"-empty-diff-cleanup"),
    re.compile(r"-rebase-noop"),
    re.compile(r"-noop-"),
    re.compile(r"-empty-commit"),
]


def _diff_stat(root: Path, base: str, branch: str) -> dict[str, Any]:
    """Return diff summary: added/removed LOC, file count, file list."""
    proc = run_git(["diff", "--numstat", f"{base}...{branch}"], root, timeout=60)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip(), "files": [], "added": 0, "removed": 0}
    added = removed = 0
    files: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, r, fpath = parts
        try:
            ai = int(a) if a != "-" else 0
            ri = int(r) if r != "-" else 0
        except ValueError:
            continue
        added += ai
        removed += ri
        files.append({"path": fpath, "added": ai, "removed": ri})
    return {"added": added, "removed": removed, "files": files}


def _commit_log(root: Path, base: str, branch: str) -> list[dict[str, str]]:
    proc = run_git(
        ["log", "--format=%H%n%s%n%b%n----", f"{base}..{branch}"],
        root,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    commits: list[dict[str, str]] = []
    chunks = proc.stdout.split("----\n")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.split("\n", 2)
        if len(lines) < 2:
            continue
        sha = lines[0]
        subject = lines[1] if len(lines) > 1 else ""
        body = lines[2] if len(lines) > 2 else ""
        commits.append({"sha": sha, "subject": subject, "body": body.strip()})
    return commits


def _is_trivial_diff(diff_stat: dict[str, Any], branch_name: str) -> tuple[bool, str]:
    """Return (is_trivial, reason)."""
    if any(p.search(branch_name) for p in TRIVIAL_BRANCH_NAME_PATTERNS):
        return True, "branch name matches no-op pattern"

    if diff_stat.get("error"):
        return False, ""  # treat errors as non-trivial; let operator review

    files = diff_stat.get("files", [])
    if not files:
        return True, "no files changed (empty diff)"

    if all(any(p.search(f["path"]) for p in TRIVIAL_FILE_PATTERNS) for f in files):
        return True, "all files match trivial patterns (.lock, .pyc, etc.)"

    total_changed = diff_stat.get("added", 0) + diff_stat.get("removed", 0)
    if total_changed == 0:
        return True, "zero net LOC changed"

    return False, ""


def _matches_auto_pr_archetype(
    diff_stat: dict[str, Any],
    commits: list[dict[str, str]],
) -> tuple[bool, str]:
    """Return (is_auto_pr_candidate, archetype_reason).

    Conservative: only auto-PR when we're highly confident the change is
    self-contained and operator-reviewable.
    """
    files = diff_stat.get("files", [])
    if not files:
        return False, ""

    if len(commits) > 3:
        return False, "more than 3 commits — likely multi-purpose"

    total_changed = diff_stat.get("added", 0) + diff_stat.get("removed", 0)
    if total_changed > 200:
        return False, "diff too large (>200 LOC) for auto-PR"
    if total_changed < 5:
        return False, "diff too small (<5 LOC) — no clear value"

    paths = [f["path"] for f in files]
    file_count = len(paths)

    # Archetype: docs-only update
    if all(p.startswith("docs/") or p.endswith(".md") for p in paths):
        return True, "docs-only update"

    # Archetype: tests-only addition
    if all(p.startswith("tests/") for p in paths) and file_count <= 3:
        return True, "tests-only addition"

    # Archetype: single-file bug fix with test
    src_files = [p for p in paths if not p.startswith("tests/") and not p.startswith("docs/")]
    test_files = [p for p in paths if p.startswith("tests/")]
    if len(src_files) == 1 and len(test_files) <= 1 and total_changed < 100:
        return True, "single-file fix" + (" with test" if test_files else "")

    return False, "no matching auto-PR archetype"


def _process_branch(
    *,
    record: dict[str, Any],
    root: Path,
    base: str,
) -> dict[str, Any]:
    branch = record.get("branch") or record.get("name")
    if not branch:
        raise KeyError("classification record is missing branch/name")
    diff_stat = _diff_stat(root, base, branch)
    commits = _commit_log(root, base, branch)

    decision: str
    reason: str
    auto_pr_archetype = ""

    is_trivial, trivial_reason = _is_trivial_diff(diff_stat, branch)
    if is_trivial:
        decision = "auto_discard"
        reason = trivial_reason
    else:
        is_auto_pr, auto_pr_archetype = _matches_auto_pr_archetype(diff_stat, commits)
        if is_auto_pr:
            decision = "auto_pr"
            reason = f"matches archetype: {auto_pr_archetype}"
        else:
            decision = "operator_review"
            reason = "non-trivial diff, no clear archetype — operator decides"

    return {
        "branch": branch,
        "head_sha": record.get("head_sha"),
        "category": record.get("category"),
        "added": diff_stat.get("added", 0),
        "removed": diff_stat.get("removed", 0),
        "file_count": len(diff_stat.get("files", [])),
        "first_3_files": [f["path"] for f in diff_stat.get("files", [])[:3]],
        "commit_count": len(commits),
        "first_commit_subject": commits[0]["subject"] if commits else "",
        "decision": decision,
        "reason": reason,
        "auto_pr_archetype": auto_pr_archetype,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Main repository root (NOT a worktree). Reads classification "
            "JSON from this path's .aragora/cleanup-state/ and writes the "
            "salvage decision artifacts there."
        ),
    )
    parser.add_argument("--base", default="origin/main")
    parser.add_argument(
        "--classification",
        default=".aragora/cleanup-state/branch-classification.json",
    )
    parser.add_argument(
        "--review-queue",
        default=".aragora/cleanup-state/salvage-review-queue.jsonl",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
        help="Cap branches processed (testing)",
    )
    args = parser.parse_args(argv)

    root = Path(args.repo).resolve()
    classification_path = root / args.classification
    if not classification_path.exists():
        print(f"ERROR: classification file not found: {classification_path}")
        return 1

    payload = json.loads(classification_path.read_text())
    records = payload.get("records", [])

    salvage_records = [r for r in records if r.get("category", "").startswith("salvage_")]
    if args.max_branches:
        salvage_records = salvage_records[: args.max_branches]

    print(f"loaded {len(records)} total records")
    print(f"  salvage_*_unique: {len(salvage_records)}")
    print("  mode: classify-only (no branch deletion, no PR creation)\n")

    decisions: list[dict[str, Any]] = []
    for i, r in enumerate(salvage_records, 1):
        d = _process_branch(
            record=r,
            root=root,
            base=args.base,
        )
        decisions.append(d)
        if i % 50 == 0:
            print(f"  processed {i}/{len(salvage_records)}")

    counts = {"auto_discard": 0, "auto_pr": 0, "operator_review": 0}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1

    print("\n--- summary ---")
    for k, v in counts.items():
        print(f"  {k:>20}: {v}")

    review_queue_path = root / args.review_queue
    review_queue_path.parent.mkdir(parents=True, exist_ok=True)
    with review_queue_path.open("w") as f:
        for d in decisions:
            if d["decision"] == "operator_review":
                f.write(json.dumps(d) + "\n")
    print(f"\n  review queue: {review_queue_path} ({counts['operator_review']} entries)")

    state_dir = root / ".aragora" / "cleanup-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    out = state_dir / f"salvage-decisions-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {"counts": counts, "decisions": decisions, "mode": "classify_only"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"  full report: {out}")

    print(
        "  Operator-review queue items: review and turn valuable entries into focused PRs "
        "or mark them discarded with evidence."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
