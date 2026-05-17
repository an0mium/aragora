#!/usr/bin/env python3
"""Read-only four-bucket PR triage classifier.

Implements ``docs/governance/OPERATOR_DELEGATION_POLICY.md`` (Stage 1
of the rollout per ``docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md``).

For every open PR on the current repo, classifies it into exactly one
of four buckets:

  A — recommend auto-merge (additive + green CI + mergeable + tests +
      no held/protected touch + ≤1500 LOC + trusted author)
  B — recommend auto-close (superseded by newer / >60d stale +
      inactive / CI red >7d)
  C — needs operator y/n (touches held / protected / large diff /
      CI red / CI pending / non-trusted author / unresolved review /
      no tests with code changes / merge-state not clean)
  D — strategic check-in (PR works but plausibly conflicts with
      canonical direction; not auto-classifiable from gh metadata
      alone — reserved for future enhancement)

The classifier is read-only by design: it NEVER mutates GitHub state.
The downstream Stage 2 (``scripts/auto_merge_bucket_a.py``) is what
acts on Bucket A; this script only emits the recommendation table.

Pure stdlib (argparse, dataclasses, datetime, json, shutil,
subprocess, sys, pathlib, typing). No ``aragora.*`` imports. No
third-party dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Policy constants — keep in sync with docs/governance/OPERATOR_DELEGATION_POLICY.md
# ---------------------------------------------------------------------------

# Held PRs (from the policy's canonical hold list). Whoever updates the
# policy doc's hold list MUST update both this set and the equivalent in
# ``scripts/apply_operator_decisions.py``.
HELD_PR_NUMBERS: frozenset[int] = frozenset({4990, 7173, 7215, 7240, 7243, 7245, 7249, 7252})

# Protected file paths — from the policy's irreducible tripwire list.
PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "CLAUDE.md",
        "aragora/__init__.py",
        ".env",
        ".envrc",
        "scripts/nomic_loop.py",
        "docs/AGENT_OPERATING_CONTRACT.md",
        "automation.toml",
    }
)

# Trusted authors — Bucket A is gated on author membership here. Adding
# a new entry is itself an operator-only tripwire (the policy doc names
# this explicitly).
TRUSTED_AUTHORS: frozenset[str] = frozenset({"an0mium"})

# Bucket A diff-size cap. PRs above this go to Bucket C regardless of
# other criteria — large diffs trip more invariants than the
# metadata-only classifier can verify.
LARGE_DIFF_LOC = 1500

# Auto-close thresholds (days). Both must hold for the stale-draft
# Bucket B path.
STALE_AGE_DAYS = 60
STALE_INACTIVITY_DAYS = 30

# CI-red threshold (days) for the auto-close path.
CI_RED_THRESHOLD_DAYS = 7

# File-overlap threshold for the supersede path. The policy doc names 0.8.
SUPERSEDE_OVERLAP_THRESHOLD = 0.80

BUCKET_A = "A"
BUCKET_B = "B"
BUCKET_C = "C"
BUCKET_D = "D"  # currently never auto-emitted; reserved (see module docstring)

_BUCKET_LABELS: dict[str, str] = {
    BUCKET_A: "BUCKET A — recommend AUTO-MERGE",
    BUCKET_B: "BUCKET B — recommend AUTO-CLOSE",
    BUCKET_C: "BUCKET C — needs operator y/n",
    BUCKET_D: "BUCKET D — strategic check-in",
}


# ---------------------------------------------------------------------------
# Result + classification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    """One PR's classification. Bucket + ≤120-char justification + the
    recommended human action (MERGE / CLOSE / DEFER / DECIDE / STAY HELD).
    """

    pr_number: int
    bucket: str
    reason: str
    title: str
    recommended_action: str


def _result(
    pr_number: int,
    title: str,
    bucket: str,
    reason: str,
    recommended_action: str,
) -> ClassificationResult:
    return ClassificationResult(
        pr_number=pr_number,
        bucket=bucket,
        reason=reason[:200],  # cap reason length defensively
        title=title[:80],
        recommended_action=recommended_action,
    )


def _is_protected(path: str) -> bool:
    return path in PROTECTED_PATHS


def _is_test_file(path: str) -> bool:
    if path.startswith("tests/"):
        return True
    if "/__tests__/" in path:
        return True
    if path.endswith((".test.tsx", ".test.ts", ".test.jsx", ".test.js")):
        return True
    if path.endswith("_test.py"):
        return True
    return False


def _is_code_file(path: str) -> bool:
    if _is_test_file(path):
        return False
    if path.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs")):
        return True
    return False


def _parse_age_days(iso: str, now: datetime.datetime) -> int:
    """Days between ``iso`` and ``now``; 0 on parse failure or empty."""

    if not iso:
        return 0
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return max(0, (now - dt).days)


def _file_paths(pr: dict[str, Any]) -> list[str]:
    files = pr.get("files") or []
    return [str(f.get("path", "")) for f in files if isinstance(f, dict) and f.get("path")]


def _find_superseder(pr: dict[str, Any], all_open: list[dict[str, Any]]) -> int | None:
    """Return the PR number of a newer open PR that supersedes ``pr``.

    Supersede = file-overlap ≥ ``SUPERSEDE_OVERLAP_THRESHOLD`` AND newer
    (higher PR number) AND zero CI failures on the newer PR. (We can't
    see merged PRs from the open list, so we use "newer + clean" as a
    proxy for "in Bucket A or about to merge.")
    """

    pr_n = int(pr.get("number", 0))
    if pr_n <= 0:
        return None
    pr_files = set(_file_paths(pr))
    if not pr_files:
        return None
    for other in all_open:
        other_n = int(other.get("number", 0) or 0)
        if other_n <= pr_n:
            continue
        other_files = set(_file_paths(other))
        if not other_files:
            continue
        overlap_count = len(pr_files & other_files)
        overlap_ratio = overlap_count / len(pr_files)
        if overlap_ratio < SUPERSEDE_OVERLAP_THRESHOLD:
            continue
        other_failures = sum(
            1
            for c in (other.get("statusCheckRollup") or [])
            if isinstance(c, dict) and c.get("conclusion") == "FAILURE"
        )
        if other_failures > 0:
            continue
        return other_n
    return None


def classify(
    pr: dict[str, Any],
    all_open: list[dict[str, Any]],
    *,
    now: datetime.datetime | None = None,
) -> ClassificationResult:
    """Run the four-bucket classification on one PR.

    Bucket precedence (most-restrictive wins): C (held) → C (protected) →
    C (large) → B (CI red 7d+) → C (CI red recent) → C (CI pending) →
    B (stale draft) → B (superseded) → C (non-trusted) → C (not
    mergeable) → C (merge-state DIRTY/BEHIND/etc) → C (code without
    tests) → C (CHANGES_REQUESTED) → A (default if all gates pass).

    Bucket D is never auto-emitted by this classifier — it's reserved
    for explicit operator/agent escalation in a future stage.
    """

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    n = int(pr.get("number") or 0)
    title = str(pr.get("title") or "")
    author_raw = pr.get("author") or {}
    author = author_raw.get("login", "") if isinstance(author_raw, dict) else str(author_raw)
    file_paths = _file_paths(pr)
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    net_loc = additions + deletions
    mergeable = str(pr.get("mergeable") or "")
    mss = str(pr.get("mergeStateStatus") or "")
    is_draft = bool(pr.get("isDraft"))
    checks = pr.get("statusCheckRollup") or []
    ci_success = sum(1 for c in checks if isinstance(c, dict) and c.get("conclusion") == "SUCCESS")
    ci_failure = sum(1 for c in checks if isinstance(c, dict) and c.get("conclusion") == "FAILURE")
    ci_pending = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") in ("IN_PROGRESS", "QUEUED")
    )
    ci_total = len(checks)
    age_days = _parse_age_days(str(pr.get("createdAt") or ""), now)
    updated_days = _parse_age_days(str(pr.get("updatedAt") or ""), now)
    review = str(pr.get("reviewDecision") or "")

    # --- Bucket C: held PR ---
    if n in HELD_PR_NUMBERS:
        return _result(n, title, BUCKET_C, f"held (#{n} is on the policy hold list)", "STAY HELD")

    # --- Bucket C: protected file edits ---
    protected_hit = [p for p in file_paths if _is_protected(p)]
    if protected_hit:
        return _result(
            n,
            title,
            BUCKET_C,
            f"edits protected file ({protected_hit[0]})",
            "DECIDE",
        )

    # --- Bucket C: large diff ---
    if net_loc > LARGE_DIFF_LOC:
        return _result(
            n,
            title,
            BUCKET_C,
            f"large diff ({net_loc} LOC > {LARGE_DIFF_LOC})",
            "DECIDE",
        )

    # --- Bucket B: CI red ≥7 days (use updated_days as a proxy for "no
    # recent fix attempts" since the rollup doesn't carry per-check age) ---
    if ci_failure > 0 and updated_days >= CI_RED_THRESHOLD_DAYS:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"CI red ≥{CI_RED_THRESHOLD_DAYS}d ({ci_failure} failures, "
                f"{updated_days}d since update)"
            ),
            "CLOSE",
        )

    # --- Bucket C: CI red but recent ---
    if ci_failure > 0:
        return _result(n, title, BUCKET_C, f"CI red ({ci_failure} failures)", "DECIDE")

    # --- Bucket C: CI pending ---
    if ci_pending > 0:
        return _result(
            n,
            title,
            BUCKET_C,
            f"CI pending ({ci_pending} in-flight, {ci_success}/{ci_total} green)",
            "DEFER",
        )

    # --- Bucket B: stale draft ---
    if is_draft and age_days >= STALE_AGE_DAYS and updated_days >= STALE_INACTIVITY_DAYS:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"stale draft ({age_days}d old, {updated_days}d inactive — "
                f"thresholds {STALE_AGE_DAYS}/{STALE_INACTIVITY_DAYS})"
            ),
            "CLOSE",
        )

    # --- Bucket B: superseded by newer green PR ---
    superseder = _find_superseder(pr, all_open)
    if superseder is not None:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"superseded by #{superseder} "
                f"(≥{int(SUPERSEDE_OVERLAP_THRESHOLD * 100)}% file overlap, "
                f"newer + zero CI failures)"
            ),
            "CLOSE",
        )

    # --- Bucket C: non-trusted author ---
    if author not in TRUSTED_AUTHORS:
        return _result(
            n,
            title,
            BUCKET_C,
            f"non-trusted author ({author or '(unknown)'})",
            "DECIDE",
        )

    # --- Bucket C: not mergeable ---
    if mergeable != "MERGEABLE":
        return _result(
            n,
            title,
            BUCKET_C,
            f"not mergeable (mergeable={mergeable or '(unknown)'})",
            "DECIDE",
        )

    # --- Bucket C: merge-state status not CLEAN/BLOCKED ---
    # CLEAN = ready to merge; BLOCKED = often just "draft state" or
    # branch-protection waiting (still safe to auto-merge once ready).
    # DIRTY = conflicts; BEHIND = needs rebase; UNSTABLE = passing
    # but flaky. All of those need a human look.
    if mss and mss not in ("CLEAN", "BLOCKED"):
        return _result(
            n,
            title,
            BUCKET_C,
            f"merge state status: {mss}",
            "DECIDE",
        )

    # --- Bucket C: code changes without tests ---
    has_tests = any(_is_test_file(p) for p in file_paths)
    has_code = any(_is_code_file(p) for p in file_paths)
    if has_code and not has_tests:
        return _result(
            n,
            title,
            BUCKET_C,
            f"code changes without test files ({len(file_paths)} files touched)",
            "DECIDE",
        )

    # --- Bucket C: changes requested in review ---
    if review == "CHANGES_REQUESTED":
        return _result(
            n,
            title,
            BUCKET_C,
            "review decision: CHANGES_REQUESTED",
            "DECIDE",
        )

    # --- Bucket A: default if all gates pass ---
    return _result(
        n,
        title,
        BUCKET_A,
        (
            f"green CI ({ci_success}/{ci_total}), {net_loc} LOC, "
            f"{len(file_paths)} files, tests present, author={author}"
        ),
        "MERGE",
    )


# ---------------------------------------------------------------------------
# I/O — gh shell-out + output formatting
# ---------------------------------------------------------------------------


_GH_JSON_FIELDS = (
    "number,title,isDraft,author,mergeable,mergeStateStatus,additions,"
    "deletions,changedFiles,createdAt,updatedAt,headRefName,"
    "statusCheckRollup,reviewDecision,labels,files"
)


def fetch_open_prs(*, limit: int = 100) -> list[dict[str, Any]]:
    """Shell out to ``gh pr list`` with the field set the classifier needs."""

    cmd = [
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "-L",
        str(limit),
        "--json",
        _GH_JSON_FIELDS,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise SystemExit(f"gh pr list failed: {stderr[:300]}") from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gh pr list returned non-JSON: {exc}") from exc


def _print_human(results: Sequence[ClassificationResult]) -> None:
    by_bucket: dict[str, list[ClassificationResult]] = {
        BUCKET_A: [],
        BUCKET_B: [],
        BUCKET_C: [],
        BUCKET_D: [],
    }
    for r in results:
        by_bucket.setdefault(r.bucket, []).append(r)

    for bucket in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D):
        entries = sorted(by_bucket[bucket], key=lambda r: r.pr_number)
        print(_BUCKET_LABELS[bucket])
        if not entries:
            print("  (none)")
        else:
            for r in entries:
                print(f"  #{r.pr_number} — {r.recommended_action} — {r.reason}")
        print()

    summary = "  ".join(
        f"{b}: {len(by_bucket[b])}" for b in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D)
    )
    print(f"summary: {summary}    total: {len(results)}")


def _print_json(results: Sequence[ClassificationResult]) -> None:
    print(
        json.dumps(
            {
                "policy_doc": ("docs/governance/OPERATOR_DELEGATION_POLICY.md"),
                "rollout_doc": ("docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md"),
                "results": [
                    dataclasses.asdict(r)
                    for r in sorted(results, key=lambda r: (r.bucket, r.pr_number))
                ],
                "summary": {
                    b: sum(1 for r in results if r.bucket == b)
                    for b in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage_open_prs.py",
        description=(
            "Read-only four-bucket PR triage classifier per "
            "docs/governance/OPERATOR_DELEGATION_POLICY.md."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON (default: human-readable table).",
    )
    parser.add_argument(
        "--bucket",
        choices=[BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D],
        help="Filter output to one bucket only.",
    )
    parser.add_argument(
        "--include-held",
        action="store_true",
        default=True,
        help=(
            "Always include held PRs (default: yes for visibility — held "
            "PRs always show as Bucket C with reason 'held')."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max PRs to fetch from gh (default: 100).",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help=(
            "Read PR data from a JSON file instead of calling gh (used by tests and offline runs)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.from_json:
        if not args.from_json.exists():
            print(f"ERROR: file not found: {args.from_json}", file=sys.stderr)
            return 2
        try:
            prs = json.loads(args.from_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        if shutil.which("gh") is None:
            print(
                "ERROR: gh CLI not found on PATH — install gh (https://cli.github.com) and retry.",
                file=sys.stderr,
            )
            return 2
        prs = fetch_open_prs(limit=args.limit)

    if not isinstance(prs, list):
        print("ERROR: PR data must be a JSON array", file=sys.stderr)
        return 2

    if args.limit:
        prs = prs[: args.limit]

    results = [classify(pr, prs) for pr in prs if isinstance(pr, dict)]

    if args.bucket:
        results = [r for r in results if r.bucket == args.bucket]

    if args.json:
        _print_json(results)
    else:
        _print_human(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
