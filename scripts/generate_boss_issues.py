#!/usr/bin/env python3
"""Generate boss-ready GitHub issues by scanning the codebase.

Scans for improvement opportunities (missing tests, silent exceptions,
broad exception handlers, TODOs, etc.), formats them as boss-ready issues,
validates through the pre-dispatch gate, deduplicates against open issues,
and creates them on GitHub.

Usage:
    python scripts/generate_boss_issues.py --dry-run            # Preview
    python scripts/generate_boss_issues.py --max-issues 10      # Create 10
    python scripts/generate_boss_issues.py --categories test_coverage silent_exception
    python scripts/generate_boss_issues.py --categories handler_validation --min-success-rate 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.issue_scanner import BossIssueCandidate, scan_all  # noqa: E402


def format_boss_ready_body(candidate: BossIssueCandidate) -> str:
    """Format a candidate into the proven boss-ready issue body."""
    parts: list[str] = []

    # Task section
    parts.append(f"## Task\n\n{candidate.description}")

    # File scope
    scope_lines: list[str] = []
    for f in candidate.file_scope:
        scope_lines.append(f"- `{f}`")
    for f in candidate.new_files:
        scope_lines.append(f"- `{f}` (create)")
    if scope_lines:
        parts.append("### File Scope\n" + "\n".join(scope_lines))

    # Validation
    if candidate.validation_command:
        parts.append(f"### Validation\n```bash\n{candidate.validation_command}\n```")

    # Acceptance criteria
    if candidate.acceptance_criteria:
        criteria = "\n".join(f"- {c}" for c in candidate.acceptance_criteria)
        parts.append(f"### Acceptance Criteria\n{criteria}")

    # Constraints
    parts.append(
        "### Constraints\n"
        "- Single-file change preferred\n"
        "- Under 100 lines of new/changed code\n"
        f"- Estimated complexity: {candidate.estimated_complexity}"
    )

    # Fingerprint for exact dedup across runs
    parts.append(f"<!-- fingerprint:{candidate.fingerprint} -->")

    return "\n\n".join(parts)


def fetch_existing_boss_issues(repo: str) -> list[dict]:
    """Fetch open boss-ready issues from GitHub."""
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo,
                "--label",
                "boss-ready",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,title,body",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout or "[]")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return []


def fetch_open_pr_files(repo: str) -> set[str]:
    """Fetch file paths changed in open PRs."""
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--limit",
                "50",
                "--json",
                "files",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            prs = json.loads(result.stdout or "[]")
            files: set[str] = set()
            for pr in prs:
                for f in pr.get("files", []):
                    path = f.get("path", "") if isinstance(f, dict) else str(f)
                    if path:
                        files.add(path)
            return files
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return set()


def _normalize_tokens(text: str) -> set[str]:
    """Normalize title into tokens for similarity comparison."""
    text = text.lower()
    # Remove common prefixes
    for prefix in [
        "add unit tests for",
        "narrow broad except exception in",
        "replace silent exception",
        "address todo/fixme",
    ]:
        text = text.replace(prefix, "")
    return {t for t in text.split() if len(t) > 2}


def is_duplicate(
    candidate: BossIssueCandidate,
    existing: list[dict],
) -> bool:
    """Check if candidate duplicates an existing issue."""
    candidate_tokens = _normalize_tokens(candidate.title)
    candidate_files = set(candidate.file_scope + candidate.new_files)

    for issue in existing:
        # Fingerprint match (exact)
        if candidate.fingerprint in (issue.get("body") or ""):
            return True

        # Title similarity (Jaccard > 0.6)
        existing_tokens = _normalize_tokens(issue.get("title", ""))
        if candidate_tokens and existing_tokens:
            intersection = candidate_tokens & existing_tokens
            union = candidate_tokens | existing_tokens
            if union and len(intersection) / len(union) > 0.6:
                return True

        # File scope overlap
        existing_body = issue.get("body", "") or ""
        for f in candidate_files:
            if f in existing_body:
                return True

    return False


def conflicts_with_pr(
    candidate: BossIssueCandidate,
    pr_files: set[str],
) -> bool:
    """Check if candidate's file scope overlaps with open PRs."""
    for f in candidate.file_scope:
        if f in pr_files:
            return True
    return False


def validate_body(body: str) -> tuple[bool, str]:
    """Validate issue body through sanitation check."""
    try:
        from aragora.swarm.boss_validation import assess_issue_body_sanitation

        ok, reason = assess_issue_body_sanitation(body)
        return ok, reason or ""
    except ImportError:
        # If boss_validation not importable, do basic checks
        if len(body.strip()) < 50:
            return False, "body_too_short"
        if "## Task" not in body:
            return False, "missing_task_section"
        return True, ""


def _count_categories(candidates: list[BossIssueCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.category] = counts.get(candidate.category, 0) + 1
    return counts


def _compute_suppressed_candidates(
    repo_root: Path,
    *,
    categories: list[str] | None,
    min_success_rate: float,
    kept_candidates: list[BossIssueCandidate],
) -> list[BossIssueCandidate]:
    if min_success_rate <= 0:
        return []

    all_candidates = scan_all(repo_root, categories=categories, min_success_rate=0.0)
    kept_fingerprints = {candidate.fingerprint for candidate in kept_candidates}
    return [
        candidate for candidate in all_candidates if candidate.fingerprint not in kept_fingerprints
    ]


def create_github_issue(repo: str, title: str, body: str, label: str) -> bool:
    """Create a GitHub issue and return success."""
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                title,
                "--body",
                body,
                "--label",
                label,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate boss-ready GitHub issues by scanning the codebase"
    )
    parser.add_argument("--repo", default="synaptent/aragora", help="GitHub repo")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    parser.add_argument("--max-issues", type=int, default=20, help="Max issues to create")
    parser.add_argument("--categories", nargs="*", help="Filter to specific categories")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.3,
        help="Hide categories below this historical success rate threshold; use 0 to disable",
    )
    parser.add_argument("--label", default="boss-ready", help="Label for created issues")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    repo_root = REPO_ROOT

    # 1. Scan
    print(f"Scanning {repo_root}...")
    candidates = scan_all(
        repo_root,
        categories=args.categories,
        min_success_rate=args.min_success_rate,
    )
    suppressed = _compute_suppressed_candidates(
        repo_root,
        categories=args.categories,
        min_success_rate=args.min_success_rate,
        kept_candidates=candidates,
    )
    print(
        f"  Found {len(candidates)} candidates across {len(set(c.category for c in candidates))} categories"
    )
    if suppressed:
        suppressed_counts = _count_categories(suppressed)
        print(
            "  Suppressed "
            f"{len(suppressed)} low-signal candidates across {len(suppressed_counts)} categories "
            f"at min success rate {args.min_success_rate:.0%}"
        )
        print("  Use --min-success-rate 0 to include them.")

    if args.verbose:
        by_cat = _count_categories(candidates)
        for cat, count in sorted(by_cat.items()):
            print(f"    {cat}: {count}")
        if suppressed:
            for cat, count in sorted(_count_categories(suppressed).items()):
                print(f"    suppressed {cat}: {count}")

    # 2. Deduplicate against existing issues (always fetch, even in dry-run)
    print("Fetching existing boss-ready issues...")
    existing = fetch_existing_boss_issues(args.repo)
    print(f"  {len(existing)} existing issues")

    # 3. Check PR conflicts (always fetch, even in dry-run)
    print("Fetching open PR files...")
    pr_files = fetch_open_pr_files(args.repo)
    print(f"  {len(pr_files)} files in open PRs")

    # 4. Filter
    filtered: list[tuple[BossIssueCandidate, str]] = []
    skipped_dup = 0
    skipped_pr = 0
    skipped_val = 0

    for candidate in candidates:
        if len(filtered) >= args.max_issues * 2:
            break

        if is_duplicate(candidate, existing):
            skipped_dup += 1
            if args.verbose:
                print(f"  SKIP (duplicate): {candidate.title}")
            continue

        if conflicts_with_pr(candidate, pr_files):
            skipped_pr += 1
            if args.verbose:
                print(f"  SKIP (PR conflict): {candidate.title}")
            continue

        body = format_boss_ready_body(candidate)
        ok, reason = validate_body(body)
        if not ok:
            skipped_val += 1
            if args.verbose:
                print(f"  SKIP (validation: {reason}): {candidate.title}")
            continue

        filtered.append((candidate, body))

    print(f"\nFiltered to {len(filtered)} valid candidates")
    print(
        f"  Skipped: {skipped_dup} duplicates, {skipped_pr} PR conflicts, {skipped_val} validation failures"
    )

    # 5. Trim to max
    to_create = filtered[: args.max_issues]

    # 6. Create or dry-run
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"DRY RUN — would create {len(to_create)} issues:")
        print(f"{'=' * 60}")
        for i, (candidate, body) in enumerate(to_create, 1):
            print(f"\n--- Issue {i}/{len(to_create)} ---")
            print(f"TITLE: {candidate.title}")
            print(f"CATEGORY: {candidate.category}")
            print(f"SUCCESS RATE: {candidate.expected_success_rate:.0%}")
            print(f"FILES: {', '.join(candidate.file_scope + candidate.new_files)}")
            print(f"FINGERPRINT: {candidate.fingerprint}")
            if args.verbose:
                print(f"\nBODY:\n{body}")
    else:
        created = 0
        failed = 0
        for i, (candidate, body) in enumerate(to_create, 1):
            print(f"  [{i}/{len(to_create)}] Creating: {candidate.title}...", end=" ")
            if create_github_issue(args.repo, candidate.title, body, args.label):
                print("OK")
                created += 1
            else:
                print("FAILED")
                failed += 1
            time.sleep(1)  # Rate limit safety

        print(f"\nDone: {created} created, {failed} failed")


if __name__ == "__main__":
    main()
