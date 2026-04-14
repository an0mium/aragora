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
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.decomposition_bridge import DecompositionBridge  # noqa: E402
from aragora.swarm.issue_scanner import BossIssueCandidate, scan_all  # noqa: E402
from aragora.swarm.issue_upgrader import upgrade_issue_heuristic  # noqa: E402
from aragora.swarm.roadmap_priority import load_roadmap_priority_policy  # noqa: E402

_GENERIC_PARENT_PHRASES: tuple[str, ...] = (
    "read the module and identify all public functions",
    "create a test file with broad coverage",
    "covering all public api surface",
    "while preserving existing behavior and keeping the work reviewable",
    "supporting tests",
    "think about it",
)
_OPEN_PR_PAGE_SIZE = 100
_OPEN_PR_MAX_PAGES = 10
_OPEN_PR_FILES_PAGE_SIZE = 100
_OPEN_PR_FILES_MAX_PAGES = 10
_UPGRADEABLE_CATEGORIES = frozenset(
    {"test_coverage", "broad_exception", "silent_exception", "type_annotation"}
)


@dataclass(slots=True)
class DecompositionTelemetry:
    """Aggregate telemetry for one generator run."""

    parents_seen: int = 0
    parents_eligible: int = 0
    parents_preserved: int = 0
    parents_replaced: int = 0
    children_emitted: int = 0
    children_rejected: int = 0
    sanitizer_rejections: int = 0


def format_boss_ready_body(candidate: BossIssueCandidate) -> str:
    """Format a candidate into the proven boss-ready issue body.

    For supported categories, use the issue upgrader to generate concrete,
    module-aware issue bodies instead of the generic template.
    """
    if candidate.category in _UPGRADEABLE_CATEGORIES:
        upgraded = upgrade_issue_heuristic(
            candidate.title,
            f"## Task\n\n{candidate.description}\n\n### File Scope\n"
            + "\n".join(f"- `{f}`" for f in candidate.file_scope),
            repo_root=REPO_ROOT,
            category=candidate.category,
            validation_command=candidate.validation_command,
            acceptance_criteria=list(candidate.acceptance_criteria),
            new_files=list(candidate.new_files),
        )
        if upgraded:
            body = upgraded.upgraded_body
            body += f"\n\n<!-- fingerprint:{candidate.fingerprint} -->"
            return body

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
        files: set[str] = set()
        for page in range(1, _OPEN_PR_MAX_PAGES + 1):
            prs_result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{repo}/pulls?state=open&per_page={_OPEN_PR_PAGE_SIZE}&page={page}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if prs_result.returncode != 0:
                return set()
            prs = json.loads(prs_result.stdout or "[]")
            if not isinstance(prs, list):
                return set()
            if not prs:
                return files
            for pr in prs:
                if not isinstance(pr, dict):
                    continue
                number = pr.get("number")
                if not isinstance(number, int):
                    continue
                for files_page in range(1, _OPEN_PR_FILES_MAX_PAGES + 1):
                    files_result = subprocess.run(
                        [
                            "gh",
                            "api",
                            (
                                f"repos/{repo}/pulls/{number}/files"
                                f"?per_page={_OPEN_PR_FILES_PAGE_SIZE}&page={files_page}"
                            ),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if files_result.returncode != 0:
                        return set()
                    payload = json.loads(files_result.stdout or "[]")
                    if not isinstance(payload, list):
                        return set()
                    if not payload:
                        break
                    for item in payload:
                        path = item.get("filename", "") if isinstance(item, dict) else str(item)
                        if path:
                            files.add(path)
                    if len(payload) < _OPEN_PR_FILES_PAGE_SIZE:
                        break
                else:
                    raise RuntimeError(
                        "open PR file pagination exceeded configured cap "
                        f"({_OPEN_PR_FILES_MAX_PAGES} pages) for PR #{number}"
                    )
            if len(prs) < _OPEN_PR_PAGE_SIZE:
                return files
        raise RuntimeError(
            f"open PR pagination exceeded configured cap ({_OPEN_PR_MAX_PAGES} pages) for {repo}"
        )
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


def is_low_quality_parent(candidate: BossIssueCandidate, body: str) -> bool:
    """Return True only for generic parent issues worth decomposing.

    The bridge should operate on broad or templated parent issues, not on
    already-bounded single-file work orders produced by the scanner.
    """
    total_scope = len(candidate.file_scope) + len(candidate.new_files)
    normalized_body = " ".join(body.lower().split())

    if total_scope == 0:
        return True
    if total_scope > 2:
        return True
    if not candidate.validation_command:
        return True
    if candidate.estimated_complexity not in {"small", "medium"}:
        return True
    return any(phrase in normalized_body for phrase in _GENERIC_PARENT_PHRASES)


def _print_decomposition_telemetry(telemetry: DecompositionTelemetry) -> None:
    print(
        "  Decomposition telemetry:"
        f" seen={telemetry.parents_seen}"
        f" eligible={telemetry.parents_eligible}"
        f" preserved={telemetry.parents_preserved}"
        f" replaced={telemetry.parents_replaced}"
        f" children={telemetry.children_emitted}"
        f" rejected={telemetry.children_rejected}"
        f" sanitizer={telemetry.sanitizer_rejections}"
    )


def maybe_decompose_candidates_with_telemetry(
    candidates: list[BossIssueCandidate],
    *,
    enabled: bool,
    max_children_per_parent: int,
    repo_root: Path,
) -> tuple[list[BossIssueCandidate], DecompositionTelemetry]:
    """Optionally replace low-quality parents with bounded child candidates."""
    telemetry = DecompositionTelemetry(parents_seen=len(candidates))
    if not enabled or not candidates:
        telemetry.parents_preserved = len(candidates)
        return list(candidates), telemetry

    bridge = DecompositionBridge(repo_root)
    expanded: list[BossIssueCandidate] = []
    for candidate in candidates:
        parent_body = format_boss_ready_body(candidate)
        if not is_low_quality_parent(candidate, parent_body):
            expanded.append(candidate)
            telemetry.parents_preserved += 1
            continue

        telemetry.parents_eligible += 1
        outcome = bridge.decompose_issue_sync_with_stats(
            candidate.title,
            parent_body,
            max_children=max_children_per_parent,
        )
        telemetry.children_rejected += outcome.stats.rejected_candidates
        telemetry.sanitizer_rejections += outcome.stats.sanitizer_rejections

        if len(outcome.children) >= 2:
            expanded.extend(outcome.children)
            telemetry.parents_replaced += 1
            telemetry.children_emitted += len(outcome.children)
        else:
            expanded.append(candidate)
            telemetry.parents_preserved += 1

    return expanded, telemetry


def maybe_decompose_candidates(
    candidates: list[BossIssueCandidate],
    *,
    enabled: bool,
    max_children_per_parent: int,
    repo_root: Path,
) -> list[BossIssueCandidate]:
    expanded, _ = maybe_decompose_candidates_with_telemetry(
        candidates,
        enabled=enabled,
        max_children_per_parent=max_children_per_parent,
        repo_root=repo_root,
    )
    return expanded


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
    parser.add_argument("--label", default="boss-ready", help="Label for created issues")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.3,
        help="Drop candidate categories below this historical success rate",
    )
    parser.add_argument(
        "--decompose-low-quality",
        action="store_true",
        help="Use the decomposition bridge to replace vague parent issues with bounded child issues",
    )
    parser.add_argument(
        "--max-children-per-parent",
        type=int,
        default=5,
        help="Maximum child issues to emit for one decomposed parent",
    )
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
    candidates, decomposition_telemetry = maybe_decompose_candidates_with_telemetry(
        candidates,
        enabled=args.decompose_low_quality,
        max_children_per_parent=args.max_children_per_parent,
        repo_root=repo_root,
    )
    if args.decompose_low_quality:
        _print_decomposition_telemetry(decomposition_telemetry)
    print(
        f"  Found {len(candidates)} candidates across {len(set(c.category for c in candidates))} categories"
    )

    if args.verbose:
        by_cat: dict[str, int] = {}
        for c in candidates:
            by_cat[c.category] = by_cat.get(c.category, 0) + 1
        for cat, count in sorted(by_cat.items()):
            print(f"    {cat}: {count}")

    # 2. Deduplicate against existing issues (always fetch, even in dry-run)
    print("Fetching existing boss-ready issues...")
    existing = fetch_existing_boss_issues(args.repo)
    print(f"  {len(existing)} existing issues")

    # 3. Check PR conflicts (always fetch, even in dry-run)
    print("Fetching open PR files...")
    try:
        pr_files = fetch_open_pr_files(args.repo)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"  {len(pr_files)} files in open PRs")

    # 4. Filter
    filtered: list[tuple[BossIssueCandidate, str]] = []
    skipped_dup = 0
    skipped_pr = 0
    skipped_val = 0
    skipped_priority = 0
    priority_policy = load_roadmap_priority_policy(repo_root)

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
        if args.label == "boss-ready" and priority_policy is not None:
            priority = priority_policy.priority_for_text(candidate.title, body)
            if priority.priority.blocks_boss_ready:
                skipped_priority += 1
                if args.verbose:
                    print(
                        "  SKIP (canonical priority): "
                        f"{candidate.title} [{', '.join(priority.blocked_codes)}]"
                    )
                continue
        ok, reason = validate_body(body)
        if not ok:
            skipped_val += 1
            if args.verbose:
                print(f"  SKIP (validation: {reason}): {candidate.title}")
            continue

        filtered.append((candidate, body))

    print(f"\nFiltered to {len(filtered)} valid candidates")
    print(
        "  Skipped: "
        f"{skipped_dup} duplicates, "
        f"{skipped_pr} PR conflicts, "
        f"{skipped_priority} canonical priority blocks, "
        f"{skipped_val} validation failures"
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
