#!/usr/bin/env python3
"""Create a controlled B0 cohort of boss-loop issues using existing primitives."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.boss_validation import assess_issue_body_sanitation  # noqa: E402
from aragora.swarm.decomposition_bridge import DecompositionBridge  # noqa: E402
from aragora.swarm.issue_scanner import BossIssueCandidate, scan_all  # noqa: E402
from aragora.swarm.task_sanitizer import (  # noqa: E402
    SanitizationOutcome,
    SanitizationResult,
    TaskSanitizer,
)


DEFAULT_CATEGORY = "test_coverage"
DEFAULT_MIN_SUCCESS_RATE = 0.3
DEFAULT_MAX_CHILDREN = 5
DEFAULT_LABEL = "boss-ready"


@dataclass(frozen=True)
class ChildReview:
    parent_title: str
    candidate: BossIssueCandidate
    prefixed_title: str
    body: str
    task_sanitizer: SanitizationResult
    body_sanitation_ok: bool
    body_sanitation_reason: str


@dataclass(frozen=True)
class ParentReview:
    parent_title: str
    children: list[ChildReview]


def _prefix_title(title: str) -> str:
    normalized = str(title or "").strip()
    if normalized.startswith("[B0-cohort]"):
        return normalized
    return f"[B0-cohort] {normalized}"


def _render_issue_body(candidate: BossIssueCandidate) -> str:
    parts = [f"## Task\n\n{candidate.description.strip()}"]
    scope_lines = [f"- `{path}`" for path in candidate.file_scope]
    scope_lines.extend(f"- `{path}` (create)" for path in candidate.new_files)
    if scope_lines:
        parts.append("### File Scope\n" + "\n".join(scope_lines))
    if candidate.validation_command:
        parts.append("### Validation\n- " + candidate.validation_command)
    if candidate.acceptance_criteria:
        parts.append(
            "### Acceptance Criteria\n"
            + "\n".join(f"- {item}" for item in candidate.acceptance_criteria)
        )
    parts.append(
        "### Constraints\n"
        f"- Estimated complexity: {candidate.estimated_complexity}\n"
        "- B0 cohort seed issue\n"
        "- Keep changes focused to the declared file scope"
    )
    return "\n\n".join(parts).strip()


def _format_parent_body(candidate: BossIssueCandidate) -> str:
    return _render_issue_body(candidate)


def _task_sanitizer_ok(result: SanitizationResult) -> bool:
    return result.outcome not in {SanitizationOutcome.DROPPED, SanitizationOutcome.QUARANTINED}


def _evaluate_candidate(
    candidate: BossIssueCandidate,
    parent_title: str,
    *,
    sanitizer: TaskSanitizer,
) -> ChildReview:
    prefixed_title = _prefix_title(candidate.title)
    body = _render_issue_body(candidate)
    sanitizer_result = sanitizer.sanitize(prefixed_title, body)
    body_ok, body_reason = assess_issue_body_sanitation(body)
    return ChildReview(
        parent_title=parent_title,
        candidate=candidate,
        prefixed_title=prefixed_title,
        body=body,
        task_sanitizer=sanitizer_result,
        body_sanitation_ok=body_ok,
        body_sanitation_reason=body_reason or "",
    )


def _select_cohort(
    candidates: list[BossIssueCandidate],
    *,
    max_issues: int,
    bridge: DecompositionBridge,
    sanitizer: TaskSanitizer,
    skip_decomposition: bool = False,
) -> list[ParentReview]:
    reviews: list[ParentReview] = []
    total_selected = 0
    for candidate in candidates:
        if total_selected >= max_issues:
            break
        parent_body = _format_parent_body(candidate)
        children: list[BossIssueCandidate] = []
        if not skip_decomposition:
            children = bridge.decompose_issue_sync(
                candidate.title,
                parent_body,
                max_children=DEFAULT_MAX_CHILDREN,
            )
        final_children = children or [candidate]
        remaining = max_issues - total_selected
        final_children = final_children[:remaining]
        if not final_children:
            continue
        child_reviews = [
            _evaluate_candidate(child, candidate.title, sanitizer=sanitizer)
            for child in final_children
        ]
        reviews.append(ParentReview(parent_title=candidate.title, children=child_reviews))
        total_selected += len(child_reviews)
    return reviews


def _all_candidates_pass(reviews: list[ParentReview]) -> bool:
    for review in reviews:
        for child in review.children:
            if not _task_sanitizer_ok(child.task_sanitizer):
                return False
            if not child.body_sanitation_ok:
                return False
    return True


def _create_issue(repo: str, title: str, body: str) -> bool:
    return _create_issue_with_label(repo, title, body, label=DEFAULT_LABEL)


def _create_issue_with_label(repo: str, title: str, body: str, *, label: str) -> bool:
    try:
        cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
        normalized_label = str(label or "").strip()
        if normalized_label:
            cmd.extend(["--label", normalized_label])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _print_review(reviews: list[ParentReview]) -> None:
    print("\n" + "=" * 72)
    print(f"B0 cohort candidates: {sum(len(r.children) for r in reviews)}")
    print("=" * 72)
    for idx, review in enumerate(reviews, 1):
        print(f"\nCandidate {idx}: {review.parent_title}")
        child_titles = ", ".join(child.prefixed_title for child in review.children)
        print(f"  Final child title(s): {child_titles}")
        for child in review.children:
            candidate = child.candidate
            print(f"  - Child: {child.prefixed_title}")
            scope = ", ".join(candidate.file_scope + candidate.new_files) or "(none)"
            print(f"    File scope: {scope}")
            print(f"    Validation command: {candidate.validation_command or '(none)'}")
            print(
                "    TaskSanitizer accepts: "
                f"{'yes' if _task_sanitizer_ok(child.task_sanitizer) else 'no'}"
                f" ({child.task_sanitizer.outcome.value})"
            )
            if child.task_sanitizer.reason:
                print(f"    TaskSanitizer reason: {child.task_sanitizer.reason}")
            print(
                "    assess_issue_body_sanitation accepts: "
                f"{'yes' if child.body_sanitation_ok else 'no'}"
            )
            if child.body_sanitation_reason:
                print(f"    assess_issue_body_sanitation reason: {child.body_sanitation_reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a controlled B0 cohort using existing primitives"
    )
    parser.add_argument("--repo", default="synaptent/aragora", help="GitHub repo")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    parser.add_argument("--publish", action="store_true", help="Create issues on GitHub")
    parser.add_argument(
        "--max-issues",
        type=int,
        default=10,
        help="Number of cohort issues to select",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=DEFAULT_MIN_SUCCESS_RATE,
        help="Drop candidate categories below this historical success rate",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[DEFAULT_CATEGORY],
        help="Scanner categories to include (default: test_coverage)",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Use raw scanner candidates instead of decomposition-bridge children",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="GitHub label applied to created issues (default: boss-ready)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run and args.publish:
        print("Error: --dry-run and --publish are mutually exclusive.")
        return 2

    if args.max_issues <= 0:
        print("Error: --max-issues must be > 0.")
        return 2

    repo_root = REPO_ROOT
    candidates = scan_all(
        repo_root,
        categories=args.categories,
        min_success_rate=args.min_success_rate,
    )
    if not candidates:
        print("No candidates found for requested categories.")
        return 1

    bridge = DecompositionBridge(repo_root)
    sanitizer = TaskSanitizer(repo_root=repo_root)
    reviews = _select_cohort(
        candidates,
        max_issues=args.max_issues,
        bridge=bridge,
        sanitizer=sanitizer,
        skip_decomposition=args.skip_decomposition,
    )
    selected_count = sum(len(review.children) for review in reviews)
    if selected_count < args.max_issues:
        print(
            f"Error: only {selected_count} candidate issues available "
            f"(requested {args.max_issues})."
        )
        _print_review(reviews)
        return 1

    _print_review(reviews)

    if args.dry_run:
        print("\nDry run only; no issues created.")
        return 0

    if args.publish:
        if not _all_candidates_pass(reviews):
            print("\nPublish aborted: not all candidates pass sanitation gates.")
            return 1
        created = 0
        failed = 0
        for review in reviews:
            for child in review.children:
                ok = _create_issue_with_label(
                    args.repo,
                    child.prefixed_title,
                    child.body,
                    label=args.label,
                )
                if ok:
                    created += 1
                else:
                    failed += 1
        print(f"\nPublish complete: {created} created, {failed} failed.")
        return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
