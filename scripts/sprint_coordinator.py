#!/usr/bin/env python3
"""Sprint Coordinator -- multi-session parallel development workflow.

Ties TaskDecomposer + BranchCoordinator + git worktrees into a complete
multi-session workflow for parallel Claude Code development.

Usage:
    python scripts/sprint_coordinator.py plan "Make Aragora production-ready" [--debate]
    python scripts/sprint_coordinator.py setup
    python scripts/sprint_coordinator.py status
    python scripts/sprint_coordinator.py merge [--all]
    python scripts/sprint_coordinator.py cleanup
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI color codes for terminal output
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

SPRINT_DIR = PROJECT_ROOT / ".aragora_beads" / "sprint"
MANIFEST_NAME = "sprint-manifest.json"


def _color(text: str, code: str) -> str:
    """Wrap *text* in ANSI color if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _run_git(
    *args: str,
    cwd: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the result."""
    cmd = ["git"] + list(args)
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _load_manifest() -> dict:
    """Load the latest sprint manifest, or exit with an error."""
    manifest_path = SPRINT_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        print(
            f"Error: No sprint manifest found at {manifest_path}\n"
            "Run 'python scripts/sprint_coordinator.py plan <goal>' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(manifest_path.read_text())


def _save_manifest(manifest: dict) -> Path:
    """Save the sprint manifest and return its path."""
    SPRINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPRINT_DIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _branch_name_for_subtask(subtask: dict) -> str:
    """Derive a branch name from a subtask entry."""
    slug = subtask.get("title", subtask["id"])
    slug = slug.lower()
    # Replace non-alphanumeric chars with hyphens
    slug = "".join(c if c.isalnum() else "-" for c in slug)
    slug = slug.strip("-")[:40]
    return f"sprint/{slug}"


def _worktree_dir_for_branch(branch: str) -> Path:
    """Return the worktree directory for a branch."""
    dir_name = branch.replace("/", "-")
    return PROJECT_ROOT / ".worktrees" / dir_name


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> None:
    """Decompose a goal into sprint tasks."""
    from aragora.nomic.task_decomposer import DecomposerConfig, TaskDecomposer

    config = DecomposerConfig(complexity_threshold=3)
    decomposer = TaskDecomposer(config=config)

    if args.debate:
        import asyncio

        result = asyncio.run(decomposer.analyze_with_debate(args.goal))
    else:
        result = decomposer.analyze(args.goal)

    # Build manifest
    manifest: dict = {
        "goal": args.goal,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complexity_score": result.complexity_score,
        "complexity_level": result.complexity_level,
        "subtasks": [],
        "worktrees": {},
    }

    for st in result.subtasks:
        manifest["subtasks"].append(
            {
                "id": st.id,
                "title": st.title,
                "description": st.description,
                "file_scope": st.file_scope,
                "complexity": st.estimated_complexity,
                "dependencies": st.dependencies,
            }
        )

    # Print summary
    print(f"\n{_color('Sprint Plan:', _BOLD)} {args.goal}")
    print(
        f"Complexity: {result.complexity_score}/10 ({result.complexity_level})"
    )
    print(f"Subtasks:   {len(result.subtasks)}")

    if result.subtasks:
        print()
        header = f"  {'ID':<14} {'Title':<30} {'Cplx':<8} {'Files'}"
        print(_color(header, _BOLD))
        print("  " + "-" * 70)
        for st in result.subtasks:
            files = ", ".join(st.file_scope[:3]) if st.file_scope else "(auto)"
            print(
                f"  {st.id:<14} {st.title:<30} {st.estimated_complexity:<8} {files}"
            )

    if args.dry_run:
        print(f"\n(dry-run -- manifest not saved)")
        return

    manifest_path = _save_manifest(manifest)
    print(f"\nManifest saved: {manifest_path}")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    """Create worktrees from the latest sprint manifest."""
    manifest = _load_manifest()
    subtasks = manifest.get("subtasks", [])

    if not subtasks:
        print("No subtasks in the manifest. Nothing to set up.")
        return

    # Ensure worktree base directory exists
    worktree_base = PROJECT_ROOT / ".worktrees"
    worktree_base.mkdir(parents=True, exist_ok=True)

    # Get the current branch to use as base
    base_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    created: list[dict] = {}
    worktrees_info: dict[str, dict] = manifest.get("worktrees", {})

    print(f"\n{_color('Setting up sprint worktrees', _BOLD)}")
    print(f"Base branch: {base_branch}")
    print()

    header = f"  {'Subtask':<14} {'Branch':<42} {'Path'}"
    print(_color(header, _BOLD))
    print("  " + "-" * 80)

    for st in subtasks:
        branch = _branch_name_for_subtask(st)
        wt_path = _worktree_dir_for_branch(branch)

        # Check if worktree already exists
        if wt_path.exists():
            print(
                f"  {st['id']:<14} {branch:<42} {wt_path}  "
                f"{_color('(exists)', _YELLOW)}"
            )
            worktrees_info[st["id"]] = {
                "branch": branch,
                "path": str(wt_path),
            }
            continue

        # Check if branch already exists
        branch_exists = _run_git(
            "rev-parse", "--verify", f"refs/heads/{branch}", check=False
        ).returncode == 0

        if branch_exists:
            # Add worktree for existing branch
            result = _run_git(
                "worktree", "add", str(wt_path), branch, check=False
            )
        else:
            # Create new branch with worktree
            result = _run_git(
                "worktree", "add", "-b", branch, str(wt_path), base_branch,
                check=False,
            )

        if result.returncode == 0:
            status = _color("(created)", _GREEN)
        else:
            status = _color(f"(error: {result.stderr.strip()[:60]})", _RED)

        print(f"  {st['id']:<14} {branch:<42} {wt_path}  {status}")

        worktrees_info[st["id"]] = {
            "branch": branch,
            "path": str(wt_path),
        }

    # Persist worktree info back to manifest
    manifest["worktrees"] = worktrees_info
    _save_manifest(manifest)
    print(f"\nManifest updated with worktree paths.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of sprint worktrees."""
    manifest = _load_manifest()
    worktrees = manifest.get("worktrees", {})

    if not worktrees:
        print("No worktrees configured. Run 'setup' first.")
        return

    base_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    print(f"\n{_color('Sprint Status', _BOLD)}")
    print(f"Base branch: {base_branch}")
    print()

    header = f"  {'Subtask':<14} {'Branch':<42} {'Commits':<10} {'Files':<8} {'Status'}"
    print(_color(header, _BOLD))
    print("  " + "-" * 90)

    for subtask_id, info in worktrees.items():
        branch = info["branch"]
        wt_path = Path(info["path"])

        if not wt_path.exists():
            status = _color("missing", _RED)
            print(f"  {subtask_id:<14} {branch:<42} {'?':<10} {'?':<8} {status}")
            continue

        # Count new commits since base
        log_result = _run_git(
            "log", "--oneline", f"{base_branch}..{branch}",
            check=False,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            commit_count = len(log_result.stdout.strip().splitlines())
        else:
            commit_count = 0

        # Count changed files
        diff_result = _run_git(
            "diff", "--stat", f"{base_branch}...{branch}",
            check=False,
        )
        if diff_result.returncode == 0 and diff_result.stdout.strip():
            # Last line of --stat is the summary; files are the other lines
            stat_lines = diff_result.stdout.strip().splitlines()
            file_count = max(0, len(stat_lines) - 1)
        else:
            file_count = 0

        # Check for merge conflicts (try a dry-run merge)
        has_conflict = False
        if commit_count > 0:
            merge_check = _run_git(
                "merge-tree",
                _run_git("merge-base", base_branch, branch, check=False).stdout.strip(),
                base_branch,
                branch,
                check=False,
            )
            if merge_check.returncode != 0 or "conflict" in merge_check.stdout.lower():
                has_conflict = True

        # Determine status color
        if has_conflict:
            status = _color("conflict", _RED)
        elif commit_count > 0:
            status = _color("active", _GREEN)
        else:
            status = _color("idle", _YELLOW)

        print(
            f"  {subtask_id:<14} {branch:<42} {commit_count:<10} {file_count:<8} {status}"
        )

    print()


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def cmd_merge(args: argparse.Namespace) -> None:
    """Test-gated merge of completed worktrees."""
    manifest = _load_manifest()
    worktrees = manifest.get("worktrees", {})

    if not worktrees:
        print("No worktrees configured. Run 'setup' first.")
        return

    base_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    # Collect branches that have commits
    mergeable: list[tuple[str, str, str]] = []  # (subtask_id, branch, wt_path)

    for subtask_id, info in worktrees.items():
        branch = info["branch"]
        wt_path = info["path"]

        log_result = _run_git(
            "log", "--oneline", f"{base_branch}..{branch}",
            check=False,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            mergeable.append((subtask_id, branch, wt_path))

    if not mergeable:
        print("No branches with new commits to merge.")
        return

    print(f"\n{_color('Merge Candidates', _BOLD)}")
    for subtask_id, branch, _ in mergeable:
        print(f"  - [{subtask_id}] {branch}")
    print()

    merged: list[str] = []
    failed: list[str] = []

    for subtask_id, branch, wt_path in mergeable:
        if not args.all:
            response = input(f"Merge {branch}? [y/N] ").strip().lower()
            if response != "y":
                print(f"  Skipped {branch}")
                continue

        print(f"\n{_color(f'Merging {branch}', _BOLD)}")

        # Step 1: Pre-merge tests in the worktree
        print("  Running pre-merge tests...")
        pre_test = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/nomic/", "-x", "-q", "--timeout=120", "--tb=short",
            ],
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if pre_test.returncode != 0:
            print(
                f"  {_color('FAIL', _RED)}: pre-merge tests failed in {branch}"
            )
            if pre_test.stdout:
                # Show last 10 lines of output
                for line in pre_test.stdout.strip().splitlines()[-10:]:
                    print(f"    {line}")
            failed.append(branch)
            continue

        print(f"  {_color('PASS', _GREEN)}: pre-merge tests")

        # Step 2: Check for conflicts
        merge_check = _run_git(
            "merge", "--no-commit", "--no-ff", branch,
            check=False,
        )

        if merge_check.returncode != 0:
            print(
                f"  {_color('FAIL', _RED)}: merge conflict detected"
            )
            _run_git("merge", "--abort", check=False)
            failed.append(branch)
            continue

        # Abort the dry-run merge, then do the real one
        _run_git("merge", "--abort", check=False)

        # Step 3: Actual merge with --no-ff
        merge_result = _run_git(
            "merge", "--no-ff", "-m",
            f"Merge sprint/{subtask_id}: {branch}",
            branch,
            check=False,
        )

        if merge_result.returncode != 0:
            print(
                f"  {_color('FAIL', _RED)}: merge failed: {merge_result.stderr.strip()[:80]}"
            )
            _run_git("merge", "--abort", check=False)
            failed.append(branch)
            continue

        # Step 4: Post-merge tests
        print("  Running post-merge tests...")
        post_test = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/nomic/", "-x", "-q", "--timeout=120", "--tb=short",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if post_test.returncode != 0:
            print(
                f"  {_color('FAIL', _RED)}: post-merge tests failed -- reverting merge"
            )
            # Revert the merge commit
            _run_git("revert", "-m", "1", "HEAD", "--no-edit", check=False)
            failed.append(branch)
            continue

        merge_sha = _run_git("rev-parse", "--short", "HEAD").stdout.strip()
        print(
            f"  {_color('MERGED', _GREEN)}: {branch} -> {base_branch} ({merge_sha})"
        )
        merged.append(branch)

    # Summary
    print(f"\n{_color('Merge Summary', _BOLD)}")
    if merged:
        print(f"  Merged:  {len(merged)}")
        for b in merged:
            print(f"    {_color('+', _GREEN)} {b}")
    if failed:
        print(f"  Failed:  {len(failed)}")
        for b in failed:
            print(f"    {_color('!', _RED)} {b}")
    if not merged and not failed:
        print("  No branches were merged.")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove merged worktrees and branches."""
    manifest = _load_manifest()
    worktrees = manifest.get("worktrees", {})

    if not worktrees:
        print("No worktrees to clean up.")
        return

    base_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    # Find which branches are merged into base
    merged_result = _run_git(
        "branch", "--merged", base_branch, check=False,
    )
    merged_branches = set()
    if merged_result.returncode == 0:
        for line in merged_result.stdout.strip().splitlines():
            # git branch --merged can prefix with: * (current), + (worktree), spaces
            merged_branches.add(line.strip().lstrip("*+ "))

    removed_count = 0
    kept_count = 0

    print(f"\n{_color('Sprint Cleanup', _BOLD)}")
    print()

    for subtask_id, info in list(worktrees.items()):
        branch = info["branch"]
        wt_path = Path(info["path"])

        if branch in merged_branches:
            # Remove worktree
            if wt_path.exists():
                rm_result = _run_git(
                    "worktree", "remove", str(wt_path), check=False,
                )
                if rm_result.returncode == 0:
                    print(f"  Removed worktree: {wt_path}")
                else:
                    # Force removal if needed
                    force_result = _run_git(
                        "worktree", "remove", "--force", str(wt_path),
                        check=False,
                    )
                    if force_result.returncode == 0 or not wt_path.exists():
                        print(f"  Force-removed worktree: {wt_path}")
                    else:
                        # Last resort: manual removal + prune
                        shutil.rmtree(wt_path, ignore_errors=True)
                        _run_git("worktree", "prune", check=False)
                        print(f"  Removed worktree (manual): {wt_path}")

            # Delete the branch
            del_result = _run_git("branch", "-d", branch, check=False)
            if del_result.returncode == 0:
                print(f"  Deleted branch: {branch}")
            else:
                print(
                    f"  {_color('Could not delete', _YELLOW)} {branch}: "
                    f"{del_result.stderr.strip()[:60]}"
                )

            removed_count += 1
        else:
            print(f"  Kept (not merged): {branch}")
            kept_count += 1

    # Prune stale worktree entries
    _run_git("worktree", "prune", check=False)

    # Update manifest: remove cleaned-up entries
    remaining = {
        sid: info
        for sid, info in worktrees.items()
        if info["branch"] not in merged_branches
    }
    manifest["worktrees"] = remaining
    _save_manifest(manifest)

    print(f"\nRemoved: {removed_count}  |  Kept: {kept_count}")
    if removed_count > 0:
        print("Manifest updated.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint Coordinator for parallel Claude Code sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # plan
    plan_parser = subparsers.add_parser(
        "plan", help="Decompose a goal into sprint tasks",
    )
    plan_parser.add_argument("goal", help="High-level goal to decompose")
    plan_parser.add_argument(
        "--debate", action="store_true",
        help="Use debate-based decomposition (slower, better for abstract goals)",
    )
    plan_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without saving manifest",
    )
    plan_parser.set_defaults(func=cmd_plan)

    # setup
    setup_parser = subparsers.add_parser(
        "setup", help="Create worktrees from manifest",
    )
    setup_parser.set_defaults(func=cmd_setup)

    # status
    status_parser = subparsers.add_parser(
        "status", help="Show sprint progress across worktrees",
    )
    status_parser.set_defaults(func=cmd_status)

    # merge
    merge_parser = subparsers.add_parser(
        "merge", help="Test-gated merge of completed worktrees",
    )
    merge_parser.add_argument(
        "--all", action="store_true",
        help="Merge all ready branches without prompting",
    )
    merge_parser.set_defaults(func=cmd_merge)

    # cleanup
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Remove merged worktrees and branches",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
