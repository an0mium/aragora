"""Worktree management CLI subcommand.

Provides first-class CLI access to git worktree management for
parallel multi-agent development sessions.

Usage:
    aragora worktree create --tracks sme developer qa
    aragora worktree list
    aragora worktree merge <branch>
    aragora worktree merge-all --test-first
    aragora worktree conflicts
    aragora worktree cleanup
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path


def add_worktree_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'worktree' subcommand and its sub-subcommands."""
    wt_parser = subparsers.add_parser(
        "worktree",
        help="Manage git worktrees for parallel agent sessions",
        description=(
            "Create, list, merge, and clean up git worktrees for isolated "
            "parallel development sessions. Each track gets its own worktree "
            "so multiple Claude Code sessions can work without conflicts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow:
  1. aragora worktree create --tracks sme developer qa
  2. Open a Claude Code session in each worktree path
  3. Work independently; each session commits to its branch
  4. aragora worktree merge-all --test-first
  5. aragora worktree cleanup
""",
    )
    wt_parser.add_argument(
        "--repo",
        default=None,
        help="Repository path (default: current directory)",
    )
    wt_parser.add_argument(
        "--base",
        default="main",
        help="Base branch (default: main)",
    )

    wt_sub = wt_parser.add_subparsers(dest="wt_action")

    # create
    create_p = wt_sub.add_parser("create", help="Create worktrees for tracks")
    create_p.add_argument(
        "--tracks",
        "-t",
        nargs="+",
        required=True,
        help="Tracks: sme, developer, self_hosted, qa, core, security",
    )

    # list
    wt_sub.add_parser("list", help="List active worktrees")

    # merge
    merge_p = wt_sub.add_parser("merge", help="Merge a branch back to base")
    merge_p.add_argument("branch", help="Branch name to merge")
    merge_p.add_argument("--test-first", action="store_true", help="Run tests before merging")
    merge_p.add_argument("--dry-run", action="store_true", help="Check without merging")

    # merge-all
    merge_all_p = wt_sub.add_parser("merge-all", help="Merge all worktree branches")
    merge_all_p.add_argument("--test-first", action="store_true", help="Run tests first")

    # conflicts
    wt_sub.add_parser("conflicts", help="Show conflict report")

    # cleanup
    wt_sub.add_parser("cleanup", help="Clean up merged worktrees")

    wt_parser.set_defaults(func=cmd_worktree)


def cmd_worktree(args: argparse.Namespace) -> None:
    """Dispatch worktree subcommand."""
    action = getattr(args, "wt_action", None)
    if not action:
        print("Usage: aragora worktree {create|list|merge|merge-all|conflicts|cleanup}")
        print("Run 'aragora worktree --help' for details.")
        return

    from aragora.nomic.branch_coordinator import (
        BranchCoordinator,
        BranchCoordinatorConfig,
    )
    from aragora.nomic.meta_planner import Track

    repo_path = Path(args.repo).resolve() if args.repo else Path.cwd()
    base_branch = args.base

    config = BranchCoordinatorConfig(
        base_branch=base_branch,
        use_worktrees=True,
    )
    coordinator = BranchCoordinator(repo_path=repo_path, config=config)

    track_map = {t.value: t for t in Track}

    if action == "create":
        tracks = []
        for name in args.tracks:
            key = name.lower().strip()
            if key not in track_map:
                print(f"Error: Unknown track '{name}'. Valid: {', '.join(sorted(track_map))}")
                return
            tracks.append(track_map[key])

        config.max_parallel_branches = len(tracks)
        print(f"Creating {len(tracks)} worktree(s) from '{base_branch}'...\n")

        for track in tracks:
            goal = f"Session work on {track.value} track"
            branch = asyncio.run(coordinator.create_track_branch(track=track, goal=goal))
            wt_path = coordinator.get_worktree_path(branch)
            print(f"  [{track.value}]  {branch}")
            print(f"    Path: {wt_path}\n")

        print("Done. Open a session in each worktree path.")

    elif action == "list":
        worktrees = coordinator.list_worktrees()
        if not worktrees:
            print("No active worktrees.")
            return
        print(f"Active worktrees ({len(worktrees)}):\n")
        for wt in worktrees:
            track_label = f" [{wt.track}]" if wt.track else ""
            print(f"  {wt.branch_name}{track_label}")
            print(f"    Path: {wt.worktree_path}")
            if wt.created_at:
                print(f"    Created: {wt.created_at:%Y-%m-%d %H:%M}")
            print()

    elif action == "merge":
        branch = args.branch
        if not coordinator.branch_exists(branch):
            print(f"Error: Branch '{branch}' does not exist.")
            return

        if getattr(args, "test_first", False):
            wt_path = coordinator.get_worktree_path(branch) or repo_path
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                cwd=wt_path, capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                print("Tests FAILED. Aborting merge.")
                return
            print("Tests passed.")

        dry_run = getattr(args, "dry_run", False)
        merge_result = asyncio.run(coordinator.safe_merge(branch, dry_run=dry_run))
        if merge_result.success:
            if dry_run:
                print("Merge would succeed (no conflicts).")
            else:
                print(f"Merged: {merge_result.commit_sha[:12]}")
        else:
            print(f"Merge failed: {merge_result.error}")
            if merge_result.conflicts:
                for f in merge_result.conflicts:
                    print(f"  - {f}")

    elif action == "merge-all":
        worktrees = coordinator.list_worktrees()
        branches = [
            wt.branch_name for wt in worktrees
            if wt.branch_name not in (base_branch, "main")
        ]
        if not branches:
            print("No branches to merge.")
            return

        merged, failed = 0, 0
        for branch in branches:
            print(f"Merging: {branch}")
            if getattr(args, "test_first", False):
                wt_path = coordinator.get_worktree_path(branch) or repo_path
                result = subprocess.run(
                    ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                    cwd=wt_path, capture_output=True, text=True, check=False,
                )
                if result.returncode != 0:
                    print("  Tests FAILED, skipping.")
                    failed += 1
                    continue

            merge_result = asyncio.run(coordinator.safe_merge(branch))
            if merge_result.success:
                print(f"  Merged: {merge_result.commit_sha[:12]}")
                merged += 1
            else:
                print(f"  FAILED: {merge_result.error}")
                failed += 1

        print(f"\n{merged} merged, {failed} failed")

    elif action == "conflicts":
        worktrees = coordinator.list_worktrees()
        branches = [
            wt.branch_name for wt in worktrees
            if wt.branch_name not in (base_branch, "main")
        ]
        if not branches:
            print("No active branches.")
            return

        conflicts = asyncio.run(coordinator.detect_conflicts(branches))
        if not conflicts:
            print("No conflicts detected.")
        else:
            print(f"{len(conflicts)} potential conflict(s):\n")
            for c in conflicts:
                print(f"  {c.source_branch} <-> {c.target_branch} [{c.severity}]")
                print(f"    Files: {', '.join(c.conflicting_files[:5])}")
                if c.resolution_hint:
                    print(f"    Hint: {c.resolution_hint}")
                print()

    elif action == "cleanup":
        deleted = coordinator.cleanup_branches()
        removed = coordinator.cleanup_worktrees()
        print(f"Deleted {deleted} merged branch(es), removed {removed} worktree(s).")
