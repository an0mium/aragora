"""Branch Coordinator for parallel development.

Manages multiple development branches for parallel nomic loops,
handles conflict detection, and coordinates merges.

Usage:
    from aragora.nomic.branch_coordinator import BranchCoordinator

    coordinator = BranchCoordinator()

    # Create branches for parallel work
    branches = await coordinator.create_track_branches([
        TrackAssignment(track=Track.SME, goal="Improve dashboard"),
        TrackAssignment(track=Track.QA, goal="Add E2E tests"),
    ])

    # Run parallel work
    result = await coordinator.coordinate_parallel_work(branches)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aragora.nomic.meta_planner import PrioritizedGoal, Track

logger = logging.getLogger(__name__)


@dataclass
class TrackAssignment:
    """Assignment of a goal to a track for parallel execution."""

    goal: PrioritizedGoal
    branch_name: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, merged
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ConflictReport:
    """Report of potential merge conflicts."""

    source_branch: str
    target_branch: str
    conflicting_files: List[str]
    severity: str  # low, medium, high
    resolution_hint: str = ""


@dataclass
class MergeResult:
    """Result of a merge operation."""

    source_branch: str
    target_branch: str
    success: bool
    commit_sha: Optional[str] = None
    error: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)


@dataclass
class CoordinationResult:
    """Result of coordinating parallel work."""

    total_branches: int
    completed_branches: int
    failed_branches: int
    merged_branches: int
    assignments: List[TrackAssignment]
    duration_seconds: float
    success: bool
    summary: str = ""


@dataclass
class BranchCoordinatorConfig:
    """Configuration for BranchCoordinator."""

    base_branch: str = "main"
    branch_prefix: str = "dev"
    auto_merge_safe: bool = True
    require_tests_pass: bool = True
    max_parallel_branches: int = 3


class BranchCoordinator:
    """Manages parallel development branches.

    Creates feature branches for each track/goal, runs nomic loops
    in parallel, and coordinates merges back to main.
    """

    def __init__(
        self,
        repo_path: Optional[Path] = None,
        config: Optional[BranchCoordinatorConfig] = None,
        on_conflict: Optional[Callable[[ConflictReport], None]] = None,
    ):
        self.repo_path = repo_path or Path.cwd()
        self.config = config or BranchCoordinatorConfig()
        self.on_conflict = on_conflict
        self._active_branches: List[str] = []

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        result = self._run_git(
            "rev-parse",
            "--verify",
            f"refs/heads/{branch_name}",
            check=False,
        )
        return result.returncode == 0

    async def create_track_branch(
        self,
        track: Track,
        goal: str,
        base_branch: Optional[str] = None,
    ) -> str:
        """Create a feature branch for a track.

        Args:
            track: Development track
            goal: Goal description (for branch naming)
            base_branch: Branch to base off of

        Returns:
            Branch name
        """
        base = base_branch or self.config.base_branch

        # Generate branch name from goal
        goal_slug = self._slugify(goal)[:30]
        timestamp = datetime.now(timezone.utc).strftime("%m%d")
        branch_name = f"{self.config.branch_prefix}/{track.value}-{goal_slug}-{timestamp}"

        # Ensure we're on base branch
        current = self.get_current_branch()
        if current != base:
            self._run_git("checkout", base)
            self._run_git("pull", "--rebase", "origin", base, check=False)

        # Create and checkout new branch
        if self.branch_exists(branch_name):
            logger.warning(f"Branch {branch_name} already exists, using it")
            self._run_git("checkout", branch_name)
        else:
            self._run_git("checkout", "-b", branch_name)
            logger.info(f"Created branch: {branch_name}")

        self._active_branches.append(branch_name)
        return branch_name

    async def create_track_branches(
        self,
        assignments: List[TrackAssignment],
    ) -> List[TrackAssignment]:
        """Create branches for all track assignments.

        Args:
            assignments: List of track assignments

        Returns:
            Updated assignments with branch names
        """
        for assignment in assignments:
            branch = await self.create_track_branch(
                track=assignment.goal.track,
                goal=assignment.goal.description,
            )
            assignment.branch_name = branch

        # Return to base branch
        self._run_git("checkout", self.config.base_branch, check=False)

        return assignments

    async def detect_conflicts(
        self,
        branches: List[str],
        target_branch: Optional[str] = None,
    ) -> List[ConflictReport]:
        """Detect potential merge conflicts between branches.

        Args:
            branches: List of branch names to check
            target_branch: Target branch for merge (default: main)

        Returns:
            List of conflict reports
        """
        target = target_branch or self.config.base_branch
        conflicts = []

        for branch in branches:
            if not self.branch_exists(branch):
                continue

            # Get files changed in this branch
            changed_files = self._get_branch_files(branch, target)

            # Check against other branches
            for other_branch in branches:
                if other_branch == branch or not self.branch_exists(other_branch):
                    continue

                other_files = self._get_branch_files(other_branch, target)
                overlap = set(changed_files) & set(other_files)

                if overlap:
                    # Determine severity
                    if len(overlap) > 5:
                        severity = "high"
                    elif len(overlap) > 2:
                        severity = "medium"
                    else:
                        severity = "low"

                    conflicts.append(
                        ConflictReport(
                            source_branch=branch,
                            target_branch=other_branch,
                            conflicting_files=list(overlap),
                            severity=severity,
                            resolution_hint=self._generate_resolution_hint(list(overlap)),
                        )
                    )

        return conflicts

    def _get_branch_files(self, branch: str, base: str) -> List[str]:
        """Get files changed in a branch relative to base."""
        result = self._run_git(
            "diff",
            "--name-only",
            f"{base}...{branch}",
            check=False,
        )
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.split("\n") if f.strip()]

    def _generate_resolution_hint(self, conflicting_files: List[str]) -> str:
        """Generate a hint for resolving conflicts."""
        if any("test" in f.lower() for f in conflicting_files):
            return "Consider merging test changes separately"
        if any(f.endswith(".py") for f in conflicting_files):
            return "Review Python module changes carefully"
        return "Manual review recommended"

    async def safe_merge(
        self,
        source: str,
        target: Optional[str] = None,
        dry_run: bool = False,
    ) -> MergeResult:
        """Merge a branch if safe.

        Args:
            source: Source branch to merge
            target: Target branch (default: main)
            dry_run: If True, only check if merge is possible

        Returns:
            MergeResult with status
        """
        target = target or self.config.base_branch

        if not self.branch_exists(source):
            return MergeResult(
                source_branch=source,
                target_branch=target,
                success=False,
                error=f"Branch {source} does not exist",
            )

        # Checkout target branch
        self._run_git("checkout", target)
        self._run_git("pull", "--rebase", "origin", target, check=False)

        if dry_run:
            # Check if merge would succeed
            result = self._run_git("merge", "--no-commit", "--no-ff", source, check=False)
            self._run_git("merge", "--abort", check=False)

            return MergeResult(
                source_branch=source,
                target_branch=target,
                success=result.returncode == 0,
                conflicts=(
                    self._parse_merge_conflicts(result.stderr) if result.returncode != 0 else []
                ),
            )

        # Perform actual merge
        result = self._run_git(
            "merge",
            "--no-ff",
            "-m",
            f"Merge {source} into {target}",
            source,
            check=False,
        )

        if result.returncode != 0:
            conflicts = self._parse_merge_conflicts(result.stderr)
            self._run_git("merge", "--abort", check=False)

            return MergeResult(
                source_branch=source,
                target_branch=target,
                success=False,
                error="Merge conflicts detected",
                conflicts=conflicts,
            )

        # Get commit SHA
        sha_result = self._run_git("rev-parse", "HEAD")
        commit_sha = sha_result.stdout.strip()

        logger.info(f"Merged {source} into {target}: {commit_sha[:8]}")

        return MergeResult(
            source_branch=source,
            target_branch=target,
            success=True,
            commit_sha=commit_sha,
        )

    def _parse_merge_conflicts(self, stderr: str) -> List[str]:
        """Parse conflicting files from git merge stderr."""
        conflicts = []
        for line in stderr.split("\n"):
            if "CONFLICT" in line and "Merge conflict in" in line:
                # Extract filename
                parts = line.split("Merge conflict in")
                if len(parts) > 1:
                    conflicts.append(parts[1].strip())
        return conflicts

    async def coordinate_parallel_work(
        self,
        assignments: List[TrackAssignment],
        run_nomic_fn: Optional[Callable[[TrackAssignment], Any]] = None,
    ) -> CoordinationResult:
        """Run nomic loops in parallel on separate branches.

        Args:
            assignments: Track assignments with goals
            run_nomic_fn: Function to run nomic loop on a branch

        Returns:
            CoordinationResult with status
        """
        start_time = datetime.now(timezone.utc)

        # Create branches for all assignments
        assignments = await self.create_track_branches(assignments)

        # Check for conflicts upfront
        branches = [a.branch_name for a in assignments if a.branch_name]
        conflicts = await self.detect_conflicts(branches)

        for conflict in conflicts:
            logger.warning(
                f"potential_conflict source={conflict.source_branch} "
                f"target={conflict.target_branch} files={conflict.conflicting_files}"
            )
            if self.on_conflict:
                self.on_conflict(conflict)

        # Run nomic loops in parallel
        if run_nomic_fn:
            tasks = []
            for assignment in assignments:
                if assignment.branch_name:
                    task = asyncio.create_task(self._run_assignment(assignment, run_nomic_fn))
                    tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

        # Attempt to merge completed branches
        merged_count = 0
        if self.config.auto_merge_safe:
            for assignment in assignments:
                if assignment.status == "completed" and assignment.branch_name:
                    merge_result = await self.safe_merge(
                        assignment.branch_name,
                        dry_run=False,
                    )
                    if merge_result.success:
                        assignment.status = "merged"
                        merged_count += 1
                    else:
                        logger.warning(
                            f"Could not auto-merge {assignment.branch_name}: {merge_result.error}"
                        )

        # Compute result
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        completed = sum(1 for a in assignments if a.status in ("completed", "merged"))
        failed = sum(1 for a in assignments if a.status == "failed")

        return CoordinationResult(
            total_branches=len(assignments),
            completed_branches=completed,
            failed_branches=failed,
            merged_branches=merged_count,
            assignments=assignments,
            duration_seconds=duration,
            success=failed == 0,
            summary=self._generate_summary(assignments),
        )

    async def _run_assignment(
        self,
        assignment: TrackAssignment,
        run_nomic_fn: Callable[[TrackAssignment], Any],
    ) -> None:
        """Run a single assignment on its branch."""
        if not assignment.branch_name:
            return

        assignment.status = "running"
        assignment.started_at = datetime.now(timezone.utc)

        try:
            # Checkout the branch
            self._run_git("checkout", assignment.branch_name)

            # Run the nomic loop
            result = await run_nomic_fn(assignment)

            assignment.status = "completed"
            assignment.result = result

        except Exception as e:
            logger.exception(f"Assignment failed: {e}")
            assignment.status = "failed"
            assignment.error = str(e)

        finally:
            assignment.completed_at = datetime.now(timezone.utc)
            # Return to base branch
            self._run_git("checkout", self.config.base_branch, check=False)

    def _generate_summary(self, assignments: List[TrackAssignment]) -> str:
        """Generate a summary of coordination result."""
        lines = ["Branch Coordination Summary:", ""]

        by_status: Dict[str, List[str]] = {}
        for a in assignments:
            if a.status not in by_status:
                by_status[a.status] = []
            by_status[a.status].append(f"{a.goal.track.value}: {a.goal.description[:40]}")

        for status, items in by_status.items():
            icon = {"completed": "+", "merged": "++", "failed": "!", "running": "~"}.get(
                status, "-"
            )
            lines.append(f"{status.upper()}:")
            for item in items:
                lines.append(f"  {icon} {item}")
            lines.append("")

        return "\n".join(lines)

    def _slugify(self, text: str) -> str:
        """Convert text to a branch-name-friendly slug."""
        import re

        slug = text.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug

    def cleanup_branches(self, branches: Optional[List[str]] = None) -> int:
        """Delete merged or stale branches.

        Args:
            branches: Specific branches to clean up (default: all active)

        Returns:
            Number of branches deleted
        """
        branches = branches or self._active_branches
        deleted = 0

        for branch in branches:
            if not self.branch_exists(branch):
                continue

            # Check if merged into main
            result = self._run_git(
                "branch",
                "--merged",
                self.config.base_branch,
                check=False,
            )
            merged_branches = result.stdout.strip().split("\n")

            if branch in merged_branches or f"  {branch}" in merged_branches:
                self._run_git("branch", "-d", branch, check=False)
                deleted += 1
                logger.info(f"Deleted merged branch: {branch}")

        return deleted


__all__ = [
    "BranchCoordinator",
    "BranchCoordinatorConfig",
    "TrackAssignment",
    "ConflictReport",
    "MergeResult",
    "CoordinationResult",
]
