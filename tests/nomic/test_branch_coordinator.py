"""Tests for BranchCoordinator - parallel branch management."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from datetime import datetime, timezone

from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    TrackAssignment,
    ConflictReport,
    MergeResult,
    CoordinationResult,
)
from aragora.nomic.meta_planner import PrioritizedGoal, Track


class TestTrackAssignment:
    """Tests for TrackAssignment dataclass."""

    def test_assignment_creation(self):
        """Should create assignment with required fields."""
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.SME,
            description="Improve dashboard",
            rationale="Better UX",
            estimated_impact="high",
            priority=1,
        )
        assignment = TrackAssignment(goal=goal)

        assert assignment.goal == goal
        assert assignment.branch_name is None
        assert assignment.status == "pending"
        assert assignment.started_at is None
        assert assignment.completed_at is None
        assert assignment.result is None
        assert assignment.error is None

    def test_assignment_with_all_fields(self):
        """Should accept all optional fields."""
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.QA,
            description="Add tests",
            rationale="Coverage",
            estimated_impact="medium",
            priority=2,
        )
        now = datetime.now(timezone.utc)
        assignment = TrackAssignment(
            goal=goal,
            branch_name="dev/qa-add-tests-0124",
            status="completed",
            started_at=now,
            completed_at=now,
            result={"success": True},
            error=None,
        )

        assert assignment.branch_name == "dev/qa-add-tests-0124"
        assert assignment.status == "completed"


class TestConflictReport:
    """Tests for ConflictReport dataclass."""

    def test_conflict_report_creation(self):
        """Should create conflict report with all fields."""
        report = ConflictReport(
            source_branch="dev/sme-dashboard",
            target_branch="dev/qa-tests",
            conflicting_files=["handlers.py", "models.py"],
            severity="medium",
            resolution_hint="Review Python module changes",
        )

        assert report.source_branch == "dev/sme-dashboard"
        assert report.target_branch == "dev/qa-tests"
        assert len(report.conflicting_files) == 2
        assert report.severity == "medium"
        assert "Python" in report.resolution_hint


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_successful_merge(self):
        """Should represent successful merge."""
        result = MergeResult(
            source_branch="dev/feature",
            target_branch="main",
            success=True,
            commit_sha="abc123",
        )

        assert result.success is True
        assert result.commit_sha == "abc123"
        assert result.error is None
        assert result.conflicts == []

    def test_failed_merge(self):
        """Should represent failed merge."""
        result = MergeResult(
            source_branch="dev/feature",
            target_branch="main",
            success=False,
            error="Merge conflicts detected",
            conflicts=["handlers.py"],
        )

        assert result.success is False
        assert "conflicts" in result.error.lower()
        assert "handlers.py" in result.conflicts


class TestCoordinationResult:
    """Tests for CoordinationResult dataclass."""

    def test_coordination_result(self):
        """Should summarize coordination outcome."""
        result = CoordinationResult(
            total_branches=3,
            completed_branches=2,
            failed_branches=1,
            merged_branches=1,
            assignments=[],
            duration_seconds=120.5,
            success=False,
            summary="Summary text",
        )

        assert result.total_branches == 3
        assert result.completed_branches == 2
        assert result.failed_branches == 1
        assert result.success is False


class TestBranchCoordinatorConfig:
    """Tests for BranchCoordinatorConfig dataclass."""

    def test_config_defaults(self):
        """Should have sensible defaults."""
        config = BranchCoordinatorConfig()

        assert config.base_branch == "main"
        assert config.branch_prefix == "dev"
        assert config.auto_merge_safe is True
        assert config.require_tests_pass is True
        assert config.max_parallel_branches == 3

    def test_config_custom(self):
        """Should accept custom values."""
        config = BranchCoordinatorConfig(
            base_branch="develop",
            branch_prefix="feature",
            auto_merge_safe=False,
        )

        assert config.base_branch == "develop"
        assert config.branch_prefix == "feature"
        assert config.auto_merge_safe is False


class TestBranchCoordinator:
    """Tests for BranchCoordinator class."""

    def test_init_defaults(self):
        """Should initialize with defaults."""
        coordinator = BranchCoordinator()

        assert coordinator.repo_path == Path.cwd()
        assert coordinator.config.base_branch == "main"
        assert coordinator.on_conflict is None
        assert coordinator._active_branches == []

    def test_init_custom(self):
        """Should accept custom config."""
        config = BranchCoordinatorConfig(base_branch="develop")
        callback = MagicMock()
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/repo"),
            config=config,
            on_conflict=callback,
        )

        assert coordinator.repo_path == Path("/tmp/repo")
        assert coordinator.config.base_branch == "develop"
        assert coordinator.on_conflict == callback


class TestSlugify:
    """Tests for _slugify method."""

    def test_slugify_simple(self):
        """Should convert simple text to slug."""
        coordinator = BranchCoordinator()
        slug = coordinator._slugify("Improve dashboard")

        assert slug == "improve-dashboard"

    def test_slugify_special_chars(self):
        """Should remove special characters."""
        coordinator = BranchCoordinator()
        slug = coordinator._slugify("Fix bug #123 (critical)")

        assert "#" not in slug
        assert "(" not in slug
        assert ")" not in slug

    def test_slugify_strips_dashes(self):
        """Should strip leading/trailing dashes."""
        coordinator = BranchCoordinator()
        slug = coordinator._slugify("---test---")

        assert not slug.startswith("-")
        assert not slug.endswith("-")


class TestRunGit:
    """Tests for _run_git method."""

    @patch("subprocess.run")
    def test_run_git_basic(self, mock_run):
        """Should run git command."""
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )
        coordinator = BranchCoordinator(repo_path=Path("/tmp/repo"))
        result = coordinator._run_git("status")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["git", "status"]
        assert call_args[1]["cwd"] == Path("/tmp/repo")

    @patch("subprocess.run")
    def test_run_git_with_args(self, mock_run):
        """Should pass arguments correctly."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()
        coordinator._run_git("checkout", "-b", "feature/test")

        call_args = mock_run.call_args
        assert call_args[0][0] == ["git", "checkout", "-b", "feature/test"]


class TestGetCurrentBranch:
    """Tests for get_current_branch method."""

    @patch("subprocess.run")
    def test_get_current_branch(self, mock_run):
        """Should return current branch name."""
        mock_run.return_value = MagicMock(
            stdout="main\n",
            returncode=0,
        )
        coordinator = BranchCoordinator()
        branch = coordinator.get_current_branch()

        assert branch == "main"


class TestBranchExists:
    """Tests for branch_exists method."""

    @patch("subprocess.run")
    def test_branch_exists_true(self, mock_run):
        """Should return True for existing branch."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()

        assert coordinator.branch_exists("main") is True

    @patch("subprocess.run")
    def test_branch_exists_false(self, mock_run):
        """Should return False for non-existing branch."""
        mock_run.return_value = MagicMock(returncode=1)
        coordinator = BranchCoordinator()

        assert coordinator.branch_exists("nonexistent") is False


class TestCreateTrackBranch:
    """Tests for create_track_branch async method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_create_new_branch(self, mock_run):
        """Should create new branch for track."""
        mock_run.return_value = MagicMock(
            stdout="main",
            returncode=0,
        )
        # Make branch_exists return False (new branch)
        mock_run.side_effect = [
            MagicMock(stdout="main\n", returncode=0),  # get_current_branch
            MagicMock(returncode=1),  # branch_exists (doesn't exist)
            MagicMock(returncode=0),  # checkout -b
        ]

        coordinator = BranchCoordinator()
        branch = await coordinator.create_track_branch(
            track=Track.SME,
            goal="Improve dashboard",
        )

        assert branch.startswith("dev/sme-")
        assert "improve-dashboard" in branch
        assert branch in coordinator._active_branches

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_create_branch_existing(self, mock_run):
        """Should use existing branch if present."""
        mock_run.side_effect = [
            MagicMock(stdout="main\n", returncode=0),  # get_current_branch
            MagicMock(returncode=0),  # branch_exists (exists)
            MagicMock(returncode=0),  # checkout existing
        ]

        coordinator = BranchCoordinator()
        branch = await coordinator.create_track_branch(
            track=Track.QA,
            goal="Add tests",
        )

        assert branch.startswith("dev/qa-")


class TestDetectConflicts:
    """Tests for detect_conflicts async method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_detect_no_conflicts(self, mock_run):
        """Should return empty list when no conflicts."""
        # The detect_conflicts method iterates twice:
        # First for branch1 checking against branch2, then branch2 against branch1
        mock_run.side_effect = [
            # First iteration: branch1
            MagicMock(returncode=0),  # branch_exists(branch1)
            MagicMock(stdout="file1.py\n", returncode=0),  # get_branch_files(branch1)
            MagicMock(returncode=0),  # branch_exists(branch2) - inner loop
            MagicMock(stdout="file2.py\n", returncode=0),  # get_branch_files(branch2)
            # Second iteration: branch2
            MagicMock(returncode=0),  # branch_exists(branch2)
            MagicMock(stdout="file2.py\n", returncode=0),  # get_branch_files(branch2)
            MagicMock(returncode=0),  # branch_exists(branch1) - inner loop
            MagicMock(stdout="file1.py\n", returncode=0),  # get_branch_files(branch1)
        ]

        coordinator = BranchCoordinator()
        conflicts = await coordinator.detect_conflicts(["branch1", "branch2"])

        assert conflicts == []

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_detect_conflicts_overlap(self, mock_run):
        """Should detect overlapping files."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists 1
            MagicMock(stdout="handlers.py\nmodels.py\n", returncode=0),
            MagicMock(returncode=0),  # branch_exists 2
            MagicMock(stdout="handlers.py\nauth.py\n", returncode=0),
            MagicMock(returncode=0),  # branch_exists 2 (second iteration)
            MagicMock(stdout="handlers.py\nauth.py\n", returncode=0),
            MagicMock(returncode=0),  # branch_exists 1 (second iteration)
            MagicMock(stdout="handlers.py\nmodels.py\n", returncode=0),
        ]

        coordinator = BranchCoordinator()
        conflicts = await coordinator.detect_conflicts(["branch1", "branch2"])

        assert len(conflicts) >= 1
        assert "handlers.py" in conflicts[0].conflicting_files


class TestGenerateResolutionHint:
    """Tests for _generate_resolution_hint method."""

    def test_hint_for_test_files(self):
        """Should suggest separate test merging."""
        coordinator = BranchCoordinator()
        hint = coordinator._generate_resolution_hint(["test_auth.py", "test_handlers.py"])

        assert "test" in hint.lower()

    def test_hint_for_python_files(self):
        """Should suggest Python review."""
        coordinator = BranchCoordinator()
        hint = coordinator._generate_resolution_hint(["handlers.py", "models.py"])

        assert "python" in hint.lower()

    def test_hint_default(self):
        """Should give generic hint for other files."""
        coordinator = BranchCoordinator()
        hint = coordinator._generate_resolution_hint(["config.json"])

        assert "manual" in hint.lower()


class TestSafeMerge:
    """Tests for safe_merge async method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_safe_merge_nonexistent_branch(self, mock_run):
        """Should fail for nonexistent branch."""
        mock_run.return_value = MagicMock(returncode=1)

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("nonexistent")

        assert result.success is False
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_safe_merge_dry_run(self, mock_run):
        """Should check merge without committing."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout target
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge --no-commit
            MagicMock(returncode=0),  # merge --abort
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("feature", dry_run=True)

        assert result.success is True
        assert result.commit_sha is None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_safe_merge_success(self, mock_run):
        """Should complete successful merge."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge
            MagicMock(stdout="abc123\n", returncode=0),  # rev-parse
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("feature")

        assert result.success is True
        assert result.commit_sha == "abc123"

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_safe_merge_conflict(self, mock_run):
        """Should handle merge conflicts."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(
                returncode=1,
                stderr="CONFLICT (content): Merge conflict in handlers.py",
            ),  # merge fails
            MagicMock(returncode=0),  # merge --abort
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("feature")

        assert result.success is False
        assert "handlers.py" in result.conflicts


class TestParseMergeConflicts:
    """Tests for _parse_merge_conflicts method."""

    def test_parse_single_conflict(self):
        """Should parse single conflict."""
        coordinator = BranchCoordinator()
        stderr = "CONFLICT (content): Merge conflict in handlers.py"
        conflicts = coordinator._parse_merge_conflicts(stderr)

        assert "handlers.py" in conflicts

    def test_parse_multiple_conflicts(self):
        """Should parse multiple conflicts."""
        coordinator = BranchCoordinator()
        stderr = """CONFLICT (content): Merge conflict in handlers.py
CONFLICT (content): Merge conflict in models.py"""
        conflicts = coordinator._parse_merge_conflicts(stderr)

        assert len(conflicts) == 2

    def test_parse_no_conflicts(self):
        """Should return empty for no conflicts."""
        coordinator = BranchCoordinator()
        stderr = "Already up to date."
        conflicts = coordinator._parse_merge_conflicts(stderr)

        assert conflicts == []


class TestGenerateSummary:
    """Tests for _generate_summary method."""

    def test_summary_with_assignments(self):
        """Should generate summary from assignments."""
        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.SME,
            description="Improve dashboard for better UX",
            rationale="User request",
            estimated_impact="high",
            priority=1,
        )
        assignments = [
            TrackAssignment(goal=goal, status="completed"),
            TrackAssignment(
                goal=PrioritizedGoal(
                    id="goal_1",
                    track=Track.QA,
                    description="Add tests",
                    rationale="Coverage",
                    estimated_impact="medium",
                    priority=2,
                ),
                status="failed",
            ),
        ]

        summary = coordinator._generate_summary(assignments)

        assert "COMPLETED" in summary
        assert "FAILED" in summary
        assert "sme" in summary.lower()


class TestCleanupBranches:
    """Tests for cleanup_branches method."""

    @patch("subprocess.run")
    def test_cleanup_merged_branches(self, mock_run):
        """Should delete merged branches."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(stdout="  feature-1\n  feature-2\n", returncode=0),
            MagicMock(returncode=0),  # branch -d
        ]

        coordinator = BranchCoordinator()
        coordinator._active_branches = ["feature-1"]
        deleted = coordinator.cleanup_branches()

        assert deleted >= 0

    @patch("subprocess.run")
    def test_cleanup_nonexistent_branches(self, mock_run):
        """Should skip nonexistent branches."""
        mock_run.return_value = MagicMock(returncode=1)  # branch doesn't exist

        coordinator = BranchCoordinator()
        coordinator._active_branches = ["nonexistent"]
        deleted = coordinator.cleanup_branches()

        assert deleted == 0


class TestCoordinateParallelWork:
    """Tests for coordinate_parallel_work async method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_coordinate_creates_branches(self, mock_run):
        """Should create branches for all assignments."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)
        mock_run.side_effect = [
            MagicMock(stdout="main\n", returncode=0),
            MagicMock(returncode=1),  # branch doesn't exist
            MagicMock(returncode=0),  # checkout -b
            MagicMock(returncode=0),  # checkout main
            MagicMock(returncode=0),  # branch_exists for conflict check
            MagicMock(stdout="", returncode=0),  # get_branch_files
        ]

        coordinator = BranchCoordinator(config=BranchCoordinatorConfig(auto_merge_safe=False))
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.SME,
            description="Test goal",
            rationale="Test",
            estimated_impact="medium",
            priority=1,
        )
        assignments = [TrackAssignment(goal=goal)]

        result = await coordinator.coordinate_parallel_work(assignments)

        assert result.total_branches == 1
        assert assignments[0].branch_name is not None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_coordinate_with_nomic_fn(self, mock_run):
        """Should run nomic function on branches."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        coordinator = BranchCoordinator(config=BranchCoordinatorConfig(auto_merge_safe=False))
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.QA,
            description="Test",
            rationale="Test",
            estimated_impact="low",
            priority=1,
        )
        assignments = [TrackAssignment(goal=goal, branch_name="test-branch")]

        async def mock_nomic_fn(assignment):
            return {"success": True}

        result = await coordinator.coordinate_parallel_work(
            assignments,
            run_nomic_fn=mock_nomic_fn,
        )

        assert result.total_branches == 1
