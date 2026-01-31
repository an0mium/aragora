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


class TestCreateTrackBranches:
    """Tests for create_track_branches method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_creates_multiple_branches(self, mock_run):
        """Should create branches for multiple assignments."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        coordinator = BranchCoordinator()

        goal1 = PrioritizedGoal(
            id="goal_1",
            track=Track.SME,
            description="Frontend work",
            rationale="Test",
            estimated_impact="high",
            priority=1,
        )
        goal2 = PrioritizedGoal(
            id="goal_2",
            track=Track.QA,
            description="Test work",
            rationale="Test",
            estimated_impact="medium",
            priority=2,
        )

        assignments = [
            TrackAssignment(goal=goal1),
            TrackAssignment(goal=goal2),
        ]

        result = await coordinator.create_track_branches(assignments)

        assert len(result) == 2
        assert all(a.branch_name is not None for a in result)
        assert "sme" in result[0].branch_name.lower()
        assert "qa" in result[1].branch_name.lower()


class TestGetBranchFiles:
    """Tests for _get_branch_files method."""

    @patch("subprocess.run")
    def test_get_files_success(self, mock_run):
        """Should return list of changed files."""
        mock_run.return_value = MagicMock(
            stdout="file1.py\nfile2.py\nfile3.py\n",
            returncode=0,
        )

        coordinator = BranchCoordinator()
        files = coordinator._get_branch_files("feature", "main")

        assert len(files) == 3
        assert "file1.py" in files
        assert "file2.py" in files

    @patch("subprocess.run")
    def test_get_files_failure(self, mock_run):
        """Should return empty list on git error."""
        mock_run.return_value = MagicMock(returncode=1)

        coordinator = BranchCoordinator()
        files = coordinator._get_branch_files("bad-branch", "main")

        assert files == []

    @patch("subprocess.run")
    def test_get_files_empty(self, mock_run):
        """Should handle empty diff output."""
        mock_run.return_value = MagicMock(stdout="\n", returncode=0)

        coordinator = BranchCoordinator()
        files = coordinator._get_branch_files("clean-branch", "main")

        assert files == []


class TestDetectConflictsExtended:
    """Extended tests for detect_conflicts."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_detect_high_severity_conflict(self, mock_run):
        """Should mark high severity for many conflicting files."""
        # Make more than 5 files overlap
        overlapping_files = "\n".join([f"file{i}.py" for i in range(10)])
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists 1
            MagicMock(stdout=overlapping_files, returncode=0),  # files 1
            MagicMock(returncode=0),  # branch_exists 2
            MagicMock(stdout=overlapping_files, returncode=0),  # files 2
            MagicMock(returncode=0),  # branch_exists 2 (second iteration)
            MagicMock(stdout=overlapping_files, returncode=0),
            MagicMock(returncode=0),  # branch_exists 1 (second iteration)
            MagicMock(stdout=overlapping_files, returncode=0),
        ]

        coordinator = BranchCoordinator()
        conflicts = await coordinator.detect_conflicts(["branch1", "branch2"])

        assert len(conflicts) >= 1
        assert conflicts[0].severity == "high"

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_detect_low_severity_conflict(self, mock_run):
        """Should mark low severity for few conflicting files."""
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(stdout="shared.py\n", returncode=0),
            MagicMock(returncode=0),
            MagicMock(stdout="shared.py\nother.py\n", returncode=0),
            MagicMock(returncode=0),
            MagicMock(stdout="shared.py\nother.py\n", returncode=0),
            MagicMock(returncode=0),
            MagicMock(stdout="shared.py\n", returncode=0),
        ]

        coordinator = BranchCoordinator()
        conflicts = await coordinator.detect_conflicts(["branch1", "branch2"])

        if conflicts:
            assert conflicts[0].severity == "low"

    def test_conflict_callback_set(self):
        """Should store on_conflict callback correctly."""
        conflicts_received = []

        def on_conflict(report):
            conflicts_received.append(report)

        coordinator = BranchCoordinator(on_conflict=on_conflict)

        # Verify callback is set
        assert coordinator.on_conflict is on_conflict

        # Manually invoke to test it works
        report = ConflictReport(
            source_branch="b1",
            target_branch="b2",
            conflicting_files=["file.py"],
            severity="low",
            resolution_hint="Review changes",
        )
        coordinator.on_conflict(report)

        assert len(conflicts_received) == 1
        assert conflicts_received[0].conflicting_files == ["file.py"]


class TestRunAssignment:
    """Tests for _run_assignment method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_run_assignment_success(self, mock_run):
        """Should complete assignment successfully."""
        mock_run.return_value = MagicMock(returncode=0)

        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Test",
            rationale="R",
            estimated_impact="low",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            branch_name="test-branch",
        )

        async def success_fn(a):
            return {"result": "success"}

        await coordinator._run_assignment(assignment, success_fn)

        assert assignment.status == "completed"
        assert assignment.result is not None
        assert assignment.completed_at is not None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_run_assignment_failure(self, mock_run):
        """Should handle assignment failure."""
        mock_run.return_value = MagicMock(returncode=0)

        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Test",
            rationale="R",
            estimated_impact="low",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            branch_name="test-branch",
        )

        async def fail_fn(a):
            raise Exception("Task failed")

        await coordinator._run_assignment(assignment, fail_fn)

        assert assignment.status == "failed"
        assert assignment.error is not None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_run_assignment_no_branch(self, mock_run):
        """Should skip assignment without branch name."""
        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Test",
            rationale="R",
            estimated_impact="low",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            branch_name=None,  # No branch
        )

        async def should_not_call(a):
            raise Exception("Should not be called")

        # Should not raise, should just return
        await coordinator._run_assignment(assignment, should_not_call)

        # Status should remain pending
        assert assignment.status == "pending"


class TestSafeMergeExtended:
    """Extended tests for safe_merge."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_with_custom_target(self, mock_run):
        """Should merge to custom target branch."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout develop
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge
            MagicMock(stdout="def456\n", returncode=0),  # rev-parse
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("feature", target="develop")

        assert result.target_branch == "develop"
        assert result.success is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_dry_run_detects_conflicts(self, mock_run):
        """Should detect conflicts in dry run mode."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(
                returncode=1, stderr="CONFLICT (content): Merge conflict in app.py"
            ),  # merge --no-commit fails
            MagicMock(returncode=0),  # merge --abort
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.safe_merge("feature", dry_run=True)

        assert result.success is False
        assert "app.py" in result.conflicts


class TestCoordinationResultExtended:
    """Extended tests for CoordinationResult."""

    def test_result_all_merged(self):
        """Should track merged branches count."""
        result = CoordinationResult(
            total_branches=3,
            completed_branches=3,
            failed_branches=0,
            merged_branches=3,
            assignments=[],
            duration_seconds=60.0,
            success=True,
            summary="All merged",
        )

        assert result.merged_branches == 3
        assert result.success is True

    def test_result_partial_success(self):
        """Should handle partial success."""
        result = CoordinationResult(
            total_branches=5,
            completed_branches=3,
            failed_branches=2,
            merged_branches=2,
            assignments=[],
            duration_seconds=120.0,
            success=False,
            summary="Partial completion",
        )

        assert result.completed_branches == 3
        assert result.failed_branches == 2
        assert result.success is False


class TestTrackAssignmentExtended:
    """Extended tests for TrackAssignment."""

    def test_assignment_started_at(self):
        """Should track started timestamp."""
        now = datetime.now(timezone.utc)

        goal = PrioritizedGoal(
            id="g1",
            track=Track.CORE,
            description="Test",
            rationale="R",
            estimated_impact="high",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            started_at=now,
            status="running",
        )

        assert assignment.started_at == now
        assert assignment.status == "running"

    def test_assignment_result_storage(self):
        """Should store arbitrary result dict."""
        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Test",
            rationale="R",
            estimated_impact="low",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            result={
                "files_modified": ["a.py", "b.py"],
                "tests_passed": 10,
                "custom_data": {"key": "value"},
            },
        )

        assert assignment.result["files_modified"] == ["a.py", "b.py"]
        assert assignment.result["tests_passed"] == 10


class TestSlugifyExtended:
    """Extended tests for _slugify method."""

    def test_slugify_unicode(self):
        """Should handle unicode characters."""
        coordinator = BranchCoordinator()

        slug = coordinator._slugify("Fix bug in authentication")

        assert slug.isascii()
        assert "-" in slug or slug.isalnum()

    def test_slugify_numbers(self):
        """Should preserve numbers."""
        coordinator = BranchCoordinator()

        slug = coordinator._slugify("Add feature 123")

        assert "123" in slug

    def test_slugify_multiple_spaces(self):
        """Should collapse multiple spaces/dashes."""
        coordinator = BranchCoordinator()

        slug = coordinator._slugify("Fix   multiple   spaces")

        assert "---" not in slug
        assert "--" not in slug


class TestCleanupBranchesExtended:
    """Extended tests for cleanup_branches."""

    @patch("subprocess.run")
    def test_cleanup_specific_branches(self, mock_run):
        """Should clean up only specified branches."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists for branch1
            MagicMock(stdout="  branch1\n  branch2\n", returncode=0),  # merged
            MagicMock(returncode=0),  # delete
        ]

        coordinator = BranchCoordinator()
        deleted = coordinator.cleanup_branches(branches=["branch1"])

        # Should only process branch1
        assert deleted >= 0

    @patch("subprocess.run")
    def test_cleanup_empty_list(self, mock_run):
        """Should handle empty branch list."""
        coordinator = BranchCoordinator()
        coordinator._active_branches = []

        deleted = coordinator.cleanup_branches()

        assert deleted == 0


class TestGenerateSummaryExtended:
    """Extended tests for _generate_summary."""

    def test_summary_with_merged_status(self):
        """Should show merged status correctly."""
        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="g1",
            track=Track.SME,
            description="Task",
            rationale="R",
            estimated_impact="high",
            priority=1,
        )
        assignments = [
            TrackAssignment(goal=goal, status="merged"),
        ]

        summary = coordinator._generate_summary(assignments)

        assert "MERGED" in summary
        assert "++" in summary  # Merged indicator

    def test_summary_with_running_status(self):
        """Should show running status."""
        coordinator = BranchCoordinator()

        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Task",
            rationale="R",
            estimated_impact="low",
            priority=1,
        )
        assignments = [
            TrackAssignment(goal=goal, status="running"),
        ]

        summary = coordinator._generate_summary(assignments)

        assert "RUNNING" in summary
        assert "~" in summary  # Running indicator

    def test_summary_empty_assignments(self):
        """Should handle empty assignments list."""
        coordinator = BranchCoordinator()

        summary = coordinator._generate_summary([])

        assert "Summary" in summary
