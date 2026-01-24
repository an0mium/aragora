"""Tests for the AutonomousOrchestrator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aragora.nomic.autonomous_orchestrator import (
    AutonomousOrchestrator,
    AgentRouter,
    FeedbackLoop,
    Track,
    TrackConfig,
    AgentAssignment,
    OrchestrationResult,
    DEFAULT_TRACK_CONFIGS,
    reset_orchestrator,
)
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset orchestrator singleton before each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


class TestAgentRouter:
    """Tests for the AgentRouter class."""

    def test_determine_track_from_file_scope(self):
        """Router should determine track from file paths."""
        router = AgentRouter()

        # Frontend file -> SME track
        subtask = SubTask(
            id="1",
            title="Add button",
            description="Add logout button",
            file_scope=["aragora/live/src/app/dashboard/page.tsx"],
        )
        assert router.determine_track(subtask) == Track.SME

        # SDK file -> Developer track
        subtask = SubTask(
            id="2",
            title="Add SDK method",
            description="Add create_debate method",
            file_scope=["sdk/python/aragora_sdk/client.py"],
        )
        assert router.determine_track(subtask) == Track.DEVELOPER

        # Docker file -> Self-Hosted track
        subtask = SubTask(
            id="3",
            title="Update Dockerfile",
            description="Optimize build",
            file_scope=["docker/Dockerfile.production"],
        )
        assert router.determine_track(subtask) == Track.SELF_HOSTED

        # Test file -> QA track
        subtask = SubTask(
            id="4",
            title="Add tests",
            description="Add unit tests",
            file_scope=["tests/server/test_auth.py"],
        )
        assert router.determine_track(subtask) == Track.QA

    def test_determine_track_from_keywords(self):
        """Router should determine track from task description when no file scope."""
        router = AgentRouter()

        # UI keywords -> SME
        subtask = SubTask(
            id="1",
            title="Improve dashboard UI",
            description="Make the user interface more intuitive",
            file_scope=[],
        )
        assert router.determine_track(subtask) == Track.SME

        # API keywords -> Developer
        subtask = SubTask(
            id="2",
            title="API Documentation",
            description="Document the REST API endpoints",
            file_scope=[],
        )
        assert router.determine_track(subtask) == Track.DEVELOPER

        # Deploy keywords -> Self-Hosted
        subtask = SubTask(
            id="3",
            title="Kubernetes Setup",
            description="Create deployment manifests for kubernetes",
            file_scope=[],
        )
        assert router.determine_track(subtask) == Track.SELF_HOSTED

        # Test keywords -> QA
        subtask = SubTask(
            id="4",
            title="Increase Coverage",
            description="Add e2e tests for critical paths",
            file_scope=[],
        )
        assert router.determine_track(subtask) == Track.QA

        # Debate keywords -> Core
        subtask = SubTask(
            id="5",
            title="Consensus Algorithm",
            description="Improve debate consensus detection",
            file_scope=[],
        )
        assert router.determine_track(subtask) == Track.CORE

    def test_select_agent_type_for_complexity(self):
        """Router should select appropriate agent based on complexity."""
        router = AgentRouter()

        # High complexity -> Claude
        subtask = SubTask(
            id="1",
            title="Refactor",
            description="Major refactoring",
            estimated_complexity="high",
        )
        agent = router.select_agent_type(subtask, Track.DEVELOPER)
        assert agent == "claude"

        # Implementation task with codex available -> Codex
        subtask = SubTask(
            id="2",
            title="Implement Feature",
            description="Implement new code feature",
            estimated_complexity="medium",
        )
        agent = router.select_agent_type(subtask, Track.DEVELOPER)
        assert agent == "codex"

        # Low complexity -> first available
        subtask = SubTask(
            id="3",
            title="Fix Typo",
            description="Fix documentation typo",
            estimated_complexity="low",
        )
        agent = router.select_agent_type(subtask, Track.QA)
        assert agent == "claude"  # First in QA's agent_types

    def test_check_conflicts_file_overlap(self):
        """Router should detect file conflicts between assignments."""
        router = AgentRouter()

        active_subtask = SubTask(
            id="1",
            title="Active Task",
            description="Working on auth",
            file_scope=["aragora/server/handlers/auth.py"],
        )
        active = AgentAssignment(
            subtask=active_subtask,
            track=Track.SME,
            agent_type="claude",
            status="running",
        )

        # Overlapping file -> conflict
        new_subtask = SubTask(
            id="2",
            title="New Task",
            description="Also modifying auth",
            file_scope=["aragora/server/handlers/auth.py"],
        )
        conflicts = router.check_conflicts(new_subtask, [active])
        assert len(conflicts) == 1
        assert "File conflict" in conflicts[0]

        # Non-overlapping file -> no conflict
        new_subtask2 = SubTask(
            id="3",
            title="Different Task",
            description="Modifying different file",
            file_scope=["aragora/server/handlers/debate.py"],
        )
        conflicts = router.check_conflicts(new_subtask2, [active])
        assert len(conflicts) == 0

    def test_check_conflicts_core_track(self):
        """Router should prevent parallel core track tasks."""
        router = AgentRouter()

        core_subtask = SubTask(
            id="1",
            title="Core Work",
            description="Working on debate consensus algorithm",
            file_scope=["aragora/debate/consensus.py"],
        )
        active = AgentAssignment(
            subtask=core_subtask,
            track=Track.CORE,
            agent_type="claude",
            status="running",
        )

        # Another core task -> conflict
        new_core = SubTask(
            id="2",
            title="More Core Work",
            description="Working on arena memory integration",
            file_scope=["aragora/debate/memory_manager.py"],
        )
        conflicts = router.check_conflicts(new_core, [active])
        # Note: This will detect as core because of keywords, and there's already
        # a running core task
        assert any("Core track conflict" in c for c in conflicts)


class TestFeedbackLoop:
    """Tests for the FeedbackLoop class."""

    def test_analyze_test_failure(self):
        """FeedbackLoop should recommend retry for test failures."""
        loop = FeedbackLoop(max_iterations=3)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "test_failure", "message": "AssertionError: Expected 5, got 3"},
        )

        assert feedback["action"] == "retry_implement"
        assert "AssertionError" in feedback.get("hints", "")

    def test_analyze_lint_error(self):
        """FeedbackLoop should recommend quick fix for lint errors."""
        loop = FeedbackLoop(max_iterations=3)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "lint_error", "message": "Missing trailing comma"},
        )

        assert feedback["action"] == "quick_fix"

    def test_analyze_design_issue(self):
        """FeedbackLoop should recommend redesign for design issues."""
        loop = FeedbackLoop(max_iterations=3)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "design_issue", "suggestion": "Use factory pattern"},
        )

        assert feedback["action"] == "redesign"

    def test_escalate_after_max_iterations(self):
        """FeedbackLoop should escalate after max iterations."""
        loop = FeedbackLoop(max_iterations=2)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        # First iteration
        loop.analyze_failure(assignment, {"type": "test_failure", "message": ""})

        # Second iteration
        loop.analyze_failure(assignment, {"type": "test_failure", "message": ""})

        # Third iteration should escalate
        feedback = loop.analyze_failure(
            assignment,
            {"type": "test_failure", "message": ""},
        )

        assert feedback["action"] == "escalate"
        assert feedback["require_human"] is True


class TestAutonomousOrchestrator:
    """Tests for the AutonomousOrchestrator class."""

    @pytest.fixture
    def mock_workflow_engine(self):
        """Create a mock workflow engine."""
        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                final_output={"status": "completed"},
                error=None,
            )
        )
        return engine

    @pytest.fixture
    def mock_task_decomposer(self):
        """Create a mock task decomposer."""
        decomposer = MagicMock()
        decomposer.analyze = MagicMock(
            return_value=TaskDecomposition(
                original_task="Test goal",
                complexity_score=5,
                complexity_level="medium",
                should_decompose=True,
                subtasks=[
                    SubTask(
                        id="1",
                        title="Frontend Changes",
                        description="Update dashboard UI",
                        file_scope=["aragora/live/src/app/dashboard/page.tsx"],
                        estimated_complexity="medium",
                    ),
                    SubTask(
                        id="2",
                        title="Add Tests",
                        description="Add e2e tests for dashboard",
                        file_scope=["aragora/live/e2e/dashboard.spec.ts"],
                        estimated_complexity="low",
                    ),
                ],
            )
        )
        return decomposer

    @pytest.mark.asyncio
    async def test_execute_goal_decomposes_and_assigns(
        self, mock_workflow_engine, mock_task_decomposer
    ):
        """Orchestrator should decompose goal and create assignments."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        result = await orchestrator.execute_goal(
            goal="Improve dashboard for SME users",
            tracks=["sme", "qa"],
            max_cycles=3,
        )

        assert result.total_subtasks == 2
        assert result.completed_subtasks == 2
        assert result.success is True

        # Check task decomposer was called
        mock_task_decomposer.analyze.assert_called_once()

        # Check workflow engine was called for each subtask
        assert mock_workflow_engine.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_goal_filters_by_track(self, mock_workflow_engine, mock_task_decomposer):
        """Orchestrator should only process tasks for specified tracks."""
        # Add a core subtask that should be filtered out
        mock_task_decomposer.analyze.return_value.subtasks.append(
            SubTask(
                id="3",
                title="Core Changes",
                description="Modify debate consensus algorithm",
                file_scope=["aragora/debate/consensus.py"],
                estimated_complexity="high",
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        result = await orchestrator.execute_goal(
            goal="Improve everything",
            tracks=["sme", "qa"],  # Exclude core
            max_cycles=3,
        )

        # Core task should be filtered out
        assert result.total_subtasks == 2

    @pytest.mark.asyncio
    async def test_execute_goal_handles_failure(self, mock_workflow_engine, mock_task_decomposer):
        """Orchestrator should handle workflow failures gracefully."""
        mock_workflow_engine.execute = AsyncMock(
            return_value=MagicMock(
                success=False,
                final_output=None,
                error="Test failure",
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            max_parallel_tasks=1,
        )

        result = await orchestrator.execute_goal(
            goal="Test goal",
            max_cycles=1,
        )

        # Should complete but with failures
        assert result.success is False
        assert result.failed_subtasks > 0

    @pytest.mark.asyncio
    async def test_execute_track_convenience_method(
        self, mock_workflow_engine, mock_task_decomposer
    ):
        """execute_track should work as a convenience wrapper."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        result = await orchestrator.execute_track(
            track="sme",
            focus_areas=["dashboard", "settings"],
            max_cycles=2,
        )

        assert result is not None
        # The goal should mention the track
        mock_task_decomposer.analyze.assert_called_once()
        call_arg = mock_task_decomposer.analyze.call_args[0][0]
        assert "sme" in call_arg.lower()

    def test_checkpoint_callback(self, mock_workflow_engine, mock_task_decomposer):
        """Orchestrator should call checkpoint callback at key phases."""
        checkpoints = []

        def on_checkpoint(phase, data):
            checkpoints.append((phase, data))

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            on_checkpoint=on_checkpoint,
        )

        # Run synchronously for checkpoint test
        import asyncio

        asyncio.run(orchestrator.execute_goal(goal="Test", max_cycles=1))

        # Should have checkpoints for: started, decomposed, assigned, completed
        phases = [c[0] for c in checkpoints]
        assert "started" in phases
        assert "decomposed" in phases
        assert "assigned" in phases
        assert "completed" in phases


class TestTrackConfig:
    """Tests for track configuration."""

    def test_default_track_configs_exist(self):
        """All tracks should have default configurations."""
        for track in Track:
            assert track in DEFAULT_TRACK_CONFIGS
            config = DEFAULT_TRACK_CONFIGS[track]
            assert config.name
            assert len(config.folders) > 0
            assert len(config.agent_types) > 0

    def test_sme_track_config(self):
        """SME track should focus on frontend and handlers."""
        config = DEFAULT_TRACK_CONFIGS[Track.SME]
        assert "aragora/live/" in config.folders
        assert "aragora/server/handlers/" in config.folders
        assert "aragora/debate/" in config.protected_folders

    def test_core_track_requires_claude(self):
        """Core track should only use Claude."""
        config = DEFAULT_TRACK_CONFIGS[Track.CORE]
        assert config.agent_types == ["claude"]
        assert config.max_concurrent_tasks == 1  # Safety limit


class TestOrchestrationResult:
    """Tests for OrchestrationResult dataclass."""

    def test_result_fields(self):
        """OrchestrationResult should have all expected fields."""
        result = OrchestrationResult(
            goal="Test goal",
            total_subtasks=5,
            completed_subtasks=4,
            failed_subtasks=1,
            skipped_subtasks=0,
            assignments=[],
            duration_seconds=120.5,
            success=False,
            error="One task failed",
            summary="Test summary",
        )

        assert result.goal == "Test goal"
        assert result.total_subtasks == 5
        assert result.completed_subtasks == 4
        assert result.failed_subtasks == 1
        assert result.success is False
        assert result.error == "One task failed"
