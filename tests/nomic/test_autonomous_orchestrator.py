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


class TestAgentAssignment:
    """Tests for AgentAssignment dataclass."""

    def test_assignment_creation(self):
        """Should create assignment with required fields."""
        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        assert assignment.subtask == subtask
        assert assignment.track == Track.QA
        assert assignment.agent_type == "claude"
        assert assignment.status == "pending"
        assert assignment.attempt_count == 0
        assert assignment.max_attempts == 3

    def test_assignment_with_all_fields(self):
        """Should accept all optional fields."""
        subtask = SubTask(id="2", title="Full", description="Full task")
        now = datetime.now(timezone.utc)
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="codex",
            priority=5,
            status="completed",
            attempt_count=2,
            max_attempts=5,
            result={"success": True},
            started_at=now,
            completed_at=now,
        )

        assert assignment.priority == 5
        assert assignment.status == "completed"
        assert assignment.attempt_count == 2
        assert assignment.result["success"] is True


class TestAgentRouterExtended:
    """Extended tests for AgentRouter."""

    def test_file_to_track_caching(self):
        """Should cache file to track mappings."""
        router = AgentRouter()

        # First call should populate cache
        subtask = SubTask(
            id="1",
            title="Test",
            description="Test",
            file_scope=["aragora/live/test.py"],
        )
        track1 = router.determine_track(subtask)

        # Check cache was populated
        assert "aragora/live/test.py" in router._file_to_track_cache
        assert router._file_to_track_cache["aragora/live/test.py"] == Track.SME

    def test_select_agent_for_track_without_codex(self):
        """Should handle tracks without codex agent."""
        router = AgentRouter()

        subtask = SubTask(
            id="1",
            title="Implement Something",
            description="Code generation",
            estimated_complexity="medium",
        )

        # Core track only has claude
        agent = router.select_agent_type(subtask, Track.CORE)
        assert agent == "claude"

    def test_select_agent_empty_agent_types(self):
        """Should default to claude when track has no agents."""
        router = AgentRouter(
            track_configs={
                Track.QA: TrackConfig(
                    name="QA",
                    folders=["tests/"],
                    agent_types=[],  # Empty
                )
            }
        )

        subtask = SubTask(id="1", title="Test", description="Test")
        agent = router.select_agent_type(subtask, Track.QA)

        assert agent == "claude"

    def test_check_conflicts_with_non_running_tasks(self):
        """Should only check running tasks for conflicts."""
        router = AgentRouter()

        completed_subtask = SubTask(
            id="1",
            title="Completed",
            description="Done",
            file_scope=["shared.py"],
        )
        completed = AgentAssignment(
            subtask=completed_subtask,
            track=Track.QA,
            agent_type="claude",
            status="completed",  # Not running
        )

        new_subtask = SubTask(
            id="2",
            title="New",
            description="New task",
            file_scope=["shared.py"],
        )

        conflicts = router.check_conflicts(new_subtask, [completed])
        assert len(conflicts) == 0  # No conflict with completed tasks


class TestAgentRouterKiloCode:
    """Tests for KiloCode harness integration."""

    def test_get_coding_harness_for_claude(self):
        """Claude should not need KiloCode harness."""
        router = AgentRouter()

        harness = router.get_coding_harness("claude", Track.DEVELOPER)

        assert harness is None  # Claude has native harness

    def test_get_coding_harness_for_codex(self):
        """Codex should not need KiloCode harness."""
        router = AgentRouter()

        harness = router.get_coding_harness("codex", Track.DEVELOPER)

        assert harness is None  # Codex has native harness

    def test_get_coding_harness_for_gemini(self):
        """Gemini should get KiloCode harness."""
        router = AgentRouter()

        harness = router.get_coding_harness("gemini", Track.SME)

        assert harness is not None
        assert harness["harness"] == "kilocode"
        assert "google" in harness["provider_id"]

    def test_get_coding_harness_for_grok(self):
        """Grok should get KiloCode harness."""
        router = AgentRouter()

        harness = router.get_coding_harness("grok", Track.DEVELOPER)

        assert harness is not None
        assert harness["harness"] == "kilocode"
        assert "grok" in harness["provider_id"]

    def test_get_coding_harness_disabled_for_track(self):
        """Should not provide harness when disabled for track."""
        router = AgentRouter(
            track_configs={
                Track.CORE: TrackConfig(
                    name="Core",
                    folders=["aragora/debate/"],
                    agent_types=["gemini"],
                    use_kilocode_harness=False,  # Disabled
                )
            }
        )

        harness = router.get_coding_harness("gemini", Track.CORE)

        assert harness is None


class TestFeedbackLoopExtended:
    """Extended tests for FeedbackLoop."""

    def test_analyze_type_error(self):
        """Should recommend quick fix for type errors."""
        loop = FeedbackLoop()

        subtask = SubTask(id="1", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "type_error", "message": "Argument type mismatch"},
        )

        assert feedback["action"] == "quick_fix"

    def test_analyze_unknown_error(self):
        """Should escalate unknown error types."""
        loop = FeedbackLoop()

        subtask = SubTask(id="1", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="claude",
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "unknown_error", "message": "Something weird"},
        )

        assert feedback["action"] == "escalate"
        assert feedback["require_human"] is True

    def test_extract_test_hints_with_expected(self):
        """Should extract Expected/Actual hints."""
        loop = FeedbackLoop()

        hints = loop._extract_test_hints("Error occurred\nExpected: 5\nActual: 3\nSome other line")

        assert "Expected" in hints
        assert "Actual" in hints

    def test_extract_test_hints_empty(self):
        """Should return default hint when no patterns match."""
        loop = FeedbackLoop()

        hints = loop._extract_test_hints("Generic error message")

        assert hints == "Review test output"

    def test_iteration_tracking_per_subtask(self):
        """Should track iterations per subtask ID."""
        loop = FeedbackLoop(max_iterations=5)

        subtask1 = SubTask(id="task1", title="Task 1", description="First")
        subtask2 = SubTask(id="task2", title="Task 2", description="Second")

        assignment1 = AgentAssignment(subtask=subtask1, track=Track.QA, agent_type="claude")
        assignment2 = AgentAssignment(subtask=subtask2, track=Track.QA, agent_type="claude")

        # Iterate task1 twice
        loop.analyze_failure(assignment1, {"type": "test_failure", "message": ""})
        loop.analyze_failure(assignment1, {"type": "test_failure", "message": ""})

        # task2 should start fresh
        feedback = loop.analyze_failure(assignment2, {"type": "test_failure", "message": ""})

        # task2 should not be at max yet
        assert feedback["action"] != "escalate"


class TestAutonomousOrchestratorExtended:
    """Extended tests for AutonomousOrchestrator."""

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
                        title="Task One",
                        description="First task",
                        file_scope=["tests/test.py"],
                        estimated_complexity="low",
                    ),
                ],
            )
        )
        return decomposer

    def test_orchestrator_initialization(self):
        """Should initialize with defaults."""
        orchestrator = AutonomousOrchestrator()

        assert orchestrator.aragora_path.exists()
        assert orchestrator.track_configs == DEFAULT_TRACK_CONFIGS
        assert orchestrator.max_parallel_tasks == 4
        assert orchestrator.require_human_approval is True

    def test_orchestrator_custom_config(self, mock_workflow_engine, mock_task_decomposer):
        """Should accept custom configuration."""
        custom_configs = {
            Track.QA: TrackConfig(
                name="CustomQA",
                folders=["custom/tests/"],
                agent_types=["claude"],
            )
        }

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            track_configs=custom_configs,
            require_human_approval=True,
            max_parallel_tasks=2,
        )

        assert orchestrator.track_configs == custom_configs
        assert orchestrator.require_human_approval is True
        assert orchestrator.max_parallel_tasks == 2

    @pytest.mark.asyncio
    async def test_execute_goal_empty_decomposition(
        self, mock_workflow_engine, mock_task_decomposer
    ):
        """Should handle empty decomposition gracefully."""
        mock_task_decomposer.analyze.return_value = TaskDecomposition(
            original_task="Empty",
            complexity_score=0,
            complexity_level="trivial",
            should_decompose=False,
            subtasks=[],
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        result = await orchestrator.execute_goal(goal="Trivial task")

        assert result.success is True
        assert result.total_subtasks == 0
        assert "trivial" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_execute_goal_with_exception(self, mock_workflow_engine, mock_task_decomposer):
        """Should handle exceptions during execution."""
        mock_task_decomposer.analyze.side_effect = RuntimeError("Decomposition failed")

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        result = await orchestrator.execute_goal(goal="Will fail")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_get_active_assignments(self, mock_workflow_engine, mock_task_decomposer):
        """Should return copy of active assignments."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        # Add a fake active assignment
        subtask = SubTask(id="1", title="Active", description="Active task")
        orchestrator._active_assignments.append(
            AgentAssignment(subtask=subtask, track=Track.QA, agent_type="claude")
        )

        active = orchestrator.get_active_assignments()

        assert len(active) == 1
        # Should be a copy
        assert active is not orchestrator._active_assignments

    @pytest.mark.asyncio
    async def test_get_completed_assignments(self, mock_workflow_engine, mock_task_decomposer):
        """Should return copy of completed assignments."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )

        subtask = SubTask(id="1", title="Done", description="Completed task")
        orchestrator._completed_assignments.append(
            AgentAssignment(
                subtask=subtask,
                track=Track.QA,
                agent_type="claude",
                status="completed",
            )
        )

        completed = orchestrator.get_completed_assignments()

        assert len(completed) == 1
        assert completed[0].status == "completed"

    @pytest.mark.asyncio
    async def test_debate_decomposition_mode(self, mock_workflow_engine, mock_task_decomposer):
        """Should use debate decomposition when enabled."""
        mock_task_decomposer.analyze_with_debate = AsyncMock(
            return_value=TaskDecomposition(
                original_task="Debate decomposed",
                complexity_score=8,
                complexity_level="high",
                should_decompose=True,
                subtasks=[
                    SubTask(
                        id="debate_1",
                        title="Debated Task",
                        description="From debate",
                        estimated_complexity="high",
                    )
                ],
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            use_debate_decomposition=True,
        )

        await orchestrator.execute_goal(goal="Complex goal")

        # Should have used debate decomposition
        mock_task_decomposer.analyze_with_debate.assert_called_once()
        mock_task_decomposer.analyze.assert_not_called()


class TestOrchestratorSingleton:
    """Tests for singleton pattern."""

    def test_get_orchestrator_creates_singleton(self):
        """get_orchestrator should create singleton."""
        from aragora.nomic.autonomous_orchestrator import get_orchestrator

        orch1 = get_orchestrator()
        orch2 = get_orchestrator()

        assert orch1 is orch2

    def test_reset_orchestrator_clears_singleton(self):
        """reset_orchestrator should clear the singleton."""
        from aragora.nomic.autonomous_orchestrator import (
            get_orchestrator,
            reset_orchestrator,
        )

        orch1 = get_orchestrator()
        reset_orchestrator()
        orch2 = get_orchestrator()

        assert orch1 is not orch2


class TestBuildSubtaskWorkflow:
    """Tests for _build_subtask_workflow method."""

    def test_workflow_has_correct_steps(self):
        """Should create workflow with design, implement, verify steps."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(
            id="test",
            title="Test Task",
            description="Description",
            file_scope=["test.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        assert workflow.id == "subtask_test"
        assert len(workflow.steps) == 3

        step_ids = [s.id for s in workflow.steps]
        assert "design" in step_ids
        assert "implement" in step_ids
        assert "verify" in step_ids

    def test_workflow_uses_implementation_step(self):
        """Should use 'implementation' step type for implement phase (gold path)."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(id="test", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="codex",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        # Find implement step - now uses implementation step type
        impl_step = next(s for s in workflow.steps if s.id == "implement")
        assert impl_step.step_type == "implementation"
        assert impl_step.config["task_id"] == "test"
        assert impl_step.config["description"] == "Test task"

    def test_workflow_uses_verification_step(self):
        """Verify step should use 'verification' step type (runs pytest)."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(id="test", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.QA,
            agent_type="gemini",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        verify_step = next(s for s in workflow.steps if s.id == "verify")
        assert verify_step.step_type == "verification"
        assert verify_step.config["run_tests"] is True


class TestGenerateSummary:
    """Tests for _generate_summary method."""

    def test_summary_groups_by_track(self):
        """Should group assignments by track."""
        orchestrator = AutonomousOrchestrator()

        assignments = [
            AgentAssignment(
                subtask=SubTask(id="1", title="SME Task", description="SME"),
                track=Track.SME,
                agent_type="claude",
                status="completed",
            ),
            AgentAssignment(
                subtask=SubTask(id="2", title="QA Task", description="QA"),
                track=Track.QA,
                agent_type="claude",
                status="completed",
            ),
        ]

        summary = orchestrator._generate_summary(assignments)

        assert "SME" in summary
        assert "QA" in summary

    def test_summary_shows_status_icons(self):
        """Should show status indicators."""
        orchestrator = AutonomousOrchestrator()

        assignments = [
            AgentAssignment(
                subtask=SubTask(id="1", title="Completed", description="Done"),
                track=Track.QA,
                agent_type="claude",
                status="completed",
            ),
            AgentAssignment(
                subtask=SubTask(id="2", title="Failed", description="Error"),
                track=Track.QA,
                agent_type="claude",
                status="failed",
            ),
        ]

        summary = orchestrator._generate_summary(assignments)

        assert "+" in summary  # Completed indicator
        assert "-" in summary  # Failed indicator


class TestAntiFragileReassignment:
    """Tests for anti-fragile agent reassignment on failure."""

    def test_feedback_loop_reassign_on_first_workflow_failure(self):
        """First workflow_failure should trigger agent reassignment."""
        loop = FeedbackLoop(max_iterations=3)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="codex",
            attempt_count=0,
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "workflow_failure", "message": "Agent timed out"},
        )

        assert feedback["action"] == "reassign_agent"
        assert feedback["original_agent"] == "codex"

    def test_feedback_loop_no_reassign_on_second_failure(self):
        """Second failure of same type should not trigger reassignment."""
        loop = FeedbackLoop(max_iterations=5)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="codex",
            attempt_count=1,  # Already retried once
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "workflow_failure", "message": "Still failing"},
        )

        # Should escalate on unknown, not reassign (attempt_count != 0)
        assert feedback["action"] == "escalate"

    def test_feedback_loop_reassign_on_agent_timeout(self):
        """Agent timeout should trigger reassignment on first attempt."""
        loop = FeedbackLoop(max_iterations=3)

        subtask = SubTask(id="1", title="Test", description="Test task")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.SME,
            agent_type="gemini",
            attempt_count=0,
        )

        feedback = loop.analyze_failure(
            assignment,
            {"type": "agent_timeout", "message": "Request timed out"},
        )

        assert feedback["action"] == "reassign_agent"

    def test_select_alternative_agent_from_track(self):
        """Should select next agent from track's preferred list."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(id="1", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,  # Has ["claude", "codex"]
            agent_type="claude",
        )

        alt = orchestrator._select_alternative_agent(assignment)
        assert alt == "codex"

    def test_select_alternative_agent_falls_back_to_claude(self):
        """Should fall back to claude when no alternatives in track."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(id="1", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.CORE,  # Has only ["claude"]
            agent_type="gemini",  # Not in track list
        )

        alt = orchestrator._select_alternative_agent(assignment)
        assert alt == "claude"

    def test_select_alternative_agent_returns_none_when_only_claude(self):
        """Should return None when already using claude and no alternatives."""
        orchestrator = AutonomousOrchestrator()

        subtask = SubTask(id="1", title="Test", description="Test")
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.CORE,  # Has only ["claude"]
            agent_type="claude",
        )

        alt = orchestrator._select_alternative_agent(assignment)
        assert alt is None

    @pytest.mark.asyncio
    async def test_execute_with_reassignment_on_failure(self):
        """Full integration: failed task should be reassigned to different agent."""
        call_count = 0

        async def mock_execute(workflow, inputs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(success=False, final_output=None, error="Agent failed")
            return MagicMock(success=True, final_output={"status": "ok"}, error=None)

        engine = MagicMock()
        engine.execute = AsyncMock(side_effect=mock_execute)

        decomposer = MagicMock()
        decomposer.analyze = MagicMock(
            return_value=TaskDecomposition(
                original_task="Test",
                complexity_score=5,
                complexity_level="medium",
                should_decompose=True,
                subtasks=[
                    SubTask(
                        id="1",
                        title="SDK Task",
                        description="Add SDK method",
                        file_scope=["sdk/python/client.py"],
                        estimated_complexity="medium",
                    ),
                ],
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=engine,
            task_decomposer=decomposer,
            max_parallel_tasks=1,
        )

        result = await orchestrator.execute_goal(goal="Test", max_cycles=2)

        # Should have succeeded on retry with different agent
        assert result.completed_subtasks == 1
        assert call_count == 2


class TestMetricsIntegration:
    """Tests for MetricsCollector integration in the orchestrator."""

    @pytest.fixture
    def mock_workflow_engine(self):
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
                        title="Fix tests",
                        description="Fix failing tests",
                        file_scope=["tests/"],
                        estimated_complexity="medium",
                        success_criteria={"test_pass_rate": ">0.95"},
                    ),
                ],
            )
        )
        return decomposer

    @pytest.mark.asyncio
    async def test_metrics_disabled_by_default(self, mock_workflow_engine, mock_task_decomposer):
        """Orchestrator should not collect metrics when enable_metrics is False."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
        )
        assert orchestrator.enable_metrics is False
        assert orchestrator._metrics_collector is None

    @pytest.mark.asyncio
    async def test_metrics_enabled_creates_collector(self):
        """Enabling metrics should create a MetricsCollector."""
        orchestrator = AutonomousOrchestrator(enable_metrics=True)
        assert orchestrator.enable_metrics is True
        assert orchestrator._metrics_collector is not None

    @pytest.mark.asyncio
    async def test_metrics_populates_result_fields(
        self, mock_workflow_engine, mock_task_decomposer
    ):
        """When metrics enabled, result should have baseline/after/delta."""
        from aragora.nomic.metrics_collector import MetricSnapshot

        mock_collector = MagicMock()
        baseline = MetricSnapshot(tests_passed=100, tests_failed=5)
        after = MetricSnapshot(tests_passed=103, tests_failed=2)
        mock_collector.collect_baseline = AsyncMock(return_value=baseline)
        mock_collector.collect_after = AsyncMock(return_value=after)
        mock_collector.compare = MagicMock(
            return_value=MagicMock(
                to_dict=MagicMock(return_value={"improved": True, "improvement_score": 0.7}),
                improvement_score=0.7,
                improved=True,
                summary="+3 tests passing",
            )
        )
        mock_collector.check_success_criteria = MagicMock(return_value=(True, []))

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            enable_metrics=True,
            metrics_collector=mock_collector,
        )

        result = await orchestrator.execute_goal(goal="Fix tests", max_cycles=2)

        assert result.baseline_metrics is not None
        assert result.after_metrics is not None
        assert result.metrics_delta is not None
        assert result.improvement_score == 0.7
        assert result.success_criteria_met is True
        mock_collector.collect_baseline.assert_called_once()
        mock_collector.collect_after.assert_called_once()
        mock_collector.compare.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_failure_does_not_break_orchestration(
        self, mock_workflow_engine, mock_task_decomposer
    ):
        """Metrics collection failure should not prevent orchestration."""
        mock_collector = MagicMock()
        mock_collector.collect_baseline = AsyncMock(side_effect=RuntimeError("metrics down"))

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            enable_metrics=True,
            metrics_collector=mock_collector,
        )

        result = await orchestrator.execute_goal(goal="Fix tests", max_cycles=2)

        # Should succeed despite metrics failure
        assert result.completed_subtasks == 1
        assert result.baseline_metrics is None
        assert result.metrics_delta is None


class TestPreflightIntegration:
    """Tests for preflight health check integration."""

    def test_preflight_disabled_by_default(self):
        """Orchestrator should not run preflight by default."""
        orchestrator = AutonomousOrchestrator()
        assert orchestrator.enable_preflight is False

    def test_preflight_enabled_stores_flag(self):
        """Enabling preflight should set the flag."""
        orchestrator = AutonomousOrchestrator(enable_preflight=True)
        assert orchestrator.enable_preflight is True

    @pytest.mark.asyncio
    async def test_preflight_failure_returns_error_result(self):
        """Failed preflight should return error OrchestrationResult."""
        from aragora.nomic.preflight import PreflightResult

        mock_result = PreflightResult(
            passed=False,
            blocking_issues=["No API keys configured"],
        )

        orchestrator = AutonomousOrchestrator(enable_preflight=True)

        with patch(
            "aragora.nomic.preflight.PreflightHealthCheck.run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await orchestrator.execute_goal(goal="Test", max_cycles=1)

        assert result.success is False
        assert "Preflight check failed" in result.error
        assert result.total_subtasks == 0

    @pytest.mark.asyncio
    async def test_preflight_passes_allows_execution(
        self,
    ):
        """Passed preflight should allow normal execution."""
        from aragora.nomic.preflight import PreflightResult

        mock_preflight = PreflightResult(
            passed=True,
            recommended_agents=["claude-visionary"],
        )

        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value=MagicMock(success=True, final_output={}, error=None)
        )
        decomposer = MagicMock()
        decomposer.analyze = MagicMock(
            return_value=TaskDecomposition(
                original_task="Test",
                complexity_score=5,
                complexity_level="medium",
                should_decompose=True,
                subtasks=[
                    SubTask(id="1", title="Task", description="Do thing", file_scope=["test.py"]),
                ],
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=engine,
            task_decomposer=decomposer,
            enable_preflight=True,
        )

        with patch(
            "aragora.nomic.preflight.PreflightHealthCheck.run",
            new_callable=AsyncMock,
            return_value=mock_preflight,
        ):
            result = await orchestrator.execute_goal(goal="Test", max_cycles=1)

        assert result.success is True
        assert result.completed_subtasks == 1


class TestStuckDetectorIntegration:
    """Tests for stuck detector integration in the orchestrator."""

    def test_stuck_detection_disabled_by_default(self):
        """Stuck detection should be off by default."""
        orchestrator = AutonomousOrchestrator()
        assert orchestrator.enable_stuck_detection is False
        assert orchestrator._stuck_detector is None

    def test_stuck_detection_enabled_creates_detector(self):
        """Enabling stuck detection should create a StuckDetector."""
        orchestrator = AutonomousOrchestrator(enable_stuck_detection=True)
        assert orchestrator.enable_stuck_detection is True
        assert orchestrator._stuck_detector is not None

    def test_stuck_detection_custom_detector(self):
        """Custom detector should be used when provided."""
        mock_detector = MagicMock()
        orchestrator = AutonomousOrchestrator(
            enable_stuck_detection=True,
            stuck_detector=mock_detector,
        )
        assert orchestrator._stuck_detector is mock_detector

    @pytest.mark.asyncio
    async def test_stuck_detection_starts_and_stops(self):
        """Stuck detector should start monitoring during execution and stop after."""
        mock_detector = MagicMock()
        mock_detector.initialize = AsyncMock()
        mock_detector.start_monitoring = AsyncMock()
        mock_detector.stop_monitoring = AsyncMock()
        mock_detector.get_health_summary = AsyncMock(
            return_value=MagicMock(
                red_count=0,
                yellow_count=0,
                total_items=3,
                health_percentage=100.0,
                recovered_count=0,
            )
        )

        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value=MagicMock(success=True, final_output={}, error=None)
        )
        decomposer = MagicMock()
        decomposer.analyze = MagicMock(
            return_value=TaskDecomposition(
                original_task="Test",
                complexity_score=3,
                complexity_level="low",
                should_decompose=True,
                subtasks=[
                    SubTask(id="1", title="Task", description="Do thing", file_scope=["x.py"]),
                ],
            )
        )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=engine,
            task_decomposer=decomposer,
            enable_stuck_detection=True,
            stuck_detector=mock_detector,
        )

        result = await orchestrator.execute_goal(goal="Test", max_cycles=1)

        assert result.success is True
        mock_detector.initialize.assert_awaited_once()
        mock_detector.start_monitoring.assert_awaited_once()
        mock_detector.stop_monitoring.assert_awaited_once()
        mock_detector.get_health_summary.assert_awaited_once()


class TestCIFeedbackIntegration:
    """Tests for CI feedback integration in the merge flow."""

    @pytest.mark.asyncio
    async def test_ci_check_before_merge(self):
        """CI result should be checked before merging a branch."""
        from aragora.nomic.ci_feedback import CIResult

        mock_ci_result = CIResult(
            workflow_run_id=123,
            branch="dev/sme-test",
            commit_sha="abc123",
            conclusion="success",
        )

        mock_coordinator = MagicMock()
        mock_coordinator._worktree_paths = {"dev/sme-test": "/tmp/sme"}
        mock_coordinator.config = MagicMock(base_branch="main")
        mock_coordinator.safe_merge = AsyncMock(
            return_value=MagicMock(success=True, commit_sha="merged123")
        )
        mock_coordinator.cleanup_worktrees = MagicMock(return_value=0)

        orchestrator = AutonomousOrchestrator(
            branch_coordinator=mock_coordinator,
        )

        assignments = [
            AgentAssignment(
                subtask=SubTask(id="1", title="Test", description="Do thing", file_scope=[]),
                track=Track.SME,
                agent_type="claude",
                status="completed",
            ),
        ]

        with patch(
            "aragora.nomic.ci_feedback.CIResultCollector.get_latest_result",
            return_value=mock_ci_result,
        ):
            await orchestrator._merge_and_cleanup(assignments)

        mock_coordinator.safe_merge.assert_awaited_once()


class TestGatesIntegration:
    """Tests for approval gate wiring in the nomic loop handlers."""

    def test_gates_module_exports(self):
        """Gates module should export all expected classes."""
        from aragora.nomic.gates import (
            DesignGate,
            TestQualityGate,
            CommitGate,
            create_standard_gates,
            GateType,
        )

        gates = create_standard_gates(dev_mode=True)
        assert GateType.DESIGN in gates
        assert GateType.TEST_QUALITY in gates
        assert GateType.COMMIT in gates
        assert isinstance(gates[GateType.DESIGN], DesignGate)
        assert isinstance(gates[GateType.TEST_QUALITY], TestQualityGate)
        assert isinstance(gates[GateType.COMMIT], CommitGate)

    @pytest.mark.asyncio
    async def test_design_gate_auto_approve_dev_mode(self):
        """DesignGate should auto-approve in dev mode."""
        from aragora.nomic.gates import DesignGate, ApprovalStatus
        import os

        gate = DesignGate(enabled=True, auto_approve_dev=True)

        with patch.dict(os.environ, {"ARAGORA_DEV_MODE": "1"}):
            decision = await gate.require_approval(
                "Design: refactor auth.py",
                context={"complexity_score": 0.5, "files_affected": ["auth.py"]},
            )

        assert decision.status == ApprovalStatus.APPROVED
        assert decision.approver == "auto_dev"

    @pytest.mark.asyncio
    async def test_design_gate_rejects_high_complexity(self):
        """DesignGate should reject designs with complexity above threshold."""
        from aragora.nomic.gates import DesignGate, ApprovalRequired

        gate = DesignGate(enabled=True, max_complexity_score=0.5)

        with pytest.raises(ApprovalRequired):
            await gate.require_approval(
                "Complex design",
                context={"complexity_score": 0.9, "files_affected": ["a.py"]},
            )

    @pytest.mark.asyncio
    async def test_test_quality_gate_passes_on_all_green(self):
        """TestQualityGate should pass when all tests pass."""
        from aragora.nomic.gates import TestQualityGate, ApprovalStatus

        gate = TestQualityGate(enabled=True, require_all_tests_pass=True)

        decision = await gate.require_approval(
            "All 42 tests passed",
            context={"tests_passed": True, "coverage": 85.0, "warnings_count": 0},
        )

        assert decision.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_test_quality_gate_rejects_failures(self):
        """TestQualityGate should reject when tests fail."""
        from aragora.nomic.gates import TestQualityGate, ApprovalRequired

        gate = TestQualityGate(enabled=True, require_all_tests_pass=True)

        with pytest.raises(ApprovalRequired):
            await gate.require_approval(
                "3 tests failed",
                context={"tests_passed": False, "coverage": 60.0, "warnings_count": 0},
            )

    @pytest.mark.asyncio
    async def test_gate_skipped_when_disabled(self):
        """Gates should skip when disabled."""
        from aragora.nomic.gates import DesignGate, ApprovalStatus

        gate = DesignGate(enabled=False)

        decision = await gate.require_approval(
            "Design",
            context={},
        )

        assert decision.status == ApprovalStatus.SKIPPED
