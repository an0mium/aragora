"""Tests for the Hierarchical Coordinator (Planner/Worker/Judge cycle)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.hierarchical_coordinator import (
    CoordinationPhase,
    CoordinatorConfig,
    HierarchicalCoordinator,
    HierarchicalResult,
    JudgeVerdict,
    WorkerReport,
)
from aragora.nomic.task_decomposer import (
    SubTask,
    TaskDecomposer,
    TaskDecomposition,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def _no_api_keys():
    """Prevent real API calls by making get_secret return None for API keys.

    Without this, the judge phase tries to create real AnthropicAPIAgent
    instances when API keys are cached in SecretManager from other tests.
    """
    with patch("aragora.config.secrets.get_secret", return_value=None):
        yield


def _make_subtask(id_: str = "subtask_1", title: str = "Test Task") -> SubTask:
    return SubTask(
        id=id_,
        title=title,
        description=f"Description for {title}",
        estimated_complexity="medium",
        file_scope=["aragora/test.py"],
    )


def _make_decomposition(
    subtasks: list[SubTask] | None = None,
    complexity_score: int = 6,
) -> TaskDecomposition:
    subtasks = subtasks or [_make_subtask()]
    return TaskDecomposition(
        original_task="test goal",
        complexity_score=complexity_score,
        complexity_level="medium",
        should_decompose=True,
        subtasks=subtasks,
    )


def _make_workflow_result(success: bool = True, error: str | None = None):
    """Create a mock WorkflowResult."""
    result = MagicMock()
    result.success = success
    result.final_output = {"done": True} if success else None
    result.error = error
    return result


@pytest.fixture
def mock_decomposer():
    decomposer = MagicMock(spec=TaskDecomposer)
    decomposer.analyze.return_value = _make_decomposition()
    decomposer.analyze_with_debate = AsyncMock(return_value=_make_decomposition())
    return decomposer


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.execute = AsyncMock(return_value=_make_workflow_result(success=True))
    return engine


@pytest.fixture
def coordinator(mock_decomposer, mock_engine):
    return HierarchicalCoordinator(
        config=CoordinatorConfig(),
        task_decomposer=mock_decomposer,
        workflow_engine=mock_engine,
    )


# =========================================================================
# TestCoordinatorConfig
# =========================================================================


class TestCoordinatorConfig:
    def test_defaults(self):
        config = CoordinatorConfig()
        assert config.max_plan_revisions == 2
        assert config.max_parallel_workers == 4
        assert config.worker_timeout_seconds == 300
        assert config.judge_agent == "claude"
        assert config.judge_rounds == 2
        assert config.quality_threshold == 0.6
        assert config.max_cycles == 3
        assert config.enable_checkpointing is False
        assert config.use_debate_decomposition is False

    def test_custom_values(self):
        config = CoordinatorConfig(
            max_cycles=5,
            quality_threshold=0.8,
            max_parallel_workers=8,
            judge_agent="gpt",
        )
        assert config.max_cycles == 5
        assert config.quality_threshold == 0.8
        assert config.max_parallel_workers == 8
        assert config.judge_agent == "gpt"

    def test_debate_decomposition_flag(self):
        config = CoordinatorConfig(use_debate_decomposition=True)
        assert config.use_debate_decomposition is True

    def test_checkpointing_flag(self):
        config = CoordinatorConfig(enable_checkpointing=True)
        assert config.enable_checkpointing is True

    def test_plan_revisions_limit(self):
        config = CoordinatorConfig(max_plan_revisions=5)
        assert config.max_plan_revisions == 5


# =========================================================================
# TestCoordinationPhase
# =========================================================================


class TestCoordinationPhase:
    def test_all_phases_exist(self):
        phases = [p.value for p in CoordinationPhase]
        assert "planning" in phases
        assert "dispatching" in phases
        assert "executing" in phases
        assert "judging" in phases
        assert "revising" in phases
        assert "completed" in phases
        assert "failed" in phases

    def test_phase_count(self):
        assert len(CoordinationPhase) == 7

    def test_enum_values(self):
        assert CoordinationPhase.PLANNING.value == "planning"
        assert CoordinationPhase.FAILED.value == "failed"


# =========================================================================
# TestWorkerReport
# =========================================================================


class TestWorkerReport:
    def test_creation_success(self):
        report = WorkerReport(
            assignment_id="assign_1",
            subtask_title="Task A",
            success=True,
            output={"result": "done"},
            duration_seconds=1.5,
        )
        assert report.success is True
        assert report.error is None
        assert report.output == {"result": "done"}

    def test_creation_failure(self):
        report = WorkerReport(
            assignment_id="assign_2",
            subtask_title="Task B",
            success=False,
            error="timeout",
        )
        assert report.success is False
        assert report.error == "timeout"
        assert report.output is None

    def test_default_values(self):
        report = WorkerReport(
            assignment_id="a",
            subtask_title="t",
            success=True,
        )
        assert report.duration_seconds == 0.0
        assert report.output is None
        assert report.error is None


# =========================================================================
# TestJudgeVerdict
# =========================================================================


class TestJudgeVerdict:
    def test_approved_verdict(self):
        verdict = JudgeVerdict(approved=True, confidence=0.9, feedback="Good work")
        assert verdict.approved is True
        assert verdict.confidence == 0.9

    def test_rejected_verdict(self):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.3,
            feedback="Needs work",
            revision_instructions=["Fix tests", "Add docs"],
        )
        assert verdict.approved is False
        assert len(verdict.revision_instructions) == 2

    def test_partial_approvals(self):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.5,
            partial_approvals={"task_1": True, "task_2": False, "task_3": True},
        )
        assert verdict.partial_approvals["task_1"] is True
        assert verdict.partial_approvals["task_2"] is False

    def test_default_values(self):
        verdict = JudgeVerdict(approved=True, confidence=1.0)
        assert verdict.feedback == ""
        assert verdict.revision_instructions == []
        assert verdict.partial_approvals == {}


# =========================================================================
# TestHierarchicalResult
# =========================================================================


class TestHierarchicalResult:
    def test_success_result(self):
        result = HierarchicalResult(
            goal="test",
            phase=CoordinationPhase.COMPLETED,
            cycles_used=1,
            worker_reports=[],
            verdict=JudgeVerdict(approved=True, confidence=1.0),
            success=True,
            duration_seconds=2.5,
        )
        assert result.success is True
        assert result.phase == CoordinationPhase.COMPLETED

    def test_failed_result(self):
        result = HierarchicalResult(
            goal="test",
            phase=CoordinationPhase.FAILED,
            cycles_used=3,
            worker_reports=[],
            verdict=None,
            success=False,
        )
        assert result.success is False
        assert result.phase == CoordinationPhase.FAILED

    def test_default_cost(self):
        result = HierarchicalResult(
            goal="test",
            phase=CoordinationPhase.COMPLETED,
            cycles_used=1,
            worker_reports=[],
            verdict=None,
            success=True,
        )
        assert result.total_cost == 0.0
        assert result.duration_seconds == 0.0


# =========================================================================
# TestPlanPhase
# =========================================================================


class TestPlanPhase:
    @pytest.mark.asyncio
    async def test_initial_plan_uses_decomposer(self, coordinator, mock_decomposer):
        decomp = _make_decomposition()
        mock_decomposer.analyze.return_value = decomp

        result = await coordinator._plan("test goal", None, None)
        mock_decomposer.analyze.assert_called_once()
        assert result.subtasks == decomp.subtasks

    @pytest.mark.asyncio
    async def test_plan_with_tracks(self, coordinator, mock_decomposer):
        await coordinator._plan("test goal", ["sme", "qa"], None)
        call_args = mock_decomposer.analyze.call_args[0][0]
        assert "sme" in call_args
        assert "qa" in call_args

    @pytest.mark.asyncio
    async def test_debate_decomposition(self, mock_decomposer, mock_engine):
        config = CoordinatorConfig(use_debate_decomposition=True)
        coord = HierarchicalCoordinator(
            config=config,
            task_decomposer=mock_decomposer,
            workflow_engine=mock_engine,
        )
        await coord._plan("abstract goal", None, None)
        mock_decomposer.analyze_with_debate.assert_called_once()

    @pytest.mark.asyncio
    async def test_revision_replanning(self, coordinator, mock_decomposer):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.3,
            feedback="Task 2 failed",
            revision_instructions=["Fix task 2"],
            partial_approvals={"subtask_1": True, "subtask_2": False},
        )
        result = await coordinator._plan("test goal", None, verdict)
        # On revision, analyze is called with focused goal
        mock_decomposer.analyze.assert_called()
        call_goal = mock_decomposer.analyze.call_args[0][0]
        assert "Revise" in call_goal
        assert "Fix task 2" in call_goal

    @pytest.mark.asyncio
    async def test_revision_tags_subtasks(self, coordinator, mock_decomposer):
        mock_decomposer.analyze.return_value = _make_decomposition(
            subtasks=[_make_subtask("s1"), _make_subtask("s2")]
        )
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.3,
            feedback="failed",
            partial_approvals={"subtask_1": True, "subtask_2": False},
        )
        result = await coordinator._plan("test", None, verdict)
        # Only subtask_2 was rejected, so one revision ID should be tagged
        assert result.subtasks[0].id == "revision_subtask_2"

    @pytest.mark.asyncio
    async def test_plan_empty_decomposition(self, mock_engine):
        decomposer = MagicMock()
        decomposer.analyze.side_effect = lambda *a, **kw: TaskDecomposition(
            original_task="test",
            complexity_score=1,
            complexity_level="low",
            should_decompose=False,
            subtasks=[],
        )
        coord = HierarchicalCoordinator(
            task_decomposer=decomposer,
            workflow_engine=mock_engine,
        )
        result = await coord._plan("trivial", None, None)
        assert result.subtasks == []

    @pytest.mark.asyncio
    async def test_plan_preserves_original_task(self, coordinator, mock_decomposer):
        result = await coordinator._plan("my goal", None, None)
        assert result.original_task == "test goal"  # from mock

    @pytest.mark.asyncio
    async def test_revision_no_rejected_ids(self, coordinator, mock_decomposer):
        """Revision with all approved (edge case) still calls analyze."""
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.4,
            feedback="General quality low",
            partial_approvals={},
        )
        result = await coordinator._plan("test", None, verdict)
        mock_decomposer.analyze.assert_called()


# =========================================================================
# TestDispatchAndExecute
# =========================================================================


class TestDispatchAndExecute:
    @pytest.mark.asyncio
    async def test_single_worker(self, coordinator, mock_engine):
        decomp = _make_decomposition(subtasks=[_make_subtask()])
        reports = await coordinator._dispatch_and_execute(decomp)
        assert len(reports) == 1
        assert reports[0].success is True

    @pytest.mark.asyncio
    async def test_parallel_workers(self, coordinator, mock_engine):
        subtasks = [_make_subtask(f"s{i}", f"Task {i}") for i in range(4)]
        decomp = _make_decomposition(subtasks=subtasks)
        reports = await coordinator._dispatch_and_execute(decomp)
        assert len(reports) == 4
        assert all(r.success for r in reports)
        assert mock_engine.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_partial_failure(self, coordinator, mock_engine):
        # First call succeeds, second fails
        mock_engine.execute = AsyncMock(
            side_effect=[
                _make_workflow_result(success=True),
                _make_workflow_result(success=False, error="compile error"),
            ]
        )
        subtasks = [_make_subtask("s1", "Pass"), _make_subtask("s2", "Fail")]
        decomp = _make_decomposition(subtasks=subtasks)
        reports = await coordinator._dispatch_and_execute(decomp)
        assert len(reports) == 2
        successes = [r for r in reports if r.success]
        failures = [r for r in reports if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1

    @pytest.mark.asyncio
    async def test_worker_exception(self, coordinator, mock_engine):
        mock_engine.execute = AsyncMock(side_effect=RuntimeError("engine crash"))
        decomp = _make_decomposition()
        reports = await coordinator._dispatch_and_execute(decomp)
        assert len(reports) == 1
        assert reports[0].success is False
        assert "Worker execution failed" in reports[0].error

    @pytest.mark.asyncio
    async def test_empty_decomposition(self, coordinator, mock_engine):
        decomp = TaskDecomposition(
            original_task="empty",
            complexity_score=1,
            complexity_level="low",
            should_decompose=False,
            subtasks=[],
        )
        reports = await coordinator._dispatch_and_execute(decomp)
        assert reports == []
        mock_engine.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_semaphore_limits_parallelism(self, mock_decomposer):
        """Verify the semaphore limits concurrent workers."""
        engine = MagicMock()
        concurrent_count = 0
        max_concurrent = 0

        async def tracked_execute(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return _make_workflow_result(success=True)

        engine.execute = tracked_execute

        config = CoordinatorConfig(max_parallel_workers=2)
        coord = HierarchicalCoordinator(
            config=config,
            task_decomposer=mock_decomposer,
            workflow_engine=engine,
        )

        subtasks = [_make_subtask(f"s{i}", f"Task {i}") for i in range(6)]
        decomp = _make_decomposition(subtasks=subtasks)
        reports = await coord._dispatch_and_execute(decomp)
        assert len(reports) == 6
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_report_has_duration(self, coordinator, mock_engine):
        decomp = _make_decomposition()
        reports = await coordinator._dispatch_and_execute(decomp)
        assert reports[0].duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_report_assignment_id(self, coordinator, mock_engine):
        decomp = _make_decomposition(subtasks=[_make_subtask("my_task")])
        reports = await coordinator._dispatch_and_execute(decomp)
        assert reports[0].assignment_id == "assign_my_task"


# =========================================================================
# TestJudgePhase
# =========================================================================


class TestJudgePhase:
    @pytest.mark.asyncio
    async def test_heuristic_approve(self, coordinator):
        reports = [
            WorkerReport("a1", "T1", success=True, duration_seconds=1.0),
            WorkerReport("a2", "T2", success=True, duration_seconds=1.0),
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert verdict.approved is True
        assert verdict.confidence == 1.0

    @pytest.mark.asyncio
    async def test_heuristic_reject(self, coordinator):
        reports = [
            WorkerReport("a1", "T1", success=False, error="fail", duration_seconds=1.0),
            WorkerReport("a2", "T2", success=False, error="fail", duration_seconds=1.0),
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert verdict.approved is False
        assert verdict.confidence == 0.0

    @pytest.mark.asyncio
    async def test_heuristic_threshold_boundary(self, coordinator):
        # 3 of 5 succeed = 60%, threshold is 0.6
        reports = [
            WorkerReport(f"a{i}", f"T{i}", success=(i < 3), duration_seconds=1.0) for i in range(5)
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert verdict.approved is True  # 60% == threshold

    @pytest.mark.asyncio
    async def test_heuristic_partial_approvals(self, coordinator):
        reports = [
            WorkerReport("assign_s1", "T1", success=True, duration_seconds=1.0),
            WorkerReport("assign_s2", "T2", success=False, error="err", duration_seconds=1.0),
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert verdict.partial_approvals["s1"] is True
        assert verdict.partial_approvals["s2"] is False

    @pytest.mark.asyncio
    async def test_heuristic_revision_instructions(self, coordinator):
        reports = [
            WorkerReport(
                "assign_s1", "Task A", success=False, error="compile error", duration_seconds=1.0
            ),
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert len(verdict.revision_instructions) == 1
        assert "Task A" in verdict.revision_instructions[0]

    @pytest.mark.asyncio
    async def test_heuristic_empty_reports(self, coordinator):
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, [])
        assert verdict.approved is False
        assert verdict.confidence == 0.0

    @pytest.mark.asyncio
    async def test_arena_import_failure_uses_heuristic(self, coordinator):
        """When Arena import fails, falls back to heuristic."""
        reports = [WorkerReport("a1", "T1", success=True, duration_seconds=1.0)]
        decomp = _make_decomposition()
        # The coordinator uses heuristic fallback by default since Arena imports
        # will fail in test environment (no API keys)
        verdict = await coordinator._judge("goal", decomp, reports)
        assert verdict.approved is True

    @pytest.mark.asyncio
    async def test_verdict_feedback_contains_rate(self, coordinator):
        reports = [
            WorkerReport("a1", "T1", success=True, duration_seconds=1.0),
            WorkerReport("a2", "T2", success=False, error="x", duration_seconds=1.0),
        ]
        decomp = _make_decomposition()
        verdict = await coordinator._judge("goal", decomp, reports)
        assert "50%" in verdict.feedback


# =========================================================================
# TestFullCycle
# =========================================================================


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_happy_path(self, coordinator, mock_decomposer, mock_engine):
        """Plan -> execute -> approve in one cycle."""
        result = await coordinator.coordinate("improve tests")
        assert result.success is True
        assert result.cycles_used == 1
        assert result.phase == CoordinationPhase.COMPLETED
        assert len(result.worker_reports) == 1

    @pytest.mark.asyncio
    async def test_rejection_then_approval(self, mock_decomposer, mock_engine):
        """First cycle rejected, second cycle approved."""
        call_count = 0

        async def engine_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_workflow_result(success=False, error="fail")
            return _make_workflow_result(success=True)

        mock_engine.execute = engine_execute

        config = CoordinatorConfig(quality_threshold=1.0, max_cycles=3)
        coord = HierarchicalCoordinator(
            config=config,
            task_decomposer=mock_decomposer,
            workflow_engine=mock_engine,
        )
        result = await coord.coordinate("fix bugs")
        assert result.cycles_used == 2
        assert result.success is True
        assert result.phase == CoordinationPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_max_cycles_reached(self, mock_decomposer):
        """All cycles fail, returns FAILED."""
        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value=_make_workflow_result(success=False, error="always fails")
        )
        config = CoordinatorConfig(max_cycles=2, quality_threshold=1.0)
        coord = HierarchicalCoordinator(
            config=config,
            task_decomposer=mock_decomposer,
            workflow_engine=engine,
        )
        result = await coord.coordinate("impossible goal")
        assert result.success is False
        assert result.cycles_used == 2
        assert result.phase == CoordinationPhase.FAILED

    @pytest.mark.asyncio
    async def test_empty_decomposition_succeeds(self, mock_engine):
        """Goal that decomposes to zero subtasks is trivially successful."""
        decomposer = MagicMock()
        decomposer.analyze.side_effect = lambda *a, **kw: TaskDecomposition(
            original_task="test",
            complexity_score=1,
            complexity_level="low",
            should_decompose=False,
            subtasks=[],
        )
        coord = HierarchicalCoordinator(
            task_decomposer=decomposer,
            workflow_engine=mock_engine,
        )
        result = await coord.coordinate("trivial goal")
        assert result.success is True
        assert result.cycles_used == 1
        assert len(result.worker_reports) == 0

    @pytest.mark.asyncio
    async def test_context_passed_through(self, coordinator, mock_decomposer, mock_engine):
        """Context dict is passed to coordinate."""
        result = await coordinator.coordinate("test", context={"key": "value"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_tracks_passed_to_plan(self, coordinator, mock_decomposer, mock_engine):
        result = await coordinator.coordinate("test", tracks=["sme"])
        call_args = mock_decomposer.analyze.call_args[0][0]
        assert "sme" in call_args


# =========================================================================
# TestOrchestratorIntegration
# =========================================================================


class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_delegation_to_hierarchical_coordinator(self):
        """AutonomousOrchestrator delegates to HierarchicalCoordinator."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        mock_hc = MagicMock()
        mock_hc.coordinate = AsyncMock(
            return_value=HierarchicalResult(
                goal="test",
                phase=CoordinationPhase.COMPLETED,
                cycles_used=1,
                worker_reports=[
                    WorkerReport("a1", "Task1", success=True, duration_seconds=1.0),
                ],
                verdict=JudgeVerdict(approved=True, confidence=0.9),
                success=True,
                duration_seconds=2.0,
            )
        )

        orch = AutonomousOrchestrator(
            hierarchical_coordinator=mock_hc,
            enable_curriculum=False,
        )
        result = await orch.execute_goal("test goal")

        mock_hc.coordinate.assert_called_once()
        assert result.success is True
        assert result.completed_subtasks == 1

    @pytest.mark.asyncio
    async def test_orchestrator_without_coordinator_uses_standard_path(self):
        """Without hierarchical_coordinator, standard path is used."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(enable_curriculum=False)
        assert orch.hierarchical_coordinator is None

    @pytest.mark.asyncio
    async def test_result_conversion(self):
        """HierarchicalResult converts to OrchestrationResult correctly."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        mock_hc = MagicMock()
        mock_hc.coordinate = AsyncMock(
            return_value=HierarchicalResult(
                goal="test",
                phase=CoordinationPhase.COMPLETED,
                cycles_used=2,
                worker_reports=[
                    WorkerReport("a1", "T1", success=True, duration_seconds=1.0),
                    WorkerReport("a2", "T2", success=False, error="err", duration_seconds=0.5),
                ],
                verdict=JudgeVerdict(approved=True, confidence=0.8),
                success=True,
                duration_seconds=3.0,
            )
        )

        orch = AutonomousOrchestrator(
            hierarchical_coordinator=mock_hc,
            enable_curriculum=False,
        )
        result = await orch.execute_goal("test")

        assert result.total_subtasks == 2
        assert result.completed_subtasks == 1
        assert result.failed_subtasks == 1
        assert "2 cycles" in result.summary

    @pytest.mark.asyncio
    async def test_delegation_passes_tracks(self):
        """Tracks are forwarded to hierarchical coordinator."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        mock_hc = MagicMock()
        mock_hc.coordinate = AsyncMock(
            return_value=HierarchicalResult(
                goal="test",
                phase=CoordinationPhase.COMPLETED,
                cycles_used=1,
                worker_reports=[],
                verdict=None,
                success=True,
            )
        )

        orch = AutonomousOrchestrator(
            hierarchical_coordinator=mock_hc,
            enable_curriculum=False,
        )
        await orch.execute_goal("test", tracks=["sme", "qa"])

        call_kwargs = mock_hc.coordinate.call_args
        assert call_kwargs.kwargs["tracks"] == ["sme", "qa"]

    @pytest.mark.asyncio
    async def test_delegation_passes_context(self):
        """Context is forwarded to hierarchical coordinator."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        mock_hc = MagicMock()
        mock_hc.coordinate = AsyncMock(
            return_value=HierarchicalResult(
                goal="test",
                phase=CoordinationPhase.COMPLETED,
                cycles_used=1,
                worker_reports=[],
                verdict=None,
                success=True,
            )
        )

        orch = AutonomousOrchestrator(
            hierarchical_coordinator=mock_hc,
            enable_curriculum=False,
        )
        await orch.execute_goal("test", context={"foo": "bar"})

        call_kwargs = mock_hc.coordinate.call_args
        assert call_kwargs.kwargs["context"] == {"foo": "bar"}


# =========================================================================
# TestBuildJudgePrompt
# =========================================================================


class TestBuildJudgePrompt:
    def test_prompt_contains_goal(self, coordinator):
        decomp = _make_decomposition()
        reports = [WorkerReport("a1", "T1", success=True, duration_seconds=1.0)]
        prompt = coordinator._build_judge_prompt("my goal", decomp, reports)
        assert "my goal" in prompt

    def test_prompt_includes_reports(self, coordinator):
        decomp = _make_decomposition()
        reports = [
            WorkerReport("a1", "Task Alpha", success=True, duration_seconds=2.0),
        ]
        prompt = coordinator._build_judge_prompt("goal", decomp, reports)
        assert "Task Alpha" in prompt
        assert "SUCCESS" in prompt

    def test_prompt_includes_failures(self, coordinator):
        decomp = _make_decomposition()
        reports = [
            WorkerReport("a1", "Bad Task", success=False, error="crash", duration_seconds=0.5),
        ]
        prompt = coordinator._build_judge_prompt("goal", decomp, reports)
        assert "FAILED" in prompt
        assert "crash" in prompt

    def test_prompt_includes_subtask_details(self, coordinator):
        subtasks = [_make_subtask("s1", "Important Work")]
        decomp = _make_decomposition(subtasks=subtasks)
        reports = []
        prompt = coordinator._build_judge_prompt("goal", decomp, reports)
        assert "Important Work" in prompt


# =========================================================================
# TestHeuristicJudge
# =========================================================================


class TestHeuristicJudge:
    def test_all_success(self, coordinator):
        reports = [
            WorkerReport("a1", "T1", success=True, duration_seconds=1.0),
            WorkerReport("a2", "T2", success=True, duration_seconds=1.0),
        ]
        verdict = coordinator._heuristic_judge("goal", reports)
        assert verdict.approved is True
        assert verdict.confidence == 1.0

    def test_all_failure(self, coordinator):
        reports = [
            WorkerReport("a1", "T1", success=False, error="e", duration_seconds=1.0),
            WorkerReport("a2", "T2", success=False, error="e", duration_seconds=1.0),
        ]
        verdict = coordinator._heuristic_judge("goal", reports)
        assert verdict.approved is False
        assert verdict.confidence == 0.0

    def test_threshold_boundary_exact(self, coordinator):
        # 3/5 = 0.6 = threshold
        reports = [
            WorkerReport(f"a{i}", f"T{i}", success=(i < 3), duration_seconds=1.0) for i in range(5)
        ]
        verdict = coordinator._heuristic_judge("goal", reports)
        assert verdict.approved is True

    def test_below_threshold(self, coordinator):
        # 1/5 = 0.2 < 0.6
        reports = [
            WorkerReport(f"a{i}", f"T{i}", success=(i == 0), duration_seconds=1.0) for i in range(5)
        ]
        verdict = coordinator._heuristic_judge("goal", reports)
        assert verdict.approved is False


# =========================================================================
# TestRevision
# =========================================================================


class TestRevision:
    def test_only_replans_rejected(self, coordinator, mock_decomposer):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.4,
            feedback="s2 failed",
            partial_approvals={"subtask_1": True, "subtask_2": False},
        )
        result = coordinator._replan_rejected("goal", verdict)
        call_goal = mock_decomposer.analyze.call_args[0][0]
        assert "Revise" in call_goal

    def test_preserves_approved_in_partial(self, coordinator, mock_decomposer):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.5,
            feedback="mixed",
            partial_approvals={"s1": True, "s2": False, "s3": True},
        )
        coordinator._replan_rejected("goal", verdict)
        call_goal = mock_decomposer.analyze.call_args[0][0]
        # The focused goal should mention revision
        assert "Revise" in call_goal

    def test_revision_includes_instructions(self, coordinator, mock_decomposer):
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.3,
            feedback="problems",
            revision_instructions=["Fix auth", "Add validation"],
            partial_approvals={"s1": False},
        )
        coordinator._replan_rejected("goal", verdict)
        call_goal = mock_decomposer.analyze.call_args[0][0]
        assert "Fix auth" in call_goal
        assert "Add validation" in call_goal

    def test_revision_tags_subtask_ids(self, coordinator, mock_decomposer):
        mock_decomposer.analyze.return_value = _make_decomposition(
            subtasks=[_make_subtask("tmp1"), _make_subtask("tmp2")]
        )
        verdict = JudgeVerdict(
            approved=False,
            confidence=0.3,
            feedback="bad",
            partial_approvals={"s1": False, "s2": False},
        )
        result = coordinator._replan_rejected("goal", verdict)
        assert result.subtasks[0].id == "revision_s1"
        assert result.subtasks[1].id == "revision_s2"


# =========================================================================
# TestPhaseTracking
# =========================================================================


class TestPhaseTracking:
    def test_initial_phase(self, coordinator):
        assert coordinator.phase == CoordinationPhase.PLANNING

    @pytest.mark.asyncio
    async def test_phase_transitions(self, coordinator, mock_decomposer, mock_engine):
        """After successful coordination, phase is COMPLETED."""
        result = await coordinator.coordinate("test")
        assert coordinator.phase == CoordinationPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_phase_on_failure(self, mock_decomposer):
        engine = MagicMock()
        engine.execute = AsyncMock(return_value=_make_workflow_result(success=False, error="fail"))
        config = CoordinatorConfig(max_cycles=1, quality_threshold=1.0)
        coord = HierarchicalCoordinator(
            config=config,
            task_decomposer=mock_decomposer,
            workflow_engine=engine,
        )
        result = await coord.coordinate("doomed")
        assert coord.phase == CoordinationPhase.FAILED

    def test_cycle_starts_at_zero(self, coordinator):
        assert coordinator.cycle == 0

    @pytest.mark.asyncio
    async def test_cycle_increments(self, coordinator, mock_decomposer, mock_engine):
        await coordinator.coordinate("test")
        assert coordinator.cycle == 1
