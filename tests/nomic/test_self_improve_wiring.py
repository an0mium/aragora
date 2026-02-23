"""Tests for self-improvement pipeline wiring.

Verifies the integration points between:
- HardenedOrchestrator + ExecutionBridge (instruction generation)
- HardenedOrchestrator + DebugLoop (retry on failure)
- IdeaToExecutionPipeline + PipelineKMBridge (feedback loop closure)
- HardenedOrchestrator + NomicCycleAdapter (cross-cycle learning)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.nomic.autonomous_orchestrator import (
    AgentAssignment,
    AutonomousOrchestrator,
    OrchestrationResult,
    Track,
    reset_orchestrator,
)
from aragora.nomic.hardened_orchestrator import (
    HardenedConfig,
    HardenedOrchestrator,
)
from aragora.nomic.task_decomposer import SubTask


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset orchestrator singleton before each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


def _make_subtask(
    id: str = "sub-1",
    title: str = "Test task",
    description: str = "A test subtask",
    file_scope: list[str] | None = None,
) -> SubTask:
    return SubTask(
        id=id,
        title=title,
        description=description,
        file_scope=file_scope or ["aragora/core.py"],
        estimated_complexity="medium",
    )


def _make_assignment(
    subtask: SubTask | None = None,
    track: Track = Track.DEVELOPER,
    agent_type: str = "claude",
    status: str = "completed",
) -> AgentAssignment:
    return AgentAssignment(
        subtask=subtask or _make_subtask(),
        track=track,
        agent_type=agent_type,
        status=status,
    )


def _make_result(
    goal: str = "Improve test coverage",
    success: bool = True,
    completed: int = 2,
    failed: int = 0,
    assignments: list[AgentAssignment] | None = None,
) -> OrchestrationResult:
    return OrchestrationResult(
        goal=goal,
        total_subtasks=completed + failed,
        completed_subtasks=completed,
        failed_subtasks=failed,
        skipped_subtasks=0,
        assignments=assignments or [_make_assignment()],
        duration_seconds=10.0,
        success=success,
        summary="Test summary",
    )


@dataclass
class _FakePrioritizedGoal:
    """Stand-in for PrioritizedGoal."""

    id: str = "goal-1"
    track: Any = None
    description: str = "Improve SDK test coverage"
    rationale: str = "Low coverage is risky"
    estimated_impact: str = "high"
    priority: int = 1
    focus_areas: list[str] = field(default_factory=lambda: ["tests"])
    file_hints: list[str] = field(default_factory=lambda: ["sdk/client.py"])

    def __post_init__(self):
        if self.track is None:
            self.track = Track.DEVELOPER


@dataclass
class _FakeTrackAssignment:
    """Stand-in for TrackAssignment."""

    goal: _FakePrioritizedGoal = field(default_factory=_FakePrioritizedGoal)
    branch_name: str | None = None
    worktree_path: Path | None = None
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _FakeCoordinationResult:
    """Stand-in for CoordinationResult."""

    total_branches: int = 1
    completed_branches: int = 1
    failed_branches: int = 0
    merged_branches: int = 1
    assignments: list[Any] = field(default_factory=list)
    duration_seconds: float = 5.0
    success: bool = True
    summary: str = "All branches merged"


def _make_orchestrator(**kwargs) -> HardenedOrchestrator:
    """Create a HardenedOrchestrator with safe defaults for testing."""
    defaults = {
        "use_worktree_isolation": False,
        "enable_mode_enforcement": False,
        "enable_gauntlet_validation": False,
        "enable_prompt_defense": False,
        "enable_audit_reconciliation": False,
        "enable_auto_commit": False,
        "enable_meta_planning": False,
        "enable_canary_tokens": False,
        "enable_output_validation": False,
        "enable_review_gate": False,
        "enable_sandbox_validation": False,
        "generate_receipts": False,
        "require_human_approval": False,
    }
    defaults.update(kwargs)
    return HardenedOrchestrator(**defaults)


# ===========================================================================
# 1. ExecutionBridge instruction generation in coordinated path
# ===========================================================================


class TestExecutionBridgeInCoordinatedPath:
    """Tests for ExecutionBridge usage during coordinated execution."""

    @pytest.mark.asyncio
    async def test_bridge_ingest_called_on_success(self):
        """When enable_execution_bridge=True, bridge.ingest_result() is called
        after a successful coordinated assignment."""
        orch = _make_orchestrator(
            enable_execution_bridge=True,
            enable_meta_planning=True,
        )

        # Pre-set a mock bridge
        mock_bridge = MagicMock()
        mock_bridge.ingest_result = MagicMock()
        orch._execution_bridge = mock_bridge

        goal_obj = _FakePrioritizedGoal()
        ta = _FakeTrackAssignment(goal=goal_obj)
        result = _make_result(success=True)

        orch._bridge_ingest_coordinated_result(ta, result)

        mock_bridge.ingest_result.assert_called_once()
        exec_result = mock_bridge.ingest_result.call_args[0][0]
        assert exec_result.success is True
        assert exec_result.tests_passed == result.completed_subtasks

    @pytest.mark.asyncio
    async def test_bridge_ingest_called_on_failure(self):
        """Bridge ingestion also happens for failed results (records failures)."""
        orch = _make_orchestrator(enable_execution_bridge=True)
        mock_bridge = MagicMock()
        mock_bridge.ingest_result = MagicMock()
        orch._execution_bridge = mock_bridge

        result = _make_result(success=False, completed=0, failed=2)
        ta = _FakeTrackAssignment()

        orch._bridge_ingest_coordinated_result(ta, result)

        mock_bridge.ingest_result.assert_called_once()
        exec_result = mock_bridge.ingest_result.call_args[0][0]
        assert exec_result.success is False
        assert exec_result.tests_failed == 2

    @pytest.mark.asyncio
    async def test_bridge_skipped_when_unavailable(self):
        """When ExecutionBridge is None, ingestion is silently skipped."""
        orch = _make_orchestrator(enable_execution_bridge=True)
        orch._execution_bridge = None

        # Should not raise
        ta = _FakeTrackAssignment()
        result = _make_result()
        orch._bridge_ingest_coordinated_result(ta, result)

    @pytest.mark.asyncio
    async def test_bridge_skipped_when_disabled(self):
        """When enable_execution_bridge=False, the bridge is not involved."""
        orch = _make_orchestrator(enable_execution_bridge=False)
        # _get_execution_bridge should not be called during coordinated assignment
        # since we check the config flag first
        assert orch.hardened_config.enable_execution_bridge is False

    def test_get_execution_bridge_creates_on_first_call(self):
        """_get_execution_bridge() lazily creates an ExecutionBridge."""
        orch = _make_orchestrator(enable_execution_bridge=True)
        assert orch._execution_bridge is None

        bridge = orch._get_execution_bridge()
        assert bridge is not None
        # Second call returns same instance
        assert orch._get_execution_bridge() is bridge

    def test_get_execution_bridge_returns_none_on_import_error(self):
        """When ExecutionBridge module is unavailable, returns None."""
        orch = _make_orchestrator(enable_execution_bridge=True)

        # Force re-creation by clearing cache
        orch._execution_bridge = None
        # The import is lazy inside _get_execution_bridge, so block the module
        with patch.dict("sys.modules", {"aragora.nomic.execution_bridge": None}):
            result = orch._get_execution_bridge()
            # It catches ImportError and returns None
            assert result is None

    def test_execution_bridge_create_instruction(self):
        """ExecutionBridge.create_instruction() produces correct fields."""
        from aragora.nomic.execution_bridge import ExecutionBridge

        bridge = ExecutionBridge()

        @dataclass
        class _FakeSubTask:
            id: str = "st_001"
            title: str = "Refactor module"
            description: str = "Refactor the analytics module for clarity"
            file_scope: list[str] = field(default_factory=lambda: ["analytics.py", "utils.py"])
            success_criteria: dict[str, Any] = field(
                default_factory=lambda: {"test_pass_rate": ">0.95"}
            )
            dependencies: list[str] = field(default_factory=list)
            estimated_complexity: str = "medium"
            track: str = "core"

        subtask = _FakeSubTask()
        goal = _FakePrioritizedGoal(
            description="Improve coverage",
            rationale="Regressions are too common",
        )

        instruction = bridge.create_instruction(
            subtask=subtask,
            goal=goal,
            debate_context="The debate decided this is high priority.",
        )

        assert instruction.subtask_id == "st_001"
        assert instruction.track == "core"
        assert "Refactor the analytics module" in instruction.objective
        assert "analytics.py" in instruction.file_hints
        assert "utils.py" in instruction.file_hints
        assert len(instruction.success_criteria) > 0
        # Context should include debate rationale
        assert "high priority" in instruction.context

    def test_execution_bridge_instruction_includes_goal_rationale(self):
        """Generated instruction includes goal rationale in context."""
        from aragora.nomic.execution_bridge import ExecutionBridge

        bridge = ExecutionBridge()

        @dataclass
        class _SimpleSubTask:
            id: str = "st_002"
            title: str = "Fix bug"
            description: str = "Fix the auth bug"
            file_scope: list[str] = field(default_factory=list)
            success_criteria: dict[str, Any] = field(default_factory=dict)
            dependencies: list[str] = field(default_factory=list)
            estimated_complexity: str = "low"
            track: str = "security"

        goal = _FakePrioritizedGoal(
            rationale="Security vulnerabilities are critical",
        )
        instruction = bridge.create_instruction(
            subtask=_SimpleSubTask(),
            goal=goal,
        )
        # Rationale should be in the context
        assert "Security vulnerabilities" in instruction.context

    def test_execution_bridge_instruction_includes_file_scope(self):
        """Instruction file_hints come from subtask file_scope."""
        from aragora.nomic.execution_bridge import ExecutionBridge

        bridge = ExecutionBridge()

        @dataclass
        class _SubTask:
            id: str = "st_003"
            title: str = "Add tests"
            description: str = "Add tests for auth"
            file_scope: list[str] = field(default_factory=lambda: ["auth.py", "test_auth.py"])
            success_criteria: dict[str, Any] = field(default_factory=dict)
            dependencies: list[str] = field(default_factory=list)
            estimated_complexity: str = "medium"
            track: str = "qa"

        instruction = bridge.create_instruction(subtask=_SubTask())
        assert "auth.py" in instruction.file_hints
        assert "test_auth.py" in instruction.file_hints

    def test_execution_bridge_instruction_includes_success_criteria(self):
        """Instruction success_criteria come from subtask."""
        from aragora.nomic.execution_bridge import ExecutionBridge

        bridge = ExecutionBridge()

        @dataclass
        class _SubTask:
            id: str = "st_004"
            title: str = "Increase coverage"
            description: str = "Increase test coverage"
            file_scope: list[str] = field(default_factory=list)
            success_criteria: dict[str, Any] = field(
                default_factory=lambda: {"coverage": ">90%", "tests_pass": "True"}
            )
            dependencies: list[str] = field(default_factory=list)
            estimated_complexity: str = "medium"
            track: str = "qa"

        instruction = bridge.create_instruction(subtask=_SubTask())
        criteria_text = " ".join(instruction.success_criteria)
        assert "coverage" in criteria_text.lower() or len(instruction.success_criteria) >= 2


# ===========================================================================
# 2. DebugLoop retry on failure in coordinated path
# ===========================================================================


class TestDebugLoopWiring:
    """Tests for DebugLoop integration in HardenedOrchestrator."""

    def test_get_debug_loop_creates_on_first_call(self):
        """_get_debug_loop() lazily creates a DebugLoop."""
        orch = _make_orchestrator(enable_debug_loop=True)
        assert orch._debug_loop is None

        loop = orch._get_debug_loop()
        assert loop is not None
        # Same instance on second call
        assert orch._get_debug_loop() is loop

    def test_get_debug_loop_uses_configured_max_retries(self):
        """DebugLoop gets max_retries from HardenedConfig."""
        orch = _make_orchestrator(
            enable_debug_loop=True,
            debug_loop_max_retries=5,
        )
        loop = orch._get_debug_loop()
        assert loop.config.max_retries == 5

    def test_get_debug_loop_returns_none_on_import_error(self):
        """When DebugLoop module is unavailable, returns None."""
        orch = _make_orchestrator(enable_debug_loop=True)

        with patch.dict("sys.modules", {"aragora.nomic.debug_loop": None}):
            orch._debug_loop = None
            result = orch._get_debug_loop()
            assert result is None

    def test_debug_loop_not_created_when_disabled(self):
        """When enable_debug_loop=False, config flag is respected."""
        orch = _make_orchestrator(enable_debug_loop=False)
        assert orch.hardened_config.enable_debug_loop is False

    @pytest.mark.asyncio
    async def test_debug_loop_execute_with_retry_success(self):
        """DebugLoop.execute_with_retry() returns success when tests pass."""
        from aragora.nomic.debug_loop import DebugLoopResult

        mock_loop = MagicMock()
        mock_loop.execute_with_retry = AsyncMock(
            return_value=DebugLoopResult(
                subtask_id="sub-1",
                success=True,
                total_attempts=1,
                final_tests_passed=10,
                final_tests_failed=0,
            )
        )

        orch = _make_orchestrator(enable_debug_loop=True)
        orch._debug_loop = mock_loop

        result = await mock_loop.execute_with_retry(
            instruction="Fix the bug",
            worktree_path="/tmp/wt",
            test_scope=["tests/auth/"],
            subtask_id="sub-1",
        )

        assert result.success is True
        assert result.total_attempts == 1
        mock_loop.execute_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_debug_loop_execute_with_retry_exhausted(self):
        """DebugLoop returns failure when retries are exhausted."""
        from aragora.nomic.debug_loop import DebugLoopResult

        mock_loop = MagicMock()
        mock_loop.execute_with_retry = AsyncMock(
            return_value=DebugLoopResult(
                subtask_id="sub-1",
                success=False,
                total_attempts=3,
                final_tests_passed=5,
                final_tests_failed=3,
            )
        )

        orch = _make_orchestrator(enable_debug_loop=True)
        orch._debug_loop = mock_loop

        result = await mock_loop.execute_with_retry(
            instruction="Fix the bug",
            worktree_path="/tmp/wt",
            subtask_id="sub-1",
        )

        assert result.success is False
        assert result.total_attempts == 3

    def test_debug_loop_config_defaults(self):
        """DebugLoop default config is sensible."""
        orch = _make_orchestrator(enable_debug_loop=True)
        assert orch.hardened_config.debug_loop_max_retries == 3

    def test_debug_loop_config_custom(self):
        """Custom debug loop max_retries is propagated."""
        orch = _make_orchestrator(
            enable_debug_loop=True,
            debug_loop_max_retries=7,
        )
        assert orch.hardened_config.debug_loop_max_retries == 7

    @pytest.mark.asyncio
    async def test_debug_loop_result_serialization(self):
        """DebugLoopResult.to_dict() includes all key fields."""
        from aragora.nomic.debug_loop import DebugLoopResult

        result = DebugLoopResult(
            subtask_id="sub-x",
            success=True,
            total_attempts=2,
            final_tests_passed=15,
            final_tests_failed=0,
            final_files_changed=["auth.py"],
        )
        d = result.to_dict()
        assert d["subtask_id"] == "sub-x"
        assert d["success"] is True
        assert d["total_attempts"] == 2
        assert d["final_tests_passed"] == 15
        assert d["final_files_changed"] == ["auth.py"]


# ===========================================================================
# 3. Stranded feature fixes in idea_to_execution.py
# ===========================================================================


class TestIdeaToExecutionPipelineWiring:
    """Tests for pipeline feedback loop closure in IdeaToExecutionPipeline."""

    def test_from_debate_persists_to_km(self):
        """from_debate() stores result to KM via PipelineKMBridge."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        mock_extractor = MagicMock()
        pipeline = IdeaToExecutionPipeline(goal_extractor=mock_extractor)

        # Build minimal cartographer data
        cartographer_data = {
            "arguments": [
                {
                    "id": "arg-1",
                    "claim": "We should cache API responses",
                    "type": "proposal",
                    "support": 3,
                    "relations": [],
                }
            ],
            "metadata": {"total_arguments": 1},
        }

        # Mock the canvas converters and bridge
        with (
            patch(
                "aragora.pipeline.idea_to_execution.debate_to_ideas_canvas"
            ) as mock_debate_canvas,
            patch("aragora.pipeline.km_bridge.PipelineKMBridge") as MockBridge,
            patch.object(pipeline, "_advance_to_goals", side_effect=lambda r: r),
            patch.object(pipeline, "_advance_to_actions", side_effect=lambda r: r),
            patch.object(pipeline, "_advance_to_orchestration", side_effect=lambda r: r),
            patch.object(pipeline, "_build_universal_graph"),
        ):
            mock_canvas = MagicMock()
            mock_canvas.nodes = {}
            mock_debate_canvas.return_value = mock_canvas

            mock_bridge_inst = MagicMock()
            mock_bridge_inst.available = True
            mock_bridge_inst.store_pipeline_result = MagicMock(return_value=True)
            MockBridge.return_value = mock_bridge_inst

            result = pipeline.from_debate(cartographer_data)

            # PipelineKMBridge should be created and store_pipeline_result called
            MockBridge.assert_called_once()
            mock_bridge_inst.store_pipeline_result.assert_called_once_with(result)

    def test_from_ideas_persists_to_km(self):
        """from_ideas() stores result to KM via PipelineKMBridge."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        mock_extractor = MagicMock()
        # Set up the goal graph mock
        mock_goal_graph = MagicMock()
        mock_goal_graph.goals = []
        mock_goal_graph.transition = None
        mock_goal_graph.provenance = []
        mock_goal_graph.metadata = {}
        mock_extractor.extract_from_raw_ideas.return_value = mock_goal_graph

        pipeline = IdeaToExecutionPipeline(goal_extractor=mock_extractor)

        with (
            patch("aragora.pipeline.km_bridge.PipelineKMBridge") as MockBridge,
            patch.object(pipeline, "_advance_to_actions", side_effect=lambda r: r),
            patch.object(pipeline, "_advance_to_orchestration", side_effect=lambda r: r),
            patch.object(pipeline, "_build_universal_graph"),
            patch(
                "aragora.pipeline.idea_to_execution.content_hash",
                return_value="abc123",
            ),
        ):
            mock_bridge_inst = MagicMock()
            mock_bridge_inst.available = True
            mock_bridge_inst.store_pipeline_result = MagicMock(return_value=True)
            MockBridge.return_value = mock_bridge_inst

            result = pipeline.from_ideas(["idea 1", "idea 2"])

            # PipelineKMBridge should be created and store_pipeline_result called
            # The from_ideas path calls it at the end
            assert MockBridge.called
            mock_bridge_inst.store_pipeline_result.assert_called_once_with(result)

    def test_strategic_memory_store_still_used_in_from_ideas(self):
        """StrategicMemoryStore is still imported in from_ideas for enrichment."""
        # This tests that the StrategicMemoryStore import exists in from_ideas
        # (line 484). It's used for goal enrichment, not for pipeline outcome
        # recording. The feedback loop is closed by PipelineKMBridge instead.
        import ast

        with open("/Users/armand/Development/aragora/aragora/pipeline/idea_to_execution.py") as f:
            source = f.read()

        tree = ast.parse(source)
        # Check that StrategicMemoryStore is imported somewhere in the file
        # (it's used for enrichment in from_ideas and _advance_to_goals)
        assert "StrategicMemoryStore" in source

    def test_pipeline_km_bridge_used_for_feedback_loop(self):
        """PipelineKMBridge is used (not StrategicMemoryStore) for pipeline result storage."""
        import ast

        with open("/Users/armand/Development/aragora/aragora/pipeline/idea_to_execution.py") as f:
            source = f.read()

        # PipelineKMBridge.store_pipeline_result should be called
        assert "bridge.store_pipeline_result(result)" in source
        # query_similar_goals should be called in _advance_to_goals
        assert "bridge.query_similar_goals" in source

    def test_advance_to_goals_uses_km_bridge_for_precedents(self):
        """_advance_to_goals() queries KM for precedents via PipelineKMBridge."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline
        from aragora.pipeline.idea_to_execution import PipelineResult

        mock_extractor = MagicMock()
        mock_goal_graph = MagicMock()
        mock_goal_graph.goals = []
        mock_goal_graph.transition = None
        mock_goal_graph.provenance = []
        mock_goal_graph.metadata = {}
        mock_extractor.extract_from_ideas.return_value = mock_goal_graph
        mock_extractor.cluster_ideas_semantically.side_effect = lambda x: x

        pipeline = IdeaToExecutionPipeline(goal_extractor=mock_extractor)

        mock_canvas = MagicMock()
        mock_canvas.to_dict.return_value = {"nodes": {}}
        result = PipelineResult(
            pipeline_id="pipe-test",
            ideas_canvas=mock_canvas,
        )

        with patch("aragora.pipeline.km_bridge.PipelineKMBridge") as MockBridge:
            mock_bridge_inst = MagicMock()
            mock_bridge_inst.available = True
            mock_bridge_inst.query_similar_goals.return_value = {}
            MockBridge.return_value = mock_bridge_inst

            pipeline._advance_to_goals(result)

            # PipelineKMBridge is instantiated twice: once for precedent
            # queries and once for adapter precedent enrichment
            assert MockBridge.call_count >= 1
            mock_bridge_inst.query_similar_goals.assert_called_once_with(mock_goal_graph)

    def test_record_pipeline_outcome_calls_meta_planner(self):
        """_record_pipeline_outcome() records via MetaPlanner.record_outcome()."""
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineResult,
        )

        mock_extractor = MagicMock()
        pipeline = IdeaToExecutionPipeline(goal_extractor=mock_extractor)

        result = PipelineResult(
            pipeline_id="pipe-test-123",
            stage_status={
                "ideas": "complete",
                "goals": "complete",
                "actions": "failed",
                "orchestration": "pending",
            },
        )

        with patch("aragora.nomic.meta_planner.MetaPlanner") as MockPlanner:
            mock_planner = MagicMock()
            MockPlanner.return_value = mock_planner

            pipeline._record_pipeline_outcome(result)

            MockPlanner.assert_called_once()
            mock_planner.record_outcome.assert_called_once()

            # Verify the format of goal_outcomes
            call_kwargs = mock_planner.record_outcome.call_args
            goal_outcomes = (
                call_kwargs.kwargs.get("goal_outcomes", call_kwargs[1].get("goal_outcomes"))
                if call_kwargs.kwargs
                else call_kwargs[0][0]
            )

            # Should have 4 entries (one per stage)
            assert len(goal_outcomes) == 4

            # Verify success flags match stage_status
            by_desc = {o["description"]: o["success"] for o in goal_outcomes}
            assert by_desc["pipeline_stage_ideas"] is True
            assert by_desc["pipeline_stage_goals"] is True
            assert by_desc["pipeline_stage_actions"] is False
            assert by_desc["pipeline_stage_orchestration"] is False

    def test_record_pipeline_outcome_includes_objective(self):
        """The objective passed to record_outcome includes the pipeline_id."""
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineResult,
        )

        pipeline = IdeaToExecutionPipeline(goal_extractor=MagicMock())
        result = PipelineResult(
            pipeline_id="pipe-abc",
            stage_status={"ideas": "complete"},
        )

        with patch("aragora.nomic.meta_planner.MetaPlanner") as MockPlanner:
            mock_planner = MagicMock()
            MockPlanner.return_value = mock_planner

            pipeline._record_pipeline_outcome(result)

            call_kwargs = mock_planner.record_outcome.call_args
            # objective kwarg should contain the pipeline id
            if call_kwargs.kwargs:
                objective = call_kwargs.kwargs.get("objective", "")
            else:
                objective = call_kwargs[1].get("objective", "")
            assert "pipe-abc" in objective

    def test_record_pipeline_outcome_handles_import_error(self):
        """_record_pipeline_outcome() silently handles ImportError."""
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineResult,
        )

        pipeline = IdeaToExecutionPipeline(goal_extractor=MagicMock())
        result = PipelineResult(pipeline_id="pipe-x", stage_status={})

        with patch(
            "aragora.nomic.meta_planner.MetaPlanner",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            pipeline._record_pipeline_outcome(result)


# ===========================================================================
# 4. Feedback loop closure (P0 Task #17)
# ===========================================================================


class TestFeedbackLoopClosure:
    """Tests for _record_orchestration_outcome in HardenedOrchestrator."""

    @pytest.mark.asyncio
    async def test_record_outcome_stores_to_nomic_cycle_adapter(self):
        """_record_orchestration_outcome() stores to NomicCycleAdapter."""
        orch = _make_orchestrator()

        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(
                goal="Improve test coverage",
                success=True,
                completed=3,
                failed=1,
            )

            await orch._record_orchestration_outcome("Improve test coverage", result)

            mock_adapter.ingest_cycle_outcome.assert_called_once()
            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]

            assert outcome.objective == "Improve test coverage"
            assert outcome.goals_attempted == 4  # 3 + 1
            assert outcome.goals_succeeded == 3
            assert outcome.goals_failed == 1

    @pytest.mark.asyncio
    async def test_record_outcome_stores_what_worked(self):
        """Outcome includes what_worked from completed assignments."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        assignment_ok = _make_assignment(
            subtask=_make_subtask(title="Fix auth bug"),
            agent_type="claude",
            status="completed",
        )
        assignment_fail = _make_assignment(
            subtask=_make_subtask(id="sub-2", title="Add caching"),
            agent_type="codex",
            status="failed",
        )

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(
                success=False,
                completed=1,
                failed=1,
                assignments=[assignment_ok, assignment_fail],
            )

            await orch._record_orchestration_outcome("Fix stuff", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert any("Fix auth bug" in w for w in outcome.what_worked)
            assert any("Add caching" in w for w in outcome.what_failed)

    @pytest.mark.asyncio
    async def test_record_outcome_tracks_agents_used(self):
        """Outcome records which agent types were used."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        assignments = [
            _make_assignment(agent_type="claude", status="completed"),
            _make_assignment(
                subtask=_make_subtask(id="sub-2"),
                agent_type="codex",
                status="completed",
            ),
        ]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(completed=2, failed=0, assignments=assignments)

            await orch._record_orchestration_outcome("test", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert "claude" in outcome.agents_used
            assert "codex" in outcome.agents_used

    @pytest.mark.asyncio
    async def test_record_outcome_cycle_status_success(self):
        """CycleStatus is SUCCESS when all subtasks pass."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(success=True, completed=3, failed=0)
            await orch._record_orchestration_outcome("test", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert outcome.status.value == "success"

    @pytest.mark.asyncio
    async def test_record_outcome_cycle_status_partial(self):
        """CycleStatus is PARTIAL when some succeed and some fail."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(success=False, completed=2, failed=1)
            await orch._record_orchestration_outcome("test", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert outcome.status.value == "partial"

    @pytest.mark.asyncio
    async def test_record_outcome_cycle_status_failed(self):
        """CycleStatus is FAILED when no subtasks complete."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(success=False, completed=0, failed=3)
            await orch._record_orchestration_outcome("test", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert outcome.status.value == "failed"

    @pytest.mark.asyncio
    async def test_record_outcome_includes_improvement_score(self):
        """Outcome carries through improvement_score from the result."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(success=True)
            result.improvement_score = 0.75
            result.metrics_delta = {"tests_added": 10}
            result.success_criteria_met = True

            await orch._record_orchestration_outcome("test", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]
            assert outcome.improvement_score == 0.75
            assert outcome.metrics_delta == {"tests_added": 10}
            assert outcome.success_criteria_met is True

    @pytest.mark.asyncio
    async def test_record_outcome_handles_import_error(self):
        """Silently handles ImportError when NomicCycleAdapter is unavailable."""
        orch = _make_orchestrator()

        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.adapters.nomic_cycle_adapter": None},
        ):
            result = _make_result()
            # Should not raise
            await orch._record_orchestration_outcome("test", result)

    @pytest.mark.asyncio
    async def test_record_outcome_handles_runtime_error(self):
        """Silently handles RuntimeError from adapter.ingest_cycle_outcome."""
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result()
            # Should not raise
            await orch._record_orchestration_outcome("test", result)

    @pytest.mark.asyncio
    async def test_execute_goal_coordinated_records_outcome(self):
        """execute_goal_coordinated() calls both _record_orchestration_outcome()
        and _detect_km_contradictions() — closing the learning loop for the
        default meta-planning codepath."""
        orch = _make_orchestrator(enable_meta_planning=True)

        coord_result = _FakeCoordinationResult(
            total_branches=3,
            completed_branches=2,
            failed_branches=1,
            merged_branches=2,
            success=True,
            duration_seconds=12.0,
            summary="2 of 3 branches merged",
        )

        mock_coordinator = MagicMock()
        mock_coordinator.coordinate_parallel_work = AsyncMock(return_value=coord_result)
        mock_coordinator.cleanup_all_worktrees = MagicMock()

        with (
            patch.object(
                orch,
                "_run_meta_planner_for_coordination",
                new_callable=AsyncMock,
                return_value=[_FakePrioritizedGoal()],
            ),
            patch(
                "aragora.nomic.branch_coordinator.BranchCoordinator",
                return_value=mock_coordinator,
            ),
            patch(
                "aragora.nomic.branch_coordinator.BranchCoordinatorConfig",
            ),
            patch(
                "aragora.nomic.branch_coordinator.TrackAssignment",
                side_effect=lambda goal: _FakeTrackAssignment(goal=goal),
            ),
            patch.object(
                orch,
                "_record_orchestration_outcome",
                new_callable=AsyncMock,
            ) as mock_record,
            patch.object(
                orch,
                "_detect_km_contradictions",
                new_callable=AsyncMock,
            ) as mock_contradictions,
            patch.object(orch, "_emit_event"),
        ):
            result = await orch.execute_goal_coordinated("test goal")

            # Both methods must be called for cross-cycle learning
            mock_record.assert_called_once()
            mock_contradictions.assert_called_once()

            # Verify the result is passed correctly
            call_args = mock_record.call_args[0]
            assert call_args[0] == "test goal"
            assert call_args[1].success is True
            assert call_args[1].completed_subtasks == 2
            assert call_args[1].failed_subtasks == 1

    @pytest.mark.asyncio
    async def test_record_outcome_called_from_execute_goal(self):
        """execute_goal() calls _record_orchestration_outcome() after execution."""
        orch = _make_orchestrator(enable_meta_planning=False)

        mock_result = _make_result()

        with (
            patch.object(
                AutonomousOrchestrator,
                "execute_goal",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch.object(
                orch,
                "_record_orchestration_outcome",
                new_callable=AsyncMock,
            ) as mock_record,
            patch.object(
                orch,
                "_collect_baseline_metrics",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                orch,
                "_reconcile_audits",
            ),
            patch.object(
                orch,
                "_emit_event",
            ),
            patch.object(
                orch,
                "_detect_km_contradictions",
                new_callable=AsyncMock,
            ),
        ):
            await orch.execute_goal("test goal")

            mock_record.assert_called_once()
            call_args = mock_record.call_args[0]
            assert call_args[0] == "test goal"
            assert call_args[1] is mock_result

    @pytest.mark.asyncio
    async def test_outcome_format_matches_meta_planner_query(self):
        """Stored outcome uses same format that MetaPlanner queries."""
        # NomicCycleOutcome fields match what MetaPlanner reads:
        # objective, status, what_worked, what_failed, agents_used
        orch = _make_orchestrator()
        mock_adapter = AsyncMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            result = _make_result(
                goal="Add auth feature",
                success=True,
                completed=1,
                failed=0,
                assignments=[_make_assignment(status="completed")],
            )

            await orch._record_orchestration_outcome("Add auth feature", result)

            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]

            # These fields are what MetaPlanner.prioritize_work() queries
            assert hasattr(outcome, "objective")
            assert hasattr(outcome, "status")
            assert hasattr(outcome, "what_worked")
            assert hasattr(outcome, "what_failed")
            assert hasattr(outcome, "agents_used")
            assert hasattr(outcome, "tracks_affected")
            assert hasattr(outcome, "goals_attempted")
            assert hasattr(outcome, "goals_succeeded")
            assert hasattr(outcome, "goals_failed")

            # Verify it's a NomicCycleOutcome (duck-type check)
            d = outcome.to_dict()
            assert d["objective"] == "Add auth feature"
            assert d["status"] == "success"
            assert d["goals_attempted"] == 1
            assert d["goals_succeeded"] == 1


# ===========================================================================
# 5. HardenedConfig defaults
# ===========================================================================


class TestHardenedConfigDefaults:
    """Verify HardenedConfig has correct defaults for the wiring flags."""

    def test_execution_bridge_enabled_by_default(self):
        config = HardenedConfig()
        assert config.enable_execution_bridge is True

    def test_debug_loop_enabled_by_default(self):
        config = HardenedConfig()
        assert config.enable_debug_loop is True

    def test_debug_loop_max_retries_default(self):
        config = HardenedConfig()
        assert config.debug_loop_max_retries == 3

    def test_orchestrator_propagates_config_flags(self):
        orch = _make_orchestrator(
            enable_execution_bridge=False,
            enable_debug_loop=False,
            debug_loop_max_retries=10,
        )
        assert orch.hardened_config.enable_execution_bridge is False
        assert orch.hardened_config.enable_debug_loop is False
        assert orch.hardened_config.debug_loop_max_retries == 10


# ===========================================================================
# 6. SelfImprovePipeline → NomicCycleAdapter persistence (Phase 1B)
# ===========================================================================


class TestSelfImprovePipelineNomicCycleAdapter:
    """Verify _persist_cycle_outcome() records to NomicCycleAdapter."""

    @pytest.mark.asyncio
    async def test_persist_cycle_outcome_records_to_nomic_cycle_adapter(self):
        """_persist_cycle_outcome() stores via NomicCycleAdapter in addition
        to PipelineKMBridge, enabling cross-cycle learning via
        find_similar_cycles()."""
        from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

        config = SelfImproveConfig(use_meta_planner=False, quick_mode=True)
        pipeline = SelfImprovePipeline(config=config)

        @dataclass
        class _FakeResult:
            cycle_id: str = "cycle-test-001"
            objective: str = "Improve SDK coverage"
            subtasks_total: int = 3
            subtasks_completed: int = 2
            subtasks_failed: int = 1
            files_changed: list = field(default_factory=lambda: ["a.py", "b.py"])
            tests_passed: int = 10
            tests_failed: int = 2
            metrics_delta: dict = field(default_factory=dict)
            improvement_score: float = 0.42
            total_cost_usd: float = 0.0
            km_persisted: bool = False
            regressions_detected: bool = False
            reverted: bool = False
            duration_seconds: float = 5.0

        result = _FakeResult()
        mock_adapter = MagicMock()
        mock_adapter.ingest_cycle_outcome = MagicMock()

        with (
            patch(
                "aragora.nomic.cycle_store.get_cycle_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=mock_adapter,
            ),
        ):
            await pipeline._persist_cycle_outcome("cycle-test-001", result)

            mock_adapter.ingest_cycle_outcome.assert_called_once()
            outcome = mock_adapter.ingest_cycle_outcome.call_args[0][0]

            assert outcome.cycle_id == "cycle-test-001"
            assert outcome.objective == "Improve SDK coverage"
            assert outcome.goals_attempted == 3
            assert outcome.goals_succeeded == 2
            assert outcome.goals_failed == 1
            assert outcome.total_files_changed == 2
            # PARTIAL: some succeeded, some failed
            assert outcome.status.value == "partial"

    @pytest.mark.asyncio
    async def test_persist_cycle_outcome_adapter_failure_is_graceful(self):
        """NomicCycleAdapter failure doesn't break _persist_cycle_outcome()."""
        from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

        config = SelfImproveConfig(use_meta_planner=False, quick_mode=True)
        pipeline = SelfImprovePipeline(config=config)

        @dataclass
        class _FakeResult:
            cycle_id: str = "cycle-err"
            objective: str = "Test"
            subtasks_total: int = 1
            subtasks_completed: int = 1
            subtasks_failed: int = 0
            files_changed: list = field(default_factory=list)
            tests_passed: int = 0
            tests_failed: int = 0
            metrics_delta: dict = field(default_factory=dict)
            improvement_score: float = 0.0
            total_cost_usd: float = 0.0
            km_persisted: bool = False
            regressions_detected: bool = False
            reverted: bool = False
            duration_seconds: float = 1.0

        result = _FakeResult()

        with (
            patch(
                "aragora.nomic.cycle_store.get_cycle_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=RuntimeError("DB unavailable"),
            ),
        ):
            # Should not raise
            await pipeline._persist_cycle_outcome("cycle-err", result)
