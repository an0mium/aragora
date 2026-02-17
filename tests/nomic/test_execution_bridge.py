"""Tests for ExecutionBridge -- translating debate decisions into agent instructions."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Any

from aragora.nomic.execution_bridge import (
    ExecutionBridge,
    ExecutionInstruction,
    ExecutionResult,
)


# ---------------------------------------------------------------------------
# Fixtures: lightweight stand-ins for SubTask and PrioritizedGoal
# ---------------------------------------------------------------------------


@dataclass
class _FakeSubTask:
    """Mimics aragora.nomic.task_decomposer.SubTask for testing."""

    id: str = "st_001"
    title: str = "Refactor module"
    description: str = "Refactor the analytics module for clarity"
    file_scope: list[str] = field(default_factory=lambda: ["analytics.py", "utils.py"])
    success_criteria: dict[str, Any] = field(
        default_factory=lambda: {"test_pass_rate": ">0.95", "lint_errors": "==0"}
    )
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"
    track: str = "core"


@dataclass
class _MinimalSubTask:
    """SubTask-like object with only description (all other fields missing)."""

    description: str = "Do something"


@dataclass
class _FakeGoal:
    """Mimics aragora.nomic.meta_planner.PrioritizedGoal for testing."""

    id: str = "goal_1"
    description: str = "Improve test coverage"
    rationale: str = "Low coverage risks regressions"
    estimated_impact: str = "high"
    priority: int = 1
    focus_areas: list[str] = field(default_factory=lambda: ["tests"])
    file_hints: list[str] = field(default_factory=lambda: ["test_analytics.py"])


# ===========================================================================
# TestExecutionInstruction
# ===========================================================================


class TestExecutionInstruction:
    """Tests for the ExecutionInstruction dataclass."""

    def test_to_agent_prompt_full(self):
        """to_agent_prompt() includes all sections when fields are populated."""
        inst = ExecutionInstruction(
            subtask_id="st_001",
            track="core",
            objective="Refactor analytics",
            context="The debate concluded this is highest priority.",
            file_hints=["analytics.py", "utils.py"],
            success_criteria=["test_pass_rate: >0.95", "lint_errors: ==0"],
            constraints=["Do not break API"],
            worktree_path="/tmp/wt",
            budget_limit_usd=1.50,
        )
        prompt = inst.to_agent_prompt()

        assert "# Task: Refactor analytics" in prompt
        assert "## Context" in prompt
        assert "highest priority" in prompt
        assert "## Relevant Files" in prompt
        assert "`analytics.py`" in prompt
        assert "`utils.py`" in prompt
        assert "## Success Criteria" in prompt
        assert "test_pass_rate: >0.95" in prompt
        assert "## Constraints" in prompt
        assert "Do not break API" in prompt
        assert "## Verification" in prompt
        assert "pytest" in prompt

    def test_to_agent_prompt_empty_optional(self):
        """to_agent_prompt() omits optional sections when lists are empty."""
        inst = ExecutionInstruction(
            subtask_id="st_002",
            track="qa",
            objective="Quick fix",
            context="Minimal context.",
            file_hints=[],
            success_criteria=[],
            constraints=[],
        )
        prompt = inst.to_agent_prompt()

        assert "# Task: Quick fix" in prompt
        assert "## Context" in prompt
        assert "## Relevant Files" not in prompt
        assert "## Success Criteria" not in prompt
        assert "## Constraints" not in prompt
        # Verification section is always present
        assert "## Verification" in prompt

    def test_to_dict_serialization(self):
        """to_dict() returns all fields in a JSON-serializable dict."""
        inst = ExecutionInstruction(
            subtask_id="st_003",
            track="security",
            objective="Audit module",
            context="Context text",
            file_hints=["auth.py"],
            success_criteria=["no critical findings"],
            constraints=["read-only"],
            worktree_path="/tmp/wt2",
            budget_limit_usd=2.0,
        )
        d = inst.to_dict()

        assert d["subtask_id"] == "st_003"
        assert d["track"] == "security"
        assert d["objective"] == "Audit module"
        assert d["worktree_path"] == "/tmp/wt2"
        assert d["budget_limit_usd"] == 2.0
        assert "auth.py" in d["file_hints"]


# ===========================================================================
# TestExecutionResult
# ===========================================================================


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_to_dict_success(self):
        """to_dict() serializes a successful result correctly."""
        result = ExecutionResult(
            subtask_id="st_001",
            success=True,
            files_changed=["a.py", "b.py"],
            tests_passed=10,
            tests_failed=0,
            duration_seconds=12.5,
            tokens_used=1500,
            diff_summary="diff --git a/a.py",
        )
        d = result.to_dict()

        assert d["subtask_id"] == "st_001"
        assert d["success"] is True
        assert d["files_changed"] == ["a.py", "b.py"]
        assert d["tests_passed"] == 10
        assert d["tests_failed"] == 0
        assert d["error"] is None
        assert d["duration_seconds"] == 12.5
        assert d["tokens_used"] == 1500

    def test_to_dict_failure(self):
        """to_dict() serializes a failed result with error."""
        result = ExecutionResult(
            subtask_id="st_002",
            success=False,
            error="Import failed: no module named foo",
            tests_passed=3,
            tests_failed=2,
        )
        d = result.to_dict()

        assert d["success"] is False
        assert "Import failed" in d["error"]

    def test_diff_summary_truncated(self):
        """to_dict() truncates diff_summary to 500 chars."""
        long_diff = "x" * 1000
        result = ExecutionResult(subtask_id="st_003", success=True, diff_summary=long_diff)
        d = result.to_dict()

        assert len(d["diff_summary"]) == 500

    def test_default_fields(self):
        """Default fields are populated correctly."""
        result = ExecutionResult(subtask_id="st_004", success=True)

        assert result.files_changed == []
        assert result.tests_passed == 0
        assert result.tests_failed == 0
        assert result.error is None
        assert result.duration_seconds == 0.0
        assert result.tokens_used == 0
        assert result.diff_summary == ""
        assert result.agent_observations == []


# ===========================================================================
# TestExecutionBridge
# ===========================================================================


class TestExecutionBridge:
    """Tests for the ExecutionBridge class."""

    def test_create_instruction_full_subtask(self):
        """create_instruction() packages all SubTask fields into an instruction."""
        bridge = ExecutionBridge()
        subtask = _FakeSubTask()

        inst = bridge.create_instruction(subtask)

        assert inst.subtask_id == "st_001"
        assert inst.track == "core"
        assert inst.objective == "Refactor the analytics module for clarity"
        assert "analytics.py" in inst.file_hints
        assert "utils.py" in inst.file_hints
        assert any("test_pass_rate" in c for c in inst.success_criteria)
        assert any("lint_errors" in c for c in inst.success_criteria)
        assert len(inst.constraints) == 4  # base constraints

    def test_create_instruction_minimal_subtask(self):
        """create_instruction() uses getattr defaults for minimal SubTask."""
        bridge = ExecutionBridge()
        subtask = _MinimalSubTask(description="Fix a bug")

        inst = bridge.create_instruction(subtask)

        assert inst.objective == "Fix a bug"
        assert inst.file_hints == []
        assert inst.success_criteria == []
        assert inst.track == "core"  # default
        # subtask_id generated from time since no id attr
        assert inst.subtask_id.startswith("subtask_")

    def test_create_instruction_with_goal_context(self):
        """create_instruction() includes goal description and rationale in context."""
        bridge = ExecutionBridge()
        subtask = _FakeSubTask()
        goal = _FakeGoal()

        inst = bridge.create_instruction(
            subtask,
            goal=goal,
            debate_context="Debate voted 4-1 in favor.",
        )

        assert "Debate voted 4-1 in favor." in inst.context
        assert "Improve test coverage" in inst.context
        assert "Low coverage risks regressions" in inst.context
        assert "high" in inst.context

    def test_create_instruction_with_extra_constraints(self):
        """Extra constraints are appended to base constraints."""
        bridge = ExecutionBridge(base_constraints=["Rule A"])
        subtask = _FakeSubTask()

        inst = bridge.create_instruction(
            subtask, extra_constraints=["Rule B", "Rule C"]
        )

        assert "Rule A" in inst.constraints
        assert "Rule B" in inst.constraints
        assert "Rule C" in inst.constraints
        assert len(inst.constraints) == 3

    def test_create_instruction_with_worktree_and_budget(self):
        """Worktree path and budget are passed through."""
        bridge = ExecutionBridge()
        subtask = _FakeSubTask()

        inst = bridge.create_instruction(
            subtask, worktree_path="/tmp/wt_test", budget_limit_usd=5.0
        )

        assert inst.worktree_path == "/tmp/wt_test"
        assert inst.budget_limit_usd == 5.0

    def test_create_instruction_dict_success_criteria(self):
        """SubTask with dict success_criteria is converted to list of 'key: value' strings."""
        bridge = ExecutionBridge()
        subtask = _FakeSubTask(
            success_criteria={"coverage": ">80%", "no_regressions": "true"}
        )

        inst = bridge.create_instruction(subtask)

        assert any("coverage" in c and ">80%" in c for c in inst.success_criteria)
        assert any("no_regressions" in c for c in inst.success_criteria)

    def test_create_instruction_list_success_criteria(self):
        """SubTask with list success_criteria (non-standard) is handled gracefully."""
        bridge = ExecutionBridge()
        subtask = _FakeSubTask()
        # Monkey-patch to list instead of dict
        subtask.success_criteria = ["all tests pass", "no lint errors"]

        inst = bridge.create_instruction(subtask)

        assert "all tests pass" in inst.success_criteria
        assert "no lint errors" in inst.success_criteria

    def test_ingest_result_with_km_adapter(self):
        """ingest_result() calls KM adapter when available."""
        import asyncio

        bridge = ExecutionBridge(enable_km_ingestion=True)
        result = ExecutionResult(
            subtask_id="st_km_001",
            success=True,
            files_changed=["a.py"],
            tests_passed=5,
            tests_failed=0,
            agent_observations=["Refactored cleanly"],
        )

        mock_adapter = MagicMock()
        mock_adapter.ingest_cycle_outcome = AsyncMock()

        mock_loop = MagicMock()

        with patch.object(
            asyncio, "get_running_loop", return_value=mock_loop
        ):
            with patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=mock_adapter,
            ):
                bridge.ingest_result(result)

        assert len(bridge._results) == 1
        assert bridge._results[0].subtask_id == "st_km_001"
        # The adapter's ingest was scheduled as a task
        mock_loop.create_task.assert_called_once()

    def test_ingest_result_km_import_error(self):
        """ingest_result() handles ImportError gracefully."""
        bridge = ExecutionBridge(enable_km_ingestion=True)
        result = ExecutionResult(subtask_id="st_no_km", success=True)

        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.adapters.nomic_cycle_adapter": None},
        ):
            # Should not raise
            bridge.ingest_result(result)

        assert len(bridge._results) == 1

    def test_ingest_result_km_disabled(self):
        """ingest_result() skips KM when disabled."""
        bridge = ExecutionBridge(enable_km_ingestion=False)
        result = ExecutionResult(subtask_id="st_skip", success=True)

        # No KM import should be attempted
        bridge.ingest_result(result)

        assert len(bridge._results) == 1

    def test_ingest_result_no_event_loop(self):
        """ingest_result() handles missing event loop gracefully."""
        import asyncio

        bridge = ExecutionBridge(enable_km_ingestion=True)
        result = ExecutionResult(
            subtask_id="st_no_loop",
            success=True,
            files_changed=["x.py"],
        )

        mock_adapter = MagicMock()

        with patch.object(
            asyncio, "get_running_loop", side_effect=RuntimeError("no loop")
        ):
            with patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=mock_adapter,
            ):
                # Should not raise
                bridge.ingest_result(result)

        assert len(bridge._results) == 1

    @pytest.mark.asyncio
    async def test_verify_changes_with_review(self):
        """verify_changes() calls PRReviewRunner and returns findings."""
        bridge = ExecutionBridge(enable_verification=True)
        result = ExecutionResult(
            subtask_id="st_verify",
            success=True,
            diff_summary="diff --git a/foo.py b/foo.py\n+new line",
        )

        mock_review = MagicMock()
        mock_review.has_critical = False
        mock_review.findings = [MagicMock(), MagicMock()]
        mock_review.critical_count = 0
        mock_review.agreement_score = 0.85

        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(return_value=mock_review)

        with patch(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            return_value=mock_runner,
        ):
            verification = await bridge.verify_changes(result)

        assert verification["verified"] is True
        assert verification["findings_count"] == 2
        assert verification["critical_count"] == 0
        assert verification["agreement_score"] == 0.85

    @pytest.mark.asyncio
    async def test_verify_changes_no_diff(self):
        """verify_changes() skips when diff_summary is empty."""
        bridge = ExecutionBridge(enable_verification=True)
        result = ExecutionResult(subtask_id="st_no_diff", success=True, diff_summary="")

        verification = await bridge.verify_changes(result)

        assert verification["verified"] is True
        assert verification["skipped"] is True
        assert verification["reason"] == "no diff to verify"

    @pytest.mark.asyncio
    async def test_verify_changes_disabled(self):
        """verify_changes() skips when verification is disabled."""
        bridge = ExecutionBridge(enable_verification=False)
        result = ExecutionResult(
            subtask_id="st_disabled",
            success=True,
            diff_summary="some diff",
        )

        verification = await bridge.verify_changes(result)

        assert verification["verified"] is True
        assert verification["skipped"] is True

    @pytest.mark.asyncio
    async def test_verify_changes_import_error(self):
        """verify_changes() handles missing PRReviewRunner gracefully."""
        bridge = ExecutionBridge(enable_verification=True)
        result = ExecutionResult(
            subtask_id="st_no_runner",
            success=True,
            diff_summary="some diff content",
        )

        with patch.dict(
            "sys.modules",
            {"aragora.compat.openclaw.pr_review_runner": None},
        ):
            verification = await bridge.verify_changes(result)

        assert verification["verified"] is True
        assert verification["skipped"] is True
        assert verification["reason"] == "reviewer unavailable"

    @pytest.mark.asyncio
    async def test_verify_changes_runtime_error(self):
        """verify_changes() handles runner exceptions."""
        bridge = ExecutionBridge(enable_verification=True)
        result = ExecutionResult(
            subtask_id="st_err",
            success=True,
            diff_summary="some diff",
        )

        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(side_effect=RuntimeError("review failed"))

        with patch(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            return_value=mock_runner,
        ):
            verification = await bridge.verify_changes(result)

        assert verification["verified"] is False
        assert "review failed" in verification["error"]

    def test_get_session_summary_empty(self):
        """get_session_summary() returns zeros when no results recorded."""
        bridge = ExecutionBridge()

        summary = bridge.get_session_summary()

        assert summary["total_subtasks"] == 0
        assert summary["successful"] == 0
        assert summary["failed"] == 0
        assert summary["total_files_changed"] == 0
        assert summary["total_tests_run"] == 0
        assert summary["results"] == []

    def test_get_session_summary_mixed(self):
        """get_session_summary() correctly aggregates mixed results."""
        bridge = ExecutionBridge(enable_km_ingestion=False)

        bridge.ingest_result(
            ExecutionResult(
                subtask_id="s1",
                success=True,
                files_changed=["a.py", "b.py"],
                tests_passed=8,
                tests_failed=0,
            )
        )
        bridge.ingest_result(
            ExecutionResult(
                subtask_id="s2",
                success=False,
                files_changed=["c.py"],
                tests_passed=3,
                tests_failed=2,
                error="Test failure",
            )
        )
        bridge.ingest_result(
            ExecutionResult(
                subtask_id="s3",
                success=True,
                files_changed=[],
                tests_passed=5,
                tests_failed=0,
            )
        )

        summary = bridge.get_session_summary()

        assert summary["total_subtasks"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["total_files_changed"] == 3  # 2 + 1 + 0
        assert summary["total_tests_run"] == 18  # (8+0) + (3+2) + (5+0)
        assert len(summary["results"]) == 3

    def test_create_batch_instructions(self):
        """create_batch_instructions() creates one instruction per subtask."""
        bridge = ExecutionBridge()
        subtasks = [
            _FakeSubTask(id="st_a", description="Task A"),
            _FakeSubTask(id="st_b", description="Task B"),
            _FakeSubTask(id="st_c", description="Task C"),
        ]
        goal = _FakeGoal()

        instructions = bridge.create_batch_instructions(
            subtasks, goal=goal, debate_context="Batch run"
        )

        assert len(instructions) == 3
        assert instructions[0].subtask_id == "st_a"
        assert instructions[1].subtask_id == "st_b"
        assert instructions[2].subtask_id == "st_c"
        for inst in instructions:
            assert "Improve test coverage" in inst.context
            assert "Batch run" in inst.context


# ===========================================================================
# Integration: full instruction -> execution -> ingestion -> verification
# ===========================================================================


class TestExecutionBridgeIntegration:
    """Integration test: create instruction, simulate execution, ingest, verify."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """End-to-end: SubTask -> instruction -> result -> ingestion -> verification."""
        bridge = ExecutionBridge(
            enable_km_ingestion=False,  # skip real KM
            enable_verification=True,
        )

        # Step 1: Create instruction
        subtask = _FakeSubTask(
            id="integration_1",
            description="Add input validation to API handler",
            file_scope=["handlers/api.py", "handlers/validation.py"],
            success_criteria={"test_pass_rate": ">0.99"},
        )
        goal = _FakeGoal(
            description="Harden API security",
            rationale="Penetration test revealed input injection risks",
            estimated_impact="high",
        )

        instruction = bridge.create_instruction(
            subtask,
            goal=goal,
            debate_context="Security review debate recommended immediate action.",
        )

        # Verify instruction is well-formed
        assert instruction.subtask_id == "integration_1"
        prompt = instruction.to_agent_prompt()
        assert "Add input validation" in prompt
        assert "Harden API security" in prompt
        assert "`handlers/api.py`" in prompt

        # Step 2: Simulate execution result
        result = ExecutionResult(
            subtask_id=instruction.subtask_id,
            success=True,
            files_changed=["handlers/api.py", "handlers/validation.py"],
            tests_passed=15,
            tests_failed=0,
            duration_seconds=45.2,
            tokens_used=3200,
            diff_summary="diff --git a/handlers/api.py\n+    validate_input(data)",
            agent_observations=["Added Pydantic model for request body"],
        )

        # Step 3: Ingest result (KM disabled, just stores locally)
        bridge.ingest_result(result)

        # Step 4: Verify changes
        mock_review = MagicMock()
        mock_review.has_critical = False
        mock_review.findings = []
        mock_review.critical_count = 0
        mock_review.agreement_score = 0.92

        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(return_value=mock_review)

        with patch(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            return_value=mock_runner,
        ):
            verification = await bridge.verify_changes(result)

        assert verification["verified"] is True
        assert verification["findings_count"] == 0
        assert verification["agreement_score"] == 0.92

        # Step 5: Check session summary
        summary = bridge.get_session_summary()
        assert summary["total_subtasks"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0
        assert summary["total_files_changed"] == 2
        assert summary["total_tests_run"] == 15


class TestExportsFromPackage:
    """Verify the execution bridge classes are accessible from aragora.nomic."""

    def test_import_execution_bridge(self):
        """ExecutionBridge is importable from aragora.nomic."""
        from aragora.nomic import ExecutionBridge as EB

        assert EB is not None

    def test_import_execution_instruction(self):
        """ExecutionInstruction is importable from aragora.nomic."""
        from aragora.nomic import ExecutionInstruction as EI

        assert EI is not None

    def test_import_execution_result(self):
        """ExecutionResult is importable from aragora.nomic."""
        from aragora.nomic import ExecutionResult as ER

        assert ER is not None
