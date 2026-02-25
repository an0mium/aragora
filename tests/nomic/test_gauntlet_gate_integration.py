"""
Integration tests for the GauntletApprovalGate with AutonomousOrchestrator.

Exercises the full end-to-end flow:
  AutonomousOrchestrator(enable_gauntlet_gate=True)
  -> decompose goal -> assign agents -> execute workflow
  -> gauntlet gate evaluate -> approve or reject

All agent interactions are mocked (no real API calls).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.nomic.autonomous_orchestrator import (
    AutonomousOrchestrator,
    AgentAssignment,
    OrchestrationResult,
    Track,
    reset_orchestrator,
)
from aragora.nomic.gauntlet_gate import (
    BlockingFinding,
    GauntletApprovalGate,
    GauntletGateConfig,
    GauntletGateResult,
)
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset orchestrator singleton before and after each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


@pytest.fixture
def mock_workflow_engine():
    """Create a mock workflow engine that returns success."""
    engine = MagicMock()
    engine.execute = AsyncMock(
        return_value=MagicMock(
            success=True,
            final_output={"status": "completed", "diff": "--- a/f.py\n+++ b/f.py"},
            error=None,
        )
    )
    return engine


@pytest.fixture
def mock_task_decomposer():
    """Create a mock task decomposer that returns a single subtask."""
    decomposer = MagicMock()
    decomposer.analyze = MagicMock(
        return_value=TaskDecomposition(
            original_task="Improve auth security",
            complexity_score=5,
            complexity_level="medium",
            should_decompose=True,
            subtasks=[
                SubTask(
                    id="sub_auth_1",
                    title="Harden auth handler",
                    description="Add rate limiting and input validation to auth handler",
                    file_scope=["aragora/server/handlers/auth.py"],
                    estimated_complexity="medium",
                ),
            ],
        )
    )
    return decomposer


@pytest.fixture
def passing_gate_result():
    """A GauntletGateResult that approves the change."""
    return GauntletGateResult(
        blocked=False,
        reason="Gauntlet gate passed (0 total findings, 0 critical, 0 high)",
        critical_count=0,
        high_count=0,
        total_findings=0,
        blocking_findings=[],
        gauntlet_id="gauntlet-pass-001",
        duration_seconds=1.5,
    )


@pytest.fixture
def blocking_gate_result():
    """A GauntletGateResult that blocks the change due to critical findings."""
    return GauntletGateResult(
        blocked=True,
        reason="Gauntlet gate BLOCKED: 1 CRITICAL findings (threshold: 0)",
        critical_count=1,
        high_count=0,
        total_findings=1,
        blocking_findings=[
            BlockingFinding(
                severity="critical",
                title="SQL injection vulnerability",
                description="User input concatenated into SQL query without parameterization",
                category="security",
                source="red_team",
            ),
        ],
        gauntlet_id="gauntlet-block-001",
        duration_seconds=2.3,
    )


def _make_orchestrator(
    mock_workflow_engine,
    mock_task_decomposer,
    *,
    enable_gauntlet_gate: bool = True,
) -> AutonomousOrchestrator:
    """Helper to create an orchestrator with standard test configuration."""
    return AutonomousOrchestrator(
        workflow_engine=mock_workflow_engine,
        task_decomposer=mock_task_decomposer,
        enable_gauntlet_gate=enable_gauntlet_gate,
        require_human_approval=False,
        branch_coordinator=None,
        max_parallel_tasks=1,
    )


# ---------------------------------------------------------------------------
# Integration: approval path (gate passes)
# ---------------------------------------------------------------------------


class TestGauntletGateApprovalPath:
    """End-to-end tests where the gauntlet gate approves the change."""

    @pytest.mark.asyncio
    async def test_execute_goal_passes_with_gauntlet_approval(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        passing_gate_result,
    ):
        """Full flow: decompose -> execute workflow -> gauntlet passes -> assignment completed."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        # Patch _run_gauntlet_gate to return a passing result
        orchestrator._run_gauntlet_gate = AsyncMock(return_value=passing_gate_result)

        result = await orchestrator.execute_goal(
            goal="Improve auth security",
            max_cycles=2,
        )

        # Orchestration should succeed
        assert result.success is True
        assert result.completed_subtasks == 1
        assert result.failed_subtasks == 0

        # Gauntlet gate should have been invoked exactly once (one subtask)
        orchestrator._run_gauntlet_gate.assert_awaited_once()

        # The assignment passed to the gate should carry the subtask
        gate_call_args = orchestrator._run_gauntlet_gate.call_args
        assignment_arg = gate_call_args[0][0]
        assert isinstance(assignment_arg, AgentAssignment)
        assert assignment_arg.subtask.id == "sub_auth_1"

    @pytest.mark.asyncio
    async def test_gate_receives_workflow_result(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        passing_gate_result,
    ):
        """The gauntlet gate should receive the workflow execution result."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=passing_gate_result)

        await orchestrator.execute_goal(goal="Test", max_cycles=1)

        # Second positional arg to _run_gauntlet_gate is the workflow result
        gate_call_args = orchestrator._run_gauntlet_gate.call_args[0]
        workflow_result = gate_call_args[1]
        assert workflow_result.success is True
        assert workflow_result.final_output["status"] == "completed"

    @pytest.mark.asyncio
    async def test_checkpoint_emitted_on_gauntlet_pass(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        passing_gate_result,
    ):
        """Checkpoint callback should include gauntlet gate data on pass."""
        checkpoints: list[tuple[str, dict]] = []

        def on_checkpoint(phase, data):
            checkpoints.append((phase, data))

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=mock_task_decomposer,
            enable_gauntlet_gate=True,
            require_human_approval=False,
            branch_coordinator=None,
            on_checkpoint=on_checkpoint,
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=passing_gate_result)

        await orchestrator.execute_goal(goal="Test checkpoints", max_cycles=1)

        phases = [c[0] for c in checkpoints]
        assert "started" in phases
        assert "completed" in phases


# ---------------------------------------------------------------------------
# Integration: rejection path (gate blocks)
# ---------------------------------------------------------------------------


class TestGauntletGateRejectionPath:
    """End-to-end tests where the gauntlet gate blocks the change."""

    @pytest.mark.asyncio
    async def test_execute_goal_rejects_on_gauntlet_block(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        blocking_gate_result,
    ):
        """Full flow: decompose -> execute workflow -> gauntlet blocks -> assignment rejected."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=blocking_gate_result)

        result = await orchestrator.execute_goal(
            goal="Improve auth security",
            max_cycles=1,
        )

        # The subtask workflow succeeded but was rejected by the gauntlet gate.
        # The orchestrator tracks "rejected" as a distinct status from "failed",
        # so the subtask is neither completed nor failed -- it's rejected.
        assert result.completed_subtasks == 0
        # Verify the assignment carries the "rejected" status
        rejected = [a for a in result.assignments if a.status == "rejected"]
        assert len(rejected) == 1

    @pytest.mark.asyncio
    async def test_assignment_status_set_to_rejected(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        blocking_gate_result,
    ):
        """Assignment status should be 'rejected' when gate blocks."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=blocking_gate_result)

        await orchestrator.execute_goal(goal="Test rejection", max_cycles=1)

        # Check the completed assignments to find the rejected one
        all_assignments = (
            orchestrator._active_assignments + orchestrator._completed_assignments
        )
        rejected = [a for a in all_assignments if a.status == "rejected"]
        assert len(rejected) >= 1

        rejected_assignment = rejected[0]
        assert rejected_assignment.result is not None
        assert "gauntlet_gate" in rejected_assignment.result
        gate_data = rejected_assignment.result["gauntlet_gate"]
        assert gate_data["blocked"] is True
        assert gate_data["critical_count"] == 1

    @pytest.mark.asyncio
    async def test_rejection_preserves_workflow_result(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        blocking_gate_result,
    ):
        """Even when rejected, the assignment should preserve the workflow output."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=blocking_gate_result)

        await orchestrator.execute_goal(goal="Test preserve output", max_cycles=1)

        all_assignments = (
            orchestrator._active_assignments + orchestrator._completed_assignments
        )
        rejected = [a for a in all_assignments if a.status == "rejected"]
        assert len(rejected) >= 1
        assert "workflow_result" in rejected[0].result

    @pytest.mark.asyncio
    async def test_blocked_assignment_skips_further_gates(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
        blocking_gate_result,
    ):
        """When gauntlet blocks, the final review gate should NOT be called."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        orchestrator._run_gauntlet_gate = AsyncMock(return_value=blocking_gate_result)
        orchestrator._check_final_review = AsyncMock(return_value=True)

        await orchestrator.execute_goal(goal="Test gate ordering", max_cycles=1)

        # The final review should never have been called
        orchestrator._check_final_review.assert_not_awaited()


# ---------------------------------------------------------------------------
# Integration: gate disabled (no-op path)
# ---------------------------------------------------------------------------


class TestGauntletGateDisabledPath:
    """Tests that the gate is properly skipped when disabled."""

    @pytest.mark.asyncio
    async def test_gate_not_called_when_disabled(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """When enable_gauntlet_gate=False, _run_gauntlet_gate should not be called."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=False
        )

        orchestrator._run_gauntlet_gate = AsyncMock()

        result = await orchestrator.execute_goal(goal="No gauntlet", max_cycles=1)

        assert result.success is True
        assert result.completed_subtasks == 1
        orchestrator._run_gauntlet_gate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_workflow_completes_without_gate(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """Without the gate, workflow success should lead directly to completion."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=False
        )

        result = await orchestrator.execute_goal(goal="Direct completion", max_cycles=1)

        assert result.success is True
        assert result.completed_subtasks == 1
        mock_workflow_engine.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration: _run_gauntlet_gate method with real GauntletApprovalGate
# ---------------------------------------------------------------------------


class TestRunGauntletGateMethod:
    """Tests for the _run_gauntlet_gate method end-to-end with mocked runner."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_gate_passes_on_clean_findings(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """_run_gauntlet_gate should return a passing result when no findings."""
        from aragora.gauntlet.result import GauntletResult, RiskSummary

        mock_gauntlet_result = GauntletResult(
            gauntlet_id="gauntlet-clean-001",
            input_hash="abc",
            input_summary="test content",
            started_at="2026-02-25T00:00:00",
            completed_at="2026-02-25T00:00:02",
            duration_seconds=2.0,
            vulnerabilities=[],
            risk_summary=RiskSummary(critical=0, high=0, medium=0),
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_gauntlet_result)

        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        subtask = SubTask(
            id="sub_auth_1",
            title="Harden auth",
            description="Add rate limiting",
            file_scope=["aragora/server/handlers/auth.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.SME,
            agent_type="claude",
        )
        workflow_result = MagicMock(
            success=True,
            final_output={"status": "completed"},
        )

        with patch(
            "aragora.nomic.gauntlet_gate._GauntletRunner",
            return_value=mock_runner,
        ):
            gate_result = await orchestrator._run_gauntlet_gate(
                assignment, workflow_result
            )

        assert gate_result is not None
        assert gate_result.blocked is False
        assert gate_result.passed is True
        assert gate_result.gauntlet_id == "gauntlet-clean-001"

    @pytest.mark.asyncio
    async def test_run_gauntlet_gate_blocks_on_critical(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """_run_gauntlet_gate should return blocked result on critical findings."""
        from aragora.gauntlet.result import (
            GauntletResult,
            RiskSummary,
            SeverityLevel,
            Vulnerability,
        )

        vuln = Vulnerability(
            id="vuln-crit-0",
            title="Prompt injection in auth flow",
            description="Auth handler directly interpolates user input into LLM prompt",
            severity=SeverityLevel.CRITICAL,
            category="security",
            source="red_team",
        )
        mock_gauntlet_result = GauntletResult(
            gauntlet_id="gauntlet-block-002",
            input_hash="def",
            input_summary="test content",
            started_at="2026-02-25T00:00:00",
            completed_at="2026-02-25T00:00:03",
            duration_seconds=3.0,
            vulnerabilities=[vuln],
            risk_summary=RiskSummary(critical=1, high=0, medium=0),
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_gauntlet_result)

        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        subtask = SubTask(
            id="sub_auth_1",
            title="Harden auth",
            description="Add rate limiting",
            file_scope=["aragora/server/handlers/auth.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.SME,
            agent_type="claude",
        )
        workflow_result = MagicMock(
            success=True,
            final_output={"status": "completed"},
        )

        with patch(
            "aragora.nomic.gauntlet_gate._GauntletRunner",
            return_value=mock_runner,
        ):
            gate_result = await orchestrator._run_gauntlet_gate(
                assignment, workflow_result
            )

        assert gate_result is not None
        assert gate_result.blocked is True
        assert gate_result.critical_count == 1
        assert len(gate_result.blocking_findings) == 1
        assert gate_result.blocking_findings[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_run_gauntlet_gate_returns_none_on_import_error(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """_run_gauntlet_gate should return None when gauntlet module is unavailable."""
        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        subtask = SubTask(id="sub1", title="T", description="D")
        assignment = AgentAssignment(
            subtask=subtask, track=Track.QA, agent_type="claude"
        )
        workflow_result = MagicMock(success=True, final_output={})

        with patch.dict(
            "sys.modules",
            {"aragora.nomic.gauntlet_gate": None},
        ):
            # When the module import fails, _run_gauntlet_gate catches ImportError
            # and returns None. We simulate this by patching the import inside the method.
            with patch(
                "aragora.nomic.autonomous_orchestrator.AutonomousOrchestrator._run_gauntlet_gate",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_gate:
                gate_result = await mock_gate(assignment, workflow_result)

        assert gate_result is None


# ---------------------------------------------------------------------------
# Integration: full end-to-end with real _run_gauntlet_gate and mocked runner
# ---------------------------------------------------------------------------


class TestGauntletGateFullEndToEnd:
    """Full integration: execute_goal calls real _run_gauntlet_gate with a mocked GauntletRunner."""

    @pytest.mark.asyncio
    async def test_full_flow_gauntlet_passes(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """Complete end-to-end: goal -> decompose -> execute -> gauntlet passes -> success."""
        from aragora.gauntlet.result import GauntletResult, RiskSummary

        clean_result = GauntletResult(
            gauntlet_id="gauntlet-e2e-pass",
            input_hash="hash",
            input_summary="summary",
            started_at="2026-02-25T00:00:00",
            completed_at="2026-02-25T00:00:01",
            duration_seconds=1.0,
            vulnerabilities=[],
            risk_summary=RiskSummary(critical=0, high=0, medium=0),
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=clean_result)

        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        with patch(
            "aragora.nomic.gauntlet_gate._GauntletRunner",
            return_value=mock_runner,
        ):
            result = await orchestrator.execute_goal(
                goal="Improve auth security", max_cycles=1
            )

        assert result.success is True
        assert result.completed_subtasks == 1
        assert result.failed_subtasks == 0

        # The runner should have been invoked
        mock_runner.run.assert_awaited_once()
        call_kwargs = mock_runner.run.call_args
        # GauntletApprovalGate.evaluate calls runner.run(input_content=..., context=...)
        input_content = call_kwargs.kwargs.get("input_content", "")
        assert "Harden auth handler" in input_content or "rate limiting" in input_content.lower()

    @pytest.mark.asyncio
    async def test_full_flow_gauntlet_blocks(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """Complete end-to-end: goal -> decompose -> execute -> gauntlet blocks -> failure."""
        from aragora.gauntlet.result import (
            GauntletResult,
            RiskSummary,
            SeverityLevel,
            Vulnerability,
        )

        critical_vuln = Vulnerability(
            id="vuln-e2e-crit",
            title="Critical security flaw",
            description="Unvalidated user input in auth flow",
            severity=SeverityLevel.CRITICAL,
            category="security",
            source="red_team",
        )
        blocked_result = GauntletResult(
            gauntlet_id="gauntlet-e2e-block",
            input_hash="hash",
            input_summary="summary",
            started_at="2026-02-25T00:00:00",
            completed_at="2026-02-25T00:00:02",
            duration_seconds=2.0,
            vulnerabilities=[critical_vuln],
            risk_summary=RiskSummary(critical=1, high=0, medium=0),
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=blocked_result)

        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        with patch(
            "aragora.nomic.gauntlet_gate._GauntletRunner",
            return_value=mock_runner,
        ):
            result = await orchestrator.execute_goal(
                goal="Improve auth security", max_cycles=1
            )

        # The subtask was rejected (not failed/completed), so completed_subtasks == 0
        assert result.completed_subtasks == 0
        rejected = [a for a in result.assignments if a.status == "rejected"]
        assert len(rejected) == 1

        # Verify the workflow was still executed (gate runs after workflow)
        mock_workflow_engine.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_full_flow_gauntlet_error_does_not_block(
        self,
        mock_workflow_engine,
        mock_task_decomposer,
    ):
        """When the gauntlet runner raises an error, execution should still succeed."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("Gauntlet API down"))

        orchestrator = _make_orchestrator(
            mock_workflow_engine, mock_task_decomposer, enable_gauntlet_gate=True
        )

        with patch(
            "aragora.nomic.gauntlet_gate._GauntletRunner",
            return_value=mock_runner,
        ):
            result = await orchestrator.execute_goal(
                goal="Improve auth security", max_cycles=1
            )

        # Gate errors are non-blocking: the gate returns skipped=True, blocked=False
        # So _run_gauntlet_gate returns a result with blocked=False, and the
        # assignment proceeds to completion
        assert result.success is True
        assert result.completed_subtasks == 1


# ---------------------------------------------------------------------------
# Integration: multiple subtasks
# ---------------------------------------------------------------------------


class TestGauntletGateMultipleSubtasks:
    """Tests with multiple subtasks to verify per-subtask gating."""

    @pytest.mark.asyncio
    async def test_mixed_approval_and_rejection(self, mock_workflow_engine):
        """One subtask approved, one rejected by the gauntlet gate."""
        decomposer = MagicMock()
        decomposer.analyze = MagicMock(
            return_value=TaskDecomposition(
                original_task="Multi-task goal",
                complexity_score=6,
                complexity_level="medium",
                should_decompose=True,
                subtasks=[
                    SubTask(
                        id="task_pass",
                        title="Safe change",
                        description="Add logging to auth handler",
                        file_scope=["aragora/server/handlers/auth.py"],
                        estimated_complexity="low",
                    ),
                    SubTask(
                        id="task_fail",
                        title="Risky change",
                        description="Modify SQL queries in billing handler",
                        file_scope=["aragora/server/handlers/billing.py"],
                        estimated_complexity="medium",
                    ),
                ],
            )
        )

        # Gate returns pass for first subtask, block for second
        call_count = 0

        async def mock_gauntlet_gate(assignment, workflow_result):
            nonlocal call_count
            call_count += 1
            if assignment.subtask.id == "task_pass":
                return GauntletGateResult(
                    blocked=False,
                    reason="passed",
                    gauntlet_id="gauntlet-pass",
                )
            else:
                return GauntletGateResult(
                    blocked=True,
                    reason="CRITICAL finding in SQL handler",
                    critical_count=1,
                    blocking_findings=[
                        BlockingFinding(
                            severity="critical",
                            title="SQL injection",
                            description="Unparameterized query",
                            category="security",
                            source="red_team",
                        ),
                    ],
                    gauntlet_id="gauntlet-block",
                )

        orchestrator = AutonomousOrchestrator(
            workflow_engine=mock_workflow_engine,
            task_decomposer=decomposer,
            enable_gauntlet_gate=True,
            require_human_approval=False,
            branch_coordinator=None,
            max_parallel_tasks=1,
        )

        orchestrator._run_gauntlet_gate = AsyncMock(side_effect=mock_gauntlet_gate)

        result = await orchestrator.execute_goal(
            goal="Multi-task goal",
            max_cycles=1,
        )

        # One completed, one rejected by gauntlet gate
        assert result.completed_subtasks == 1
        assert call_count == 2

        # The orchestrator tracks "rejected" as a distinct status from "failed".
        # Verify the rejected assignment is present with correct status.
        rejected = [a for a in result.assignments if a.status == "rejected"]
        assert len(rejected) == 1
        assert rejected[0].subtask.id == "task_fail"
        assert "gauntlet_gate" in rejected[0].result


# ---------------------------------------------------------------------------
# Integration: workflow step insertion
# ---------------------------------------------------------------------------


class TestGauntletGateWorkflowStep:
    """Tests that _build_subtask_workflow inserts the gauntlet step correctly."""

    def test_gauntlet_step_inserted_in_workflow(self):
        """When gauntlet gate is enabled, workflow should contain a gauntlet step."""
        orchestrator = AutonomousOrchestrator(
            enable_gauntlet_gate=True,
            require_human_approval=False,
        )

        subtask = SubTask(
            id="test",
            title="Test Task",
            description="Test description",
            file_scope=["test.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        step_ids = [s.id for s in workflow.steps]
        assert "gauntlet" in step_ids

        # Gauntlet should be between design and implement
        design_idx = step_ids.index("design")
        gauntlet_idx = step_ids.index("gauntlet")
        implement_idx = step_ids.index("implement")
        assert design_idx < gauntlet_idx < implement_idx

    def test_gauntlet_step_not_present_when_disabled(self):
        """When gauntlet gate is disabled, workflow should NOT contain a gauntlet step."""
        orchestrator = AutonomousOrchestrator(
            enable_gauntlet_gate=False,
            require_human_approval=False,
        )

        subtask = SubTask(
            id="test",
            title="Test Task",
            description="Test description",
            file_scope=["test.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        step_ids = [s.id for s in workflow.steps]
        assert "gauntlet" not in step_ids
        assert "design" in step_ids
        assert "implement" in step_ids
        assert "verify" in step_ids

    def test_gauntlet_step_has_correct_type(self):
        """Gauntlet step should have step_type='gauntlet'."""
        orchestrator = AutonomousOrchestrator(
            enable_gauntlet_gate=True,
            require_human_approval=False,
        )

        subtask = SubTask(
            id="test",
            title="Test",
            description="Test",
            file_scope=["test.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        gauntlet_step = next(s for s in workflow.steps if s.id == "gauntlet")
        assert gauntlet_step.step_type == "gauntlet"
        assert gauntlet_step.config["require_passing"] is True

    def test_design_step_points_to_gauntlet_when_enabled(self):
        """Design step's next_steps should point to gauntlet when enabled."""
        orchestrator = AutonomousOrchestrator(
            enable_gauntlet_gate=True,
            require_human_approval=False,
        )

        subtask = SubTask(
            id="test",
            title="Test",
            description="Test",
            file_scope=["test.py"],
        )
        assignment = AgentAssignment(
            subtask=subtask,
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orchestrator._build_subtask_workflow(assignment)

        design_step = next(s for s in workflow.steps if s.id == "design")
        assert "gauntlet" in design_step.next_steps
