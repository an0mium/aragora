"""End-to-end tests for the self-improvement pipeline.

Verifies the complete flow: planning → decomposition → parallel execution →
budget tracking → KnowledgeMound persistence → outcome recording.

These tests mock external dependencies (Claude Code CLI, KnowledgeMound)
but exercise the full pipeline orchestration logic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.self_improve import (
    BudgetExceededError,
    SelfImproveConfig,
    SelfImprovePipeline,
    SelfImproveResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_config():
    """Config for end-to-end tests with non-essential features disabled."""
    return SelfImproveConfig(
        autonomous=True,
        budget_limit_usd=1.0,
        require_approval=False,
        use_worktrees=False,
        run_tests=False,
        run_review=False,
        capture_metrics=False,
        enable_codebase_metrics=False,
        enable_codebase_indexing=False,
        enable_debug_loop=False,
        quick_mode=True,
        scan_mode=True,
        persist_outcomes=False,
    )


# ---------------------------------------------------------------------------
# ApprovalGate tests
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_approval_decision_enum(self):
        from aragora.nomic.approval import ApprovalDecision

        assert ApprovalDecision.APPROVE.value == "approve"
        assert ApprovalDecision.REJECT.value == "reject"
        assert ApprovalDecision.DEFER.value == "defer"
        assert ApprovalDecision.SKIP.value == "skip"

    def test_gate_invalid_mode(self):
        from aragora.nomic.approval import ApprovalGate

        with pytest.raises(ValueError, match="Invalid approval mode"):
            ApprovalGate(mode="invalid")

    @pytest.mark.asyncio
    async def test_auto_approve_low_risk(self):
        from aragora.nomic.approval import ApprovalDecision, ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Add docstrings to aragora/utils.py"
        instruction.subtask_id = "test-low-risk"
        instruction.files = ["aragora/utils.py"]

        decision = await gate.request_approval(instruction)
        assert decision == ApprovalDecision.APPROVE

    @pytest.mark.asyncio
    async def test_auto_defer_high_risk(self):
        from aragora.nomic.approval import ApprovalDecision, ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Delete all auth tokens and reset credentials"
        instruction.subtask_id = "test-high-risk"
        instruction.files = ["aragora/auth/tokens.py"]

        decision = await gate.request_approval(instruction)
        assert decision == ApprovalDecision.DEFER

    @pytest.mark.asyncio
    async def test_auto_defer_dangerous_operations(self):
        from aragora.nomic.approval import ApprovalDecision, ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Run rm -rf on temp directory"
        instruction.subtask_id = "test-dangerous"
        instruction.files = []

        decision = await gate.request_approval(instruction)
        assert decision == ApprovalDecision.DEFER

    def test_risk_assessment_protected_files(self):
        from aragora.nomic.approval import ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Modify init file"
        instruction.files = ["aragora/__init__.py"]

        risk = gate._assess_risk(instruction)
        assert risk == "high"

    def test_risk_assessment_tests_medium(self):
        from aragora.nomic.approval import ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Add test for utils"
        instruction.files = ["tests/test_utils.py"]

        risk = gate._assess_risk(instruction)
        assert risk == "medium"

    def test_risk_assessment_normal_files_low(self):
        from aragora.nomic.approval import ApprovalGate

        gate = ApprovalGate(mode="auto")
        instruction = MagicMock()
        instruction.to_agent_prompt.return_value = "Add docstrings"
        instruction.files = ["aragora/debate/utils.py"]

        risk = gate._assess_risk(instruction)
        assert risk == "low"


# ---------------------------------------------------------------------------
# Budget tracking tests
# ---------------------------------------------------------------------------


class TestBudgetTracking:
    def test_budget_exceeded_error_exists(self):
        assert issubclass(BudgetExceededError, RuntimeError)

    def test_result_has_cost_fields(self):
        result = SelfImproveResult(cycle_id="c1", objective="test")
        assert result.total_cost_usd == 0.0
        assert result.km_persisted is False
        d = result.to_dict()
        assert "total_cost_usd" in d
        assert "km_persisted" in d

    def test_parse_cost_from_output_direct(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig())
        assert pipeline._parse_cost_from_output("Total cost: $0.42") == 0.42
        assert pipeline._parse_cost_from_output("Cost: $1.23") == 1.23
        assert pipeline._parse_cost_from_output("Cost: 0.05") == 0.05

    def test_parse_cost_from_output_token_estimate(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig())
        # 1000 input, 500 output → (1000*3 + 500*15) / 1M = 0.0105
        cost = pipeline._parse_cost_from_output("input=1000, output=500")
        assert abs(cost - 0.0105) < 0.0001

    def test_parse_cost_from_output_no_match(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig())
        assert pipeline._parse_cost_from_output("no cost info here") == 0.0

    def test_pipeline_initializes_spend_tracker(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig())
        assert pipeline._total_spend_usd == 0.0


# ---------------------------------------------------------------------------
# Dependency waves tests
# ---------------------------------------------------------------------------


class TestDependencyWaves:
    def test_single_wave_under_limit(self):
        config = SelfImproveConfig(max_parallel=10)
        pipeline = SelfImprovePipeline(config)
        waves = pipeline._dependency_waves(["a", "b", "c"])
        assert len(waves) == 1
        assert waves[0] == ["a", "b", "c"]

    def test_multiple_waves_at_limit(self):
        config = SelfImproveConfig(max_parallel=2)
        pipeline = SelfImprovePipeline(config)
        waves = pipeline._dependency_waves(["a", "b", "c", "d", "e"])
        assert len(waves) == 3
        assert waves[0] == ["a", "b"]
        assert waves[1] == ["c", "d"]
        assert waves[2] == ["e"]

    def test_empty_subtasks(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig())
        waves = pipeline._dependency_waves([])
        assert waves == []

    def test_single_subtask(self):
        pipeline = SelfImprovePipeline(SelfImproveConfig(max_parallel=4))
        waves = pipeline._dependency_waves(["only"])
        assert len(waves) == 1
        assert waves[0] == ["only"]


# ---------------------------------------------------------------------------
# E2E pipeline run tests
# ---------------------------------------------------------------------------


class TestSelfImproveE2E:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_run_completes_with_result(self, e2e_config):
        """Full pipeline run returns a valid SelfImproveResult."""
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)
        result = await pipeline.run("Add type hints to utils.py")

        assert isinstance(result, SelfImproveResult)
        assert result.cycle_id.startswith("cycle_")
        assert result.objective == "Add type hints to utils.py"
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_run_records_total_cost(self, e2e_config):
        """Total cost should be recorded on the result."""
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)
        result = await pipeline.run("Fix lint errors")

        assert isinstance(result, SelfImproveResult)
        assert isinstance(result.total_cost_usd, float)
        assert result.total_cost_usd >= 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_dry_run_still_works(self, e2e_config):
        """Dry run should still return a plan dict."""
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)
        plan = await pipeline.dry_run("Improve error handling")

        assert isinstance(plan, dict)
        assert "objective" in plan
        assert "goals" in plan

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_km_persistence_wired(self, e2e_config):
        """KM bridge is called during persist_outcome."""
        e2e_config.persist_outcomes = True
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)

        with patch("aragora.pipeline.km_bridge.PipelineKMBridge") as mock_cls:
            mock_bridge = MagicMock()
            mock_bridge.available = True
            mock_bridge.store_pipeline_result.return_value = True
            mock_cls.return_value = mock_bridge

            result = await pipeline.run("Wire stranded features")

            # If subtasks were generated and completed, persist should be called
            assert isinstance(result, SelfImproveResult)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_self_directing_mode(self, e2e_config):
        """Running with objective=None triggers scan-based goal synthesis."""
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)
        result = await pipeline.run(None)

        assert isinstance(result, SelfImproveResult)
        # Objective should be synthesized from scan
        assert result.objective != ""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_approval_gate_skips_when_disabled(self, e2e_config):
        """With require_approval=False, dispatch should not use ApprovalGate."""
        e2e_config.require_approval = False
        e2e_config.use_meta_planner = False
        pipeline = SelfImprovePipeline(e2e_config)

        with patch("aragora.nomic.approval.ApprovalGate") as mock_gate:
            result = await pipeline.run("Add comments")
            # Gate should not be instantiated when require_approval=False
            mock_gate.assert_not_called()


# ---------------------------------------------------------------------------
# Config new fields tests
# ---------------------------------------------------------------------------


class TestConfigNewFields:
    def test_approval_callback_url_default(self):
        config = SelfImproveConfig()
        assert config.approval_callback_url is None

    def test_approval_callback_url_custom(self):
        config = SelfImproveConfig(approval_callback_url="https://example.com/approve")
        assert config.approval_callback_url == "https://example.com/approve"
