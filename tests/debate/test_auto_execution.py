"""Tests for auto-execution of debate results via the Decision Pipeline.

Covers:
1. AutoExecutionConfig defaults and field values
2. Config integration with ArenaConfig sub-config delegation
3. _auto_execute_plan function - plan creation, metadata storage, execution
4. Graceful failure on import/executor errors
5. Approval mode and risk level mapping
6. Config propagation through Arena constructor
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import DebateResult
from aragora.debate.arena_sub_configs import AutoExecutionConfig


# ===========================================================================
# AutoExecutionConfig Defaults
# ===========================================================================


class TestAutoExecutionConfigDefaults:
    """Test that AutoExecutionConfig has correct defaults."""

    def test_defaults(self):
        cfg = AutoExecutionConfig()
        assert cfg.enable_auto_execution is False
        assert cfg.auto_execution_mode == "workflow"
        assert cfg.auto_approval_mode == "risk_based"
        assert cfg.auto_max_risk == "low"

    def test_is_dataclass(self):
        field_names = {f.name for f in dataclass_fields(AutoExecutionConfig)}
        assert "enable_auto_execution" in field_names
        assert "auto_execution_mode" in field_names
        assert "auto_approval_mode" in field_names
        assert "auto_max_risk" in field_names

    def test_enable_override(self):
        cfg = AutoExecutionConfig(enable_auto_execution=True)
        assert cfg.enable_auto_execution is True

    def test_custom_mode(self):
        cfg = AutoExecutionConfig(auto_execution_mode="hybrid")
        assert cfg.auto_execution_mode == "hybrid"

    def test_custom_approval_mode(self):
        cfg = AutoExecutionConfig(auto_approval_mode="never")
        assert cfg.auto_approval_mode == "never"

    def test_custom_max_risk(self):
        cfg = AutoExecutionConfig(auto_max_risk="high")
        assert cfg.auto_max_risk == "high"


# ===========================================================================
# ArenaConfig Integration
# ===========================================================================


class TestAutoExecutionArenaConfig:
    """Test AutoExecutionConfig integration with ArenaConfig."""

    def test_arena_config_has_auto_execution_config(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert hasattr(config, "auto_execution_config")
        assert isinstance(config.auto_execution_config, AutoExecutionConfig)

    def test_arena_config_delegation(self):
        """Test that ArenaConfig delegates auto-execution fields to sub-config."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_auto_execution=True, auto_max_risk="medium")
        assert config.enable_auto_execution is True
        assert config.auto_max_risk == "medium"
        assert config.auto_execution_config.enable_auto_execution is True
        assert config.auto_execution_config.auto_max_risk == "medium"

    def test_arena_config_default_disabled(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_auto_execution is False

    def test_arena_config_builder(self):
        from aragora.debate.arena_config import ArenaConfig

        config = (
            ArenaConfig.builder()
            .with_auto_execution(enable_auto_execution=True, auto_execution_mode="fabric")
            .build()
        )
        assert config.enable_auto_execution is True
        assert config.auto_execution_mode == "fabric"


# ===========================================================================
# _auto_execute_plan Function
# ===========================================================================


def _make_result(**overrides) -> DebateResult:
    """Create a minimal DebateResult for testing."""
    defaults = {
        "task": "Design a rate limiter",
        "debate_id": "test-debate-123",
        "final_answer": "Use token bucket algorithm",
        "confidence": 0.9,
        "consensus_reached": True,
        "metadata": {},
    }
    defaults.update(overrides)
    return DebateResult(**defaults)


def _make_arena(**overrides) -> MagicMock:
    """Create a mock Arena with auto-execution config attributes."""
    arena = MagicMock()
    arena.enable_auto_execution = overrides.get("enable_auto_execution", True)
    arena.auto_execution_mode = overrides.get("auto_execution_mode", "workflow")
    arena.auto_approval_mode = overrides.get("auto_approval_mode", "risk_based")
    arena.auto_max_risk = overrides.get("auto_max_risk", "low")
    return arena


class TestAutoExecutePlan:
    """Tests for the _auto_execute_plan function."""

    @pytest.mark.asyncio
    async def test_creates_plan_from_result(self):
        """Plan is created and its ID is stored in result metadata."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-abc"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = True  # Don't execute

        with patch(
            "aragora.debate.orchestrator_runner.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            # Patch imports inside the function
            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(
                            RISK_BASED="risk_based",
                            ALWAYS="always",
                            CONFIDENCE_BASED="confidence_based",
                            NEVER="never",
                        ),
                    ),
                    "aragora.pipeline.executor": MagicMock(),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(
                            LOW="low",
                            MEDIUM="medium",
                            HIGH="high",
                            CRITICAL="critical",
                        ),
                    ),
                },
            ):
                arena = _make_arena()
                result = _make_result()

                updated = await _auto_execute_plan(arena, result)

                assert updated.metadata["decision_plan_id"] == "plan-abc"
                assert updated.metadata["decision_plan_status"] == "approved"

    @pytest.mark.asyncio
    async def test_plan_stored_in_metadata(self):
        """Plan ID and status are stored in result.metadata."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-xyz"
        mock_plan.status = MagicMock(value="awaiting_approval")
        mock_plan.requires_human_approval = True

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(
                            RISK_BASED="risk_based",
                        ),
                    ),
                    "aragora.pipeline.executor": MagicMock(),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena()
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                assert "decision_plan_id" in updated.metadata
                assert "decision_plan_status" in updated.metadata
                assert updated.metadata["decision_plan_id"] == "plan-xyz"

    @pytest.mark.asyncio
    async def test_execution_skipped_when_approval_required(self):
        """Executor is NOT called when plan requires human approval."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-needs-approval"
        mock_plan.status = MagicMock(value="awaiting_approval")
        mock_plan.requires_human_approval = True

        mock_executor_cls = MagicMock()

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(RISK_BASED="risk_based"),
                    ),
                    "aragora.pipeline.executor": MagicMock(
                        PlanExecutor=mock_executor_cls,
                    ),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena()
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                # Executor should NOT have been called
                mock_executor_cls.return_value.execute.assert_not_called()
                # plan_outcome should NOT be in metadata
                assert "plan_outcome" not in updated.metadata

    @pytest.mark.asyncio
    async def test_execution_runs_when_auto_approved(self):
        """Executor IS called when plan does not require human approval."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-auto"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = False

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.tasks_completed = 3
        mock_outcome.tasks_total = 3

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_outcome)

        mock_executor_cls = MagicMock(return_value=mock_executor)

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(RISK_BASED="risk_based"),
                    ),
                    "aragora.pipeline.executor": MagicMock(
                        PlanExecutor=mock_executor_cls,
                    ),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena()
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                mock_executor.execute.assert_awaited_once_with(mock_plan)
                assert updated.metadata["plan_outcome"]["success"] is True
                assert updated.metadata["plan_outcome"]["tasks_completed"] == 3
                assert updated.metadata["plan_outcome"]["tasks_total"] == 3

    @pytest.mark.asyncio
    async def test_graceful_failure_on_import_error(self):
        """ImportError is caught and recorded in metadata."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        with patch.dict("sys.modules", {"aragora.pipeline.decision_plan": None}):
            arena = _make_arena()
            result = _make_result()
            updated = await _auto_execute_plan(arena, result)

            assert updated.metadata.get("auto_execution_error") == "ImportError"

    @pytest.mark.asyncio
    async def test_graceful_failure_on_executor_error(self):
        """RuntimeError during execution is caught and recorded."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-fail"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = False

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("Connection lost"))

        mock_executor_cls = MagicMock(return_value=mock_executor)

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(RISK_BASED="risk_based"),
                    ),
                    "aragora.pipeline.executor": MagicMock(
                        PlanExecutor=mock_executor_cls,
                    ),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena()
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                assert updated.metadata["auto_execution_error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_result_always_returned(self):
        """Result is always returned even on failure."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        with patch.dict("sys.modules", {"aragora.pipeline.decision_plan": None}):
            arena = _make_arena()
            result = _make_result()
            updated = await _auto_execute_plan(arena, result)

            assert updated is result
            assert updated.task == "Design a rate limiter"

    @pytest.mark.asyncio
    async def test_metadata_initialized_if_missing(self):
        """If result.metadata is not a dict, it gets initialized."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        with patch.dict("sys.modules", {"aragora.pipeline.decision_plan": None}):
            arena = _make_arena()
            result = _make_result()
            result.metadata = None  # type: ignore[assignment]
            updated = await _auto_execute_plan(arena, result)

            assert isinstance(updated.metadata, dict)
            assert "auto_execution_error" in updated.metadata


# ===========================================================================
# Approval Mode Mapping
# ===========================================================================


class TestApprovalModeMapping:
    """Test that approval mode strings map correctly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode_str",
        ["always", "risk_based", "confidence_based", "never"],
    )
    async def test_approval_modes(self, mode_str):
        """All valid approval mode strings are handled."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-mode"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = True

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(
                            ALWAYS="always",
                            RISK_BASED="risk_based",
                            CONFIDENCE_BASED="confidence_based",
                            NEVER="never",
                        ),
                    ),
                    "aragora.pipeline.executor": MagicMock(),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena(auto_approval_mode=mode_str)
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                assert "decision_plan_id" in updated.metadata


# ===========================================================================
# Risk Level Mapping
# ===========================================================================


class TestRiskLevelMapping:
    """Test that risk level strings map correctly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "risk_str",
        ["low", "medium", "high", "critical"],
    )
    async def test_risk_levels(self, risk_str):
        """All valid risk level strings are handled."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-risk"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = True

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(RISK_BASED="risk_based"),
                    ),
                    "aragora.pipeline.executor": MagicMock(),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(
                            LOW="low",
                            MEDIUM="medium",
                            HIGH="high",
                            CRITICAL="critical",
                        ),
                    ),
                },
            ):
                arena = _make_arena(auto_max_risk=risk_str)
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                assert "decision_plan_id" in updated.metadata

    @pytest.mark.asyncio
    async def test_unknown_risk_falls_back(self):
        """Unknown risk level falls back to RiskLevel.LOW."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-unknown-risk"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = True

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(RISK_BASED="risk_based"),
                    ),
                    "aragora.pipeline.executor": MagicMock(),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena(auto_max_risk="unknown_level")
                result = _make_result()
                updated = await _auto_execute_plan(arena, result)

                # Should still succeed using the fallback
                assert "decision_plan_id" in updated.metadata


# ===========================================================================
# Execution Mode Passthrough
# ===========================================================================


class TestExecutionModePassthrough:
    """Test that execution mode is passed to PlanExecutor."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["workflow", "hybrid", "fabric"])
    async def test_execution_mode_passed(self, mode):
        """Execution mode from config is passed to PlanExecutor constructor."""
        from aragora.debate.orchestrator_runner import _auto_execute_plan

        mock_plan = MagicMock()
        mock_plan.id = "plan-mode-pass"
        mock_plan.status = MagicMock(value="approved")
        mock_plan.requires_human_approval = False

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.tasks_completed = 1
        mock_outcome.tasks_total = 1

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_outcome)

        mock_executor_cls = MagicMock(return_value=mock_executor)

        with patch(
            "aragora.pipeline.decision_plan.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            with patch.dict(
                "sys.modules",
                {
                    "aragora.pipeline.decision_plan": MagicMock(
                        DecisionPlanFactory=mock_factory_cls,
                    ),
                    "aragora.pipeline.decision_plan.core": MagicMock(
                        ApprovalMode=MagicMock(NEVER="never"),
                    ),
                    "aragora.pipeline.executor": MagicMock(
                        PlanExecutor=mock_executor_cls,
                    ),
                    "aragora.pipeline.risk_register": MagicMock(
                        RiskLevel=MagicMock(LOW="low"),
                    ),
                },
            ):
                arena = _make_arena(
                    auto_execution_mode=mode,
                    auto_approval_mode="never",
                )
                result = _make_result()
                await _auto_execute_plan(arena, result)

                mock_executor_cls.assert_called_once_with(execution_mode=mode)


# ===========================================================================
# cleanup_debate_resources Integration
# ===========================================================================


class TestCleanupDebateResourcesIntegration:
    """Test that auto-execution hook is called in cleanup_debate_resources."""

    @pytest.mark.asyncio
    async def test_auto_execute_called_when_enabled(self):
        """_auto_execute_plan is called when enable_auto_execution is True."""
        from aragora.debate.orchestrator_runner import cleanup_debate_resources

        mock_result = _make_result()

        arena = MagicMock()
        arena.enable_auto_execution = True
        arena.auto_execution_mode = "workflow"
        arena.auto_approval_mode = "risk_based"
        arena.auto_max_risk = "low"
        arena.protocol = MagicMock(
            checkpoint_cleanup_on_success=False,
            enable_translation=False,
        )
        arena._cleanup_convergence_cache = MagicMock()
        arena._teardown_agent_channels = AsyncMock()
        arena.cleanup_checkpoints = AsyncMock()
        arena._translate_conclusions = AsyncMock()

        state = MagicMock()
        state.debate_status = "completed"
        state.debate_id = "test-debate"
        state.ctx.finalize_result.return_value = mock_result

        with patch(
            "aragora.debate.orchestrator_runner._auto_execute_plan",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_auto:
            result = await cleanup_debate_resources(arena, state)
            mock_auto.assert_awaited_once_with(arena, mock_result)

    @pytest.mark.asyncio
    async def test_auto_execute_not_called_when_disabled(self):
        """_auto_execute_plan is NOT called when enable_auto_execution is False."""
        from aragora.debate.orchestrator_runner import cleanup_debate_resources

        mock_result = _make_result()

        arena = MagicMock()
        arena.enable_auto_execution = False
        arena.protocol = MagicMock(
            checkpoint_cleanup_on_success=False,
            enable_translation=False,
        )
        arena._cleanup_convergence_cache = MagicMock()
        arena._teardown_agent_channels = AsyncMock()

        state = MagicMock()
        state.debate_status = "completed"
        state.debate_id = "test-debate"
        state.ctx.finalize_result.return_value = mock_result

        with patch(
            "aragora.debate.orchestrator_runner._auto_execute_plan",
            new_callable=AsyncMock,
        ) as mock_auto:
            await cleanup_debate_resources(arena, state)
            mock_auto.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_execute_not_called_when_result_is_none(self):
        """_auto_execute_plan is NOT called when result is None."""
        from aragora.debate.orchestrator_runner import cleanup_debate_resources

        arena = MagicMock()
        arena.enable_auto_execution = True
        arena.protocol = MagicMock(
            checkpoint_cleanup_on_success=False,
            enable_translation=False,
        )
        arena._cleanup_convergence_cache = MagicMock()
        arena._teardown_agent_channels = AsyncMock()

        state = MagicMock()
        state.debate_status = "completed"
        state.debate_id = "test-debate"
        state.ctx.finalize_result.return_value = None

        with patch(
            "aragora.debate.orchestrator_runner._auto_execute_plan",
            new_callable=AsyncMock,
        ) as mock_auto:
            await cleanup_debate_resources(arena, state)
            mock_auto.assert_not_called()
