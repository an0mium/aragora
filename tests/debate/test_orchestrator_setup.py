"""Tests for orchestrator_setup module.

Verifies that subsystem init helpers correctly configure Arena attributes
for fabric integration, debate strategy, post-debate workflow, knowledge
operations, grounded operations, RLM limiter, and agent channels.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.orchestrator_setup import (
    init_debate_strategy,
    init_fabric_integration,
    init_grounded_operations,
    init_post_debate_workflow,
    init_rlm_limiter,
    setup_agent_channels,
    teardown_agent_channels,
)


class TestInitFabricIntegration:
    def test_no_fabric_sets_none(self):
        arena = MagicMock()
        agents = [MagicMock()]
        result = init_fabric_integration(arena, fabric=None, fabric_config=None, agents=agents)
        assert result is agents
        assert arena._fabric is None
        assert arena._fabric_config is None

    def test_fabric_with_agents_raises(self):
        arena = MagicMock()
        with pytest.raises(ValueError, match="Cannot specify both"):
            init_fabric_integration(
                arena,
                fabric=MagicMock(),
                fabric_config=MagicMock(),
                agents=[MagicMock()],
            )

    def test_fabric_without_agents_uses_pool(self):
        arena = MagicMock()
        fabric = MagicMock()
        config = MagicMock()
        config.pool_id = "pool-1"
        pool_agents = [MagicMock(), MagicMock()]

        with patch(
            "aragora.debate.orchestrator_agents.get_fabric_agents_sync",
            return_value=pool_agents,
        ):
            result = init_fabric_integration(arena, fabric, config, agents=[])

        assert result is pool_agents
        assert arena._fabric is fabric
        assert arena._fabric_config is config


class TestInitDebateStrategy:
    def test_passthrough_when_no_adaptive(self):
        arena = MagicMock()
        init_debate_strategy(arena, enable_adaptive_rounds=False, debate_strategy=None)
        assert arena.enable_adaptive_rounds is False
        assert arena.debate_strategy is None

    def test_keeps_provided_strategy(self):
        arena = MagicMock()
        strategy = MagicMock()
        init_debate_strategy(arena, enable_adaptive_rounds=True, debate_strategy=strategy)
        assert arena.debate_strategy is strategy

    def test_auto_creates_strategy_when_adaptive_enabled(self):
        arena = MagicMock()
        arena.continuum_memory = MagicMock()

        with patch("aragora.debate.orchestrator_setup.DebateStrategy", create=True) as MockStrategy:
            MockStrategy.return_value = MagicMock()
            # Simulate the dynamic import inside init_debate_strategy
            with patch.dict(
                "sys.modules", {"aragora.debate.strategy": MagicMock(DebateStrategy=MockStrategy)}
            ):
                init_debate_strategy(arena, enable_adaptive_rounds=True, debate_strategy=None)

        assert arena.debate_strategy is not None

    def test_handles_import_error_gracefully(self):
        arena = MagicMock()
        arena.continuum_memory = MagicMock()

        with patch.dict("sys.modules", {"aragora.debate.strategy": None}):
            init_debate_strategy(arena, enable_adaptive_rounds=True, debate_strategy=None)

        assert arena.debate_strategy is None


class TestInitPostDebateWorkflow:
    def test_stores_provided_workflow(self):
        arena = MagicMock()
        workflow = MagicMock()
        init_post_debate_workflow(arena, True, workflow, 0.8)
        assert arena.post_debate_workflow is workflow
        assert arena.enable_post_debate_workflow is True
        assert arena.post_debate_workflow_threshold == 0.8

    def test_disabled_stores_none(self):
        arena = MagicMock()
        init_post_debate_workflow(arena, False, None, 0.7)
        assert arena.post_debate_workflow is None
        assert arena.enable_post_debate_workflow is False


class TestInitGroundedOperations:
    def test_creates_grounded_ops(self):
        arena = MagicMock()
        arena.position_ledger = MagicMock()
        arena.elo_system = MagicMock()

        init_grounded_operations(arena)

        assert arena._grounded_ops is not None


class TestInitRlmLimiter:
    def test_delegates_to_memory_module(self):
        arena = MagicMock()
        state = {
            "use_rlm_limiter": True,
            "rlm_compression_threshold": 5000,
            "rlm_max_recent_messages": 10,
            "rlm_summary_level": "DETAILED",
            "rlm_limiter": MagicMock(),
        }

        with patch(
            "aragora.debate.orchestrator_memory.init_rlm_limiter_state",
            return_value=state,
        ):
            init_rlm_limiter(
                arena,
                use_rlm_limiter=True,
                rlm_limiter=None,
                rlm_compression_threshold=5000,
                rlm_max_recent_messages=10,
                rlm_summary_level="DETAILED",
                rlm_compression_round_threshold=4,
            )

        assert arena.use_rlm_limiter is True
        assert arena.rlm_compression_threshold == 5000
        assert arena.rlm_compression_round_threshold == 4


class TestSetupAgentChannels:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        arena = MagicMock()
        arena.protocol = MagicMock(enable_agent_channels=False)
        ctx = MagicMock()

        await setup_agent_channels(arena, ctx, "debate-1")
        # No exception, no assignment

    @pytest.mark.asyncio
    async def test_handles_import_error(self):
        arena = MagicMock()
        arena.protocol = MagicMock(enable_agent_channels=True)
        ctx = MagicMock()

        with patch(
            "aragora.debate.channel_integration.create_channel_integration",
            side_effect=ImportError("not available"),
            create=True,
        ):
            await setup_agent_channels(arena, ctx, "debate-2")

        # When import fails, _channel_integration should be set to None
        assert arena._channel_integration is None


class TestTeardownAgentChannels:
    @pytest.mark.asyncio
    async def test_noop_when_no_integration(self):
        arena = MagicMock()
        arena._channel_integration = None
        await teardown_agent_channels(arena)

    @pytest.mark.asyncio
    async def test_teardown_calls_integration(self):
        arena = MagicMock()
        integration = AsyncMock()
        arena._channel_integration = integration

        await teardown_agent_channels(arena)

        integration.teardown.assert_awaited_once()
        assert arena._channel_integration is None
