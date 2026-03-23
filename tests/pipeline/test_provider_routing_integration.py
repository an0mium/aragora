"""Test ProviderRouter integration in UnifiedOrchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core_types import Agent, DebateResult
from aragora.debate.service import DebateService
from aragora.pipeline.unified_orchestrator import (
    OrchestratorConfig,
    UnifiedOrchestrator,
)


class RuntimeAgent(Agent):
    async def generate(self, prompt, context=None):
        return f"{self.name}:{prompt}"

    async def critique(self, proposal, task, context=None, target_agent=None):
        return "critique"


@pytest.fixture
def mock_arena_factory():
    result = MagicMock()
    result.final_answer = "Use approach A"
    result.participants = ["agent-claude", "agent-gpt"]
    result.consensus_reached = True
    result.metadata = {}
    return AsyncMock(return_value=result)


@pytest.fixture
def mock_provider_router():
    router = MagicMock()
    router.select_providers_for_debate.return_value = [
        "claude-sonnet-4",
        "gpt-4o",
        "deepseek-r1",
    ]
    return router


@pytest.mark.asyncio
async def test_provider_router_selects_before_debate(mock_arena_factory, mock_provider_router):
    """ProviderRouter selections are passed to arena_factory."""
    orch = UnifiedOrchestrator(
        arena_factory=mock_arena_factory,
        provider_router=mock_provider_router,
    )

    result = await orch.run("Design a rate limiter")

    # Router was called
    mock_provider_router.select_providers_for_debate.assert_called_once()

    # Arena factory received provider hints
    call_kwargs = mock_arena_factory.call_args
    assert call_kwargs is not None
    assert "provider_hints" in (call_kwargs.kwargs or {})
    assert result.debate_result is not None
    assert result.debate_result.metadata["provider_hints"] == [
        "claude-sonnet-4",
        "gpt-4o",
        "deepseek-r1",
    ]
    assert result.debate_result.metadata["provider_names"] == [
        "claude-sonnet-4",
        "gpt-4o",
        "deepseek-r1",
    ]


@pytest.mark.asyncio
async def test_provider_router_records_outcome(mock_arena_factory, mock_provider_router):
    """After debate, outcomes are recorded back to the router."""
    orch = UnifiedOrchestrator(
        arena_factory=mock_arena_factory,
        provider_router=mock_provider_router,
    )

    await orch.run("Design a rate limiter")

    # Outcome is recorded against provider IDs, not agent display names.
    recorded = [call.args[0] for call in mock_provider_router.record_outcome.call_args_list]
    assert recorded == ["claude-sonnet-4", "gpt-4o", "deepseek-r1"]


@pytest.mark.asyncio
async def test_provider_router_does_not_break_factory_without_provider_hints_kwarg(
    mock_provider_router,
):
    """Factories that don't accept provider_hints still run successfully."""

    async def arena_factory(
        prompt: str,
        agents=None,
        rounds: int = 3,
        agent_count: int = 3,
        consensus_threshold: float = 0.6,
    ):
        result = MagicMock()
        result.final_answer = "Use approach A"
        result.participants = ["agent-claude", "agent-gpt"]
        result.consensus_reached = True
        result.metadata = {}
        return result

    orch = UnifiedOrchestrator(
        arena_factory=arena_factory,
        provider_router=mock_provider_router,
    )

    result = await orch.run("Design a rate limiter")

    assert "debate" in result.stages_completed
    assert result.errors == []
    assert result.debate_result is not None
    assert result.debate_result.metadata["provider_names"] == [
        "claude-sonnet-4",
        "gpt-4o",
        "deepseek-r1",
    ]


@pytest.mark.asyncio
async def test_provider_router_does_not_overwrite_existing_provider_metadata(
    mock_provider_router,
):
    """If the debate runtime already stamps provider metadata, preserve it."""

    async def arena_factory(
        prompt: str,
        agents=None,
        rounds: int = 3,
        agent_count: int = 3,
        consensus_threshold: float = 0.6,
        provider_hints=None,
    ):
        result = MagicMock()
        result.final_answer = "Use approach A"
        result.participants = ["agent-claude", "agent-gpt"]
        result.consensus_reached = True
        result.metadata = {
            "provider_names": ["preexisting-provider"],
            "provider_hints": ["preexisting-provider"],
        }
        return result

    orch = UnifiedOrchestrator(
        arena_factory=arena_factory,
        provider_router=mock_provider_router,
    )

    result = await orch.run("Design a rate limiter")

    assert result.debate_result is not None
    assert result.debate_result.metadata["provider_names"] == ["preexisting-provider"]
    assert result.debate_result.metadata["provider_hints"] == ["preexisting-provider"]


@pytest.mark.asyncio
async def test_no_router_no_change(mock_arena_factory):
    """Without a provider_router, debate runs as normal."""
    orch = UnifiedOrchestrator(arena_factory=mock_arena_factory)
    result = await orch.run("Design a rate limiter")

    assert "debate" in result.stages_completed


@pytest.mark.asyncio
async def test_provider_router_exercises_real_debate_service_runtime_selection(
    mock_provider_router,
):
    """The default routed provider list should shape the real Arena roster."""
    mock_provider_router.select_providers_for_debate.return_value = [
        "gpt-4o",
        "claude-sonnet-4",
    ]
    agents = [
        RuntimeAgent(name="anthropic-proposer", model="claude-sonnet-4"),
        RuntimeAgent(name="openai-critic", model="gpt-4o"),
        RuntimeAgent(name="openai-backup", model="gpt-4o-mini"),
    ]
    debate_result = DebateResult(
        debate_id="debate-runtime-1",
        task="Design a rate limiter",
        final_answer="Use approach A",
        consensus_reached=True,
    )

    with patch("aragora.debate.orchestrator.Arena") as mock_arena_cls:
        arena_inst = MagicMock()
        arena_inst.run = AsyncMock(return_value=debate_result)
        mock_arena_cls.return_value = arena_inst

        service = DebateService()
        orch = UnifiedOrchestrator(
            arena_factory=service.run,
            provider_router=mock_provider_router,
        )
        cfg = OrchestratorConfig(
            agent_count=2,
            min_providers=2,
        )
        result = await orch.run("Design a rate limiter", config=cfg, agents=agents)

    selected_agents = mock_arena_cls.call_args[0][1]
    assert [agent.name for agent in selected_agents] == [
        "openai-critic",
        "anthropic-proposer",
    ]
    assert result.debate_result is not None
    assert result.debate_result.metadata["provider_names"] == [
        "gpt-4o",
        "claude-sonnet-4",
    ]
    recorded = [call.args[0] for call in mock_provider_router.record_outcome.call_args_list]
    assert recorded == ["gpt-4o", "claude-sonnet-4"]
