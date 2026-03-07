"""Test ProviderRouter integration in UnifiedOrchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.pipeline.unified_orchestrator import (
    OrchestratorConfig,
    UnifiedOrchestrator,
)


@pytest.fixture
def mock_arena_factory():
    result = MagicMock()
    result.final_answer = "Use approach A"
    result.participants = ["claude-sonnet-4", "gpt-4o"]
    result.consensus = True
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


@pytest.mark.asyncio
async def test_provider_router_records_outcome(mock_arena_factory, mock_provider_router):
    """After debate, outcomes are recorded back to the router."""
    orch = UnifiedOrchestrator(
        arena_factory=mock_arena_factory,
        provider_router=mock_provider_router,
    )

    await orch.run("Design a rate limiter")

    # Outcome was recorded for each participant
    assert mock_provider_router.record_outcome.call_count >= 1


@pytest.mark.asyncio
async def test_no_router_no_change(mock_arena_factory):
    """Without a provider_router, debate runs as normal."""
    orch = UnifiedOrchestrator(arena_factory=mock_arena_factory)
    result = await orch.run("Design a rate limiter")

    assert "debate" in result.stages_completed
