"""Integration tests for Supermemory wiring in the debate pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core_types import DebateResult, Environment
from aragora.debate.knowledge_manager import ArenaKnowledgeManager
from aragora.debate.prompt_builder import PromptBuilder
from aragora.debate.protocol import DebateProtocol
from aragora.knowledge.mound.adapters.supermemory_adapter import (
    ContextInjectionResult,
    SyncOutcomeResult,
)


@pytest.mark.asyncio
async def test_supermemory_outcome_sync_once():
    """Supermemory outcome sync should run once per debate_id."""
    adapter = MagicMock()
    adapter.sync_debate_outcome = AsyncMock(
        return_value=SyncOutcomeResult(success=True, memory_id="mem_123")
    )

    manager = ArenaKnowledgeManager(
        enable_supermemory=True,
        supermemory_adapter=adapter,
        supermemory_sync_on_conclusion=True,
    )

    result = DebateResult(task="Test debate", debate_id="debate-123")
    env = Environment(task="Test debate")

    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_called_once()
    assert "debate-123" in manager._supermemory_synced_debate_ids

    # Second ingest should not resync
    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_called_once()


@pytest.mark.asyncio
async def test_prompt_builder_injects_supermemory_context():
    """PromptBuilder should format and cache Supermemory context."""
    adapter = MagicMock()
    adapter.inject_context = AsyncMock(
        return_value=ContextInjectionResult(
            memories_injected=2,
            context_content=["First memory", "Second memory"],
            total_tokens_estimate=123,
            search_time_ms=5,
        )
    )

    env = Environment(task="Test topic")
    protocol = DebateProtocol()
    builder = PromptBuilder(protocol=protocol, env=env, supermemory_adapter=adapter)

    rendered = await builder.inject_supermemory_context()

    assert "External Memory Context" in rendered
    assert "First memory" in rendered
    assert "Second memory" in rendered
    assert builder.get_supermemory_context() == rendered
    adapter.inject_context.assert_called_once()
