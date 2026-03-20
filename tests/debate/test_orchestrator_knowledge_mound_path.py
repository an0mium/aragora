from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aragora.core import Environment
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class CaptureAgent:
    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.role = "proposer"
        self._response = response
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


@pytest.mark.asyncio
async def test_arena_run_enriches_prompt_and_closes_knowledge_loop():
    node = SimpleNamespace(
        id="km-node-1",
        node_type="fact",
        confidence=0.91,
        content="Rate limiters need tenant-aware burst controls.",
    )
    knowledge_mound = AsyncMock()
    knowledge_mound.workspace_id = "ws-debate"
    knowledge_mound.query_semantic = AsyncMock(return_value=SimpleNamespace(nodes=[node]))
    knowledge_mound.store = AsyncMock(return_value=SimpleNamespace(node_id="stored-node"))

    agent = CaptureAgent("agent-1", "Use a token bucket with tenant-aware burst limits.")
    arena = Arena(
        Environment(task="Design a rate limiter for multi-tenant APIs"),
        [agent],
        DebateProtocol(rounds=1, consensus="none"),
        knowledge_mound=knowledge_mound,
        enable_knowledge_retrieval=True,
        enable_knowledge_ingestion=True,
    )

    result = await arena.run()

    knowledge_mound.query_semantic.assert_awaited_once_with(
        text="Design a rate limiter for multi-tenant APIs",
        limit=5,
        min_confidence=0.5,
    )
    knowledge_mound.store.assert_awaited_once()
    assert "Background evidence from Knowledge Mound" in agent.prompts[0]
    assert "Rate limiters need tenant-aware burst controls." in agent.prompts[0]
    assert result.metadata["knowledge_mound_context_applied"] is True
    assert result.metadata["knowledge_mound_read_hits"] == 1
    assert result.metadata["knowledge_mound_item_ids"] == ["km-node-1"]
