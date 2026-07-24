"""Tests for the debate-home executor factory (P4a queue inversion Q2).

``create_default_executor`` relocated here from ``aragora.queue.worker``
(docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §6.2, §10 Q2): its nested
``execute_debate`` lazily imports ``aragora.agents.base``, ``aragora.core``,
and ``aragora.debate.orchestrator`` to run a debate through the Arena, which
would otherwise pull those domain packages into the infrastructure-layer
``aragora.queue`` package (an illegal upward edge under .importlinter).
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.queue_executor import create_default_executor
from aragora.exceptions import InfrastructureError
from aragora.queue.job import create_debate_job


@pytest.mark.asyncio
async def test_create_default_executor_returns_callable():
    executor = await create_default_executor()
    assert callable(executor)


@pytest.mark.asyncio
async def test_execute_debate_builds_result_from_arena_run():
    executor = await create_default_executor()
    job = create_debate_job(
        question="Should we ship it?", agents=["claude", "openai-api"], rounds=2
    )

    fake_result = MagicMock()
    fake_result.debate_id = "debate-123"
    fake_result.consensus_reached = True
    fake_result.final_answer = "Yes"
    fake_result.confidence = 0.9
    fake_result.rounds_used = 2

    fake_arena = MagicMock()
    fake_arena.run = AsyncMock(return_value=fake_result)

    with (
        patch("aragora.agents.base.create_agent", return_value=MagicMock()) as mock_create_agent,
        patch("aragora.debate.orchestrator.Arena", return_value=fake_arena) as mock_arena_cls,
    ):
        result = await executor(job)

    assert mock_create_agent.call_count == 2
    mock_arena_cls.assert_called_once()
    assert result["debate_id"] == "debate-123"
    assert result["consensus_reached"] is True
    assert result["final_answer"] == "Yes"
    assert result["confidence"] == 0.9
    assert result["rounds_used"] == 2
    assert result["participants"] == ["claude", "openai-api"]


@pytest.mark.asyncio
async def test_execute_debate_skips_agent_types_that_create_agent_rejects():
    executor = await create_default_executor()
    job = create_debate_job(question="Q", agents=["claude", "unknown-agent"], rounds=1)

    fake_arena = MagicMock()
    fake_arena.run = AsyncMock(return_value=MagicMock())

    def _create_agent_side_effect(agent_type):
        return MagicMock() if agent_type == "claude" else None

    with (
        patch("aragora.agents.base.create_agent", side_effect=_create_agent_side_effect),
        patch("aragora.debate.orchestrator.Arena", return_value=fake_arena) as mock_arena_cls,
    ):
        await executor(job)

    _, kwargs = mock_arena_cls.call_args
    assert len(kwargs["agents"]) == 1


@pytest.mark.asyncio
async def test_execute_debate_raises_infrastructure_error_when_debate_stack_missing():
    executor = await create_default_executor()
    job = create_debate_job(question="Q", agents=[], rounds=1)

    with patch.dict(sys.modules, {"aragora.debate.orchestrator": None}):
        with pytest.raises(InfrastructureError):
            await executor(job)
