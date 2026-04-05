import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.debate_rounds import DebateRoundsPhase
from tests.debate.phases.test_debate_rounds import (
    MockAgent,
    MockCritique,
    MockDebateContext,
    MockProtocol,
    MockResult,
)


@pytest.mark.asyncio
async def test_debate_round_timeout_raises_and_cleans_up_slow_critique():
    protocol = MockProtocol(rounds=1)
    protocol.round_timeout_seconds = 0.02
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def slow_critique(critic, proposal, task, context, target_agent=None):
        started.set()
        try:
            await asyncio.sleep(0.1)
            return MockCritique(agent=critic.name, target_agent=target_agent or "unknown")
        finally:
            cleaned_up.set()

    phase = DebateRoundsPhase(protocol=protocol, critique_with_agent=slow_critique)
    proposer = MockAgent(name="proposer", role="proposer")
    critic = MockAgent(name="critic", role="critic")
    ctx = MockDebateContext(
        agents=[proposer, critic],
        proposers=[proposer],
        proposals={"proposer": "initial proposal"},
        result=MockResult(critiques=[]),
    )

    perf_monitor = SimpleNamespace(
        track_round=lambda *args, **kwargs: nullcontext(),
        track_phase=lambda *args, **kwargs: nullcontext(),
        slow_round_threshold=60.0,
    )
    governor = MagicMock()
    governor.get_scaled_timeout.return_value = 30.0

    with (
        patch("aragora.debate.phases.debate_rounds.get_debate_monitor", return_value=perf_monitor),
        patch(
            "aragora.debate.phases.debate_rounds.get_complexity_governor",
            return_value=governor,
        ),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(phase.execute(ctx), timeout=protocol.round_timeout_seconds)

    assert started.is_set()
    await asyncio.wait_for(cleaned_up.wait(), timeout=0.2)
    assert ctx.result.rounds_used == 0
    assert ctx.result.critiques == []
