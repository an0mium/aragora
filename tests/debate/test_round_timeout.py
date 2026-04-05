import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.debate_rounds import DebateRoundsPhase


@pytest.mark.asyncio
async def test_round_timeout_seconds_raises_for_delayed_critique():
    protocol = SimpleNamespace(
        rounds=1,
        round_timeout_seconds=0.01,
        asymmetric_stances=False,
        rotate_stances=False,
        use_structured_phases=False,
    )

    async def slow_critique(critic, proposal, task, messages, target_agent=None):
        await asyncio.sleep(0.05)
        return None

    phase = DebateRoundsPhase(protocol=protocol, critique_with_agent=slow_critique)
    proposer = SimpleNamespace(
        name="proposer", role="proposer", timeout=30.0, provider="test", model_type="test"
    )
    critic = SimpleNamespace(
        name="critic", role="critic", timeout=30.0, provider="test", model_type="test"
    )
    ctx = SimpleNamespace(
        result=SimpleNamespace(messages=[], critiques=[], rounds_used=0, metadata={}),
        env=SimpleNamespace(task="test task"),
        proposals={"proposer": "proposal"},
        context_messages=[],
        agents=[proposer, critic],
        proposers=[proposer],
        debate_id="debate-1",
        hook_manager=None,
        cancellation_token=None,
        budget_check_callback=None,
        loop_id="loop-1",
    )
    ctx.add_message = ctx.context_messages.append

    perf_monitor = MagicMock()
    perf_monitor.track_phase.side_effect = lambda *args, **kwargs: nullcontext()
    perf_monitor.slow_round_threshold = 60.0

    with patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov:
        mock_gov.return_value.get_scaled_timeout.return_value = 30.0

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                phase._execute_round(ctx, perf_monitor, round_num=1, total_rounds=1),
                timeout=protocol.round_timeout_seconds,
            )

        await asyncio.sleep(0.06)
