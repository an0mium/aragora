"""Smoke benchmark for orchestration speed policy.

This test validates that low-contention fast-first routing materially reduces
critique-phase latency while using deterministic in-memory mocks.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from aragora.debate.phases.debate_rounds import DebateRoundsPhase


@dataclass
class _Agent:
    name: str
    role: str = "critic"
    timeout: float = 30.0
    provider: str = "mock"
    model_type: str = "mock"


@dataclass
class _Critique:
    agent: str
    target_agent: str
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    severity: float = 0.0

    def to_prompt(self) -> str:
        return f"{self.agent} -> {self.target_agent}"


@dataclass
class _Result:
    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)


@dataclass
class _Env:
    task: str = "benchmark task"


@dataclass
class _Protocol:
    rounds: int = 1
    fast_first_routing: bool = False
    fast_first_min_round: int = 1
    fast_first_low_contention_agent_threshold: int = 4
    fast_first_max_critics_per_proposal: int = 1
    max_parallel_critiques: int = 8


class _Context:
    def __init__(self):
        self.result = _Result()
        self.env = _Env()
        self.context_messages: list = []
        self.proposals = {"agent-a": "proposal a", "agent-b": "proposal b"}

    def add_message(self, msg) -> None:
        self.context_messages.append(msg)


async def _measure_once(fast_first_enabled: bool) -> float:
    protocol = _Protocol(fast_first_routing=fast_first_enabled)

    async def critique_fn(critic, proposal, task, context, target_agent=None):
        await asyncio.sleep(0.01)
        return _Critique(
            agent=critic.name,
            target_agent=target_agent or "unknown",
            issues=["minor tradeoff"],
            severity=0.05,
        )

    phase = DebateRoundsPhase(
        protocol=protocol,
        critique_with_agent=critique_fn,
        select_critics_for_proposal=lambda _proposal_agent, critics: list(critics),
    )
    ctx = _Context()
    critics = [_Agent(name=f"critic-{i}", timeout=10.0 + i) for i in range(6)]

    with patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov:
        mock_gov.return_value.get_scaled_timeout.return_value = 30.0
        start = time.perf_counter()
        await phase._critique_phase(ctx=ctx, critics=critics, round_num=1)
        return time.perf_counter() - start


@pytest.mark.asyncio
async def test_fast_first_policy_reduces_median_critique_latency():
    """Fast-first policy should reduce median critique latency by >=25%."""
    baseline = [await _measure_once(False) for _ in range(6)]
    fast_first = [await _measure_once(True) for _ in range(6)]

    baseline_median = statistics.median(baseline)
    fast_median = statistics.median(fast_first)

    assert fast_median <= baseline_median * 0.75
