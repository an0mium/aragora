"""Integration test for truth-scorer vote weighting in consensus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.core_types import DebateResult, Vote
from aragora.debate.context import DebateContext
from aragora.debate.phases.consensus_phase import (
    ConsensusCallbacks,
    ConsensusDependencies,
    ConsensusPhase,
)


@dataclass
class MockAgent:
    """Minimal agent used for deterministic vote collection."""

    name: str
    provider: str = "test"
    model_type: str = "mock"
    timeout: float = 30.0


@dataclass
class MockProtocol:
    """Protocol surface needed by majority consensus for this integration test."""

    consensus: str = "majority"
    rounds: int = 3
    consensus_timeout: float = 5.0
    consensus_threshold: float = 0.6
    min_participation_ratio: float = 0.5
    min_participation_count: int = 2
    user_vote_weight: float = 0.5
    formal_verification_enabled: bool = False
    judge_selection: str = "random"
    enable_judge_deliberation: bool = False
    enable_position_shuffling: bool = False
    position_shuffling_permutations: int = 3
    enable_self_vote_mitigation: bool = False
    enable_verbosity_normalization: bool = False
    verify_claims_during_consensus: bool = False
    enable_evidence_weighting: bool = False
    enable_process_evaluation: bool = False
    enable_epistemic_hygiene: bool = False
    enable_truth_ratio_weighting: bool = True
    truth_ratio_bonus: float = 0.3


@dataclass
class MockEnvironment:
    """Minimal debate environment."""

    task: str = "Choose the most trustworthy migration plan"


def make_context() -> tuple[DebateContext, MockProtocol]:
    """Create a debate context with a tied base vote scenario."""
    protocol = MockProtocol()
    result = DebateResult()
    proposals = {
        "agent_evidence": (
            "According to the 2024 migration report, dual-read support reduced deploy "
            "incidents by 42% across 120 runs."
        ),
        "agent_rhetoric": (
            "Everyone knows this migration is obviously the only sensible option and "
            "it would be catastrophic to hesitate."
        ),
    }
    ctx = DebateContext(
        env=MockEnvironment(),
        agents=[MockAgent("agent_evidence"), MockAgent("agent_rhetoric")],
        proposals=proposals,
        result=result,
        start_time=time.time(),
        debate_id="truth-ratio-integration-2033",
    )
    return ctx, protocol


async def vote_with_agent(agent: MockAgent, proposals: dict[str, str], task: str) -> Vote:
    """Return a deterministic self-vote so truth weighting breaks the tie."""
    del proposals, task
    return Vote(
        agent=agent.name,
        choice=agent.name,
        reasoning=f"{agent.name} votes for its own proposal",
        confidence=1.0,
    )


@pytest.mark.asyncio
async def test_truth_ratio_bonus_influences_majority_consensus_metadata():
    ctx, protocol = make_context()
    phase = ConsensusPhase(
        deps=ConsensusDependencies(protocol=protocol),
        callbacks=ConsensusCallbacks(vote_with_agent=vote_with_agent),
    )

    mock_scorer = MagicMock()
    mock_scorer.score.side_effect = [
        MagicMock(truth_ratio=0.9),
        MagicMock(truth_ratio=0.2),
    ]

    with patch("aragora.debate.truth_scorer.TruthScorer", return_value=mock_scorer):
        await phase._handle_majority_consensus(ctx)

    assert ctx.vote_tally["agent_evidence"] == pytest.approx(1.24)
    assert ctx.vote_tally["agent_rhetoric"] == pytest.approx(1.0)
    assert ctx.result.winner == "agent_evidence"
    assert ctx.result.final_answer == ctx.proposals["agent_evidence"]
    assert ctx.result.consensus_reached is True
    assert ctx.result.metadata["truth_ratio"] == {
        "scores": {"agent_evidence": 0.9, "agent_rhetoric": 0.2},
        "average": pytest.approx(0.55),
    }
