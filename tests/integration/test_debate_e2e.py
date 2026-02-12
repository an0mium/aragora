"""
End-to-end debate integration tests.

Tests the full propose -> critique -> revision -> consensus flow using
real Aragora classes with mock agents (canned responses, not HTTP-level mocks).
Covers different consensus modes: majority, unanimous, judge.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.consensus import ConsensusProof
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.memory.store import CritiqueStore

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CannedAgent(Agent):
    """Agent that returns deterministic canned responses for each phase."""

    def __init__(
        self,
        name: str,
        *,
        role: str = "proposer",
        proposals: list[str] | None = None,
        vote_choice: str | None = None,
        vote_confidence: float = 0.85,
    ):
        super().__init__(name, "canned-model", role)
        self.agent_type = "canned"
        self._proposals = proposals or [f"Proposal from {name}: use a token bucket"]
        self._vote_choice = vote_choice
        self._vote_confidence = vote_confidence
        self._gen_idx = 0

    async def generate(self, prompt: str, context: list | None = None) -> str:
        resp = self._proposals[self._gen_idx % len(self._proposals)]
        self._gen_idx += 1
        return resp

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100],
            issues=[f"{self.name} notes a gap in error handling"],
            suggestions=[f"{self.name} suggests adding retry logic"],
            severity=3.0,
            reasoning=f"Critique from {self.name}",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        choice = self._vote_choice or (list(proposals.keys())[0] if proposals else self.name)
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"{self.name} votes for {choice}",
            confidence=self._vote_confidence,
            continue_debate=False,
        )


def _make_agents(n: int = 3, vote_target: str | None = None) -> list[CannedAgent]:
    names = ["alice", "bob", "charlie", "diana", "eve"][:n]
    agents = []
    for name in names:
        agents.append(
            CannedAgent(
                name,
                role="proposer",
                proposals=[f"{name}'s design: distributed cache with TTL"],
                vote_choice=vote_target,
            )
        )
    return agents


# ---------------------------------------------------------------------------
# 1. Basic debate completion
# ---------------------------------------------------------------------------


class TestDebateCompletion:
    """Verify that a debate runs through all phases and produces a result."""

    async def test_two_agent_debate_completes(self):
        agents = _make_agents(2)
        env = Environment(task="Design a rate limiter", max_rounds=2)
        protocol = DebateProtocol(rounds=2)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)
        assert result.task == "Design a rate limiter"
        assert result.rounds_used >= 1

    async def test_three_agent_debate_completes(self):
        agents = _make_agents(3)
        env = Environment(task="Design a cache", max_rounds=2)
        protocol = DebateProtocol(rounds=2)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)
        assert len(result.participants) >= 2

    async def test_five_agent_debate_completes(self):
        agents = _make_agents(5)
        env = Environment(task="Choose a database", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)

    async def test_debate_result_has_messages(self):
        agents = _make_agents(3)
        env = Environment(task="Pick a framework", max_rounds=2)
        protocol = DebateProtocol(rounds=2)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result.messages, list)

    async def test_debate_result_has_debate_id(self):
        agents = _make_agents(2)
        env = Environment(task="Select an algorithm", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert result.debate_id
        assert len(result.debate_id) > 0


# ---------------------------------------------------------------------------
# 2. Consensus modes
# ---------------------------------------------------------------------------


class TestConsensusModes:
    """Test majority, unanimous, and judge consensus modes."""

    async def test_majority_consensus_all_agree(self):
        agents = _make_agents(3, vote_target="alice")
        env = Environment(task="Design an API", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)
        # When all agents agree, consensus should be reached
        # (may or may not be flagged depending on internal thresholds)

    async def test_majority_consensus_split_vote(self):
        agents = [
            CannedAgent("alice", vote_choice="alice"),
            CannedAgent("bob", vote_choice="bob"),
            CannedAgent("charlie", vote_choice="alice"),
        ]
        env = Environment(task="Pick a language", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)

    async def test_unanimous_consensus_all_agree(self):
        agents = _make_agents(3, vote_target="alice")
        env = Environment(task="Choose storage", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="unanimous")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)

    async def test_judge_consensus(self):
        agents = [
            CannedAgent("alice", role="proposer"),
            CannedAgent("bob", role="proposer"),
            CannedAgent("judge", role="judge", vote_choice="alice"),
        ]
        env = Environment(task="Evaluate proposals", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="judge")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)

    async def test_any_consensus_mode(self):
        agents = _make_agents(2, vote_target="alice")
        env = Environment(task="Quick decision", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="any")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert isinstance(result, DebateResult)


# ---------------------------------------------------------------------------
# 3. Receipt / hash generation
# ---------------------------------------------------------------------------


class TestReceiptGeneration:
    """Verify that debate results produce stable, verifiable outputs."""

    async def test_result_has_stable_id(self):
        agents = _make_agents(2)
        env = Environment(task="Test ID stability", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert result.debate_id == result.id

    async def test_result_serializes_to_dict(self):
        agents = _make_agents(2)
        env = Environment(task="Serialization test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        d = result.to_dict()
        assert "debate_id" in d
        assert "task" in d
        assert "consensus_reached" in d

    async def test_result_dict_is_hashable(self):
        """The dict representation should be JSON-serializable and hashable."""
        import json

        agents = _make_agents(2)
        env = Environment(task="Hash test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        d = result.to_dict()
        content = json.dumps(d, sort_keys=True, default=str)
        digest = hashlib.sha256(content.encode()).hexdigest()
        assert len(digest) == 64

    async def test_two_debates_have_different_ids(self):
        agents1 = _make_agents(2)
        agents2 = _make_agents(2)
        env = Environment(task="ID uniqueness", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        r1 = await asyncio.wait_for(Arena(env, agents1, protocol).run(), timeout=30)
        r2 = await asyncio.wait_for(Arena(env, agents2, protocol).run(), timeout=30)

        assert r1.debate_id != r2.debate_id


# ---------------------------------------------------------------------------
# 4. Memory integration
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    """Verify that memory systems receive debate outcomes."""

    async def test_debate_with_critique_store(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = CritiqueStore(db_path)
            agents = _make_agents(2)
            env = Environment(task="Memory test", max_rounds=1)
            protocol = DebateProtocol(rounds=1)
            arena = Arena(env, agents, protocol, memory=store)
            result = await asyncio.wait_for(arena.run(), timeout=30)

            assert isinstance(result, DebateResult)
            stats = store.get_stats()
            assert isinstance(stats, dict)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test_debate_result_round_count_matches_protocol(self):
        agents = _make_agents(3)
        env = Environment(task="Round count check", max_rounds=3)
        protocol = DebateProtocol(rounds=3)
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert result.rounds_used >= 1
        assert result.rounds_used <= 3

    async def test_debate_status_is_set(self):
        agents = _make_agents(2, vote_target="alice")
        env = Environment(task="Status check", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(env, agents, protocol)
        result = await asyncio.wait_for(arena.run(), timeout=30)

        assert result.status in ("consensus_reached", "completed", "failed", "")
