from __future__ import annotations

import importlib
import os

import pytest

from aragora.core import Agent, Critique, Vote


def test_import_path_has_no_secrets_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARAGORA_SKIP_SECRETS_HYDRATION", raising=False)
    package = importlib.import_module("aragora")
    assert package.__name__ == "aragora"
    assert "ARAGORA_SKIP_SECRETS_HYDRATION" not in os.environ


class MockAgent(Agent):
    """Minimal offline agent that returns a fixed answer.

    Inherits from the real Agent ABC so that all Agent contract attributes
    (name, model, role, system_prompt, agent_type, stance, tool_manifest)
    are populated, and so that the abstract methods are explicitly
    implemented with shape-correct return types (Critique, Vote dataclasses).
    """

    def __init__(self, name: str, answer: str) -> None:
        super().__init__(name=name, model="mock", role="proposer")
        self._answer = answer

    async def generate(self, prompt, context=None):
        return self._answer

    async def critique(self, proposal, task, context=None, target_agent=None):
        return Critique(
            agent=self.name,
            target_agent=target_agent or "",
            target_content=proposal,
            issues=[],
            suggestions=[],
            severity=0.0,
            reasoning="offline mock",
        )

    async def vote(self, proposals, task):
        # Vote for the first proposal that matches our answer string, or the
        # first available proposal as a fallback. continue_debate=False so
        # the consensus phase finalizes after round 1.
        choice = next(
            (agent for agent, proposal in proposals.items() if proposal == self._answer),
            next(iter(proposals), self.name),
        )
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="offline mock vote",
            confidence=1.0,
            continue_debate=False,
        )


@pytest.fixture
def mock_agents() -> list[MockAgent]:
    agents: list[MockAgent] = []
    # Use unique names per fixture run so we don't collide with global
    # circuit-breaker state seeded by other tests in the session.
    for name, answer in [
        ("smoke_mock_alpha", "Use a narrow review lane."),
        ("smoke_mock_beta", "Use a narrow review lane."),
    ]:
        agents.append(MockAgent(name, answer))
    return agents


@pytest.fixture(autouse=True)
def _reset_circuit_breakers() -> None:
    """Reset global circuit breakers so prior test failures don't gate this
    debate's proposers (the offline pipeline relies on proposers passing the
    circuit-breaker filter)."""
    from aragora.resilience import reset_all_circuit_breakers

    reset_all_circuit_breakers()


@pytest.mark.asyncio
async def test_minimal_offline_debate_completes(mock_agents: list[MockAgent]) -> None:
    from aragora.core import Environment
    from aragora.debate import Arena, DebateProtocol

    env = Environment(task="How should we ship the debate package?")
    protocol = DebateProtocol(rounds=1, consensus="majority")
    result = await Arena(env, mock_agents, protocol).run()

    # Smoke contract: the offline pipeline must complete without crashing
    # and produce a DebateResult with both agents listed as participants.
    # Asserting deeper invariants (consensus_reached, exact final_answer)
    # would couple this offline test to embedding/synthesis backends that
    # aren't available in the integration environment.
    assert result.status == "completed"
    assert set(result.participants) == {"smoke_mock_alpha", "smoke_mock_beta"}
    # Votes are produced by the MockAgent.vote override and should be
    # collected even if synthesis falls back.
    assert len(result.votes) == 2
    assert {vote.agent for vote in result.votes} == {
        "smoke_mock_alpha",
        "smoke_mock_beta",
    }
