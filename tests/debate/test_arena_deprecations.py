"""Tests for Arena config deprecation warnings.

Verifies that passing knowledge_*, evolution_*, and ml_* individual kwargs
triggers DeprecationWarning, while using the corresponding config objects
does not.
"""

from __future__ import annotations

import warnings

import pytest

from aragora.core import Agent, Critique, Environment, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class _MockAgent(Agent):
    """Minimal mock agent for deprecation tests."""

    def __init__(self, name: str = "mock-agent"):
        super().__init__(name=name, model="mock-model", role="proposer")

    async def generate(self, prompt: str, context: list = None) -> str:
        return "ok"

    async def generate_stream(self, prompt: str, context: list = None):
        yield "ok"

    async def critique(
        self, proposal: str, task: str, context: list = None, target_agent: str = None
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["issue"],
            suggestions=["suggestion"],
            severity=0.5,
            reasoning="reason",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(agent=self.name, choice=choice, reasoning="vote", confidence=0.8)


@pytest.fixture()
def _arena_deps():
    """Provide minimal dependencies for Arena construction."""
    env = Environment(task="test")
    agents = [_MockAgent(name="a1"), _MockAgent(name="a2"), _MockAgent(name="a3")]
    protocol = DebateProtocol(rounds=1)
    return env, agents, protocol


class TestKnowledgeDeprecation:
    """knowledge_* individual kwargs trigger DeprecationWarning."""

    def test_enable_knowledge_extraction_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, enable_knowledge_extraction=True)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "knowledge" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_extraction_min_confidence_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, extraction_min_confidence=0.9)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "knowledge" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_enable_belief_guidance_false_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, enable_belief_guidance=False)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "knowledge" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_knowledge_config_does_not_warn(self, _arena_deps):
        from aragora.debate.arena_primary_configs import KnowledgeConfig

        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, knowledge_config=KnowledgeConfig())
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "knowledge" in str(x.message).lower()
        ]
        assert len(dep_warnings) == 0

    def test_knowledge_params_still_function(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            arena = Arena(env, agents, protocol, enable_knowledge_extraction=True)
        assert arena.enable_knowledge_extraction is True


class TestEvolutionDeprecation:
    """evolution_* individual kwargs trigger DeprecationWarning."""

    def test_auto_evolve_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, auto_evolve=True)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "evolution" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_breeding_threshold_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, breeding_threshold=0.5)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "evolution" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_evolution_config_does_not_warn(self, _arena_deps):
        from aragora.debate.arena_primary_configs import EvolutionConfig

        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, evolution_config=EvolutionConfig())
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "evolution" in str(x.message).lower()
        ]
        assert len(dep_warnings) == 0

    def test_evolution_params_still_function(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            arena = Arena(env, agents, protocol, breeding_threshold=0.5)
        assert arena.breeding_threshold == 0.5


class TestMLDeprecation:
    """ml_* individual kwargs trigger DeprecationWarning."""

    def test_ml_delegation_disabled_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, enable_ml_delegation=False)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "ml" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_quality_gate_threshold_warns(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, quality_gate_threshold=0.9)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "ml" in str(x.message).lower()
        ]
        assert len(dep_warnings) >= 1

    def test_ml_config_does_not_warn(self, _arena_deps):
        from aragora.debate.arena_primary_configs import MLConfig

        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol, ml_config=MLConfig())
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning) and "ml" in str(x.message).lower()
        ]
        assert len(dep_warnings) == 0

    def test_ml_params_still_function(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            arena = Arena(env, agents, protocol, enable_ml_delegation=False)
        assert arena.enable_ml_delegation is False


class TestDefaultParamsNoWarning:
    """No deprecation warnings when all params are at defaults."""

    def test_no_warnings_with_defaults(self, _arena_deps):
        env, agents, protocol = _arena_deps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Arena(env, agents, protocol)
        dep_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and any(
                kw in str(x.message).lower()
                for kw in ("knowledge", "evolution", "ml params", "individual ml")
            )
        ]
        assert len(dep_warnings) == 0
