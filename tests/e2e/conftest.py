"""
E2E test fixtures for Aragora.

Provides fixtures for full end-to-end testing of the complete
debate lifecycle including:
- Debate creation via API
- Debate execution with real-ish agents
- Memory persistence
- ELO updates
- Archival
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def use_jaccard_similarity():
    """Use simple Jaccard similarity backend to avoid sentence-transformer mutex issues.

    The sentence-transformers library can cause mutex deadlocks when
    multiple tests run concurrently. Using Jaccard backend avoids this
    while still testing vote grouping logic.
    """
    old_value = os.environ.get("ARAGORA_SIMILARITY_BACKEND")
    os.environ["ARAGORA_SIMILARITY_BACKEND"] = "jaccard"
    yield
    if old_value is not None:
        os.environ["ARAGORA_SIMILARITY_BACKEND"] = old_value
    else:
        os.environ.pop("ARAGORA_SIMILARITY_BACKEND", None)


from aragora.core import Agent, Vote, Critique, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol


def pytest_collection_modifyitems(items):
    """Automatically add e2e marker to all tests in this directory."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


class E2EAgent(Agent):
    """
    Agent for E2E testing that simulates realistic debate behavior.

    Unlike MockAgent, this agent provides more realistic responses
    that evolve over rounds, simulating actual debate dynamics.
    """

    def __init__(
        self,
        name: str,
        position: str = "neutral",
        personality: str = "analytical",
        stubbornness: float = 0.5,
    ):
        super().__init__(name, model="e2e-test", role="proposer")
        self.position = position
        self.personality = personality
        self.stubbornness = stubbornness
        self._round = 0
        self._proposals: dict[str, str] = {}

    async def generate(self, prompt: str, context: list | None = None) -> str:
        self._round += 1

        # Generate response based on personality and round
        if self.personality == "analytical":
            return f"[Round {self._round}] After careful analysis, I propose: {self.position}. This approach balances trade-offs effectively."
        elif self.personality == "creative":
            return f"[Round {self._round}] Here's an innovative approach: {self.position}. We should explore unconventional solutions."
        else:
            return f"[Round {self._round}] My position: {self.position}."

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        # Record proposals for voting
        self._proposals[self.name] = proposal

        # Use provided target_agent or fallback to "target"
        target = target_agent or "target"

        # Generate critique based on stubbornness
        if self.stubbornness > 0.7:
            return Critique(
                agent=self.name,
                target_agent=target,
                target_content=proposal[:100],
                issues=["The approach has fundamental flaws"],
                suggestions=["Consider my alternative"],
                severity=0.7,
                reasoning="Strong disagreement",
            )
        else:
            return Critique(
                agent=self.name,
                target_agent=target,
                target_content=proposal[:100],
                issues=["Minor improvements possible"],
                suggestions=["Consider edge cases"],
                severity=0.3,
                reasoning="Generally acceptable",
            )

    async def vote(self, proposals: dict, task: str) -> Vote:
        # Vote based on stubbornness - stubborn agents vote for themselves
        if self.stubbornness > 0.6 and self.name in proposals:
            choice = self.name
            confidence = self.stubbornness
        else:
            # Prefer proposals that align with our position for majority convergence
            position = self.position.lower()
            aligned = [
                agent
                for agent, proposal in proposals.items()
                if position and position in str(proposal).lower()
            ]
            if aligned:
                choice = aligned[0]
            else:
                # Fallback to first non-self proposal
                other_choices = [k for k in proposals.keys() if k != self.name]
                choice = other_choices[0] if other_choices else self.name
            confidence = 0.7

        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Supporting {choice}'s approach",
            confidence=confidence,
            continue_debate=self._round < 3,
        )


@pytest.fixture
def temp_e2e_dir():
    """Create a temporary directory for E2E test data."""
    with tempfile.TemporaryDirectory(prefix="aragora_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def e2e_agents():
    """Create a set of E2E agents with different personalities."""
    return [
        E2EAgent(
            "analyst", position="systematic_approach", personality="analytical", stubbornness=0.3
        ),
        E2EAgent(
            "innovator", position="creative_solution", personality="creative", stubbornness=0.4
        ),
        E2EAgent(
            "pragmatist", position="practical_approach", personality="neutral", stubbornness=0.5
        ),
    ]


@pytest.fixture
def e2e_environment():
    """Create an environment for E2E testing."""
    return Environment(
        task="Design a scalable microservices architecture for a real-time collaboration platform",
        context="The platform needs to support 10,000 concurrent users with sub-100ms latency.",
    )


@pytest.fixture
def e2e_protocol():
    """Create a minimal debate protocol for E2E testing.

    Disables all database/embedding-dependent features to prevent
    SQLite mutex deadlock when multiple tests run concurrently.
    E2E tests focus on debate flow, not individual subsystem features.
    """
    return DebateProtocol(
        rounds=3,
        consensus="majority",
        # Disable all DB/embedding-dependent features to prevent mutex deadlock
        enable_calibration=False,
        convergence_detection=False,
        enable_trickster=False,
        enable_rhetorical_observer=False,
        enable_evolution=False,
        enable_research=False,
        enable_breakpoints=False,
        enable_evidence_weighting=False,
    )


@pytest.fixture
def mock_external_apis():
    """Mock external API calls for E2E tests."""
    with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock) as mock:
        mock.return_value = None
        yield mock
