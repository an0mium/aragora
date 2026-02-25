"""
Smoke test configuration and shared fixtures.

Registers the ``smoke`` marker and provides lightweight fixtures
used by the production smoke test suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core_types import Agent, Critique, Environment, Message


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: production smoke tests for critical paths")


# ---------------------------------------------------------------------------
# Mock agent implementation
# ---------------------------------------------------------------------------


class SmokeTestAgent(Agent):
    """Minimal concrete Agent for smoke tests.

    Returns deterministic responses so assertions are predictable.
    """

    def __init__(self, name: str = "smoke-agent", model: str = "smoke-model") -> None:
        super().__init__(name=name, model=model, role="proposer")

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        return f"Proposal from {self.name}: Use a token bucket algorithm for rate limiting."

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=["Could improve error handling"],
            suggestions=["Add retry logic"],
            severity=3.0,
            reasoning="Standard smoke-test critique",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_env() -> Environment:
    """A simple Environment for smoke tests."""
    return Environment(task="Design a rate limiter for an API gateway")


@pytest.fixture
def smoke_agents() -> list[SmokeTestAgent]:
    """Three mock agents sufficient for a minimal debate."""
    return [
        SmokeTestAgent(name="agent-alpha", model="model-a"),
        SmokeTestAgent(name="agent-beta", model="model-b"),
        SmokeTestAgent(name="agent-gamma", model="model-c"),
    ]


@pytest.fixture
def smoke_protocol():
    """A minimal DebateProtocol with expensive features disabled."""
    from aragora.debate.protocol import DebateProtocol

    return DebateProtocol(
        rounds=1,
        consensus="majority",
        timeout_seconds=0,
        use_structured_phases=False,
        convergence_detection=False,
        early_stopping=False,
        enable_trickster=False,
        enable_rhetorical_observer=False,
        enable_calibration=False,
        enable_evolution=False,
        enable_research=False,
        role_rotation=False,
        role_matching=False,
        enable_breakpoints=False,
        enable_evidence_weighting=False,
        verify_claims_during_consensus=False,
        enable_molecule_tracking=False,
        enable_agent_channels=False,
    )


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for SQLite databases."""
    return tmp_path / "smoke_test.db"
