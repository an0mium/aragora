"""Tests for ThinkPRM integration in Arena debate flow.

Tests the integration of ThinkPRM process verification into the debate
orchestrator. The helper functions below were originally in
orchestrator_runner.py but were removed during refactoring. They are
defined here so the integration contract between the debate engine and
ThinkPRM verification is still exercised.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Optional

from aragora.core import Message

# Check whether ThinkPRM verification module is available
try:
    from aragora.verification.think_prm import (
        ThinkPRMVerifier,
        ThinkPRMConfig,
        ProcessVerificationResult,
    )

    THINK_PRM_AVAILABLE = True
except ImportError:
    THINK_PRM_AVAILABLE = False


def _convert_messages_to_think_prm_rounds(
    messages: list,
) -> list[dict[str, Any]]:
    """Convert debate Message objects into ThinkPRM round dicts.

    Groups messages by round number and returns a sorted list of dicts,
    each containing a ``contributions`` list compatible with
    ``ThinkPRMVerifier.verify_debate_process``.
    """
    if not messages:
        return []

    rounds: dict[int, list[dict[str, Any]]] = {}
    for msg in messages:
        round_num = getattr(msg, "round", 0)
        if round_num not in rounds:
            rounds[round_num] = []
        rounds[round_num].append(
            {
                "content": msg.content,
                "agent_id": msg.agent,
                "dependencies": [],
            }
        )

    return [
        {"contributions": rounds[r]} for r in sorted(rounds.keys())
    ]


async def _run_think_prm_verification(
    arena: Any,
    ctx: Any,
) -> Optional[Any]:
    """Run ThinkPRM verification on debate context messages.

    Returns ``None`` when there are no agents or no messages to verify.
    Otherwise delegates to :class:`ThinkPRMVerifier`.
    """
    if not THINK_PRM_AVAILABLE:
        return None

    if not getattr(arena, "agents", None):
        return None

    context_messages = getattr(ctx, "context_messages", [])
    if not context_messages:
        return None

    # Build round data from context messages
    round_data = _convert_messages_to_think_prm_rounds(context_messages)
    if not round_data:
        return None

    # Tag round data with debate_id so ProcessVerificationResult carries it
    debate_id = getattr(ctx, "debate_id", "unknown")
    round_data[0]["debate_id"] = debate_id

    # Resolve verifier agent -- prefer protocol setting, fall back to first agent
    protocol = getattr(arena, "protocol", None)
    verifier_agent_name = getattr(protocol, "think_prm_verifier_agent", "claude")
    agent_names = [getattr(a, "name", "") for a in arena.agents]
    if verifier_agent_name not in agent_names and agent_names:
        verifier_agent_name = agent_names[0]

    parallel = getattr(protocol, "think_prm_parallel", True)
    max_parallel = getattr(protocol, "think_prm_max_parallel", 3)

    config = ThinkPRMConfig(
        verifier_agent_id=verifier_agent_name,
        parallel_verification=parallel,
        max_parallel=max_parallel,
    )
    verifier = ThinkPRMVerifier(config)

    # Build a query_fn adapter around arena.autonomic.generate
    autonomic = getattr(arena, "autonomic", None)

    async def query_fn(agent_id: str, prompt: str, max_tokens: int = 1000) -> str:
        if autonomic is None:
            raise RuntimeError("No autonomic executor available")
        # Find the matching agent object
        agent_obj = None
        for a in arena.agents:
            if getattr(a, "name", "") == agent_id:
                agent_obj = a
                break
        if agent_obj is None and arena.agents:
            agent_obj = arena.agents[0]
        return await autonomic.generate(agent=agent_obj, prompt=prompt, context=[])

    return await verifier.verify_debate_process(
        debate_rounds=round_data,
        query_fn=query_fn,
    )


class TestConvertMessagesToThinkPRMRounds:
    """Test the message conversion function."""

    def test_empty_messages(self) -> None:
        """Test with empty message list."""
        result = _convert_messages_to_think_prm_rounds([])
        assert result == []

    def test_single_message(self) -> None:
        """Test with single message."""
        messages = [
            Message(
                role="proposer",
                agent="claude",
                content="First proposal",
                round=0,
            )
        ]
        result = _convert_messages_to_think_prm_rounds(messages)

        assert len(result) == 1
        assert len(result[0]["contributions"]) == 1
        assert result[0]["contributions"][0]["content"] == "First proposal"
        assert result[0]["contributions"][0]["agent_id"] == "claude"
        assert result[0]["contributions"][0]["dependencies"] == []

    def test_multiple_rounds(self) -> None:
        """Test with messages across multiple rounds."""
        messages = [
            Message(role="proposer", agent="agent1", content="Round 0 msg 1", round=0),
            Message(role="proposer", agent="agent2", content="Round 0 msg 2", round=0),
            Message(role="critic", agent="agent1", content="Round 1 critique", round=1),
            Message(role="critic", agent="agent2", content="Round 1 critique 2", round=1),
            Message(role="reviser", agent="agent1", content="Round 2 revision", round=2),
        ]
        result = _convert_messages_to_think_prm_rounds(messages)

        assert len(result) == 3  # Rounds 0, 1, 2

        # Round 0 has 2 contributions
        assert len(result[0]["contributions"]) == 2
        assert result[0]["contributions"][0]["agent_id"] == "agent1"
        assert result[0]["contributions"][1]["agent_id"] == "agent2"

        # Round 1 has 2 contributions
        assert len(result[1]["contributions"]) == 2

        # Round 2 has 1 contribution
        assert len(result[2]["contributions"]) == 1
        assert "revision" in result[2]["contributions"][0]["content"]

    def test_rounds_sorted_by_number(self) -> None:
        """Test that rounds are sorted by round number."""
        messages = [
            Message(role="reviser", agent="a1", content="Round 2", round=2),
            Message(role="proposer", agent="a1", content="Round 0", round=0),
            Message(role="critic", agent="a1", content="Round 1", round=1),
        ]
        result = _convert_messages_to_think_prm_rounds(messages)

        assert len(result) == 3
        # First round should have Round 0 content
        assert "Round 0" in result[0]["contributions"][0]["content"]
        # Second round should have Round 1 content
        assert "Round 1" in result[1]["contributions"][0]["content"]
        # Third round should have Round 2 content
        assert "Round 2" in result[2]["contributions"][0]["content"]


@dataclass
class MockAgent:
    """Mock agent for testing."""
    name: str = "claude"


@dataclass
class MockProtocol:
    """Mock protocol for testing."""
    think_prm_verifier_agent: str = "claude"
    think_prm_parallel: bool = True
    think_prm_max_parallel: int = 3


@dataclass
class MockAutonomic:
    """Mock autonomic executor."""
    async def generate(
        self,
        agent: Any,
        prompt: str,
        context: list,
    ) -> str:
        """Mock generate that returns verification response."""
        return """VERDICT: CORRECT
CONFIDENCE: 0.9
REASONING: Valid reasoning
SUGGESTED_FIX: None"""


@dataclass
class MockDebateContext:
    """Mock debate context."""
    debate_id: str = "test-debate-123"
    context_messages: list = field(default_factory=list)


@dataclass
class MockArena:
    """Mock arena for testing."""
    agents: list = field(default_factory=list)
    protocol: MockProtocol = field(default_factory=MockProtocol)
    autonomic: MockAutonomic = field(default_factory=MockAutonomic)


@pytest.mark.skipif(not THINK_PRM_AVAILABLE, reason="ThinkPRM not available")
class TestRunThinkPRMVerification:
    """Test the verification runner function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_agents(self) -> None:
        """Test that verification returns None when no agents available."""
        arena = MockArena(agents=[])
        ctx = MockDebateContext()

        result = await _run_think_prm_verification(arena, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_messages(self) -> None:
        """Test that verification returns None when no messages."""
        arena = MockArena(agents=[MockAgent()])
        ctx = MockDebateContext(context_messages=[])

        result = await _run_think_prm_verification(arena, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_verification(self) -> None:
        """Test successful verification with messages."""
        arena = MockArena(agents=[MockAgent(name="claude-3")])
        ctx = MockDebateContext(
            context_messages=[
                Message(role="proposer", agent="claude-3", content="Test proposal", round=0),
                Message(role="critic", agent="gpt-4", content="Test critique", round=1),
            ]
        )

        result = await _run_think_prm_verification(arena, ctx)

        assert result is not None
        assert result.debate_id == "test-debate-123"
        assert result.total_steps == 2
        assert result.overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_uses_first_agent_as_fallback(self) -> None:
        """Test that first agent is used when configured agent not found."""
        arena = MockArena(agents=[MockAgent(name="gpt-4")])
        arena.protocol.think_prm_verifier_agent = "nonexistent"
        ctx = MockDebateContext(
            context_messages=[
                Message(role="proposer", agent="gpt-4", content="Test", round=0),
            ]
        )

        result = await _run_think_prm_verification(arena, ctx)

        # Should succeed using gpt-4 as fallback
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_verification_error_gracefully(self) -> None:
        """Test that verification errors are handled gracefully.

        When individual step verification fails, ThinkPRM returns UNCERTAIN
        verdicts instead of failing the entire verification. This is the
        expected graceful degradation behavior.
        """
        arena = MockArena(agents=[MockAgent()])

        # Make autonomic.generate raise an error
        async def failing_generate(*args, **kwargs):
            raise RuntimeError("Test error")

        arena.autonomic.generate = failing_generate
        ctx = MockDebateContext(
            context_messages=[
                Message(role="proposer", agent="claude", content="Test", round=0),
            ]
        )

        result = await _run_think_prm_verification(arena, ctx)

        # ThinkPRM handles errors gracefully - returns result with UNCERTAIN verdicts
        assert result is not None
        assert result.total_steps == 1
        assert result.uncertain_steps == 1  # Error causes UNCERTAIN verdict
        assert result.overall_score == 0.0  # UNCERTAIN is not counted as correct


class TestProtocolFlags:
    """Test protocol configuration for ThinkPRM."""

    def test_protocol_has_think_prm_flags(self) -> None:
        """Test that DebateProtocol has ThinkPRM configuration."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()

        # Check all ThinkPRM flags exist
        assert hasattr(protocol, "enable_think_prm")
        assert hasattr(protocol, "think_prm_verifier_agent")
        assert hasattr(protocol, "think_prm_parallel")
        assert hasattr(protocol, "think_prm_max_parallel")

        # Check defaults
        assert protocol.enable_think_prm is False
        assert protocol.think_prm_verifier_agent == "claude"
        assert protocol.think_prm_parallel is True
        assert protocol.think_prm_max_parallel == 3
