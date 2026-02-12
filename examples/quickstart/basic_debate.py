#!/usr/bin/env python3
"""
Basic Debate -- Simplest possible Aragora example.

Runs a 2-round adversarial debate between multiple AI models.
In demo mode, uses mocked responses so no API keys are needed.

Usage:
    python examples/quickstart/basic_debate.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aragora import Arena, DebateProtocol, Environment


def make_mock_agent(name: str, role: str, response: str) -> MagicMock:
    """Create a mock agent that returns a fixed response."""
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.model_type = "mock"
    agent.generate = AsyncMock(return_value=response)
    agent.get_response = AsyncMock(return_value=response)
    return agent


async def main():
    # Create agents with predetermined responses for demo
    agents = [
        make_mock_agent(
            "claude",
            "proposer",
            "Use a token bucket algorithm. It handles bursts well, "
            "is simple to implement with Redis INCR + EXPIRE, and "
            "provides predictable rate limiting per API key.",
        ),
        make_mock_agent(
            "gpt",
            "critic",
            "Token bucket is a solid choice, but consider sliding "
            "window log for more precise rate limiting. The tradeoff "
            "is higher memory usage per client. For 10K req/s, "
            "the token bucket's O(1) operations are preferable.",
        ),
        make_mock_agent(
            "gemini",
            "synthesizer",
            "Consensus: token bucket for the hot path with a sliding "
            "window counter as a secondary check. This gives O(1) "
            "performance for most requests while catching edge cases "
            "that slip through bucket refill timing.",
        ),
    ]

    # Define the debate
    env = Environment(task="Design a rate limiter for 10,000 requests/second")
    protocol = DebateProtocol(rounds=2, consensus="majority")

    # Run it
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    # Print results
    print(f"Question: {env.task}")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nFinal answer:\n{result.final_answer[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
