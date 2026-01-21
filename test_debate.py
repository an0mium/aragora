#!/usr/bin/env python3
"""
Quick test of the aragora multi-agent debate framework.
"""

import asyncio
import sys

sys.path.insert(0, "/Users/armand/Development/aragora")

from aragora.agents import create_agent
from aragora.debate import Arena, DebateProtocol
from aragora.core import Environment
from aragora.memory import CritiqueStore


async def test_simple_debate():
    """Run a simple 2-agent debate using Codex."""

    # Create two codex agents with different roles
    agents = [
        create_agent("codex", name="proposer", role="proposer"),
        create_agent("codex", name="critic", role="critic"),
    ]

    # Define a simple task
    env = Environment(
        task="Design a simple in-memory cache in Python with TTL (time-to-live) support. Keep it under 50 lines.",
        max_rounds=2,
    )

    # Configure debate with just 2 rounds
    protocol = DebateProtocol(
        rounds=2,
        consensus="majority",
    )

    # Create memory store
    memory = CritiqueStore("/tmp/aragora_test.db")

    # Run debate
    arena = Arena(env, agents, protocol, memory)
    result = await arena.run()

    # Print results

    # Show stats
    stats = memory.get_stats()
    for key, value in stats.items():
        pass

    return result


if __name__ == "__main__":
    result = asyncio.run(test_simple_debate())
