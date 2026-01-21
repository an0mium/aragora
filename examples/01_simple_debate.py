#!/usr/bin/env python3
"""
Simple Multi-Agent Debate Example
==================================

This example shows Aragora's core value: heterogeneous AI agents debating
to produce better answers through critique and synthesis.

Time: ~2-5 minutes
Requirements: At least one API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or XAI_API_KEY)

Usage:
    python examples/01_simple_debate.py

Expected output:
    Starting debate: "Design a rate limiter API"
    Agents: ['grok-proposer', 'gemini-critic']
    Rounds: 2
    ...
    Consensus: Yes (85% confidence)
    Final Answer: [synthesized design with all perspectives]
"""

import asyncio
import sys
from pathlib import Path

# Add aragora to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent


async def run_simple_debate():
    """Run a simple multi-agent debate on API design."""

    # Try to create agents based on available API keys
    # Prioritize paid APIs (more reliable) over free tiers
    agent_configs = [
        ("anthropic-api", "proposer"),  # Claude API (reliable)
        ("openai-api", "critic"),  # OpenAI API (reliable)
        ("grok", "synthesizer"),  # xAI Grok
        ("kimi", "critic"),  # Moonshot Kimi
        ("gemini", "critic"),  # Google Gemini (often quota limited)
    ]

    agents = []

    for agent_type, role in agent_configs:
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore
                name=f"{agent_type}-{role}",
                role=role,
            )
            agents.append(agent)
        except Exception:
            # Skip agents that can't be created (missing API keys)
            pass

    if len(agents) < 2:
        return None

    # Define the debate environment
    env = Environment(
        task="""Design a rate limiter API for a high-traffic web service.

Requirements:
- Handle 10,000 requests/second
- Support per-user and global limits
- Gracefully degrade under load

Provide a concrete API design with:
1. Key endpoints
2. Data structures
3. Algorithm choice (token bucket, sliding window, etc.)
4. Error handling strategy""",
    )

    # Configure debate protocol
    protocol = DebateProtocol(
        rounds=2,  # Keep short for demo
        consensus="majority",  # Majority vote for consensus
        early_stopping=True,  # Stop early if consensus reached
    )

    # Create and run debate

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    # Display results

    # Show final answer (truncated for display)
    answer = result.final_answer
    if len(answer) > 800:
        pass
    else:
        pass

    return result


if __name__ == "__main__":
    result = asyncio.run(run_simple_debate())

    if result and result.consensus_reached:
        pass
    elif result:
        pass
    else:
        pass
