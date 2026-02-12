#!/usr/bin/env python3
"""
Simple Multi-Agent Debate Example
==================================

This example shows Aragora's core value: heterogeneous AI agents debating
to produce better answers through critique and synthesis.

Time: ~10 seconds (demo) | ~2-5 minutes (live)
Requirements: None for --demo, or at least one API key for live mode

Usage:
    python examples/01_simple_debate.py --demo      # No API keys needed
    python examples/01_simple_debate.py              # Uses real AI agents
"""

import argparse
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add aragora to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol


def _create_mock_agents():
    """Create mock agents for demo mode (no API keys needed)."""
    mock_responses = [
        "I recommend a token bucket algorithm with a sliding window fallback. "
        "Key endpoints: POST /api/v1/check-rate (returns allow/deny + remaining tokens), "
        "GET /api/v1/limits/{user_id} (current usage). Data structure: Redis sorted sets "
        "for O(log N) window queries. Graceful degradation via priority queues.",
        "The token bucket approach is solid but I'd add a leaky bucket for global limits. "
        "Consider: PUT /api/v1/limits (admin config), DELETE /api/v1/limits/{user_id}/reset. "
        "Error handling: 429 with Retry-After header, X-RateLimit-* response headers. "
        "Add circuit breaker for Redis failures with local fallback cache.",
        "Synthesizing both perspectives: hybrid approach using token bucket per-user "
        "and sliding window globally. Final design includes 4 endpoints, Redis + local "
        "LRU cache fallback, structured 429 responses with Retry-After. Algorithm handles "
        "10K req/s via sharded Redis with consistent hashing.",
    ]
    agents = []
    for i, (name, role) in enumerate([
        ("claude-proposer", "proposer"),
        ("gpt-critic", "critic"),
        ("gemini-synthesizer", "synthesizer"),
    ]):
        agent = MagicMock()
        agent.name = name
        agent.role = role
        agent.model_type = name.split("-")[0]
        agent.generate = AsyncMock(return_value=mock_responses[i])
        agent.get_response = AsyncMock(return_value=mock_responses[i])
        agents.append(agent)
    return agents


async def run_simple_debate(demo: bool = False):
    """Run a simple multi-agent debate on API design."""

    if demo:
        agents = _create_mock_agents()
        print("Running in demo mode (mock agents, no API keys needed)")
    else:
        from aragora.agents.base import create_agent

        agent_configs = [
            ("anthropic-api", "proposer"),
            ("openai-api", "critic"),
            ("grok", "synthesizer"),
            ("kimi", "critic"),
            ("gemini", "critic"),
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
                pass

        if len(agents) < 2:
            print("Error: Need at least 2 agents. Set API keys or use --demo.")
            return None

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

    protocol = DebateProtocol(
        rounds=2,
        consensus="majority",
        early_stopping=True,
    )

    print(f"Starting debate with {len(agents)} agents: {[a.name for a in agents]}")
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\nConsensus: {'Yes' if result.consensus_reached else 'No'}")
    answer = result.final_answer
    if len(answer) > 800:
        print(f"Final answer ({len(answer)} chars): {answer[:800]}...")
    else:
        print(f"Final answer: {answer}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Multi-Agent Debate")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with mock agents (no API keys needed)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_simple_debate(demo=args.demo))
    sys.exit(0 if result else 1)
