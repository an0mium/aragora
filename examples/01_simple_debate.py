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
import time
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

TASK = """Design a rate limiter API for a high-traffic web service.

Requirements:
- Handle 10,000 requests/second
- Support per-user and global limits
- Gracefully degrade under load

Provide a concrete API design with:
1. Key endpoints
2. Data structures
3. Algorithm choice (token bucket, sliding window, etc.)
4. Error handling strategy"""


def run_demo():
    """Simulate a debate with pre-built responses (no API keys needed)."""
    print("=== Aragora Multi-Agent Debate (Demo Mode) ===\n")
    print(f"Task: {TASK.splitlines()[0]}")
    print()

    agents = [
        ("claude-proposer", "proposer"),
        ("gpt4-critic", "critic"),
        ("gemini-synthesizer", "synthesizer"),
    ]
    print(f"Agents: {[name for name, _ in agents]}")
    print("Rounds: 2\n")

    # Round 1: Proposals
    proposals = {
        "claude-proposer": (
            "I recommend a token bucket algorithm with a sliding window fallback. "
            "Key endpoints: POST /api/v1/check-rate (returns allow/deny + remaining "
            "tokens), GET /api/v1/limits/{user_id} (current usage). Data structure: "
            "Redis sorted sets for O(log N) window queries. Graceful degradation via "
            "priority queues that shed low-priority traffic first."
        ),
        "gpt4-critic": (
            "The token bucket approach is solid but I'd add a leaky bucket for global "
            "limits. Consider: PUT /api/v1/limits (admin config), DELETE "
            "/api/v1/limits/{user_id}/reset. Error handling: 429 with Retry-After "
            "header, X-RateLimit-Remaining/Limit/Reset response headers. Add circuit "
            "breaker for Redis failures with local LRU fallback cache."
        ),
    }
    critiques = {
        "gpt4-critic": (
            "Missing admin endpoints for limit configuration. No mention of "
            "distributed rate limiting across multiple nodes. Retry-After header "
            "not specified. Severity: 4/10."
        ),
        "claude-proposer": (
            "Good addition of admin endpoints, but leaky bucket for global limits "
            "adds complexity without clear benefit over sliding window. Circuit "
            "breaker is a strong suggestion. Severity: 3/10."
        ),
    }
    synthesis = (
        "FINAL DESIGN: Hybrid rate limiter using token bucket per-user and sliding "
        "window globally.\n\n"
        "Endpoints:\n"
        "  POST /api/v1/check-rate - Check and consume rate limit token\n"
        "  GET  /api/v1/limits/{user_id} - Query current usage\n"
        "  PUT  /api/v1/limits - Admin: configure limits\n"
        "  DELETE /api/v1/limits/{user_id}/reset - Admin: reset user limits\n\n"
        "Data structures: Redis sorted sets (per-user), atomic counters (global).\n"
        "Fallback: Local LRU cache with circuit breaker on Redis failure.\n"
        "Headers: X-RateLimit-Remaining, X-RateLimit-Limit, Retry-After on 429.\n"
        "Handles 10K req/s via sharded Redis with consistent hashing."
    )

    # Simulate debate progression with timing
    print("--- Round 1: Proposals ---")
    for agent_name, proposal in proposals.items():
        time.sleep(0.3)
        print(f"\n[{agent_name}]:")
        print(f"  {proposal}")

    print("\n--- Round 1: Critiques ---")
    for agent_name, critique in critiques.items():
        time.sleep(0.2)
        print(f"\n[{agent_name}] critique:")
        print(f"  {critique}")

    print("\n--- Round 2: Synthesis ---")
    time.sleep(0.3)
    print("\n[gemini-synthesizer]:")
    for line in synthesis.split("\n"):
        print(f"  {line}")

    print("\n--- Result ---")
    print("Consensus: Yes (85% confidence)")
    print("Votes: claude-proposer -> synthesis, gpt4-critic -> synthesis")
    print(f"Final answer length: {len(synthesis)} chars")
    return True


async def run_live_debate():
    """Run a real multi-agent debate (requires API keys)."""
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

    env = Environment(task=TASK)
    protocol = DebateProtocol(rounds=2, consensus="majority", early_stopping=True)

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

    if args.demo:
        ok = run_demo()
        sys.exit(0 if ok else 1)
    else:
        result = asyncio.run(run_live_debate())
        sys.exit(0 if result else 1)
