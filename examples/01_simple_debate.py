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

    print("\n" + "=" * 60)
    print("ARAGORA: Simple Multi-Agent Debate")
    print("=" * 60)

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
    print("\nInitializing agents...")

    for agent_type, role in agent_configs:
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore
                name=f"{agent_type}-{role}",
                role=role,
            )
            agents.append(agent)
            print(f"  + {agent.name} ready ({role})")
        except Exception as e:
            # Skip agents that can't be created (missing API keys)
            print(f"  - {agent_type} unavailable: {str(e)[:50]}")

    if len(agents) < 2:
        print("\nError: Need at least 2 agents for a debate.")
        print("Set at least two of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY")
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
    print(f"\n{'='*60}")
    print("DEBATE: Rate Limiter API Design")
    print(f"{'='*60}")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Rounds: {protocol.rounds}")
    print(f"Consensus mode: {protocol.consensus}")
    print("\nDebate in progress...")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Consensus reached: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds used: {result.rounds_used}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    # Show final answer (truncated for display)
    print(f"\n--- Final Answer ---")
    answer = result.final_answer
    if len(answer) > 800:
        print(answer[:800] + "\n\n[... truncated for display ...]")
    else:
        print(answer)

    return result


if __name__ == "__main__":
    print("Aragora Simple Debate Example")
    print("This demonstrates multi-agent debate with propose/critique/synthesize.")

    result = asyncio.run(run_simple_debate())

    if result and result.consensus_reached:
        print("\n[SUCCESS] Debate completed with consensus!")
    elif result:
        print("\n[INFO] Debate completed without full consensus")
    else:
        print("\n[ERROR] Debate could not run - check API keys")
