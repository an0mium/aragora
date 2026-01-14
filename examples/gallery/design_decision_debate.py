#!/usr/bin/env python3
"""
Design Decision Debate Example
==============================

Multi-agent architectural debate to evaluate different design options
and reach consensus on the best approach.

Use case: Making informed technical decisions with diverse AI perspectives.

Time: ~3-5 minutes
Requirements: At least one API key

Usage:
    python examples/gallery/design_decision_debate.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent


DESIGN_QUESTION = """
We need to choose a database for our new microservices platform.

Requirements:
- 100,000 read/write operations per second
- Strong consistency for financial transactions
- Horizontal scalability across regions
- Schema flexibility for evolving data models
- Must support ACID transactions

Options to evaluate:
1. PostgreSQL with read replicas
2. CockroachDB (distributed SQL)
3. MongoDB with transactions
4. Amazon Aurora

Team context:
- 5 backend engineers familiar with SQL
- Existing PostgreSQL deployment
- 99.99% uptime requirement
- Budget: $50k/year for database infrastructure
"""


async def run_design_debate():
    """Run a multi-agent design decision debate."""

    print("\n" + "=" * 60)
    print("ARAGORA: Design Decision Debate")
    print("=" * 60)

    # Create agents with different perspectives
    agent_configs = [
        ("anthropic-api", "architect"),
        ("openai-api", "critic"),
        ("grok", "pragmatist"),
    ]

    agents = []
    for agent_type, role in agent_configs:
        try:
            agent = create_agent(model_type=agent_type, name=f"{role}", role=role)  # type: ignore
            agents.append(agent)
            print(f"  + {agent.name} ready")
        except Exception as e:
            print(f"  - {agent_type} unavailable: {str(e)[:40]}")

    if len(agents) < 2:
        print("\nError: Need at least 2 agents. Check API keys.")
        return None

    env = Environment(
        task=f"""Evaluate the database options and recommend the best choice.

{DESIGN_QUESTION}

Provide:
1. Evaluation of each option against requirements
2. Risk analysis for each option
3. Final recommendation with justification
4. Migration strategy if switching from PostgreSQL""",
        context="Enterprise microservices platform design decision",
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        early_stopping=True,
    )

    print(f"\nDebating database choice with {len(agents)} agents...")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\n{'='*60}")
    print("DESIGN DECISION RESULTS")
    print(f"{'='*60}")
    print(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds: {result.rounds_used}")

    print(f"\n--- Recommendation ---")
    answer = result.final_answer
    print(answer[:1500] if len(answer) > 1500 else answer)

    return result


if __name__ == "__main__":
    result = asyncio.run(run_design_debate())
    if result and result.consensus_reached:
        print("\n[SUCCESS] Design decision completed with consensus!")
