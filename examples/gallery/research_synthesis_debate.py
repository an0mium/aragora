#!/usr/bin/env python3
"""
Research Synthesis Debate Example
==================================

Multi-agent synthesis of research findings from different sources
to produce balanced, evidence-based conclusions.

Use case: Literature review, evidence aggregation, systematic analysis.

Time: ~4-6 minutes
Requirements: At least one API key

Usage:
    python examples/gallery/research_synthesis_debate.py
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


RESEARCH_TOPIC = """
Research Question: What are the most effective strategies for
reducing technical debt in large software organizations?

Source 1 - Academic Study (2024):
"Microservice Migration Patterns" - IEEE Software
- Found 73% reduction in deployment time after decomposition
- However, 45% increase in operational complexity
- Recommends gradual migration over 18-24 months

Source 2 - Industry Report (2025):
"State of DevOps" - DORA/Google
- Teams with <10% tech debt spend 44% more time on features
- Automated testing coverage >80% correlates with 5x faster recovery
- Top performers allocate 20% sprint capacity to debt reduction

Source 3 - Case Study (2024):
"Shopify's Modular Monolith Journey"
- Rejected microservices in favor of modular monolith
- 3x improvement in developer productivity
- Key: strong module boundaries, not deployment units

Source 4 - Contrarian View (2025):
"The Tech Debt Myth" - ACM Queue
- Argues "tech debt" is often just "code we don't like"
- Proposes measuring by actual maintenance cost, not opinion
- Only 23% of identified "debt" caused measurable problems
"""


async def run_research_synthesis_debate():
    """Run a multi-agent research synthesis debate."""

    print("\n" + "=" * 60)
    print("ARAGORA: Research Synthesis Debate")
    print("=" * 60)

    # Create agents with different analytical perspectives
    agent_configs = [
        ("anthropic-api", "academic_analyst"),
        ("openai-api", "practitioner"),
        ("grok", "skeptic"),
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
        task=f"""Synthesize these research sources and provide evidence-based recommendations.

{RESEARCH_TOPIC}

Analyze:
1. Areas of agreement across sources
2. Contradictions and how to reconcile them
3. Strength of evidence for each claim
4. Gaps in the research

Provide:
1. Key findings ranked by evidence strength
2. Practical recommendations for engineering teams
3. Caveats and limitations
4. Suggested metrics to track success""",
        context="Research synthesis for engineering leadership",
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        early_stopping=True,
    )

    print(f"\nSynthesizing research with {len(agents)} agents...")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\n{'='*60}")
    print("RESEARCH SYNTHESIS RESULTS")
    print(f"{'='*60}")
    print(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")

    print(f"\n--- Synthesis ---")
    answer = result.final_answer
    print(answer[:2000] if len(answer) > 2000 else answer)

    return result


if __name__ == "__main__":
    result = asyncio.run(run_research_synthesis_debate())
    if result and result.consensus_reached:
        print("\n[SUCCESS] Research synthesis completed with consensus!")
