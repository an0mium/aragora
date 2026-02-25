"""Example: Debate with Knowledge Mound + Freshness Weighting.

Demonstrates running a debate that pulls context from the Knowledge Mound,
with results sorted by a composite score of importance, freshness, and recency.

Prerequisites:
    pip install aragora
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...

Usage:
    python examples/enterprise-integration/knowledge_debate.py
"""

from __future__ import annotations

import asyncio

from aragora import Arena, Environment
from aragora.agents.base import create_agent
from aragora.debate.protocol import DebateProtocol
from aragora.knowledge.mound.core import KnowledgeMound


async def main() -> None:
    # Initialize the Knowledge Mound
    km = KnowledgeMound(workspace_id="demo-workspace")

    # Ingest some sample knowledge
    await km.ingest(
        {
            "title": "Q4 Revenue Report",
            "content": "Revenue grew 23% YoY to $4.2M in Q4 2025.",
            "source": "finance",
            "importance": 0.9,
        }
    )
    await km.ingest(
        {
            "title": "Customer Churn Analysis",
            "content": "Churn rate dropped to 2.1% after onboarding improvements.",
            "source": "customer_success",
            "importance": 0.7,
        }
    )

    # Query with freshness-weighted retrieval (importance 50%, freshness 30%, recency 20%)
    results = await km.query("What are the key business metrics?", limit=5)
    print(f"Retrieved {results.total_count} knowledge items:")
    for item in results.items:
        print(f"  - {item.title} (importance={item.importance})")

    # Create agents
    agents = [
        create_agent("anthropic-api", name="analyst", role="proposer"),
        create_agent("openai-api", name="strategist", role="critic"),
        create_agent("anthropic-api", name="synthesizer", role="synthesizer"),
    ]

    # Build context from knowledge retrieval
    context = "\n".join(
        f"- {item.title}: {item.content}" for item in results.items
    )

    # Set up debate with knowledge context
    env = Environment(
        task="Based on our latest business metrics, what should be our top strategic priority for Q1 2026?",
        context=context,
        max_rounds=3,
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        early_stopping=True,
    )

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\nDebate completed:")
    print(f"  Consensus: {result.consensus_reached}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Answer: {result.final_answer}")


if __name__ == "__main__":
    asyncio.run(main())
