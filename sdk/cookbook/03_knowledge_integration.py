#!/usr/bin/env python3
"""
03_knowledge_integration.py - Query and inject knowledge into debates.

This example demonstrates how to:
1. Query the Knowledge Mound for relevant context
2. Inject organizational knowledge into debates
3. Collect and store evidence from debate outcomes

Usage:
    python 03_knowledge_integration.py --dry-run
    python 03_knowledge_integration.py --query "rate limiting patterns"
"""

import argparse
import asyncio
from aragora_sdk import ArenaClient, DebateConfig, Agent
from aragora_sdk.knowledge import KnowledgeMound, Evidence, KnowledgeQuery


async def run_knowledge_debate(query: str, dry_run: bool = False) -> dict:
    """Run a debate enriched with organizational knowledge."""

    # Initialize clients
    client = ArenaClient()
    knowledge = KnowledgeMound()

    if dry_run:
        print(f"[DRY RUN] Query: {query}")
        print("[DRY RUN] Would search Knowledge Mound for relevant context")
        print("[DRY RUN] Would inject knowledge into debate")
        print("[DRY RUN] Would collect evidence from outcome")
        return {"status": "dry_run", "query": query}

    # Step 1: Query Knowledge Mound for relevant context
    print(f"Querying knowledge base for: {query}")
    knowledge_query = KnowledgeQuery(
        text=query,
        limit=5,  # Max results to retrieve
        min_confidence=0.7,  # Only high-confidence knowledge
        include_evidence=True,  # Include supporting evidence
    )
    knowledge_results = await knowledge.search(knowledge_query)

    print(f"Found {len(knowledge_results)} relevant knowledge items")
    for item in knowledge_results[:3]:
        print(f"  - {item.title} (confidence: {item.confidence:.2%})")

    # Step 2: Configure debate with knowledge injection
    agents = [
        Agent(name="claude", model="claude-sonnet-4-20250514"),
        Agent(name="gpt", model="gpt-4o"),
    ]

    config = DebateConfig(
        topic=f"Based on our organizational knowledge, {query}",
        agents=agents,
        rounds=2,
        # Inject retrieved knowledge as context
        knowledge_context=knowledge_results,
        # Enable evidence collection during debate
        collect_evidence=True,
    )

    # Step 3: Run the debate
    result = await client.run_debate(config)

    # Step 4: Store evidence from the debate outcome
    if result.consensus_reached:
        evidence = Evidence(
            claim=result.decision,
            confidence=result.confidence,
            sources=[agent.name for agent in agents],
            debate_id=result.debate_id,
        )
        await knowledge.store_evidence(evidence)
        print(f"Stored new evidence with confidence {evidence.confidence:.2%}")

    print(f"\nDecision: {result.decision}")
    print(f"Informed by {len(knowledge_results)} knowledge items")

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run a knowledge-enriched debate")
    parser.add_argument(
        "--query", default="best practices for API rate limiting", help="Knowledge query topic"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_knowledge_debate(args.query, args.dry_run))
    return result


if __name__ == "__main__":
    main()
