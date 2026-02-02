"""
Knowledge Integration Example

Demonstrates using the Knowledge system with debates.
Shows how to create facts, run knowledge-powered debates,
and validate results with the Gauntlet.

Usage:
    python examples/knowledge_integration.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from aragora_sdk import AragoraAsyncClient

# =============================================================================
# Knowledge System Overview
# =============================================================================


async def knowledge_overview(client: AragoraAsyncClient) -> None:
    """Overview of Knowledge system capabilities."""
    print("=== Knowledge System Overview ===\n")

    print("The Knowledge system provides:")
    print("  - Fact storage with confidence scores")
    print("  - Semantic search across knowledge base")
    print("  - Evidence linking and citations")
    print("  - Cross-debate knowledge sharing")
    print("  - Validation and contradiction detection")
    print()


# =============================================================================
# Creating and Managing Facts
# =============================================================================


async def create_facts(client: AragoraAsyncClient) -> list[dict[str, Any]]:
    """Create facts in the knowledge system."""
    print("=== Creating Facts ===\n")

    facts_data = [
        {
            "content": "Python was created by Guido van Rossum in 1991.",
            "source": "Wikipedia",
            "confidence": 0.95,
            "tags": ["python", "history", "programming"],
        },
        {
            "content": "Python emphasizes code readability with significant whitespace.",
            "source": "Python Documentation",
            "confidence": 0.99,
            "tags": ["python", "syntax", "design"],
        },
        {
            "content": "Python supports multiple programming paradigms including procedural, object-oriented, and functional.",
            "source": "Python Documentation",
            "confidence": 0.99,
            "tags": ["python", "paradigms", "features"],
        },
        {
            "content": "The Python Package Index (PyPI) hosts over 400,000 packages.",
            "source": "PyPI Statistics",
            "confidence": 0.85,
            "tags": ["python", "ecosystem", "packages"],
        },
    ]

    created_facts = []
    for fact_data in facts_data:
        print(f"Creating fact: {fact_data['content'][:50]}...")
        fact = await client.knowledge.create_fact(
            content=fact_data["content"],
            source=fact_data["source"],
            confidence=fact_data["confidence"],
            tags=fact_data["tags"],
        )
        created_facts.append(fact)
        print(f"  Created: {fact['id']} (confidence: {fact['confidence']})")

    print(f"\nCreated {len(created_facts)} facts")
    return created_facts


# =============================================================================
# Querying Knowledge
# =============================================================================


async def query_knowledge(client: AragoraAsyncClient) -> list[dict[str, Any]]:
    """Query the knowledge base with semantic search."""
    print("\n=== Querying Knowledge ===\n")

    # Semantic search
    query = "What are Python's main features?"
    print(f"Query: '{query}'")

    results = await client.knowledge.search(
        query=query,
        limit=5,
        min_confidence=0.7,
    )

    print(f"\nFound {len(results.get('facts', []))} relevant facts:")
    for fact in results.get("facts", []):
        print(f"\n  [{fact.get('confidence', 0):.0%}] {fact['content'][:80]}...")
        print(f"       Source: {fact.get('source', 'N/A')}")
        print(f"       Tags: {', '.join(fact.get('tags', []))}")

    return results.get("facts", [])


# =============================================================================
# Knowledge-Powered Debate
# =============================================================================


async def run_knowledge_debate(
    client: AragoraAsyncClient,
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a debate powered by the knowledge system."""
    print("\n=== Knowledge-Powered Debate ===\n")

    # Extract fact IDs for context
    fact_ids = [f["id"] for f in facts if "id" in f]

    print("Creating debate with knowledge context...")
    print(f"  Using {len(fact_ids)} facts as context")

    debate = await client.debates.create(
        task="Based on Python's design philosophy and ecosystem, what makes it suitable for beginners?",
        agents=["claude", "gpt-4"],
        rounds=2,
        consensus="weighted",
        # Include knowledge context
        knowledge_context={
            "fact_ids": fact_ids,
            "include_related": True,  # Also include related facts
            "max_facts": 10,
        },
    )

    debate_id = debate["debate_id"]
    print(f"Debate created: {debate_id}")

    # Wait for completion
    print("Waiting for debate to complete...")
    while debate.get("status") in ("running", "pending"):
        await asyncio.sleep(2)
        debate = await client.debates.get(debate_id)
        print(f"  Status: {debate['status']}")

    # Show results
    if debate.get("status") == "completed":
        consensus = debate.get("consensus", {})
        print("\n--- Debate Results ---")
        print(f"Final Answer: {consensus.get('final_answer', 'N/A')[:200]}...")
        print(f"Confidence: {consensus.get('confidence', 0):.1%}")

        # Show which facts were cited
        citations = debate.get("citations", [])
        if citations:
            print(f"\nCitations used: {len(citations)}")
            for cite in citations[:3]:
                print(f"  - {cite.get('fact_content', 'N/A')[:60]}...")

    return debate


# =============================================================================
# Validating with Gauntlet
# =============================================================================


async def validate_with_gauntlet(
    client: AragoraAsyncClient,
    debate: dict[str, Any],
) -> dict[str, Any]:
    """Validate debate results using the Gauntlet."""
    print("\n=== Gauntlet Validation ===\n")

    debate_id = debate["debate_id"]
    print(f"Running Gauntlet validation for debate {debate_id}...")

    # Run Gauntlet validation
    gauntlet = await client.gauntlet.run(
        debate_id=debate_id,
        validation_types=["factual", "logical", "bias"],
        severity="standard",
    )

    gauntlet_id = gauntlet["gauntlet_id"]
    print(f"Gauntlet started: {gauntlet_id}")

    # Wait for completion
    while gauntlet.get("status") in ("running", "pending"):
        await asyncio.sleep(2)
        gauntlet = await client.gauntlet.get(gauntlet_id)
        print(f"  Status: {gauntlet['status']}")

    # Show validation results
    if gauntlet.get("status") == "completed":
        print("\n--- Validation Results ---")
        print(f"Overall Score: {gauntlet.get('overall_score', 0):.1%}")
        print(f"Passed: {gauntlet.get('passed', False)}")

        findings = gauntlet.get("findings", [])
        if findings:
            print(f"\nFindings ({len(findings)}):")
            for finding in findings[:5]:
                severity = finding.get("severity", "info")
                message = finding.get("message", "N/A")
                print(f"  [{severity.upper()}] {message[:60]}...")

        # Get decision receipt
        receipt = await client.gauntlet.get_receipt(gauntlet_id)
        print(f"\nDecision Receipt: {receipt.get('receipt_id', 'N/A')}")
        print(f"  Hash: {receipt.get('hash', 'N/A')[:16]}...")
        print(f"  Timestamp: {receipt.get('timestamp', 'N/A')}")

    return gauntlet


# =============================================================================
# Learning from Debates
# =============================================================================


async def extract_learnings(
    client: AragoraAsyncClient,
    debate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract new knowledge from debate results."""
    print("\n=== Extracting Learnings ===\n")

    debate_id = debate["debate_id"]

    # Extract insights that could become new facts
    print("Analyzing debate for new insights...")
    insights = await client.knowledge.extract_insights(
        debate_id=debate_id,
        min_confidence=0.8,
    )

    extracted = insights.get("insights", [])
    print(f"Found {len(extracted)} potential new facts:")

    new_facts = []
    for insight in extracted[:3]:
        print(f"\n  Content: {insight['content'][:80]}...")
        print(f"  Confidence: {insight.get('confidence', 0):.1%}")
        print(f"  Source: Debate {debate_id}")

        # Optionally save as new fact
        if insight.get("confidence", 0) >= 0.85:
            print("  -> Saving as new fact...")
            fact = await client.knowledge.create_fact(
                content=insight["content"],
                source=f"debate:{debate_id}",
                confidence=insight["confidence"],
                tags=["extracted", "debate-derived"],
            )
            new_facts.append(fact)

    print(f"\nSaved {len(new_facts)} new facts to knowledge base")
    return new_facts


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run knowledge integration demonstration."""
    print("Aragora SDK Knowledge Integration Example")
    print("=" * 60)

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Overview
        await knowledge_overview(client)

        # Create facts
        facts = await create_facts(client)

        # Query knowledge
        relevant_facts = await query_knowledge(client)
        print(f"\nRetrieved {len(relevant_facts)} facts for debate context")

        # Run knowledge-powered debate
        debate = await run_knowledge_debate(client, facts)

        # Validate with Gauntlet
        if debate.get("status") == "completed":
            await validate_with_gauntlet(client, debate)

            # Extract learnings
            await extract_learnings(client, debate)

    print("\n" + "=" * 60)
    print("Knowledge integration example complete!")
    print("\nKey Concepts:")
    print("  1. Facts: Structured knowledge with confidence scores")
    print("  2. Semantic Search: Find relevant facts by meaning")
    print("  3. Knowledge Context: Inform debates with facts")
    print("  4. Gauntlet: Validate debate results")
    print("  5. Learning: Extract new facts from debates")


if __name__ == "__main__":
    asyncio.run(main())
