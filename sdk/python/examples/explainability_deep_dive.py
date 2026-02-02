"""
Explainability Deep Dive Example

Demonstrates the explainability features of the Aragora SDK.
Shows how to get decision factors, counterfactuals, narratives,
and trace reasoning provenance.

Usage:
    python examples/explainability_deep_dive.py

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
# Run a Debate for Explanation
# =============================================================================


async def create_debate_for_explanation(
    client: AragoraAsyncClient,
) -> dict[str, Any]:
    """Create a debate that we'll explain."""
    print("=== Creating Debate for Explanation ===\n")

    debate = await client.debates.create(
        task="Should our startup adopt a microservices architecture or stay with a monolith?",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        consensus="weighted",
        # Enable explainability tracking
        options={
            "track_reasoning": True,
            "capture_evidence": True,
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

    if debate.get("status") == "completed":
        consensus = debate.get("consensus", {})
        print(f"\nDecision: {consensus.get('final_answer', 'N/A')[:100]}...")
        print(f"Confidence: {consensus.get('confidence', 0):.1%}")

    return debate


# =============================================================================
# Get Decision Factors
# =============================================================================


async def get_decision_factors(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Get the factors that influenced the decision."""
    print("\n=== Decision Factors ===\n")

    factors = await client.explainability.get_factors(debate_id)

    print(f"Found {len(factors.get('factors', []))} decision factors:\n")

    for i, factor in enumerate(factors.get("factors", []), 1):
        name = factor.get("name", "Unknown")
        weight = factor.get("weight", 0)
        direction = factor.get("direction", "neutral")  # positive/negative/neutral
        evidence = factor.get("evidence", [])

        # Direction indicator
        direction_icon = {"positive": "+", "negative": "-", "neutral": "~"}.get(direction, "?")

        print(f"{i}. [{direction_icon}] {name} (weight: {weight:.1%})")
        print(f"   Description: {factor.get('description', 'N/A')[:80]}...")

        if evidence:
            print(f"   Evidence ({len(evidence)} items):")
            for ev in evidence[:2]:
                print(f"     - {ev.get('content', 'N/A')[:60]}...")

        print()

    # Show factor summary
    summary = factors.get("summary", {})
    if summary:
        print("Factor Summary:")
        print(f"  Primary factor: {summary.get('primary_factor', 'N/A')}")
        print(f"  Decision confidence: {summary.get('confidence', 0):.1%}")
        print(f"  Consensus level: {summary.get('consensus_level', 'N/A')}")

    return factors


# =============================================================================
# Generate Counterfactuals
# =============================================================================


async def generate_counterfactuals(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Generate counterfactual scenarios - what would change the decision."""
    print("\n=== Counterfactual Analysis ===\n")

    counterfactuals = await client.explainability.get_counterfactuals(
        debate_id=debate_id,
        max_scenarios=5,
    )

    print("What conditions would change this decision?\n")

    for i, cf in enumerate(counterfactuals.get("counterfactuals", []), 1):
        condition = cf.get("condition", "Unknown")
        new_outcome = cf.get("new_outcome", "Unknown")
        likelihood = cf.get("likelihood", 0)
        impact = cf.get("impact", "medium")

        print(f"{i}. IF: {condition}")
        print(f"   THEN: {new_outcome}")
        print(f"   Likelihood: {likelihood:.1%} | Impact: {impact}")

        # Show factors affected
        affected = cf.get("affected_factors", [])
        if affected:
            print(f"   Affects: {', '.join(affected[:3])}")

        print()

    # Show sensitivity analysis
    sensitivity = counterfactuals.get("sensitivity", {})
    if sensitivity:
        print("Sensitivity Analysis:")
        print(f"  Most sensitive factor: {sensitivity.get('most_sensitive', 'N/A')}")
        print(f"  Decision stability: {sensitivity.get('stability', 'N/A')}")
        print(f"  Confidence range: {sensitivity.get('confidence_range', 'N/A')}")

    return counterfactuals


# =============================================================================
# Create Narrative Explanation
# =============================================================================


async def create_narrative(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Create a human-readable narrative explanation."""
    print("\n=== Narrative Explanation ===\n")

    narrative = await client.explainability.get_narrative(
        debate_id=debate_id,
        style="executive",  # Options: executive, technical, casual
        max_length=500,
    )

    print("Executive Summary:")
    print("-" * 40)
    print(narrative.get("narrative", "No narrative available"))
    print("-" * 40)

    # Show different narrative styles
    print("\nAvailable narrative styles:")
    print("  - executive: High-level summary for decision makers")
    print("  - technical: Detailed technical reasoning")
    print("  - casual: Conversational explanation")

    # Get technical narrative too
    tech_narrative = await client.explainability.get_narrative(
        debate_id=debate_id,
        style="technical",
        max_length=300,
    )

    print("\nTechnical Summary (excerpt):")
    print("-" * 40)
    print(tech_narrative.get("narrative", "N/A")[:300] + "...")
    print("-" * 40)

    return narrative


# =============================================================================
# Trace Reasoning Provenance
# =============================================================================


async def trace_provenance(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Trace the provenance of reasoning - where did each claim come from."""
    print("\n=== Reasoning Provenance ===\n")

    provenance = await client.explainability.get_provenance(debate_id)

    print("Reasoning Chain:\n")

    claims = provenance.get("claims", [])
    for i, claim in enumerate(claims[:5], 1):
        content = claim.get("content", "Unknown")
        source = claim.get("source", {})
        confidence = claim.get("confidence", 0)
        supports = claim.get("supports", [])
        contradicts = claim.get("contradicts", [])

        print(f"{i}. Claim: {content[:80]}...")
        print(f"   Source: {source.get('agent', 'N/A')} (Round {source.get('round', '?')})")
        print(f"   Confidence: {confidence:.1%}")

        if supports:
            print(f"   Supports: {len(supports)} other claims")
        if contradicts:
            print(f"   Contradicts: {len(contradicts)} claims")

        print()

    # Show reasoning graph summary
    graph = provenance.get("graph_summary", {})
    if graph:
        print("Reasoning Graph Summary:")
        print(f"  Total claims: {graph.get('total_claims', 0)}")
        print(f"  Support links: {graph.get('support_links', 0)}")
        print(f"  Contradiction links: {graph.get('contradiction_links', 0)}")
        print(f"  Key premises: {graph.get('key_premises', 0)}")

    return provenance


# =============================================================================
# Compare Agent Reasoning
# =============================================================================


async def compare_agent_reasoning(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Compare how different agents reasoned about the problem."""
    print("\n=== Agent Reasoning Comparison ===\n")

    comparison = await client.explainability.compare_reasoning(debate_id)

    agents = comparison.get("agents", {})
    for agent_name, agent_data in agents.items():
        print(f"{agent_name}:")
        print(f"  Position: {agent_data.get('position', 'N/A')[:60]}...")
        print(f"  Key arguments: {len(agent_data.get('key_arguments', []))}")
        print(f"  Evidence cited: {len(agent_data.get('evidence', []))}")
        print(f"  Confidence: {agent_data.get('confidence', 0):.1%}")

        # Show unique insights
        unique = agent_data.get("unique_insights", [])
        if unique:
            print("  Unique insights:")
            for insight in unique[:2]:
                print(f"    - {insight[:50]}...")

        print()

    # Show areas of agreement/disagreement
    print("Agreement Analysis:")
    agreement = comparison.get("agreement", {})
    print(f"  Points of agreement: {len(agreement.get('agreed', []))}")
    print(f"  Points of disagreement: {len(agreement.get('disagreed', []))}")
    print(f"  Overall alignment: {agreement.get('alignment_score', 0):.1%}")

    return comparison


# =============================================================================
# Export Explanation Report
# =============================================================================


async def export_explanation_report(
    client: AragoraAsyncClient,
    debate_id: str,
) -> dict[str, Any]:
    """Export a complete explanation report."""
    print("\n=== Exporting Explanation Report ===\n")

    report = await client.explainability.export_report(
        debate_id=debate_id,
        format="markdown",  # Options: markdown, html, pdf, json
        include=[
            "summary",
            "factors",
            "counterfactuals",
            "narrative",
            "provenance",
            "agent_comparison",
        ],
    )

    print(f"Report generated: {report.get('report_id', 'N/A')}")
    print(f"Format: {report.get('format', 'N/A')}")
    print(f"Size: {report.get('size_bytes', 0)} bytes")

    if report.get("download_url"):
        print(f"Download URL: {report['download_url']}")

    # Show report preview
    preview = report.get("preview", "")
    if preview:
        print("\nReport Preview:")
        print("-" * 40)
        print(preview[:500] + "...")
        print("-" * 40)

    return report


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run explainability demonstrations."""
    print("Aragora SDK Explainability Deep Dive")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nExplainability features demonstrated:")
        print("  1. Decision Factors: What influenced the decision")
        print("  2. Counterfactuals: What would change the outcome")
        print("  3. Narratives: Human-readable explanations")
        print("  4. Provenance: Trace reasoning chains")
        print("  5. Agent Comparison: How agents differed")
        print("  6. Reports: Export complete explanations")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Create a debate
        debate = await create_debate_for_explanation(client)

        if debate.get("status") != "completed":
            print("Debate did not complete successfully")
            return

        debate_id = debate["debate_id"]

        # Get decision factors
        await get_decision_factors(client, debate_id)

        # Generate counterfactuals
        await generate_counterfactuals(client, debate_id)

        # Create narrative
        await create_narrative(client, debate_id)

        # Trace provenance
        await trace_provenance(client, debate_id)

        # Compare agent reasoning
        await compare_agent_reasoning(client, debate_id)

        # Export report
        await export_explanation_report(client, debate_id)

    print("\n" + "=" * 60)
    print("Explainability deep dive complete!")
    print("\nKey Takeaways:")
    print("  - Factors show WHAT influenced the decision")
    print("  - Counterfactuals show WHAT WOULD CHANGE it")
    print("  - Narratives explain WHY in plain language")
    print("  - Provenance traces WHERE reasoning came from")
    print("  - Comparison shows HOW agents differed")


if __name__ == "__main__":
    asyncio.run(main())
