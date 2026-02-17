#!/usr/bin/env python3
"""
Minimal Debate -- Three mock agents debate a technical decision.

No API keys required. Works completely offline.

Usage:
    pip install aragora-debate
    python examples/quickstart/01_simple_debate.py
"""

import asyncio

from aragora_debate import Debate, create_agent


async def main():
    # Create a debate with 2 rounds of propose/critique/vote
    debate = Debate(
        topic="Should we migrate from a monolith to microservices?",
        rounds=2,
        consensus="majority",
    )

    # Add three agents with different perspectives (no API keys needed)
    debate.add_agent(create_agent("mock", name="pro-microservices", proposal=(
        "Yes, migrate to microservices. Independent deployability reduces "
        "release risk, enables per-service scaling, and lets teams own their "
        "domain end-to-end. Start with the highest-churn bounded contexts "
        "(auth, billing) and extract incrementally."
    )))

    debate.add_agent(create_agent("mock", name="pro-monolith", proposal=(
        "Stay with the monolith. At our scale (50 req/s, 3 developers), "
        "the operational overhead of service mesh, distributed tracing, and "
        "cross-service transactions outweighs the benefits. A modular "
        "monolith with clear module boundaries gives 80% of the value."
    )))

    debate.add_agent(create_agent("mock", name="pragmatist", proposal=(
        "Take a hybrid approach: extract the two services with the most "
        "independent scaling needs (notification service, analytics pipeline) "
        "while keeping the core domain in a modular monolith. Measure the "
        "operational cost delta before extracting more."
    )))

    # Run the debate
    result = await debate.run()

    # Print results
    print(f"Topic: {result.task}")
    print(f"Agents: {', '.join(result.participants)}")
    print(f"Rounds used: {result.rounds_used}")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nFinal answer:\n{result.final_answer[:500]}")

    # The receipt is automatically generated
    if result.receipt:
        print(f"\nReceipt ID: {result.receipt.receipt_id}")
        print(f"Verdict: {result.receipt.verdict.value}")


if __name__ == "__main__":
    asyncio.run(main())
