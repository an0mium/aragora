#!/usr/bin/env python3
"""
Basic Debate -- Simplest possible Aragora example.

Runs a 2-round adversarial debate between mock AI agents.
No API keys needed -- works completely offline.

Usage:
    python examples/quickstart/basic_debate.py

    # Or with ARAGORA_OFFLINE=1 for the full platform:
    ARAGORA_OFFLINE=1 python examples/quickstart/basic_debate.py
"""

import asyncio

from aragora_debate import Debate, create_agent, ReceiptBuilder


async def main():
    # Create a debate on any topic
    debate = Debate(
        topic="Design a rate limiter for 10,000 requests/second",
        rounds=2,
        consensus="majority",
    )

    # Add mock agents (no API keys required)
    debate.add_agent(
        create_agent(
            "mock",
            name="claude",
            proposal=(
                "Use a token bucket algorithm. It handles bursts well, "
                "is simple to implement with Redis INCR + EXPIRE, and "
                "provides predictable rate limiting per API key."
            ),
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="gpt",
            proposal=(
                "Token bucket is a solid choice, but consider sliding "
                "window log for more precise rate limiting. The tradeoff "
                "is higher memory usage per client. For 10K req/s, "
                "the token bucket's O(1) operations are preferable."
            ),
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="gemini",
            proposal=(
                "Consensus: token bucket for the hot path with a sliding "
                "window counter as a secondary check. This gives O(1) "
                "performance for most requests while catching edge cases "
                "that slip through bucket refill timing."
            ),
        )
    )

    # Run the debate
    result = await debate.run()

    # Print results
    print(f"Question: {result.task}")
    print(f"Agents: {result.participants}")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds: {result.rounds_used}")
    print(f"\nFinal answer:\n{result.final_answer[:500]}")

    # Print the decision receipt
    if result.receipt:
        print("\n" + "=" * 60)
        print(result.receipt.to_markdown())

        # Sign the receipt for audit compliance
        ReceiptBuilder.sign_hmac(result.receipt, key="demo-signing-key")
        print(f"\nSigned with HMAC-SHA256: {result.receipt.signature[:32]}...")

        # Verify the signature
        valid = ReceiptBuilder.verify_hmac(result.receipt, key="demo-signing-key")
        print(f"Signature valid: {valid}")


if __name__ == "__main__":
    asyncio.run(main())
