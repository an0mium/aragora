#!/usr/bin/env python3
"""
Decision Receipt -- Run a debate and produce an auditable, signed receipt.

Demonstrates:
- Running a debate with styled mock agents (realistic varied responses)
- Generating a Markdown decision receipt
- Signing the receipt with HMAC-SHA256 for tamper detection
- Verifying the signature

No API keys required. Works completely offline.

Usage:
    pip install aragora-debate
    python examples/quickstart/02_with_receipt.py
"""

import asyncio

from aragora_debate import Arena, ConsensusMethod, DebateConfig, ReceiptBuilder, StyledMockAgent


async def main():
    # StyledMockAgent produces varied, realistic responses based on style.
    # Styles: "supportive", "critical", "balanced", "contrarian"
    agents = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("devil-advocate", style="critical"),
        StyledMockAgent("synthesizer", style="balanced"),
    ]

    # Arena is the lower-level API with more configuration options
    arena = Arena(
        question="Should we adopt GraphQL to replace our REST API?",
        agents=agents,
        config=DebateConfig(
            rounds=2,
            consensus_method=ConsensusMethod.MAJORITY,
            early_stopping=True,
        ),
    )

    result = await arena.run()

    # --- Print the debate outcome ---
    print(f"Topic: {result.task}")
    print(f"Consensus: {result.consensus_reached} ({result.confidence:.0%})")
    print(f"Rounds: {result.rounds_used}")
    print()

    # --- Print the decision receipt in Markdown ---
    receipt = result.receipt
    if receipt:
        print("=" * 60)
        print(receipt.to_markdown())
        print("=" * 60)

        # --- Sign the receipt for audit compliance ---
        signing_key = "my-organization-signing-key"
        ReceiptBuilder.sign_hmac(receipt, key=signing_key)
        print("\nSigned with HMAC-SHA256")
        print(f"Signature: {receipt.signature[:48]}...")

        # --- Verify the signature ---
        is_valid = ReceiptBuilder.verify_hmac(receipt, key=signing_key)
        print(f"Signature valid: {is_valid}")

        # --- Tamper detection demo ---
        receipt.confidence = 0.99  # tamper with the data
        is_still_valid = ReceiptBuilder.verify_hmac(receipt, key=signing_key)
        print(f"After tampering: {is_still_valid}")

        # --- Export as JSON ---
        print("\nJSON export (first 300 chars):")
        json_str = ReceiptBuilder.to_json(receipt)
        print(json_str[:300] + "...")


if __name__ == "__main__":
    asyncio.run(main())
