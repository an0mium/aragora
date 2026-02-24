#!/usr/bin/env python3
"""
14_decision_receipts.py - Generate and verify decision receipts.

Decision receipts provide cryptographic audit trails for every debate
and decision made through Aragora. They include SHA-256 integrity hashes,
full provenance chains, and can be verified independently.

Usage:
    python 14_decision_receipts.py                    # Generate receipt
    python 14_decision_receipts.py --dry-run          # Preview
    python 14_decision_receipts.py --verify RECEIPT_ID # Verify a receipt
"""

import argparse
import asyncio
from aragora_sdk import AragoraClient, DebateConfig, Agent


async def generate_receipt(dry_run: bool = False) -> dict:
    """Run a debate and generate a decision receipt."""

    client = AragoraClient()

    if dry_run:
        print("[DRY RUN] Would run debate and generate receipt")
        return {"status": "dry_run"}

    # Run a debate that produces a receipt
    config = DebateConfig(
        topic="Should we migrate from REST to GraphQL for our public API?",
        agents=[
            Agent(name="architect", model="claude-sonnet-4-20250514"),
            Agent(name="pragmatist", model="gpt-4o"),
            Agent(name="security", model="gemini-2.0-flash"),
        ],
        rounds=3,
        generate_receipt=True,  # Enable receipt generation
    )

    result = await client.run_debate(config)
    receipt_id = result.get("receipt_id")
    print(f"Debate complete. Receipt: {receipt_id}")

    # Fetch the full receipt
    receipt = await client.receipts.get(receipt_id)

    print("\nDecision Receipt")
    print(f"  ID: {receipt['id']}")
    print(f"  Topic: {receipt['topic']}")
    print(f"  Consensus: {receipt.get('consensus', 'none')}")
    print(f"  Confidence: {receipt.get('confidence', 0):.1%}")
    print(f"  Content Hash: {receipt.get('content_hash', '?')}")
    print(f"  Agents: {', '.join(receipt.get('agents', []))}")
    print(f"  Rounds: {receipt.get('rounds', 0)}")
    print(f"  Created: {receipt.get('created_at', '?')}")

    return receipt


async def verify_receipt(receipt_id: str) -> dict:
    """Verify a receipt's integrity."""

    client = AragoraClient()

    verification = await client.receipts.verify(receipt_id)
    valid = verification.get("valid", False)
    icon = "VALID" if valid else "INVALID"

    print(f"Receipt Verification: {icon}")
    print(f"  Hash match: {verification.get('hash_match', False)}")
    print(f"  Chain intact: {verification.get('chain_intact', False)}")
    print(f"  Agents verified: {verification.get('agents_verified', False)}")

    return verification


async def list_receipts() -> list:
    """List recent decision receipts."""

    client = AragoraClient()

    receipts = await client.receipts.list(limit=10)
    print(f"Recent Receipts ({len(receipts)}):")
    for r in receipts:
        consensus = r.get("consensus", "?")
        print(f"  [{r.get('id', '?')[:8]}] {r.get('topic', '?')[:60]} -> {consensus}")

    return receipts


def main():
    parser = argparse.ArgumentParser(description="Decision receipt operations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=str, help="Verify a receipt by ID")
    parser.add_argument("--list", action="store_true", help="List recent receipts")
    args = parser.parse_args()

    if args.verify:
        asyncio.run(verify_receipt(args.verify))
    elif args.list:
        asyncio.run(list_receipts())
    else:
        asyncio.run(generate_receipt(args.dry_run))


if __name__ == "__main__":
    main()
