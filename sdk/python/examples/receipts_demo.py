"""
Decision Receipts Example

Demonstrates how to work with decision receipts — cryptographic audit
trails for debate outcomes. Includes listing, verification, and export.

Usage:
    python examples/receipts_demo.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import os

from aragora import AragoraClient


def main() -> None:
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    )

    print("=== Decision Receipts Demo ===\n")

    # 1. List recent receipts
    print("1. Listing recent decision receipts...")
    result = client.gauntlet.list(limit=5)
    receipts = result.get("receipts", [])
    print(f"   Found {len(receipts)} receipts")

    if not receipts:
        print("   No receipts found. Run a debate first to generate receipts.")
        return

    for r in receipts:
        rid = r.get("receipt_id", "unknown")
        verdict = r.get("verdict", "N/A")
        confidence = r.get("confidence", "N/A")
        print(f"   - {rid}: {verdict} (confidence: {confidence})")

    # 2. Get receipt details
    print("\n2. Getting receipt details...")
    receipt_id = receipts[0]["receipt_id"]
    receipt = client.gauntlet.get(receipt_id)
    print(f"   ID: {receipt.get('receipt_id')}")
    print(f"   Verdict: {receipt.get('verdict')}")
    print(f"   Confidence: {receipt.get('confidence')}")
    consensus = receipt.get("consensus_reached", "N/A")
    print(f"   Consensus: {'Reached' if consensus else 'Not reached'}")
    agents = receipt.get("participating_agents", [])
    print(f"   Participating Agents: {', '.join(agents)}")

    # Check for dissent
    dissenters = receipt.get("dissenting_agents", [])
    if dissenters:
        print(f"   Dissenting Agents: {', '.join(dissenters)}")

    # 3. Verify receipt integrity
    print("\n3. Verifying receipt integrity...")
    verification = client.receipts.verify(receipt_id)
    print(f"   Valid: {verification.get('valid')}")
    print(f"   Hash: {verification.get('hash')}")
    print(f"   Verified at: {verification.get('verified_at')}")

    # 4. Export receipt in different formats
    print("\n4. Exporting receipt...")
    for fmt in ("markdown", "sarif"):
        try:
            exported = client.receipts.export(receipt_id, format=fmt)
            content = exported.get("content", "")
            print(f"   {fmt}: {len(content)} bytes")
        except Exception as e:
            print(f"   {fmt}: not available ({e})")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
