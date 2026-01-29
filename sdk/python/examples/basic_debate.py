"""
Basic Debate Example

Demonstrates how to create and run a simple debate using the Aragora SDK.

Usage:
    python examples/basic_debate.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import os
import time

from aragora import AragoraClient


def main() -> None:
    # Create client
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    )

    print("Creating debate...")

    # Create a debate
    debate = client.debates.create(
        task="What is the best programming language for building web APIs?",
        agents=["claude", "gpt-4"],
        rounds=3,
        consensus="majority",
    )

    debate_id = debate["debate_id"]
    print(f"Debate created: {debate_id}")
    print(f"Status: {debate['status']}")

    # Poll for completion
    current = debate
    while current.get("status") in ("running", "pending"):
        time.sleep(2)
        current = client.debates.get(debate_id)
        print(f"Status: {current['status']}")

    # Get results
    if current.get("status") == "completed":
        consensus = current.get("consensus", {})
        print("\n--- Debate Results ---")
        print(f"Final Answer: {consensus.get('final_answer', 'N/A')}")
        print(f"Confidence: {consensus.get('confidence', 'N/A')}")
        print(f"Rounds: {len(current.get('rounds', []))}")

        # Get messages
        messages = client.debates.get_messages(debate_id)
        for msg in messages.get("messages", [])[:5]:
            print(f"\n[{msg['agent']}]: {msg['content'][:200]}...")


if __name__ == "__main__":
    main()
