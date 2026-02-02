"""
Streaming Debate Example

Demonstrates real-time debate streaming using WebSockets.
Events are received as they happen during the debate.

Usage:
    python examples/streaming_debate.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os

from aragora_sdk import AragoraAsyncClient


async def main() -> None:
    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Create a debate
        print("Creating debate...")
        debate = await client.debates.create(
            task="Should AI systems be required to explain their decisions?",
            agents=["claude", "gpt-4", "gemini"],
            rounds=2,
            consensus="weighted",
        )

        debate_id = debate["debate_id"]
        print(f"Debate created: {debate_id}")

        # Connect to WebSocket stream
        ws = client.stream.connect()
        await ws.open()
        print("WebSocket connected")

        # Subscribe to debate events
        await ws.subscribe(debate_id)

        # Process events as they arrive
        async for event in ws.events():
            if event.type == "debate_start":
                print("\n=== Debate Started ===")
                print(f"Task: {event.data['task']}")
                print(f"Agents: {', '.join(event.data.get('agents', []))}")

            elif event.type == "round_start":
                print(f"\n--- Round {event.data['round_number']} ---")

            elif event.type == "agent_message":
                agent = event.data["agent"]
                content = event.data["content"][:200]
                print(f"\n[{agent}]: {content}...")

            elif event.type == "critique":
                critic = event.data["critic"]
                target = event.data["target"]
                score = event.data.get("score", "N/A")
                print(f"\n[{critic} critiques {target}] Score: {score}/10")

            elif event.type == "vote":
                voter = event.data["agent"]
                choice = event.data["vote"]
                print(f"Vote: {voter} -> {choice}")

            elif event.type == "consensus":
                answer = event.data.get("final_answer", "N/A")
                confidence = event.data.get("confidence", 0)
                print("\n=== Consensus Reached ===")
                print(f"Answer: {answer}")
                print(f"Confidence: {confidence:.1%}")

            elif event.type == "debate_end":
                print(f"\nDebate ended: {event.data.get('status', 'unknown')}")
                break

        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
