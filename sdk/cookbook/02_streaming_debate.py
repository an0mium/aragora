#!/usr/bin/env python3
"""
02_streaming_debate.py - Real-time WebSocket streaming with reconnection.

This example shows how to stream debate events in real-time using WebSockets.
Includes automatic reconnection logic and event handlers for all debate phases.

Usage:
    python 02_streaming_debate.py                    # Run with streaming
    python 02_streaming_debate.py --dry-run          # Test without connections
    python 02_streaming_debate.py --topic "Your topic"
"""

import argparse
import asyncio
from aragora_sdk import StreamingClient, DebateConfig, Agent
from aragora_sdk.streaming import EventType, ReconnectPolicy


class DebateEventHandler:
    """Handler for real-time debate events."""

    def __init__(self):
        self.round_count = 0
        self.messages = []

    async def on_debate_start(self, event: dict) -> None:
        """Called when debate begins."""
        print(f"\n[DEBATE START] Topic: {event.get('topic')}")
        print(f"[DEBATE START] Agents: {event.get('agents')}")

    async def on_round_start(self, event: dict) -> None:
        """Called at the beginning of each round."""
        self.round_count += 1
        print(f"\n--- Round {self.round_count} ---")

    async def on_agent_message(self, event: dict) -> None:
        """Called when an agent submits their argument."""
        agent = event.get("agent", "unknown")
        content = event.get("content", "")[:100]  # Truncate for display
        print(f"[{agent}] {content}...")
        self.messages.append(event)

    async def on_critique(self, event: dict) -> None:
        """Called when an agent critiques another's argument."""
        critic = event.get("critic", "unknown")
        target = event.get("target", "unknown")
        print(f"[CRITIQUE] {critic} -> {target}: {event.get('summary', '')[:80]}...")

    async def on_vote(self, event: dict) -> None:
        """Called when voting occurs."""
        print(f"[VOTE] {event.get('agent')}: {event.get('position')}")

    async def on_consensus(self, event: dict) -> None:
        """Called when consensus is reached."""
        print(f"\n[CONSENSUS] Decision: {event.get('decision')}")
        print(f"[CONSENSUS] Confidence: {event.get('confidence', 0):.2%}")

    async def on_debate_end(self, event: dict) -> None:
        """Called when debate concludes."""
        print(f"\n[DEBATE END] Total rounds: {self.round_count}")
        print(f"[DEBATE END] Total messages: {len(self.messages)}")


async def run_streaming_debate(topic: str, dry_run: bool = False) -> dict:
    """Run a debate with real-time streaming events."""

    # Create event handler
    handler = DebateEventHandler()

    # Configure reconnection policy for resilience
    reconnect_policy = ReconnectPolicy(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_backoff=True,
    )

    # Initialize streaming client
    client = StreamingClient(reconnect_policy=reconnect_policy)

    # Define agents
    agents = [
        Agent(name="claude", model="claude-sonnet-4-20250514"),
        Agent(name="gpt", model="gpt-4o"),
        Agent(name="gemini", model="gemini-2.0-flash"),
    ]

    config = DebateConfig(topic=topic, agents=agents, rounds=3)

    if dry_run:
        # Simulate streaming events without actual connection
        print("[DRY RUN] Simulating streaming events...")
        await handler.on_debate_start({"topic": topic, "agents": [a.name for a in agents]})
        await handler.on_round_start({"round": 1})
        await handler.on_agent_message({"agent": "claude", "content": "Mock argument..."})
        await handler.on_debate_end({"status": "completed"})
        return {"status": "dry_run", "events_simulated": 4}

    # Register event handlers
    client.on(EventType.DEBATE_START, handler.on_debate_start)
    client.on(EventType.ROUND_START, handler.on_round_start)
    client.on(EventType.AGENT_MESSAGE, handler.on_agent_message)
    client.on(EventType.CRITIQUE, handler.on_critique)
    client.on(EventType.VOTE, handler.on_vote)
    client.on(EventType.CONSENSUS, handler.on_consensus)
    client.on(EventType.DEBATE_END, handler.on_debate_end)

    # Connect and run with streaming
    async with client.connect() as stream:
        result = await stream.run_debate(config)

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run a streaming debate")
    parser.add_argument(
        "--topic", default="What is the best approach to AI alignment?", help="Topic for the debate"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without connections")
    args = parser.parse_args()

    result = asyncio.run(run_streaming_debate(args.topic, args.dry_run))
    return result


if __name__ == "__main__":
    main()
