"""
Real-Time Collaboration Example

Demonstrates WebSocket streaming for real-time debate interaction.
Shows how to receive events, handle user voting, and submit suggestions.

Usage:
    python examples/realtime_collaboration.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aragora_sdk import AragoraAsyncClient

# =============================================================================
# Event Types
# =============================================================================


@dataclass
class DebateEvent:
    """Parsed debate event."""

    type: str
    timestamp: datetime
    data: dict[str, Any]

    @classmethod
    def from_raw(cls, raw: Any) -> DebateEvent:
        """Create from raw WebSocket event."""
        return cls(
            type=raw.type,
            timestamp=datetime.now(),
            data=raw.data,
        )


# =============================================================================
# Event Handlers
# =============================================================================


class DebateEventHandler:
    """Handler for debate events with callbacks."""

    def __init__(self) -> None:
        self.events: list[DebateEvent] = []
        self.agent_messages: list[dict[str, Any]] = []
        self.votes: list[dict[str, Any]] = []

    async def handle_event(self, event: DebateEvent) -> None:
        """Route event to appropriate handler."""
        self.events.append(event)

        handler = getattr(self, f"on_{event.type}", self.on_unknown)
        await handler(event)

    async def on_debate_start(self, event: DebateEvent) -> None:
        """Handle debate start."""
        print("\n" + "=" * 50)
        print("DEBATE STARTED")
        print("=" * 50)
        print(f"Task: {event.data.get('task', 'N/A')}")
        print(f"Agents: {', '.join(event.data.get('agents', []))}")
        print(f"Rounds: {event.data.get('rounds', 'N/A')}")

    async def on_round_start(self, event: DebateEvent) -> None:
        """Handle round start."""
        round_num = event.data.get("round_number", "?")
        print(f"\n--- Round {round_num} ---")

    async def on_agent_message(self, event: DebateEvent) -> None:
        """Handle agent message."""
        agent = event.data.get("agent", "Unknown")
        content = event.data.get("content", "")
        role = event.data.get("role", "response")

        self.agent_messages.append(event.data)

        # Truncate for display
        preview = content[:150] + "..." if len(content) > 150 else content
        print(f"\n[{agent}] ({role}):")
        print(f"  {preview}")

    async def on_critique(self, event: DebateEvent) -> None:
        """Handle critique."""
        critic = event.data.get("critic", "Unknown")
        target = event.data.get("target", "Unknown")
        score = event.data.get("score", "N/A")
        feedback = event.data.get("feedback", "")[:100]

        print(f"\n[CRITIQUE] {critic} -> {target}")
        print(f"  Score: {score}/10")
        print(f"  Feedback: {feedback}...")

    async def on_vote(self, event: DebateEvent) -> None:
        """Handle vote."""
        voter = event.data.get("agent", event.data.get("user", "Unknown"))
        choice = event.data.get("vote", "N/A")
        confidence = event.data.get("confidence", 0)

        self.votes.append(event.data)
        print(f"\n[VOTE] {voter} -> {choice} (confidence: {confidence:.0%})")

    async def on_user_suggestion(self, event: DebateEvent) -> None:
        """Handle user suggestion."""
        user = event.data.get("user_id", "Anonymous")
        suggestion = event.data.get("suggestion", "")[:100]

        print(f"\n[USER SUGGESTION] from {user}:")
        print(f"  {suggestion}...")

    async def on_consensus(self, event: DebateEvent) -> None:
        """Handle consensus reached."""
        answer = event.data.get("final_answer", "N/A")
        confidence = event.data.get("confidence", 0)
        method = event.data.get("method", "N/A")

        print("\n" + "=" * 50)
        print("CONSENSUS REACHED")
        print("=" * 50)
        print(f"Answer: {answer[:200]}...")
        print(f"Confidence: {confidence:.1%}")
        print(f"Method: {method}")

    async def on_debate_end(self, event: DebateEvent) -> None:
        """Handle debate end."""
        status = event.data.get("status", "unknown")
        duration = event.data.get("duration_seconds", 0)

        print("\n" + "=" * 50)
        print(f"DEBATE ENDED: {status}")
        print(f"Duration: {duration:.1f}s")
        print(f"Total events: {len(self.events)}")
        print(f"Agent messages: {len(self.agent_messages)}")
        print(f"Votes: {len(self.votes)}")
        print("=" * 50)

    async def on_unknown(self, event: DebateEvent) -> None:
        """Handle unknown event type."""
        print(f"\n[{event.type}] {event.data}")


# =============================================================================
# Basic Streaming
# =============================================================================


async def basic_streaming(client: AragoraAsyncClient) -> None:
    """Basic WebSocket streaming example."""
    print("=== Basic WebSocket Streaming ===\n")

    # Create a debate
    print("Creating debate...")
    debate = await client.debates.create(
        task="What is the most important skill for a software engineer?",
        agents=["claude", "gpt-4"],
        rounds=2,
    )

    debate_id = debate["debate_id"]
    print(f"Debate created: {debate_id}")

    # Connect to WebSocket
    ws = client.stream.connect()
    await ws.open()
    print("WebSocket connected")

    # Subscribe to debate events
    await ws.subscribe(debate_id)
    print(f"Subscribed to debate {debate_id}")

    # Create event handler
    handler = DebateEventHandler()

    # Process events
    try:
        async for raw_event in ws.events():
            event = DebateEvent.from_raw(raw_event)
            await handler.handle_event(event)

            # Exit on debate end
            if event.type == "debate_end":
                break

    finally:
        await ws.close()
        print("\nWebSocket closed")


# =============================================================================
# User Voting
# =============================================================================


async def user_voting(client: AragoraAsyncClient) -> None:
    """Demonstrate user voting during a debate."""
    print("\n=== User Voting ===\n")

    # Create a debate with voting enabled
    debate = await client.debates.create(
        task="Should code reviews be mandatory for all pull requests?",
        agents=["claude", "gpt-4"],
        rounds=2,
        options={
            "allow_user_votes": True,
            "voting_window_seconds": 30,
        },
    )

    debate_id = debate["debate_id"]
    print(f"Debate created: {debate_id}")

    ws = client.stream.connect()
    await ws.open()
    await ws.subscribe(debate_id)

    print("Watching for voting opportunities...")

    async for event in ws.events():
        if event.type == "voting_open":
            # Voting window is open
            print("\n>>> VOTING WINDOW OPEN <<<")
            print(f"Vote deadline: {event.data.get('deadline', 'N/A')}")

            # Submit a vote
            print("Submitting vote for 'yes'...")
            await ws.send_vote(
                debate_id=debate_id,
                vote="yes",
                confidence=0.8,
                reason="Code reviews improve code quality",
            )
            print("Vote submitted!")

        elif event.type == "vote_received":
            voter = event.data.get("voter", "Unknown")
            vote = event.data.get("vote", "N/A")
            print(f"Vote received: {voter} -> {vote}")

        elif event.type == "debate_end":
            break

    await ws.close()


# =============================================================================
# User Suggestions
# =============================================================================


async def user_suggestions(client: AragoraAsyncClient) -> None:
    """Demonstrate submitting user suggestions during a debate."""
    print("\n=== User Suggestions ===\n")

    # Create a debate with suggestions enabled
    debate = await client.debates.create(
        task="How can we improve developer productivity?",
        agents=["claude", "gpt-4"],
        rounds=3,
        options={
            "allow_user_suggestions": True,
        },
    )

    debate_id = debate["debate_id"]
    print(f"Debate created: {debate_id}")

    ws = client.stream.connect()
    await ws.open()
    await ws.subscribe(debate_id)

    suggestion_sent = False

    async for event in ws.events():
        if event.type == "round_start" and not suggestion_sent:
            # Send a suggestion at the start of a round
            print("\nSubmitting suggestion...")
            await ws.send_suggestion(
                debate_id=debate_id,
                suggestion="Consider the impact of meeting-free days on productivity",
                context="Studies show developers need 4+ hours of uninterrupted time",
            )
            suggestion_sent = True
            print("Suggestion submitted!")

        elif event.type == "suggestion_acknowledged":
            print(f"Suggestion acknowledged: {event.data.get('status', 'N/A')}")

        elif event.type == "agent_message":
            # Check if suggestion was incorporated
            content = event.data.get("content", "")
            if "meeting" in content.lower() or "uninterrupted" in content.lower():
                print(">>> Agent may have incorporated your suggestion!")

        elif event.type == "debate_end":
            break

    await ws.close()


# =============================================================================
# Multiple Subscriptions
# =============================================================================


async def multiple_subscriptions(client: AragoraAsyncClient) -> None:
    """Monitor multiple debates simultaneously."""
    print("\n=== Multiple Debate Subscriptions ===\n")

    # Create multiple debates
    debates = []
    for i in range(3):
        debate = await client.debates.create(
            task=f"Question {i + 1}: What is important?",
            agents=["claude"],
            rounds=1,
        )
        debates.append(debate)
        print(f"Created debate {i + 1}: {debate['debate_id']}")

    # Connect and subscribe to all
    ws = client.stream.connect()
    await ws.open()

    for debate in debates:
        await ws.subscribe(debate["debate_id"])

    print(f"\nSubscribed to {len(debates)} debates")
    print("Monitoring all debates...\n")

    completed = 0
    async for event in ws.events():
        debate_id = event.data.get("debate_id", "unknown")[:8]

        if event.type == "agent_message":
            agent = event.data.get("agent", "?")
            print(f"[{debate_id}...] {agent}: message received")

        elif event.type == "debate_end":
            completed += 1
            print(f"[{debate_id}...] Debate ended ({completed}/{len(debates)})")

            if completed >= len(debates):
                break

    await ws.close()
    print("\nAll debates completed")


# =============================================================================
# Event Filtering
# =============================================================================


async def event_filtering(client: AragoraAsyncClient) -> None:
    """Demonstrate filtering events by type."""
    print("\n=== Event Filtering ===\n")

    debate = await client.debates.create(
        task="What is the best approach?",
        agents=["claude", "gpt-4"],
        rounds=2,
    )

    debate_id = debate["debate_id"]
    print(f"Debate: {debate_id}")

    ws = client.stream.connect()
    await ws.open()

    # Subscribe with event filter
    await ws.subscribe(
        debate_id,
        events=["agent_message", "consensus", "debate_end"],  # Only these events
    )

    print("Filtering for: agent_message, consensus, debate_end")

    async for event in ws.events():
        print(f"  Received: {event.type}")
        if event.type == "debate_end":
            break

    await ws.close()


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run real-time collaboration demonstrations."""
    print("Aragora SDK Real-Time Collaboration")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nReal-time collaboration features:")
        print("  1. WebSocket Streaming: Receive events as they happen")
        print("  2. User Voting: Vote during debates")
        print("  3. User Suggestions: Submit suggestions to influence debates")
        print("  4. Multiple Subscriptions: Monitor multiple debates")
        print("  5. Event Filtering: Subscribe to specific event types")
        print("\nEvent types:")
        print("  - debate_start, debate_end")
        print("  - round_start, round_end")
        print("  - agent_message, critique")
        print("  - vote, consensus")
        print("  - user_suggestion, suggestion_acknowledged")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Basic streaming
        await basic_streaming(client)

        # User voting
        await user_voting(client)

        # User suggestions
        await user_suggestions(client)

        # Multiple subscriptions
        await multiple_subscriptions(client)

        # Event filtering
        await event_filtering(client)

    print("\n" + "=" * 60)
    print("Real-time collaboration complete!")
    print("\nKey Patterns:")
    print("  - Use async for event in ws.events() to iterate")
    print("  - Handle events by type with if/elif or handler class")
    print("  - Always close WebSocket connection when done")
    print("  - Subscribe to specific events to reduce traffic")


if __name__ == "__main__":
    asyncio.run(main())
