#!/usr/bin/env python3
"""
01_simple_debate.py - Basic 3-agent debate example.

This is the simplest way to run a debate using the Aragora SDK.
Three agents (Claude, GPT, Gemini) will debate a topic and reach consensus.

Usage:
    python 01_simple_debate.py                    # Run actual debate
    python 01_simple_debate.py --dry-run          # Test without API calls
    python 01_simple_debate.py --topic "Your topic here"
"""

import argparse
import asyncio
from aragora_sdk import ArenaClient, DebateConfig, Agent


async def run_simple_debate(topic: str, dry_run: bool = False) -> dict:
    """Run a basic 3-agent debate on the given topic."""

    # Initialize the client (uses ARAGORA_API_URL and ARAGORA_API_TOKEN from env)
    client = ArenaClient()

    # Define our three agents - each brings a different perspective
    agents = [
        Agent(name="claude", model="claude-sonnet-4-20250514"),
        Agent(name="gpt", model="gpt-4o"),
        Agent(name="gemini", model="gemini-2.0-flash"),
    ]

    # Configure the debate
    config = DebateConfig(
        topic=topic,
        agents=agents,
        rounds=3,  # Number of debate rounds
        consensus_threshold=0.7,  # 70% agreement needed for consensus
    )

    if dry_run:
        # In dry-run mode, return mock result without API calls
        print(f"[DRY RUN] Would run debate on: {topic}")
        print(f"[DRY RUN] Agents: {[a.name for a in agents]}")
        print(f"[DRY RUN] Rounds: {config.rounds}")
        return {"status": "dry_run", "topic": topic, "agents": [a.name for a in agents]}

    # Run the debate and wait for result
    result = await client.run_debate(config)

    # Print the outcome
    print("\n=== Debate Results ===")
    print(f"Topic: {result.topic}")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Final decision: {result.decision}")
    print(f"Confidence: {result.confidence:.2%}")

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run a simple 3-agent debate")
    parser.add_argument(
        "--topic",
        default="Should AI development prioritize safety over capability?",
        help="Topic for the debate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_simple_debate(args.topic, args.dry_run))
    return result


if __name__ == "__main__":
    main()
