#!/usr/bin/env python3
"""
Aragora Python SDK Demo CLI

A simple command-line tool demonstrating the Aragora Python SDK.

Usage:
    python main.py debate "Your topic here"
    python main.py stream "Your topic here"
    python main.py gauntlet document.md --persona gdpr
    python main.py rankings

Requirements:
    pip install aragora
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora.client import (
    AragoraClient,
    AragoraAPIError,
    stream_debate,
)


def get_client(base_url: str = "http://localhost:8080") -> AragoraClient:
    """Create an Aragora client."""
    return AragoraClient(base_url=base_url)


def cmd_debate(args: argparse.Namespace) -> int:
    """Run a debate and wait for completion."""
    client = get_client(args.server)

    print("\nCreating debate...")
    print(f"  Topic: {args.topic}")
    print(f"  Agents: {', '.join(args.agents)}")
    print(f"  Rounds: {args.rounds}")

    try:
        result = client.debates.run(
            task=args.topic,
            agents=args.agents,
            rounds=args.rounds,
        )

        print("\nResults:")
        print(f"  Consensus: {'Yes' if result.consensus.reached else 'No'}")
        print(f"  Confidence: {result.consensus.confidence:.1%}")
        print(f"  Rounds completed: {result.rounds_completed}")

        print("\nFinal Answer:")
        answer = result.consensus.final_answer or "No answer generated"
        # Wrap text for readability
        for line in answer.split("\n"):
            print(f"  {line}")

        return 0

    except AragoraAPIError as e:
        print(f"\nError: {e.message}", file=sys.stderr)
        return 1


async def cmd_stream_async(args: argparse.Namespace) -> int:
    """Stream a debate in real-time."""
    print("\nStarting real-time debate stream...")
    print(f"  Topic: {args.topic}")
    print(f"  Agents: {', '.join(args.agents)}")
    print()

    try:
        async for event in stream_debate(
            base_url=args.server,
            task=args.topic,
            agents=args.agents,
            rounds=args.rounds,
        ):
            if event.type == "debate_start":
                print(f"[STARTED] Debate ID: {event.debate_id}")
            elif event.type == "round_start":
                print(f"\n[ROUND {event.round}]")
            elif event.type == "agent_message":
                agent = event.agent or "Unknown"
                content = (event.content or "")[:200]
                print(f"  [{agent}]: {content}...")
            elif event.type == "critique":
                critic = event.critic or "Unknown"
                print(f"  [CRITIQUE by {critic}]: {event.summary or ''}")
            elif event.type == "consensus":
                print(f"\n[CONSENSUS] {event.data}")
            elif event.type == "debate_end":
                print("\n[COMPLETED]")

        return 0

    except Exception as e:
        print(f"\nStream error: {e}", file=sys.stderr)
        return 1


def cmd_stream(args: argparse.Namespace) -> int:
    """Wrapper for async stream command."""
    return asyncio.run(cmd_stream_async(args))


def cmd_gauntlet(args: argparse.Namespace) -> int:
    """Run Gauntlet validation on a document."""
    client = get_client(args.server)

    # Read document
    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: File not found: {args.document}", file=sys.stderr)
        return 1

    content = doc_path.read_text()

    print("\nRunning Gauntlet validation...")
    print(f"  Document: {args.document}")
    print(f"  Persona: {args.persona}")
    print(f"  Profile: {args.profile}")

    try:
        receipt = client.gauntlet.run_and_wait(
            input_content=content,
            input_type=args.type,
            persona=args.persona,
            profile=args.profile,
            timeout=args.timeout,
        )

        print("\nResults:")
        print(f"  Verdict: {receipt.verdict}")
        print(f"  Risk Score: {receipt.risk_score}")

        if receipt.findings:
            print(f"\nFindings ({len(receipt.findings)}):")
            for finding in receipt.findings:
                print(f"  [{finding.severity}] {finding.title}")
                if finding.description:
                    print(f"    {finding.description[:100]}...")

        return 0

    except AragoraAPIError as e:
        print(f"\nError: {e.message}", file=sys.stderr)
        return 1
    except TimeoutError:
        print(f"\nError: Gauntlet timed out after {args.timeout}s", file=sys.stderr)
        return 1


def cmd_rankings(args: argparse.Namespace) -> int:
    """Display agent rankings."""
    client = get_client(args.server)

    print("\nAgent Rankings:")
    print("-" * 50)

    try:
        rankings = client.leaderboard.list(limit=args.limit)

        if not rankings:
            print("  No agents found.")
            return 0

        for i, agent in enumerate(rankings, 1):
            elo = agent.elo or 1500
            wins = agent.wins or 0
            losses = agent.losses or 0
            print(f"  {i:2}. {agent.name:<20} {elo:>6.0f} ELO  ({wins}W/{losses}L)")

        return 0

    except AragoraAPIError as e:
        print(f"\nError: {e.message}", file=sys.stderr)
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Check server health."""
    client = get_client(args.server)

    print(f"\nChecking server at {args.server}...")

    try:
        health = client.system.health()
        print(f"  Status: {health.status}")
        print(f"  Version: {health.version or 'unknown'}")
        return 0

    except AragoraAPIError as e:
        print(f"\nError: Server not reachable - {e.message}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aragora Python SDK Demo CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080",
        help="Aragora server URL (default: http://localhost:8080)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # debate command
    debate_parser = subparsers.add_parser("debate", help="Run a debate")
    debate_parser.add_argument("topic", help="Debate topic")
    debate_parser.add_argument(
        "--agents",
        nargs="+",
        default=["anthropic-api", "openai-api"],
        help="Agents to use (default: anthropic-api openai-api)",
    )
    debate_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds (default: 3)",
    )
    debate_parser.set_defaults(func=cmd_debate)

    # stream command
    stream_parser = subparsers.add_parser("stream", help="Stream a debate in real-time")
    stream_parser.add_argument("topic", help="Debate topic")
    stream_parser.add_argument(
        "--agents",
        nargs="+",
        default=["anthropic-api", "openai-api"],
        help="Agents to use",
    )
    stream_parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of rounds",
    )
    stream_parser.set_defaults(func=cmd_stream)

    # gauntlet command
    gauntlet_parser = subparsers.add_parser("gauntlet", help="Run Gauntlet validation")
    gauntlet_parser.add_argument("document", help="Document to validate")
    gauntlet_parser.add_argument(
        "--persona",
        default="security",
        help="Validation persona (default: security)",
    )
    gauntlet_parser.add_argument(
        "--profile",
        default="default",
        choices=["quick", "default", "thorough"],
        help="Validation depth (default: default)",
    )
    gauntlet_parser.add_argument(
        "--type",
        default="policy",
        choices=["text", "policy", "code", "spec"],
        help="Document type (default: policy)",
    )
    gauntlet_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )
    gauntlet_parser.set_defaults(func=cmd_gauntlet)

    # rankings command
    rankings_parser = subparsers.add_parser("rankings", help="Show agent rankings")
    rankings_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of agents to show (default: 10)",
    )
    rankings_parser.set_defaults(func=cmd_rankings)

    # health command
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.set_defaults(func=cmd_health)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
