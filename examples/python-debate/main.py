#!/usr/bin/env python3
"""
Aragora Python SDK Demo CLI

A command-line tool demonstrating the Aragora Python SDK capabilities.

Usage:
    python main.py debate "Your topic here"
    python main.py stream "Your topic here"
    python main.py gauntlet document.md --persona gdpr
    python main.py rankings
    python main.py tournament --name "Q1 Showdown" --agents claude gpt gemini
    python main.py onboarding
    python main.py login --email user@example.com

Requirements:
    pip install aragora-client python-dotenv
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora_client import (
    AragoraClient,
    AragoraError,
    AragoraAuthenticationError,
    AragoraConnectionError,
)
from aragora_client.websocket import stream_debate


async def get_client_async(base_url: str = "http://localhost:8080") -> AragoraClient:
    """Create an async Aragora client."""
    return AragoraClient(base_url=base_url)


# =============================================================================
# Debate Commands
# =============================================================================


async def cmd_debate_async(args: argparse.Namespace) -> int:
    """Run a debate and wait for completion."""
    async with AragoraClient(base_url=args.server) as client:
        print("\nCreating debate...")
        print(f"  Topic: {args.topic}")
        print(f"  Agents: {', '.join(args.agents)}")
        print(f"  Rounds: {args.rounds}")

        try:
            result = await client.debates.run(
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
            for line in answer.split("\n"):
                print(f"  {line}")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_debate(args: argparse.Namespace) -> int:
    """Run a debate (sync wrapper)."""
    return asyncio.run(cmd_debate_async(args))


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


# =============================================================================
# Gauntlet Commands
# =============================================================================


async def cmd_gauntlet_async(args: argparse.Namespace) -> int:
    """Run Gauntlet validation on a document."""
    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: File not found: {args.document}", file=sys.stderr)
        return 1

    content = doc_path.read_text()

    print("\nRunning Gauntlet validation...")
    print(f"  Document: {args.document}")
    print(f"  Persona: {args.persona}")
    print(f"  Profile: {args.profile}")

    async with AragoraClient(base_url=args.server) as client:
        try:
            receipt = await client.gauntlet.run_and_wait(
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

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1
        except TimeoutError:
            print(f"\nError: Gauntlet timed out after {args.timeout}s", file=sys.stderr)
            return 1


def cmd_gauntlet(args: argparse.Namespace) -> int:
    """Run Gauntlet validation (sync wrapper)."""
    return asyncio.run(cmd_gauntlet_async(args))


# =============================================================================
# Rankings Commands
# =============================================================================


async def cmd_rankings_async(args: argparse.Namespace) -> int:
    """Display agent rankings."""
    print("\nAgent Rankings:")
    print("-" * 50)

    async with AragoraClient(base_url=args.server) as client:
        try:
            rankings = await client.leaderboard.list(limit=args.limit)

            if not rankings:
                print("  No agents found.")
                return 0

            for i, agent in enumerate(rankings, 1):
                elo = agent.elo or 1500
                wins = agent.wins or 0
                losses = agent.losses or 0
                print(f"  {i:2}. {agent.name:<20} {elo:>6.0f} ELO  ({wins}W/{losses}L)")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_rankings(args: argparse.Namespace) -> int:
    """Display agent rankings (sync wrapper)."""
    return asyncio.run(cmd_rankings_async(args))


# =============================================================================
# Tournament Commands
# =============================================================================


async def cmd_tournament_async(args: argparse.Namespace) -> int:
    """Create and run a tournament."""
    print("\nCreating tournament...")
    print(f"  Name: {args.name}")
    print(f"  Participants: {', '.join(args.agents)}")
    print(f"  Format: {args.format}")

    async with AragoraClient(base_url=args.server) as client:
        try:
            tournament = await client.tournaments.create(
                name=args.name,
                participants=args.agents,
                format=args.format,
                task_template=args.task
                or "Discuss the merits of {participant_a} vs {participant_b}",
            )

            print(f"\nTournament created: {tournament.id}")
            print(f"  Status: {tournament.status}")
            print(f"  Total matches: {tournament.total_matches}")

            if args.wait:
                print("\nWaiting for tournament completion...")
                while tournament.status != "completed":
                    await asyncio.sleep(5)
                    tournament = await client.tournaments.get(tournament.id)
                    completed = tournament.completed_matches or 0
                    total = tournament.total_matches or 0
                    print(f"  Progress: {completed}/{total} matches")

                print("\nFinal Standings:")
                standings = await client.tournaments.get_standings(tournament.id)
                for i, standing in enumerate(standings.standings[:5], 1):
                    print(f"  {i}. {standing.participant} - {standing.wins}W/{standing.losses}L")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_tournament(args: argparse.Namespace) -> int:
    """Create and run a tournament (sync wrapper)."""
    return asyncio.run(cmd_tournament_async(args))


async def cmd_tournament_list_async(args: argparse.Namespace) -> int:
    """List tournaments."""
    async with AragoraClient(base_url=args.server) as client:
        try:
            tournaments = await client.tournaments.list(limit=args.limit)

            if not tournaments:
                print("\nNo tournaments found.")
                return 0

            print("\nTournaments:")
            print("-" * 60)
            for t in tournaments:
                print(f"  {t.id[:8]}  {t.name:<30} {t.status}")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_tournament_list(args: argparse.Namespace) -> int:
    """List tournaments (sync wrapper)."""
    return asyncio.run(cmd_tournament_list_async(args))


# =============================================================================
# Authentication Commands
# =============================================================================


async def cmd_login_async(args: argparse.Namespace) -> int:
    """Login to Aragora."""
    email = args.email
    password = args.password or getpass.getpass("Password: ")

    print(f"\nLogging in as {email}...")

    async with AragoraClient(base_url=args.server) as client:
        try:
            token = await client.auth.login(email, password)
            print("\nLogin successful!")
            print(f"  User ID: {token.user_id}")
            print(f"  Token expires: {token.expires_at}")

            # Save token for future use
            token_file = Path.home() / ".aragora" / "token"
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(token.access_token)
            print(f"\n  Token saved to {token_file}")

            return 0

        except AragoraAuthenticationError as e:
            print(f"\nLogin failed: {e}", file=sys.stderr)
            return 1
        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_login(args: argparse.Namespace) -> int:
    """Login to Aragora (sync wrapper)."""
    return asyncio.run(cmd_login_async(args))


async def cmd_logout_async(args: argparse.Namespace) -> int:
    """Logout from Aragora."""
    async with AragoraClient(base_url=args.server) as client:
        try:
            await client.auth.logout()
            print("\nLogged out successfully.")

            token_file = Path.home() / ".aragora" / "token"
            if token_file.exists():
                token_file.unlink()
                print("  Local token removed.")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_logout(args: argparse.Namespace) -> int:
    """Logout from Aragora (sync wrapper)."""
    return asyncio.run(cmd_logout_async(args))


async def cmd_apikey_list_async(args: argparse.Namespace) -> int:
    """List API keys."""
    async with AragoraClient(base_url=args.server) as client:
        try:
            keys = await client.auth.list_api_keys()

            if not keys:
                print("\nNo API keys found.")
                return 0

            print("\nAPI Keys:")
            print("-" * 60)
            for key in keys:
                status = "active" if key.is_active else "revoked"
                print(f"  {key.id[:8]}  {key.name:<20} [{status}]")
                if key.last_used_at:
                    print(f"           Last used: {key.last_used_at}")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_apikey_list(args: argparse.Namespace) -> int:
    """List API keys (sync wrapper)."""
    return asyncio.run(cmd_apikey_list_async(args))


async def cmd_apikey_create_async(args: argparse.Namespace) -> int:
    """Create an API key."""
    print(f"\nCreating API key: {args.name}")

    async with AragoraClient(base_url=args.server) as client:
        try:
            result = await client.auth.create_api_key(
                name=args.name,
                scopes=args.scopes,
                expires_in_days=args.expires,
            )

            print("\nAPI key created!")
            print(f"  Name: {result.key.name}")
            print(f"  ID: {result.key.id}")
            print(f"  Scopes: {', '.join(result.key.scopes or ['all'])}")
            print(f"\n  SECRET KEY: {result.secret}")
            print("\n  ⚠️  Save this key now - it won't be shown again!")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_apikey_create(args: argparse.Namespace) -> int:
    """Create an API key (sync wrapper)."""
    return asyncio.run(cmd_apikey_create_async(args))


# =============================================================================
# Onboarding Commands
# =============================================================================


async def cmd_onboarding_async(args: argparse.Namespace) -> int:
    """Start or continue onboarding."""
    async with AragoraClient(base_url=args.server) as client:
        try:
            # Check for existing flow
            flow = await client.onboarding.get_flow()

            if flow is None:
                print("\nStarting onboarding...")
                flow = await client.onboarding.init_flow(template=args.template)
                print(f"  New flow created: {flow.id}")
            else:
                print("\nContinuing onboarding...")
                print(f"  Flow ID: {flow.id}")

            print(f"  Progress: {flow.progress_percent:.0%}")
            print(f"  Status: {flow.status}")

            print("\nSteps:")
            for step in flow.steps:
                status_icon = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "●",
                    "skipped": "⊘",
                }.get(step.status, "?")
                required = "*" if step.required else " "
                print(f"  {status_icon} {step.name}{required}")
                if step.description:
                    print(f"      {step.description[:60]}...")

            if args.complete:
                current = next(
                    (s for s in flow.steps if s.status in ("pending", "in_progress")), None
                )
                if current:
                    print(f"\nCompleting step: {current.name}")
                    flow = await client.onboarding.complete_step(current.id)
                    print(f"  Progress: {flow.progress_percent:.0%}")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_onboarding(args: argparse.Namespace) -> int:
    """Start or continue onboarding (sync wrapper)."""
    return asyncio.run(cmd_onboarding_async(args))


# =============================================================================
# Health Commands
# =============================================================================


async def cmd_health_async(args: argparse.Namespace) -> int:
    """Check server health."""
    print(f"\nChecking server at {args.server}...")

    async with AragoraClient(base_url=args.server) as client:
        try:
            health = await client.health.check()
            print(f"  Status: {health.status}")
            print(f"  Version: {health.version or 'unknown'}")
            return 0

        except AragoraConnectionError as e:
            print(f"\nError: Server not reachable - {e}", file=sys.stderr)
            return 1
        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Check server health (sync wrapper)."""
    return asyncio.run(cmd_health_async(args))


# =============================================================================
# Memory Analytics Commands
# =============================================================================


async def cmd_memory_async(args: argparse.Namespace) -> int:
    """View memory analytics."""
    async with AragoraClient(base_url=args.server) as client:
        try:
            analytics = await client.memory.get_analytics()

            print("\nMemory Analytics:")
            print("-" * 50)
            print(f"  Total entries: {analytics.get('total_entries', 0):,}")
            print(f"  Active debates: {analytics.get('active_debates', 0)}")
            print(f"  Consensuses: {analytics.get('total_consensuses', 0)}")

            tiers = analytics.get("tiers", {})
            if tiers:
                print("\n  Memory Tiers:")
                for tier, count in tiers.items():
                    print(f"    {tier}: {count:,} entries")

            return 0

        except AragoraError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1


def cmd_memory(args: argparse.Namespace) -> int:
    """View memory analytics (sync wrapper)."""
    return asyncio.run(cmd_memory_async(args))


# =============================================================================
# Main Entry Point
# =============================================================================


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

    # tournament commands
    tournament_parser = subparsers.add_parser("tournament", help="Tournament commands")
    tournament_sub = tournament_parser.add_subparsers(dest="tournament_command")

    # tournament create
    create_parser = tournament_sub.add_parser("create", help="Create a tournament")
    create_parser.add_argument("--name", required=True, help="Tournament name")
    create_parser.add_argument("--agents", nargs="+", required=True, help="Participating agents")
    create_parser.add_argument(
        "--format",
        default="single_elimination",
        choices=["single_elimination", "double_elimination", "round_robin"],
        help="Tournament format",
    )
    create_parser.add_argument("--task", help="Task template for matches")
    create_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    create_parser.set_defaults(func=cmd_tournament)

    # tournament list
    list_parser = tournament_sub.add_parser("list", help="List tournaments")
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=cmd_tournament_list)

    # auth commands
    auth_parser = subparsers.add_parser("auth", help="Authentication commands")
    auth_sub = auth_parser.add_subparsers(dest="auth_command")

    # auth login
    login_parser = auth_sub.add_parser("login", help="Login to Aragora")
    login_parser.add_argument("--email", required=True, help="Email address")
    login_parser.add_argument("--password", help="Password (prompted if not provided)")
    login_parser.set_defaults(func=cmd_login)

    # auth logout
    logout_parser = auth_sub.add_parser("logout", help="Logout from Aragora")
    logout_parser.set_defaults(func=cmd_logout)

    # auth apikeys
    apikey_parser = auth_sub.add_parser("apikeys", help="Manage API keys")
    apikey_sub = apikey_parser.add_subparsers(dest="apikey_command")

    apikey_list_parser = apikey_sub.add_parser("list", help="List API keys")
    apikey_list_parser.set_defaults(func=cmd_apikey_list)

    apikey_create_parser = apikey_sub.add_parser("create", help="Create an API key")
    apikey_create_parser.add_argument("--name", required=True, help="Key name")
    apikey_create_parser.add_argument("--scopes", nargs="*", help="Permission scopes")
    apikey_create_parser.add_argument("--expires", type=int, default=365, help="Expiry in days")
    apikey_create_parser.set_defaults(func=cmd_apikey_create)

    # onboarding command
    onboarding_parser = subparsers.add_parser("onboarding", help="User onboarding")
    onboarding_parser.add_argument(
        "--template",
        default="default",
        help="Onboarding template",
    )
    onboarding_parser.add_argument(
        "--complete",
        action="store_true",
        help="Complete the current step",
    )
    onboarding_parser.set_defaults(func=cmd_onboarding)

    # health command
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.set_defaults(func=cmd_health)

    # memory command
    memory_parser = subparsers.add_parser("memory", help="View memory analytics")
    memory_parser.set_defaults(func=cmd_memory)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Handle nested subcommands
    if args.command == "tournament" and not getattr(args, "tournament_command", None):
        tournament_parser.print_help()
        return 1
    if args.command == "auth" and not getattr(args, "auth_command", None):
        auth_parser.print_help()
        return 1
    if (
        args.command == "auth"
        and args.auth_command == "apikeys"
        and not getattr(args, "apikey_command", None)
    ):
        apikey_parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
