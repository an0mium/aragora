#!/usr/bin/env python3
"""
Tournament & Leaderboard Example
================================

This example demonstrates Aragora's competitive ranking system:
- Multiple agents compete across different debate topics
- ELO ratings track agent skill over time
- Leaderboard shows which AI performs best

Time: ~10-30 minutes (depends on number of tasks)
Requirements: At least 2 API keys

Usage:
    python examples/02_tournament.py

Expected output:
    === ARAGORA TOURNAMENT ===
    Format: round_robin
    Agents: grok, gemini, anthropic-api
    Tasks: 2

    Match 1/3: grok vs gemini
      Topic: API Design
      Winner: grok
    ...

    === FINAL STANDINGS ===
    1. gemini      | Wins: 2 | Points: 6.0 | ELO: 1532
    2. grok        | Wins: 1 | Points: 3.0 | ELO: 1508
    3. anthropic   | Wins: 0 | Points: 0.0 | ELO: 1492
"""

import asyncio
import sys
from pathlib import Path

# Add aragora to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import (
    Arena,
    Environment,
    DebateProtocol,
    Tournament,
    TournamentFormat,
    TournamentTask,
    EloSystem,
)
from aragora.agents.base import create_agent
from aragora.core import Agent, DebateResult


def create_agents():
    """Create available agents based on API keys."""
    agent_types = [
        ("grok", "proposer"),
        ("gemini", "critic"),
        ("anthropic-api", "synthesizer"),
        ("openai-api", "critic"),
    ]

    agents = []
    for agent_type, role in agent_types:
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore
                name=agent_type,
                role=role,
            )
            agents.append(agent)
            print(f"  + {agent_type} ready")
        except Exception:
            pass

    return agents


def create_tournament_tasks():
    """Create a small set of tasks for the demo tournament."""
    return [
        TournamentTask(
            task_id="api-design",
            description="""Design a REST API for a todo list application.
Include: endpoints, data models, authentication approach, and error handling.""",
            domain="api_design",
            difficulty=0.5,
            time_limit=120,
        ),
        TournamentTask(
            task_id="error-handling",
            description="""Design an error handling strategy for a payment processing system.
Consider: retry logic, idempotency, user feedback, and logging.""",
            domain="error_handling",
            difficulty=0.5,
            time_limit=120,
        ),
    ]


async def run_debate(env: Environment, agents: list[Agent]) -> DebateResult:
    """Run a single debate for tournament scoring."""
    protocol = DebateProtocol(
        rounds=2,
        consensus="majority",
        early_stopping=True,
    )
    arena = Arena(env, agents, protocol)
    return await arena.run()


async def run_tournament():
    """Run a round-robin tournament between agents."""

    print("\n" + "=" * 60)
    print("ARAGORA TOURNAMENT")
    print("=" * 60)

    print("\nInitializing agents...")
    agents = create_agents()

    if len(agents) < 2:
        print("\nError: Need at least 2 agents for tournament")
        print("Set at least two of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY")
        return None

    tasks = create_tournament_tasks()

    # Use in-memory ELO for demo (won't persist between runs)
    elo = EloSystem(db_path=":memory:")

    print(f"\nCreating tournament...")
    tournament = Tournament(
        name="Aragora Demo Tournament",
        agents=agents,
        tasks=tasks,
        format=TournamentFormat.ROUND_ROBIN,
        elo_system=elo,
        db_path=":memory:",  # In-memory for demo
    )

    print(f"Format: {tournament.format.value}")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Tasks: {len(tasks)}")

    # Calculate expected matches
    n_agents = len(agents)
    expected_matches = (n_agents * (n_agents - 1) // 2) * len(tasks)
    print(f"Expected matches: {expected_matches}")

    # Run tournament
    print(f"\n{'='*60}")
    print("RUNNING MATCHES")
    print(f"{'='*60}")

    result = await tournament.run(
        run_debate_fn=run_debate,
        parallel=False,  # Sequential for clearer output
    )

    # Display results
    print(f"\n{'='*60}")
    print("FINAL STANDINGS")
    print(f"{'='*60}")
    print(f"Champion: {result.champion}")
    print(f"Total matches: {len(result.matches)}")
    print(f"\nLeaderboard:")

    for i, standing in enumerate(result.standings, 1):
        rating = elo.get_rating(standing.agent_name)
        rating_value = rating.rating if rating else 1500
        print(
            f"  {i}. {standing.agent_name:15} | "
            f"Wins: {standing.wins} | "
            f"Points: {standing.points:.1f} | "
            f"ELO: {rating_value:.0f}"
        )

    # Show match history
    print(f"\nMatch History:")
    for match in result.matches:
        winner = match.winner or "Draw"
        print(f"  - {' vs '.join(match.participants)}: {winner}")

    return result


if __name__ == "__main__":
    print("Aragora Tournament Example")
    print("This demonstrates competitive AI debates with ELO ranking.")

    result = asyncio.run(run_tournament())

    if result:
        print(f"\n[SUCCESS] Tournament completed! Champion: {result.champion}")
    else:
        print("\n[ERROR] Tournament could not run - check API keys")
