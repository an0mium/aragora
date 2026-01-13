"""
CLI demo command - run quick compelling demos.

Extracted from main.py for modularity.
Provides pre-configured debate scenarios for demonstration purposes.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

# Demo task configurations
DEMO_TASKS: dict[str, dict[str, Any]] = {
    "rate-limiter": {
        "task": "Design a distributed rate limiter that handles 1M requests/second across multiple regions",
        "agents": "demo,demo,demo",
        "rounds": 2,
    },
    "auth": {
        "task": "Design a secure authentication system with passwordless login and MFA support",
        "agents": "demo,demo,demo",
        "rounds": 2,
    },
    "cache": {
        "task": "Design a cache invalidation strategy for a social media feed with 100M users",
        "agents": "demo,demo,demo",
        "rounds": 2,
    },
    "consensus": {
        "task": "Design a consensus algorithm for a distributed database with strong consistency",
        "agents": "demo,demo,demo",
        "rounds": 2,
    },
    "search": {
        "task": "Design a real-time search system for an e-commerce platform with 10M products",
        "agents": "demo,demo,demo",
        "rounds": 2,
    },
}


def list_demos() -> list[str]:
    """Return available demo names."""
    return list(DEMO_TASKS.keys())


def run_demo(demo_name: str) -> None:
    """Run a specific demo by name."""
    from aragora.cli.ask import run_debate

    if demo_name not in DEMO_TASKS:
        print(f"Unknown demo: {demo_name}")
        print(f"Available demos: {', '.join(DEMO_TASKS.keys())}")
        return

    demo = DEMO_TASKS[demo_name]
    task = str(demo["task"])
    agents = str(demo["agents"])
    rounds = int(demo["rounds"])

    print("\n" + "=" * 60)
    print("ARAGORA DEMO - Decision Stress-Test")
    print("=" * 60)
    print(f"\nTask: {task[:80]}...")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print("\n" + "-" * 60)
    print("Starting debate...")
    print("-" * 60 + "\n")

    start = time.time()

    result = asyncio.run(
        run_debate(
            task=task,
            agents_str=agents,
            rounds=rounds,
            consensus="majority",
            learn=False,
            enable_audience=False,
            protocol_overrides={
                "convergence_detection": False,
                "vote_grouping": False,
                "enable_trickster": False,
                "enable_research": False,
                "enable_rhetorical_observer": False,
                "role_rotation": False,
                "role_matching": False,
            },
        )
    )

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("DEBATE COMPLETE")
    print("=" * 60)
    print(f"Duration: {elapsed:.1f}s")
    print(f"Consensus: {'Reached' if result.consensus_reached else 'Not reached'}")
    print(f"Confidence: {result.confidence:.0%}")
    print("\n" + "-" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(result.final_answer[:1000])
    if len(result.final_answer) > 1000:
        print("...")
    print("\n" + "=" * 60)


def main(args: argparse.Namespace) -> None:
    """Handle 'demo' command - run a quick compelling demo."""
    demo_name = getattr(args, "name", None) or "rate-limiter"
    run_demo(demo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Aragora demos")
    parser.add_argument(
        "name",
        nargs="?",
        default="rate-limiter",
        help=f"Demo name (available: {', '.join(DEMO_TASKS.keys())})",
    )
    main(parser.parse_args())
