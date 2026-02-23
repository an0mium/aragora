#!/usr/bin/env python3
"""
11_self_improvement.py - Trigger and monitor self-improvement cycles.

Shows how to use the SDK to start a self-improvement cycle, monitor
its progress in real time, and review the results. The self-improvement
system uses the Nomic Loop to autonomously plan, implement, test, and
verify code changes.

Usage:
    python 11_self_improvement.py                    # Run actual cycle
    python 11_self_improvement.py --dry-run          # Preview without executing
    python 11_self_improvement.py --goal "Improve test coverage for debate module"
"""

import argparse
import asyncio
import time
from aragora_sdk import AragoraClient


async def run_self_improvement(goal: str, dry_run: bool = False) -> dict:
    """Start a self-improvement cycle and monitor its progress."""

    client = AragoraClient()

    if dry_run:
        # Preview the plan without executing
        print(f"[DRY RUN] Goal: {goal}")
        result = await client.self_improve.dry_run(goal=goal)
        print(f"\nPlanned goals ({len(result.get('goals', []))}):")
        for g in result.get("goals", []):
            print(f"  - [{g.get('track', '?')}] {g.get('description', '')}")
        return result

    # Start a new cycle with approval gates
    cycle = await client.self_improve.start(
        goal=goal,
        require_approval=True,  # Pause at checkpoints for human review
        budget_limit_usd=5.0,  # Cap API spending
    )

    cycle_id = cycle["cycle_id"]
    print(f"Started cycle: {cycle_id}")

    # Poll for status updates
    while True:
        status = await client.self_improve.status(cycle_id)
        phase = status.get("phase", "unknown")
        progress = status.get("progress", 0)

        print(f"  Phase: {phase} ({progress}%)")

        if phase == "awaiting_approval":
            # Auto-approve in this example (in production, show UI)
            print("  -> Auto-approving checkpoint...")
            await client.self_improve.approve(cycle_id)

        if status.get("completed", False):
            break

        await asyncio.sleep(3)

    # Review results
    result = await client.self_improve.status(cycle_id)
    print(f"\nCycle complete!")
    print(f"  Files changed: {result.get('files_changed', 0)}")
    print(f"  Tests passed: {result.get('tests_passed', 0)}")
    print(f"  Improvement score: {result.get('score', 0):.2f}")

    return result


async def list_history() -> list:
    """List past self-improvement cycles."""
    client = AragoraClient()
    history = await client.self_improve.history(limit=10)
    print(f"\nRecent cycles ({len(history)}):")
    for entry in history:
        status_icon = "+" if entry.get("success") else "x"
        print(f"  [{status_icon}] {entry.get('goal', '?')} ({entry.get('duration_s', 0):.0f}s)")
    return history


def main():
    parser = argparse.ArgumentParser(description="Self-improvement via Aragora SDK")
    parser.add_argument("--goal", default="Improve error handling in server handlers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--history", action="store_true", help="Show past cycles")
    args = parser.parse_args()

    if args.history:
        asyncio.run(list_history())
    else:
        result = asyncio.run(run_self_improvement(args.goal, args.dry_run))
        print(f"\nResult: {result.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
