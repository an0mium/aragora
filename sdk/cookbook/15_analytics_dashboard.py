#!/usr/bin/env python3
"""
15_analytics_dashboard.py - Query debate analytics and agent performance.

Shows how to use the SDK to access debate statistics, agent ELO rankings,
calibration scores, and usage metrics for building custom dashboards.

Usage:
    python 15_analytics_dashboard.py                    # Show all metrics
    python 15_analytics_dashboard.py --dry-run          # Preview
    python 15_analytics_dashboard.py --agents           # Agent rankings only
"""

import argparse
import asyncio
from aragora_sdk import AragoraClient


async def show_analytics(dry_run: bool = False) -> dict:
    """Fetch and display comprehensive analytics."""

    client = AragoraClient()

    if dry_run:
        print("[DRY RUN] Would fetch analytics")
        return {"status": "dry_run"}

    # Debate statistics
    stats = await client.analytics.debate_stats()
    print("Debate Statistics")
    print(f"  Total debates: {stats.get('total', 0)}")
    print(f"  Consensus rate: {stats.get('consensus_rate', 0):.1%}")
    print(f"  Avg duration: {stats.get('avg_duration_s', 0):.1f}s")
    print(f"  Avg rounds: {stats.get('avg_rounds', 0):.1f}")

    # Agent leaderboard
    rankings = await client.analytics.agent_rankings(limit=10)
    print(f"\nAgent Leaderboard (top {len(rankings)}):")
    for i, agent in enumerate(rankings, 1):
        elo = agent.get("elo", 1500)
        calibration = agent.get("brier_score", None)
        cal_str = f" (Brier: {calibration:.3f})" if calibration is not None else ""
        print(f"  {i}. {agent['name']} - ELO {elo}{cal_str}")

    # Usage metrics
    usage = await client.analytics.usage(period="7d")
    print("\nUsage (last 7 days):")
    print(f"  API calls: {usage.get('api_calls', 0)}")
    print(f"  Tokens used: {usage.get('tokens', 0):,}")
    print(f"  Cost: ${usage.get('cost_usd', 0):.2f}")

    return {"stats": stats, "rankings": rankings, "usage": usage}


async def show_agent_detail(agent_name: str) -> dict:
    """Show detailed analytics for a specific agent."""

    client = AragoraClient()

    detail = await client.analytics.agent_detail(agent_name)
    print(f"\nAgent: {agent_name}")
    print(f"  ELO: {detail.get('elo', 1500)}")
    print(f"  Debates: {detail.get('total_debates', 0)}")
    print(f"  Win rate: {detail.get('win_rate', 0):.1%}")
    print(f"  Avg proposal quality: {detail.get('avg_quality', 0):.2f}")
    print(f"  Calibration (Brier): {detail.get('brier_score', 'N/A')}")

    return detail


def main():
    parser = argparse.ArgumentParser(description="Analytics dashboard via SDK")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agents", action="store_true", help="Show agent rankings only")
    parser.add_argument("--agent", type=str, help="Show detail for specific agent")
    args = parser.parse_args()

    if args.agent:
        asyncio.run(show_agent_detail(args.agent))
    elif args.agents:
        asyncio.run(show_analytics(args.dry_run))
    else:
        asyncio.run(show_analytics(args.dry_run))


if __name__ == "__main__":
    main()
