"""
Analytics CLI commands: debate stats, agent leaderboard, costs, trends.

Commands for viewing platform analytics:
- summary: Overall debate statistics
- agents: Agent performance leaderboard
- costs: Cost breakdown
- trends: Usage trends over time
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)


def add_analytics_parser(subparsers) -> None:
    """Register the 'analytics' subcommand."""
    analytics_parser = subparsers.add_parser(
        "analytics",
        help="View debate analytics and platform usage",
        description="View debate statistics, agent performance, costs, and usage trends.",
    )

    analytics_sub = analytics_parser.add_subparsers(dest="analytics_command")

    # Shared args
    for name, help_text in [
        ("summary", "Overall debate statistics (default)"),
        ("agents", "Agent performance leaderboard"),
        ("costs", "Cost breakdown"),
        ("trends", "Usage trends over time"),
    ]:
        p = analytics_sub.add_parser(name, help=help_text)
        p.add_argument(
            "--days", type=int, default=30, help="Look-back period in days (default: 30)"
        )
        p.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")
        p.add_argument("--org-id", help="Filter by organization ID")
        if name == "agents":
            p.add_argument(
                "--limit", type=int, default=10, help="Number of agents to show (default: 10)"
            )
            p.add_argument(
                "--sort-by",
                choices=["elo", "debates", "accuracy", "cost"],
                default="elo",
                help="Sort agents by metric (default: elo)",
            )
        p.set_defaults(func=cmd_analytics)

    analytics_parser.set_defaults(
        func=cmd_analytics, analytics_command="summary", _parser=analytics_parser
    )


def cmd_analytics(args: argparse.Namespace) -> None:
    """Handle analytics commands."""
    subcommand = getattr(args, "analytics_command", "summary")
    days = getattr(args, "days", 30)
    as_json = getattr(args, "as_json", False)
    org_id = getattr(args, "org_id", None)

    try:
        from aragora.analytics.debate_analytics import get_debate_analytics

        analytics = get_debate_analytics()
    except ImportError:
        print("Error: Analytics module not available", file=sys.stderr)
        sys.exit(1)

    if subcommand == "summary":
        _cmd_summary(analytics, days, org_id, as_json)
    elif subcommand == "agents":
        limit = getattr(args, "limit", 10)
        sort_by = getattr(args, "sort_by", "elo")
        _cmd_agents(analytics, days, limit, sort_by, as_json)
    elif subcommand == "costs":
        _cmd_costs(analytics, days, org_id, as_json)
    elif subcommand == "trends":
        _cmd_trends(analytics, days, org_id, as_json)


def _cmd_summary(analytics, days: int, org_id: str | None, as_json: bool) -> None:
    """Show debate summary statistics."""
    stats = asyncio.run(analytics.get_debate_stats(org_id=org_id, days_back=days))

    if as_json:
        print(json.dumps(stats.to_dict(), indent=2, default=str))
        return

    print(f"\nDebate Analytics (last {days} days)")
    print("=" * 50)
    print(f"Total Debates:     {stats.total_debates}")
    print(f"Completed:         {stats.completed_debates}")
    print(f"Failed:            {stats.failed_debates}")
    print(f"Consensus Rate:    {stats.consensus_rate:.1%}")
    print(f"Avg Rounds:        {stats.avg_rounds:.1f}")
    print(f"Avg Duration:      {stats.avg_duration_seconds:.0f}s")
    print(f"Avg Agents/Debate: {stats.avg_agents_per_debate:.1f}")
    print(f"Total Messages:    {stats.total_messages}")
    print(f"Total Votes:       {stats.total_votes}")
    if stats.by_protocol:
        print("\nBy Protocol:")
        for proto, count in stats.by_protocol.items():
            print(f"  {proto}: {count}")
    print()


def _cmd_agents(analytics, days: int, limit: int, sort_by: str, as_json: bool) -> None:
    """Show agent performance leaderboard."""
    agents = asyncio.run(
        analytics.get_agent_leaderboard(limit=limit, days_back=days, sort_by=sort_by)
    )

    if as_json:
        print(json.dumps([a.to_dict() for a in agents], indent=2, default=str))
        return

    print(f"\nAgent Leaderboard (last {days} days, top {limit})")
    print("=" * 80)
    print(f"{'#':<4} {'AGENT':<20} {'ELO':>7} {'DEBATES':>8} {'VOTES':>7} {'ERR%':>6} {'COST':>10}")
    print("-" * 80)

    for agent in agents:
        print(
            f"{agent.rank:<4} {agent.agent_name[:19]:<20} "
            f"{agent.current_elo:>7.0f} {agent.debates_participated:>8} "
            f"{agent.positive_votes:>7} {agent.error_rate:>5.1%} "
            f"${agent.total_cost:>9}"
        )
    print()


def _cmd_costs(analytics, days: int, org_id: str | None, as_json: bool) -> None:
    """Show cost breakdown."""
    costs = asyncio.run(analytics.get_cost_breakdown(days_back=days, org_id=org_id))

    if as_json:
        print(json.dumps(costs.to_dict(), indent=2, default=str))
        return

    print(f"\nCost Breakdown (last {days} days)")
    print("=" * 50)
    d = costs.to_dict()
    print(f"Total Cost:        ${d.get('total_cost', 0)}")
    print(f"Total Tokens In:   {d.get('total_tokens_in', 0):,}")
    print(f"Total Tokens Out:  {d.get('total_tokens_out', 0):,}")
    print(f"Avg Cost/Debate:   ${d.get('avg_cost_per_debate', 0)}")
    if d.get("by_agent"):
        print("\nBy Agent:")
        for agent_name, cost in d["by_agent"].items():
            print(f"  {agent_name}: ${cost}")
    print()


def _cmd_trends(analytics, days: int, org_id: str | None, as_json: bool) -> None:
    """Show usage trends."""
    from aragora.analytics.debate_analytics import DebateMetricType, DebateTimeGranularity

    trends = asyncio.run(
        analytics.get_usage_trends(
            metric=DebateMetricType.DEBATE_COUNT,
            granularity=DebateTimeGranularity.DAILY,
            days_back=days,
            org_id=org_id,
        )
    )

    if as_json:
        print(json.dumps([t.to_dict() for t in trends], indent=2, default=str))
        return

    print(f"\nUsage Trends - Daily Debate Count (last {days} days)")
    print("=" * 50)

    if not trends:
        print("No data available.")
        return

    max_val = max(t.value for t in trends) if trends else 1
    bar_width = 30

    for point in trends:
        date_str = point.timestamp.strftime("%Y-%m-%d")
        bar_len = int((point.value / max_val) * bar_width) if max_val > 0 else 0
        bar = "#" * bar_len
        print(f"  {date_str} | {bar:<{bar_width}} {point.value:.0f}")
    print()
