"""
Billing operations CLI commands.

Provides CLI access to the billing and cost management system via server API endpoints.
Commands:
- aragora costs usage     - Current usage summary
- aragora costs budget    - Budget status and limits
- aragora costs forecast  - Cost forecast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aragora._lazy_imports import lazy_module

# Optional runtime dep: imported lazily so the parser can register this
# command on a base install that lacks httpx (httpx is only needed when a
# command actually issues an HTTP request).
if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import httpx
else:
    httpx = lazy_module("httpx")

from aragora.config.settings import get_settings

logger = logging.getLogger(__name__)


def _get_api_base() -> str:
    """Get the API base URL from settings."""
    settings = get_settings()
    host = getattr(settings, "server_host", "localhost")
    port = getattr(settings, "server_port", 8000)
    return f"http://{host}:{port}"


def _get_auth_headers() -> dict[str, str]:
    """Get authentication headers if available."""
    settings = get_settings()
    token = getattr(settings, "api_token", None)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def _api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make GET request to API."""
    base = _get_api_base()
    url = f"{base}{endpoint}"
    headers = _get_auth_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def cmd_billing_ops(args: argparse.Namespace) -> None:
    """Handle 'billing' command - dispatch to subcommands."""
    subcommand = getattr(args, "billing_command", None)

    if subcommand == "usage":
        asyncio.run(_cmd_usage(args))
    elif subcommand == "budget":
        asyncio.run(_cmd_budget(args))
    elif subcommand == "forecast":
        asyncio.run(_cmd_forecast(args))
    elif subcommand == "dashboard":
        _cmd_dashboard(args)
    elif subcommand == "report":
        _cmd_report(args)
    elif subcommand == "agents":
        _cmd_agents(args)
    else:
        print("\nUsage: aragora costs <command>")
        print("\nCommands:")
        print("  usage      Current usage summary (API)")
        print("  budget     Budget status and limits (API)")
        print("  forecast   Cost forecast and projections (API)")
        print("  dashboard  Cost dashboard summary (local)")
        print("  report     Generate cost report (local)")
        print("  agents     Per-agent cost breakdown (local)")


async def _cmd_usage(args: argparse.Namespace) -> None:
    """Show current usage summary."""
    as_json = getattr(args, "json", False)
    period = getattr(args, "period", "current")

    params: dict[str, Any] = {}
    if period != "current":
        params["period"] = period

    try:
        result = await _api_get("/api/v1/billing/usage", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("USAGE SUMMARY")
        print("=" * 60 + "\n")

        total_cost = result.get("total_cost", 0)
        total_tokens = result.get("total_tokens", 0)
        period_label = result.get("period", period)
        providers = result.get("by_provider", {})

        print(f"  Period:       {period_label}")
        print(f"  Total cost:   ${total_cost:.4f}")
        print(f"  Total tokens: {total_tokens:,}")

        if providers:
            print("\n  By provider:")
            for provider, data in sorted(providers.items()):
                cost = data.get("cost", 0) if isinstance(data, dict) else data
                print(f"    {provider:20} ${cost:.4f}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


async def _cmd_budget(args: argparse.Namespace) -> None:
    """Show budget status."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/v1/billing/budget")
        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("BUDGET STATUS")
        print("=" * 60 + "\n")

        budget_limit = result.get("limit", 0)
        spent = result.get("spent", 0)
        remaining = result.get("remaining", budget_limit - spent)
        utilization = result.get("utilization", 0)

        print(f"  Budget limit: ${budget_limit:.2f}")
        print(f"  Spent:        ${spent:.4f}")
        print(f"  Remaining:    ${remaining:.4f}")
        print(f"  Utilization:  {utilization:.1%}")

        alerts = result.get("alerts", [])
        if alerts:
            print("\n  Active alerts:")
            for alert in alerts:
                print(f"    - {alert}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


async def _cmd_forecast(args: argparse.Namespace) -> None:
    """Show cost forecast."""
    as_json = getattr(args, "json", False)
    days = getattr(args, "days", 30)

    params: dict[str, Any] = {"days": days}

    try:
        result = await _api_get("/api/v1/billing/forecast", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("COST FORECAST")
        print("=" * 60 + "\n")

        projected = result.get("projected_cost", 0)
        daily_avg = result.get("daily_average", 0)
        horizon = result.get("horizon_days", days)
        confidence = result.get("confidence", 0)

        print(f"  Forecast horizon: {horizon} days")
        print(f"  Daily average:    ${daily_avg:.4f}")
        print(f"  Projected total:  ${projected:.2f}")
        if confidence:
            print(f"  Confidence:       {confidence:.0%}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


def _get_cost_tracker():
    """Get a CostTracker instance for local cost operations."""
    try:
        from aragora.billing.cost_tracker import get_cost_tracker

        return get_cost_tracker()
    except ImportError:
        logger.debug("CostTracker not available")
        return None


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Show cost dashboard summary (local, no server required)."""
    as_json = getattr(args, "json", False)
    tracker = _get_cost_tracker()
    if tracker is None:
        print("\nError: CostTracker not available. Install billing dependencies.")
        return

    try:
        summary = tracker.get_dashboard_summary()
    except (AttributeError, TypeError):
        print("\nError: CostTracker.get_dashboard_summary() not available.")
        return

    if as_json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print("\n" + "=" * 60)
    print("COST DASHBOARD")
    print("=" * 60 + "\n")

    print(f"  Total spend:    ${summary.get('total_spend', 0):.4f}")
    print(f"  Budget limit:   ${summary.get('budget_limit', 0):.2f}")
    print(f"  Budget used:    {summary.get('budget_used_pct', 0):.1f}%")
    print(f"  Debates today:  {summary.get('debates_today', 0)}")

    top_agents = summary.get("top_agents", [])
    if top_agents:
        print("\n  Top agents by cost:")
        for entry in top_agents[:5]:
            name = entry.get("agent", "unknown")
            cost = entry.get("cost", 0)
            print(f"    {name:25} ${cost:.4f}")

    projections = summary.get("projections", {})
    if projections:
        daily = projections.get("daily_average", 0)
        monthly = projections.get("monthly_projected", 0)
        print(f"\n  Daily average:  ${daily:.4f}")
        print(f"  Monthly proj:   ${monthly:.2f}")


def _period_bounds(days: int) -> tuple[datetime, datetime]:
    """Translate a ``--days`` window into (period_start, period_end)."""
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=max(1, days))
    return period_start, period_end


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate a cost report (local, no server required)."""
    as_json = getattr(args, "json", False)
    days = getattr(args, "days", 30)
    workspace_id = getattr(args, "workspace", None)
    tracker = _get_cost_tracker()
    if tracker is None:
        print("\nError: CostTracker not available. Install billing dependencies.")
        return

    period_start, period_end = _period_bounds(days)
    try:
        report_obj = asyncio.run(
            tracker.generate_report(
                workspace_id=workspace_id,
                period_start=period_start,
                period_end=period_end,
            )
        )
    except AttributeError:
        print("\nError: CostTracker.generate_report() not available.")
        return

    # CostReport is a dataclass; normalize to a plain dict for display/JSON.
    report = report_obj.to_dict() if hasattr(report_obj, "to_dict") else dict(report_obj)

    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return

    total_cost = float(report.get("total_cost_usd", 0) or 0)
    total_tokens = int(report.get("total_tokens_in", 0) or 0) + int(
        report.get("total_tokens_out", 0) or 0
    )
    total_calls = int(report.get("total_api_calls", 0) or 0)

    print("\n" + "=" * 60)
    print(f"COST REPORT (Last {days} days)")
    print("=" * 60 + "\n")

    print(f"  Period:         {report.get('period_start', '')} -> {report.get('period_end', '')}")
    print(f"  Total cost:     ${total_cost:.4f}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Total API calls:{total_calls:>8,}")

    by_provider = report.get("cost_by_provider", {})
    if by_provider:
        print("\n  By provider:")
        for provider, data in sorted(by_provider.items()):
            cost = data.get("cost", 0) if isinstance(data, dict) else float(data or 0)
            print(f"    {provider:20} ${float(cost):.4f}")

    top_agents = report.get("top_agents_by_cost", [])
    if top_agents:
        print("\n  Top agents by cost:")
        for entry in top_agents[:7]:
            name = entry.get("agent", "unknown")
            cost = float(entry.get("cost_usd", 0) or 0)
            print(f"    {name:25} ${cost:.4f}")


def _cmd_agents(args: argparse.Namespace) -> None:
    """Show per-agent cost breakdown (local, no server required)."""
    as_json = getattr(args, "json", False)
    days = getattr(args, "days", 30)
    workspace_id = getattr(args, "workspace", None) or "default"
    tracker = _get_cost_tracker()
    if tracker is None:
        print("\nError: CostTracker not available. Install billing dependencies.")
        return

    period_start, period_end = _period_bounds(days)
    try:
        agent_costs = asyncio.run(
            tracker.get_agent_costs(
                workspace_id,
                period_start=period_start,
                period_end=period_end,
            )
        )
    except AttributeError:
        print("\nError: CostTracker.get_agent_costs() not available.")
        return

    if as_json:
        print(json.dumps(agent_costs, indent=2, default=str))
        return

    print("\n" + "=" * 60)
    print(f"AGENT COSTS (Last {days} days)")
    print("=" * 60 + "\n")

    if not agent_costs:
        print("  No agent cost data available.")
        return

    print(f"  {'Agent':<25} {'Cost':>12} {'Share':>8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 8}")
    for name, data in sorted(
        agent_costs.items(),
        key=lambda kv: float(kv[1].get("cost_usd", 0) or 0),
        reverse=True,
    ):
        cost = float(data.get("cost_usd", 0) or 0)
        pct = float(data.get("percentage", 0) or 0)
        print(f"  {name:<25} ${cost:>11.4f} {pct:>6.1f}%")


def _print_api_error(e: httpx.HTTPStatusError) -> None:
    """Print a human-readable API error message."""
    try:
        error_data = e.response.json()
        print(f"  {error_data.get('error', error_data.get('detail', 'Unknown error'))}")
    except (ValueError, KeyError):
        print(f"  {e.response.text}")


def add_billing_ops_parser(subparsers: Any) -> None:
    """Add billing operations subparser to CLI."""
    bp = subparsers.add_parser(
        "costs",
        help="Cost tracking and billing management commands",
        description="View usage, budget status, and cost forecasts via API.",
    )
    bp.set_defaults(func=cmd_billing_ops)

    bp_sub = bp.add_subparsers(dest="billing_command")

    # usage
    usage_p = bp_sub.add_parser("usage", help="Current usage summary")
    usage_p.add_argument(
        "--period", "-p", default="current", help="Billing period (default: current)"
    )
    usage_p.add_argument("--json", action="store_true", help="Output as JSON")

    # budget
    budget_p = bp_sub.add_parser("budget", help="Budget status and limits")
    budget_p.add_argument("--json", action="store_true", help="Output as JSON")

    # forecast
    forecast_p = bp_sub.add_parser("forecast", help="Cost forecast and projections")
    forecast_p.add_argument(
        "--days", "-d", type=int, default=30, help="Forecast horizon in days (default: 30)"
    )
    forecast_p.add_argument("--json", action="store_true", help="Output as JSON")

    # dashboard (local)
    dashboard_p = bp_sub.add_parser("dashboard", help="Cost dashboard summary (local)")
    dashboard_p.add_argument("--json", action="store_true", help="Output as JSON")

    # report (local)
    report_p = bp_sub.add_parser("report", help="Generate cost report (local)")
    report_p.add_argument(
        "--days", "-d", type=int, default=30, help="Report period in days (default: 30)"
    )
    report_p.add_argument(
        "--workspace", "-w", default=None, help="Filter by workspace ID (optional)"
    )
    report_p.add_argument("--json", action="store_true", help="Output as JSON")

    # agents (local)
    agents_p = bp_sub.add_parser("agents", help="Per-agent cost breakdown (local)")
    agents_p.add_argument("--days", "-d", type=int, default=30, help="Period in days (default: 30)")
    agents_p.add_argument(
        "--workspace", "-w", default=None, help="Workspace ID to break down (default: 'default')"
    )
    agents_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_billing_ops",
    "add_billing_ops_parser",
]
