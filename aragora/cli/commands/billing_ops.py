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
from typing import Any

import httpx

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
    else:
        print("\nUsage: aragora costs <command>")
        print("\nCommands:")
        print("  usage      Current usage summary")
        print("  budget     Budget status and limits")
        print("  forecast   Cost forecast and projections")


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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        print(f"\nError: {e}")


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


__all__ = [
    "cmd_billing_ops",
    "add_billing_ops_parser",
]
