"""
Memory operations CLI commands.

Provides CLI access to the multi-tier memory system via server API endpoints.
Commands:
- aragora memory query <text>        - Search across memory tiers
- aragora memory store <text> --tier - Store a memory entry
- aragora memory stats               - Show tier statistics
- aragora memory promote <id> --to   - Promote entry to higher tier
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

TIER_ORDER = ["fast", "medium", "slow", "glacial"]
TIER_TTLS = {"fast": "1 min", "medium": "1 hour", "slow": "1 day", "glacial": "1 week"}


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


async def _api_post(endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make POST request to API."""
    base = _get_api_base()
    url = f"{base}{endpoint}"
    headers = _get_auth_headers()
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=data or {}, headers=headers)
        response.raise_for_status()
        return response.json()


def cmd_memory_ops(args: argparse.Namespace) -> None:
    """Handle 'memory' command - dispatch to subcommands."""
    subcommand = getattr(args, "memory_command", None)

    if subcommand == "query":
        asyncio.run(_cmd_query(args))
    elif subcommand == "store":
        asyncio.run(_cmd_store(args))
    elif subcommand == "stats":
        asyncio.run(_cmd_stats(args))
    elif subcommand == "promote":
        asyncio.run(_cmd_promote(args))
    else:
        print("\nUsage: aragora memory <command>")
        print("\nCommands:")
        print("  query <text>                Search across memory tiers")
        print("  store <text> --tier <tier>  Store a memory entry")
        print("  stats                       Show tier sizes, hit rates, TTL info")
        print("  promote <id> --to <tier>    Promote entry to higher tier")
        print("\nTiers: fast (1 min), medium (1 hour), slow (1 day), glacial (1 week)")


async def _cmd_query(args: argparse.Namespace) -> None:
    """Search across memory tiers."""
    text = getattr(args, "text", "")
    tier = getattr(args, "tier", None)
    limit = getattr(args, "limit", 10)
    as_json = getattr(args, "json", False)

    if not text:
        print("Error: Search text is required")
        print("Usage: aragora memory query <text>")
        return

    params: dict[str, Any] = {"q": text, "limit": limit}
    if tier:
        params["tier"] = tier

    try:
        result = await _api_get("/api/v1/memory/continuum/search", params)

        if as_json:
            print(json.dumps(result, indent=2))
            return

        results = result.get("results", [])
        total = result.get("total", len(results))
        print(f"\nFound {total} memory entries matching '{text}':\n")

        for entry in results:
            entry_id = entry.get("id", "unknown")[:12]
            tier_name = entry.get("tier", "unknown")
            score = entry.get("score", 0)
            content = entry.get("content", "")[:80]
            print(f"  [{entry_id}] (tier: {tier_name}, score: {score:.2f})")
            print(f"    {content}")
            print()

        if not results:
            print("  No matching entries found.")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_store(args: argparse.Namespace) -> None:
    """Store a memory entry."""
    text = getattr(args, "text", "")
    tier = getattr(args, "tier", "fast")
    as_json = getattr(args, "json", False)

    if not text:
        print("Error: Text content is required")
        print("Usage: aragora memory store <text> --tier <tier>")
        return

    data: dict[str, Any] = {
        "content": text,
        "tier": tier,
    }

    try:
        result = await _api_post("/api/v1/memory", data)

        if as_json:
            print(json.dumps(result, indent=2))
            return

        entry_id = result.get("id", "unknown")
        stored_tier = result.get("tier", tier)
        print("\nMemory entry stored successfully.")
        print(f"  ID:   {entry_id}")
        print(f"  Tier: {stored_tier} (TTL: {TIER_TTLS.get(stored_tier, 'unknown')})")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_stats(args: argparse.Namespace) -> None:
    """Show tier sizes, hit rates, TTL info."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/v1/memory/continuum/tier-stats")

        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("MEMORY TIER STATISTICS")
        print("=" * 60 + "\n")

        tiers = result.get("tiers", {})
        total = result.get("total_entries", 0)
        print(f"  Total entries: {total}\n")

        for tier_name in TIER_ORDER:
            tier_data = tiers.get(tier_name, {})
            count = tier_data.get("count", 0)
            hit_rate = tier_data.get("hit_rate", 0)
            ttl = TIER_TTLS.get(tier_name, "unknown")
            bar = "#" * min(count // 5, 30) if count > 0 else ""
            print(f"  {tier_name:8}  entries: {count:5}  hit_rate: {hit_rate:.1%}  TTL: {ttl}")
            if bar:
                print(f"            {bar}")

        promotions = result.get("promotions", 0)
        demotions = result.get("demotions", 0)
        if promotions or demotions:
            print(f"\n  Promotions: {promotions}  Demotions: {demotions}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_promote(args: argparse.Namespace) -> None:
    """Promote a memory entry to a higher tier."""
    entry_id = getattr(args, "id", None)
    target_tier = getattr(args, "to", None)
    as_json = getattr(args, "json", False)

    if not entry_id:
        print("Error: Entry ID is required")
        print("Usage: aragora memory promote <id> --to <tier>")
        return

    if not target_tier:
        print("Error: Target tier is required (--to <tier>)")
        print("Usage: aragora memory promote <id> --to <tier>")
        return

    data: dict[str, Any] = {
        "action": "promote",
        "entry_id": entry_id,
        "target_tier": target_tier,
    }

    try:
        result = await _api_post("/api/v1/memory", data)

        if as_json:
            print(json.dumps(result, indent=2))
            return

        if result.get("success", False):
            prev_tier = result.get("previous_tier", "unknown")
            print("\nEntry promoted successfully.")
            print(f"  ID:   {entry_id}")
            print(f"  From: {prev_tier}")
            print(f"  To:   {target_tier} (TTL: {TIER_TTLS.get(target_tier, 'unknown')})")
        else:
            error = result.get("error", "Unknown error")
            print(f"\nPromotion failed: {error}")

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
    except Exception:
        print(f"  {e.response.text}")


def add_memory_ops_parser(subparsers: Any) -> None:
    """Add memory operations subparser to CLI.

    This replaces the existing simple memory parser with sub-subcommands
    for query, store, stats, and promote.
    """
    mp = subparsers.add_parser(
        "memory",
        help="Memory management commands",
        description="Search, store, and manage multi-tier memory entries.",
    )
    mp.set_defaults(func=cmd_memory_ops)

    mp_sub = mp.add_subparsers(dest="memory_command")

    # query
    query_p = mp_sub.add_parser("query", help="Search across memory tiers")
    query_p.add_argument("text", help="Search text")
    query_p.add_argument(
        "--tier",
        "-t",
        choices=TIER_ORDER,
        help="Filter by tier",
    )
    query_p.add_argument("--limit", "-l", type=int, default=10, help="Max results (default: 10)")
    query_p.add_argument("--json", action="store_true", help="Output as JSON")

    # store
    store_p = mp_sub.add_parser("store", help="Store a memory entry")
    store_p.add_argument("text", help="Content to store")
    store_p.add_argument(
        "--tier",
        "-t",
        choices=TIER_ORDER,
        default="fast",
        help="Memory tier (default: fast)",
    )
    store_p.add_argument("--json", action="store_true", help="Output as JSON")

    # stats
    stats_p = mp_sub.add_parser("stats", help="Show tier sizes, hit rates, TTL info")
    stats_p.add_argument("--json", action="store_true", help="Output as JSON")

    # promote
    promote_p = mp_sub.add_parser("promote", help="Promote entry to higher tier")
    promote_p.add_argument("id", help="Memory entry ID")
    promote_p.add_argument(
        "--to",
        required=True,
        choices=TIER_ORDER,
        help="Target tier",
    )
    promote_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_memory_ops",
    "add_memory_ops_parser",
]
