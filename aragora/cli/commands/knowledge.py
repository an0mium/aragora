"""
Knowledge management CLI commands.

Provides CLI access to the Knowledge Mound via server API endpoints.
Commands:
- aragora km query <text>              - Search knowledge base
- aragora km store <text> --source <s> - Store knowledge entry
- aragora km stats                     - Show KM statistics
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


def cmd_knowledge(args: argparse.Namespace) -> None:
    """Handle 'knowledge' command - dispatch to subcommands."""
    subcommand = getattr(args, "km_command", None)

    if subcommand == "query":
        asyncio.run(_cmd_query(args))
    elif subcommand == "store":
        asyncio.run(_cmd_store(args))
    elif subcommand == "stats":
        asyncio.run(_cmd_stats(args))
    else:
        print("\nUsage: aragora km <command>")
        print("\nCommands:")
        print("  query <text>                Search the knowledge base")
        print("  store <text> --source <s>   Store a knowledge entry")
        print("  stats                       Show knowledge base statistics")


async def _cmd_query(args: argparse.Namespace) -> None:
    """Search the knowledge base."""
    text = getattr(args, "text", "")
    limit = getattr(args, "limit", 10)
    as_json = getattr(args, "json", False)

    if not text:
        print("Error: Search text is required")
        print("Usage: aragora km query <text>")
        return

    params: dict[str, Any] = {"q": text, "limit": limit}

    try:
        result = await _api_get("/api/km/search", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        results = result.get("results", [])
        total = result.get("total", len(results))
        print(f"\nFound {total} knowledge entries matching '{text}':\n")

        for entry in results:
            entry_id = entry.get("id", "unknown")[:12]
            source = entry.get("source", "unknown")
            score = entry.get("score", 0)
            content = entry.get("content", "")[:80]
            print(f"  [{entry_id}] (source: {source}, score: {score:.2f})")
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
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


async def _cmd_store(args: argparse.Namespace) -> None:
    """Store a knowledge entry."""
    text = getattr(args, "text", "")
    source = getattr(args, "source", "cli")
    as_json = getattr(args, "json", False)

    if not text:
        print("Error: Text content is required")
        print("Usage: aragora km store <text> --source <source>")
        return

    data: dict[str, Any] = {
        "content": text,
        "source": source,
    }

    try:
        result = await _api_post("/api/km/entries", data)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        entry_id = result.get("id", "unknown")
        stored_source = result.get("source", source)
        print("\nKnowledge entry stored successfully.")
        print(f"  ID:     {entry_id}")
        print(f"  Source: {stored_source}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


async def _cmd_stats(args: argparse.Namespace) -> None:
    """Show knowledge base statistics."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/km/stats")
        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("KNOWLEDGE MOUND STATISTICS")
        print("=" * 60 + "\n")

        total = result.get("total_entries", 0)
        adapters = result.get("adapters", 0)
        sources = result.get("sources", {})
        print(f"  Total entries:   {total}")
        print(f"  Active adapters: {adapters}")

        if sources:
            print("\n  Entries by source:")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"    {src:20} {count}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")


def _print_api_error(e: httpx.HTTPStatusError) -> None:
    """Print a human-readable API error message."""
    try:
        error_data = e.response.json()
        print(f"  {error_data.get('error', error_data.get('detail', 'Unknown error'))}")
    except (ValueError, KeyError):
        print(f"  {e.response.text}")


def add_knowledge_ops_parser(subparsers: Any) -> None:
    """Add knowledge operations subparser to CLI."""
    kp = subparsers.add_parser(
        "km",
        help="Knowledge Mound management commands",
        description="Search, store, and inspect Knowledge Mound entries via API.",
    )
    kp.set_defaults(func=cmd_knowledge)

    kp_sub = kp.add_subparsers(dest="km_command")

    # query
    query_p = kp_sub.add_parser("query", help="Search the knowledge base")
    query_p.add_argument("text", help="Search text")
    query_p.add_argument("--limit", "-l", type=int, default=10, help="Max results (default: 10)")
    query_p.add_argument("--json", action="store_true", help="Output as JSON")

    # store
    store_p = kp_sub.add_parser("store", help="Store a knowledge entry")
    store_p.add_argument("text", help="Content to store")
    store_p.add_argument("--source", "-s", default="cli", help="Knowledge source (default: cli)")
    store_p.add_argument("--json", action="store_true", help="Output as JSON")

    # stats
    stats_p = kp_sub.add_parser("stats", help="Show knowledge base statistics")
    stats_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_knowledge",
    "add_knowledge_ops_parser",
]
