"""
Connectors CLI commands.

Provides CLI access to connector management via server API endpoints.
Commands:
- aragora connectors list            - List all connectors
- aragora connectors status <name>   - Get connector health
- aragora connectors test <name>     - Test a connector
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

CONNECTOR_TYPES = [
    "slack",
    "teams",
    "discord",
    "telegram",
    "whatsapp",
    "email",
    "webhook",
    "kafka",
    "rabbitmq",
    "zapier",
    "github",
]


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


def cmd_connectors(args: argparse.Namespace) -> None:
    """Handle 'connectors' command - dispatch to subcommands."""
    subcommand = getattr(args, "conn_command", None)

    if subcommand == "list":
        asyncio.run(_cmd_list(args))
    elif subcommand == "status":
        asyncio.run(_cmd_status(args))
    elif subcommand == "test":
        asyncio.run(_cmd_test(args))
    else:
        print("\nUsage: aragora connectors <command>")
        print("\nCommands:")
        print("  list              List all connectors")
        print("  status <name>     Get connector health status")
        print("  test <name>       Test a connector connection")
        print(f"\nKnown connector types: {', '.join(CONNECTOR_TYPES)}")


async def _cmd_list(args: argparse.Namespace) -> None:
    """List all connectors."""
    as_json = getattr(args, "json", False)
    conn_type = getattr(args, "type", None)

    params: dict[str, Any] = {}
    if conn_type:
        params["type"] = conn_type

    try:
        result = await _api_get("/api/v1/connectors", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        connectors = result.get("connectors", [])
        print(f"\nConnectors ({len(connectors)}):\n")
        for conn in connectors:
            name = conn.get("name", "unknown")
            ctype = conn.get("type", "unknown")
            status = conn.get("status", "unknown")
            icon = "+" if status == "connected" else "-"
            print(f"  [{icon}] {name:20} type: {ctype:12} status: {status}")

        if not connectors:
            print("  No connectors configured.")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_status(args: argparse.Namespace) -> None:
    """Get connector health status."""
    name = getattr(args, "name", None)
    as_json = getattr(args, "json", False)

    if not name:
        print("Error: Connector name is required")
        print("Usage: aragora connectors status <name>")
        return

    try:
        result = await _api_get(f"/api/v1/connectors/{name}/status")
        if as_json:
            print(json.dumps(result, indent=2))
            return

        status = result.get("status", "unknown")
        latency = result.get("latency_ms", None)
        last_seen = result.get("last_seen", "never")
        error = result.get("error", None)

        print(f"\nConnector: {name}")
        print(f"  Status:    {status}")
        if latency is not None:
            print(f"  Latency:   {latency}ms")
        print(f"  Last seen: {last_seen}")
        if error:
            print(f"  Error:     {error}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_test(args: argparse.Namespace) -> None:
    """Test a connector connection."""
    name = getattr(args, "name", None)
    as_json = getattr(args, "json", False)

    if not name:
        print("Error: Connector name is required")
        print("Usage: aragora connectors test <name>")
        return

    try:
        result = await _api_post(f"/api/v1/connectors/{name}/test")
        if as_json:
            print(json.dumps(result, indent=2))
            return

        success = result.get("success", False)
        latency = result.get("latency_ms", None)
        message = result.get("message", "")

        if success:
            print(f"\nConnector '{name}' test passed.")
            if latency is not None:
                print(f"  Latency: {latency}ms")
        else:
            print(f"\nConnector '{name}' test failed.")
            if message:
                print(f"  Message: {message}")

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


def add_connectors_parser(subparsers: Any) -> None:
    """Add connectors subparser to CLI."""
    cp = subparsers.add_parser(
        "connectors",
        help="Connector management commands",
        description="List, check status, and test external connectors.",
    )
    cp.set_defaults(func=cmd_connectors)

    cp_sub = cp.add_subparsers(dest="conn_command")

    # list
    list_p = cp_sub.add_parser("list", help="List all connectors")
    list_p.add_argument("--type", "-t", help="Filter by connector type")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    status_p = cp_sub.add_parser("status", help="Get connector health status")
    status_p.add_argument("name", help="Connector name")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")

    # test
    test_p = cp_sub.add_parser("test", help="Test a connector connection")
    test_p.add_argument("name", help="Connector name")
    test_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_connectors",
    "add_connectors_parser",
]
