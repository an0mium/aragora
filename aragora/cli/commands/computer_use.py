"""
Computer use CLI commands.

Provides CLI access to the computer use task system via server API endpoints.
Commands:
- aragora computer-use run <goal>        - Start a computer use task
- aragora computer-use status <task_id>  - Get task status
- aragora computer-use list              - List all tasks
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


def cmd_computer_use(args: argparse.Namespace) -> None:
    """Handle 'computer-use' command - dispatch to subcommands."""
    subcommand = getattr(args, "cu_command", None)

    if subcommand == "run":
        asyncio.run(_cmd_run(args))
    elif subcommand == "status":
        asyncio.run(_cmd_status(args))
    elif subcommand == "list":
        asyncio.run(_cmd_list(args))
    else:
        print("\nUsage: aragora computer-use <command>")
        print("\nCommands:")
        print("  run <goal>          Start a computer use task")
        print("  status <task_id>    Get task status")
        print("  list                List all tasks")


async def _cmd_run(args: argparse.Namespace) -> None:
    """Start a computer use task."""
    goal = getattr(args, "goal", "")
    as_json = getattr(args, "json", False)

    if not goal:
        print("Error: Goal is required")
        print("Usage: aragora computer-use run <goal>")
        return

    data: dict[str, Any] = {"goal": goal}
    try:
        result = await _api_post("/api/v1/computer-use/tasks", data)
        if as_json:
            print(json.dumps(result, indent=2))
            return
        task_id = result.get("task_id", result.get("id", "unknown"))
        status = result.get("status", "created")
        print("\nComputer use task created.")
        print(f"  Task ID: {task_id}")
        print(f"  Status:  {status}")
        print(f"  Goal:    {goal}")
    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_status(args: argparse.Namespace) -> None:
    """Get task status."""
    task_id = getattr(args, "task_id", None)
    as_json = getattr(args, "json", False)

    if not task_id:
        print("Error: Task ID is required")
        print("Usage: aragora computer-use status <task_id>")
        return

    try:
        result = await _api_get(f"/api/v1/computer-use/tasks/{task_id}")
        if as_json:
            print(json.dumps(result, indent=2))
            return
        status = result.get("status", "unknown")
        goal = result.get("goal", "")
        steps = result.get("steps_completed", 0)
        print(f"\nTask {task_id}:")
        print(f"  Status: {status}")
        print(f"  Goal:   {goal}")
        print(f"  Steps:  {steps}")
    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_list(args: argparse.Namespace) -> None:
    """List all computer use tasks."""
    as_json = getattr(args, "json", False)
    limit = getattr(args, "limit", 20)

    try:
        result = await _api_get("/api/v1/computer-use/tasks", params={"limit": limit})
        if as_json:
            print(json.dumps(result, indent=2))
            return
        tasks = result.get("tasks", [])
        print(f"\nComputer use tasks ({len(tasks)}):\n")
        for task in tasks:
            tid = task.get("task_id", task.get("id", "unknown"))[:12]
            status = task.get("status", "unknown")
            goal = task.get("goal", "")[:60]
            print(f"  [{tid}] {status:12} {goal}")
        if not tasks:
            print("  No tasks found.")
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


def add_computer_use_parser(subparsers: Any) -> None:
    """Add computer-use subparser to CLI."""
    cp = subparsers.add_parser(
        "computer-use",
        help="Computer use task management",
        description="Start, monitor, and list computer use tasks.",
    )
    cp.set_defaults(func=cmd_computer_use)

    cp_sub = cp.add_subparsers(dest="cu_command")

    # run
    run_p = cp_sub.add_parser("run", help="Start a computer use task")
    run_p.add_argument("goal", help="Goal for the computer use task")
    run_p.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    status_p = cp_sub.add_parser("status", help="Get task status")
    status_p.add_argument("task_id", help="Task ID")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")

    # list
    list_p = cp_sub.add_parser("list", help="List all tasks")
    list_p.add_argument("--limit", "-l", type=int, default=20, help="Max results (default: 20)")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_computer_use",
    "add_computer_use_parser",
]
