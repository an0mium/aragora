"""
Verticals CLI commands.

Provides CLI access to vertical specialist configuration via API endpoints.
Commands:
- aragora verticals list
- aragora verticals get <vertical_id>
- aragora verticals tools <vertical_id>
- aragora verticals compliance <vertical_id>
- aragora verticals suggest --task "..."
"""

from __future__ import annotations

import argparse
import json
import os

DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def _get_api_client(api_url: str | None = None, api_key: str | None = None):
    """Get API client if available and server is reachable."""
    try:
        from aragora.client import AragoraClient

        url = api_url or DEFAULT_API_URL
        key = api_key or os.environ.get("ARAGORA_API_KEY") or os.environ.get("ARAGORA_API_TOKEN")
        client = AragoraClient(base_url=url, api_key=key)
        client.system.health()
        return client
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def cmd_verticals(args: argparse.Namespace) -> None:
    """Handle 'verticals' command - dispatch to subcommands."""
    subcommand = getattr(args, "verticals_command", None)
    api_url = getattr(args, "api_url", DEFAULT_API_URL)
    api_key = getattr(args, "api_key", None)
    as_json = bool(getattr(args, "json", False))

    client = _get_api_client(api_url, api_key)
    if client is None:
        print("\nError: API server not reachable.")
        print(f"  URL: {api_url}")
        print("  Start the server with: aragora serve")
        return

    if subcommand == "list":
        keyword = getattr(args, "keyword", None)
        verticals = client.verticals.list(keyword=keyword)
        if as_json:
            print(json.dumps({"verticals": verticals}, indent=2))
            return
        if not verticals:
            print("\nNo verticals found.")
            return
        print("\nAvailable Verticals:\n")
        for v in verticals:
            vid = v.get("vertical_id") or v.get("id") or "unknown"
            name = v.get("display_name") or v.get("name") or vid
            desc = v.get("description") or ""
            print(f"  [{vid}] {name}")
            if desc:
                print(f"    {desc}")
            tags = v.get("tags") or []
            if tags:
                print(f"    Tags: {', '.join(tags)}")
            print()
        return

    if subcommand == "get":
        vertical_id = getattr(args, "id", None)
        if not vertical_id:
            print("\nError: vertical_id required")
            return
        data = client.verticals.get(vertical_id)
        if as_json:
            print(json.dumps(data, indent=2))
            return
        print(f"\nVertical: {data.get('display_name', vertical_id)}")
        print(f"ID: {data.get('vertical_id', vertical_id)}")
        if data.get("description"):
            print(f"Description: {data['description']}")
        if data.get("expertise_areas"):
            print("Expertise Areas:")
            for area in data["expertise_areas"]:
                print(f"  - {area}")
        return

    if subcommand == "tools":
        vertical_id = getattr(args, "id", None)
        if not vertical_id:
            print("\nError: vertical_id required")
            return
        data = client.verticals.tools(vertical_id)
        if as_json:
            print(json.dumps(data, indent=2))
            return
        tools = data.get("tools", data)
        print(f"\nTools for {vertical_id}:")
        for tool in tools:
            name = tool.get("name") or "unknown"
            desc = tool.get("description") or ""
            print(f"  - {name}: {desc}")
        return

    if subcommand == "compliance":
        vertical_id = getattr(args, "id", None)
        if not vertical_id:
            print("\nError: vertical_id required")
            return
        data = client.verticals.compliance(vertical_id)
        if as_json:
            print(json.dumps(data, indent=2))
            return
        frameworks = data.get("frameworks", data)
        print(f"\nCompliance frameworks for {vertical_id}:")
        for fw in frameworks:
            name = fw.get("framework") or fw.get("name") or "unknown"
            version = fw.get("version") or ""
            level = fw.get("level") or ""
            suffix = f" ({version})" if version else ""
            print(f"  - {name}{suffix} [{level}]" if level else f"  - {name}{suffix}")
        return

    if subcommand == "suggest":
        task = getattr(args, "task", None)
        if not task:
            print("\nError: --task is required for suggest")
            return
        data = client.verticals.suggest(task)
        if as_json:
            print(json.dumps(data, indent=2))
            return
        print("\nSuggested Vertical:")
        print(f"  {data.get('vertical_id', data.get('id', 'unknown'))}")
        if data.get("reason"):
            print(f"  Reason: {data['reason']}")
        return

    # Default help
    print("\nUsage: aragora verticals <command>")
    print("\nCommands:")
    print("  list                 List available verticals")
    print("  get <vertical_id>     Show vertical configuration")
    print("  tools <vertical_id>   List tools for a vertical")
    print("  compliance <vertical_id>  Show compliance frameworks")
    print("  suggest --task <task>  Suggest vertical for a task")
    print("\nOptions:")
    print("  --keyword <term>      Filter list by keyword")
    print("  --api-url <url>       API server URL")
    print("  --api-key <key>       API key (or ARAGORA_API_KEY)")
    print("  --json                Output JSON")


def add_verticals_parser(subparsers) -> None:
    """Add the 'verticals' subcommand parser."""
    parser = subparsers.add_parser(
        "verticals",
        help="Manage vertical specialist configurations",
        description="List and inspect vertical specialist configurations.",
    )
    parser.add_argument(
        "verticals_command",
        nargs="?",
        help="Subcommand (list, get, tools, compliance, suggest)",
    )
    parser.add_argument("id", nargs="?", help="Vertical ID")
    parser.add_argument("--task", help="Task description for suggest")
    parser.add_argument("--keyword", help="Keyword filter for list")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.set_defaults(func=cmd_verticals)
