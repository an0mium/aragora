"""
Delegated CLI commands.

These are thin wrappers that delegate to implementations in other
aragora.cli submodules. They exist here so all command handlers
can be imported from a single location.
"""

import argparse
import os

import httpx

from aragora.security.safe_http import safe_get

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def cmd_agents(args: argparse.Namespace) -> None:
    """Handle 'agents' command - list available agents and their configuration."""
    from aragora.cli.agents import main as agents_main

    agents_main(args)


def cmd_control_plane(args: argparse.Namespace) -> None:
    """Handle 'control-plane' command - show control plane status and management."""
    server_url = getattr(args, "server", DEFAULT_API_URL)
    subcommand = getattr(args, "subcommand", "status")

    print("\n" + "=" * 60)
    print("ARAGORA CONTROL PLANE")
    print("=" * 60)

    if subcommand in (None, "status"):
        # Get control plane metrics
        try:
            resp = safe_get(
                f"{server_url}/api/v1/control-plane/metrics",
                headers={"Accept": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            print("\nControl Plane Status: ONLINE")
            print("\nAgents:")
            print(f"  Total:     {data.get('total_agents', 0)}")
            print(f"  Available: {data.get('agents_available', 0)}")
            print(f"  Busy:      {data.get('agents_busy', 0)}")

            print("\nJobs:")
            print(f"  Active:    {data.get('active_jobs', 0)}")
            print(f"  Queued:    {data.get('queued_jobs', 0)}")
            print(f"  Completed: {data.get('completed_jobs', 0)}")

            print("\nActivity (24h):")
            print(f"  Documents: {data.get('documents_processed_today', 0)}")
            print(f"  Audits:    {data.get('audits_completed_today', 0)}")

        except (httpx.HTTPError, OSError, TimeoutError) as e:
            print("\nControl Plane Status: OFFLINE")
            print(f"  Server not reachable at {server_url}")
            print(f"  Error: {e}")
            print("\n  Start the server with: aragora serve")

    elif subcommand == "agents":
        # List registered agents
        try:
            resp = safe_get(
                f"{server_url}/api/v1/control-plane/agents",
                headers={"Accept": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            agents = data.get("agents", [])
            print(f"\nRegistered Agents: {len(agents)}")
            print("-" * 40)
            for agent in agents:
                status = agent.get("status", "unknown")
                status_icon = "+" if status == "idle" else ("*" if status == "busy" else "-")
                print(f"  [{status_icon}] {agent.get('id', 'unknown')}")
                print(f"      Type: {agent.get('type', 'unknown')}")
                print(f"      Status: {status}")

        except (httpx.HTTPError, OSError, TimeoutError) as e:
            print(f"\nCannot reach control plane at {server_url}")
            print(f"  Error: {e}")

    elif subcommand == "channels":
        # List connected channels
        try:
            resp = safe_get(
                f"{server_url}/api/v1/integrations/status",
                headers={"Accept": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            print("\nConnected Channels:")
            print("-" * 40)
            channels = data.get("channels", data.get("integrations", []))
            if channels:
                for channel in channels:
                    name = channel.get("name", channel.get("type", "unknown"))
                    status = channel.get("status", "unknown")
                    icon = "+" if status in ("connected", "healthy", "ok") else "-"
                    print(f"  [{icon}] {name}: {status}")
            else:
                print("  No channels configured")
                print("  Configure in: CLAUDE.md or via API")

        except (httpx.HTTPError, OSError, TimeoutError) as e:
            print(f"\nCannot reach control plane at {server_url}")
            print(f"  Error: {e}")

    print("\n" + "=" * 60)


def cmd_demo(args: argparse.Namespace) -> None:
    """Handle 'demo' command - run a quick compelling demo."""
    from aragora.cli.demo import main as demo_main

    demo_main(args)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle 'export' command - export debate artifacts."""
    from aragora.cli.export import main as export_main

    export_main(args)


def cmd_init(args: argparse.Namespace) -> None:
    """Handle 'init' command - project scaffolding."""
    from aragora.cli.init import cmd_init as init_handler

    init_handler(args)


def cmd_setup(args: argparse.Namespace) -> None:
    """Run interactive setup wizard."""
    from aragora.cli.setup import cmd_setup as setup_handler

    setup_handler(args)


def cmd_repl(args: argparse.Namespace) -> None:
    """Handle 'repl' command - interactive debate mode."""
    from aragora.cli.repl import cmd_repl as repl_handler

    repl_handler(args)


def cmd_config(args: argparse.Namespace) -> None:
    """Handle 'config' command - manage configuration."""
    from aragora.cli.config import cmd_config as config_handler

    config_handler(args)


def cmd_replay(args: argparse.Namespace) -> None:
    """Handle 'replay' command - replay stored debates."""
    from aragora.cli.replay import cmd_replay as replay_handler

    replay_handler(args)


def cmd_bench(args: argparse.Namespace) -> None:
    """Handle 'bench' command - benchmark agents."""
    from aragora.cli.bench import cmd_bench as bench_handler

    bench_handler(args)


def cmd_review(args: argparse.Namespace) -> int:
    """Handle 'review' command - AI red team code review."""
    from aragora.cli.review import cmd_review as review_handler

    return review_handler(args)


def cmd_gauntlet(args: argparse.Namespace) -> None:
    """Handle 'gauntlet' command - adversarial stress-testing."""
    from aragora.cli.gauntlet import cmd_gauntlet as gauntlet_handler

    return gauntlet_handler(args)


def cmd_badge(args) -> None:
    """Generate badge markdown for README."""
    from aragora.cli.badge import main as badge_main

    badge_main(args)


def cmd_billing(args: argparse.Namespace) -> int:
    """Handle 'billing' command - manage billing and usage."""
    from aragora.cli.billing import main as billing_main

    return billing_main(args)


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Handle 'mcp-server' command - run MCP server."""
    try:
        from aragora.mcp.server import main as mcp_main  # type: ignore[attr-defined]

        mcp_main()
    except ImportError as e:
        print("\nError: MCP dependencies not installed")
        print(f"\nMissing: {e}")
        print("\nTo install MCP support:")
        print("  pip install mcp")
        print("  # or")
        print("  pip install aragora[mcp]")
        print(
            "\nMCP (Model Context Protocol) enables integration with Claude Desktop and other MCP clients."
        )


def cmd_marketplace(args: argparse.Namespace) -> None:
    """Handle 'marketplace' command - manage agent templates."""
    from aragora.cli.marketplace import marketplace

    # Build args list from parsed args
    click_args = []
    if hasattr(args, "subcommand") and args.subcommand:
        click_args.append(args.subcommand)
    if hasattr(args, "args") and args.args:
        click_args.extend(args.args)

    # Call click command
    marketplace(click_args, standalone_mode=False)
