# mypy: ignore-errors
"""
Gas Town CLI (gt) - Multi-agent orchestration command line interface.

Commands:
    gt convoy list              # List active convoys
    gt convoy create <title>    # Create convoy with beads
    gt convoy status <id>       # Convoy progress
    gt bead list [--status]     # List beads with filters
    gt bead assign <id> <agent> # Assign bead to agent
    gt agent list               # List agents with roles
    gt agent promote <id> MAYOR # Promote agent role
    gt witness status           # Patrol status
    gt workspace init           # Initialize GT workspace

Usage via aragora CLI:
    aragora gt convoy list
    aragora gt bead list --status pending

Note: This module uses Gas Town APIs that are still being developed.
Type checking is disabled due to API signature mismatches.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "N/A"


def _print_table(headers: List[str], rows: List[List[str]], widths: Optional[List[int]] = None):
    """Print a simple text table."""
    if not widths:
        widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


# =============================================================================
# Convoy Commands
# =============================================================================


def cmd_convoy_list(args: argparse.Namespace) -> int:
    """List active convoys."""
    try:
        from aragora.nomic.convoys import ConvoyManager, ConvoyStatus

        manager = ConvoyManager()  # type: ignore[call-arg]

        # Filter by status if specified
        status_filter = None
        if args.status:
            try:
                status_filter = ConvoyStatus(args.status)
            except ValueError:
                print(f"Invalid status: {args.status}")
                print(f"Valid: {', '.join(s.value for s in ConvoyStatus)}")
                return 1

        convoys = _run_async(manager.list_convoys(status=status_filter))

        if not convoys:
            print("No convoys found")
            return 0

        headers = ["ID", "Title", "Status", "Beads", "Progress", "Created"]
        rows = []
        for c in convoys[: args.limit]:
            progress = f"{c.completed_beads}/{c.total_beads}"
            rows.append(
                [
                    c.id[:8],
                    c.title[:30] if c.title else "Untitled",
                    c.status.value,
                    str(c.total_beads),
                    progress,
                    _format_timestamp(c.created_at),
                ]
            )

        print(f"Found {len(convoys)} convoy(s):\n")
        _print_table(headers, rows, [10, 32, 12, 8, 10, 20])
        return 0

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error listing convoys: {e}")
        return 1


def cmd_convoy_create(args: argparse.Namespace) -> int:
    """Create a new convoy."""
    try:
        from aragora.nomic.convoys import ConvoyManager, ConvoySpec  # type: ignore[attr-defined]
        from aragora.nomic.beads import BeadSpec, BeadPriority  # type: ignore[attr-defined]

        manager = ConvoyManager()  # type: ignore[call-arg]

        # Parse beads from comma-separated list
        bead_specs = []
        if args.beads:
            for bead_title in args.beads.split(","):
                bead_title = bead_title.strip()
                if bead_title:
                    bead_specs.append(
                        BeadSpec(
                            title=bead_title,
                            priority=BeadPriority[args.priority.upper()],
                        )
                    )

        if not bead_specs:
            print("Error: At least one bead is required (--beads task1,task2,...)")
            return 1

        spec = ConvoySpec(
            title=args.title,
            description=args.description or "",
            beads=bead_specs,
        )

        convoy = _run_async(manager.create_convoy(spec))  # type: ignore[call-arg]

        print("Convoy created successfully!")
        print(f"  ID: {convoy.id}")
        print(f"  Title: {convoy.title}")
        print(f"  Beads: {len(convoy.beads)}")
        print(f"  Status: {convoy.status.value}")
        return 0

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error creating convoy: {e}")
        return 1


def cmd_convoy_status(args: argparse.Namespace) -> int:
    """Get convoy status and progress."""
    try:
        from aragora.nomic.convoys import ConvoyManager

        manager = ConvoyManager()  # type: ignore[call-arg]
        convoy = _run_async(manager.get_convoy(args.convoy_id))

        if not convoy:
            print(f"Convoy not found: {args.convoy_id}")
            return 1

        print(f"Convoy: {convoy.title}")
        print(f"  ID: {convoy.id}")
        print(f"  Status: {convoy.status.value}")
        print(f"  Created: {_format_timestamp(convoy.created_at)}")
        print(f"  Progress: {convoy.completed_beads}/{convoy.total_beads} beads")

        if convoy.beads:
            print("\nBeads:")
            for bead in convoy.beads[:10]:
                status_icon = {
                    "pending": "[ ]",
                    "in_progress": "[~]",
                    "completed": "[x]",
                    "failed": "[!]",
                }.get(bead.status.value, "[ ]")
                print(f"  {status_icon} {bead.title[:50]}")

            if len(convoy.beads) > 10:
                print(f"  ... and {len(convoy.beads) - 10} more")

        return 0

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error getting convoy status: {e}")
        return 1


# =============================================================================
# Bead Commands
# =============================================================================


def cmd_bead_list(args: argparse.Namespace) -> int:
    """List beads with optional filters."""
    try:
        from aragora.nomic.beads import BeadManager, BeadStatus  # type: ignore[attr-defined]

        manager = BeadManager()  # type: ignore[misc]

        # Filter by status if specified
        status_filter = None
        if args.status:
            try:
                status_filter = BeadStatus(args.status)
            except ValueError:
                print(f"Invalid status: {args.status}")
                print(f"Valid: {', '.join(s.value for s in BeadStatus)}")
                return 1

        beads = _run_async(
            manager.list_beads(
                status=status_filter,
                convoy_id=args.convoy,
                limit=args.limit,
            )
        )

        if not beads:
            print("No beads found")
            return 0

        headers = ["ID", "Title", "Status", "Priority", "Assigned", "Convoy"]
        rows = []
        for b in beads:
            rows.append(
                [
                    b.id[:8],
                    b.title[:30] if b.title else "Untitled",
                    b.status.value,
                    b.priority.value if hasattr(b, "priority") else "normal",
                    b.assigned_to[:8] if b.assigned_to else "-",
                    b.convoy_id[:8] if b.convoy_id else "-",
                ]
            )

        print(f"Found {len(beads)} bead(s):\n")
        _print_table(headers, rows, [10, 32, 12, 10, 10, 10])
        return 0

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error listing beads: {e}")
        return 1


def cmd_bead_assign(args: argparse.Namespace) -> int:
    """Assign a bead to an agent."""
    try:
        from aragora.nomic.beads import BeadManager  # type: ignore[attr-defined]

        manager = BeadManager()  # type: ignore[misc]
        success = _run_async(manager.assign_bead(args.bead_id, args.agent_id))

        if success:
            print(f"Bead {args.bead_id} assigned to agent {args.agent_id}")
            return 0
        else:
            print("Failed to assign bead")
            return 1

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error assigning bead: {e}")
        return 1


# =============================================================================
# Agent Commands
# =============================================================================


def cmd_agent_list(args: argparse.Namespace) -> int:
    """List agents with their roles."""
    try:
        from aragora.nomic.agent_roles import AgentHierarchy, AgentRole

        hierarchy = AgentHierarchy()

        # Filter by role if specified
        role_filter = None
        if args.role:
            try:
                role_filter = AgentRole(args.role.lower())
            except ValueError:
                print(f"Invalid role: {args.role}")
                print(f"Valid: {', '.join(r.value for r in AgentRole)}")
                return 1

        agents = _run_async(hierarchy.list_agents(role=role_filter))  # type: ignore[attr-defined]

        if not agents:
            print("No agents found")
            return 0

        headers = ["Agent ID", "Role", "Supervised By", "Registered"]
        rows = []
        for a in agents:
            rows.append(
                [
                    a.agent_id[:20],
                    a.role.value.upper(),
                    a.supervised_by[:10] if a.supervised_by else "-",
                    _format_timestamp(a.assigned_at),
                ]
            )

        print(f"Found {len(agents)} agent(s):\n")
        _print_table(headers, rows, [22, 10, 12, 20])
        return 0

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error listing agents: {e}")
        return 1


def cmd_agent_promote(args: argparse.Namespace) -> int:
    """Promote an agent to a new role."""
    try:
        from aragora.nomic.agent_roles import AgentHierarchy, AgentRole

        hierarchy = AgentHierarchy()

        # Parse role
        try:
            new_role = AgentRole(args.role.lower())
        except ValueError:
            print(f"Invalid role: {args.role}")
            print(f"Valid: {', '.join(r.value for r in AgentRole)}")
            return 1

        # Update role
        success = _run_async(hierarchy.update_agent_role(args.agent_id, new_role))  # type: ignore[attr-defined]

        if success:
            print(f"Agent {args.agent_id} promoted to {new_role.value.upper()}")
            return 0
        else:
            print("Failed to promote agent (not found?)")
            return 1

    except ImportError as e:
        print(f"Gas Town modules not available: {e}")
        return 1
    except Exception as e:
        print(f"Error promoting agent: {e}")
        return 1


# =============================================================================
# Witness Commands
# =============================================================================


def cmd_witness_status(args: argparse.Namespace) -> int:
    """Get witness patrol status."""
    try:
        from aragora.server.startup import get_witness_behavior

        witness = get_witness_behavior()
        if not witness:
            print("Witness patrol not initialized")
            print("Start the server to enable witness patrol")
            return 1

        print("Witness Patrol Status")
        print("=" * 40)
        print(f"  Patrolling: {'Yes' if witness._running else 'No'}")
        print(f"  Patrol Interval: {witness.config.patrol_interval_seconds}s")
        print(f"  Heartbeat Timeout: {witness.config.heartbeat_timeout_seconds}s")

        # Try to get health report
        try:
            report = _run_async(witness.generate_health_report())
            print(f"\nHealth Report ({report.report_id[:8]}):")
            print(f"  Overall Status: {report.overall_status.value}")
            print(f"  Agents Checked: {len(report.agent_checks)}")
            print(f"  Convoys Checked: {len(report.convoy_checks)}")
            print(f"  Active Alerts: {len(report.alerts)}")

            if report.recommendations:
                print("\nRecommendations:")
                for rec in report.recommendations[:5]:
                    print(f"  - {rec}")
        except Exception as e:
            logger.debug(f"Could not generate health report: {e}")

        return 0

    except ImportError as e:
        print(f"Witness module not available: {e}")
        return 1
    except Exception as e:
        print(f"Error getting witness status: {e}")
        return 1


# =============================================================================
# Workspace Commands
# =============================================================================


def cmd_workspace_init(args: argparse.Namespace) -> int:
    """Initialize a Gas Town workspace."""
    workspace_dir = Path(args.directory or ".").resolve()

    # Create GT directories
    gt_dirs = [
        workspace_dir / ".gt",
        workspace_dir / ".gt" / "convoys",
        workspace_dir / ".gt" / "beads",
        workspace_dir / ".gt" / "agents",
        workspace_dir / ".gt" / "hooks",
    ]

    for d in gt_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_file = workspace_dir / ".gt" / "config.json"
    if not config_file.exists() or args.force:
        config = {
            "version": "1.0",
            "workspace": str(workspace_dir),
            "created_at": datetime.now().isoformat(),
            "settings": {
                "auto_assign_beads": True,
                "max_concurrent_agents": 5,
                "witness_patrol_enabled": True,
            },
        }
        config_file.write_text(json.dumps(config, indent=2))

    print(f"Gas Town workspace initialized at: {workspace_dir}")
    print(f"  Config: {config_file}")
    print("\nNext steps:")
    print("  aragora gt convoy create 'My First Convoy' --beads task1,task2")
    print("  aragora gt agent list")
    return 0


# =============================================================================
# Main GT Command Handler
# =============================================================================


def add_gt_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add GT subcommand to the main CLI parser."""
    gt_parser = subparsers.add_parser(
        "gt",
        help="Gas Town multi-agent orchestration",
        description="""
Gas Town (GT) - Multi-agent orchestration for complex tasks.

Organize work into Convoys (batch orders) containing Beads (work units).
Agents are assigned roles: MAYOR (coordinator), WITNESS (monitor),
POLECAT (ephemeral worker), or CREW (persistent worker).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    gt_subparsers = gt_parser.add_subparsers(dest="gt_command", help="GT commands")

    # --- Convoy commands ---
    convoy_parser = gt_subparsers.add_parser("convoy", help="Manage convoys")
    convoy_subparsers = convoy_parser.add_subparsers(dest="convoy_action")

    # convoy list
    convoy_list = convoy_subparsers.add_parser("list", help="List convoys")
    convoy_list.add_argument("--status", "-s", help="Filter by status")
    convoy_list.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    convoy_list.set_defaults(func=cmd_convoy_list)

    # convoy create
    convoy_create = convoy_subparsers.add_parser("create", help="Create a convoy")
    convoy_create.add_argument("title", help="Convoy title")
    convoy_create.add_argument("--beads", "-b", required=True, help="Comma-separated bead titles")
    convoy_create.add_argument("--description", "-d", help="Convoy description")
    convoy_create.add_argument(
        "--priority", "-p", default="normal", choices=["low", "normal", "high", "critical"]
    )
    convoy_create.set_defaults(func=cmd_convoy_create)

    # convoy status
    convoy_status = convoy_subparsers.add_parser("status", help="Get convoy status")
    convoy_status.add_argument("convoy_id", help="Convoy ID")
    convoy_status.set_defaults(func=cmd_convoy_status)

    # --- Bead commands ---
    bead_parser = gt_subparsers.add_parser("bead", help="Manage beads")
    bead_subparsers = bead_parser.add_subparsers(dest="bead_action")

    # bead list
    bead_list = bead_subparsers.add_parser("list", help="List beads")
    bead_list.add_argument("--status", "-s", help="Filter by status")
    bead_list.add_argument("--convoy", "-c", help="Filter by convoy ID")
    bead_list.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    bead_list.set_defaults(func=cmd_bead_list)

    # bead assign
    bead_assign = bead_subparsers.add_parser("assign", help="Assign bead to agent")
    bead_assign.add_argument("bead_id", help="Bead ID")
    bead_assign.add_argument("agent_id", help="Agent ID")
    bead_assign.set_defaults(func=cmd_bead_assign)

    # --- Agent commands ---
    agent_parser = gt_subparsers.add_parser("agent", help="Manage agents")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_action")

    # agent list
    agent_list = agent_subparsers.add_parser("list", help="List agents")
    agent_list.add_argument("--role", "-r", help="Filter by role (mayor, witness, polecat, crew)")
    agent_list.set_defaults(func=cmd_agent_list)

    # agent promote
    agent_promote = agent_subparsers.add_parser("promote", help="Promote agent to role")
    agent_promote.add_argument("agent_id", help="Agent ID")
    agent_promote.add_argument("role", help="New role (mayor, witness, polecat, crew)")
    agent_promote.set_defaults(func=cmd_agent_promote)

    # --- Witness commands ---
    witness_parser = gt_subparsers.add_parser("witness", help="Witness patrol operations")
    witness_subparsers = witness_parser.add_subparsers(dest="witness_action")

    # witness status
    witness_status = witness_subparsers.add_parser("status", help="Get patrol status")
    witness_status.set_defaults(func=cmd_witness_status)

    # --- Workspace commands ---
    workspace_parser = gt_subparsers.add_parser("workspace", help="Workspace management")
    workspace_subparsers = workspace_parser.add_subparsers(dest="workspace_action")

    # workspace init
    workspace_init = workspace_subparsers.add_parser("init", help="Initialize workspace")
    workspace_init.add_argument("directory", nargs="?", help="Target directory")
    workspace_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    workspace_init.set_defaults(func=cmd_workspace_init)


def cmd_gt(args: argparse.Namespace) -> int:
    """Handle the 'gt' command group."""
    if not hasattr(args, "func") or args.func is None:
        # No subcommand specified - show help
        print("Gas Town CLI - Multi-agent orchestration")
        print("\nUsage: aragora gt <command> [options]")
        print("\nCommands:")
        print("  convoy    Manage convoys (batch work orders)")
        print("  bead      Manage beads (individual work units)")
        print("  agent     Manage agents and roles")
        print("  witness   Witness patrol operations")
        print("  workspace Workspace management")
        print("\nRun 'aragora gt <command> --help' for more information")
        return 0

    return args.func(args)
