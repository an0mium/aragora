"""
Aragora OpenClaw CLI.

Commands for managing OpenClaw Enterprise Gateway deployments:
- init: Scaffold secure OpenClaw deployment
- status: Check gateway health
- policy: Manage policy rules
- audit: Query audit trail

Usage:
    aragora openclaw init --output-dir ./deploy --template enterprise
    aragora openclaw status
    aragora openclaw policy list
    aragora openclaw policy validate policy.yaml
    aragora openclaw audit --user-id user-123 --limit 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")

# Template content embedded directly to avoid importlib.resources complexity
DOCKER_COMPOSE_BASIC = """\
# Aragora + OpenClaw Basic Deployment
# Usage: docker-compose up -d

version: '3.8'

services:
  aragora-gateway:
    image: aragora/gateway:latest
    ports:
      - "8080:8080"
    environment:
      ARAGORA_ENV: production
      OPENCLAW_URL: http://openclaw:3000
      POLICY_FILE: /etc/aragora/policies/policy.yaml
      DEFAULT_POLICY: deny
    volumes:
      - ./policies:/etc/aragora/policies:ro
    depends_on:
      - openclaw
    networks:
      - aragora-net

  openclaw:
    image: openclawai/openclaw:latest
    environment:
      DISPLAY: ":99"
      OPENCLAW_PORT: 3000
      WORKSPACE_ROOT: /workspace
    volumes:
      - openclaw-workspace:/workspace
    networks:
      - aragora-net
    cap_drop:
      - ALL
    cap_add:
      - SYS_ADMIN

volumes:
  openclaw-workspace:

networks:
  aragora-net:
    driver: bridge
"""

POLICY_STRICT = """\
# Strict Enterprise Policy
version: 1
default_decision: deny

rules:
  - name: block_system_directories
    action_types: [file_read, file_write, file_delete]
    decision: deny
    priority: 100
    path_patterns: ["/etc/**", "/sys/**", "/proc/**", "/root/**"]

  - name: block_dangerous_commands
    action_types: [shell]
    decision: deny
    priority: 100
    command_deny_patterns:
      - "rm\\\\s+-rf\\\\s+/"
      - "sudo"
      - "mkfs"

  - name: approve_elevated_commands
    action_types: [shell]
    decision: require_approval
    priority: 50
    command_patterns: ["^sudo\\\\s+"]

  - name: allow_workspace_files
    action_types: [file_read, file_write, file_delete]
    decision: allow
    priority: 10
    workspace_only: true

  - name: allow_safe_commands
    action_types: [shell]
    decision: allow
    priority: 10
    command_patterns:
      - "^(ls|cat|head|tail|grep|find|wc|echo|pwd)\\\\s+"
      - "^(python|node|npm|pip|git)\\\\s+"

  - name: rate_limit_browser
    action_types: [browser]
    decision: allow
    priority: 5
    rate_limit: 30
    rate_limit_window: 60
"""

POLICY_PERMISSIVE = """\
# Permissive Development Policy
version: 1
default_decision: allow

rules:
  - name: block_system_directories
    action_types: [file_read, file_write, file_delete]
    decision: deny
    priority: 100
    path_patterns: ["/etc/shadow", "/etc/sudoers", "/root/**"]

  - name: block_dangerous_commands
    action_types: [shell]
    decision: deny
    priority: 100
    command_deny_patterns:
      - "rm\\\\s+-rf\\\\s+/"
      - "mkfs"
      - "dd\\\\s+if=.*of=/dev/"
"""

GATEWAY_ENV = """\
# Aragora Gateway Configuration
ARAGORA_ENV=production
ARAGORA_PORT=8080
ARAGORA_LOG_LEVEL=info

# OpenClaw Backend
OPENCLAW_URL=http://openclaw:3000
OPENCLAW_TIMEOUT=30

# Policy
POLICY_FILE=/etc/aragora/policies/policy.yaml
DEFAULT_POLICY=deny

# Security (generate your own keys!)
# SIGNING_KEY=your-signing-key
# ENCRYPTION_KEY=your-encryption-key

# Optional: API keys
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
"""


# =============================================================================
# Init Command
# =============================================================================


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize OpenClaw enterprise deployment."""
    output_dir = Path(args.output_dir)
    template = args.template
    policy_preset = args.policy_preset
    force = args.force
    dry_run = args.dry_run

    files_to_create = {
        "docker-compose.yml": DOCKER_COMPOSE_BASIC,
        "policies/policy.yaml": POLICY_STRICT if policy_preset == "strict" else POLICY_PERMISSIVE,
        "gateway.env": GATEWAY_ENV,
    }

    # For enterprise template, use the full docker-compose from deploy/
    if template == "enterprise":
        enterprise_compose = (
            Path(__file__).parent.parent.parent / "deploy" / "openclaw" / "docker-compose.yml"
        )
        if enterprise_compose.exists():
            files_to_create["docker-compose.yml"] = enterprise_compose.read_text()

        enterprise_policy = (
            Path(__file__).parent.parent.parent
            / "deploy"
            / "openclaw"
            / "policies"
            / "enterprise.yaml"
        )
        if enterprise_policy.exists():
            files_to_create["policies/policy.yaml"] = enterprise_policy.read_text()

    if dry_run:
        print(f"\nWould create deployment in: {output_dir}")
        print(f"Template: {template}")
        print(f"Policy preset: {policy_preset}")
        print("\nFiles to create:")
        for name in files_to_create:
            print(f"  {output_dir / name}")
        return 0

    # Check for existing files
    if output_dir.exists() and not force:
        existing = [f for f in files_to_create if (output_dir / f).exists()]
        if existing:
            print(f"Error: Files already exist in {output_dir}:")
            for f in existing:
                print(f"  {f}")
            print("\nUse --force to overwrite.")
            return 1

    # Create files
    print(f"\nInitializing OpenClaw enterprise deployment in {output_dir}")
    print(f"Template: {template}")
    print(f"Policy: {policy_preset}")
    print()

    for name, content in files_to_create.items():
        filepath = output_dir / name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        print(f"  Created {filepath}")

    print(f"\nDeployment scaffolded in {output_dir}")
    print("\nNext steps:")
    print(f"  1. Review and customize {output_dir / 'policies/policy.yaml'}")
    print(f"  2. Set environment variables in {output_dir / 'gateway.env'}")
    print(f"  3. Run: cd {output_dir} && docker-compose up -d")
    print("  4. Check status: aragora openclaw status")

    return 0


# =============================================================================
# Status Command
# =============================================================================


def cmd_status(args: argparse.Namespace) -> int:
    """Check OpenClaw gateway status."""
    import httpx

    server_url = args.server
    print("\n" + "=" * 60)
    print("OPENCLAW ENTERPRISE GATEWAY STATUS")
    print("=" * 60)

    # Check gateway health
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"\nGateway: ONLINE ({server_url})")
    except (httpx.HTTPError, OSError):
        print(f"\nGateway: OFFLINE ({server_url})")
        print("  Start the server with: aragora serve")
        return 1

    # Check proxy stats
    try:
        resp = httpx.get(
            f"{server_url}/api/v1/openclaw/stats",
            headers={"Accept": "application/json"},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            print("\nProxy Statistics:")
            print(f"  Active sessions: {data.get('active_sessions', 0)}")
            print(f"  Actions allowed: {data.get('actions_allowed', 0)}")
            print(f"  Actions denied:  {data.get('actions_denied', 0)}")
            print(f"  Pending approvals: {data.get('pending_approvals', 0)}")
            print(f"  Policy rules:  {data.get('policy_rules', 0)}")
    except (httpx.HTTPError, OSError):
        pass

    return 0


# =============================================================================
# Policy Commands
# =============================================================================


def cmd_policy(args: argparse.Namespace) -> int:
    """Manage OpenClaw policy rules."""
    action = args.policy_action

    if action == "list":
        return _policy_list(args)
    elif action == "validate":
        return _policy_validate(args)
    else:
        print("Usage: aragora openclaw policy {list|validate}")
        return 1


def _policy_list(args: argparse.Namespace) -> int:
    """List active policy rules."""
    import httpx

    server_url = args.server

    try:
        resp = httpx.get(
            f"{server_url}/api/v1/openclaw/policy/rules",
            headers={"Accept": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        rules = data.get("rules", [])

        print(f"\nPolicy Rules ({len(rules)} total)")
        print("-" * 70)
        print(f"{'Name':<30} {'Decision':<15} {'Priority':<10} {'Actions'}")
        print("-" * 70)

        for rule in rules:
            name = rule.get("name", "")[:29]
            decision = rule.get("decision", "")
            priority = rule.get("priority", 0)
            actions = ", ".join(rule.get("action_types", []))
            print(f"{name:<30} {decision:<15} {priority:<10} {actions}")

    except (httpx.HTTPError, OSError) as e:
        print(f"\nError: Could not reach gateway at {server_url}")
        print(f"  {e}")

        # Fallback: try to load from local file
        print("\nTip: You can validate a local policy file with:")
        print("  aragora openclaw policy validate <path-to-policy.yaml>")
        return 1

    return 0


def _policy_validate(args: argparse.Namespace) -> int:
    """Validate a policy YAML file."""
    policy_file = getattr(args, "file", None)
    if not policy_file:
        print("Usage: aragora openclaw policy validate <policy-file.yaml>")
        return 1

    policy_path = Path(policy_file)
    if not policy_path.exists():
        print(f"Error: File not found: {policy_path}")
        return 1

    try:
        from aragora.gateway.openclaw_policy import OpenClawPolicy

        policy = OpenClawPolicy(policy_file=str(policy_path))
        rules = policy.get_rules()

        print(f"\nPolicy file is valid: {policy_path}")
        print(f"Rules: {len(rules)}")
        print(f"Default decision: {policy._default_decision.value}")
        print()

        for rule in rules:
            actions = ", ".join(at.value for at in rule.action_types)
            print(f"  [{rule.priority:>3}] {rule.name:<30} {rule.decision.value:<15} ({actions})")

        return 0

    except Exception as e:
        print(f"\nPolicy validation failed: {e}")
        return 1


# =============================================================================
# Audit Command
# =============================================================================


def cmd_audit(args: argparse.Namespace) -> int:
    """Query audit trail."""
    import httpx

    server_url = args.server
    params: dict[str, Any] = {}

    if args.user_id:
        params["user_id"] = args.user_id
    if args.session_id:
        params["session_id"] = args.session_id
    if args.event_type:
        params["event_type"] = args.event_type
    if args.limit:
        params["limit"] = args.limit

    try:
        resp = httpx.get(
            f"{server_url}/api/v1/openclaw/audit",
            params=params,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])

        print(f"\nAudit Trail ({len(records)} records)")
        print("-" * 80)
        print(f"{'Timestamp':<22} {'Event':<20} {'User':<15} {'Action':<12} {'Status'}")
        print("-" * 80)

        for record in records:
            from datetime import datetime

            ts = record.get("timestamp", 0)
            try:
                ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError):
                ts_str = str(ts)[:21]

            event = record.get("event_type", "")[:19]
            user = (record.get("user_id") or "")[:14]
            action = (record.get("action_type") or "")[:11]
            success = "OK" if record.get("success", True) else "FAIL"

            print(f"{ts_str:<22} {event:<20} {user:<15} {action:<12} {success}")

    except (httpx.HTTPError, OSError) as e:
        print(f"\nError: Could not reach gateway at {server_url}")
        print(f"  {e}")
        return 1

    return 0


def _cmd_serve_gateway(args: argparse.Namespace) -> int:
    """Launch the standalone OpenClaw governance gateway."""
    from aragora.compat.openclaw.standalone import cmd_openclaw_serve

    cmd_openclaw_serve(args)
    return 0


# =============================================================================
# Parser Registration
# =============================================================================


def create_openclaw_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create openclaw subcommand parser."""
    openclaw_parser = subparsers.add_parser(
        "openclaw",
        help="OpenClaw Enterprise Gateway management",
        description="""
OpenClaw Enterprise Gateway - secure proxy for OpenClaw deployments.

Commands:
    init      - Initialize a secure OpenClaw deployment
    status    - Check gateway health and statistics
    policy    - Manage policy rules
    audit     - Query audit trail

Examples:
    aragora openclaw init --output-dir ./deploy --template enterprise
    aragora openclaw status
    aragora openclaw policy validate ./policy.yaml
    aragora openclaw audit --user-id user-123
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    openclaw_subparsers = openclaw_parser.add_subparsers(dest="openclaw_action")

    # Init command
    init_parser = openclaw_subparsers.add_parser(
        "init",
        help="Initialize OpenClaw enterprise deployment",
    )
    init_parser.add_argument(
        "--output-dir",
        "-o",
        default="./openclaw-deploy",
        help="Output directory (default: ./openclaw-deploy)",
    )
    init_parser.add_argument(
        "--template",
        "-t",
        default="basic",
        choices=["basic", "enterprise"],
        help="Deployment template (default: basic)",
    )
    init_parser.add_argument(
        "--policy-preset",
        default="strict",
        choices=["strict", "permissive"],
        help="Policy preset (default: strict)",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without creating",
    )
    init_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files",
    )
    init_parser.set_defaults(func=cmd_init)

    # Status command
    status_parser = openclaw_subparsers.add_parser(
        "status",
        help="Check gateway health",
    )
    status_parser.add_argument(
        "--server",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    status_parser.set_defaults(func=cmd_status)

    # Policy command
    policy_parser = openclaw_subparsers.add_parser(
        "policy",
        help="Manage policy rules",
    )
    policy_subparsers = policy_parser.add_subparsers(dest="policy_action")

    policy_list = policy_subparsers.add_parser("list", help="List active rules")
    policy_list.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    policy_list.set_defaults(func=cmd_policy)

    policy_validate = policy_subparsers.add_parser("validate", help="Validate policy file")
    policy_validate.add_argument("file", help="Path to policy YAML file")
    policy_validate.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    policy_validate.set_defaults(func=cmd_policy)

    # Audit command
    audit_parser = openclaw_subparsers.add_parser(
        "audit",
        help="Query audit trail",
    )
    audit_parser.add_argument("--user-id", "-u", help="Filter by user ID")
    audit_parser.add_argument("--session-id", "-s", help="Filter by session ID")
    audit_parser.add_argument("--event-type", "-e", help="Filter by event type")
    audit_parser.add_argument(
        "--limit", "-l", type=int, default=50, help="Max results (default: 50)"
    )
    audit_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    audit_parser.set_defaults(func=cmd_audit)

    # Serve command -- standalone gateway
    serve_parser = openclaw_subparsers.add_parser(
        "serve",
        help="Run standalone OpenClaw governance gateway",
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", "-p", type=int, default=8100,
        help="Port to listen on (default: 8100)",
    )
    serve_parser.add_argument(
        "--policy", help="Path to policy YAML file",
    )
    serve_parser.add_argument(
        "--default-policy", default="deny", choices=["allow", "deny"],
        help="Default policy when no rule matches (default: deny)",
    )
    serve_parser.add_argument(
        "--cors", default="*", help="CORS allowed origins (comma-separated)",
    )
    serve_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    serve_parser.set_defaults(func=_cmd_serve_gateway)
