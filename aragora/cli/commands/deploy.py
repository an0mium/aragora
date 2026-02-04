"""
Deployment CLI commands.

Provides CLI access to deployment validation and security configuration.
Commands:
- gt deploy validate - Validate deployment readiness
- gt deploy secrets - Generate security secrets
- gt deploy status - Show deployment status summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def cmd_deploy(args: argparse.Namespace) -> None:
    """Handle 'deploy' command - dispatch to subcommands."""
    subcommand = getattr(args, "deploy_command", None)

    if subcommand == "validate":
        asyncio.run(_cmd_validate(args))
    elif subcommand == "secrets":
        _cmd_secrets(args)
    elif subcommand == "status":
        asyncio.run(_cmd_status(args))
    else:
        # Default: show help
        print("\nUsage: aragora deploy <command>")
        print("\nCommands:")
        print("  validate             Validate deployment readiness")
        print("  secrets              Generate security secrets")
        print("  status               Show deployment status summary")
        print("\nOptions for 'validate':")
        print("  --strict             Fail on critical issues (exit code 1)")
        print("  --production         Check production-specific requirements")
        print("  --json               Output as JSON")
        print("\nOptions for 'secrets':")
        print("  --type <type>        Secret type: jwt, encryption, api_token, all")
        print("  --output <file>      Write to file instead of stdout")
        print("  --format <format>    Output format: env, json, plain")


async def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate deployment readiness."""
    strict = getattr(args, "strict", False)
    production = getattr(args, "production", False)
    as_json = getattr(args, "json", False)

    try:
        from aragora.ops.deployment_validator import (
            validate_deployment,
            DeploymentNotReadyError,
            Severity,
        )

        if not as_json:
            print("\nValidating deployment...")
            print("=" * 60)

        env_backup: str | None = None
        env_set = False
        if production:
            env_backup = os.environ.get("ARAGORA_ENV")
            os.environ["ARAGORA_ENV"] = "production"
            env_set = True

        try:
            result = await validate_deployment(strict=strict)
        except DeploymentNotReadyError as e:
            result = e.result
        finally:
            if env_set:
                if env_backup is None:
                    os.environ.pop("ARAGORA_ENV", None)
                else:
                    os.environ["ARAGORA_ENV"] = env_backup

        if as_json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
            if not result.ready:
                sys.exit(1)
            return

        # Display results
        status = "READY" if result.ready else "NOT READY"
        live = "LIVE" if result.live else "NOT LIVE"

        print(f"\n  Status:     {status}")
        print(f"  Live:       {live}")
        print(f"  Duration:   {result.validation_duration_ms:.0f}ms")

        # Group issues by severity
        critical = [i for i in result.issues if i.severity == Severity.CRITICAL]
        warnings = [i for i in result.issues if i.severity == Severity.WARNING]
        info = [i for i in result.issues if i.severity == Severity.INFO]

        if critical:
            print(f"\n  CRITICAL ISSUES ({len(critical)}):")
            for issue in critical:
                print(f"    [-] {issue.component}: {issue.message}")
                if issue.suggestion:
                    print(f"        Suggestion: {issue.suggestion}")

        if warnings:
            print(f"\n  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"    [!] {issue.component}: {issue.message}")
                if issue.suggestion:
                    print(f"        Suggestion: {issue.suggestion}")

        if info and not as_json:
            print(f"\n  INFO ({len(info)}):")
            for issue in info[:5]:  # Limit to 5
                print(f"    [i] {issue.component}: {issue.message}")
            if len(info) > 5:
                print(f"    ... and {len(info) - 5} more")

        # Component health
        print("\n  COMPONENTS:")
        for comp in result.components:
            status_icon = {
                "healthy": "+",
                "degraded": "!",
                "unhealthy": "-",
                "unknown": "?",
            }.get(comp.status.value, "?")
            latency = f" ({comp.latency_ms:.0f}ms)" if comp.latency_ms else ""
            print(f"    [{status_icon}] {comp.name}: {comp.status.value}{latency}")

        print()

        if not result.ready and strict:
            print("Deployment validation FAILED (--strict mode)")
            sys.exit(1)

    except ImportError as e:
        print(f"\nError: Failed to import deployment validator: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Validation failed")
        print(f"\nError: {e}")
        sys.exit(1)


def _cmd_secrets(args: argparse.Namespace) -> None:
    """Generate security secrets."""
    secret_type = getattr(args, "type", "all")
    output_file = getattr(args, "output", None)
    output_format = getattr(args, "format", "env")

    generated: dict[str, str] = {}

    # Generate requested secrets
    if secret_type in ("jwt", "all"):
        # JWT secret: 64 character hex string (256 bits)
        jwt_secret = secrets.token_hex(32)
        generated["ARAGORA_JWT_SECRET"] = jwt_secret

    if secret_type in ("encryption", "all"):
        # Encryption key: 32 bytes base64 encoded (for AES-256)
        import base64

        encryption_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        generated["ARAGORA_ENCRYPTION_KEY"] = encryption_key

    if secret_type in ("api_token", "all"):
        # API token: URL-safe token
        api_token = secrets.token_urlsafe(32)
        generated["ARAGORA_API_TOKEN"] = api_token

    if secret_type in ("session", "all"):
        # Session secret: 48 bytes hex
        session_secret = secrets.token_hex(24)
        generated["ARAGORA_SESSION_SECRET"] = session_secret

    if not generated:
        print(f"\nError: Unknown secret type: {secret_type}")
        print("Valid types: jwt, encryption, api_token, session, all")
        return

    # Format output
    if output_format == "env":
        lines = [f"{k}={v}" for k, v in generated.items()]
        output = "\n".join(lines) + "\n"
    elif output_format == "json":
        output = json.dumps(generated, indent=2)
    else:  # plain
        output = "\n".join(generated.values())

    # Write or print
    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\nSecrets written to: {output_file}")
        print(f"  Generated {len(generated)} secret(s)")
        print("\nIMPORTANT: Keep this file secure and add to .gitignore!")
    else:
        if output_format == "env":
            print("\n# Generated secrets - add to .env file")
            print("# KEEP SECURE - DO NOT COMMIT TO VERSION CONTROL")
            print()
        print(output)

        if output_format == "env":
            print("\nUsage:")
            print("  1. Add these to your .env file")
            print("  2. Ensure .env is in .gitignore")
            print("  3. For production, use a secrets manager instead")


async def _cmd_status(args: argparse.Namespace) -> None:
    """Show deployment status summary."""
    as_json = getattr(args, "json", False)

    try:
        import os
        from aragora.config.settings import get_settings

        settings = get_settings()
        env = os.environ.get("ARAGORA_ENV", "development")

        # Collect status information
        status: dict[str, Any] = {
            "environment": env,
            "timestamp": datetime.now().isoformat(),
            "config": {},
            "secrets": {},
            "services": {},
        }

        # Config status
        status["config"]["server_host"] = getattr(settings, "server_host", "localhost")
        status["config"]["server_port"] = getattr(settings, "server_port", 8000)
        status["config"]["debug"] = getattr(settings, "debug", False)

        # Secrets status (presence only, not values)
        env_vars = [
            "ARAGORA_JWT_SECRET",
            "ARAGORA_API_TOKEN",
            "ARAGORA_ENCRYPTION_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                # Show partial for security
                status["secrets"][var] = f"***{value[-4:]}" if len(value) > 4 else "***"
            else:
                status["secrets"][var] = None

        # Service connectivity
        status["services"]["database"] = (
            "configured"
            if os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_URL")
            else "sqlite"
        )
        status["services"]["redis"] = (
            "configured" if os.environ.get("REDIS_URL") else "not configured"
        )

        if as_json:
            print(json.dumps(status, indent=2))
            return

        print("\n" + "=" * 60)
        print("DEPLOYMENT STATUS")
        print("=" * 60)

        print(f"\n  Environment:  {status['environment']}")
        print(
            f"  Server:       {status['config']['server_host']}:{status['config']['server_port']}"
        )
        print(f"  Debug:        {status['config']['debug']}")

        print("\n  Secrets:")
        for name, value in status["secrets"].items():
            icon = "+" if value else "-"
            display = value if value else "not set"
            print(f"    [{icon}] {name}: {display}")

        print("\n  Services:")
        for name, value in status["services"].items():
            icon = "+" if value != "not configured" else "-"
            print(f"    [{icon}] {name}: {value}")

        print()

    except Exception as e:
        print(f"\nError getting status: {e}")


def add_deploy_parser(subparsers: Any) -> None:
    """Add deploy subparser to CLI."""
    dp = subparsers.add_parser(
        "deploy",
        help="Deployment validation and configuration",
        description="Validate deployment readiness and manage security configuration",
    )
    dp.set_defaults(func=cmd_deploy)

    dp_sub = dp.add_subparsers(dest="deploy_command")

    # Validate
    validate_p = dp_sub.add_parser("validate", help="Validate deployment readiness")
    validate_p.add_argument("--strict", action="store_true", help="Fail on critical issues")
    validate_p.add_argument(
        "--production", action="store_true", help="Check production requirements"
    )
    validate_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Secrets
    secrets_p = dp_sub.add_parser("secrets", help="Generate security secrets")
    secrets_p.add_argument(
        "--type",
        choices=["jwt", "encryption", "api_token", "session", "all"],
        default="all",
        help="Secret type to generate (default: all)",
    )
    secrets_p.add_argument("--output", "-o", help="Output file path")
    secrets_p.add_argument(
        "--format",
        choices=["env", "json", "plain"],
        default="env",
        help="Output format (default: env)",
    )

    # Status
    status_p = dp_sub.add_parser("status", help="Show deployment status")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")
