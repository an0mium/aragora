"""
Environment status and validation CLI commands.

Contains commands for checking environment health, validating API keys,
and verifying backend connectivity.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def cmd_status(args: argparse.Namespace) -> None:
    """Handle 'status' command - show environment health and agent availability."""
    import os
    import shutil

    print("\nAragora Environment Status")
    print("=" * 60)

    # Check API keys
    print("\n\U0001f4e1 API Keys:")
    api_keys = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT/Codex)"),
        ("OPENROUTER_API_KEY", "OpenRouter (Fallback)"),
        ("GEMINI_API_KEY", "Google (Gemini)"),
        ("XAI_API_KEY", "xAI (Grok)"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]
    for env_var, name in api_keys:
        value = os.environ.get(env_var, "")
        if value:
            # Show masked key
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  \u2713 {name}: {masked}")
        else:
            print(f"  \u2717 {name}: not set")

    # Check CLI tools
    print("\n\U0001f527 CLI Tools:")
    cli_tools = [
        ("claude", "Claude Code CLI"),
        ("codex", "OpenAI Codex CLI"),
        ("gemini", "Gemini CLI"),
        ("grok", "Grok CLI"),
    ]
    for cmd, name in cli_tools:
        path = shutil.which(cmd)
        if path:
            print(f"  \u2713 {name}: {path}")
        else:
            print(f"  \u2717 {name}: not installed")

    # Check server health
    print("\n\U0001f310 Server Status:")
    server_url = args.server if hasattr(args, "server") else DEFAULT_API_URL
    try:
        import httpx

        resp = httpx.get(f"{server_url}/api/health", timeout=2)
        if resp.status_code == 200:
            print(f"  \u2713 Server running at {server_url}")
        else:
            print(f"  \u26a0 Server returned status {resp.status_code}")
    except (httpx.HTTPError, OSError, TimeoutError):
        print(f"  \u2717 Server not reachable at {server_url}")

    # Check database
    print("\n\U0001f4be Databases:")
    from aragora.persistence.db_config import DatabaseType, get_db_path

    db_paths = [
        (get_db_path(DatabaseType.CONTINUUM_MEMORY), "Memory store"),
        (get_db_path(DatabaseType.INSIGHTS), "Insights store"),
        (get_db_path(DatabaseType.ELO), "ELO rankings"),
    ]
    for db_path, name in db_paths:
        if Path(db_path).exists():
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            print(f"  \u2713 {name}: {size_mb:.1f} MB")
        else:
            print(f"  \u2717 {name}: not found")

    # Show nomic loop state if available
    nomic_state = Path(".nomic/nomic_state.json")
    if nomic_state.exists():
        print("\n\U0001f504 Nomic Loop:")
        try:
            import json

            with open(nomic_state) as f:
                state = json.load(f)
            total_cycles = state.get("total_cycles", 0)
            last_cycle = state.get("last_cycle_timestamp", "unknown")
            print(f"  Total cycles: {total_cycles}")
            print(f"  Last run: {last_cycle}")
        except OSError as e:
            logger.warning("Could not read nomic state file: %s", e)
            print(f"  \u26a0 Could not read state: {e}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Invalid nomic state file format: %s", e)
            print(f"  \u26a0 Could not read state: {e}")

    print("\n" + "=" * 60)
    print("Run 'aragora ask' to start a debate or 'aragora serve' to start the server")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Handle 'doctor' command - run system health checks."""
    from aragora.cli.doctor import main as doctor_main

    sys.exit(doctor_main())


def cmd_validate(_: argparse.Namespace) -> None:
    """Handle 'validate' command - validate API keys."""
    # run_validate doesn't exist; reuse doctor main for now
    from aragora.cli.doctor import main as doctor_main

    sys.exit(doctor_main())


def cmd_validate_env(args: argparse.Namespace) -> None:
    """Handle 'validate-env' command - validate environment and backend connectivity."""

    verbose = getattr(args, "verbose", False)
    json_output = getattr(args, "json", False)
    strict = getattr(args, "strict", False)

    async def run_validation() -> dict:
        """Run all environment validations."""
        import os
        from typing import Any

        results: dict[str, Any] = {
            "valid": True,
            "checks": {},
            "errors": [],
            "warnings": [],
        }

        # 1. Environment mode
        env = os.environ.get("ARAGORA_ENV", "development")
        is_production = env == "production"
        results["checks"]["environment"] = {
            "status": "ok",
            "value": env,
            "is_production": is_production,
        }

        # 2. Check distributed state requirement
        try:
            from aragora.control_plane.leader import is_distributed_state_required

            distributed_required = is_distributed_state_required()
        except ImportError:
            distributed_required = False

        results["checks"]["distributed_state"] = {
            "status": "ok" if not distributed_required else "required",
            "required": distributed_required,
            "reason": (
                "ARAGORA_MULTI_INSTANCE=true or ARAGORA_ENV=production"
                if distributed_required
                else "single instance mode"
            ),
        }

        # 3. Encryption key
        encryption_key = os.environ.get("ARAGORA_ENCRYPTION_KEY")
        if encryption_key:
            key_len = len(encryption_key) // 2  # hex string
            results["checks"]["encryption"] = {
                "status": "ok",
                "configured": True,
                "key_length_bytes": key_len,
            }
        elif is_production:
            results["checks"]["encryption"] = {
                "status": "error",
                "configured": False,
                "message": "ARAGORA_ENCRYPTION_KEY required in production",
            }
            results["errors"].append("Encryption key not configured")
            results["valid"] = False
        else:
            results["checks"]["encryption"] = {
                "status": "warning",
                "configured": False,
                "message": "Encryption key not set (optional in development)",
            }
            results["warnings"].append("Encryption key not configured")

        # 4. Redis connectivity
        try:
            from aragora.server.startup import validate_redis_connectivity

            redis_ok, redis_msg = await validate_redis_connectivity(timeout_seconds=5.0)
            if redis_ok:
                results["checks"]["redis"] = {
                    "status": "ok",
                    "connected": True,
                    "message": redis_msg,
                }
            elif distributed_required:
                results["checks"]["redis"] = {
                    "status": "error",
                    "connected": False,
                    "message": redis_msg,
                }
                results["errors"].append(f"Redis: {redis_msg}")
                results["valid"] = False
            else:
                results["checks"]["redis"] = {
                    "status": "warning",
                    "connected": False,
                    "message": redis_msg,
                }
                results["warnings"].append(f"Redis: {redis_msg}")
        except ImportError as e:
            results["checks"]["redis"] = {
                "status": "skip",
                "message": f"Startup module not available: {e}",
            }

        # 5. PostgreSQL connectivity
        try:
            from aragora.server.startup import validate_database_connectivity

            db_ok, db_msg = await validate_database_connectivity(timeout_seconds=5.0)
            require_database = os.environ.get("ARAGORA_REQUIRE_DATABASE", "").lower() in (
                "true",
                "1",
                "yes",
            )

            if db_ok:
                results["checks"]["postgresql"] = {
                    "status": "ok",
                    "connected": True,
                    "message": db_msg,
                }
            elif require_database:
                results["checks"]["postgresql"] = {
                    "status": "error",
                    "connected": False,
                    "message": db_msg,
                }
                results["errors"].append(f"PostgreSQL: {db_msg}")
                results["valid"] = False
            else:
                results["checks"]["postgresql"] = {
                    "status": "info",
                    "connected": False,
                    "message": db_msg,
                }
        except ImportError as e:
            results["checks"]["postgresql"] = {
                "status": "skip",
                "message": f"Startup module not available: {e}",
            }

        # 6. AI provider check
        providers_configured = []
        provider_keys = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "xai": "XAI_API_KEY",
        }

        for name, key in provider_keys.items():
            if os.environ.get(key):
                providers_configured.append(name)

        if providers_configured:
            results["checks"]["ai_providers"] = {
                "status": "ok",
                "configured": providers_configured,
            }
        else:
            results["checks"]["ai_providers"] = {
                "status": "error",
                "configured": [],
                "message": "No AI provider API key configured",
            }
            results["errors"].append("No AI provider configured")
            results["valid"] = False

        # 7. JWT secret check
        jwt_secret = os.environ.get("JWT_SECRET") or os.environ.get("ARAGORA_JWT_SECRET")
        if jwt_secret:
            results["checks"]["jwt_secret"] = {
                "status": "ok",
                "configured": True,
            }
        elif is_production:
            results["checks"]["jwt_secret"] = {
                "status": "warning",
                "configured": False,
                "message": "JWT secret not set - using derived key",
            }
            results["warnings"].append("JWT secret not configured")
        else:
            results["checks"]["jwt_secret"] = {
                "status": "info",
                "configured": False,
            }

        return results

    # Run the async validation
    results = asyncio.run(run_validation())

    # Determine final validity (strict mode treats warnings as errors)
    is_valid = results["valid"]
    if strict and results["warnings"]:
        is_valid = False

    # Output
    if json_output:
        import json

        results["strict_mode"] = strict
        results["final_valid"] = is_valid
        print(json.dumps(results, indent=2))
        sys.exit(0 if is_valid else 1)

    # Pretty output
    print("\n" + "=" * 60)
    print("ARAGORA ENVIRONMENT VALIDATION")
    print("=" * 60 + "\n")

    def status_icon(status: str) -> str:
        icons = {
            "ok": "\u2713",  # checkmark
            "error": "\u2717",  # X
            "warning": "!",
            "info": "-",
            "skip": "?",
            "required": "\u2713",
        }
        return icons.get(status, "?")

    for check_name, check_data in results["checks"].items():
        status = check_data.get("status", "unknown")
        icon = status_icon(status)

        # Format check name
        display_name = check_name.replace("_", " ").title()

        # Build detail string
        details = []
        if "value" in check_data:
            details.append(check_data["value"])
        if "configured" in check_data:
            if isinstance(check_data["configured"], list):
                details.append(", ".join(check_data["configured"]))
            elif check_data["configured"]:
                details.append("configured")
        if "connected" in check_data and check_data["connected"]:
            details.append("connected")
        if "message" in check_data and verbose:
            details.append(check_data["message"])

        detail_str = f" ({', '.join(details)})" if details else ""

        # Color based on status (using ANSI codes)
        if status == "ok":
            print(f"  {icon} {display_name}{detail_str}")
        elif status == "error":
            print(f"  {icon} {display_name}: FAILED{detail_str}")
        elif status == "warning":
            print(f"  {icon} {display_name}: WARNING{detail_str}")
        else:
            print(f"  {icon} {display_name}{detail_str}")

    print()

    if results["errors"]:
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
        print()

    if results["warnings"]:
        print("Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
        print()

    # Determine final status
    failed = not results["valid"]
    if strict and results["warnings"]:
        failed = True

    if failed:
        if strict and results["warnings"] and results["valid"]:
            print("Result: VALIDATION FAILED (strict mode - warnings treated as errors)")
        else:
            print("Result: VALIDATION FAILED - fix errors before production deployment")
    else:
        print("Result: All production requirements met")

    print()
    sys.exit(1 if failed else 0)
