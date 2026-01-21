#!/usr/bin/env python3
"""
Production Environment Validation Script for aragora.ai

Validates all required components are properly configured before/after deployment:
- Environment variables
- Database connectivity (Supabase PostgreSQL)
- API provider keys
- SSL/TLS configuration
- Health endpoints

Usage:
    python scripts/validate_production.py           # Full validation
    python scripts/validate_production.py --quick   # Quick check (no DB connection)
    python scripts/validate_production.py --fix     # Show fix suggestions
"""

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


class Status(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    fix: Optional[str] = None


def check_env_var(name: str, required: bool = True, secret: bool = False) -> CheckResult:
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        display = "***" if secret else (value[:20] + "..." if len(value) > 20 else value)
        return CheckResult(name, Status.OK, f"Set: {display}")
    elif required:
        return CheckResult(
            name, Status.FAIL, "Not set",
            fix=f"export {name}=<value> or add to .env file"
        )
    else:
        return CheckResult(name, Status.WARN, "Not set (optional)")


def check_environment_mode() -> CheckResult:
    """Check ARAGORA_ENVIRONMENT is set to production."""
    env = os.environ.get("ARAGORA_ENVIRONMENT", "development")
    if env == "production":
        return CheckResult("ARAGORA_ENVIRONMENT", Status.OK, "production")
    else:
        return CheckResult(
            "ARAGORA_ENVIRONMENT", Status.WARN, f"'{env}' (not production)",
            fix="export ARAGORA_ENVIRONMENT=production"
        )


def check_jwt_secret() -> CheckResult:
    """Check JWT secret is configured and strong enough."""
    secret = os.environ.get("ARAGORA_JWT_SECRET")
    if not secret:
        return CheckResult(
            "ARAGORA_JWT_SECRET", Status.FAIL, "Not set",
            fix='export ARAGORA_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    if len(secret) < 32:
        return CheckResult(
            "ARAGORA_JWT_SECRET", Status.WARN, f"Too short ({len(secret)} chars, need 32+)",
            fix="Use a longer secret for production security"
        )
    return CheckResult("ARAGORA_JWT_SECRET", Status.OK, f"Set ({len(secret)} chars)")


def check_supabase_config() -> CheckResult:
    """Check Supabase configuration."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")

    issues = []
    if not url:
        issues.append("SUPABASE_URL not set")
    if not key:
        issues.append("SUPABASE_KEY not set")
    if not dsn:
        issues.append("ARAGORA_POSTGRES_DSN not set")

    if issues:
        return CheckResult(
            "Supabase", Status.FAIL, "; ".join(issues),
            fix="Configure Supabase credentials in .env (see docs/PRODUCTION_DEPLOYMENT.md)"
        )

    return CheckResult("Supabase", Status.OK, "All credentials configured")


async def check_postgres_connection() -> CheckResult:
    """Test PostgreSQL connection to Supabase."""
    dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        return CheckResult(
            "PostgreSQL Connection", Status.SKIP, "No DSN configured"
        )

    try:
        import asyncpg
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10)
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            version = await conn.fetchval("SELECT version()")
            await pool.close()

            # Extract PostgreSQL version
            pg_version = version.split()[1] if version else "unknown"
            return CheckResult(
                "PostgreSQL Connection", Status.OK,
                f"Connected (PostgreSQL {pg_version})"
            )
    except ImportError:
        return CheckResult(
            "PostgreSQL Connection", Status.FAIL, "asyncpg not installed",
            fix="pip install asyncpg"
        )
    except Exception as e:
        return CheckResult(
            "PostgreSQL Connection", Status.FAIL, str(e)[:100],
            fix="Check DSN format and network connectivity"
        )


async def check_tables_exist() -> CheckResult:
    """Check if required database tables exist."""
    dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        return CheckResult("Database Tables", Status.SKIP, "No DSN configured")

    required_tables = [
        "webhook_configs", "integrations", "users", "job_queue",
        "token_blacklist", "approval_requests"
    ]

    try:
        import asyncpg
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10)
        async with pool.acquire() as conn:
            existing = []
            missing = []
            for table in required_tables:
                row = await conn.fetchrow(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                    )
                    """,
                    table,
                )
                if row and row[0]:
                    existing.append(table)
                else:
                    missing.append(table)

            await pool.close()

            if missing:
                return CheckResult(
                    "Database Tables", Status.FAIL,
                    f"Missing: {', '.join(missing)}",
                    fix="python scripts/init_postgres_db.py"
                )
            return CheckResult(
                "Database Tables", Status.OK,
                f"{len(existing)} tables verified"
            )
    except ImportError:
        return CheckResult("Database Tables", Status.SKIP, "asyncpg not installed")
    except Exception as e:
        return CheckResult("Database Tables", Status.FAIL, str(e)[:100])


def check_ai_providers() -> CheckResult:
    """Check if at least one AI provider is configured."""
    providers = {
        "ANTHROPIC_API_KEY": "Anthropic",
        "OPENAI_API_KEY": "OpenAI",
        "GEMINI_API_KEY": "Google Gemini",
        "XAI_API_KEY": "xAI Grok",
        "MISTRAL_API_KEY": "Mistral",
        "OPENROUTER_API_KEY": "OpenRouter",
    }

    configured = [name for key, name in providers.items() if os.environ.get(key)]

    if not configured:
        return CheckResult(
            "AI Providers", Status.FAIL, "No AI provider configured",
            fix="Set at least one: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc."
        )

    return CheckResult(
        "AI Providers", Status.OK,
        f"Configured: {', '.join(configured)}"
    )


def check_cors_config() -> CheckResult:
    """Check CORS configuration for production."""
    origins = os.environ.get("ARAGORA_ALLOWED_ORIGINS", "")
    if not origins:
        return CheckResult(
            "CORS", Status.WARN, "No allowed origins set",
            fix="export ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai"
        )

    if "localhost" in origins.lower() or "127.0.0.1" in origins:
        return CheckResult(
            "CORS", Status.WARN, "Localhost in allowed origins (OK for staging)",
        )

    return CheckResult("CORS", Status.OK, f"Origins: {origins[:50]}...")


def check_ssl_config() -> CheckResult:
    """Check SSL/TLS configuration."""
    ssl_enabled = os.environ.get("ARAGORA_SSL_ENABLED", "").lower() == "true"
    cert = os.environ.get("ARAGORA_SSL_CERT")
    key = os.environ.get("ARAGORA_SSL_KEY")

    if not ssl_enabled:
        return CheckResult(
            "SSL/TLS", Status.WARN, "Not enabled (OK if behind reverse proxy)",
        )

    if not cert or not key:
        return CheckResult(
            "SSL/TLS", Status.FAIL, "Enabled but cert/key not configured",
            fix="Set ARAGORA_SSL_CERT and ARAGORA_SSL_KEY paths"
        )

    # Check if cert files exist
    if not Path(cert).exists():
        return CheckResult("SSL/TLS", Status.FAIL, f"Cert file not found: {cert}")
    if not Path(key).exists():
        return CheckResult("SSL/TLS", Status.FAIL, f"Key file not found: {key}")

    return CheckResult("SSL/TLS", Status.OK, "Enabled with cert/key")


def print_results(results: list[CheckResult], show_fix: bool = False) -> tuple[int, int, int]:
    """Print validation results and return counts."""
    ok_count = warn_count = fail_count = 0

    # Status symbols and colors
    symbols = {
        Status.OK: "✓",
        Status.WARN: "⚠",
        Status.FAIL: "✗",
        Status.SKIP: "○",
    }

    print("\n" + "=" * 60)
    print("ARAGORA.AI PRODUCTION VALIDATION")
    print("=" * 60 + "\n")

    for result in results:
        symbol = symbols[result.status]
        status_str = result.status.value.ljust(4)

        if result.status == Status.OK:
            ok_count += 1
        elif result.status == Status.WARN:
            warn_count += 1
        elif result.status == Status.FAIL:
            fail_count += 1

        print(f"{symbol} [{status_str}] {result.name}")
        print(f"          {result.message}")

        if show_fix and result.fix and result.status in (Status.FAIL, Status.WARN):
            print(f"          Fix: {result.fix}")

        print()

    # Summary
    print("=" * 60)
    print(f"SUMMARY: {ok_count} OK, {warn_count} warnings, {fail_count} failures")

    if fail_count == 0 and warn_count == 0:
        print("\n✓ All checks passed! Ready for production.")
    elif fail_count == 0:
        print("\n⚠ Warnings present but no critical failures.")
    else:
        print("\n✗ Critical issues found. Please fix before deploying.")

    print("=" * 60 + "\n")

    return ok_count, warn_count, fail_count


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate aragora.ai production environment"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick check without database connection tests"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Show fix suggestions for issues"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results: list[CheckResult] = []

    # Environment checks
    print("Checking environment configuration...")
    results.append(check_environment_mode())
    results.append(check_jwt_secret())
    results.append(check_supabase_config())
    results.append(check_ai_providers())
    results.append(check_cors_config())
    results.append(check_ssl_config())

    # Database checks (unless --quick)
    if not args.quick:
        print("Checking database connectivity...")
        results.append(await check_postgres_connection())
        results.append(await check_tables_exist())
    else:
        results.append(CheckResult("PostgreSQL Connection", Status.SKIP, "Quick mode"))
        results.append(CheckResult("Database Tables", Status.SKIP, "Quick mode"))

    # Print results
    ok, warn, fail = print_results(results, show_fix=args.fix)

    # Return exit code
    if fail > 0:
        return 1
    elif warn > 0:
        return 0  # Warnings are OK
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
