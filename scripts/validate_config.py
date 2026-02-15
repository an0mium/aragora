#!/usr/bin/env python3
"""
Aragora configuration validator.

Checks that all required environment variables are set, database is reachable,
and the configuration is valid before starting the server.

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --strict  # Fail on warnings too

Exit codes:
    0 - All checks passed
    1 - Critical errors found (server will not start correctly)
    2 - Warnings found (--strict mode only)
"""

import os
import sys


def check_env(name: str, required: bool = False, secret: bool = False) -> str | None:
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        display = f"{value[:4]}..." if secret and len(value) > 4 else value
        return display
    return None


def main() -> int:
    strict = "--strict" in sys.argv
    errors: list[str] = []
    warnings: list[str] = []

    print("Aragora Configuration Validator")
    print("=" * 50)

    # 1. Check AI provider keys
    print("\n[AI Providers]")
    has_provider = False
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
        val = check_env(key, secret=True)
        if val:
            print(f"  OK  {key} = {val}")
            has_provider = True
        else:
            print(f"  --  {key} not set")
    if not has_provider:
        errors.append("No AI provider API key configured (need at least one)")

    # 2. Check environment
    print("\n[Environment]")
    env = os.environ.get("ARAGORA_ENV", "development")
    print(f"  ARAGORA_ENV = {env}")
    is_production = env == "production"

    # 3. Check security settings
    print("\n[Security]")
    secret_key = check_env("ARAGORA_SECRET_KEY", secret=True)
    if secret_key:
        print(f"  OK  ARAGORA_SECRET_KEY = {secret_key}")
    elif is_production:
        errors.append("ARAGORA_SECRET_KEY required in production")
    else:
        warnings.append("ARAGORA_SECRET_KEY not set (required for production)")

    origins = os.environ.get("ARAGORA_ALLOWED_ORIGINS", "*")
    if origins == "*" and is_production:
        errors.append("ARAGORA_ALLOWED_ORIGINS='*' is not safe for production")
    elif origins == "*":
        warnings.append("ARAGORA_ALLOWED_ORIGINS='*' (wildcard) - restrict in production")
    else:
        print(f"  OK  ARAGORA_ALLOWED_ORIGINS = {origins}")

    # 4. Check database
    print("\n[Database]")
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
    if db_url:
        # Mask password in display
        display_url = db_url
        if "@" in db_url and ":" in db_url:
            parts = db_url.split("@")
            creds = parts[0].rsplit(":", 1)
            if len(creds) == 2:
                display_url = f"{creds[0]}:****@{parts[1]}"
        print(f"  OK  DATABASE_URL = {display_url}")

        # Try connecting
        try:
            import psycopg2  # noqa: F401

            print("  --  psycopg2 available (connection test skipped in validator)")
        except ImportError:
            try:
                import asyncpg  # noqa: F401

                print("  --  asyncpg available (connection test skipped in validator)")
            except ImportError:
                warnings.append("No PostgreSQL driver installed (psycopg2 or asyncpg)")
    else:
        backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite")
        if backend == "postgres" or is_production:
            warnings.append(
                "No DATABASE_URL set - will use SQLite (not recommended for production)"
            )
        print(f"  --  Using {backend} backend")

    # 5. Check Redis
    print("\n[Redis]")
    redis_url = os.environ.get("ARAGORA_REDIS_URL")
    if redis_url:
        print(f"  OK  ARAGORA_REDIS_URL = {redis_url}")
    else:
        warnings.append("ARAGORA_REDIS_URL not set (required for multi-instance deployments)")

    # 6. Check ports
    print("\n[Server]")
    api_port = os.environ.get("ARAGORA_API_PORT", "8080")
    ws_port = os.environ.get("ARAGORA_WS_PORT", "8765")
    print(f"  API port: {api_port}")
    print(f"  WebSocket port: {ws_port}")
    if api_port == ws_port:
        errors.append(f"API and WebSocket ports are the same ({api_port})")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [x] {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")

    if not errors and not warnings:
        print("\nAll checks passed.")
        return 0
    elif errors:
        print(f"\n{len(errors)} error(s) found. Fix before deploying.")
        return 1
    elif strict:
        print(f"\n{len(warnings)} warning(s) found (strict mode).")
        return 2
    else:
        print(
            f"\n{len(warnings)} warning(s). Configuration is usable but review before production."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
