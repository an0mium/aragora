"""
Doctor command - Comprehensive health checks for Aragora.

Checks:
- Python version and required packages
- API key configuration
- Database connectivity
- Redis availability
- Storage backends
- Server endpoints (if running)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from aragora.config.provider_readiness import discover_provider_credentials

HealthCheck = tuple[str, str, bool | None]


def check_icon(ok: bool | None) -> str:
    """Return status icon."""
    if ok is True:
        return "\033[92m✓\033[0m"  # Green checkmark
    elif ok is False:
        return "\033[91m✗\033[0m"  # Red X
    return "\033[93m○\033[0m"  # Yellow circle (optional)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n\033[1m{title}\033[0m")
    print("-" * 40)


def check_packages() -> list[HealthCheck]:
    """Check required and optional packages."""
    checks: list[HealthCheck] = []

    # Required packages
    required = ["aiohttp", "pydantic", "sqlite3", "asyncio"]
    for pkg in required:
        try:
            __import__(pkg)
            checks.append((pkg, "installed", True))
        except Exception as exc:  # noqa: BLE001 - doctor should surface broken imports, not crash
            checks.append((pkg, f"MISSING ({type(exc).__name__})", False))

    # Optional ML packages
    optional_ml = ["torch", "transformers", "sentence_transformers"]
    for pkg in optional_ml:
        try:
            __import__(pkg)
            checks.append((f"{pkg} (ML)", "installed", True))
        except Exception as exc:  # noqa: BLE001 - optional imports may fail due broken transitive deps
            checks.append((f"{pkg} (ML)", f"not installed ({type(exc).__name__})", None))

    # Optional integrations
    optional_int = ["redis", "asyncpg", "boto3", "opentelemetry"]
    for pkg in optional_int:
        try:
            __import__(pkg)
            checks.append((f"{pkg} (integration)", "installed", True))
        except Exception as exc:  # noqa: BLE001 - doctor should not crash on broken optional deps
            checks.append((f"{pkg} (integration)", f"not installed ({type(exc).__name__})", None))

    return checks


def check_api_keys(validate_live: bool = False) -> list[HealthCheck]:
    """Check API key configuration."""
    checks: list[HealthCheck] = []
    report = discover_provider_credentials()
    invalid_providers = []

    for provider in report.providers:
        env_label = "/".join(provider.checked_env_vars)
        if provider.configured:
            status = "configured"
            ok = True
            if validate_live:
                from aragora.cli.api_keys import get_supported_provider_names, validate_provider_key

                validation_provider = "grok" if provider.provider == "xai" else provider.provider
                if validation_provider in set(get_supported_provider_names()):
                    validation_report = validate_provider_key(validation_provider)
                    status = f"{status}; live {validation_report.remote_status}"
                    if not validation_report.is_valid:
                        status = f"{status}: {validation_report.message}"
                        invalid_providers.append(provider.provider)
                        ok = False
                else:
                    status = f"{status}; live skipped"
            checks.append((env_label, status, ok))
        else:
            checks.append((env_label, "not set", None))

    if invalid_providers:
        checks.append(
            ("LLM Provider", f"invalid provider(s): {', '.join(invalid_providers)}", False)
        )
    elif report.any_configured:
        configured = ", ".join(report.configured_providers)
        checks.append(("LLM Provider", f"configured: {configured}", True))
    else:
        detail = "NO API KEY SET"
        if report.discovery_errors:
            detail += f" ({'; '.join(report.discovery_errors)})"
        checks.append(("LLM Provider", detail, False))

    return checks


def check_storage() -> list[HealthCheck]:
    """Check storage backends."""
    checks: list[HealthCheck] = []

    # Check data directory
    data_dir = Path.home() / ".aragora"
    if data_dir.exists():
        checks.append(("Data directory", str(data_dir), True))
    else:
        checks.append(("Data directory", "will be created", None))

    # Check SQLite
    try:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT 1")
        conn.close()
        checks.append(("SQLite", "working", True))
    except (OSError, RuntimeError) as e:
        checks.append(("SQLite", f"error: {e}", False))

    # Check PostgreSQL
    try:
        import asyncpg  # noqa: F401

        checks.append(("PostgreSQL driver", "available", True))
        if os.getenv("DATABASE_URL"):
            checks.append(("DATABASE_URL", "configured", True))
        else:
            checks.append(("DATABASE_URL", "not set (using SQLite)", None))
    except ImportError:
        checks.append(("PostgreSQL driver", "not installed", None))

    # Check Redis
    try:
        import redis  # noqa: F401

        checks.append(("Redis driver", "available", True))
        if os.getenv("ARAGORA_REDIS_URL"):
            checks.append(("ARAGORA_REDIS_URL", "configured", True))
        else:
            checks.append(("ARAGORA_REDIS_URL", "not set (using memory)", None))
    except ImportError:
        checks.append(("Redis driver", "not installed", None))

    return checks


async def check_server() -> list[HealthCheck]:
    """Check if server is running and responsive."""
    checks: list[HealthCheck] = []

    try:
        from aragora.server.http_client_pool import get_http_pool

        pool = get_http_pool()
        try:
            async with pool.get_session("health_check") as client:
                resp = await client.get("http://localhost:8080/health", timeout=5)
                if resp.status_code == 200:
                    checks.append(("Server (localhost:8080)", "running", True))
                else:
                    checks.append(
                        ("Server (localhost:8080)", f"unhealthy ({resp.status_code})", False)
                    )
        except Exception:  # noqa: BLE001 — diagnostic tool must never crash
            checks.append(("Server (localhost:8080)", "not running", None))
    except ImportError:
        checks.append(("Server check", "http pool not available", None))

    return checks


def check_environment() -> list[HealthCheck]:
    """Check environment configuration."""
    checks: list[HealthCheck] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", py_ver, sys.version_info >= (3, 10)))

    # Environment
    env = os.getenv("ARAGORA_ENV", "development")
    checks.append(("Environment", env, True))

    # Debug mode
    debug = os.getenv("ARAGORA_DEBUG", "false").lower() == "true"
    checks.append(("Debug mode", "enabled" if debug else "disabled", True))

    return checks


def main(validate_keys: bool = False) -> int:
    """Run comprehensive health checks."""
    print("\n\033[1;36m" + "=" * 50 + "\033[0m")
    print("\033[1;36m       ARAGORA HEALTH CHECK\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")

    all_ok = True
    all_checks = []

    # Environment
    print_section("Environment")
    env_checks = check_environment()
    all_checks.extend(env_checks)
    for name, status, ok in env_checks:
        print(f"  {check_icon(ok)} {name}: {status}")
        if ok is False:
            all_ok = False

    # Packages
    print_section("Packages")
    pkg_checks = check_packages()
    all_checks.extend(pkg_checks)
    for name, status, ok in pkg_checks:
        print(f"  {check_icon(ok)} {name}: {status}")
        if ok is False:
            all_ok = False

    # API Keys
    print_section("API Keys")
    key_checks = check_api_keys(validate_live=validate_keys)
    all_checks.extend(key_checks)
    for name, status, ok in key_checks:
        print(f"  {check_icon(ok)} {name}: {status}")
        if ok is False:
            all_ok = False

    # Storage
    print_section("Storage")
    storage_checks = check_storage()
    all_checks.extend(storage_checks)
    for name, status, ok in storage_checks:
        print(f"  {check_icon(ok)} {name}: {status}")
        if ok is False:
            all_ok = False

    # Server
    print_section("Server")
    try:
        server_checks = asyncio.run(check_server())
        all_checks.extend(server_checks)
        for name, status, ok in server_checks:
            print(f"  {check_icon(ok)} {name}: {status}")
            if ok is False:
                all_ok = False
    except Exception as e:  # noqa: BLE001 — doctor must never crash
        print(f"  {check_icon(None)} Server check: skipped ({type(e).__name__}: {e})")

    # Summary
    passed = sum(1 for _, _, ok in all_checks if ok is True)
    failed = sum(1 for _, _, ok in all_checks if ok is False)
    optional = sum(1 for _, _, ok in all_checks if ok is None)

    print("\n" + "=" * 50)
    print(f"\033[1mSummary:\033[0m {passed} passed, {failed} failed, {optional} optional")

    if all_ok:
        print("\n\033[92m✓ Aragora is ready to use!\033[0m\n")
    else:
        print("\n\033[91m✗ Some required checks failed. Please fix the issues above.\033[0m\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
