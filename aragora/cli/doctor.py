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
import importlib.util
import os
import sys
from pathlib import Path

import httpx

from aragora.exceptions import REDIS_CONNECTION_ERRORS

_SERVER_UNAVAILABLE_ERRORS = REDIS_CONNECTION_ERRORS + (
    httpx.RequestError,
    RuntimeError,
)


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


def _module_available(module_name: str) -> bool:
    """Check whether a module can be resolved without importing it."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (AttributeError, ImportError, ValueError):
        return False


def check_packages() -> list[tuple[str, str, bool | None]]:
    """Check required and optional packages."""
    checks = []

    # Required packages
    required = ["aiohttp", "pydantic", "sqlite3", "asyncio"]
    for pkg in required:
        if _module_available(pkg):
            checks.append((pkg, "installed", True))
        else:
            checks.append((pkg, "MISSING", False))

    # Optional ML packages
    optional_ml = ["torch", "transformers", "sentence_transformers"]
    for pkg in optional_ml:
        if _module_available(pkg):
            checks.append((f"{pkg} (ML)", "installed", True))
        else:
            checks.append((f"{pkg} (ML)", "not installed", None))

    # Optional integrations
    optional_int = ["redis", "asyncpg", "boto3", "opentelemetry"]
    for pkg in optional_int:
        if _module_available(pkg):
            checks.append((f"{pkg} (integration)", "installed", True))
        else:
            checks.append((f"{pkg} (integration)", "not installed", None))

    return checks


def check_api_keys() -> list[tuple[str, str, bool | None]]:
    """Check API key configuration."""
    checks = []

    # At least one LLM provider required
    llm_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    has_llm = False
    for key in llm_keys:
        if os.getenv(key):
            checks.append((key, "configured", True))
            has_llm = True
        else:
            checks.append((key, "not set", None))

    if not has_llm:
        checks.append(("LLM Provider", "NO API KEY SET", False))

    # Optional providers
    optional_keys = ["GEMINI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"]
    for key in optional_keys:
        if os.getenv(key):
            checks.append((key, "configured", True))
        else:
            checks.append((key, "not set", None))

    return checks


def check_storage() -> list[tuple[str, str, bool | None]]:
    """Check storage backends."""
    checks = []

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
    if _module_available("asyncpg"):
        checks.append(("PostgreSQL driver", "available", True))
        if os.getenv("DATABASE_URL"):
            checks.append(("DATABASE_URL", "configured", True))
        else:
            checks.append(("DATABASE_URL", "not set (using SQLite)", None))
    else:
        checks.append(("PostgreSQL driver", "not installed", None))

    # Check Redis
    if _module_available("redis"):
        checks.append(("Redis driver", "available", True))
        if os.getenv("ARAGORA_REDIS_URL"):
            checks.append(("ARAGORA_REDIS_URL", "configured", True))
        else:
            checks.append(("ARAGORA_REDIS_URL", "not set (using memory)", None))
    else:
        checks.append(("Redis driver", "not installed", None))

    return checks


async def check_server() -> list[tuple[str, str, bool | None]]:
    """Check if server is running and responsive."""
    checks = []

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
        except _SERVER_UNAVAILABLE_ERRORS as exc:
            detail = str(exc).strip()
            status = f"not running ({detail})" if detail else "not running"
            checks.append(("Server (localhost:8080)", status, None))
    except ImportError:
        checks.append(("Server check", "http pool not available", None))

    return checks


def check_environment() -> list[tuple[str, str, bool | None]]:
    """Check environment configuration."""
    checks = []

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


def main() -> int:
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
    key_checks = check_api_keys()
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
    except (httpx.RequestError, OSError, ConnectionError, RuntimeError) as e:
        print(f"  {check_icon(None)} Server check: skipped ({e})")

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
