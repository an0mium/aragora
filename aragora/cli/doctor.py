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

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional


def check_icon(ok: Optional[bool]) -> str:
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


def check_packages() -> list[tuple[str, str, Optional[bool]]]:
    """Check required and optional packages."""
    checks = []

    # Required packages
    required = ["aiohttp", "pydantic", "sqlite3", "asyncio"]
    for pkg in required:
        try:
            __import__(pkg)
            checks.append((pkg, "installed", True))
        except ImportError:
            checks.append((pkg, "MISSING", False))

    # Optional ML packages
    optional_ml = ["torch", "transformers", "sentence_transformers"]
    for pkg in optional_ml:
        try:
            __import__(pkg)
            checks.append((f"{pkg} (ML)", "installed", True))
        except ImportError:
            checks.append((f"{pkg} (ML)", "not installed", None))

    # Optional integrations
    optional_int = ["redis", "asyncpg", "boto3", "opentelemetry"]
    for pkg in optional_int:
        try:
            __import__(pkg)
            checks.append((f"{pkg} (integration)", "installed", True))
        except ImportError:
            checks.append((f"{pkg} (integration)", "not installed", None))

    return checks


def check_api_keys() -> list[tuple[str, str, Optional[bool]]]:
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


def check_storage() -> list[tuple[str, str, Optional[bool]]]:
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
    except Exception as e:
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


async def check_server() -> list[tuple[str, str, Optional[bool]]]:
    """Check if server is running and responsive."""
    checks = []

    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "http://localhost:8080/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        checks.append(("Server (localhost:8080)", "running", True))
                    else:
                        checks.append(
                            ("Server (localhost:8080)", f"unhealthy ({resp.status})", False)
                        )
            except aiohttp.ClientError:
                checks.append(("Server (localhost:8080)", "not running", None))
    except ImportError:
        checks.append(("Server check", "aiohttp not available", None))

    return checks


def check_environment() -> list[tuple[str, str, Optional[bool]]]:
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
    except Exception as e:
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
