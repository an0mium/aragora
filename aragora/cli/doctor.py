#!/usr/bin/env python3
"""
Aragora Doctor - System Health Check CLI

Usage:
    python -m aragora doctor
    aragora doctor
"""

import os
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required


class CheckResult(NamedTuple):
    """Result of a single health check."""
    name: str
    status: str  # "pass", "warn", "fail"
    message: str


def check_env_vars() -> list[CheckResult]:
    """Check required and optional environment variables."""
    results = []

    # Required: at least one API key
    api_keys = [
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("GEMINI_API_KEY", "Gemini"),
        ("GOOGLE_API_KEY", "Google"),
        ("XAI_API_KEY", "xAI"),
        ("GROK_API_KEY", "Grok"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("KIMI_API_KEY", "Kimi"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]

    found_keys = []
    for env_var, provider in api_keys:
        value = os.environ.get(env_var, "")
        if value and len(value) >= 10:
            found_keys.append(provider)

    if found_keys:
        results.append(CheckResult(
            "api_keys",
            "pass",
            f"Found: {', '.join(found_keys)}"
        ))
    else:
        results.append(CheckResult(
            "api_keys",
            "fail",
            "No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
        ))

    # Check for fallback key (helpful warning)
    if found_keys and "OpenRouter" not in found_keys:
        results.append(CheckResult(
            "fallback",
            "warn",
            "OPENROUTER_API_KEY not set (recommended for fallback)"
        ))

    return results


def check_configuration() -> list[CheckResult]:
    """Validate configuration using existing infrastructure."""
    results = []

    try:
        from aragora.config.legacy import validate_configuration

        validation = validate_configuration(strict=False)

        if validation["valid"]:
            results.append(CheckResult(
                "configuration",
                "pass",
                "Configuration valid"
            ))
        else:
            for error in validation["errors"]:
                results.append(CheckResult("config_error", "fail", error))

        for warning in validation.get("warnings", []):
            results.append(CheckResult("config_warning", "warn", warning))

    except ImportError as e:
        results.append(CheckResult(
            "configuration",
            "warn",
            f"Could not load config module: {e}"
        ))
    except Exception as e:
        results.append(CheckResult(
            "configuration",
            "fail",
            f"Configuration error: {e}"
        ))

    return results


def check_databases() -> list[CheckResult]:
    """Check SQLite database accessibility."""
    results = []

    nomic_dir = Path(".nomic")
    if not nomic_dir.exists():
        results.append(CheckResult(
            "databases",
            "warn",
            ".nomic directory not found (first run?)"
        ))
        return results

    # Key databases to check
    db_files = [
        ("agent_elo.db", "ELO rankings"),
        ("agent_memories.db", "Memory store"),
        ("agent_calibration.db", "Calibration"),
        ("consensus_memory.db", "Consensus"),
    ]

    accessible = 0
    for db_file, description in db_files:
        db_path = nomic_dir / db_file
        if not db_path.exists():
            continue  # Skip missing files silently

        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            conn.execute("SELECT 1")
            conn.close()
            accessible += 1
            size_mb = db_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                results.append(CheckResult(
                    f"db_{db_file}",
                    "warn",
                    f"{description}: {size_mb:.1f} MB (large)"
                ))
        except sqlite3.Error as e:
            results.append(CheckResult(
                f"db_{db_file}",
                "fail",
                f"{description}: {e}"
            ))

    if accessible > 0:
        results.insert(0, CheckResult(
            "databases",
            "pass",
            f"{accessible} database(s) accessible"
        ))

    return results


def check_circuit_breakers() -> list[CheckResult]:
    """Check circuit breaker status."""
    results = []

    try:
        from aragora.resilience import get_circuit_breaker_status

        status = get_circuit_breaker_status()

        if not status:
            results.append(CheckResult(
                "circuit_breakers",
                "pass",
                "No circuit breakers registered (agents not initialized)"
            ))
            return results

        open_circuits = []
        for name, info in status.items():
            if info.get("status") == "open":
                open_circuits.append(name)

        if not open_circuits:
            results.append(CheckResult(
                "circuit_breakers",
                "pass",
                f"All {len(status)} circuit breakers closed"
            ))
        else:
            results.append(CheckResult(
                "circuit_breakers",
                "warn",
                f"{len(open_circuits)}/{len(status)} open: {', '.join(open_circuits)}"
            ))

    except ImportError:
        results.append(CheckResult(
            "circuit_breakers",
            "warn",
            "Resilience module not available"
        ))

    return results


def check_server() -> list[CheckResult]:
    """Check if API server is running."""
    results = []

    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8080/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                results.append(CheckResult(
                    "server",
                    "pass",
                    "API server running at localhost:8080"
                ))
    except Exception:
        results.append(CheckResult(
            "server",
            "warn",
            "API server not running (optional)"
        ))

    return results


def print_results(results: list[CheckResult]) -> int:
    """Print results and return exit code."""
    symbols = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
    colors = {"pass": "\033[32m", "warn": "\033[33m", "fail": "\033[31m"}
    reset = "\033[0m"

    use_color = sys.stdout.isatty()

    print("\nAragora Doctor - System Health Check")
    print("=" * 50)

    has_failures = False

    for result in results:
        symbol = symbols[result.status]
        if use_color:
            symbol = f"{colors[result.status]}{symbol}{reset}"

        print(f"{symbol} {result.name}: {result.message}")

        if result.status == "fail":
            has_failures = True

    print("=" * 50)

    if has_failures:
        print("\nSome checks failed. See above for details.")
        return 1
    else:
        print("\nAll critical checks passed!")
        return 0


def main() -> int:
    """Run all health checks."""
    results = []

    results.extend(check_env_vars())
    results.extend(check_configuration())
    results.extend(check_databases())
    results.extend(check_circuit_breakers())
    results.extend(check_server())

    return print_results(results)


if __name__ == "__main__":
    sys.exit(main())
