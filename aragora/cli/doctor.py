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


# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


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
        results.append(CheckResult("api_keys", "pass", f"Found: {', '.join(found_keys)}"))
    else:
        results.append(
            CheckResult(
                "api_keys", "fail", "No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )
        )

    # Check for fallback key (helpful warning)
    if found_keys and "OpenRouter" not in found_keys:
        results.append(
            CheckResult("fallback", "warn", "OPENROUTER_API_KEY not set (recommended for fallback)")
        )

    return results


def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Validate an API key by making a minimal test call.

    Returns (is_valid, message).
    """
    import json
    import urllib.error
    import urllib.request

    try:
        if provider == "Anthropic":
            # Anthropic messages API - minimal request
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                method="POST",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ).encode(),
            )

        elif provider == "OpenAI":
            # OpenAI models list - read-only, cheap
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                method="GET",
                headers={"Authorization": f"Bearer {api_key}"},
            )

        elif provider in ("Gemini", "Google"):
            # Gemini models list
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}",
                method="GET",
            )

        elif provider in ("xAI", "Grok"):
            # xAI models list
            req = urllib.request.Request(
                "https://api.x.ai/v1/models",
                method="GET",
                headers={"Authorization": f"Bearer {api_key}"},
            )

        elif provider == "OpenRouter":
            # OpenRouter models list
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                method="GET",
                headers={"Authorization": f"Bearer {api_key}"},
            )

        elif provider == "DeepSeek":
            # DeepSeek models list
            req = urllib.request.Request(
                "https://api.deepseek.com/models",
                method="GET",
                headers={"Authorization": f"Bearer {api_key}"},
            )

        else:
            return True, "Validation not implemented"

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                return True, "Valid"

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid key"
        elif e.code == 403:
            return False, "Access denied"
        elif e.code == 429:
            return True, "Valid (rate limited)"
        else:
            return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"

    return False, "Unknown error"


def check_api_keys_validity(validate: bool = False) -> list[CheckResult]:
    """
    Check API keys and optionally validate them.

    Args:
        validate: If True, make test API calls to verify keys work.
    """
    results = []

    api_keys = [
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("GEMINI_API_KEY", "Gemini"),
        ("XAI_API_KEY", "xAI"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]

    valid_keys = []
    invalid_keys = []

    for env_var, provider in api_keys:
        value = os.environ.get(env_var, "")
        if not value or len(value) < 10:
            continue

        if validate:
            is_valid, message = validate_api_key(provider, value)
            if is_valid:
                valid_keys.append(provider)
                results.append(
                    CheckResult(f"api_{provider.lower()}", "pass", f"{provider}: {message}")
                )
            else:
                invalid_keys.append(provider)
                results.append(
                    CheckResult(f"api_{provider.lower()}", "fail", f"{provider}: {message}")
                )
        else:
            valid_keys.append(provider)

    if not validate:
        if valid_keys:
            results.append(CheckResult("api_keys", "pass", f"Found: {', '.join(valid_keys)}"))
        else:
            results.append(
                CheckResult(
                    "api_keys", "fail", "No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
                )
            )

    if validate and invalid_keys:
        results.append(
            CheckResult(
                "api_keys_summary",
                "fail",
                f"Invalid keys: {', '.join(invalid_keys)}. Run `aragora config set <KEY> <value>` to fix.",
            )
        )

    return results


def check_configuration() -> list[CheckResult]:
    """Validate configuration using existing infrastructure."""
    results = []

    try:
        from aragora.config.legacy import validate_configuration

        validation = validate_configuration(strict=False)

        if validation["valid"]:
            results.append(CheckResult("configuration", "pass", "Configuration valid"))
        else:
            for error in validation["errors"]:
                results.append(CheckResult("config_error", "fail", error))

        for warning in validation.get("warnings", []):
            results.append(CheckResult("config_warning", "warn", warning))

    except ImportError as e:
        results.append(CheckResult("configuration", "warn", f"Could not load config module: {e}"))
    except Exception as e:
        results.append(CheckResult("configuration", "fail", f"Configuration error: {e}"))

    return results


def check_databases() -> list[CheckResult]:
    """Check SQLite database accessibility."""
    results = []

    nomic_dir = Path(".nomic")
    if not nomic_dir.exists():
        results.append(CheckResult("databases", "warn", ".nomic directory not found (first run?)"))
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
                results.append(
                    CheckResult(f"db_{db_file}", "warn", f"{description}: {size_mb:.1f} MB (large)")
                )
        except sqlite3.Error as e:
            results.append(CheckResult(f"db_{db_file}", "fail", f"{description}: {e}"))

    if accessible > 0:
        results.insert(0, CheckResult("databases", "pass", f"{accessible} database(s) accessible"))

    return results


def check_circuit_breakers() -> list[CheckResult]:
    """Check circuit breaker status."""
    results = []

    try:
        from aragora.resilience import get_circuit_breaker_status

        status = get_circuit_breaker_status()

        if not status:
            results.append(
                CheckResult(
                    "circuit_breakers",
                    "pass",
                    "No circuit breakers registered (agents not initialized)",
                )
            )
            return results

        open_circuits = []
        for name, info in status.items():
            # Skip metadata keys like _registry_size
            if name.startswith("_") or not isinstance(info, dict):
                continue
            if info.get("status") == "open":
                open_circuits.append(name)

        # Count actual circuit breakers (exclude metadata keys)
        cb_count = len([k for k in status.keys() if not k.startswith("_")])

        if not open_circuits:
            results.append(
                CheckResult("circuit_breakers", "pass", f"All {cb_count} circuit breakers closed")
            )
        else:
            results.append(
                CheckResult(
                    "circuit_breakers",
                    "warn",
                    f"{len(open_circuits)}/{cb_count} open: {', '.join(open_circuits)}",
                )
            )

    except ImportError:
        results.append(CheckResult("circuit_breakers", "warn", "Resilience module not available"))

    return results


def check_server() -> list[CheckResult]:
    """Check if API server is running."""
    results = []

    try:
        import urllib.request

        # Use health endpoint
        health_url = f"{DEFAULT_API_URL}/healthz"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                results.append(
                    CheckResult("server", "pass", f"API server running at {DEFAULT_API_URL}")
                )
    except (OSError, TimeoutError):
        results.append(
            CheckResult("server", "warn", f"API server not running at {DEFAULT_API_URL} (optional)")
        )

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


def main(validate_keys: bool = False) -> int:
    """
    Run all health checks.

    Args:
        validate_keys: If True, validate API keys by making test calls.
    """
    results = []

    if validate_keys:
        print("Validating API keys (making test calls)...")
        results.extend(check_api_keys_validity(validate=True))
    else:
        results.extend(check_env_vars())

    results.extend(check_configuration())
    results.extend(check_databases())
    results.extend(check_circuit_breakers())
    results.extend(check_server())

    return print_results(results)


def run_validate() -> int:
    """
    Run API key validation only.

    Returns exit code 0 if all keys valid, 1 if any invalid.
    """
    print("\nAragora Validate - API Key Verification")
    print("=" * 50)

    results = check_api_keys_validity(validate=True)

    symbols = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
    colors = {"pass": "\033[32m", "warn": "\033[33m", "fail": "\033[31m"}
    reset = "\033[0m"

    use_color = sys.stdout.isatty()

    has_failures = False
    for result in results:
        symbol = symbols[result.status]
        if use_color:
            symbol = f"{colors[result.status]}{symbol}{reset}"

        if result.status == "fail":
            has_failures = True

        print(f"  {symbol} {result.message}")

    print("=" * 50)

    if has_failures:
        print("\nSuggestion: Update invalid keys in .env or run:")
        print("  aragora config set <KEY> <value>")
        return 1
    else:
        print("\nAll API keys validated successfully!")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aragora system health check")
    parser.add_argument(
        "--validate", "-v", action="store_true", help="Validate API keys by making test calls"
    )
    args = parser.parse_args()

    sys.exit(main(validate_keys=args.validate))
