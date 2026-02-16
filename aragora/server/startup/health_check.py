"""Startup health checks for Aragora server.

Validates critical subsystems at server startup and provides
a summary report. Also used by ``aragora doctor`` CLI command.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def run_startup_health_checks() -> dict[str, Any]:
    """Run startup health checks and return a summary report.

    Checks:
    - API key availability (at least one LLM provider configured)
    - Database connectivity (SQLite or Postgres)
    - Knowledge Mound accessibility
    - Health registry status

    Returns:
        Dict with check results and overall status.
    """
    results: dict[str, dict[str, Any]] = {}

    # 1. Check API key availability
    results["api_keys"] = _check_api_keys()

    # 2. Check database connectivity
    results["database"] = _check_database()

    # 3. Check Knowledge Mound
    results["knowledge_mound"] = _check_knowledge_mound()

    # 4. Check Health Registry
    results["health_registry"] = _check_health_registry()

    # Compute overall status
    all_passed = all(r.get("status") == "ok" for r in results.values())
    warnings = [name for name, r in results.items() if r.get("status") == "warning"]
    errors = [name for name, r in results.items() if r.get("status") == "error"]

    overall = "ok" if all_passed else ("warning" if not errors else "error")

    report = {
        "overall": overall,
        "checks": results,
        "warnings": warnings,
        "errors": errors,
    }

    _print_report(report)
    return report


def _check_api_keys() -> dict[str, Any]:
    """Check that at least one LLM provider API key is configured."""
    provider_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
    }

    available = []
    for provider, env_var in provider_keys.items():
        if os.environ.get(env_var):
            available.append(provider)

    if available:
        return {
            "status": "ok",
            "message": f"API keys configured: {', '.join(available)}",
            "providers": available,
        }

    return {
        "status": "warning",
        "message": "No LLM provider API keys found. Set at least one of: "
        + ", ".join(provider_keys.values()),
        "providers": [],
    }


def _check_database() -> dict[str, Any]:
    """Check database connectivity."""
    try:
        from aragora.storage.database import get_database

        db = get_database()
        if db is None:
            return {
                "status": "warning",
                "message": "Database not configured (using in-memory fallback)",
            }

        # Try a simple connectivity check
        if hasattr(db, "is_connected"):
            connected = db.is_connected()
        else:
            connected = True  # Assume connected if no check method

        if connected:
            backend = getattr(db, "backend", "unknown")
            return {
                "status": "ok",
                "message": f"Database connected ({backend})",
            }

        return {
            "status": "error",
            "message": "Database configured but not connected",
        }
    except ImportError:
        return {
            "status": "warning",
            "message": "Database module not available",
        }
    except (OSError, RuntimeError, ValueError) as e:
        return {
            "status": "error",
            "message": f"Database check failed: {e}",
        }


def _check_knowledge_mound() -> dict[str, Any]:
    """Check Knowledge Mound accessibility."""
    try:
        from aragora.knowledge.mound import get_knowledge_mound

        mound = get_knowledge_mound()
        if mound is not None:
            return {
                "status": "ok",
                "message": "Knowledge Mound accessible",
            }

        return {
            "status": "warning",
            "message": "Knowledge Mound not initialized (will auto-create on first use)",
        }
    except ImportError:
        return {
            "status": "warning",
            "message": "Knowledge Mound module not available",
        }
    except (OSError, RuntimeError, ValueError) as e:
        return {
            "status": "warning",
            "message": f"Knowledge Mound check failed: {e}",
        }


def _check_health_registry() -> dict[str, Any]:
    """Check global health registry status."""
    try:
        from aragora.resilience.health import get_global_health_registry

        registry = get_global_health_registry()
        if registry is None:
            return {
                "status": "warning",
                "message": "Health registry not available",
            }

        if hasattr(registry, "get_report"):
            report = registry.get_report()
            healthy_count = sum(1 for r in report.values() if r) if isinstance(report, dict) else 0
            total = len(report) if isinstance(report, dict) else 0
            return {
                "status": "ok",
                "message": f"Health registry: {healthy_count}/{total} checks passing",
            }

        return {
            "status": "ok",
            "message": "Health registry available",
        }
    except ImportError:
        return {
            "status": "warning",
            "message": "Health registry module not available",
        }
    except (OSError, RuntimeError, ValueError) as e:
        return {
            "status": "warning",
            "message": f"Health registry check failed: {e}",
        }


def _print_report(report: dict[str, Any]) -> None:
    """Print a formatted startup health check report."""
    overall = report["overall"]
    status_icon = {"ok": "+", "warning": "!", "error": "x"}.get(overall, "?")

    print(f"\n[{status_icon}] Startup Health Check: {overall.upper()}")
    print("-" * 50)

    for name, check in report["checks"].items():
        status = check.get("status", "unknown")
        message = check.get("message", "")
        icon = {"ok": "+", "warning": "!", "error": "x"}.get(status, "?")
        print(f"  [{icon}] {name}: {message}")

    if report["errors"]:
        print(f"\n  Errors: {', '.join(report['errors'])}")
    if report["warnings"]:
        print(f"  Warnings: {', '.join(report['warnings'])}")

    print()
