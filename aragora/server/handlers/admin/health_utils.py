"""
Health check utility functions.

Standalone functions for health checks that can be used by HealthHandler
and other components. Extracted from health.py for better modularity.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def check_filesystem_health(test_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Check filesystem write access to data directory.

    Args:
        test_dir: Directory to test write access. Defaults to temp dir.

    Returns:
        Dict with healthy status and details.
    """
    try:
        if test_dir is None or not test_dir.exists():
            test_dir = Path(tempfile.gettempdir())

        test_file = test_dir / f".health_check_{os.getpid()}"
        try:
            # Write test
            test_file.write_text("health_check")
            # Read verify
            content = test_file.read_text()
            if content != "health_check":
                return {"healthy": False, "error": "Read verification failed"}
            return {"healthy": True, "path": str(test_dir)}
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    except PermissionError as e:
        return {"healthy": False, "error": f"Permission denied: {e}"}
    except OSError as e:
        return {"healthy": False, "error": f"Filesystem error: {e}"}


def check_redis_health(redis_url: Optional[str] = None) -> Dict[str, Any]:
    """Check Redis connectivity if configured.

    Args:
        redis_url: Redis connection URL. If None, reads from environment.

    Returns:
        Dict with healthy status and details.
    """
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL") or os.environ.get("CACHE_REDIS_URL")

    if not redis_url:
        return {"healthy": True, "configured": False, "note": "Redis not configured"}

    try:
        import redis

        client = redis.from_url(redis_url, socket_timeout=2.0)
        ping_start = time.time()
        pong = client.ping()
        ping_latency = round((time.time() - ping_start) * 1000, 2)

        if pong:
            return {"healthy": True, "configured": True, "latency_ms": ping_latency}
        else:
            return {"healthy": False, "configured": True, "error": "Ping returned False"}

    except ImportError:
        return {"healthy": True, "configured": True, "warning": "redis package not installed"}
    except Exception as e:
        return {
            "healthy": False,
            "configured": True,
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_ai_providers_health() -> Dict[str, Any]:
    """Check AI provider API key availability.

    Returns:
        Dict with provider availability status.
    """
    providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "grok": "GROK_API_KEY",
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    available = {}
    for name, env_var in providers.items():
        key = os.environ.get(env_var)
        available[name] = key is not None and len(key) > 10

    any_available = any(available.values())
    available_count = sum(1 for v in available.values() if v)

    return {
        "healthy": True,
        "any_available": any_available,
        "available_count": available_count,
        "providers": available,
    }


def check_security_services(is_production: Optional[bool] = None) -> Dict[str, Any]:
    """Check security services health.

    Verifies:
    - Encryption service is available and functional
    - RBAC module is available
    - Audit logger is configured
    - Production encryption key is set (in production mode)

    Args:
        is_production: Whether running in production. If None, reads from env.

    Returns:
        Dict with security services status.
    """
    from aragora.config.secrets import get_secret

    if is_production is None:
        is_production = os.environ.get("ARAGORA_ENV") == "production"

    result: Dict[str, Any] = {"healthy": True}

    # Check encryption service
    try:
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        result["encryption_available"] = service is not None
        result["encryption_configured"] = bool(get_secret("ARAGORA_ENCRYPTION_KEY"))

        if is_production and not result["encryption_configured"]:
            result["healthy"] = False
            result["encryption_warning"] = (
                "ARAGORA_ENCRYPTION_KEY not set - secrets may be unencrypted"
            )
    except ImportError:
        result["encryption_available"] = False
        result["encryption_warning"] = "Encryption module not available"
    except Exception as e:
        result["encryption_available"] = False
        result["encryption_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # Check RBAC module
    try:
        from aragora.rbac import AuthorizationContext, check_permission  # noqa: F401

        result["rbac_available"] = True
    except ImportError:
        result["rbac_available"] = False
        result["rbac_warning"] = "RBAC module not available"

    # Check audit logger
    try:
        from aragora.server.middleware.audit_logger import get_audit_logger

        audit_logger = get_audit_logger()
        result["audit_logger_configured"] = audit_logger is not None
    except ImportError:
        result["audit_logger_configured"] = False
        result["audit_warning"] = "Audit logger module not available"
    except Exception as e:
        result["audit_logger_configured"] = False
        result["audit_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # Check JWT auth module
    try:
        from aragora.billing.jwt_auth import extract_user_from_request  # noqa: F401

        result["jwt_auth_available"] = True
    except ImportError:
        result["jwt_auth_available"] = False
        result["jwt_warning"] = "JWT auth module not available"

    return result


def check_database_health(database_url: Optional[str] = None) -> Dict[str, Any]:
    """Check database connectivity.

    Args:
        database_url: Database connection URL. If None, reads from environment.

    Returns:
        Dict with database health status.
    """
    if database_url is None:
        database_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")

    if not database_url:
        return {"healthy": True, "configured": False, "note": "Database not configured"}

    try:
        import asyncio
        from aragora.server.startup import validate_database_connectivity

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, validate_database_connectivity(timeout_seconds=2.0)
                )
                db_ok, db_msg = future.result(timeout=3.0)
        else:
            db_ok, db_msg = asyncio.run(validate_database_connectivity(timeout_seconds=2.0))

        return {
            "healthy": db_ok,
            "configured": True,
            "message": db_msg,
        }

    except ImportError:
        return {"healthy": True, "configured": True, "status": "check_skipped"}
    except Exception as e:
        return {
            "healthy": False,
            "configured": True,
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def get_uptime_info(start_time: float) -> Dict[str, Any]:
    """Get server uptime information.

    Args:
        start_time: Server start timestamp (time.time()).

    Returns:
        Dict with uptime details.
    """
    uptime_seconds = time.time() - start_time

    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    if days > 0:
        uptime_str = f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        uptime_str = f"{hours}h {minutes}m {seconds}s"
    else:
        uptime_str = f"{minutes}m {seconds}s"

    return {
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_human": uptime_str,
    }


def check_stripe_health() -> Dict[str, Any]:
    """Check Stripe API connectivity.

    Validates the Stripe API key by making a lightweight API call.

    Returns:
        Dict with Stripe health status.
    """
    stripe_key = os.environ.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_API_KEY")

    if not stripe_key:
        return {"healthy": True, "configured": False, "note": "Stripe not configured"}

    try:
        import stripe

        stripe.api_key = stripe_key
        ping_start = time.time()
        # Use a lightweight API call - list 1 customer to verify connectivity
        stripe.Customer.list(limit=1)
        ping_latency = round((time.time() - ping_start) * 1000, 2)

        return {
            "healthy": True,
            "configured": True,
            "latency_ms": ping_latency,
        }

    except ImportError:
        return {"healthy": True, "configured": True, "warning": "stripe package not installed"}
    except stripe.error.AuthenticationError as e:  # type: ignore[attr-defined]
        return {
            "healthy": False,
            "configured": True,
            "error": f"Authentication failed: {str(e)[:80]}",
        }
    except stripe.error.APIConnectionError as e:  # type: ignore[attr-defined]
        return {
            "healthy": False,
            "configured": True,
            "error": f"Connection failed: {str(e)[:80]}",
        }
    except Exception as e:
        return {
            "healthy": False,
            "configured": True,
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_slack_health() -> Dict[str, Any]:
    """Check Slack API connectivity.

    Validates the Slack bot token by calling auth.test endpoint.

    Returns:
        Dict with Slack health status.
    """
    slack_token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")

    if not slack_token:
        return {"healthy": True, "configured": False, "note": "Slack not configured"}

    try:
        import httpx

        ping_start = time.time()
        response = httpx.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {slack_token}"},
            timeout=5.0,
        )
        ping_latency = round((time.time() - ping_start) * 1000, 2)

        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                return {
                    "healthy": True,
                    "configured": True,
                    "latency_ms": ping_latency,
                    "team": data.get("team"),
                    "user": data.get("user"),
                }
            else:
                return {
                    "healthy": False,
                    "configured": True,
                    "error": data.get("error", "Unknown error"),
                }
        else:
            return {
                "healthy": False,
                "configured": True,
                "error": f"HTTP {response.status_code}",
            }

    except ImportError:
        return {"healthy": True, "configured": True, "warning": "httpx package not installed"}
    except httpx.TimeoutException:
        return {
            "healthy": False,
            "configured": True,
            "error": "Request timeout",
        }
    except Exception as e:
        return {
            "healthy": False,
            "configured": True,
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }
