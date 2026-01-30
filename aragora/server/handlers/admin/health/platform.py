# mypy: ignore-errors
"""
Platform health, encryption, and startup check implementations.

Provides health checks for:
- /api/health/platform - Platform resilience (circuit breakers, rate limiters, DLQ)
- /api/health/encryption - Encryption service status
- /api/health/startup - Startup report and SLO status
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


def startup_health(handler) -> HandlerResult:
    """Startup health status - reports server startup information.

    Returns:
    - Success status
    - Total startup duration
    - SLO compliance (target: 30s)
    - Components initialized/failed
    - Checkpoints reached
    """
    start_time = time.time()

    try:
        from aragora.server.startup_transaction import (
            get_last_startup_report,
        )

        report = get_last_startup_report()
        if report is None:
            return json_response(
                {
                    "status": "unknown",
                    "message": "No startup report available",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                }
            )

        response_time_ms = round((time.time() - start_time) * 1000, 2)

        # Determine overall status
        if report.success and report.slo_met:
            status = "healthy"
        elif report.success:
            status = "warning"  # Started but SLO exceeded
        else:
            status = "degraded"

        return json_response(
            {
                "status": status,
                "startup": {
                    "success": report.success,
                    "duration_seconds": round(report.total_duration_seconds, 2),
                    "slo_seconds": report.slo_seconds,
                    "slo_met": report.slo_met,
                },
                "components": {
                    "initialized": report.components_initialized,
                    "failed": report.components_failed,
                },
                "checkpoints": [
                    {
                        "name": cp.name,
                        "elapsed_seconds": round(cp.elapsed_seconds, 2),
                    }
                    for cp in report.checkpoints
                ]
                if report.checkpoints
                else None,
                "error": report.error,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    except ImportError:
        return json_response(
            {
                "status": "not_available",
                "message": "Startup transaction module not installed",
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            },
            status=503,
        )


def encryption_health(handler) -> HandlerResult:
    """Encryption health check - verifies encryption service status.

    Checks:
    - Cryptography library availability
    - Encryption service initialization
    - Active encryption key status
    - Key age and rotation recommendations
    - Encrypt/decrypt round-trip verification
    """
    start_time = time.time()
    issues: list[str] = []
    warnings: list[str] = []
    health: dict[str, Any] = {}

    # Check 1: Crypto library availability
    try:
        from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            health["cryptography_library"] = {"healthy": True, "status": "installed"}
        else:
            health["cryptography_library"] = {"healthy": False, "status": "not_installed"}
            issues.append("Cryptography library not installed")
    except ImportError:
        health["cryptography_library"] = {"healthy": False, "status": "import_error"}
        issues.append("Cannot import encryption module")
        return json_response(
            {
                "status": "error",
                "issues": issues,
                "health": health,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            },
            status=503,
        )

    # Check 2: Encryption service initialization
    try:
        service = get_encryption_service()
        health["encryption_service"] = {"healthy": True, "status": "initialized"}
    except Exception as e:
        health["encryption_service"] = {
            "healthy": False,
            "status": "error",
            "error": str(e)[:100],
        }
        issues.append(f"Encryption service error: {str(e)[:50]}")
        return json_response(
            {
                "status": "error",
                "issues": issues,
                "health": health,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            },
            status=503,
        )

    # Check 3: Active key status
    active_key = service.get_active_key()
    if active_key:
        age_days = (datetime.now(timezone.utc) - active_key.created_at).days
        health["active_key"] = {
            "healthy": True,
            "key_id": service.get_active_key_id(),
            "version": active_key.version,
            "age_days": age_days,
            "created_at": active_key.created_at.isoformat(),
        }

        # Key age warnings
        if age_days > 90:
            warnings.append(f"Key is {age_days} days old (>90 days). Rotation recommended.")
            health["active_key"]["rotation_recommended"] = True
        elif age_days > 60:
            health["active_key"]["days_until_rotation"] = 90 - age_days
    else:
        health["active_key"] = {"healthy": False, "status": "no_active_key"}
        issues.append("No active encryption key")

    # Check 4: Encrypt/decrypt round-trip
    try:
        test_data = b"encryption_health_check"
        encrypted = service.encrypt(test_data)
        decrypted = service.decrypt(encrypted)

        if decrypted == test_data:
            health["roundtrip_test"] = {"healthy": True, "status": "passed"}
        else:
            health["roundtrip_test"] = {"healthy": False, "status": "data_mismatch"}
            issues.append("Encrypt/decrypt round-trip failed")
    except Exception as e:
        health["roundtrip_test"] = {
            "healthy": False,
            "status": "error",
            "error": str(e)[:100],
        }
        issues.append(f"Encrypt/decrypt error: {str(e)[:50]}")

    # Calculate overall status
    response_time_ms = round((time.time() - start_time) * 1000, 2)

    if issues:
        status = "error"
        http_status = 503
    elif warnings:
        status = "warning"
        http_status = 200
    else:
        status = "healthy"
        http_status = 200

    return json_response(
        {
            "status": status,
            "health": health,
            "issues": issues if issues else None,
            "warnings": warnings if warnings else None,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        },
        status=http_status,
    )


def platform_health(handler) -> HandlerResult:
    """Platform resilience health check for chat integrations.

    Checks:
    - Platform circuit breakers status (Slack, Discord, Teams, etc.)
    - Platform-specific rate limiters
    - Dead letter queue status
    - Platform delivery metrics
    - Webhook health

    Returns:
        JSON response with comprehensive platform health metrics
    """
    start_time = time.time()
    components: dict[str, dict[str, Any]] = {}
    all_healthy = True
    warnings: list[str] = []

    # 1. Check platform rate limiters
    try:
        from aragora.server.middleware.rate_limit.platform_limiter import (
            PLATFORM_RATE_LIMITS,
            get_platform_rate_limiter,
        )

        platform_limiters = {}
        for platform_name in PLATFORM_RATE_LIMITS.keys():
            limiter = get_platform_rate_limiter(platform_name)
            platform_limiters[platform_name] = {
                "rpm": limiter.rpm,
                "burst_size": limiter.burst_size,
                "daily_limit": limiter.daily_limit,
            }

        components["rate_limiters"] = {
            "healthy": True,
            "status": "active",
            "platforms": list(PLATFORM_RATE_LIMITS.keys()),
            "config": platform_limiters,
        }
    except ImportError:
        components["rate_limiters"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Platform rate limiter module not installed",
        }
    except Exception as e:
        components["rate_limiters"] = {
            "healthy": False,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }
        all_healthy = False

    # 2. Check platform resilience module
    try:
        from aragora.integrations.platform_resilience import (
            get_platform_resilience,
            DLQ_ENABLED,
        )

        resilience = get_platform_resilience()
        stats = resilience.get_stats()

        components["resilience"] = {
            "healthy": True,
            "status": "active",
            "dlq_enabled": DLQ_ENABLED,
            "platforms_tracked": stats.get("platforms_tracked", 0),
            "circuit_breakers": stats.get("circuit_breakers", {}),
        }
    except ImportError:
        components["resilience"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Platform resilience module not installed",
        }
    except Exception as e:
        components["resilience"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 3. Check dead letter queue
    try:
        from aragora.integrations.platform_resilience import get_dlq

        dlq = get_dlq()
        dlq_stats = dlq.get_stats()

        components["dead_letter_queue"] = {
            "healthy": True,
            "status": "active",
            "pending_count": dlq_stats.get("pending", 0),
            "failed_count": dlq_stats.get("failed", 0),
            "processed_count": dlq_stats.get("processed", 0),
        }

        # Warn if DLQ is backing up
        pending = dlq_stats.get("pending", 0)
        if pending > 100:
            warnings.append(f"DLQ has {pending} pending messages")
        elif pending > 50:
            warnings.append(f"DLQ has {pending} pending messages (elevated)")

    except ImportError:
        components["dead_letter_queue"] = {
            "healthy": True,
            "status": "not_available",
            "note": "DLQ module not installed",
        }
    except Exception as e:
        components["dead_letter_queue"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 4. Check platform metrics
    try:
        from aragora.observability.metrics.platform import (
            get_platform_metrics_summary,
        )

        metrics = get_platform_metrics_summary()
        components["metrics"] = {
            "healthy": True,
            "status": "active",
            "prometheus_enabled": True,
            "summary": metrics,
        }
    except ImportError:
        components["metrics"] = {
            "healthy": True,
            "status": "not_available",
            "prometheus_enabled": False,
        }
    except Exception as e:
        components["metrics"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 5. Check individual platform circuit breakers
    try:
        from aragora.resilience import get_circuit_breaker

        platform_circuits = {}
        platforms = ["slack", "discord", "teams", "telegram", "whatsapp", "matrix"]

        for plat in platforms:
            try:
                cb = get_circuit_breaker(f"platform_{plat}")
                if cb:
                    platform_circuits[plat] = {
                        "state": (cb.state.value if hasattr(cb.state, "value") else str(cb.state)),
                        "failure_count": getattr(cb, "failure_count", cb.failures),
                        "success_count": getattr(cb, "success_count", 0),
                    }
            except Exception as e:
                logger.debug(f"Error getting circuit breaker for {plat}: {e}")
                platform_circuits[plat] = {"state": "not_configured"}

        components["platform_circuits"] = {
            "healthy": True,
            "status": "active",
            "circuits": platform_circuits,
        }

        # Check for open circuits
        open_circuits = [p for p, c in platform_circuits.items() if c.get("state") == "open"]
        if open_circuits:
            all_healthy = False
            warnings.append(f"Open circuit breakers: {', '.join(open_circuits)}")

    except ImportError:
        components["platform_circuits"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Circuit breaker module not available",
        }
    except Exception as e:
        components["platform_circuits"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # Calculate response time
    response_time_ms = round((time.time() - start_time) * 1000, 2)

    # Determine overall status
    healthy_count = sum(1 for c in components.values() if c.get("healthy", False))
    active_count = sum(1 for c in components.values() if c.get("status") == "active")

    status = "healthy" if all_healthy else "degraded"
    if active_count == 0:
        status = "not_configured"
    elif warnings and all_healthy:
        status = "healthy_with_warnings"

    return json_response(
        {
            "status": status,
            "summary": {
                "total_components": len(components),
                "healthy": healthy_count,
                "active": active_count,
            },
            "components": components,
            "warnings": warnings if warnings else None,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )
