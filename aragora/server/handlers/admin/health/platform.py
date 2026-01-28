"""
Platform health check implementations.

Provides comprehensive health checks for platform integrations and deployment.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


class PlatformMixin:
    """Mixin providing platform health and deployment diagnostics.

    Should be mixed into a handler class that provides json_response access.
    """

    def platform_health(self) -> HandlerResult:
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
        components: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        warnings: List[str] = []

        # 1. Check platform rate limiters
        components["rate_limiters"] = self._check_platform_rate_limiters()

        # 2. Check platform resilience module
        components["resilience"] = self._check_platform_resilience()

        # 3. Check dead letter queue
        dlq_result = self._check_dead_letter_queue()
        components["dead_letter_queue"] = dlq_result["component"]
        if dlq_result.get("warnings"):
            warnings.extend(dlq_result["warnings"])

        # 4. Check platform metrics
        components["metrics"] = self._check_platform_metrics()

        # 5. Check individual platform circuit breakers
        circuits_result = self._check_platform_circuits()
        components["platform_circuits"] = circuits_result["component"]
        if circuits_result.get("warnings"):
            warnings.extend(circuits_result["warnings"])
        if circuits_result.get("unhealthy"):
            all_healthy = False

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

    def _check_platform_rate_limiters(self) -> Dict[str, Any]:
        """Check platform rate limiters."""
        try:
            from aragora.server.middleware.rate_limit.platform_limiter import (
                PLATFORM_RATE_LIMITS,
                get_platform_rate_limiter,
            )

            platform_limiters = {}
            for platform in PLATFORM_RATE_LIMITS.keys():
                limiter = get_platform_rate_limiter(platform)
                platform_limiters[platform] = {
                    "rpm": limiter.rpm,
                    "burst_size": limiter.burst_size,
                    "daily_limit": limiter.daily_limit,
                }

            return {
                "healthy": True,
                "status": "active",
                "platforms": list(PLATFORM_RATE_LIMITS.keys()),
                "config": platform_limiters,
            }
        except ImportError:
            return {
                "healthy": True,
                "status": "not_available",
                "note": "Platform rate limiter module not installed",
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

    def _check_platform_resilience(self) -> Dict[str, Any]:
        """Check platform resilience module."""
        try:
            from aragora.integrations.platform_resilience import (
                get_platform_resilience,
                DLQ_ENABLED,
            )

            resilience = get_platform_resilience()
            stats = resilience.get_stats()

            return {
                "healthy": True,
                "status": "active",
                "dlq_enabled": DLQ_ENABLED,
                "platforms_tracked": stats.get("platforms_tracked", 0),
                "circuit_breakers": stats.get("circuit_breakers", {}),
            }
        except ImportError:
            return {
                "healthy": True,
                "status": "not_available",
                "note": "Platform resilience module not installed",
            }
        except Exception as e:
            return {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

    def _check_dead_letter_queue(self) -> Dict[str, Any]:
        """Check dead letter queue status.

        Returns:
            Dict with 'component' (health status) and 'warnings' (list of warnings)
        """
        warnings: List[str] = []
        try:
            from aragora.integrations.platform_resilience import get_dlq

            dlq = get_dlq()
            dlq_stats = dlq.get_stats()

            component = {
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

            return {"component": component, "warnings": warnings}

        except ImportError:
            return {
                "component": {
                    "healthy": True,
                    "status": "not_available",
                    "note": "DLQ module not installed",
                },
                "warnings": warnings,
            }
        except Exception as e:
            return {
                "component": {
                    "healthy": True,
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                },
                "warnings": warnings,
            }

    def _check_platform_metrics(self) -> Dict[str, Any]:
        """Check platform metrics availability."""
        try:
            from aragora.observability.metrics.platform import (
                get_platform_metrics_summary,
            )

            metrics = get_platform_metrics_summary()
            return {
                "healthy": True,
                "status": "active",
                "prometheus_enabled": True,
                "summary": metrics,
            }
        except ImportError:
            return {
                "healthy": True,
                "status": "not_available",
                "prometheus_enabled": False,
            }
        except Exception as e:
            return {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

    def _check_platform_circuits(self) -> Dict[str, Any]:
        """Check individual platform circuit breakers.

        Returns:
            Dict with 'component' (health status), 'warnings' (list), and 'unhealthy' (bool)
        """
        warnings: List[str] = []
        unhealthy = False

        try:
            from aragora.resilience import get_circuit_breaker

            platform_circuits = {}
            platforms = ["slack", "discord", "teams", "telegram", "whatsapp", "matrix"]

            for platform in platforms:
                try:
                    cb = get_circuit_breaker(f"platform_{platform}")
                    if cb:
                        platform_circuits[platform] = {
                            "state": (
                                cb.state.value if hasattr(cb.state, "value") else str(cb.state)
                            ),
                            "failure_count": cb.failure_count,  # type: ignore[attr-defined]
                            "success_count": getattr(cb, "success_count", 0),
                        }
                except Exception as e:
                    logger.debug(f"Error getting circuit breaker for {platform}: {e}")
                    platform_circuits[platform] = {"state": "not_configured"}

            component = {
                "healthy": True,
                "status": "active",
                "circuits": platform_circuits,
            }

            # Check for open circuits
            open_circuits = [p for p, c in platform_circuits.items() if c.get("state") == "open"]
            if open_circuits:
                unhealthy = True
                warnings.append(f"Open circuit breakers: {', '.join(open_circuits)}")

            return {"component": component, "warnings": warnings, "unhealthy": unhealthy}

        except ImportError:
            return {
                "component": {
                    "healthy": True,
                    "status": "not_available",
                    "note": "Circuit breaker module not available",
                },
                "warnings": warnings,
                "unhealthy": unhealthy,
            }
        except Exception as e:
            return {
                "component": {
                    "healthy": True,
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                },
                "warnings": warnings,
                "unhealthy": unhealthy,
            }

    def deployment_diagnostics(self) -> HandlerResult:
        """Comprehensive deployment diagnostics endpoint.

        Runs the full deployment validator and returns detailed results
        including all production readiness checks:
        - JWT secret strength and configuration
        - AI provider API key availability
        - Database connectivity (Supabase/PostgreSQL)
        - Redis configuration for distributed state
        - CORS and security settings
        - Rate limiting configuration
        - TLS/HTTPS settings
        - Encryption key configuration
        - Storage accessibility

        This endpoint is useful for:
        - Pre-deployment validation
        - Production readiness verification
        - Debugging configuration issues
        - CI/CD deployment checks

        Returns:
            JSON response with comprehensive deployment validation results
        """
        start_time = time.time()

        try:
            from aragora.ops.deployment_validator import validate_deployment

            # Run async validation in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                # Already in async context - use thread pool
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, validate_deployment())
                    result = future.result(timeout=30.0)
            else:
                result = asyncio.run(validate_deployment())

            # Convert to response format
            response_data = result.to_dict()

            # Add summary information
            critical_issues = [i for i in result.issues if i.severity.value == "critical"]
            warning_issues = [i for i in result.issues if i.severity.value == "warning"]
            info_issues = [i for i in result.issues if i.severity.value == "info"]

            # Add component summary
            healthy_components = [c for c in result.components if c.status.value == "healthy"]
            degraded_components = [c for c in result.components if c.status.value == "degraded"]
            unhealthy_components = [c for c in result.components if c.status.value == "unhealthy"]

            response_data["summary"] = {
                "ready": result.ready,
                "live": result.live,
                "issues": {
                    "critical": len(critical_issues),
                    "warning": len(warning_issues),
                    "info": len(info_issues),
                    "total": len(result.issues),
                },
                "components": {
                    "healthy": len(healthy_components),
                    "degraded": len(degraded_components),
                    "unhealthy": len(unhealthy_components),
                    "total": len(result.components),
                },
            }

            # Add production readiness checklist
            response_data["checklist"] = self._generate_deployment_checklist(result)

            response_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            response_data["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"

            # Return appropriate status code
            if not result.ready:
                return json_response(response_data, status=503)
            elif len(warning_issues) > 0:
                return json_response(response_data, status=200)
            else:
                return json_response(response_data, status=200)

        except ImportError as e:
            return json_response(
                {
                    "status": "error",
                    "error": f"Deployment validator not available: {e}",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=500,
            )
        except concurrent.futures.TimeoutError:
            return json_response(
                {
                    "status": "error",
                    "error": "Deployment validation timed out after 30 seconds",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=504,
            )
        except Exception as e:
            logger.warning(f"Deployment diagnostics failed: {e}")
            return json_response(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=500,
            )

    def _generate_deployment_checklist(self, result: Any) -> Dict[str, Any]:
        """Generate a production readiness checklist from validation results.

        Args:
            result: ValidationResult from deployment validator

        Returns:
            Dictionary with checklist items and their status
        """
        # Build component lookup
        components = {c.name: c for c in result.components}
        issues_by_component: Dict[str, List[Any]] = {}
        for issue in result.issues:
            if issue.component not in issues_by_component:
                issues_by_component[issue.component] = []
            issues_by_component[issue.component].append(issue)

        def get_status(component_name: str) -> str:
            comp = components.get(component_name)
            if not comp:
                return "not_checked"
            if comp.status.value == "healthy":
                return "pass"
            elif comp.status.value == "degraded":
                return "warning"
            elif comp.status.value == "unhealthy":
                return "fail"
            return "unknown"

        def has_critical_issue(component_name: str) -> bool:
            issues = issues_by_component.get(component_name, [])
            return any(i.severity.value == "critical" for i in issues)

        return {
            "security": {
                "jwt_secret": {
                    "status": get_status("jwt_secret"),
                    "critical": has_critical_issue("jwt_secret"),
                    "description": "JWT secret configured with 32+ characters",
                },
                "encryption_key": {
                    "status": get_status("encryption"),
                    "critical": has_critical_issue("encryption"),
                    "description": "Encryption key configured (32-byte hex)",
                },
                "cors": {
                    "status": get_status("cors"),
                    "critical": has_critical_issue("cors"),
                    "description": "CORS origins properly restricted",
                },
                "tls": {
                    "status": get_status("tls"),
                    "critical": has_critical_issue("tls"),
                    "description": "TLS/HTTPS configured or behind proxy",
                },
            },
            "infrastructure": {
                "database": {
                    "status": get_status("database"),
                    "critical": has_critical_issue("database"),
                    "description": "Database connectivity verified",
                },
                "redis": {
                    "status": get_status("redis"),
                    "critical": has_critical_issue("redis"),
                    "description": "Redis configured for distributed state",
                },
                "storage": {
                    "status": get_status("storage"),
                    "critical": has_critical_issue("storage"),
                    "description": "Data directory writable",
                },
                "supabase": {
                    "status": get_status("supabase"),
                    "critical": has_critical_issue("supabase"),
                    "description": "Supabase configured (if used)",
                },
            },
            "api": {
                "api_keys": {
                    "status": get_status("api_keys"),
                    "critical": has_critical_issue("api_keys"),
                    "description": "At least one AI provider configured",
                },
                "rate_limiting": {
                    "status": get_status("rate_limiting"),
                    "critical": has_critical_issue("rate_limiting"),
                    "description": "Rate limiting enabled and configured",
                },
            },
            "environment": {
                "env_mode": {
                    "status": get_status("environment"),
                    "critical": has_critical_issue("environment"),
                    "description": "Environment mode set correctly",
                },
            },
        }


__all__ = ["PlatformMixin"]
