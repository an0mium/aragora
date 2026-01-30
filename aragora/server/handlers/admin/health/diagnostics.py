"""
Deployment diagnostics and production readiness checklist.

Provides:
- /api/diagnostics - Full deployment validation (requires admin:diagnostics)
- /api/diagnostics/deployment - Same as above
- Production readiness checklist generation

Security: All endpoints require admin:diagnostics permission as they expose
sensitive deployment configuration including API key availability, database
connectivity details, and security configuration status.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...base import HandlerResult, json_response, error_response

if TYPE_CHECKING:
    from aragora.ops.deployment_validator import ValidationResult

logger = logging.getLogger(__name__)


def _check_diagnostics_permission(handler: Any) -> HandlerResult | None:
    """Check admin:diagnostics permission for diagnostics endpoints.

    This endpoint exposes sensitive deployment information and must be
    protected with proper RBAC checks.

    Returns:
        None if authorized, error HandlerResult if not authorized.
    """
    from aragora.server.auth import auth_config
    from aragora.billing.jwt_auth import extract_user_from_request

    # If auth is disabled globally, allow access (development mode)
    if not auth_config.enabled:
        return None

    # Extract user context from request
    user_ctx = extract_user_from_request(handler, None)
    if not user_ctx.is_authenticated:
        return error_response("Authentication required for diagnostics endpoint", 401)

    # Check for admin:diagnostics permission
    try:
        from aragora.rbac.checker import get_permission_checker

        checker = get_permission_checker()
        if checker and hasattr(user_ctx, "user_id"):
            from aragora.rbac.models import AuthorizationContext

            raw_roles = getattr(user_ctx, "roles", [])
            roles_set = set(raw_roles) if raw_roles else set()
            auth_ctx = AuthorizationContext(
                user_id=user_ctx.user_id,
                org_id=getattr(user_ctx, "org_id", None),
                roles=roles_set,
            )
            result = checker.check_permission(auth_ctx, "admin:diagnostics")
            if not result.allowed:
                return error_response("Permission denied: admin:diagnostics required", 403)
    except ImportError:
        # RBAC module not available, fall back to auth-only check
        logger.debug("RBAC module not available, allowing authenticated access")

    return None


def deployment_diagnostics(handler: Any) -> HandlerResult:
    """Comprehensive deployment diagnostics endpoint.

    Requires: admin:diagnostics permission

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
    # Check permission before exposing sensitive deployment info
    permission_error = _check_diagnostics_permission(handler)
    if permission_error:
        return permission_error

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
        response_data["checklist"] = _generate_checklist(result)

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


def _generate_checklist(result: "ValidationResult") -> dict[str, Any]:
    """Generate a production readiness checklist from validation results.

    Args:
        result: ValidationResult from deployment validator

    Returns:
        Dictionary with checklist items and their status
    """
    # Build component lookup
    components = {c.name: c for c in result.components}
    issues_by_component: dict[str, list] = {}
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
