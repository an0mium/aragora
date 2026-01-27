"""
Skills endpoint handlers.

Endpoints:
- GET /api/skills - List all registered skills
- GET /api/skills/:name - Get skill details
- POST /api/skills/invoke - Invoke a skill by name
- GET /api/skills/:name/metrics - Get skill execution metrics
"""

from __future__ import annotations

__all__ = [
    "SkillsHandler",
]

import asyncio
import logging
from typing import Any, Optional

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.rbac.decorators import require_permission
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for skill endpoints (30 invocations per minute)
_skills_limiter = RateLimiter(requests_per_minute=30)

# Lazy imports for skills system
try:
    from aragora.skills import (
        SkillContext,
        SkillRegistry,
        SkillStatus,
        get_skill_registry,
    )

    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    get_skill_registry = None  # type: ignore[assignment]
    SkillRegistry = None  # type: ignore[assignment]
    SkillContext = None  # type: ignore[assignment]
    SkillStatus = None  # type: ignore[assignment]


class SkillsHandler(BaseHandler):
    """Handler for skills system endpoints."""

    ROUTES: list[str] = [
        "/api/skills",
        "/api/skills/invoke",
        "/api/skills/*/metrics",
        "/api/skills/*",  # Must be last due to wildcard
    ]

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)  # type: ignore[arg-type]
        self._registry: Optional["SkillRegistry"] = None

    def _get_registry(self) -> Optional["SkillRegistry"]:
        """Get or create the skill registry singleton."""
        if self._registry is None and SKILLS_AVAILABLE and get_skill_registry:
            self._registry = get_skill_registry()
        return self._registry

    @handle_errors
    async def handle_get(self, path: str, request: Any) -> HandlerResult:
        """Handle GET requests for skills endpoints."""
        path = strip_version_prefix(path)

        if not SKILLS_AVAILABLE:
            return error_response(
                503,
                "Skills system not available",
                code="SKILLS_UNAVAILABLE",
            )

        # Rate limit check
        client_ip = get_client_ip(request)
        if not _skills_limiter.check(client_ip):
            return error_response(
                429,
                "Rate limit exceeded for skills endpoints",
                code="RATE_LIMITED",
            )

        # GET /api/skills - List all skills
        if path == "/api/skills":
            return await self._list_skills(request)

        # GET /api/skills/:name/metrics
        if "/metrics" in path:
            parts = path.split("/")
            if len(parts) >= 4:
                skill_name = parts[3]
                return await self._get_skill_metrics(skill_name, request)

        # GET /api/skills/:name - Get skill details
        parts = path.split("/")
        if len(parts) >= 4:
            skill_name = parts[3]
            return await self._get_skill(skill_name, request)

        return error_response(404, f"Unknown skills endpoint: {path}")

    @handle_errors
    async def handle_post(self, path: str, request: Any) -> HandlerResult:
        """Handle POST requests for skills endpoints."""
        path = strip_version_prefix(path)

        if not SKILLS_AVAILABLE:
            return error_response(
                503,
                "Skills system not available",
                code="SKILLS_UNAVAILABLE",
            )

        # Rate limit check (stricter for invocations)
        client_ip = get_client_ip(request)
        if not _skills_limiter.check(client_ip):
            return error_response(
                429,
                "Rate limit exceeded for skill invocations",
                code="RATE_LIMITED",
            )

        # POST /api/skills/invoke
        if path == "/api/skills/invoke":
            return await self._invoke_skill(request)

        return error_response(404, f"Unknown skills endpoint: {path}")

    @require_permission("skills:read")
    async def _list_skills(self, request: Any) -> HandlerResult:
        """List all registered skills."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                503,
                "Skill registry not available",
                code="REGISTRY_UNAVAILABLE",
            )

        skills = []
        for skill in registry.list_skills():
            manifest = skill.manifest
            skills.append(
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description or "",
                    "capabilities": [cap.value for cap in manifest.capabilities],
                    "input_schema": manifest.input_schema or {},
                    "tags": getattr(manifest, "tags", []),
                }
            )

        return json_response(
            {
                "skills": skills,
                "total": len(skills),
            }
        )

    @require_permission("skills:read")
    async def _get_skill(self, name: str, request: Any) -> HandlerResult:
        """Get details for a specific skill."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                503,
                "Skill registry not available",
                code="REGISTRY_UNAVAILABLE",
            )

        skill = registry.get(name)
        if not skill:
            return error_response(
                404,
                f"Skill not found: {name}",
                code="SKILL_NOT_FOUND",
            )

        manifest = skill.manifest
        return json_response(
            {
                "name": manifest.name,
                "version": manifest.version,
                "description": manifest.description or "",
                "capabilities": [cap.value for cap in manifest.capabilities],
                "input_schema": manifest.input_schema or {},
                "output_schema": manifest.output_schema or {},
                "rate_limit_per_minute": manifest.rate_limit_per_minute,
                "timeout_seconds": manifest.timeout_seconds,
                "tags": getattr(manifest, "tags", []),
            }
        )

    @require_permission("skills:read")
    async def _get_skill_metrics(self, name: str, request: Any) -> HandlerResult:
        """Get execution metrics for a specific skill."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                503,
                "Skill registry not available",
                code="REGISTRY_UNAVAILABLE",
            )

        skill = registry.get(name)
        if not skill:
            return error_response(
                404,
                f"Skill not found: {name}",
                code="SKILL_NOT_FOUND",
            )

        # Get metrics from registry
        metrics = registry.get_metrics(name)
        if not metrics:
            return json_response(
                {
                    "skill": name,
                    "total_invocations": 0,
                    "successful_invocations": 0,
                    "failed_invocations": 0,
                    "average_latency_ms": 0,
                    "last_invoked": None,
                }
            )

        return json_response(
            {
                "skill": name,
                "total_invocations": metrics.total_invocations,
                "successful_invocations": metrics.successful_invocations,
                "failed_invocations": metrics.failed_invocations,
                "average_latency_ms": metrics.average_latency_ms,
                "last_invoked": metrics.last_invoked.isoformat() if metrics.last_invoked else None,
            }
        )

    @require_permission("skills:invoke")
    async def _invoke_skill(self, request: Any) -> HandlerResult:
        """Invoke a skill by name."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                503,
                "Skill registry not available",
                code="REGISTRY_UNAVAILABLE",
            )

        # Parse request body
        try:
            if hasattr(request, "json"):
                body = await request.json()
            else:
                body = request.get("body", {})
        except Exception:
            return error_response(400, "Invalid JSON body")

        skill_name = body.get("skill")
        if not skill_name:
            return error_response(400, "Missing required field: skill")

        input_data = body.get("input", {})
        user_id = body.get("user_id", "api")
        permissions = set(body.get("permissions", ["skills:invoke"]))
        timeout = body.get("timeout", 30.0)

        # Validate skill exists
        skill = registry.get(skill_name)
        if not skill:
            return error_response(
                404,
                f"Skill not found: {skill_name}",
                code="SKILL_NOT_FOUND",
            )

        # Create skill context
        ctx = SkillContext(
            user_id=user_id,
            permissions=permissions,
            metadata=body.get("metadata", {}),
        )

        # Invoke with timeout
        try:
            result = await asyncio.wait_for(
                registry.invoke(skill_name, input_data, ctx),
                timeout=min(timeout, 60.0),  # Cap at 60 seconds
            )

            if result.status == SkillStatus.SUCCESS:
                return json_response(
                    {
                        "status": "success",
                        "output": result.output,
                        "execution_time_ms": result.execution_time_ms,
                        "metadata": result.metadata or {},
                    }
                )
            elif result.status == SkillStatus.ERROR:
                return json_response(
                    {
                        "status": "error",
                        "error": result.error or "Unknown error",
                        "execution_time_ms": result.execution_time_ms,
                    },
                    status=500,
                )
            elif result.status == SkillStatus.RATE_LIMITED:
                return error_response(
                    429,
                    result.error or "Skill rate limited",
                    code="SKILL_RATE_LIMITED",
                )
            elif result.status == SkillStatus.PERMISSION_DENIED:
                return error_response(
                    403,
                    result.error or "Permission denied",
                    code="PERMISSION_DENIED",
                )
            else:
                return json_response(
                    {
                        "status": result.status.value,
                        "output": result.output,
                        "error": result.error,
                        "execution_time_ms": result.execution_time_ms,
                    }
                )

        except asyncio.TimeoutError:
            return error_response(
                408,
                f"Skill invocation timed out after {timeout}s",
                code="TIMEOUT",
            )
        except Exception as e:
            logger.exception(f"Skill invocation error for {skill_name}")
            return error_response(
                500,
                f"Skill invocation failed: {str(e)}",
                code="INVOCATION_ERROR",
            )
