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
    get_skill_registry = None  # type: ignore[misc,assignment]
    SkillRegistry = None  # type: ignore[misc,assignment]
    SkillContext = None  # type: ignore[misc,assignment]
    SkillStatus = None  # type: ignore[misc,assignment]


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

    @handle_errors("skills GET request")
    async def handle_get(self, path: str, request: Any) -> HandlerResult:
        """Handle GET requests for skills endpoints."""
        path = strip_version_prefix(path)

        if not SKILLS_AVAILABLE:
            return error_response(
                "Skills system not available",
                503,
                code="SKILLS_UNAVAILABLE",
            )

        # Rate limit check
        client_ip = get_client_ip(request)
        if not _skills_limiter.is_allowed(client_ip):
            return error_response(
                "Rate limit exceeded for skills endpoints",
                429,
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

        return error_response(f"Unknown skills endpoint: {path}", 404)

    @handle_errors("skills POST request")
    async def handle_post(self, path: str, request: Any) -> HandlerResult:
        """Handle POST requests for skills endpoints."""
        path = strip_version_prefix(path)

        if not SKILLS_AVAILABLE:
            return error_response(
                "Skills system not available",
                503,
                code="SKILLS_UNAVAILABLE",
            )

        # Rate limit check (stricter for invocations)
        client_ip = get_client_ip(request)
        if not _skills_limiter.is_allowed(client_ip):
            return error_response(
                "Rate limit exceeded for skill invocations",
                429,
                code="RATE_LIMITED",
            )

        # POST /api/skills/invoke
        if path == "/api/skills/invoke":
            return await self._invoke_skill(request)

        return error_response(f"Unknown skills endpoint: {path}", 404)

    @require_permission("skills:read")
    async def _list_skills(self, request: Any) -> HandlerResult:
        """List all registered skills."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                "Skill registry not available",
                503,
                code="REGISTRY_UNAVAILABLE",
            )

        skills = []
        for manifest in registry.list_skills():
            skills.append(
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description or "",
                    "capabilities": [cap.value for cap in manifest.capabilities],
                    "input_schema": manifest.input_schema or {},
                    "tags": manifest.tags,
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
                "Skill registry not available",
                503,
                code="REGISTRY_UNAVAILABLE",
            )

        skill = registry.get(name)
        if not skill:
            return error_response(
                f"Skill not found: {name}",
                404,
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
                "timeout_seconds": manifest.max_execution_time_seconds,
                "tags": manifest.tags,
            }
        )

    @require_permission("skills:read")
    async def _get_skill_metrics(self, name: str, request: Any) -> HandlerResult:
        """Get execution metrics for a specific skill."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                "Skill registry not available",
                503,
                code="REGISTRY_UNAVAILABLE",
            )

        skill = registry.get(name)
        if not skill:
            return error_response(
                f"Skill not found: {name}",
                404,
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
                "total_invocations": metrics.get("total_invocations", 0),
                "successful_invocations": metrics.get("successful_invocations", 0),
                "failed_invocations": metrics.get("failed_invocations", 0),
                "average_latency_ms": metrics.get("average_latency_ms", 0),
                "last_invoked": metrics.get("last_invoked").isoformat()
                if metrics.get("last_invoked")
                else None,
            }
        )

    @require_permission("skills:invoke")
    async def _invoke_skill(self, request: Any) -> HandlerResult:
        """Invoke a skill by name."""
        registry = self._get_registry()
        if not registry:
            return error_response(
                "Skill registry not available",
                503,
                code="REGISTRY_UNAVAILABLE",
            )

        # Parse request body
        try:
            if hasattr(request, "json"):
                body = await request.json()
            else:
                body = request.get("body", {})
        except Exception:
            return error_response("Invalid JSON body", 400)

        skill_name = body.get("skill")
        if not skill_name:
            return error_response("Missing required field: skill", 400)

        input_data = body.get("input", {})
        user_id = body.get("user_id", "api")
        permissions = set(body.get("permissions", ["skills:invoke"]))
        timeout = body.get("timeout", 30.0)

        # Validate skill exists
        skill = registry.get(skill_name)
        if not skill:
            return error_response(
                f"Skill not found: {skill_name}",
                404,
                code="SKILL_NOT_FOUND",
            )

        # Create skill context
        ctx = SkillContext(
            user_id=user_id,
            permissions=list(permissions),
            config=body.get("metadata", {}),
        )

        # Invoke with timeout
        try:
            result = await asyncio.wait_for(
                registry.invoke(skill_name, input_data, ctx),
                timeout=min(timeout, 60.0),  # Cap at 60 seconds
            )

            # Compute execution time in ms from duration_seconds
            exec_time_ms = int(result.duration_seconds * 1000) if result.duration_seconds else None

            if result.status == SkillStatus.SUCCESS:
                return json_response(
                    {
                        "status": "success",
                        "output": result.data,
                        "execution_time_ms": exec_time_ms,
                        "metadata": result.metadata or {},
                    }
                )
            elif result.status == SkillStatus.FAILURE:
                return json_response(
                    {
                        "status": "error",
                        "error": result.error_message or "Unknown error",
                        "execution_time_ms": exec_time_ms,
                    },
                    status=500,
                )
            elif result.status == SkillStatus.RATE_LIMITED:
                return error_response(
                    result.error_message or "Skill rate limited",
                    429,
                    code="SKILL_RATE_LIMITED",
                )
            elif result.status == SkillStatus.PERMISSION_DENIED:
                return error_response(
                    result.error_message or "Permission denied",
                    403,
                    code="PERMISSION_DENIED",
                )
            else:
                return json_response(
                    {
                        "status": result.status.value,
                        "output": result.data,
                        "error": result.error_message,
                        "execution_time_ms": exec_time_ms,
                    }
                )

        except asyncio.TimeoutError:
            return error_response(
                f"Skill invocation timed out after {timeout}s",
                408,
                code="TIMEOUT",
            )
        except Exception as e:
            logger.exception(f"Skill invocation error for {skill_name}")
            return error_response(
                f"Skill invocation failed: {str(e)}",
                500,
                code="INVOCATION_ERROR",
            )
