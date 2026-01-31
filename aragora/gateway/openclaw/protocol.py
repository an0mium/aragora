"""
OpenClaw Protocol Translator.

Translates between Aragora request/response formats and OpenClaw task formats.
Handles context injection, tenant isolation, and result normalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .sandbox import OpenClawTask

logger = logging.getLogger(__name__)


@dataclass
class AragoraRequest:
    """Aragora-format request for OpenClaw gateway."""

    content: str
    request_type: str = "task"  # task, query, action
    capabilities: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high
    timeout_seconds: int = 300


@dataclass
class AragoraResponse:
    """Aragora-format response from OpenClaw gateway."""

    request_id: str
    status: str  # pending, running, completed, failed
    result: Any | None = None
    error: str | None = None
    execution_time_ms: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantContext:
    """Tenant context for multi-tenancy isolation."""

    tenant_id: str
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    enabled_capabilities: set[str] = field(default_factory=set)
    enabled_plugins: set[str] = field(default_factory=set)
    quotas: dict[str, int] = field(default_factory=dict)


@dataclass
class AuthorizationContext:
    """Authorization context from RBAC."""

    actor_id: str
    actor_type: str = "user"  # user, service, agent
    permissions: set[str] = field(default_factory=set)
    roles: list[str] = field(default_factory=list)
    session_id: str | None = None


class OpenClawProtocolTranslator:
    """
    Translate between Aragora and OpenClaw message formats.

    Handles:
    - Request conversion (Aragora → OpenClaw)
    - Response conversion (OpenClaw → Aragora)
    - Context injection (tenant, auth, tracing)
    - Metadata normalization
    """

    def __init__(
        self,
        default_timeout: int = 300,
        include_debug_info: bool = False,
    ) -> None:
        """
        Initialize protocol translator.

        Args:
            default_timeout: Default timeout for tasks
            include_debug_info: Whether to include debug info in responses
        """
        self.default_timeout = default_timeout
        self.include_debug_info = include_debug_info

    def aragora_to_openclaw(
        self,
        request: AragoraRequest,
        auth_context: AuthorizationContext | None = None,
        tenant_context: TenantContext | None = None,
    ) -> OpenClawTask:
        """
        Convert Aragora request to OpenClaw task.

        Args:
            request: Aragora-format request
            auth_context: Authorization context
            tenant_context: Tenant isolation context

        Returns:
            OpenClawTask ready for sandbox execution
        """
        task_id = str(uuid4())

        # Determine task type from request type and capabilities
        task_type = self._infer_task_type(request)

        # Build payload
        payload: dict[str, Any] = {
            "content": request.content,
            "priority": request.priority,
        }

        # Merge request context into payload
        if request.context:
            payload["context"] = request.context

        # Filter capabilities based on tenant enablement
        capabilities = request.capabilities.copy()
        if tenant_context and tenant_context.enabled_capabilities:
            # Only include capabilities enabled for tenant
            capabilities = [
                cap for cap in capabilities if cap in tenant_context.enabled_capabilities
            ]

        # Filter plugins based on tenant enablement
        plugins = request.plugins.copy()
        if tenant_context and tenant_context.enabled_plugins:
            plugins = [plugin for plugin in plugins if plugin in tenant_context.enabled_plugins]

        # Build metadata with tracing info
        metadata: dict[str, Any] = {
            "source": "aragora_gateway",
            "translated_at": datetime.now(timezone.utc).isoformat(),
            **request.metadata,
        }

        if auth_context:
            metadata["actor_id"] = auth_context.actor_id
            metadata["actor_type"] = auth_context.actor_type
            if auth_context.session_id:
                metadata["session_id"] = auth_context.session_id

        if tenant_context:
            metadata["tenant_id"] = tenant_context.tenant_id
            if tenant_context.organization_id:
                metadata["organization_id"] = tenant_context.organization_id
            if tenant_context.workspace_id:
                metadata["workspace_id"] = tenant_context.workspace_id

        return OpenClawTask(
            id=task_id,
            type=task_type,
            payload=payload,
            capabilities=capabilities,
            plugins=plugins,
            metadata=metadata,
        )

    def openclaw_to_aragora(
        self,
        task: OpenClawTask,
        openclaw_result: dict[str, Any],
    ) -> AragoraResponse:
        """
        Convert OpenClaw result to Aragora response.

        Args:
            task: Original OpenClaw task
            openclaw_result: Result from OpenClaw runtime

        Returns:
            AragoraResponse in standard format
        """
        status = openclaw_result.get("status", "completed")
        if status == "success":
            status = "completed"

        # Extract result or error
        result = openclaw_result.get("result")
        error = openclaw_result.get("error")

        # Build metadata
        metadata: dict[str, Any] = {
            "task_id": task.id,
            "task_type": task.type,
        }

        if self.include_debug_info:
            metadata["capabilities_used"] = task.capabilities
            metadata["plugins_used"] = task.plugins
            metadata["openclaw_raw"] = openclaw_result

        # Parse timestamps
        completed_at = None
        if openclaw_result.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(
                    openclaw_result["completed_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                completed_at = datetime.now(timezone.utc)
        elif status == "completed":
            completed_at = datetime.now(timezone.utc)

        return AragoraResponse(
            request_id=task.id,
            status=status,
            result=result,
            error=error,
            execution_time_ms=openclaw_result.get("execution_time_ms", 0),
            completed_at=completed_at,
            metadata=metadata,
        )

    def wrap_with_context(
        self,
        task: OpenClawTask,
        auth_context: AuthorizationContext,
        tenant_context: TenantContext,
    ) -> OpenClawTask:
        """
        Inject enterprise context into OpenClaw task.

        Args:
            task: Existing OpenClaw task
            auth_context: Authorization context
            tenant_context: Tenant isolation context

        Returns:
            Task with injected context
        """
        # Update metadata with context
        task.metadata.update(
            {
                "actor_id": auth_context.actor_id,
                "actor_type": auth_context.actor_type,
                "tenant_id": tenant_context.tenant_id,
            }
        )

        if tenant_context.organization_id:
            task.metadata["organization_id"] = tenant_context.organization_id
        if tenant_context.workspace_id:
            task.metadata["workspace_id"] = tenant_context.workspace_id
        if auth_context.session_id:
            task.metadata["session_id"] = auth_context.session_id

        return task

    def _infer_task_type(self, request: AragoraRequest) -> str:
        """Infer OpenClaw task type from request."""
        # Map request types to OpenClaw task types
        type_mapping = {
            "task": "general",
            "query": "search",
            "action": "execute",
            "chat": "conversation",
            "code": "code_generation",
            "file": "file_operation",
        }

        base_type = type_mapping.get(request.request_type, "general")

        # Refine based on capabilities
        if "code_execution" in request.capabilities:
            return "code_execution"
        if "file_system_write" in request.capabilities:
            return "file_operation"
        if "browser_automation" in request.capabilities:
            return "browser_automation"
        if "email_send" in request.capabilities:
            return "email"

        return base_type

    def extract_capabilities_from_content(self, content: str) -> list[str]:
        """
        Infer required capabilities from request content.

        This is a heuristic method that can be extended with ML-based
        capability detection.
        """
        capabilities = []

        # Simple keyword-based detection
        content_lower = content.lower()

        if any(kw in content_lower for kw in ["write file", "save to", "create file"]):
            capabilities.append("file_system_write")
        if any(kw in content_lower for kw in ["read file", "open file", "load"]):
            capabilities.append("file_system_read")
        if any(kw in content_lower for kw in ["browse", "web", "fetch url", "http"]):
            capabilities.append("network_external")
        if any(kw in content_lower for kw in ["run code", "execute", "shell"]):
            capabilities.append("code_execution")
        if any(kw in content_lower for kw in ["send email", "email to"]):
            capabilities.append("email_send")
        if any(kw in content_lower for kw in ["calendar", "schedule", "meeting"]):
            capabilities.append("calendar_read")

        return capabilities


__all__ = [
    "AragoraRequest",
    "AragoraResponse",
    "TenantContext",
    "AuthorizationContext",
    "OpenClawProtocolTranslator",
]
