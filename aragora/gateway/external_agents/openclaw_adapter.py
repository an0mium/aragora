"""
OpenClaw External Agent Adapter.

Wraps the existing OpenClawGatewayAdapter to conform to the
BaseExternalAgentAdapter interface, enabling integration with
the TaskRouter and FallbackChain orchestration components.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.gateway.external_agents.base import (
    AgentCapability,
    BaseExternalAgentAdapter,
    ExternalAgentResult,
    ExternalAgentTask,
    IsolationLevel,
)
from aragora.gateway.openclaw.adapter import (
    OpenClawGatewayAdapter,
    GatewayResult,
)
from aragora.gateway.openclaw.protocol import (
    AragoraRequest,
    AuthorizationContext,
    TenantContext,
)

logger = logging.getLogger(__name__)


# Mapping from AgentCapability to OpenClaw capability strings
CAPABILITY_MAPPING: dict[AgentCapability, str] = {
    AgentCapability.WEB_SEARCH: "web_search",
    AgentCapability.WEB_BROWSE: "web_browse",
    AgentCapability.FILE_READ: "file_read",
    AgentCapability.FILE_WRITE: "file_write",
    AgentCapability.DATABASE_READ: "database_read",
    AgentCapability.DATABASE_WRITE: "database_write",
    AgentCapability.SEND_EMAIL: "send_email",
    AgentCapability.SEND_MESSAGE: "send_message",
    AgentCapability.MAKE_API_CALL: "api_call",
    AgentCapability.EXECUTE_CODE: "code_execution",
    AgentCapability.SCHEDULE_TASK: "scheduler",
    AgentCapability.SHELL_ACCESS: "shell",
    AgentCapability.NETWORK_ACCESS: "network",
    AgentCapability.CLIPBOARD_ACCESS: "clipboard",
    AgentCapability.SCREEN_CAPTURE: "screen_capture",
    AgentCapability.BROWSER_CONTROL: "browser",
}

# Reverse mapping for converting OpenClaw capabilities to AgentCapability
REVERSE_CAPABILITY_MAPPING: dict[str, AgentCapability] = {
    v: k for k, v in CAPABILITY_MAPPING.items()
}


class OpenClawExternalAdapter(BaseExternalAgentAdapter):
    """
    External agent adapter for OpenClaw integration.

    Wraps OpenClawGatewayAdapter to provide:
    - Unified interface for TaskRouter and FallbackChain
    - Automatic capability translation
    - Isolation level enforcement
    - Execution metrics collection

    Usage:
        adapter = OpenClawExternalAdapter(
            openclaw_endpoint="http://localhost:8081",
            isolation_level=IsolationLevel.CONTAINER,
        )

        result = await adapter.execute(task)
    """

    def __init__(
        self,
        openclaw_endpoint: str = "http://localhost:8081",
        isolation_level: IsolationLevel = IsolationLevel.CONTAINER,
        gateway_adapter: OpenClawGatewayAdapter | None = None,
        default_timeout_seconds: int = 300,
    ) -> None:
        """
        Initialize OpenClaw external adapter.

        Args:
            openclaw_endpoint: OpenClaw runtime endpoint
            isolation_level: Default isolation level for executions
            gateway_adapter: Pre-configured gateway adapter (optional)
            default_timeout_seconds: Default timeout for executions
        """
        self._endpoint = openclaw_endpoint
        self._isolation_level = isolation_level
        self._default_timeout = default_timeout_seconds

        # Use provided adapter or create new one
        self._gateway = gateway_adapter or OpenClawGatewayAdapter(
            openclaw_endpoint=openclaw_endpoint,
        )

        # Track available capabilities
        self._capabilities: set[AgentCapability] = self._detect_capabilities()

    @property
    def name(self) -> str:
        """Agent name for identification."""
        return "openclaw"

    @property
    def version(self) -> str:
        """Agent version string."""
        return "1.0.0"

    @property
    def capabilities(self) -> set[AgentCapability]:
        """Set of capabilities this agent provides."""
        return self._capabilities

    @property
    def isolation_level(self) -> IsolationLevel:
        """Default isolation level for this agent."""
        return self._isolation_level

    def _detect_capabilities(self) -> set[AgentCapability]:
        """Detect available capabilities from the OpenClaw gateway."""
        # OpenClaw supports most capabilities
        return {
            AgentCapability.WEB_SEARCH,
            AgentCapability.WEB_BROWSE,
            AgentCapability.BROWSER_CONTROL,
            AgentCapability.EXECUTE_CODE,
            AgentCapability.FILE_READ,
            AgentCapability.FILE_WRITE,
            AgentCapability.SHELL_ACCESS,
            AgentCapability.NETWORK_ACCESS,
            AgentCapability.MAKE_API_CALL,
            AgentCapability.SCREEN_CAPTURE,
            AgentCapability.SCHEDULE_TASK,
            AgentCapability.DATABASE_READ,
            AgentCapability.DATABASE_WRITE,
            AgentCapability.SEND_MESSAGE,
        }

    def _translate_capabilities(
        self,
        capabilities: list[AgentCapability],
    ) -> list[str]:
        """Translate AgentCapability to OpenClaw capability strings."""
        return [CAPABILITY_MAPPING.get(cap, cap.value) for cap in capabilities]

    async def execute(  # type: ignore[override]
        self,
        task: ExternalAgentTask,
    ) -> ExternalAgentResult:
        """
        Execute a task via OpenClaw.

        Args:
            task: The task to execute

        Returns:
            ExternalAgentResult with execution details
        """
        import time

        start_time = time.time()

        try:
            # Create Aragora request from task
            request = AragoraRequest(
                content=task.prompt,
                capabilities=self._translate_capabilities(task.required_capabilities),
                plugins=task.metadata.get("plugins", []) if task.metadata else [],
                request_type=task.task_type,
                metadata={"request_id": task.task_id},
            )

            # Create auth context
            auth_context = AuthorizationContext(
                actor_id=task.user_id or "anonymous",
                roles=task.metadata.get("roles", []) if task.metadata else [],
            )

            # Create tenant context if provided
            tenant_context = None
            if task.tenant_id:
                tenant_context = TenantContext(
                    tenant_id=task.tenant_id,
                    workspace_id=task.workspace_id,
                )

            # Execute via gateway
            result: GatewayResult = await self._gateway.execute_task(
                request=request,
                auth_context=auth_context,
                tenant_context=tenant_context,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Convert GatewayResult to ExternalAgentResult
            if result.success and result.response:
                return ExternalAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    agent_version=self.version,
                    success=True,
                    output=result.response.result,
                    error=None,
                    execution_time_ms=execution_time_ms,
                    tokens_used=result.metadata.get("tokens_used", 0),
                    capabilities_used=task.required_capabilities,
                    was_sandboxed=True,  # OpenClaw always sandboxed
                    isolation_level=self._isolation_level,
                    output_redacted=False,  # Redaction handled by output filter
                    redaction_count=0,
                )
            else:
                return ExternalAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    agent_version=self.version,
                    success=False,
                    output="",
                    error=result.error or result.blocked_reason or "Unknown error",
                    execution_time_ms=execution_time_ms,
                    capabilities_used=task.required_capabilities,
                    was_sandboxed=True,
                    isolation_level=self._isolation_level,
                )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"OpenClaw execution failed: {e}")

            return ExternalAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                agent_version=self.version,
                success=False,
                output="",
                error=str(e),
                execution_time_ms=execution_time_ms,
                capabilities_used=task.required_capabilities,
                was_sandboxed=True,
                isolation_level=self._isolation_level,
            )

    async def health_check(self) -> bool:
        """Check if OpenClaw is healthy and reachable."""
        try:
            # Simple connectivity check
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"OpenClaw health check failed: {e}")
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current adapter configuration."""
        return {
            "endpoint": self._endpoint,
            "isolation_level": self._isolation_level.value,
            "default_timeout": self._default_timeout,
            "capabilities": [c.value for c in self._capabilities],
        }


__all__ = [
    "OpenClawExternalAdapter",
    "CAPABILITY_MAPPING",
    "REVERSE_CAPABILITY_MAPPING",
]
