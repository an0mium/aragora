"""
External Agent Gateway - Base protocol and types.

Defines the core interfaces for integrating external AI agents securely.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class IsolationLevel(str, Enum):
    """Isolation level for external agent execution."""

    NONE = "none"  # No isolation (NOT recommended for untrusted agents)
    PROCESS = "process"  # Separate process with resource limits
    CONTAINER = "container"  # Docker/Podman container isolation
    VM = "vm"  # Full VM isolation (highest security)


class AgentCapability(str, Enum):
    """Capabilities that external agents may request."""

    # Information access
    WEB_SEARCH = "web_search"
    WEB_BROWSE = "web_browse"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    DATABASE_READ = "database_read"
    DATABASE_WRITE = "database_write"

    # Actions
    SEND_EMAIL = "send_email"
    SEND_MESSAGE = "send_message"
    MAKE_API_CALL = "make_api_call"
    EXECUTE_CODE = "execute_code"
    SCHEDULE_TASK = "schedule_task"

    # System
    SHELL_ACCESS = "shell_access"
    NETWORK_ACCESS = "network_access"
    CLIPBOARD_ACCESS = "clipboard_access"
    SCREEN_CAPTURE = "screen_capture"
    BROWSER_CONTROL = "browser_control"


@dataclass
class ExternalAgentTask:
    """Task to be executed by an external agent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "text_generation"
    prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[AgentCapability] = field(default_factory=list)
    timeout_seconds: float = 300.0
    max_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    # Security context
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None


@dataclass
class ExternalAgentResult:
    """Result from external agent execution."""

    task_id: str
    success: bool
    output: str = ""
    error: str | None = None

    # Execution metadata
    agent_name: str = ""
    agent_version: str = ""
    execution_time_ms: float = 0.0
    tokens_used: int = 0

    # Security metadata
    capabilities_used: list[AgentCapability] = field(default_factory=list)
    was_sandboxed: bool = True
    isolation_level: IsolationLevel = IsolationLevel.CONTAINER
    output_redacted: bool = False
    redaction_count: int = 0

    # Audit trail
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate execution duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return self.execution_time_ms / 1000.0


@runtime_checkable
class ExternalAgentAdapter(Protocol):
    """Protocol for external agent adapters.

    Each external agent (OpenClaw, OpenHands, etc.) implements this
    protocol to integrate with the gateway.
    """

    @property
    def agent_name(self) -> str:
        """Unique name of the agent."""
        ...

    @property
    def agent_version(self) -> str:
        """Version of the agent."""
        ...

    @property
    def supported_capabilities(self) -> list[AgentCapability]:
        """Capabilities this agent supports."""
        ...

    async def execute(
        self,
        task: ExternalAgentTask,
        credentials: dict[str, str],
        sandbox_config: dict[str, Any],
    ) -> ExternalAgentResult:
        """Execute a task using this agent.

        Args:
            task: The task to execute
            credentials: Credentials injected at runtime (never persisted)
            sandbox_config: Sandbox configuration for isolation

        Returns:
            Result of the execution
        """
        ...

    async def health_check(self) -> bool:
        """Check if the agent is healthy and available."""
        ...

    async def get_capabilities(self) -> dict[str, Any]:
        """Get detailed capability information."""
        ...


class BaseExternalAgentAdapter(ABC):
    """Base class for external agent adapters with common functionality."""

    def __init__(
        self,
        agent_name: str,
        agent_version: str,
        endpoint: str | None = None,
    ):
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._endpoint = endpoint
        self._supported_capabilities: list[AgentCapability] = []

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def agent_version(self) -> str:
        return self._agent_version

    @property
    def supported_capabilities(self) -> list[AgentCapability]:
        return self._supported_capabilities

    @abstractmethod
    async def execute(
        self,
        task: ExternalAgentTask,
        credentials: dict[str, str],
        sandbox_config: dict[str, Any],
    ) -> ExternalAgentResult:
        """Execute a task. Subclasses must implement."""
        ...

    async def health_check(self) -> bool:
        """Default health check - override for specific agents."""
        return True

    async def get_capabilities(self) -> dict[str, Any]:
        """Get capability details."""
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "supported_capabilities": [c.value for c in self.supported_capabilities],
        }


@dataclass
class GatewayConfig:
    """Configuration for the External Agent Gateway."""

    # Isolation
    default_isolation: IsolationLevel = IsolationLevel.CONTAINER
    container_image: str = "aragora/agent-sandbox:latest"
    container_memory_limit: str = "2g"
    container_cpu_limit: float = 1.0
    container_timeout_seconds: float = 600.0

    # Security
    enable_credential_vault: bool = True
    enable_output_redaction: bool = True
    enable_audit_logging: bool = True
    max_concurrent_agents: int = 10

    # Policy
    require_explicit_capability_approval: bool = True
    blocked_capabilities: list[AgentCapability] = field(
        default_factory=lambda: [
            AgentCapability.SHELL_ACCESS,
            AgentCapability.SCREEN_CAPTURE,
        ]
    )

    # Networking
    allow_external_network: bool = False
    allowed_domains: list[str] = field(default_factory=list)


class ExternalAgentGateway:
    """
    Enterprise-secure gateway for orchestrating external AI agents.

    This gateway provides:
    - Container/VM isolation for all external agents
    - Credential vault with runtime injection (never exposed to agents)
    - PII/secret redaction on all outputs
    - Full audit trail of external agent actions
    - Policy-based routing (sensitive tasks → aragora agents only)

    Usage:
        gateway = ExternalAgentGateway(config=GatewayConfig())

        # Register adapters
        gateway.register_adapter(OpenClawAdapter())
        gateway.register_adapter(OpenHandsAdapter())

        # Execute task
        result = await gateway.execute(
            adapter_name="openclaw",
            task=ExternalAgentTask(prompt="Search for news..."),
            tenant_id="acme-corp",
            credentials={"OPENAI_API_KEY": "..."},
        )
    """

    def __init__(
        self,
        config: GatewayConfig | None = None,
        credential_vault: Any | None = None,
        output_filter: Any | None = None,
        audit_bridge: Any | None = None,
        policy_engine: Any | None = None,
    ):
        self.config = config or GatewayConfig()
        self._adapters: dict[str, ExternalAgentAdapter] = {}
        self._credential_vault = credential_vault
        self._output_filter = output_filter
        self._audit_bridge = audit_bridge
        self._policy_engine = policy_engine
        self._execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        self._active_executions: dict[str, ExternalAgentTask] = {}

    def register_adapter(self, adapter: ExternalAgentAdapter) -> None:
        """Register an external agent adapter."""
        self._adapters[adapter.agent_name] = adapter
        logger.info("Registered external agent adapter: %s", adapter.agent_name)

    def get_adapter(self, name: str) -> ExternalAgentAdapter | None:
        """Get a registered adapter by name."""
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    async def execute(
        self,
        adapter_name: str,
        task: ExternalAgentTask,
        credentials: dict[str, str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> ExternalAgentResult:
        """
        Execute a task using a registered external agent.

        Args:
            adapter_name: Name of the adapter to use
            task: Task to execute
            credentials: Credentials to inject (retrieved from vault if not provided)
            tenant_id: Tenant context for multi-tenancy
            user_id: User context for audit

        Returns:
            ExternalAgentResult with execution details
        """
        start_time = time.monotonic()

        # Get adapter
        adapter = self._adapters.get(adapter_name)
        if not adapter:
            return ExternalAgentResult(
                task_id=task.task_id,
                success=False,
                error=f"Unknown adapter: {adapter_name}",
                agent_name=adapter_name,
            )

        # Apply tenant/user context
        task.tenant_id = tenant_id or task.tenant_id
        task.user_id = user_id or task.user_id

        # Check policy
        if self._policy_engine:
            policy_result = await self._check_policy(adapter, task)
            if not policy_result.allowed:
                return ExternalAgentResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Policy violation: {policy_result.reason}",
                    agent_name=adapter.agent_name,
                    agent_version=adapter.agent_version,
                )

        # Get credentials from vault if not provided
        if credentials is None and self._credential_vault:
            credentials = await self._get_credentials(adapter, task)
        credentials = credentials or {}

        # Prepare sandbox config
        sandbox_config = self._build_sandbox_config(adapter, task)

        # Execute with concurrency limit
        async with self._execution_semaphore:
            self._active_executions[task.task_id] = task
            try:
                # Log audit start
                if self._audit_bridge:
                    await self._audit_bridge.log_execution_start(
                        adapter_name=adapter.agent_name,
                        task=task,
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )

                # Execute task
                result = await adapter.execute(task, credentials, sandbox_config)

                # Apply output redaction
                if self.config.enable_output_redaction and self._output_filter:
                    result = await self._redact_output(result)

                # Update timing
                result.execution_time_ms = (time.monotonic() - start_time) * 1000
                result.completed_at = datetime.now(timezone.utc)

                # Log audit completion
                if self._audit_bridge:
                    await self._audit_bridge.log_execution_complete(
                        result=result,
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )

                return result

            except asyncio.TimeoutError:
                return ExternalAgentResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Execution timed out after {task.timeout_seconds}s",
                    agent_name=adapter.agent_name,
                    agent_version=adapter.agent_version,
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                logger.exception("External agent execution failed: %s", e)
                return ExternalAgentResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    agent_name=adapter.agent_name,
                    agent_version=adapter.agent_version,
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            finally:
                self._active_executions.pop(task.task_id, None)

    async def _check_policy(self, adapter: ExternalAgentAdapter, task: ExternalAgentTask) -> Any:
        """Check policy for task execution.

        Uses the injected PolicyEngine if available, otherwise falls back
        to basic capability blocking based on config.
        """
        from aragora.gateway.external_agents.policy import PolicyDecision, PolicyAction

        # Use policy engine if available
        if self._policy_engine:
            return await self._policy_engine.evaluate(
                adapter=adapter,
                task=task,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
            )

        # Fallback: basic capability blocking
        for cap in task.required_capabilities:
            if cap in self.config.blocked_capabilities:
                return PolicyDecision(
                    allowed=False,
                    action=PolicyAction.DENY,
                    reason=f"Capability {cap.value} is blocked by gateway config",
                )

        return PolicyDecision(
            allowed=True,
            action=PolicyAction.ALLOW,
            reason="Basic policy check passed (no policy engine configured)",
        )

    async def _get_credentials(
        self, adapter: ExternalAgentAdapter, task: ExternalAgentTask
    ) -> dict[str, str]:
        """Get credentials from vault for agent execution.

        Uses the injected CredentialVault if available. Credentials are
        scoped by tenant and agent for security.
        """
        # Use credential vault if available
        if self._credential_vault:
            return await self._credential_vault.get_credentials_for_execution(
                agent_name=adapter.agent_name,
                tenant_id=task.tenant_id,
                required_credentials=task.metadata.get("required_credentials"),
            )

        # Fallback: no credentials available without vault
        logger.warning(
            "No credential vault configured - agent %s will execute without injected credentials",
            adapter.agent_name,
        )
        return {}

    def _build_sandbox_config(
        self, adapter: ExternalAgentAdapter, task: ExternalAgentTask
    ) -> dict[str, Any]:
        """Build sandbox configuration for agent execution."""
        return {
            "isolation_level": self.config.default_isolation.value,
            "container_image": self.config.container_image,
            "memory_limit": self.config.container_memory_limit,
            "cpu_limit": self.config.container_cpu_limit,
            "timeout_seconds": min(task.timeout_seconds, self.config.container_timeout_seconds),
            "allow_network": self.config.allow_external_network,
            "allowed_domains": self.config.allowed_domains,
            "capabilities": [c.value for c in task.required_capabilities],
        }

    async def _redact_output(self, result: ExternalAgentResult) -> ExternalAgentResult:
        """Redact sensitive content from output."""
        if not self._output_filter:
            return result

        redacted_output, count = await self._output_filter.redact(result.output)
        result.output = redacted_output
        result.output_redacted = count > 0
        result.redaction_count = count
        return result

    async def health_check(self) -> dict[str, Any]:
        """Check health of all registered adapters."""
        results: dict[str, dict[str, bool | str | None]] = {}
        for name, adapter in self._adapters.items():
            try:
                healthy = await adapter.health_check()
                results[name] = {"healthy": healthy, "error": None}
            except (OSError, ConnectionError, RuntimeError):
                results[name] = {"healthy": False, "error": "Health check failed"}
        return {
            "gateway_healthy": all(r["healthy"] for r in results.values()),
            "adapters": results,
            "active_executions": len(self._active_executions),
            "max_concurrent": self.config.max_concurrent_agents,
        }
