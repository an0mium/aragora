"""External agent framework integration for Aragora.

Provides adapters for integrating external AI agent frameworks
(OpenHands, AutoGPT, CrewAI, etc.) with Aragora's enterprise
security, governance, and observability layer.

Example:
    from aragora.agents.external import (
        ExternalAgentRegistry,
        TaskRequest,
        ToolPermission,
    )

    # Create an adapter
    adapter = ExternalAgentRegistry.create("openhands")

    # Submit a task
    request = TaskRequest(
        task_type="code",
        prompt="Create a Python function that...",
        tool_permissions={ToolPermission.FILE_WRITE, ToolPermission.SHELL_EXECUTE},
    )
    task_id = await adapter.submit_task(request)

    # Get result
    result = await adapter.get_task_result(task_id)
"""

from aragora.agents.external.base import (
    EventCallback,
    ExternalAgentAdapter,
    ExternalAgentError,
    TaskNotCompleteError,
    TaskNotFoundError,
)
from aragora.agents.external.config import (
    ApprovalMode,
    AutoGPTConfig,
    CircuitBreakerConfig,
    CrewAIConfig,
    ExternalAgentConfig,
    OpenHandsConfig,
    ToolConfig,
    WorkspaceConfig,
    WorkspaceType,
    get_config_for_adapter,
)
from aragora.agents.external.models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
    ToolInvocation,
    ToolPermission,
)
from aragora.agents.external.proxy import (
    ExternalAgentProxy,
    PolicyDeniedError,
    ProxyConfig,
)
from aragora.agents.external.registry import (
    ExternalAdapterSpec,
    ExternalAgentRegistry,
    register_all_adapters,
)
from aragora.agents.external.security import (
    ExternalAgentSecurityPolicy,
    PolicyCheckResult,
    ToolPermissionGate,
)

__all__ = [
    # Base classes and errors
    "ExternalAgentAdapter",
    "ExternalAgentError",
    "TaskNotFoundError",
    "TaskNotCompleteError",
    "EventCallback",
    # Models
    "TaskStatus",
    "TaskRequest",
    "TaskResult",
    "TaskProgress",
    "HealthStatus",
    "ToolPermission",
    "ToolInvocation",
    # Configuration
    "ExternalAgentConfig",
    "OpenHandsConfig",
    "AutoGPTConfig",
    "CrewAIConfig",
    "CircuitBreakerConfig",
    "WorkspaceConfig",
    "ToolConfig",
    "WorkspaceType",
    "ApprovalMode",
    "get_config_for_adapter",
    # Registry
    "ExternalAgentRegistry",
    "ExternalAdapterSpec",
    "register_all_adapters",
    # Security
    "ExternalAgentSecurityPolicy",
    "PolicyCheckResult",
    "ToolPermissionGate",
    # Proxy
    "ExternalAgentProxy",
    "ProxyConfig",
    "PolicyDeniedError",
]
