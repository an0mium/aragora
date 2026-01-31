"""
External Agent Gateway - Secure orchestration of third-party AI agents.

Provides enterprise-grade security wrapper for external AI agents like
OpenClaw, OpenHands, and custom agents with:
- Container/VM isolation for all external agents
- Credential vault with runtime injection (never exposed to agents)
- PII/secret redaction on all outputs
- Full audit trail of external agent actions
- Policy-based routing (sensitive tasks → aragora agents only)
"""

from aragora.gateway.external_agents.base import (
    BaseExternalAgentAdapter,
    ExternalAgentGateway,
    ExternalAgentResult,
    ExternalAgentTask,
    AgentCapability,
    IsolationLevel,
)
from aragora.gateway.external_agents.policy import (
    PolicyEngine,
    PolicyDecision,
    PolicyAction,
    CapabilityRule,
    SensitivityLevel,
)
from aragora.gateway.external_agents.openclaw_adapter import (
    OpenClawExternalAdapter,
)
from aragora.gateway.external_agents.sandbox import (
    SandboxConfig,
    SandboxExecution,
    SandboxManager,
    SandboxState,
    ProcessSandbox,
    DockerSandbox,
)

__all__ = [
    # Core
    "BaseExternalAgentAdapter",
    "ExternalAgentGateway",
    "ExternalAgentResult",
    "ExternalAgentTask",
    "AgentCapability",
    "IsolationLevel",
    # Policy
    "PolicyEngine",
    "PolicyDecision",
    "PolicyAction",
    "CapabilityRule",
    "SensitivityLevel",
    # Adapters
    "OpenClawExternalAdapter",
    # Sandbox
    "SandboxConfig",
    "SandboxExecution",
    "SandboxManager",
    "SandboxState",
    "ProcessSandbox",
    "DockerSandbox",
]
