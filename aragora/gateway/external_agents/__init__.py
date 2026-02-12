"""
External Agent Gateway - Secure orchestration of third-party AI agents.

Provides enterprise-grade security wrapper for external AI agents like
OpenClaw, OpenHands, and custom agents with:
- Container/VM isolation for all external agents
- Credential vault with runtime injection (never exposed to agents)
- PII/secret redaction on all outputs
- Full audit trail of external agent actions
- Policy-based routing (sensitive tasks → aragora agents only)

Quick Start:
    # Create a fully-configured gateway
    gateway = create_gateway()

    # Register adapters
    gateway.register_adapter(OpenClawExternalAdapter())

    # Execute tasks securely
    result = await gateway.execute(
        adapter_name="openclaw",
        task=ExternalAgentTask(prompt="Search for..."),
    )
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.gateway.external_agents.base import (
    BaseExternalAgentAdapter,
    ExternalAgentGateway,
    ExternalAgentResult,
    ExternalAgentTask,
    AgentCapability,
    GatewayConfig,
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

logger = logging.getLogger(__name__)


def create_gateway(
    config: GatewayConfig | None = None,
    enable_credential_vault: bool = True,
    enable_policy_engine: bool = True,
    audit_logger: Any | None = None,
) -> ExternalAgentGateway:
    """
    Create a fully-configured External Agent Gateway.

    This factory wires up all the security components:
    - CredentialVault for secure credential injection
    - PolicyEngine for capability-based access control
    - AuditBridge for compliance logging (if audit_logger provided)

    Args:
        config: Gateway configuration (uses defaults if not provided)
        enable_credential_vault: Enable credential vault integration
        enable_policy_engine: Enable policy-based access control
        audit_logger: Optional audit logger for compliance

    Returns:
        Fully configured ExternalAgentGateway

    Example:
        # Create with all security features
        gateway = create_gateway()

        # Create with custom config
        from aragora.gateway.external_agents import GatewayConfig, IsolationLevel
        gateway = create_gateway(
            config=GatewayConfig(
                default_isolation=IsolationLevel.CONTAINER,
                max_concurrent_agents=5,
            )
        )

        # Register adapters and execute
        gateway.register_adapter(OpenClawExternalAdapter())
        result = await gateway.execute(...)
    """
    from aragora.gateway.external_agents.base import GatewayConfig

    # Initialize config
    gateway_config = config or GatewayConfig()

    # Initialize credential vault if enabled
    credential_vault = None
    if enable_credential_vault and gateway_config.enable_credential_vault:
        try:
            from aragora.gateway.security.credential_vault import CredentialVault

            credential_vault = CredentialVault(audit_logger=audit_logger)
            logger.info("Credential vault initialized for external agent gateway")
        except ImportError:
            logger.warning(
                "CredentialVault not available - credentials will not be injected. "
                "Ensure aragora.gateway.security is installed."
            )

    # Initialize policy engine if enabled
    policy_engine = None
    if enable_policy_engine:
        policy_engine = PolicyEngine(audit_logger=audit_logger)
        logger.info("Policy engine initialized for external agent gateway")

    # Create and return the gateway
    gateway = ExternalAgentGateway(
        config=gateway_config,
        credential_vault=credential_vault,
        policy_engine=policy_engine,
        audit_bridge=audit_logger,
    )

    logger.info(
        f"External Agent Gateway created: "
        f"vault={'enabled' if credential_vault else 'disabled'}, "
        f"policy={'enabled' if policy_engine else 'disabled'}, "
        f"max_concurrent={gateway_config.max_concurrent_agents}"
    )

    return gateway


__all__ = [
    # Core
    "BaseExternalAgentAdapter",
    "ExternalAgentGateway",
    "ExternalAgentResult",
    "ExternalAgentTask",
    "AgentCapability",
    "GatewayConfig",
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
    # Factory
    "create_gateway",
]
