"""
Federation Gateway - External agent framework integration.

This module provides registration, discovery, and lifecycle management for
external AI agent frameworks (OpenClaw, AutoGPT, CrewAI, LangGraph, etc.)
that integrate with Aragora's gateway.

Key Components:
- FederationRegistry: Central registry for external framework management
- ExternalFramework: Dataclass representing a registered framework
- FrameworkCapability: Capability descriptor for routing and discovery
- RegistrationResult: Result of framework registration

Usage:
    from aragora.gateway.federation import FederationRegistry, FrameworkCapability

    registry = FederationRegistry()
    await registry.connect()

    # Register an external framework
    result = await registry.register(
        name="autogpt",
        version="0.5.0",
        capabilities=[
            FrameworkCapability(
                name="autonomous_task",
                description="Execute autonomous multi-step tasks",
                parameters={"task": "str", "max_steps": "int"},
                returns="TaskResult",
            ),
        ],
        endpoints={"base": "http://localhost:8090", "health": "/health"},
    )

    # Find frameworks by capability
    frameworks = await registry.find_by_capability("autonomous_task")

    await registry.close()
"""

from aragora.gateway.federation.registry import (
    FederationRegistry,
    ExternalFramework,
    FrameworkCapability,
    RegistrationResult,
    HealthStatus,
    FrameworkStatus,
    LifecycleHook,
)

__all__ = [
    # Core registry
    "FederationRegistry",
    # Dataclasses
    "ExternalFramework",
    "FrameworkCapability",
    "RegistrationResult",
    # Enums
    "HealthStatus",
    "FrameworkStatus",
    # Type aliases
    "LifecycleHook",
]
