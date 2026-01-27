"""
Sandbox Isolation Module (Clawdbot-inspired).

Provides safe code execution environments for agent evaluation:
- Docker-based isolation
- Tool policy enforcement
- Resource limits
- Execution timeouts
- Container pool management
- Session lifecycle management
"""

from aragora.sandbox.executor import (
    ExecutionResult,
    SandboxConfig,
    SandboxExecutor,
)
from aragora.sandbox.lifecycle import (
    ContainerSession,
    SessionConfig,
    SessionContainerManager,
    SessionExecutionResult,
    get_session_manager,
)
from aragora.sandbox.policies import (
    ToolPolicy,
    ToolPolicyChecker,
    create_default_policy,
)
from aragora.sandbox.pool import (
    ContainerPool,
    ContainerPoolConfig,
    PooledContainer,
    get_container_pool,
)

__all__ = [
    # Executor
    "ExecutionResult",
    "SandboxConfig",
    "SandboxExecutor",
    # Policies
    "ToolPolicy",
    "ToolPolicyChecker",
    "create_default_policy",
    # Pool
    "ContainerPool",
    "ContainerPoolConfig",
    "PooledContainer",
    "get_container_pool",
    # Lifecycle
    "ContainerSession",
    "SessionConfig",
    "SessionContainerManager",
    "SessionExecutionResult",
    "get_session_manager",
]
