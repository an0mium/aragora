"""
Sandbox Isolation Module (Clawdbot-inspired).

Provides safe code execution environments for agent evaluation:
- Docker-based isolation
- Tool policy enforcement
- Resource limits
- Execution timeouts
"""

from aragora.sandbox.executor import (
    ExecutionResult,
    SandboxConfig,
    SandboxExecutor,
)
from aragora.sandbox.policies import (
    ToolPolicy,
    ToolPolicyChecker,
    create_default_policy,
)

__all__ = [
    "ExecutionResult",
    "SandboxConfig",
    "SandboxExecutor",
    "ToolPolicy",
    "ToolPolicyChecker",
    "create_default_policy",
]
