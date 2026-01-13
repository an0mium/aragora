"""
Policy Engine for Aragora.

This module provides per-tool and per-task policy enforcement,
enabling fine-grained control over what agents can do and when
human approval is required.

The policy engine is the foundation for enterprise trust - it ensures
that agent actions are bounded, auditable, and reversible.

Usage:
    from aragora.policy import PolicyEngine, Policy, Action

    engine = PolicyEngine()

    # Define a tool with its capabilities and risk level
    engine.register_tool(Tool(
        name="code_writer",
        capabilities=["write_file", "create_file", "delete_file"],
        risk_level=RiskLevel.HIGH,
        requires_human_approval=True,
    ))

    # Check if an action is allowed
    result = engine.check_action(
        agent="claude",
        tool="code_writer",
        action="write_file",
        context={"file": "aragora/core.py"},
    )

    if not result.allowed:
        if result.requires_human_approval:
            await request_human_approval(result)
        else:
            raise PolicyViolation(result.reason)
"""

from aragora.policy.engine import (
    Policy,
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
)
from aragora.policy.risk import (
    BlastRadius,
    RiskBudget,
    RiskLevel,
)
from aragora.policy.tools import (
    Tool,
    ToolCapability,
    ToolRegistry,
)

__all__ = [
    "PolicyEngine",
    "Policy",
    "PolicyResult",
    "PolicyViolation",
    "Tool",
    "ToolCapability",
    "ToolRegistry",
    "RiskLevel",
    "BlastRadius",
    "RiskBudget",
]
