"""
Autonomous DevOps Agent.

Handles repository operations (PR review, issue triage, releases)
through OpenClaw policy-controlled execution. Dogfood for Aragora —
the agent uses Aragora's own tools to manage Aragora's own repo.

Usage:
    aragora agent run devops --repo an0mium/aragora --task review-prs
    aragora agent run devops --repo an0mium/aragora --mode watch
"""

from aragora.agents.devops.agent import (
    DevOpsAgent,
    DevOpsAgentConfig,
    DevOpsTask,
    TaskResult,
)

__all__ = [
    "DevOpsAgent",
    "DevOpsAgentConfig",
    "DevOpsTask",
    "TaskResult",
]
