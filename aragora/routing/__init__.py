"""
Agent routing and selection.

Provides adaptive agent selection for optimal team composition,
including domain detection and auto-routing.
"""

from aragora.routing.selection import (
    AgentSelector,
    AgentProfile,
    TaskRequirements,
    TeamComposition,
    DomainDetector,
    DOMAIN_KEYWORDS,
    DEFAULT_AGENT_EXPERTISE,
)

__all__ = [
    "AgentSelector",
    "AgentProfile",
    "TaskRequirements",
    "TeamComposition",
    "DomainDetector",
    "DOMAIN_KEYWORDS",
    "DEFAULT_AGENT_EXPERTISE",
]
