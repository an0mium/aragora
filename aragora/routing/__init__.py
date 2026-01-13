"""
Agent routing and selection.

Provides adaptive agent selection for optimal team composition,
including domain detection and auto-routing.
"""

from aragora.routing.selection import (
    DEFAULT_AGENT_EXPERTISE,
    DOMAIN_KEYWORDS,
    AgentProfile,
    AgentSelector,
    DomainDetector,
    TaskRequirements,
    TeamComposition,
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
