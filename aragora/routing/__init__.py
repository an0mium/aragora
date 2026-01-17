"""
Agent routing and selection.

Provides adaptive agent selection for optimal team composition,
including domain detection and auto-routing.
"""

from aragora.routing.selection import (
    DEFAULT_AGENT_EXPERTISE,
    DOMAIN_KEYWORDS,
    PHASE_ROLES,
    AgentProfile,
    AgentSelector,
    DomainDetector,
    TaskRequirements,
    TeamBuilder,
    TeamComposition,
)

__all__ = [
    "AgentSelector",
    "AgentProfile",
    "TaskRequirements",
    "TeamComposition",
    "TeamBuilder",
    "DomainDetector",
    "DOMAIN_KEYWORDS",
    "PHASE_ROLES",
    "DEFAULT_AGENT_EXPERTISE",
]
