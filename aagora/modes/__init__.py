"""
Debate modes for different use cases.

Provides specialized debate protocols for:
- Adversarial red-teaming
- Code review
- Policy analysis
- Research synthesis
"""

from aagora.modes.redteam import (
    RedTeamMode,
    RedTeamProtocol,
    RedTeamResult,
    RedTeamRound,
    Attack,
    Defense,
    AttackType,
    redteam_code_review,
    redteam_policy,
)

__all__ = [
    "RedTeamMode",
    "RedTeamProtocol",
    "RedTeamResult",
    "RedTeamRound",
    "Attack",
    "Defense",
    "AttackType",
    "redteam_code_review",
    "redteam_policy",
]
