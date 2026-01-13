"""
Ranking and reputation system.

Provides ELO-based skill tracking for agents, calibration scoring,
relationship tracking, and red team integration.
"""

from aragora.ranking.calibration_engine import (
    BucketStats,
    CalibrationEngine,
    CalibrationPrediction,
    DomainCalibrationEngine,
)
from aragora.ranking.elo import AgentRating, EloSystem, MatchResult
from aragora.ranking.redteam import (
    RedTeamIntegrator,
    RedTeamResult,
    VulnerabilitySummary,
)
from aragora.ranking.relationships import (
    RelationshipMetrics,
    RelationshipStats,
    RelationshipTracker,
)
from aragora.ranking.tournaments import (
    Tournament,
    TournamentManager,
    TournamentMatch,
    TournamentStanding,
)

__all__ = [
    # Core ELO
    "EloSystem",
    "AgentRating",
    "MatchResult",
    # Calibration
    "CalibrationEngine",
    "DomainCalibrationEngine",
    "CalibrationPrediction",
    "BucketStats",
    # Relationships
    "RelationshipTracker",
    "RelationshipStats",
    "RelationshipMetrics",
    # Red Team
    "RedTeamIntegrator",
    "RedTeamResult",
    "VulnerabilitySummary",
    # Tournaments
    "TournamentManager",
    "Tournament",
    "TournamentMatch",
    "TournamentStanding",
]
