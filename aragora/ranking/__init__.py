"""
Ranking and reputation system.

Provides ELO-based skill tracking for agents, calibration scoring,
relationship tracking, and red team integration.
"""

from aragora.ranking.elo import EloSystem, AgentRating, MatchResult
from aragora.ranking.calibration_engine import (
    CalibrationEngine,
    DomainCalibrationEngine,
    CalibrationPrediction,
    BucketStats,
)
from aragora.ranking.relationships import (
    RelationshipTracker,
    RelationshipStats,
    RelationshipMetrics,
)
from aragora.ranking.redteam import (
    RedTeamIntegrator,
    RedTeamResult,
    VulnerabilitySummary,
)
from aragora.ranking.tournaments import (
    TournamentManager,
    Tournament,
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
