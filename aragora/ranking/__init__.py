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
    AgentRelationship,
    RelationshipMetrics,
    RelationshipStats,
    RelationshipTracker,
)
from aragora.ranking.tournaments import (
    Tournament,
    TournamentEvent,
    TournamentHistoryEntry,
    TournamentManager,
    TournamentMatch,
    TournamentStanding,
    TournamentStatus,
)
from aragora.ranking.km_elo_bridge import (
    KMEloBridge,
    KMEloBridgeConfig,
    KMEloBridgeSyncResult,
    create_km_elo_bridge,
)
from aragora.ranking.pattern_matcher import (
    PatternAffinity,
    TaskPatternMatcher,
    classify_task,
    get_pattern_matcher,
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
    "AgentRelationship",
    # Red Team
    "RedTeamIntegrator",
    "RedTeamResult",
    "VulnerabilitySummary",
    # Tournaments
    "TournamentManager",
    "Tournament",
    "TournamentEvent",
    "TournamentHistoryEntry",
    "TournamentMatch",
    "TournamentStanding",
    "TournamentStatus",
    # Knowledge Mound Bridge
    "KMEloBridge",
    "KMEloBridgeConfig",
    "KMEloBridgeSyncResult",
    "create_km_elo_bridge",
    # Pattern Matching
    "TaskPatternMatcher",
    "PatternAffinity",
    "get_pattern_matcher",
    "classify_task",
]
