"""
Modular HTTP request handlers for the unified server.

Each module handles a specific domain of endpoints:
- debates: Debate history and management
- agents: Agent profiles, rankings, and metrics
- system: Health checks, nomic state, modes
- pulse: Trending topics from multiple sources
- analytics: Aggregated metrics and statistics
- consensus: Consensus memory and dissent tracking

Usage:
    from aragora.server.handlers import DebatesHandler, AgentsHandler, SystemHandler

    # Create handlers with server context
    ctx = {"storage": storage, "elo_system": elo, "nomic_dir": nomic_dir}
    debates = DebatesHandler(ctx)
    agents = AgentsHandler(ctx)
    system = SystemHandler(ctx)

    # Handle requests
    if debates.can_handle(path):
        result = debates.handle(path, query_params, handler)
"""

from __future__ import annotations

from aragora.config.stability import Stability
from .base import HandlerResult, BaseHandler, json_response, error_response
from .debates import DebatesHandler
from .agents import AgentsHandler
from .system import SystemHandler
from .pulse import PulseHandler
from .analytics import AnalyticsHandler
from .metrics import MetricsHandler
from .consensus import ConsensusHandler
from .belief import BeliefHandler
from .critique import CritiqueHandler
from .genesis import GenesisHandler
from .replays import ReplaysHandler
from .tournaments import TournamentHandler
from .memory import MemoryHandler
from .leaderboard import LeaderboardViewHandler
from .relationship import RelationshipHandler
from .moments import MomentsHandler
from .documents import DocumentHandler
from .verification import VerificationHandler
from .auditing import AuditingHandler
from .dashboard import DashboardHandler
from .persona import PersonaHandler
from .introspection import IntrospectionHandler
from .calibration import CalibrationHandler
from .routing import RoutingHandler
from .evolution import EvolutionHandler
from .evolution_ab_testing import EvolutionABTestingHandler
from .plugins import PluginsHandler
from .broadcast import BroadcastHandler
from .audio import AudioHandler
from .social import SocialMediaHandler
from .laboratory import LaboratoryHandler
from .probes import ProbesHandler
from .insights import InsightsHandler
from .gallery import GalleryHandler
from .breakpoints import BreakpointsHandler
from .learning import LearningHandler
from .auth import AuthHandler
from .billing import BillingHandler
from .organizations import OrganizationsHandler
from .oauth import OAuthHandler
from .graph_debates import GraphDebatesHandler
from .matrix_debates import MatrixDebatesHandler
from .features import FeaturesHandler
from .memory_analytics import MemoryAnalyticsHandler

# List of all handler classes for automatic dispatch registration
# Order matters: more specific handlers should come first
ALL_HANDLERS = [
    GraphDebatesHandler,  # More specific path: /api/debates/graph
    MatrixDebatesHandler,  # More specific path: /api/debates/matrix
    DebatesHandler,
    AgentsHandler,
    SystemHandler,
    PulseHandler,
    AnalyticsHandler,
    MetricsHandler,
    ConsensusHandler,
    BeliefHandler,
    CritiqueHandler,
    GenesisHandler,
    ReplaysHandler,
    TournamentHandler,
    MemoryHandler,
    LeaderboardViewHandler,
    RelationshipHandler,
    MomentsHandler,
    DocumentHandler,
    VerificationHandler,
    AuditingHandler,
    DashboardHandler,
    PersonaHandler,
    IntrospectionHandler,
    CalibrationHandler,
    RoutingHandler,
    EvolutionABTestingHandler,  # More specific: /api/evolution/ab-tests
    EvolutionHandler,
    PluginsHandler,
    AudioHandler,
    SocialMediaHandler,
    BroadcastHandler,
    LaboratoryHandler,
    ProbesHandler,
    InsightsHandler,
    GalleryHandler,
    BreakpointsHandler,
    LearningHandler,
    AuthHandler,
    BillingHandler,
    OrganizationsHandler,
    OAuthHandler,
    FeaturesHandler,
    MemoryAnalyticsHandler,
]

# Handler stability classifications
# - STABLE: Production-ready, extensively tested, API stable
# - EXPERIMENTAL: Works but may change, use with awareness
# - PREVIEW: Early access, expect changes and potential issues
# - DEPRECATED: Being phased out, use alternative
HANDLER_STABILITY: dict[str, Stability] = {
    # Core - Stable
    "DebatesHandler": Stability.STABLE,
    "AgentsHandler": Stability.STABLE,
    "SystemHandler": Stability.STABLE,
    "AnalyticsHandler": Stability.STABLE,
    "ConsensusHandler": Stability.STABLE,
    "MetricsHandler": Stability.STABLE,
    "MemoryHandler": Stability.STABLE,
    "LeaderboardViewHandler": Stability.STABLE,
    "ReplaysHandler": Stability.STABLE,
    "FeaturesHandler": Stability.STABLE,
    "AuthHandler": Stability.STABLE,

    # Extended - Stable
    "TournamentHandler": Stability.STABLE,
    "CritiqueHandler": Stability.STABLE,
    "RelationshipHandler": Stability.STABLE,
    "DashboardHandler": Stability.STABLE,
    "RoutingHandler": Stability.STABLE,

    # Experimental - Works but may change
    "GraphDebatesHandler": Stability.EXPERIMENTAL,
    "MatrixDebatesHandler": Stability.EXPERIMENTAL,
    "EvolutionHandler": Stability.EXPERIMENTAL,
    "EvolutionABTestingHandler": Stability.EXPERIMENTAL,
    "CalibrationHandler": Stability.EXPERIMENTAL,
    "IntrospectionHandler": Stability.EXPERIMENTAL,
    "PersonaHandler": Stability.EXPERIMENTAL,
    "BeliefHandler": Stability.EXPERIMENTAL,
    "LaboratoryHandler": Stability.EXPERIMENTAL,
    "ProbesHandler": Stability.EXPERIMENTAL,
    "InsightsHandler": Stability.EXPERIMENTAL,
    "LearningHandler": Stability.EXPERIMENTAL,
    "MemoryAnalyticsHandler": Stability.EXPERIMENTAL,

    # Preview - Early access
    "BillingHandler": Stability.PREVIEW,
    "OrganizationsHandler": Stability.PREVIEW,
    "OAuthHandler": Stability.PREVIEW,
    "BroadcastHandler": Stability.PREVIEW,
    "AudioHandler": Stability.PREVIEW,
    "SocialMediaHandler": Stability.PREVIEW,
    "GenesisHandler": Stability.PREVIEW,
    "VerificationHandler": Stability.PREVIEW,
    "MomentsHandler": Stability.PREVIEW,
    "DocumentHandler": Stability.PREVIEW,
    "AuditingHandler": Stability.PREVIEW,
    "GalleryHandler": Stability.PREVIEW,
    "BreakpointsHandler": Stability.PREVIEW,
    "PluginsHandler": Stability.PREVIEW,
    "PulseHandler": Stability.PREVIEW,
}


def get_handler_stability(handler_name: str) -> Stability:
    """Get the stability level for a handler.

    Args:
        handler_name: Handler class name (e.g., 'DebatesHandler')

    Returns:
        Stability level, defaults to EXPERIMENTAL if not classified
    """
    return HANDLER_STABILITY.get(handler_name, Stability.EXPERIMENTAL)


def get_all_handler_stability() -> dict[str, str]:
    """Get all handler stability levels as strings for API response."""
    return {name: stability.value for name, stability in HANDLER_STABILITY.items()}


__all__ = [
    # Base utilities
    "HandlerResult",
    "BaseHandler",
    "json_response",
    "error_response",
    # Handler registry
    "ALL_HANDLERS",
    # Individual handlers
    "DebatesHandler",
    "AgentsHandler",
    "SystemHandler",
    "PulseHandler",
    "AnalyticsHandler",
    "MetricsHandler",
    "ConsensusHandler",
    "BeliefHandler",
    "CritiqueHandler",
    "GenesisHandler",
    "ReplaysHandler",
    "TournamentHandler",
    "MemoryHandler",
    "LeaderboardViewHandler",
    "RelationshipHandler",
    "MomentsHandler",
    "DocumentHandler",
    "VerificationHandler",
    "AuditingHandler",
    "DashboardHandler",
    "PersonaHandler",
    "IntrospectionHandler",
    "CalibrationHandler",
    "RoutingHandler",
    "EvolutionHandler",
    "EvolutionABTestingHandler",
    "PluginsHandler",
    "AudioHandler",
    "SocialMediaHandler",
    "BroadcastHandler",
    "LaboratoryHandler",
    "ProbesHandler",
    "InsightsHandler",
    "GalleryHandler",
    "BreakpointsHandler",
    "LearningHandler",
    "AuthHandler",
    "BillingHandler",
    "OrganizationsHandler",
    "OAuthHandler",
    "GraphDebatesHandler",
    "MatrixDebatesHandler",
    "FeaturesHandler",
    "MemoryAnalyticsHandler",
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
